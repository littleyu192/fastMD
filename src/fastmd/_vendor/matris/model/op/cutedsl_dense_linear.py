"""Experimental low-overhead CuTeDSL dense Linear wrapper.

This module is intentionally still an opt-in staging path for the indexed
gather/cat + GEMM work.  The forward path uses a Blackwell persistent dense
GEMM launched from a torch custom op with raw A/B pointers; backward only
computes grad_input because MatRIS MD inference keeps these weights frozen.
"""

from __future__ import annotations

from fastmd._vendor.matris.config import OPTION_REGISTRY, current_options_environment, env_value

from collections.abc import Iterable
import itertools
import json
import os
import subprocess
import sys
from typing import Any, Optional

import cuda.bindings.driver as cuda
import torch

from .fast_custom_op import fast_custom_op


Tactic = tuple[bool, tuple[int, int], tuple[int, int], int, str]

_COMPILED: dict[tuple[Any, ...], Any] = {}
_TACTIC_CACHE: dict[tuple[Any, ...], Tactic] = {}
_LAST_ERROR: str | None = None
_TUNE_RESULT_PREFIX = "MATRIS_CUTEDSL_TUNE_RESULT="
_TUNING_OPTIONS = tuple(sorted(
    name for name in OPTION_REGISTRY if name.startswith("MATRIS_CUTEDSL_")
))


def _tuning_configuration() -> tuple:
    """Do not reuse a selected tactic across different tuning overrides."""
    return tuple(
        env_value(name, OPTION_REGISTRY[name].default) for name in _TUNING_OPTIONS
    )


def _parse_pair(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected two comma-separated ints, got {value!r}")
    return parts[0], parts[1]


def _parse_swizzle_size(value: str | None) -> int | None:
    if not value:
        return None
    swizzle_size = int(value.strip())
    if swizzle_size not in {1, 2, 4, 8}:
        raise ValueError(f"unsupported swizzle size {swizzle_size}; expected one of 1,2,4,8")
    return swizzle_size


def _parse_raster_along(value: str | None) -> str | None:
    if not value:
        return None
    raster_along = value.strip().lower()
    if raster_along not in {"m", "n"}:
        raise ValueError(f"unsupported raster direction {value!r}; expected 'm' or 'n'")
    return raster_along


def _env_flag(name: str, default: bool) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


def _use_tvm_ffi() -> bool:
    return _env_flag("MATRIS_CUTEDSL_USE_TVM_FFI", False)


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _load_kernel_module():
    from .cutedsl_blackwell import dense_gemm_persistent as dense_gemm
    from .cutedsl_blackwell import utils as bw_utils

    return dense_gemm, bw_utils


def _cutlass_type_for(t: torch.Tensor):
    dense_gemm, _ = _load_kernel_module()
    if t.dtype is torch.float32:
        return dense_gemm.cutlass.TFloat32, dense_gemm.cutlass.Float32
    if t.dtype is torch.float16:
        return dense_gemm.cutlass.Float16, dense_gemm.cutlass.Float16
    if t.dtype is torch.bfloat16:
        return dense_gemm.cutlass.BFloat16, dense_gemm.cutlass.BFloat16
    return None


def _candidate_tactics(
    m: int,
    n: int,
    k: int,
) -> Iterable[Tactic]:
    tile_override = _parse_pair(env_value("MATRIS_CUTEDSL_MMA_TILER_MN"))
    cluster_override = _parse_pair(env_value("MATRIS_CUTEDSL_CLUSTER_SHAPE_MN"))
    twocta_override = env_value("MATRIS_CUTEDSL_USE_2CTA")
    swizzle_override = _parse_swizzle_size(env_value("MATRIS_CUTEDSL_SWIZZLE_SIZE"))
    raster_override = _parse_raster_along(env_value("MATRIS_CUTEDSL_RASTER_ALONG"))
    tune_scheduler = _env_flag("MATRIS_CUTEDSL_AUTOTUNE_SCHEDULER", True)

    def scheduler_candidates() -> Iterable[tuple[int, str]]:
        swizzles = (swizzle_override,) if swizzle_override is not None else (1,)
        rasters = (raster_override,) if raster_override is not None else (
            ("m", "n") if tune_scheduler else ("m",)
        )
        return itertools.product(swizzles, rasters)

    def append_scheduler(
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> Iterable[Tactic]:
        for swizzle_size, raster_along in scheduler_candidates():
            yield use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, swizzle_size, raster_along

    if tile_override is not None:
        yield from append_scheduler(
            (twocta_override != "0") if twocta_override is not None else tile_override[0] >= 128,
            tile_override,
            cluster_override if cluster_override is not None else (2, 1),
        )
        return

    if _env_flag("MATRIS_CUTEDSL_AUTOTUNE", True):
        if _env_flag("MATRIS_CUTEDSL_AUTOTUNE_EXHAUSTIVE", False):
            use_2cta_candidates = (False, True)
            tile_candidates = (
                (64, 128),
                (128, 128),
                (128, 256),
                (256, 128),
                (256, 256),
            )
            cluster_candidates = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1))
            for use_2cta_instrs, mma_tiler_mn, cluster_shape_mn in itertools.product(
                use_2cta_candidates, tile_candidates, cluster_candidates
            ):
                yield from append_scheduler(use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
            return
        yield from append_scheduler(True, (128, 256), (2, 1))
        yield from append_scheduler(True, (256, 256), (2, 1))
        yield from append_scheduler(False, (128, 256), (1, 1))
        yield from append_scheduler(True, (128, 128), (2, 1))
        yield from append_scheduler(False, (128, 128), (1, 1))
        return

    if n == 256 and k >= 512:
        yield from append_scheduler(True, (256, 256), (2, 1))
    elif n == 256:
        yield from append_scheduler(True, (128, 256), (2, 1))
    else:
        yield from append_scheduler(False, (128, 128), (1, 1))


def _valid_tactics(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> list[Tactic]:
    dense_gemm, _ = _load_kernel_module()
    dtype_pair = _cutlass_type_for(x)
    if dtype_pair is None:
        return []
    ab_dtype, c_dtype = dtype_pair
    m, k = x.shape
    n = weight.shape[0]
    candidates = []
    seen = set()
    for tactic in _candidate_tactics(m, n, k):
        use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, _swizzle_size, _raster_along = tactic
        if tactic in seen:
            continue
        seen.add(tactic)
        if dense_gemm.PersistentDenseGemmKernel.can_implement(
            ab_dtype,
            dense_gemm.cutlass.Float32,
            c_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            1,
            "k",
            "k",
            "n",
        ):
            candidates.append(tactic)
    return candidates


def _dense_jit_cache_key(
    x: torch.Tensor,
    out: torch.Tensor,
    tactic: Tactic,
) -> tuple[Any, ...]:
    """Key for the compiled CuTeDSL wrapper.

    ``m/n/k`` are runtime ``Int32`` wrapper arguments, so they belong to the
    tactic-selection key, not the JIT key.
    """
    use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, swizzle_size, raster_along = tactic
    return (
        _device_index(x.device),
        torch.cuda.get_device_capability(x.device),
        x.dtype,
        out.dtype,
        "PersistentDenseGemmKernel",
        "Float32",
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        swizzle_size,
        raster_along,
        True,  # use_tma_store
        _use_tvm_ffi(),
    )


def _dense_tactic_cache_key(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[Any, ...]:
    m, k = x.shape
    n = weight.shape[0]
    return (
        _device_index(x.device),
        torch.cuda.get_device_capability(x.device),
        x.dtype,
        m,
        n,
        k,
        bias is not None,
        _tuning_configuration(),
    )


def _tactic_from_json(value: object) -> Tactic:
    if not (
        isinstance(value, list)
        and len(value) == 5
        and isinstance(value[0], bool)
        and isinstance(value[1], list)
        and isinstance(value[2], list)
        and len(value[1]) == 2
        and len(value[2]) == 2
        and isinstance(value[3], int)
        and isinstance(value[4], str)
    ):
        raise ValueError(f"invalid tactic payload: {value!r}")
    raster_along = _parse_raster_along(value[4])
    if raster_along is None:
        raise ValueError(f"invalid tactic raster direction: {value!r}")
    return (
        value[0],
        (int(value[1][0]), int(value[1][1])),
        (int(value[2][0]), int(value[2][1])),
        _parse_swizzle_size(str(value[3])) or 1,
        raster_along,
    )


def _make_cute_output_tensor(out: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(out.unsqueeze(-1)).mark_layout_dynamic(leading_dim=1)


def _make_ab_ptrs(x: torch.Tensor, weight: torch.Tensor):
    dense_gemm, bw_utils = _load_kernel_module()
    dtype_pair = _cutlass_type_for(x)
    if dtype_pair is None:
        raise TypeError(f"unsupported dtype for CuTeDSL dense linear: {x.dtype}")
    ab_dtype, _ = dtype_pair
    a_ptr = bw_utils.make_ptr(
        ab_dtype,
        x.data_ptr(),
        dense_gemm.cute.AddressSpace.gmem,
        assumed_align=16,
    )
    b_ptr = bw_utils.make_ptr(
        ab_dtype,
        weight.data_ptr(),
        dense_gemm.cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return a_ptr, b_ptr


def _compile_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tactic: Tactic,
):
    dense_gemm, bw_utils = _load_kernel_module()
    dtype_pair = _cutlass_type_for(x)
    if dtype_pair is None:
        raise TypeError(f"unsupported dtype for CuTeDSL dense linear: {x.dtype}")
    ab_dtype, c_dtype = dtype_pair
    use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, swizzle_size, raster_along = tactic
    m, k = x.shape
    n = weight.shape[0]
    key = _dense_jit_cache_key(x, out, tactic)
    compiled = _COMPILED.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CuTeDSL dense GEMM is not compiled before CUDA graph capture")

    a_ptr, b_ptr = _make_ab_ptrs(x, weight)
    c_tensor = _make_cute_output_tensor(out)
    if _use_tvm_ffi():
        stream = dense_gemm.cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compile_options = "--opt-level 2 --enable-tvm-ffi"
    else:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compile_options = "--opt-level 2"
    gemm = dense_gemm.PersistentDenseGemmKernel(
        dense_gemm.cutlass.Float32,
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_tma_store=True,
        swizzle_size=swizzle_size,
        raster_along=raster_along,
    )
    max_active_clusters = dense_gemm.utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    compiled = dense_gemm.cute.compile(
        gemm.wrapper,
        m,
        n,
        k,
        1,
        a_ptr,
        b_ptr,
        c_tensor,
        max_active_clusters=max_active_clusters,
        stream=stream,
        options=compile_options,
    )
    _COMPILED[key] = compiled
    return compiled


def _launch_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tactic: Tactic,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    compiled = _compile_tactic(x, weight, out, tactic)
    m, k = x.shape
    n = weight.shape[0]
    if _use_tvm_ffi():
        compiled(m, n, k, 1, x.data_ptr(), weight.data_ptr(), out.unsqueeze(-1))
    else:
        a_ptr, b_ptr = _make_ab_ptrs(x, weight)
        c_tensor = _make_cute_output_tensor(out)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(m, n, k, 1, a_ptr, b_ptr, c_tensor, stream=stream)
    if bias is not None:
        out.add_(bias)
    return out


def _profile_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    tactic: Tactic,
    bias: torch.Tensor | None,
    timer_override: str | None = None,
) -> float:
    warmup = int(env_value("MATRIS_CUTEDSL_TUNE_WARMUP", "10"))
    repeat = int(env_value("MATRIS_CUTEDSL_TUNE_REPEAT", "50"))
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    _compile_tactic(x, weight, out, tactic)
    for _ in range(warmup):
        _launch_tactic(x, weight, out, tactic, bias)
    timer = (timer_override or env_value("MATRIS_CUTEDSL_TUNE_TIMER", "cupti")).strip().lower()
    if timer in {"cupti", "cupti_inline"}:
        try:
            from .cupti_activity import profile_kernel_ms

            return profile_kernel_ms(lambda: _launch_tactic(x, weight, out, tactic, bias), repeat)
        except Exception:
            if env_value("MATRIS_CUTEDSL_TUNE_CUPTI_STRICT") == "1":
                raise
    return _profile_tactic_cuda_event(x, weight, out, tactic, bias, repeat)


def _profile_tactic_cuda_event(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tactic: Tactic,
    bias: torch.Tensor | None,
    repeat: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        _launch_tactic(x, weight, out, tactic, bias)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _select_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> Tactic:
    key = _dense_tactic_cache_key(x, weight, bias)
    cached = _TACTIC_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CuTeDSL dense GEMM tactic is not selected before CUDA graph capture")

    valid = _valid_tactics(x, weight, out)
    if not valid:
        m, k = x.shape
        n = weight.shape[0]
        raise RuntimeError(f"no valid CuTeDSL dense GEMM tactics for {(m, n, k)}")
    if not _env_flag("MATRIS_CUTEDSL_AUTOTUNE", True) or len(valid) == 1:
        best = valid[0]
    else:
        best = _select_tactic_by_autotune(x, weight, bias, valid)
    _TACTIC_CACHE[key] = best
    return best


def _select_tactic_by_autotune(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    valid: list[Tactic],
) -> Tactic:
    timer = env_value("MATRIS_CUTEDSL_TUNE_TIMER", "cupti").strip().lower()
    if timer in {"cupti", "cupti_subprocess"} and not _env_flag("_MATRIS_CUTEDSL_TUNE_WORKER", False):
        try:
            tactic = _select_tactic_cupti_subprocess(x, weight, bias)
            if tactic in valid:
                return tactic
            raise RuntimeError(f"CUPTI worker returned invalid tactic {tactic!r}")
        except Exception:
            if env_value("MATRIS_CUTEDSL_TUNE_CUPTI_STRICT") == "1":
                raise
            return min(
                valid,
                key=lambda tactic: _profile_tactic(
                    x, weight, tactic, bias, timer_override="cuda_event"
                ),
            )
    return min(valid, key=lambda tactic: _profile_tactic(x, weight, tactic, bias))


def _select_tactic_cupti_subprocess(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> Tactic:
    timeout_s = float(env_value("MATRIS_CUTEDSL_TUNE_CUPTI_TIMEOUT_S", "300"))
    env = os.environ.copy()
    # The active inference profile is scoped, not written into os.environ.
    # Export its full snapshot so the tuning subprocess sees the same choices.
    for name in OPTION_REGISTRY:
        env.pop(name, None)
    env.update(current_options_environment())
    env["_MATRIS_CUTEDSL_TUNE_WORKER"] = "1"
    env["MATRIS_CUTEDSL_TUNE_TIMER"] = "cupti_inline"
    env.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
    command = [
        sys.executable,
        "-m",
        "fastmd._vendor.matris.model.op.cutedsl_dense_tune_worker",
        "--device",
        str(_device_index(x.device)),
        "--dtype",
        str(x.dtype).removeprefix("torch."),
        "--m",
        str(x.shape[0]),
        "--n",
        str(weight.shape[0]),
        "--k",
        str(x.shape[1]),
    ]
    if bias is not None:
        command.append("--bias")
    result = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "CUPTI autotune worker failed "
            f"(exit={result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_TUNE_RESULT_PREFIX):
            payload = json.loads(line[len(_TUNE_RESULT_PREFIX) :])
            return _tactic_from_json(payload["best"])
    raise RuntimeError(f"CUPTI autotune worker did not return a result: {result.stdout[-2000:]}")


def _run_cutedsl_dense_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    tactic = _select_tactic(x, weight, bias, out)
    return _launch_tactic(x, weight, out, tactic, bias)


@fast_custom_op("fastmd_matris::cutedsl_dense_linear", mutates_args=(), device_types="CUDA")
def _cutedsl_dense_linear_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return _run_cutedsl_dense_linear(x, weight, bias)


@_cutedsl_dense_linear_op.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)


def _setup_context(ctx, inputs, output) -> None:
    _x, weight, _bias = inputs
    ctx.save_for_backward(weight)


def _backward(ctx, grad_out: torch.Tensor):
    (weight,) = ctx.saved_tensors
    grad_x = grad_out.contiguous().matmul(weight)
    return grad_x, None, None


torch.library.register_autograd(
    "fastmd_matris::cutedsl_dense_linear",
    _backward,
    setup_context=_setup_context,
)


def cutedsl_dense_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return ``x @ weight.T + bias`` using CuTeDSL forward, or ``None`` on setup failure."""
    global _LAST_ERROR
    if not (
        x.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.dtype == weight.dtype
        and x.dim() == 2
        and weight.dim() == 2
        and weight.shape[1] == x.shape[1]
    ):
        return None
    if bias is not None and (
        not bias.is_cuda or bias.dtype != x.dtype or not bias.is_contiguous()
    ):
        return None
    try:
        return _cutedsl_dense_linear_op(x, weight, bias)
    except Exception as exc:
        _LAST_ERROR = repr(exc)
        return None


def last_cutedsl_dense_linear_error() -> str | None:
    return _LAST_ERROR
