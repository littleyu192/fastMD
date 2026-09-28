"""CuTeDSL dense first Linear with fused bias+SiLU split postprocess.

This is an opt-in staging backend for GatedMLP first-layer fusion.  It keeps
the efficient single N=256 CuTeDSL GEMM, then runs one Triton kernel to turn the
merged preactivation into contiguous core/gate SiLU outputs.  An experimental
CuTeDSL dual-output epilogue is available with
``MATRIS_CUTEDSL_SILU_EPILOGUE=1`` but is not the default path.  Backward
computes only grad_input because the caller guards these first-layer weights as
frozen.
"""

from __future__ import annotations

from fastmd._vendor.matris.config import env_value

import itertools
from typing import Any

import cuda.bindings.driver as cuda
import torch
from torch.autograd import Function

import triton
import triton.language as tl

from .cutedsl_dense_linear import (
    Tactic,
    _candidate_tactics,
    _device_index,
    _env_flag,
    _make_ab_ptrs,
    _tuning_configuration,
    _use_tvm_ffi,
    cutedsl_dense_linear,
)


_LAST_ERROR: str | None = None
_COMPILED: dict[tuple[Any, ...], Any] = {}
_TACTIC_CACHE: dict[tuple[Any, ...], Tactic] = {}


def _env_int(name: str, default: int) -> int:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


_SPLIT_BM = _env_int("MATRIS_CUTEDSL_SILU_SPLIT_BM", 2)


def _load_epilogue_kernel_module():
    from .cutedsl_blackwell import dense_gemm_silu_split_persistent as dense_gemm
    from .cutedsl_blackwell import utils as bw_utils

    return dense_gemm, bw_utils


def _cutlass_type_for(t: torch.Tensor):
    dense_gemm, _ = _load_epilogue_kernel_module()
    if t.dtype is torch.float32:
        return dense_gemm.cutlass.TFloat32, dense_gemm.cutlass.Float32
    if t.dtype is torch.float16:
        return dense_gemm.cutlass.Float16, dense_gemm.cutlass.Float16
    if t.dtype is torch.bfloat16:
        return dense_gemm.cutlass.BFloat16, dense_gemm.cutlass.BFloat16
    return None


def _make_cute_output_tensor(out: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(out.unsqueeze(-1)).mark_layout_dynamic(leading_dim=1)


def _make_bias_ptr(x: torch.Tensor, bias: torch.Tensor | None):
    dense_gemm, bw_utils = _load_epilogue_kernel_module()
    dtype_pair = _cutlass_type_for(x)
    if dtype_pair is None:
        raise TypeError(f"unsupported dtype for CuTeDSL dense SiLU linear: {x.dtype}")
    _ab_dtype, c_dtype = dtype_pair
    ptr_value = bias.data_ptr() if bias is not None else x.data_ptr()
    return bw_utils.make_ptr(
        c_dtype,
        ptr_value,
        dense_gemm.cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _valid_tactics(
    x: torch.Tensor,
    weight: torch.Tensor,
    preact: torch.Tensor,
) -> list[Tactic]:
    dense_gemm, _ = _load_epilogue_kernel_module()
    dtype_pair = _cutlass_type_for(x)
    if dtype_pair is None:
        return []
    ab_dtype, c_dtype = dtype_pair
    m, k = x.shape
    n = weight.shape[0]
    candidates = []
    seen = set()
    tactic_iter = _candidate_tactics(m, n, k)
    if n == 256:
        tactic_iter = itertools.chain([(True, (256, 256), (2, 1))], tactic_iter)
    for use_2cta_instrs, mma_tiler_mn, cluster_shape_mn in tactic_iter:
        if n == 256 and (not use_2cta_instrs or mma_tiler_mn != (256, 256)):
            continue
        tactic = (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn)
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


def _epilogue_jit_cache_key(
    x: torch.Tensor,
    preact: torch.Tensor,
    core_act: torch.Tensor,
    gate_act: torch.Tensor,
    bias: torch.Tensor | None,
    tactic: Tactic,
) -> tuple[Any, ...]:
    use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = tactic
    return (
        _device_index(x.device),
        torch.cuda.get_device_capability(x.device),
        x.dtype,
        preact.dtype,
        core_act.dtype,
        gate_act.dtype,
        "DenseSiluSplitPersistentDenseGemmKernel",
        "Float32",
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        False,  # use_tma_store
        bias is not None,
        gate_act.shape[1],
        _use_tvm_ffi(),
    )


def _epilogue_tactic_cache_key(
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


def _compile_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    preact: torch.Tensor,
    core_act: torch.Tensor,
    gate_act: torch.Tensor,
    bias: torch.Tensor | None,
    tactic: Tactic,
):
    dense_gemm, _ = _load_epilogue_kernel_module()
    dtype_pair = _cutlass_type_for(x)
    if dtype_pair is None:
        raise TypeError(f"unsupported dtype for CuTeDSL dense SiLU linear: {x.dtype}")
    use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = tactic
    m, k = x.shape
    n = weight.shape[0]
    h = n // 2
    key = _epilogue_jit_cache_key(x, preact, core_act, gate_act, bias, tactic)
    compiled = _COMPILED.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CuTeDSL dense SiLU epilogue GEMM is not compiled before CUDA graph capture")

    a_ptr, b_ptr = _make_ab_ptrs(x, weight)
    preact_tensor = _make_cute_output_tensor(preact)
    core_tensor = _make_cute_output_tensor(core_act)
    gate_tensor = _make_cute_output_tensor(gate_act)
    bias_ptr = _make_bias_ptr(x, bias)
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
        use_tma_store=False,
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
        preact_tensor,
        core_tensor,
        gate_tensor,
        bias_ptr,
        bias is not None,
        h,
        max_active_clusters=max_active_clusters,
        stream=stream,
        options=compile_options,
    )
    _COMPILED[key] = compiled
    return compiled


def _launch_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    preact: torch.Tensor,
    core_act: torch.Tensor,
    gate_act: torch.Tensor,
    bias: torch.Tensor | None,
    tactic: Tactic,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compiled = _compile_tactic(x, weight, preact, core_act, gate_act, bias, tactic)
    m, k = x.shape
    n = weight.shape[0]
    if _use_tvm_ffi():
        compiled(
            m,
            n,
            k,
            1,
            x.data_ptr(),
            weight.data_ptr(),
            preact.unsqueeze(-1),
            core_act.unsqueeze(-1),
            gate_act.unsqueeze(-1),
            bias.data_ptr() if bias is not None else x.data_ptr(),
        )
    else:
        a_ptr, b_ptr = _make_ab_ptrs(x, weight)
        preact_tensor = _make_cute_output_tensor(preact)
        core_tensor = _make_cute_output_tensor(core_act)
        gate_tensor = _make_cute_output_tensor(gate_act)
        bias_ptr = _make_bias_ptr(x, bias)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(
            m,
            n,
            k,
            1,
            a_ptr,
            b_ptr,
            preact_tensor,
            core_tensor,
            gate_tensor,
            bias_ptr,
            stream=stream,
        )
    return preact, core_act, gate_act


def _profile_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    tactic: Tactic,
) -> float:
    warmup = int(env_value("MATRIS_CUTEDSL_SILU_TUNE_WARMUP", "2"))
    repeat = int(env_value("MATRIS_CUTEDSL_SILU_TUNE_REPEAT", "5"))
    m = x.shape[0]
    n = weight.shape[0]
    h = n // 2
    preact = torch.empty((m, n), device=x.device, dtype=x.dtype)
    core_act = torch.empty((m, h), device=x.device, dtype=x.dtype)
    gate_act = torch.empty((m, h), device=x.device, dtype=x.dtype)
    _compile_tactic(x, weight, preact, core_act, gate_act, bias, tactic)
    for _ in range(warmup):
        _launch_tactic(x, weight, preact, core_act, gate_act, bias, tactic)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        _launch_tactic(x, weight, preact, core_act, gate_act, bias, tactic)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _select_tactic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    preact: torch.Tensor,
) -> Tactic:
    key = _epilogue_tactic_cache_key(x, weight, bias)
    cached = _TACTIC_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CuTeDSL dense SiLU epilogue tactic is not selected before CUDA graph capture")

    valid = _valid_tactics(x, weight, preact)
    if not valid:
        m, k = x.shape
        n = weight.shape[0]
        raise RuntimeError(f"no valid CuTeDSL dense SiLU epilogue tactics for {(m, n, k)}")
    if not _env_flag("MATRIS_CUTEDSL_AUTOTUNE", True) or len(valid) == 1:
        best = valid[0]
    else:
        best = min(valid, key=lambda tactic: _profile_tactic(x, weight, bias, tactic))
    _TACTIC_CACHE[key] = best
    return best


def _run_cutedsl_dense_silu_epilogue(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    n = weight.shape[0]
    h = n // 2
    preact = torch.empty((m, n), device=x.device, dtype=x.dtype)
    core_act = torch.empty((m, h), device=x.device, dtype=x.dtype)
    gate_act = torch.empty((m, h), device=x.device, dtype=x.dtype)
    tactic = _select_tactic(x, weight, bias, preact)
    return _launch_tactic(x, weight, preact, core_act, gate_act, bias, tactic)


@triton.jit
def _split_bias_silu_fwd(
    merged,
    bias,
    core_act,
    gate_act,
    M: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)

    base = rows[:, None] * N
    core = tl.load(merged + base + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(merged + base + H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        core += tl.load(bias + cols, mask=cols < H, other=0.0).to(tl.float32)[None, :]
        gate += tl.load(bias + H + cols, mask=cols < H, other=0.0).to(tl.float32)[None, :]

    out_base = rows[:, None] * H
    tl.store(merged + base + cols[None, :], core, mask=mask)
    tl.store(merged + base + H + cols[None, :], gate, mask=mask)
    tl.store(core_act + out_base + cols[None, :], core * tl.sigmoid(core), mask=mask)
    tl.store(gate_act + out_base + cols[None, :], gate * tl.sigmoid(gate), mask=mask)


@triton.jit
def _split_silu_bwd(
    grad_core,
    grad_gate,
    merged,
    grad_merged,
    M: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)

    in_base = rows[:, None] * N
    out_base = rows[:, None] * H
    core = tl.load(merged + in_base + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(merged + in_base + H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gcore = tl.load(grad_core + out_base + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    ggate = tl.load(grad_gate + out_base + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    sig_core = tl.sigmoid(core)
    sig_gate = tl.sigmoid(gate)
    dcore = gcore * sig_core * (1.0 + core * (1.0 - sig_core))
    dgate = ggate * sig_gate * (1.0 + gate * (1.0 - sig_gate))
    tl.store(grad_merged + in_base + cols[None, :], dcore, mask=mask)
    tl.store(grad_merged + in_base + H + cols[None, :], dgate, mask=mask)


def _launch_split_bias_silu(
    merged: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = merged.shape
    h = n // 2
    core_act = torch.empty((m, h), device=merged.device, dtype=merged.dtype)
    gate_act = torch.empty((m, h), device=merged.device, dtype=merged.dtype)
    block = triton.next_power_of_2(h)
    bm = _SPLIT_BM
    _split_bias_silu_fwd[(triton.cdiv(m, bm),)](
        merged,
        bias,
        core_act,
        gate_act,
        M=m,
        H=h,
        N=n,
        BM=bm,
        BLK=block,
        HAS_BIAS=bias is not None,
        num_warps=4 if block >= 128 else 1,
    )
    return core_act, gate_act


def _launch_split_silu_bwd(
    grad_core: torch.Tensor,
    grad_gate: torch.Tensor,
    merged: torch.Tensor,
) -> torch.Tensor:
    m, n = merged.shape
    h = n // 2
    grad_merged = torch.empty_like(merged)
    block = triton.next_power_of_2(h)
    bm = _SPLIT_BM
    _split_silu_bwd[(triton.cdiv(m, bm),)](
        grad_core,
        grad_gate,
        merged,
        grad_merged,
        M=m,
        H=h,
        N=n,
        BM=bm,
        BLK=block,
        num_warps=4 if block >= 128 else 1,
    )
    return grad_merged


class _CuTeDSLDenseSiluLinear(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        if _env_flag("MATRIS_CUTEDSL_SILU_EPILOGUE", False):
            preact, core_act, gate_act = _run_cutedsl_dense_silu_epilogue(x, weight, bias)
        else:
            preact = cutedsl_dense_linear(x, weight, None)
            if preact is None:
                raise RuntimeError("CuTeDSL dense GEMM returned None")
            core_act, gate_act = _launch_split_bias_silu(preact, bias)
        ctx.save_for_backward(preact, weight)
        return core_act, gate_act

    @staticmethod
    def backward(ctx, grad_core: torch.Tensor, grad_gate: torch.Tensor):
        preact, weight = ctx.saved_tensors
        grad_merged = _launch_split_silu_bwd(
            grad_core.contiguous(),
            grad_gate.contiguous(),
            preact,
        )
        grad_x = grad_merged.matmul(weight)
        return grad_x, None, None


def cutedsl_dense_silu_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return contiguous ``SiLU(linear_core(x))`` and ``SiLU(linear_gate(x))``.

    ``weight`` is the merged ``cat([core_weight, gate_weight], dim=0)`` matrix.
    """
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
        and weight.shape[0] % 2 == 0
    ):
        return None
    if bias is not None and (
        not bias.is_cuda
        or bias.dtype != x.dtype
        or not bias.is_contiguous()
        or bias.dim() != 1
        or bias.shape[0] != weight.shape[0]
    ):
        return None
    try:
        return _CuTeDSLDenseSiluLinear.apply(x, weight, bias)
    except Exception as exc:
        _LAST_ERROR = repr(exc)
        return None


def last_cutedsl_dense_silu_linear_error() -> str | None:
    return _LAST_ERROR
