"""Decomposed indexed concat first Linear with fused gather-add+SiLU.

This path exploits reuse in the line-graph indices:

    cat3: A@W0.T + B[index_a]@W1.T + B[index_b]@W2.T
    cat4: A@W0.T + X[index_a]@W1.T + B[index_b]@W2.T + B[index_c]@W3.T

The dense projections stay as high-throughput GEMMs over the source tensors;
one Triton epilogue then gathers projected rows, adds bias, applies SiLU, and
returns contiguous core/gate activations for the rest of GatedMLP.
"""

from __future__ import annotations

from fastmd._vendor.matris.config import env_value


import torch
import triton
import triton.language as tl
from torch.autograd import Function


_LAST_ERROR: str | None = None
_DEBUG_PRINTED: set[str] = set()


def _env_int(name: str, default: int) -> int:
    value = env_value(name)
    if value is None:
        return default
    return int(value)


def _env_flag(name: str, default: bool) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


_EPILOGUE_BM = _env_int("MATRIS_DECOMPOSED_INDEXED_SILU_BM", 2)
_EPILOGUE_BWD_BM = _env_int("MATRIS_DECOMPOSED_INDEXED_SILU_BWD_BM", 8)
_SORTED_A_PAIR_BWD = _env_flag("MATRIS_DECOMPOSED_INDEXED_SORTED_A_PAIR_BWD", True)
_GEMM_BACKEND = env_value("MATRIS_DECOMPOSED_INDEXED_GEMM_BACKEND", "torch").lower()


def _debug_check_index(name: str, index: torch.Tensor, limit: int) -> None:
    if not _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
        return
    if index.numel() == 0:
        return
    lo = int(index.min().item())
    hi = int(index.max().item())
    if lo < 0 or hi >= limit:
        raise RuntimeError(f"{name} index out of bounds: min={lo} max={hi} limit={limit}")


def _debug_print_once(name: str, *parts: object) -> None:
    if not _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
        return
    if name in _DEBUG_PRINTED:
        return
    _DEBUG_PRINTED.add(name)
    print("[decomposed_indexed_debug]", name, *parts, flush=True)


def _project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if _GEMM_BACKEND == "cutedsl":
        from .cutedsl_dense_linear import cutedsl_dense_linear

        out = cutedsl_dense_linear(x, weight, None)
        if out is not None:
            return out
        if not _env_flag("MATRIS_DECOMPOSED_INDEXED_CUTEDSL_FALLBACK", True):
            from .cutedsl_dense_linear import last_cutedsl_dense_linear_error

            raise RuntimeError(
                "CuTeDSL decomposed projection failed: "
                f"{last_cutedsl_dense_linear_error()}"
            )
    return torch.nn.functional.linear(x, weight, None)


@triton.jit
def _cat3_gather_add_silu_fwd(
    p0,
    p1,
    p2,
    index_a,
    index_b,
    bias,
    preact,
    core_act,
    gate_act,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    SP0: tl.constexpr,
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
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0)

    base_n = rows[:, None] * N
    base_p0 = rows[:, None] * SP0
    core = tl.load(p0 + base_p0 + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(p0 + base_p0 + H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    core += tl.load(
        p1 + idx_a[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p1 + idx_a[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    core += tl.load(
        p2 + idx_b[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p2 + idx_b[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_BIAS:
        core += tl.load(bias + cols, mask=cols < H, other=0.0).to(tl.float32)[None, :]
        gate += tl.load(bias + H + cols, mask=cols < H, other=0.0).to(tl.float32)[
            None, :
        ]

    out_base = rows[:, None] * H
    tl.store(preact + base_n + cols[None, :], core, mask=mask)
    tl.store(preact + base_n + H + cols[None, :], gate, mask=mask)
    tl.store(core_act + out_base + cols[None, :], core * tl.sigmoid(core), mask=mask)
    tl.store(gate_act + out_base + cols[None, :], gate * tl.sigmoid(gate), mask=mask)


@triton.jit
def _cat4_gather_add_silu_fwd(
    p0,
    p1,
    p2,
    p3,
    index_a,
    index_b,
    index_c,
    bias,
    preact,
    core_act,
    gate_act,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    SIC: tl.constexpr,
    SP0: tl.constexpr,
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
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0)
    idx_c = tl.load(index_c + rows * SIC, mask=rows < M, other=0)

    base_n = rows[:, None] * N
    base_p0 = rows[:, None] * SP0
    core = tl.load(p0 + base_p0 + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(p0 + base_p0 + H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    core += tl.load(
        p1 + idx_a[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p1 + idx_a[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    core += tl.load(
        p2 + idx_b[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p2 + idx_b[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    core += tl.load(
        p3 + idx_c[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gate += tl.load(
        p3 + idx_c[:, None] * N + H + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_BIAS:
        core += tl.load(bias + cols, mask=cols < H, other=0.0).to(tl.float32)[None, :]
        gate += tl.load(bias + H + cols, mask=cols < H, other=0.0).to(tl.float32)[
            None, :
        ]

    out_base = rows[:, None] * H
    tl.store(preact + base_n + cols[None, :], core, mask=mask)
    tl.store(preact + base_n + H + cols[None, :], gate, mask=mask)
    tl.store(core_act + out_base + cols[None, :], core * tl.sigmoid(core), mask=mask)
    tl.store(gate_act + out_base + cols[None, :], gate * tl.sigmoid(gate), mask=mask)


@triton.jit
def _cat3_gather_add_silu_bwd(
    grad_core,
    grad_gate,
    preact,
    index_a,
    index_b,
    grad_p0,
    grad_p1,
    grad_p2,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
    SORTED_A: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0)

    in_base = rows[:, None] * N
    out_base = rows[:, None] * H
    core = tl.load(preact + in_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(preact + in_base + H + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gcore = tl.load(grad_core + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    ggate = tl.load(grad_gate + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    sig_core = tl.sigmoid(core)
    sig_gate = tl.sigmoid(gate)
    dcore = gcore * sig_core * (1.0 + core * (1.0 - sig_core))
    dgate = ggate * sig_gate * (1.0 + gate * (1.0 - sig_gate))

    tl.store(grad_p0 + in_base + cols[None, :], dcore, mask=mask)
    tl.store(grad_p0 + in_base + H + cols[None, :], dgate, mask=mask)
    if SORTED_A:
        lanes = tl.arange(0, BM)
        even = (lanes % 2) == 0
        next_rows = rows + 1
        next_valid = even & (next_rows < M)
        idx_next = tl.load(index_a + next_rows * SIA, mask=next_valid, other=-1)
        next_mask = next_valid[:, None] & (cols[None, :] < H)
        next_in_base = next_rows[:, None] * N
        next_out_base = next_rows[:, None] * H
        core_next = tl.load(
            preact + next_in_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gate_next = tl.load(
            preact + next_in_base + H + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gcore_next = tl.load(
            grad_core + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        ggate_next = tl.load(
            grad_gate + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        sig_core_next = tl.sigmoid(core_next)
        sig_gate_next = tl.sigmoid(gate_next)
        dcore_next = gcore_next * sig_core_next * (
            1.0 + core_next * (1.0 - sig_core_next)
        )
        dgate_next = ggate_next * sig_gate_next * (
            1.0 + gate_next * (1.0 - sig_gate_next)
        )
        same_next = next_valid & (idx_next == idx_a)
        pair_dcore = dcore + tl.where(same_next[:, None], dcore_next, 0.0)
        pair_dgate = dgate + tl.where(same_next[:, None], dgate_next, 0.0)
        even_mask = mask & even[:, None]
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            pair_dcore,
            mask=even_mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            pair_dgate,
            mask=even_mask,
            sem="relaxed",
        )

        prev_rows = rows - 1
        odd = ~even
        prev_valid = odd & (rows > 0)
        idx_prev = tl.load(index_a + prev_rows * SIA, mask=prev_valid, other=-2)
        same_prev = prev_valid & (idx_prev == idx_a)
        odd_mask = mask & odd[:, None] & (~same_prev[:, None])
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=odd_mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=odd_mask,
            sem="relaxed",
        )
    else:
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=mask,
            sem="relaxed",
        )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + cols[None, :], dcore, mask=mask, sem="relaxed"
    )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + H + cols[None, :],
        dgate,
        mask=mask,
        sem="relaxed",
    )


@triton.jit
def _cat4_gather_add_silu_bwd(
    grad_core,
    grad_gate,
    preact,
    index_a,
    index_b,
    index_c,
    grad_p0,
    grad_p1,
    grad_p2,
    grad_p3,
    SIA: tl.constexpr,
    SIB: tl.constexpr,
    SIC: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BLK: tl.constexpr,
    SORTED_A: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BLK)
    mask = (rows[:, None] < M) & (cols[None, :] < H)
    idx_a = tl.load(index_a + rows * SIA, mask=rows < M, other=0)
    idx_b = tl.load(index_b + rows * SIB, mask=rows < M, other=0)
    idx_c = tl.load(index_c + rows * SIC, mask=rows < M, other=0)

    in_base = rows[:, None] * N
    out_base = rows[:, None] * H
    core = tl.load(preact + in_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(preact + in_base + H + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    gcore = tl.load(grad_core + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    ggate = tl.load(grad_gate + out_base + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    sig_core = tl.sigmoid(core)
    sig_gate = tl.sigmoid(gate)
    dcore = gcore * sig_core * (1.0 + core * (1.0 - sig_core))
    dgate = ggate * sig_gate * (1.0 + gate * (1.0 - sig_gate))

    tl.store(grad_p0 + in_base + cols[None, :], dcore, mask=mask)
    tl.store(grad_p0 + in_base + H + cols[None, :], dgate, mask=mask)
    if SORTED_A:
        lanes = tl.arange(0, BM)
        even = (lanes % 2) == 0
        next_rows = rows + 1
        next_valid = even & (next_rows < M)
        idx_next = tl.load(index_a + next_rows * SIA, mask=next_valid, other=-1)
        next_mask = next_valid[:, None] & (cols[None, :] < H)
        next_in_base = next_rows[:, None] * N
        next_out_base = next_rows[:, None] * H
        core_next = tl.load(
            preact + next_in_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gate_next = tl.load(
            preact + next_in_base + H + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        gcore_next = tl.load(
            grad_core + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        ggate_next = tl.load(
            grad_gate + next_out_base + cols[None, :], mask=next_mask, other=0.0
        ).to(tl.float32)
        sig_core_next = tl.sigmoid(core_next)
        sig_gate_next = tl.sigmoid(gate_next)
        dcore_next = gcore_next * sig_core_next * (
            1.0 + core_next * (1.0 - sig_core_next)
        )
        dgate_next = ggate_next * sig_gate_next * (
            1.0 + gate_next * (1.0 - sig_gate_next)
        )
        same_next = next_valid & (idx_next == idx_a)
        pair_dcore = dcore + tl.where(same_next[:, None], dcore_next, 0.0)
        pair_dgate = dgate + tl.where(same_next[:, None], dgate_next, 0.0)
        even_mask = mask & even[:, None]
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            pair_dcore,
            mask=even_mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            pair_dgate,
            mask=even_mask,
            sem="relaxed",
        )

        prev_rows = rows - 1
        odd = ~even
        prev_valid = odd & (rows > 0)
        idx_prev = tl.load(index_a + prev_rows * SIA, mask=prev_valid, other=-2)
        same_prev = prev_valid & (idx_prev == idx_a)
        odd_mask = mask & odd[:, None] & (~same_prev[:, None])
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=odd_mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=odd_mask,
            sem="relaxed",
        )
    else:
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + cols[None, :],
            dcore,
            mask=mask,
            sem="relaxed",
        )
        tl.atomic_add(
            grad_p1 + idx_a[:, None] * N + H + cols[None, :],
            dgate,
            mask=mask,
            sem="relaxed",
        )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + cols[None, :], dcore, mask=mask, sem="relaxed"
    )
    tl.atomic_add(
        grad_p2 + idx_b[:, None] * N + H + cols[None, :],
        dgate,
        mask=mask,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_p3 + idx_c[:, None] * N + cols[None, :], dcore, mask=mask, sem="relaxed"
    )
    tl.atomic_add(
        grad_p3 + idx_c[:, None] * N + H + cols[None, :],
        dgate,
        mask=mask,
        sem="relaxed",
    )


def _launch_cat3_gather_add_silu(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = p0.shape
    h = n // 2
    # p0 may be a row-strided narrow view of a wider merged projection
    # (stride(1) must stay 1); preact is always contiguous.
    preact = torch.empty((m, n), device=p0.device, dtype=p0.dtype)
    core_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    gate_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    block = triton.next_power_of_2(h)
    _cat3_gather_add_silu_fwd[(triton.cdiv(m, _EPILOGUE_BM),)](
        p0,
        p1,
        p2,
        index_a,
        index_b,
        bias if bias is not None else p0,
        preact,
        core_act,
        gate_act,
        SIA=index_a.stride(0),
        SIB=index_b.stride(0),
        SP0=p0.stride(0),
        M=m,
        H=h,
        N=n,
        BM=_EPILOGUE_BM,
        BLK=block,
        HAS_BIAS=bias is not None,
        num_warps=4 if block >= 128 else 1,
    )
    return preact, core_act, gate_act


def _launch_cat4_gather_add_silu(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
    index_c: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = p0.shape
    h = n // 2
    # p0 may be a row-strided narrow view (stride(1)==1); preact stays
    # contiguous.
    preact = torch.empty((m, n), device=p0.device, dtype=p0.dtype)
    core_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    gate_act = torch.empty((m, h), device=p0.device, dtype=p0.dtype)
    block = triton.next_power_of_2(h)
    _cat4_gather_add_silu_fwd[(triton.cdiv(m, _EPILOGUE_BM),)](
        p0,
        p1,
        p2,
        p3,
        index_a,
        index_b,
        index_c,
        bias if bias is not None else p0,
        preact,
        core_act,
        gate_act,
        SIA=index_a.stride(0),
        SIB=index_b.stride(0),
        SIC=index_c.stride(0),
        SP0=p0.stride(0),
        M=m,
        H=h,
        N=n,
        BM=_EPILOGUE_BM,
        BLK=block,
        HAS_BIAS=bias is not None,
        num_warps=4 if block >= 128 else 1,
    )
    return preact, core_act, gate_act


class _Cat3GatherAddSilu(Function):
    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        index_a: torch.Tensor,
        index_b: torch.Tensor,
        bias: torch.Tensor | None,
        index_a_sorted: bool,
    ):
        preact, core_act, gate_act = _launch_cat3_gather_add_silu(
            p0, p1, p2, index_a, index_b, bias
        )
        ctx.save_for_backward(preact, index_a, index_b)
        ctx.p1_shape = p1.shape
        ctx.p2_shape = p2.shape
        ctx.index_strides = (index_a.stride(0), index_b.stride(0))
        ctx.index_a_sorted = bool(index_a_sorted and _SORTED_A_PAIR_BWD)
        return core_act, gate_act

    @staticmethod
    def backward(ctx, grad_core: torch.Tensor, grad_gate: torch.Tensor):
        preact, index_a, index_b = ctx.saved_tensors
        m, n = preact.shape
        h = n // 2
        block = triton.next_power_of_2(h)
        grad_p0 = torch.empty_like(preact)
        grad_p1 = torch.zeros(ctx.p1_shape, device=preact.device, dtype=preact.dtype)
        grad_p2 = torch.zeros(ctx.p2_shape, device=preact.device, dtype=preact.dtype)
        _cat3_gather_add_silu_bwd[(triton.cdiv(m, _EPILOGUE_BWD_BM),)](
            grad_core.contiguous(),
            grad_gate.contiguous(),
            preact,
            index_a,
            index_b,
            grad_p0,
            grad_p1,
            grad_p2,
            SIA=ctx.index_strides[0],
            SIB=ctx.index_strides[1],
            M=m,
            H=h,
            N=n,
            BM=_EPILOGUE_BWD_BM,
            BLK=block,
            SORTED_A=ctx.index_a_sorted,
            num_warps=4 if block >= 128 else 1,
        )
        return grad_p0, grad_p1, grad_p2, None, None, None, None


class _Cat4GatherAddSilu(Function):
    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        index_a: torch.Tensor,
        index_b: torch.Tensor,
        index_c: torch.Tensor,
        bias: torch.Tensor | None,
        index_a_sorted: bool,
    ):
        preact, core_act, gate_act = _launch_cat4_gather_add_silu(
            p0, p1, p2, p3, index_a, index_b, index_c, bias
        )
        ctx.save_for_backward(preact, index_a, index_b, index_c)
        ctx.p1_shape = p1.shape
        ctx.p2_shape = p2.shape
        ctx.p3_shape = p3.shape
        ctx.index_strides = (index_a.stride(0), index_b.stride(0), index_c.stride(0))
        ctx.index_a_sorted = bool(index_a_sorted and _SORTED_A_PAIR_BWD)
        return core_act, gate_act

    @staticmethod
    def backward(ctx, grad_core: torch.Tensor, grad_gate: torch.Tensor):
        preact, index_a, index_b, index_c = ctx.saved_tensors
        m, n = preact.shape
        h = n // 2
        block = triton.next_power_of_2(h)
        grad_p0 = torch.empty_like(preact)
        grad_p1 = torch.zeros(ctx.p1_shape, device=preact.device, dtype=preact.dtype)
        grad_p2 = torch.zeros(ctx.p2_shape, device=preact.device, dtype=preact.dtype)
        grad_p3 = torch.zeros(ctx.p3_shape, device=preact.device, dtype=preact.dtype)
        _cat4_gather_add_silu_bwd[(triton.cdiv(m, _EPILOGUE_BWD_BM),)](
            grad_core.contiguous(),
            grad_gate.contiguous(),
            preact,
            index_a,
            index_b,
            index_c,
            grad_p0,
            grad_p1,
            grad_p2,
            grad_p3,
            SIA=ctx.index_strides[0],
            SIB=ctx.index_strides[1],
            SIC=ctx.index_strides[2],
            M=m,
            H=h,
            N=n,
            BM=_EPILOGUE_BWD_BM,
            BLK=block,
            SORTED_A=ctx.index_a_sorted,
            num_warps=4 if block >= 128 else 1,
        )
        return grad_p0, grad_p1, grad_p2, grad_p3, None, None, None, None, None


def decomposed_indexed_cat3_silu_linear(
    aligned: torch.Tensor,
    gathered: torch.Tensor,
    index_a: torch.Tensor,
    index_b: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None = None,
    index_a_sorted: bool = False,
    precomputed_p0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _LAST_ERROR
    if precomputed_p0 is not None:
        # aligned is consumed only by the p0 projection; a caller that merged
        # that projection into a wider GEMM supplies p0 directly (possibly as
        # a row-strided narrow view: stride(1) must be 1).
        if not (
            precomputed_p0.is_cuda
            and precomputed_p0.stride(1) == 1
            and precomputed_p0.dim() == 2
            and precomputed_p0.shape[0] == aligned.shape[0]
            and precomputed_p0.shape[1] == weights[0].shape[0]
            and precomputed_p0.dtype == aligned.dtype
        ):
            return None
    elif not (aligned.is_cuda and aligned.is_contiguous()):
        return None
    if not (
        gathered.is_cuda
        and gathered.is_contiguous()
        and len(weights) == 3
        and all(w.is_cuda and w.is_contiguous() and w.dtype == aligned.dtype for w in weights)
    ):
        return None
    if bias is not None and (
        not bias.is_cuda
        or not bias.is_contiguous()
        or bias.dtype != aligned.dtype
        or bias.dim() != 1
    ):
        return None
    try:
        _debug_check_index("cat3 index_a", index_a, gathered.shape[0])
        _debug_check_index("cat3 index_b", index_b, gathered.shape[0])
        _debug_print_once(
            "cat3",
            "aligned",
            tuple(aligned.shape),
            aligned.stride(),
            "gathered",
            tuple(gathered.shape),
            gathered.stride(),
            "weights",
            [tuple(w.shape) for w in weights],
            "bias",
            None if bias is None else tuple(bias.shape),
        )
        p0 = (
            precomputed_p0
            if precomputed_p0 is not None
            else _project(aligned, weights[0])
        )
        p1 = _project(gathered, weights[1])
        p2 = _project(gathered, weights[2])
        return _Cat3GatherAddSilu.apply(
            p0, p1, p2, index_a, index_b, bias, bool(index_a_sorted)
        )
    except Exception as exc:
        _LAST_ERROR = repr(exc)
        if _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
            raise
        return None


def decomposed_indexed_cat4_silu_linear(
    aligned: torch.Tensor,
    gathered_a: torch.Tensor,
    index_a: torch.Tensor,
    gathered_b: torch.Tensor,
    index_b: torch.Tensor,
    index_c: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None = None,
    index_a_sorted: bool = False,
    precomputed_p0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _LAST_ERROR
    if precomputed_p0 is not None:
        if not (
            precomputed_p0.is_cuda
            and precomputed_p0.stride(1) == 1
            and precomputed_p0.dim() == 2
            and precomputed_p0.shape[0] == aligned.shape[0]
            and precomputed_p0.shape[1] == weights[0].shape[0]
            and precomputed_p0.dtype == aligned.dtype
        ):
            return None
    elif not (aligned.is_cuda and aligned.is_contiguous()):
        return None
    if not (
        gathered_a.is_cuda
        and gathered_b.is_cuda
        and gathered_a.is_contiguous()
        and gathered_b.is_contiguous()
        and len(weights) == 4
        and all(w.is_cuda and w.is_contiguous() and w.dtype == aligned.dtype for w in weights)
    ):
        return None
    if bias is not None and (
        not bias.is_cuda
        or not bias.is_contiguous()
        or bias.dtype != aligned.dtype
        or bias.dim() != 1
    ):
        return None
    try:
        _debug_check_index("cat4 index_a", index_a, gathered_a.shape[0])
        _debug_check_index("cat4 index_b", index_b, gathered_b.shape[0])
        _debug_check_index("cat4 index_c", index_c, gathered_b.shape[0])
        _debug_print_once(
            "cat4",
            "aligned",
            tuple(aligned.shape),
            aligned.stride(),
            "gathered_a",
            tuple(gathered_a.shape),
            gathered_a.stride(),
            "gathered_b",
            tuple(gathered_b.shape),
            gathered_b.stride(),
            "weights",
            [tuple(w.shape) for w in weights],
            "bias",
            None if bias is None else tuple(bias.shape),
        )
        p0 = (
            precomputed_p0
            if precomputed_p0 is not None
            else _project(aligned, weights[0])
        )
        p1 = _project(gathered_a, weights[1])
        p2 = _project(gathered_b, weights[2])
        p3 = _project(gathered_b, weights[3])
        return _Cat4GatherAddSilu.apply(
            p0,
            p1,
            p2,
            p3,
            index_a,
            index_b,
            index_c,
            bias,
            bool(index_a_sorted),
        )
    except Exception as exc:
        _LAST_ERROR = repr(exc)
        if _env_flag("MATRIS_DECOMPOSED_INDEXED_DEBUG_CHECKS", False):
            raise
        return None


def last_decomposed_indexed_cat_silu_linear_error() -> str | None:
    return _LAST_ERROR
