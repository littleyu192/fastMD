"""Triton kernels for the MACE density tail (RealAgnosticDensity*InteractionBlock).

MACE (blocks.py, density blocks):

    edge_density = torch.tanh(density_fn(edge_feats) ** 2)          # [E,1]
    density      = scatter_sum(edge_density, receiver, dim=0, dim_size=N)

``density_fn`` (the e3nn FullyConnectedNet[NB,1] module) is NOT touched: it is called by
the caller exactly as in MACE; only the elementwise tail + scatter is fused:

    fwd:  out[rcv[e]] += tanh(d[e] * d[e])                           (relaxed fp atomics)
    bwd:  g_d[e] = g_out[rcv[e]] * (1 - tanh(d^2)^2) * (2 d[e])     (gather, no atomics)

Numerics: d*d == torch ``d ** 2`` (pow special case), tanh is libdevice (bitwise equal to
torch on this build in fp32 and fp64), and the backward multiplies in the same order as
torch's tanh_backward / pow_backward.  The scatter uses atomics exactly like
``scatter_add_``; summation order is therefore nondeterministic in both.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _density_tail_fwd_kernel(d_ptr, ei_ptr, out_ptr, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    rcv = tl.load(ei_ptr + E + offs, mask=m, other=0)
    d = tl.load(d_ptr + offs, mask=m, other=0.0)
    ed = libdevice.tanh(d * d)
    tl.atomic_add(out_ptr + rcv, ed, mask=m, sem="relaxed")


@triton.jit
def _density_tail_bwd_kernel(d_ptr, ei_ptr, gout_ptr, gd_ptr, E, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    rcv = tl.load(ei_ptr + E + offs, mask=m, other=0)
    d = tl.load(d_ptr + offs, mask=m, other=0.0)
    g = tl.load(gout_ptr + rcv, mask=m, other=0.0)
    th = libdevice.tanh(d * d)
    gd = g * (1.0 - th * th) * (2.0 * d)
    tl.store(gd_ptr + offs, gd, mask=m)


# --------------------------------------------------------------------------- energy / force tails
@triton.jit
def _energy_tail_fwd_kernel(
    e0_ptr, e1_ptr, e2_ptr, s0, s1, s2, scale_ptr, shift_ptr, ez_ptr, out_ptr, N,
    NUM: tl.constexpr, BLOCK: tl.constexpr,
):
    """out[0] = sum_i (scale * (es_0[i] + es_1[i] (+ es_2[i])) + shift);  out[1] = ez + out[0].

    (ScaleShiftMACE: stack(node_es).sum(0) -> scale_shift -> scatter_sum over the one graph
    -> + E0.)  Single program; es_k may be strided views (stride s_k)."""
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(e0_ptr + offs * s0, mask=m, other=0.0)
    if NUM > 1:
        x = x + tl.load(e1_ptr + offs * s1, mask=m, other=0.0)
    if NUM > 2:
        x = x + tl.load(e2_ptr + offs * s2, mask=m, other=0.0)
    scale = tl.load(scale_ptr)
    shift = tl.load(shift_ptr)
    node = tl.where(m, scale * x + shift, 0.0)
    tot = tl.sum(node, axis=0)
    tl.store(out_ptr + 0, tot)
    tl.store(out_ptr + 1, tl.load(ez_ptr) + tot)


@triton.jit
def _force_stress_kernel(gpos_ptr, gdisp_ptr, cell_ptr, f_ptr, s_ptr, N3,
                         STRESS: tl.constexpr, BLOCK: tl.constexpr):
    """forces = -g_pos;  stress = g_disp / |det(cell)| with MACE's |s| < 1e10 guard."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N3
    g = tl.load(gpos_ptr + offs, mask=m, other=0.0)
    tl.store(f_ptr + offs, -g, mask=m)
    if STRESS:
        if pid == 0:
            c00 = tl.load(cell_ptr + 0)
            c01 = tl.load(cell_ptr + 1)
            c02 = tl.load(cell_ptr + 2)
            c10 = tl.load(cell_ptr + 3)
            c11 = tl.load(cell_ptr + 4)
            c12 = tl.load(cell_ptr + 5)
            c20 = tl.load(cell_ptr + 6)
            c21 = tl.load(cell_ptr + 7)
            c22 = tl.load(cell_ptr + 8)
            det = c00 * (c11 * c22 - c12 * c21) - c01 * (c10 * c22 - c12 * c20) + c02 * (c10 * c21 - c11 * c20)
            vol = tl.abs(det)
            o9 = tl.arange(0, 16)
            m9 = o9 < 9
            gd = tl.load(gdisp_ptr + o9, mask=m9, other=0.0)
            s = libdevice.div_rn(gd, vol)
            s = tl.where(tl.abs(s) < 1e10, s, 0.0)
            tl.store(s_ptr + o9, s, mask=m9)


# --------------------------------------------------------------------------- density normalisation
@triton.jit
def _density_norm_fwd_kernel(x_ptr, d_ptr, out_ptr, F, BLOCK_F: tl.constexpr):
    """out[i, :] = x[i, :] / (d[i] + 1)   (MACE density blocks: linear(message) / (density + 1))."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_F)
    m = offs < F
    x = tl.load(x_ptr + row * F + offs, mask=m, other=0.0)
    o = tl.load(d_ptr + row) + 1.0
    tl.store(out_ptr + row * F + offs, libdevice.div_rn(x, o), mask=m)


@triton.jit
def _density_norm_bwd_kernel(x_ptr, d_ptr, g_ptr, gx_ptr, gd_ptr, F, BLOCK_F: tl.constexpr):
    """gx = g / (d+1);  gd[i] = sum_f (-g x) / ((d+1)(d+1))   (torch div backward formulas)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_F)
    m = offs < F
    x = tl.load(x_ptr + row * F + offs, mask=m, other=0.0)
    g = tl.load(g_ptr + row * F + offs, mask=m, other=0.0)
    o = tl.load(d_ptr + row) + 1.0
    tl.store(gx_ptr + row * F + offs, libdevice.div_rn(g, o), mask=m)
    gd = tl.sum(tl.where(m, libdevice.div_rn(-g * x, o * o), 0.0), axis=0)
    tl.store(gd_ptr + row, gd)
