"""Edge-level fusions for MACE (ScaleShiftMACE, mace-torch 0.3.16).

Public API
----------
``EdgeEmbedding(model, node_attrs, *, fuse_density=True, fuse_zbl=True, store_lengths=False,
                block=32, num_warps=1, bessel_recurrence=True)``
    Built once from a (frozen) MACE model and the fixed per-atom one-hot.  Calling it::

        edge_attrs, edge_feats, lengths, node_out = emb(positions, displacement, cell,
                                                         edge_index, unit_shifts)

    returns what the MACE forward computes from these inputs (to fp round-off):

    * ``edge_attrs``  [E,16]  = model.spherical_harmonics(vectors)
    * ``edge_feats``  [E,NB]  = model.radial_embedding(lengths, ...)[0]  (Bessel x cutoff, Agnesi)
    * ``lengths``     [E,1] (only when ``store_lengths``; else an empty tensor)
    * ``node_out``    [N, ND + ZBL]: columns ``0..ND-1`` = ``scatter_sum(tanh(density_fn_l(edge_feats)^2), receiver)``
      for every interaction l that has a ``density_fn`` (when ``fuse_density``); last column =
      ZBL node energy ``pair_repulsion_fn(lengths, ...)`` (when ``fuse_zbl`` and the model has ZBL).

    ``positions`` and ``displacement`` are differentiable (analytic Triton backward, atomic
    scatter into ``grad_positions``; block-reduced symmetric ``grad_displacement``).
    ``displacement`` may be ``None`` (no stress).  Only a single graph (cell [3,3]) is supported.
    Edge order is arbitrary (receiver-sorted or not).

``zbl_node_energy(emb, lengths, edge_index, n)``  standalone fused ZBL (autograd on ``lengths``).
``density_tail(d, edge_index, n)``               ``scatter_sum(tanh(d**2), receiver)`` (d = density_fn output).
``energy_tail(scale, shift, e0, es)``             Triton ScaleShiftMACE energy tail (autograd).
``force_stress_tail(g_pos, g_disp, cell)``        -g_pos and g_disp/|det cell| (no autograd).
``reference_edge_embed(model, ...)``              the unfused torch path (MACE's own functions/modules).

Capture safety: no host syncs; all parameters are packed into device tensors at build time
(outside capture).  Triton kernels are JIT-compiled on first call -> call once (warm-up)
before ``torch.cuda.graph``.  The atomics make the summation order nondeterministic (the
official model's scatter_add / cueq kernels are nondeterministic too); measured spread in
docs/fusion_edge.md.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton

from mace.modules.utils import get_edge_vectors_and_lengths, get_symmetric_displacement
from mace.tools.scatter import scatter_sum

from . import triton_edge_embed as K
from . import triton_edge_tail as KT

L = K.LAYOUT

__all__ = [
    "EdgeEmbedding",
    "zbl_node_energy",
    "density_tail",
    "energy_tail",
    "density_norm",
    "force_stress_tail",
    "reference_geometry",
    "reference_edge_embed",
    "default_dtype",
    "EDGE_BLOCK",
]

EDGE_BLOCK = 32
_NUM_WARPS = 1


def _poly_coeffs(p: int):
    """(P as int, C1, C2, C3) of MACE PolynomialCutoff: 1 - C1 u^p + C2 u^(p+1) - C3 u^(p+2)."""
    pi = int(p)
    pf = float(pi)
    return pi, (pf + 1.0) * (pf + 2.0) / 2.0, pf * (pf + 2.0), pf * (pf + 1.0) / 2.0


class EdgeEmbedding(torch.nn.Module):
    """Fused edge geometry + SH + radial basis (+ density, + ZBL) for one MACE model."""

    def __init__(
        self,
        model: torch.nn.Module,
        node_attrs: torch.Tensor,
        *,
        fuse_density: bool = True,
        fuse_zbl: bool = True,
        store_lengths: bool = False,
        block: int = EDGE_BLOCK,
        num_warps: int = _NUM_WARPS,
        bessel_recurrence: bool = True,
    ):
        super().__init__()
        dev = node_attrs.device
        dt = next(model.parameters()).dtype
        self.dtype = dt
        self.block = int(block)
        self.num_warps = int(num_warps)
        sh = model.spherical_harmonics
        if not (sh.normalize and sh.normalization == "component" and sh._lmax == 3 and sh._is_range_lmax):
            raise NotImplementedError("fused SH supports lmax=3, normalize=True, 'component' only")
        re = model.radial_embedding
        bf = re.bessel_fn
        if type(bf).__name__ != "BesselBasis":
            raise NotImplementedError("fused radial supports BesselBasis only")
        if hasattr(re, "apply_cutoff") and not re.apply_cutoff:
            raise NotImplementedError("apply_cutoff=False (cutoff applied inside blocks) not supported")
        self.nb = int(bf.bessel_weights.numel())
        if self.nb > 16:
            raise NotImplementedError("num_bessel > 16")
        self.nb_pad = 16
        self.agnesi = hasattr(re, "distance_transform")
        if self.agnesi and type(re.distance_transform).__name__ != "AgnesiTransform":
            raise NotImplementedError("only Agnesi distance transform is supported")
        p_cut = int(re.cutoff_fn.p)
        self.P, self.C1, self.C2, self.C3 = _poly_coeffs(p_cut)
        self.has_zbl = hasattr(model, "pair_repulsion")
        self.fuse_zbl = bool(fuse_zbl and self.has_zbl)
        self.store_lengths = bool(store_lengths)
        # density layers (MPA-0 style blocks)
        self.density_layers = [i for i, inter in enumerate(model.interactions) if hasattr(inter, "density_fn")]
        self.fuse_density = bool(fuse_density and len(self.density_layers) > 0)
        self.nd = len(self.density_layers) if self.fuse_density else 0
        if self.nd > 4:
            raise NotImplementedError("more than 4 density layers")
        self.no = self.nd + (1 if self.fuse_zbl else 0)

        with torch.no_grad():
            prm = torch.zeros(L["SIZE"], dtype=dt, device=dev)
            prm[L["RMAX"]] = re.cutoff_fn.r_max.to(dt)
            prm[L["BPREF"]] = bf.prefactor.to(dt)
            prm[L["BW"] : L["BW"] + self.nb] = bf.bessel_weights.to(dt)
            if self.agnesi:
                ag = re.distance_transform
                prm[L["AG_A"]] = ag.a.to(dt)
                prm[L["AG_Q"]] = ag.q.to(dt)
                prm[L["AG_QMP"]] = (ag.q - ag.p).to(dt)
                prm[L["AG_P"]] = ag.p.to(dt)
            if self.has_zbl:
                zb = model.pair_repulsion_fn
                prm[L["ZBL_APREF"]] = (zb.a_prefactor * 0.529).to(dt)
                prm[L["ZBL_C0"] : L["ZBL_C0"] + 4] = zb.c.to(dt)
                zp = int(zb.p)
                self.ZP, self.ZC1, self.ZC2, self.ZC3 = _poly_coeffs(zp)
            else:
                self.ZP, self.ZC1, self.ZC2, self.ZC3 = _poly_coeffs(5)
            if self.fuse_density:
                for j, li in enumerate(self.density_layers):
                    lay = model.interactions[li].density_fn.layer0
                    if lay.act is not None or len(model.interactions[li].density_fn.hs) != 2:
                        raise NotImplementedError("density_fn must be a single linear layer")
                    w = lay.weight / (lay.h_in * lay.var_in / lay.var_out) ** 0.5  # [NB,1], as e3nn
                    prm[L["WD"] + j * self.nb_pad : L["WD"] + j * self.nb_pad + self.nb] = w[:, 0].to(dt)
            # per-node table
            z_table = model.atomic_numbers
            idx = torch.argmax(node_attrs, dim=1)
            Z = z_table[idx].to(torch.int64)
            nodef = torch.zeros(node_attrs.shape[0], L["NODE_COLS"], dtype=dt, device=dev)
            if self.agnesi:
                nodef[:, 0] = re.distance_transform.covalent_radii[Z].to(dt)
            if self.has_zbl:
                zb = model.pair_repulsion_fn
                nodef[:, 1] = zb.covalent_radii[Z].to(dt)
                nodef[:, 2] = torch.pow(Z, zb.a_exp).to(dt)
                nodef[:, 3] = Z.to(dt)
            # Bessel weights uniformly spaced (w_k == k * w_1)?  -> rotation recurrence in-kernel
            bw = bf.bessel_weights.to(dt)
            kk = torch.arange(1, self.nb + 1, dtype=dt, device=bw.device)
            tol = 1e-12 if dt == torch.float64 else 1e-6
            self.bessel_rec = bool(bessel_recurrence) and bool(
                torch.allclose(bw, bw[0] * kk, rtol=tol, atol=0.0)
            )
        self.register_buffer("prm", prm, persistent=False)
        self.register_buffer("nodef", nodef, persistent=False)

    # ------------------------------------------------------------------
    def constexprs(self):
        return dict(
            AGNESI=self.agnesi, ND=self.nd, ZBL=self.fuse_zbl, NO=max(self.no, 1), BESSEL_REC=self.bessel_rec,
            NB=self.nb, NB_PAD=self.nb_pad,
            P=self.P, C1=self.C1, C2=self.C2, C3=self.C3,
            ZP=self.ZP, ZC1=self.ZC1, ZC2=self.ZC2, ZC3=self.ZC3,
        )

    def forward(
        self,
        positions: torch.Tensor,
        displacement: Optional[torch.Tensor],
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        unit_shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _EdgeEmbedFn.apply(positions, displacement, cell, edge_index, unit_shifts, self)


class _EdgeEmbedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, disp, cell, edge_index, unit_shifts, emb: EdgeEmbedding):
        # kernels index flat row-major buffers; copy only if a caller passes a strided view
        # (no-op for the runtime's static buffers)
        pos_c = pos.detach()
        if not pos_c.is_contiguous():
            pos_c = pos_c.contiguous()
        if not edge_index.is_contiguous():
            edge_index = edge_index.contiguous()
        if not unit_shifts.is_contiguous():
            unit_shifts = unit_shifts.contiguous()
        cell_c = cell.detach().reshape(9)
        if not cell_c.is_contiguous():
            cell_c = cell_c.contiguous()
        E = edge_index.shape[1]
        N = pos_c.shape[0]
        dt = pos_c.dtype
        dev = pos_c.device
        attrs = torch.empty(E, 16, dtype=dt, device=dev)
        feats = torch.empty(E, emb.nb, dtype=dt, device=dev)
        lengths = torch.empty(E, 1, dtype=dt, device=dev) if emb.store_lengths else pos_c.new_empty(0)
        nout = torch.zeros(N, emb.no, dtype=dt, device=dev) if emb.no > 0 else pos_c.new_empty(0)
        dummy = emb.prm  # valid pointer for arguments disabled by constexpr flags
        has_disp = disp is not None
        disp_c = disp.detach().reshape(9) if has_disp else cell_c
        grid = (triton.cdiv(E, emb.block),)
        K._edge_embed_fwd_kernel[grid](
            pos_c, cell_c, disp_c, edge_index, unit_shifts, emb.nodef, emb.prm,
            attrs, feats, lengths if emb.store_lengths else dummy, nout if emb.no > 0 else dummy, E,
            HAS_DISP=has_disp, STORE_LEN=emb.store_lengths,
            BLOCK=emb.block, num_warps=emb.num_warps, **emb.constexprs(),
        )
        ctx.save_for_backward(pos_c, cell_c, disp_c, edge_index, unit_shifts)
        ctx.emb = emb
        ctx.has_disp = has_disp
        ctx.set_materialize_grads(False)
        if not emb.store_lengths:
            ctx.mark_non_differentiable(lengths)
        if emb.no == 0:
            ctx.mark_non_differentiable(nout)
        return attrs, feats, lengths, nout

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_attrs, g_feats, g_len, g_nout):
        pos_c, cell_c, disp_c, edge_index, unit_shifts = ctx.saved_tensors
        emb: EdgeEmbedding = ctx.emb
        E = edge_index.shape[1]
        N = pos_c.shape[0]
        need_pos = ctx.needs_input_grad[0]
        need_disp = ctx.has_disp and ctx.needs_input_grad[1]
        if not (need_pos or need_disp):
            return None, None, None, None, None, None
        buf = torch.zeros(N * 3 + 9, dtype=pos_c.dtype, device=pos_c.device)
        gpos = buf[: N * 3]
        gdisp = buf[N * 3 :]
        dummy = emb.prm  # valid pointer for arguments disabled by constexpr flags

        def _c(t):
            return t if (t is None or t.is_contiguous()) else t.contiguous()

        g_attrs, g_feats, g_len, g_nout = _c(g_attrs), _c(g_feats), _c(g_len), _c(g_nout)
        grid = (triton.cdiv(E, emb.block),)
        K._edge_embed_bwd_kernel[grid](
            pos_c, cell_c, disp_c, edge_index, unit_shifts, emb.nodef, emb.prm,
            g_attrs if g_attrs is not None else dummy,
            g_feats if g_feats is not None else dummy,
            g_len if (g_len is not None and emb.store_lengths) else dummy,
            g_nout if (g_nout is not None and emb.no > 0) else dummy,
            gpos, gdisp, E,
            HAS_DISP=ctx.has_disp,
            HAS_GA=g_attrs is not None, HAS_GF=g_feats is not None,
            HAS_GL=(g_len is not None and emb.store_lengths),
            HAS_GN=(g_nout is not None and emb.no > 0),
            NEED_DISP_GRAD=need_disp,
            BLOCK=emb.block, num_warps=emb.num_warps, **emb.constexprs(),
        )
        return (
            gpos.view(N, 3) if need_pos else None,
            gdisp.view(1, 3, 3) if need_disp else None,
            None, None, None, None,
        )


class _ZBLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lengths, edge_index, emb: EdgeEmbedding, n_nodes: int):
        ln = lengths.detach().reshape(-1)
        if not ln.is_contiguous():
            ln = ln.contiguous()
        E = ln.shape[0]
        out = torch.zeros(n_nodes, dtype=ln.dtype, device=ln.device)
        grid = (triton.cdiv(E, emb.block),)
        K._zbl_fwd_kernel[grid](
            ln, edge_index, emb.nodef, emb.prm, out, E,
            ZP=emb.ZP, ZC1=emb.ZC1, ZC2=emb.ZC2, ZC3=emb.ZC3,
            BLOCK=emb.block, num_warps=emb.num_warps,
        )
        ctx.save_for_backward(ln, edge_index)
        ctx.emb = emb
        ctx.shape = lengths.shape
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_out):
        ln, edge_index = ctx.saved_tensors
        emb = ctx.emb
        E = ln.shape[0]
        g_out = g_out.contiguous()
        glen = torch.empty_like(ln)
        grid = (triton.cdiv(E, emb.block),)
        K._zbl_bwd_kernel[grid](
            ln, edge_index, emb.nodef, emb.prm, g_out, glen, E,
            ZP=emb.ZP, ZC1=emb.ZC1, ZC2=emb.ZC2, ZC3=emb.ZC3,
            BLOCK=emb.block, num_warps=emb.num_warps,
        )
        return glen.view(ctx.shape), None, None, None


def zbl_node_energy(emb: EdgeEmbedding, lengths: torch.Tensor, edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Fused ZBL (== model.pair_repulsion_fn(lengths, node_attrs, edge_index, atomic_numbers))."""
    if not emb.has_zbl:
        raise ValueError("model has no pair repulsion")
    return _ZBLFn.apply(lengths, edge_index, emb, n_nodes)


class _DensityTailFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, edge_index, n_nodes: int, block: int):
        dc = d.detach().reshape(-1)
        if not dc.is_contiguous():
            dc = dc.contiguous()
        E = dc.shape[0]
        out = torch.zeros(n_nodes, 1, dtype=dc.dtype, device=dc.device)
        KT._density_tail_fwd_kernel[(triton.cdiv(E, block),)](dc, edge_index, out, E, BLOCK=block, num_warps=max(1, block // 32))
        ctx.save_for_backward(dc, edge_index)
        ctx.shape = d.shape
        ctx.block = block
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_out):
        dc, edge_index = ctx.saved_tensors
        E = dc.shape[0]
        g_out = g_out.contiguous()
        gd = torch.empty_like(dc)
        KT._density_tail_bwd_kernel[(triton.cdiv(E, ctx.block),)](
            dc, edge_index, g_out, gd, E, BLOCK=ctx.block, num_warps=max(1, ctx.block // 32)
        )
        return gd.view(ctx.shape), None, None, None


def density_tail(d: torch.Tensor, edge_index: torch.Tensor, n_nodes: int, block: int = EDGE_BLOCK) -> torch.Tensor:
    """``scatter_sum(tanh(d ** 2), edge_index[1], dim=0, dim_size=n_nodes)`` for d [E,1] -> [N,1]
    (one Triton kernel forward, one backward; ``d`` is the output of the original ``density_fn``)."""
    if not edge_index.is_contiguous():
        edge_index = edge_index.contiguous()
    return _DensityTailFn.apply(d, edge_index, n_nodes, block)


class _EnergyTailFn(torch.autograd.Function):
    """inter_e = sum_i(scale*(sum_k es_k[i]) + shift) (shape [1]), total = e0 + inter_e.

    Backward: d inter_e / d es_k[i] = scale for every k, i -> one broadcast (expand) of a
    1-element tensor per input (no per-node kernels)."""

    @staticmethod
    def forward(ctx, scale_h, shift_h, e0, *es):
        n = es[0].shape[0]
        out = torch.empty(2, dtype=es[0].dtype, device=es[0].device)
        ptrs = list(es) + [es[0]] * (3 - len(es))
        strides = [t.stride(0) for t in ptrs]
        block = max(32, triton.next_power_of_2(n))
        KT._energy_tail_fwd_kernel[(1,)](
            ptrs[0], ptrs[1], ptrs[2], strides[0], strides[1], strides[2], scale_h, shift_h, e0, out, n,
            NUM=len(es), BLOCK=block, num_warps=4,
        )
        ctx.save_for_backward(scale_h)
        ctx.n = n
        ctx.num = len(es)
        ctx.set_materialize_grads(False)
        return out[0:1], out[1:2]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_inter, g_total):
        (scale_h,) = ctx.saved_tensors
        g = g_inter
        if g_total is not None:
            g = g_total if g is None else g + g_total
        if g is None:
            return (None,) * (3 + ctx.num)
        gs = (scale_h * g).expand(ctx.n)
        return (None, None, None) + (gs,) * ctx.num


def energy_tail(scale_h, shift_h, e0, es):
    """Hand-written Triton version of the ScaleShiftMACE energy tail.

    The kernel takes <= 3 node-energy terms (MPA-0: ZBL + 2 readouts).  With more terms (models
    with > 2 interactions) they are first summed exactly as MACE does
    (``torch.sum(torch.stack(es), dim=0)``) and passed as one term."""
    es = list(es)
    assert len(es) >= 1
    if len(es) > 3:
        es = [torch.sum(torch.stack(es, dim=0), dim=0)]
    return _EnergyTailFn.apply(scale_h, shift_h, e0, *es)


class _DensityNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, d):
        xc = x.contiguous()
        dc = d.reshape(-1).contiguous()
        n, f = xc.shape
        out = torch.empty_like(xc)
        bf = triton.next_power_of_2(f)
        nw = 4 if bf <= 1024 else 8
        KT._density_norm_fwd_kernel[(n,)](xc, dc, out, f, BLOCK_F=bf, num_warps=nw)
        ctx.save_for_backward(xc, dc)
        ctx.dshape = d.shape
        ctx.cfg = (bf, nw)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        xc, dc = ctx.saved_tensors
        n, f = xc.shape
        g = g.contiguous()
        gx = torch.empty_like(xc)
        gd = torch.empty_like(dc)
        bf, nw = ctx.cfg
        KT._density_norm_bwd_kernel[(n,)](xc, dc, g, gx, gd, f, BLOCK_F=bf, num_warps=nw)
        return gx, gd.view(ctx.dshape)


def density_norm(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``x / (d + 1)`` for x [N,F], d [N,1] (one Triton kernel forward, one backward)."""
    assert x.dim() == 2 and d.shape == (x.shape[0], 1)
    return _DensityNormFn.apply(x, d)


def force_stress_tail(g_pos: torch.Tensor, g_disp: Optional[torch.Tensor], cell: torch.Tensor):
    """forces = -g_pos; stress = g_disp/|det(cell)| (1e10 guard) in one Triton kernel (no grad)."""
    n3 = g_pos.numel()
    forces = torch.empty_like(g_pos)
    stress = torch.empty(3, 3, dtype=g_pos.dtype, device=g_pos.device) if g_disp is not None else None
    block = 512
    gp = g_pos if g_pos.is_contiguous() else g_pos.contiguous()
    KT._force_stress_kernel[(triton.cdiv(n3, block),)](
        gp, g_disp if g_disp is not None else gp, cell, forces, stress if stress is not None else forces, n3,
        STRESS=g_disp is not None, BLOCK=block, num_warps=4,
    )
    return forces, stress


# ----------------------------------------------------------------------
# Reference (unfused) path: MACE's own functions and modules
# ----------------------------------------------------------------------


class default_dtype:
    """Context manager: temporarily ``torch.set_default_dtype(dtype)`` (host-only, capture safe).

    MACE's ZBLBasis computes ``14.3996 * Z_u * Z_v`` with int64 Z, i.e. in the *global default*
    dtype; ``MACECalculator`` sets the default dtype to the model dtype, so the reference path
    must do the same to reproduce the calculator bit-for-bit.
    """

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.prev = None

    def __enter__(self):
        self.prev = torch.get_default_dtype()
        if self.prev != self.dtype:
            torch.set_default_dtype(self.dtype)
        return self

    def __exit__(self, *exc):
        if self.prev != self.dtype:
            torch.set_default_dtype(self.prev)
        return False


def reference_geometry(positions, displacement, cell, edge_index, unit_shifts):
    """prepare_graph() geometry for one graph.  Returns (vectors [E,3], lengths [E,1])."""
    n = positions.shape[0]
    batch = torch.zeros(n, dtype=torch.int64, device=positions.device)
    if displacement is not None:
        pos2, shifts, _ = get_symmetric_displacement(
            positions=positions,
            unit_shifts=unit_shifts,
            cell=cell.reshape(-1, 3),
            edge_index=edge_index,
            num_graphs=1,
            batch=batch,
            displacement=displacement,
        )
    else:
        pos2 = positions
        shifts = torch.einsum("be,bec->bc", unit_shifts, cell.reshape(-1, 3, 3)[batch[edge_index[0]]])
    return get_edge_vectors_and_lengths(positions=pos2, edge_index=edge_index, shifts=shifts)


def reference_edge_embed(model, node_attrs, positions, displacement, cell, edge_index, unit_shifts):
    """Unfused torch reference returning (edge_attrs, edge_feats, lengths, densities[list], zbl[N] or None)."""
    vectors, lengths = reference_geometry(positions, displacement, cell, edge_index, unit_shifts)
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, _cut = model.radial_embedding(lengths, node_attrs, edge_index, model.atomic_numbers)
    dens = []
    for inter in model.interactions:
        if hasattr(inter, "density_fn"):
            ed = torch.tanh(inter.density_fn(edge_feats) ** 2)
            dens.append(scatter_sum(src=ed, index=edge_index[1], dim=0, dim_size=node_attrs.shape[0]))
    zbl = None
    if hasattr(model, "pair_repulsion"):
        zbl = model.pair_repulsion_fn(lengths, node_attrs, edge_index, model.atomic_numbers)
    return edge_attrs, edge_feats, lengths, dens, zbl
