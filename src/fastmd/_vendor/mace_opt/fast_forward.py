"""FastMace: ScaleShiftMACE forward (+forces, +stress) with edge-level fusions.

Re-implementation of ``mace.modules.models.ScaleShiftMACE.forward`` (mace-torch 0.3.16)
for a single periodic graph, for both foundation checkpoints:

* MACE-MP-0 ("medium"):   2x RealAgnosticResidualInteractionBlock, Bessel x PolynomialCutoff.
* MACE-MPA-0 ("medium-mpa-0"): RealAgnosticDensityInteractionBlock +
  RealAgnosticDensityResidualInteractionBlock, Agnesi transform, ZBL pair repulsion, heads.

All GEMM-level work is done by the ORIGINAL submodules of the loaded model (node_embedding,
linear_up, skip_tp, conv_tp_weights / density_fn radial MLPs, cueq conv_tp, linear,
symmetric_contractions, product linear, readouts); nothing about them is changed.
Only edge-level / elementwise pieces are replaced, each behind its own flag (``FastFlags``):

``edge_embed``   Triton kernel pair (fwd + analytic bwd) for
                 positions/cell/displacement/unit_shifts -> edge vectors -> lengths ->
                 e3nn SH(lmax 3, normalize, 'component') + Bessel(Agnesi(len)) * PolynomialCutoff(len)
                 (+ ZBL, + density when requested).  The backward accumulates into
                 grad_positions with atomics (no sort-based ``index_put_`` backward) and
                 block-reduces grad_displacement (no bmm/einsum for the stress).
``zbl``          "fused"  : ZBL node energy inside the edge kernel (needs edge_embed)
                 "triton" : standalone Triton ZBL kernel on the lengths
                 "torch"  : ``model.pair_repulsion_fn`` (MACE code)
``density``      "torch"  : ``scatter_sum(tanh(density_fn(edge_feats)**2))`` (MACE code)
                 "tail"   : original ``density_fn`` module + Triton tanh(d^2)+scatter (fwd/bwd)
                 "fused"  : density_fn (a bias-free 10->1 linear) evaluated as an in-kernel dot
                            product inside the edge kernel.  NOTE: this replaces the call of the
                            original density_fn module (radial-MLP family) -> opt-in only.
``glue``         exact / round-off-level glue simplifications: reshape_irreps as a view (cueq
                 ir_mul layout with equal multiplicities), ``[arange, node_heads]`` -> ``[:, head]``,
                 no ``zeros`` pair term when the model has no ZBL, cached E0 sum and element
                 indices (they depend only on the fixed species), scale/shift by a python index.
``norm``         "triton": the density blocks' ``linear(message) / (density + 1)`` as one Triton
                 kernel fwd + one bwd (the linear itself is the original module); "eager": torch.
``tail``         energy tail (sum of node energies, scale/shift, graph sum, +E0) and
                 forces/stress tail (-grad, |det cell|, /V, 1e10 guard):
                 "eager"    : MACE's torch ops (stack/sum/scale_shift/scatter_sum, linalg.det)
                 "compiled" : torch.compile(dynamic=False, fullgraph=True, default mode = no
                              Inductor cudagraphs) of both tails (matris compiled_lowerings style;
                              the Inductor kernels are captured inside the external graph)
                 "triton"   : hand-written Triton kernels (1 fwd kernel + a broadcast bwd for the
                              energy tail, 1 kernel for forces+stress)

``compile_modules`` opt-in (default False): the ORIGINAL GEMM-level submodules are called through
                 ``torch.compile(module, dynamic=False)`` (Inductor default mode, no Inductor
                 cudagraphs; the kernels are captured in the caller's graph).  Same modules and
                 parameters; Inductor fuses their elementwise/copy work.  NonLinearReadoutBlock stays
                 eager (Dynamo guard failure on e3nn Irreps).  Raises Dynamo recompile limits
                 process-wide.  Variant "fast_cm".  See docs/fusion_edge.md section 6.6.

``FastFlags.off()`` reproduces the official model op-for-op (it calls MACE's own
get_symmetric_displacement / get_edge_vectors_and_lengths / SphericalHarmonics /
RadialEmbeddingBlock / ZBLBasis / scatter_sum, ``linear(message) / (density + 1)`` and the
eager tails; every flag above is at its MACE-code setting).

Runtime hook (see mace_opt/md_runtime.py):
    fn = make_model_fn("medium-mpa-0", "float64", True, variant="fast")
    energy0d, forces, stress_or_None = fn(inputs, compute_stress)

or directly from a loaded model: ``make_fast_model_fn(model, node_attrs, flags)``.

Capture rules: no host syncs in ``__call__``; call it once (warm-up) before capturing
(Triton JIT + torch.compile happen on the first call); pass the static buffers; the static
tensors are never mutated (a detached leaf view of ``positions`` is used for the gradient).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from mace.modules.utils import get_edge_vectors_and_lengths, get_symmetric_displacement
from mace.tools.scatter import scatter_sum

from .fusion import edge_fusions as EF

__all__ = ["FastFlags", "FastMace", "make_fast_model_fn", "make_model_fn", "FastModelFn", "VARIANT_FLAGS"]


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FastFlags:
    edge_embed: bool = True
    zbl: str = "fused"  # fused | triton | torch
    density: str = "tail"  # torch | tail | fused
    glue: bool = True
    tail: str = "triton"  # eager | compiled | triton
    norm: str = "triton"  # eager | triton : density blocks' linear(message) / (density + 1)
    block: int = 32  # edges per Triton program (tuned on GB300, S01: 32 x 1 warp)
    num_warps: int = 1
    bessel_rec: bool = True  # Bessel sin(k w x) by rotation recurrence (else NB libdevice sin/cos)
    # opt-in: run the ORIGINAL GEMM-level submodules (node_embedding, linear_up, skip_tp, radial MLP
    # conv_tp_weights / density_fn, linear, symmetric_contractions, product linear, readouts) through
    # torch.compile (Inductor, default mode = no Inductor cudagraphs, dynamic=False) -- the matris
    # "compiled lowerings" pattern; the compiled kernels are captured in the caller's CUDA graph.
    # Same modules and parameters; Inductor fuses their elementwise/copy work (round-off-level
    # numerical differences).  Whether this is inside the "leave GEMM-level modules unchanged"
    # scope is a policy decision -> not in the default.
    compile_modules: bool = False

    @staticmethod
    def off() -> "FastFlags":
        """Every replaceable piece set to MACE's own code (op-for-op the official model).
        ``compile_modules`` is already False by default; block/num_warps/bessel_rec only
        affect the Triton edge kernel, which is off."""
        return FastFlags(edge_embed=False, zbl="torch", density="torch", glue=False, tail="eager", norm="eager")

    def replace(self, **kw) -> "FastFlags":
        return dataclasses.replace(self, **kw)

    def resolved(self, has_zbl: bool, has_density: bool) -> "FastFlags":
        """Normalise combinations that do not apply to a model."""
        zbl = self.zbl if has_zbl else "torch"
        if zbl == "fused" and not self.edge_embed:
            zbl = "triton"
        dens = self.density if has_density else "torch"
        if dens == "fused" and not self.edge_embed:
            dens = "tail"
        for name, val, ok in (("zbl", zbl, ("fused", "triton", "torch")), ("density", dens, ("torch", "tail", "fused")),
                              ("tail", self.tail, ("eager", "compiled", "triton")),
                              ("norm", self.norm, ("eager", "triton"))):
            if val not in ok:
                raise ValueError(f"FastFlags.{name}={val!r} not in {ok}")
        norm = self.norm if has_density else "eager"
        return dataclasses.replace(self, zbl=zbl, density=dens, norm=norm)


VARIANT_FLAGS: Dict[str, FastFlags] = {
    "fast": FastFlags(),  # default: all in-scope fusions except the opt-in in-kernel density
    "fast_off": FastFlags.off(),  # re-implementation, no fusion (== official op-for-op)
    "fast_edge": FastFlags.off().replace(edge_embed=True),  # edge kernel only (ZBL/density MACE code)
    "fast_all": FastFlags(density="fused"),  # everything incl. the opt-in in-kernel density
    "fast_cm": FastFlags(compile_modules=True),  # fast + Inductor-compiled original submodules (opt-in)
}


# --------------------------------------------------------------------------- #
# Compiled tails (Inductor, no cudagraphs, static shapes)
# --------------------------------------------------------------------------- #
def _energy_tail(scale: torch.Tensor, shift: torch.Tensor, e0: torch.Tensor, es: List[torch.Tensor]):
    x = es[0]
    for t in es[1:]:
        x = x + t
    node_inter = scale * x + shift
    inter_e = node_inter.sum(dim=-1, keepdim=True)
    return inter_e, e0 + inter_e


def _force_stress_tail(g_pos: torch.Tensor, g_disp: torch.Tensor, cell: torch.Tensor):
    c = cell.reshape(3, 3)
    det = (
        c[0, 0] * (c[1, 1] * c[2, 2] - c[1, 2] * c[2, 1])
        - c[0, 1] * (c[1, 0] * c[2, 2] - c[1, 2] * c[2, 0])
        + c[0, 2] * (c[1, 0] * c[2, 1] - c[1, 1] * c[2, 0])
    )
    s = g_disp.reshape(3, 3) / det.abs()
    s = torch.where(s.abs() < 1e10, s, torch.zeros_like(s))
    return -1 * g_pos, s


_COMPILED: Dict[str, object] = {}


def _compiled(name: str):
    fn = _COMPILED.get(name)
    if fn is None:
        src = {"energy": _energy_tail, "force_stress": _force_stress_tail}[name]
        fn = torch.compile(src, dynamic=False, fullgraph=True)
        _COMPILED[name] = fn
    return fn


class _CompiledOrEager:
    """torch.compile(module, dynamic=False) (Inductor default mode, no Inductor cudagraphs); if
    Dynamo cannot compile the module (e.g. e3nn Activation: Irreps.__len__ in guards), fall back
    to the eager module permanently.  Compilation/fallback happens on the first (warm-up) call,
    which must be outside CUDA-graph capture."""

    # known Dynamo failure (guard creation calls e3nn Irreps.__len__ -> NotImplementedError):
    # not attempted, to avoid the compile attempt and its error log
    SKIP = {"NonLinearReadoutBlock": "not compiled: Dynamo guard on e3nn Irreps.__len__ (NotImplementedError)"}

    def __init__(self, mod):
        self.mod = mod
        self.failed: Optional[str] = self.SKIP.get(type(mod).__name__)
        self.fn = None if self.failed else torch.compile(mod, dynamic=False)

    def __call__(self, *args):
        if self.fn is not None:
            try:
                return self.fn(*args)
            except Exception as exc:  # noqa: BLE001  (Dynamo/Inductor failure -> eager)
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.failed = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
                self.fn = None
        return self.mod(*args)


# --------------------------------------------------------------------------- #
# FastMace
# --------------------------------------------------------------------------- #
_KINDS = {
    "RealAgnosticResidualInteractionBlock": ("residual", "avg"),
    "RealAgnosticDensityResidualInteractionBlock": ("residual", "density"),
    "RealAgnosticDensityInteractionBlock": ("skip_after", "density"),
}


class FastMace:
    """ScaleShiftMACE forward for one fixed set of atoms (``node_attrs``), single graph."""

    def __init__(self, model: torch.nn.Module, node_attrs: torch.Tensor, flags: Optional[FastFlags] = None,
                 head: int = 0):
        if type(model).__name__ != "ScaleShiftMACE":
            raise NotImplementedError(f"only ScaleShiftMACE is supported, got {type(model).__name__}")
        p0 = next(model.parameters())
        self.model = model
        self.dtype = p0.dtype
        self.device = p0.device
        self.head = int(head)
        self.has_zbl = hasattr(model, "pair_repulsion")
        self.kinds = []
        for inter in model.interactions:
            nm = type(inter).__name__
            if nm not in _KINDS:
                raise NotImplementedError(f"interaction block {nm} not supported")
            self.kinds.append(_KINDS[nm])
        has_density = any(k[1] == "density" for k in self.kinds)
        self.flags = (flags or FastFlags()).resolved(self.has_zbl, has_density)
        f = self.flags
        re_ = model.radial_embedding
        if hasattr(re_, "apply_cutoff") and not re_.apply_cutoff:
            raise NotImplementedError("radial_embedding.apply_cutoff=False (cutoff inside blocks) not supported")
        for prod in model.products:
            if getattr(prod, "use_agnostic_product", False):
                raise NotImplementedError("use_agnostic_product not supported")

        na = node_attrs.detach().to(self.device, self.dtype).contiguous()
        self.node_attrs = na
        self.n = int(na.shape[0])
        N = self.n
        with torch.no_grad():
            self.batch = torch.zeros(N, dtype=torch.int64, device=self.device)
            self.arange = torch.arange(N, dtype=torch.int64, device=self.device)
            self.node_heads = torch.full((N,), self.head, dtype=torch.int64, device=self.device)
            self.index_attrs = na.argmax(dim=-1).int()
            self.zero_disp = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
            node_e0 = model.atomic_energies_fn(na)[self.arange, self.node_heads]
            self.e0_const = scatter_sum(src=node_e0, index=self.batch, dim=0, dim_size=1).to(self.dtype)
            self.scale_h = torch.atleast_1d(model.scale_shift.scale)[self.head].clone()
            self.shift_h = torch.atleast_1d(model.scale_shift.shift)[self.head].clone()

        # product-block config (mirrors EquivariantProductBasisBlock.forward)
        self.prod_cfg = []
        for prod in model.products:
            use_cueq = mul_ir = False
            cfg = getattr(prod, "cueq_config", None)
            if cfg is not None:
                if cfg.enabled and (cfg.optimize_all or cfg.optimize_symmetric):
                    use_cueq = True
                if cfg.layout_str == "mul_ir":
                    mul_ir = True
            self.prod_cfg.append((use_cueq, mul_ir))
        # reshape_irreps as a view: only for cueq ir_mul with all multiplicities equal
        self.reshape_view = []
        for inter in model.interactions:
            r = inter.reshape
            cfg = getattr(r, "cueq_config", None)
            ok = cfg is not None and cfg.layout_str == "ir_mul" and len(set(r.muls)) == 1
            self.reshape_view.append((sum(r.dims), r.muls[0]) if ok else None)

        # fused edge embedding
        self.dens_col: Dict[int, int] = {}
        self.emb = None
        self.emb_zbl = None
        if f.edge_embed:
            self.emb = EF.EdgeEmbedding(
                model, na, fuse_density=(f.density == "fused"), fuse_zbl=(f.zbl == "fused"),
                store_lengths=(self.has_zbl and f.zbl != "fused"), block=f.block, num_warps=f.num_warps,
                bessel_recurrence=f.bessel_rec,
            )
            if f.density == "fused":
                for j, li in enumerate(self.emb.density_layers):
                    self.dens_col[li] = j
            if f.zbl == "triton":
                self.emb_zbl = self.emb
        elif f.zbl == "triton":
            self.emb_zbl = EF.EdgeEmbedding(model, na, fuse_density=False, fuse_zbl=False,
                                            block=f.block, num_warps=f.num_warps)
        self._energy_tail = self._fs_tail = None
        if f.tail == "compiled":
            self._energy_tail = _compiled("energy")
            self._fs_tail = _compiled("force_stress")
        elif f.tail == "triton":
            self._energy_tail = EF.energy_tail
            self._fs_tail = EF.force_stress_tail
        with torch.no_grad():
            self.ones1 = torch.ones(1, dtype=self.dtype, device=self.device)
        self._cmods: Optional[Dict[int, object]] = {} if f.compile_modules else None

    # ------------------------------------------------------------------ helpers
    def _m(self, mod):
        """The original submodule, or its torch.compile wrapper (compile_modules=True)."""
        if self._cmods is None:
            return mod
        fn = self._cmods.get(id(mod))
        if fn is None:
            fn = _CompiledOrEager(mod)
            self._cmods[id(mod)] = fn
        return fn

    def compiled_module_report(self) -> Dict[str, List[str]]:
        """Which submodules run compiled / fell back to eager (compile_modules=True)."""
        rep: Dict[str, List[str]] = {"compiled": [], "eager_fallback": []}
        for w in (self._cmods or {}).values():
            (rep["eager_fallback"] if w.failed else rep["compiled"]).append(type(w.mod).__name__)
        return rep

    def _reshape(self, i: int, inter, message: torch.Tensor) -> torch.Tensor:
        rv = self.reshape_view[i]
        if self.flags.glue and rv is not None:
            return message.reshape(message.shape[0], rv[0], rv[1])
        return inter.reshape(message)

    def _interaction(self, i: int, inter, node_feats, edge_attrs, edge_feats, edge_index, fused_density):
        kind, norm = self.kinds[i]
        na = self.node_attrs
        N = self.n
        sc = None
        if kind == "residual":
            sc = self._m(inter.skip_tp)(node_feats, na)
        nf = self._m(inter.linear_up)(node_feats)
        tp_weights = self._m(inter.conv_tp_weights)(edge_feats)
        density = None
        if norm == "density":
            if fused_density is not None:
                density = fused_density
            elif self.flags.density == "tail":
                density = EF.density_tail(self._m(inter.density_fn)(edge_feats), edge_index, N, self.flags.block)
            else:
                edge_density = torch.tanh(self._m(inter.density_fn)(edge_feats) ** 2)
                density = scatter_sum(src=edge_density, index=edge_index[1], dim=0, dim_size=N)
        if hasattr(inter, "conv_fusion"):
            message = inter.conv_tp(nf, edge_attrs, tp_weights, edge_index)
        else:
            mji = inter.conv_tp(nf[edge_index[0]], edge_attrs, tp_weights)
            message = scatter_sum(src=mji, index=edge_index[1], dim=0, dim_size=N)
        if norm == "density":
            if self.flags.norm == "triton":
                message = EF.density_norm(self._m(inter.linear)(message), density)
            else:
                message = self._m(inter.linear)(message) / (density + 1)
        else:
            message = self._m(inter.linear)(message) / inter.avg_num_neighbors
        if kind == "skip_after":
            message = self._m(inter.skip_tp)(message, na)
        return self._reshape(i, inter, message), sc

    def _product(self, j: int, prod, node_feats, sc):
        use_cueq, mul_ir = self.prod_cfg[j]
        if use_cueq:
            if mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = self.index_attrs if self.flags.glue else self.node_attrs.argmax(dim=-1).int()
            node_feats = self._m(prod.symmetric_contractions)(node_feats.flatten(1), index_attrs)
        else:
            node_feats = self._m(prod.symmetric_contractions)(node_feats, self.node_attrs)
        if prod.use_sc and sc is not None:
            return self._m(prod.linear)(node_feats) + sc
        return self._m(prod.linear)(node_feats)

    # ------------------------------------------------------------------ forward
    def __call__(self, inputs: Dict[str, torch.Tensor], compute_stress: bool = False):
        if self.flags.compile_modules:
            import torch._dynamo as dynamo
            limits = {name: max(getattr(dynamo.config, name), limit)
                      for name, limit in (("recompile_limit", 64), ("cache_size_limit", 64),
                                          ("accumulated_recompile_limit", 512))
                      if hasattr(dynamo.config, name)}
            with dynamo.config.patch(limits):
                return self.forward(inputs, compute_stress)
        return self.forward(inputs, compute_stress)

    def forward(self, inputs: Dict[str, torch.Tensor], compute_stress: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m = self.model
        f = self.flags
        na = self.node_attrs
        N = self.n
        edge_index = inputs["edge_index"]
        unit_shifts = inputs["unit_shifts"]
        cell = inputs["cell"]
        # leaf view of the static positions buffer (the buffer itself is never mutated)
        pos = inputs["positions"].detach().requires_grad_(True)
        disp = self.zero_disp.detach().requires_grad_(True) if compute_stress else None

        # ---------------- edge embedding ----------------
        pair = None
        fused_dens: Dict[int, torch.Tensor] = {}
        lengths = None
        if f.edge_embed:
            edge_attrs, edge_feats, lengths, nout = self.emb(pos, disp, cell, edge_index, unit_shifts)
            for li, j in self.dens_col.items():
                fused_dens[li] = nout[:, j : j + 1]
            if self.emb.fuse_zbl:
                pair = nout[:, -1]
        else:
            if compute_stress:
                pos2, shifts, _ = get_symmetric_displacement(
                    positions=pos, unit_shifts=unit_shifts, cell=cell, edge_index=edge_index,
                    num_graphs=1, batch=self.batch, displacement=disp,
                )
            else:
                pos2 = pos
                shifts = inputs.get("shifts")
                if shifts is None:
                    shifts = unit_shifts @ cell.reshape(3, 3)
            vectors, lengths = get_edge_vectors_and_lengths(positions=pos2, edge_index=edge_index, shifts=shifts)
            edge_attrs = m.spherical_harmonics(vectors)
            edge_feats, cutoff = m.radial_embedding(lengths, na, edge_index, m.atomic_numbers)
            assert cutoff is None
        if self.has_zbl and pair is None:
            if f.zbl == "triton":
                pair = EF.zbl_node_energy(self.emb_zbl, lengths, edge_index, N)
            else:
                with EF.default_dtype(self.dtype):
                    pair = m.pair_repulsion_fn(lengths, na, edge_index, m.atomic_numbers)

        # ---------------- E0 ----------------
        if f.glue:
            e0 = self.e0_const
            node_e0 = None
        else:
            node_e0 = m.atomic_energies_fn(na)[self.arange, self.node_heads]
            e0 = scatter_sum(src=node_e0, index=self.batch, dim=0, dim_size=1).to(self.dtype)

        # ---------------- interactions + products ----------------
        node_feats = self._m(m.node_embedding)(na)
        node_feats_list: List[torch.Tensor] = []
        for i, (inter, prod) in enumerate(zip(m.interactions, m.products)):
            node_feats, sc = self._interaction(i, inter, node_feats, edge_attrs, edge_feats, edge_index,
                                               fused_dens.get(i))
            node_feats = self._product(i, prod, node_feats, sc)
            node_feats_list.append(node_feats)

        # ---------------- readouts / energy ----------------
        es: List[torch.Tensor] = []
        if pair is not None:
            es.append(pair)
        elif not f.glue:
            es.append(torch.zeros_like(node_e0))
        for i, readout in enumerate(m.readouts):
            feat_idx = -1 if len(m.readouts) == 1 else i
            r = self._m(readout)(node_feats_list[feat_idx], self.node_heads)
            es.append(r[:, self.head] if f.glue else r[self.arange, self.node_heads])
        if self._energy_tail is not None:
            inter_e, total_energy = self._energy_tail(self.scale_h, self.shift_h, e0, es)
        else:
            node_inter_es = torch.sum(torch.stack(es, dim=0), dim=0)
            if f.glue:
                node_inter_es = self.scale_h * node_inter_es + self.shift_h
            else:
                node_inter_es = m.scale_shift(node_inter_es, self.node_heads)
            inter_e = scatter_sum(src=node_inter_es, index=self.batch, dim=-1, dim_size=1)
            total_energy = e0 + inter_e

        # ---------------- forces / stress (MACE get_outputs, training=False) ----------------
        ones = self.ones1 if f.glue else torch.ones_like(inter_e)
        if compute_stress:
            g_pos, g_disp = torch.autograd.grad([inter_e], [pos, disp], grad_outputs=[ones],
                                                retain_graph=False, create_graph=False, allow_unused=True)
            if g_pos is None:
                g_pos = torch.zeros_like(pos)
            if g_disp is None:
                g_disp = torch.zeros_like(disp)
            if self._fs_tail is not None:
                forces, stress = self._fs_tail(g_pos, g_disp, cell)
            else:
                cell3 = cell.view(-1, 3, 3)
                volume = torch.linalg.det(cell3).abs().unsqueeze(-1)
                stress = g_disp / volume.view(-1, 1, 1)
                stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
                forces = -1 * g_pos
                stress = stress.reshape(3, 3)
            return total_energy.reshape(()), forces, stress
        (g_pos,) = torch.autograd.grad([inter_e], [pos], grad_outputs=[ones], retain_graph=False,
                                       create_graph=False, allow_unused=True)
        if g_pos is None:
            g_pos = torch.zeros_like(pos)
        if f.tail == "triton":
            return total_energy.reshape(()), EF.force_stress_tail(g_pos, None, cell)[0], None
        return total_energy.reshape(()), -1 * g_pos, None


def make_fast_model_fn(model: torch.nn.Module, node_attrs: torch.Tensor, flags: Optional[FastFlags] = None,
                       head: int = 0) -> FastMace:
    """``model_fn(inputs, compute_stress) -> (energy 0-dim, forces [N,3], stress [3,3] | None)``."""
    return FastMace(model, node_attrs, flags, head=head)


# --------------------------------------------------------------------------- #
# Runtime hook (md_runtime.make_model_fn falls back to this for unknown variants)
# --------------------------------------------------------------------------- #
class FastModelFn:
    """Model hook with the attributes md_runtime expects (r_max, dtype, z_table, head_index).

    A FastMace instance is built lazily from ``inputs["node_attrs"]`` on the first call with a
    given node_attrs buffer; make that first call a warm-up outside CUDA-graph capture.  Instances
    are cached per buffer and never dropped (together with a reference to the buffer), so graphs
    captured earlier keep valid pointers to their per-atom tables and constants.  The tables
    assume the *contents* of a node_attrs buffer (the species) never change.
    """

    def __init__(self, model_name: str, dtype: str = "float64", cueq: bool = True, variant: str = "fast",
                 device: str = "cuda", flags: Optional[FastFlags] = None, **flag_overrides):
        from . import tensor_batch as tb

        self.model, self.info = tb.load_mace_model(model_name, device=device, default_dtype=dtype,
                                                   enable_cueq=bool(cueq))
        self.model_name = model_name
        self.variant = variant
        self.r_max = self.info.r_max
        self.dtype = self.info.dtype
        self.z_table = list(self.info.z_table)
        self.head_index = self.info.head_index
        self.cueq = bool(cueq)
        if flags is None:
            if variant not in VARIANT_FLAGS:
                raise ValueError(f"unknown fast variant {variant!r}; known: {sorted(VARIANT_FLAGS)}")
            flags = VARIANT_FLAGS[variant]
        if flag_overrides:
            flags = flags.replace(**flag_overrides)
        self.flags = flags
        # (data_ptr, shape) -> (node_attrs buffer, FastMace); entries are never removed
        self._cache: Dict[tuple, Tuple[torch.Tensor, FastMace]] = {}

    def build(self, node_attrs: torch.Tensor) -> FastMace:
        key = (node_attrs.data_ptr(), tuple(node_attrs.shape))
        fm = FastMace(self.model, node_attrs, self.flags, head=self.head_index)
        self._cache[key] = (node_attrs, fm)
        return fm

    def __call__(self, inputs: Dict[str, torch.Tensor], compute_stress: bool):
        na = inputs["node_attrs"]
        hit = self._cache.get((na.data_ptr(), tuple(na.shape)))
        if hit is None:
            if torch.cuda.is_current_stream_capturing():
                # building packs parameters with host-side checks (syncs) -> never inside capture
                raise RuntimeError("FastModelFn: first call with this node_attrs buffer happens inside "
                                   "CUDA-graph capture; call it once (warm-up) outside capture first")
            fm = self.build(na)
        else:
            fm = hit[1]
        return fm(inputs, bool(compute_stress))


def make_model_fn(model_name: str, dtype: str = "float64", cueq: bool = True, variant: str = "fast",
                  device: str = "cuda", **kwargs) -> FastModelFn:
    return FastModelFn(model_name, dtype, cueq, variant=variant, device=device, **kwargs)
