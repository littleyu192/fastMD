"""Tensor-native MACE inputs (no torch_geometric / AtomicData) + model loader.

Target: mace-torch 0.3.16, ScaleShiftMACE checkpoints MACE-MP-0 ("medium") and
MACE-MPA-0 ("medium-mpa-0").

What ScaleShiftMACE.forward actually reads from the input dict
(mace/modules/models.py:455-621 and mace/modules/utils.py:596-669):

    key          shape        dtype           used for
    -----------  -----------  --------------  ------------------------------------------
    positions    [N, 3]       model dtype     edge vectors; forces = -dE/dpositions
    node_attrs   [N, Z]       model dtype     one-hot over model.atomic_numbers (Z=89)
    edge_index   [2, E]       int64           [0]=sender, [1]=receiver;
                                              vec = pos[recv] - pos[send] + shift
    unit_shifts  [E, 3]       model dtype     integer image offsets S (as float);
                                              shift = S @ cell (recomputed when stress on)
    shifts       [E, 3]       model dtype     S @ cell; only read when compute_stress=False
    cell         [3G, 3]      model dtype     rows = lattice vectors, graphs stacked
    batch        [N]          int64           node -> graph index
    ptr          [G+1]        int64           ONLY ptr.numel() is used (num_graphs, static)
    head         [G]          int64           head index per graph (0 for both checkpoints)

Note: the model mutates the dict it is given (prepare_graph overwrites
data["positions"] / data["shifts"] with displaced tensors when stress is
requested, and calls positions.requires_grad_(True)).  Always pass a shallow
copy -- ``mace_forward`` does this for you.

Everything here is plain torch; nothing requires AtomicData, Batch or matscipy
(``neighbor_list_matscipy`` exists only to reproduce the reference neighbour
list bit-for-bit).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


_DTYPE_LOCK = RLock()


@contextmanager
def default_dtype(dtype):
    """Scope MACE's legacy default-dtype assumptions and always restore them."""
    with _DTYPE_LOCK:
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


# ----------------------------------------------------------------------------
# Model loading (mirrors mace_mp(...) -> MACECalculator.__init__)
# ----------------------------------------------------------------------------

_CACHE = os.path.expanduser("~/.cache/mace")
MODEL_PATHS = {
    "medium": os.path.join(_CACHE, "20231203mace128L1_epoch199model"),  # MACE-MP-0
    "mace-mp-0": os.path.join(_CACHE, "20231203mace128L1_epoch199model"),
    "medium-mpa-0": os.path.join(_CACHE, "macempa0mediummodel"),  # MACE-MPA-0
    "mace-mpa-0": os.path.join(_CACHE, "macempa0mediummodel"),
}


def resolve_model_path(model: str) -> str:
    key = str(model).lower()
    if key in MODEL_PATHS and os.path.isfile(MODEL_PATHS[key]):
        return MODEL_PATHS[key]
    if os.path.isfile(str(model)):
        return str(model)
    if key not in MODEL_PATHS:
        raise FileNotFoundError(f"MACE checkpoint does not exist: {model}")
    model = {"mace-mp-0": "medium", "mace-mpa-0": "medium-mpa-0"}.get(key, key)
    # Use MACE's downloader only for explicitly supported foundation-model names.
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    return download_mace_mp_checkpoint(model)


@dataclass
class MaceModelInfo:
    name: str
    path: str
    dtype: torch.dtype
    device: torch.device
    enable_cueq: bool
    conv_fusion: bool
    z_table: List[int]
    r_max: float
    heads: List[str]
    head_index: int
    extra: Dict = field(default_factory=dict)


def load_mace_model(
    model: str = "medium",
    device: str = "cuda",
    default_dtype: str = "float64",
    enable_cueq: bool = False,
    restore_default_dtype: bool = True,
    head: str | None = None,
) -> Tuple[torch.nn.Module, MaceModelInfo]:
    """Load a MACE foundation model exactly like
    ``mace_mp(model=..., device=..., default_dtype=..., enable_cueq=...)``.

    MACECalculator.__init__ (mace/calculators/mace.py:239-339) does:
      torch.load -> .to(device) -> .double()/.float() if dtype differs ->
      run_e3nn_to_cueq(model, device=device) (conv_fusion = device=="cuda",
      layout ir_mul, group O3_e3nn, optimize_all) -> requires_grad=False.
    ``run_e3nn_to_cueq`` calls torch.set_default_dtype(model dtype) as a side
    effect; we restore the previous default unless restore_default_dtype=False.
    """
    path = resolve_model_path(model)
    dev = torch.device(device)
    if default_dtype not in {"float64", "float32"}:
        raise ValueError("default_dtype must be 'float64' or 'float32'")
    target = {"float64": torch.float64, "float32": torch.float32}[default_dtype]
    # Keep the historical argument for internal callers; process state is always restored.
    with globals()["default_dtype"](target):
        m = torch.load(path, map_location=dev, weights_only=False).to(device=dev, dtype=target).eval()
        conv_fusion = False
        if enable_cueq:
            from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
            # Upstream 0.3.16 enables convolution fusion only for device == 'cuda'.
            # Select the index through a scoped device guard, preserving cuda:1 etc.
            with torch.cuda.device(dev) if dev.type == "cuda" else nullcontext():
                m = run_e3nn_to_cueq(m, device=dev.type).to(dev).eval()
            conv_fusion = dev.type == "cuda"
        for p in m.parameters():
            p.requires_grad_(False)

    try:
        heads = list(m.heads)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        heads = ["Default"]
    if head is not None:
        if head not in heads:
            raise ValueError(f"Unknown MACE head {head!r}; choose from {heads}")
        head_index = heads.index(head)
    elif len(heads) == 1:
        head_index = 0
    else:
        cands = [i for i, h in enumerate(heads) if h.lower() == "default"]
        if not cands:
            raise ValueError(f"no default head in {heads}")
        head_index = cands[0]
    info = MaceModelInfo(
        name=str(model),
        path=path,
        dtype=target,
        device=dev,
        enable_cueq=enable_cueq,
        conv_fusion=conv_fusion,
        z_table=[int(z) for z in m.atomic_numbers],
        r_max=float(m.r_max),
        heads=heads,
        head_index=head_index,
    )
    return m, info


# ----------------------------------------------------------------------------
# Node attributes
# ----------------------------------------------------------------------------


def z_lookup(z_table: Sequence[int], device, max_z: int = 128) -> torch.Tensor:
    """LUT atomic number -> index in z_table (-1 if absent)."""
    lut = torch.full((max_z,), -1, dtype=torch.int64, device=device)
    lut[torch.as_tensor(list(z_table), dtype=torch.int64, device=device)] = torch.arange(
        len(z_table), dtype=torch.int64, device=device
    )
    return lut


def one_hot_node_attrs(
    atomic_numbers, z_table: Sequence[int], dtype: torch.dtype, device
) -> torch.Tensor:
    """[N, len(z_table)] one-hot (same as mace.tools.to_one_hot(atomic_numbers_to_indices))."""
    z = torch.as_tensor(np.asarray(atomic_numbers), dtype=torch.int64, device=device)
    idx = z_lookup(z_table, device)[z]
    if bool((idx < 0).any()):  # build-time check only (host sync is fine here)
        raise ValueError("atomic number not in model z_table")
    oh = torch.zeros(z.shape[0], len(z_table), dtype=dtype, device=device)
    oh.scatter_(1, idx.unsqueeze(1), 1.0)
    return oh


# ----------------------------------------------------------------------------
# Neighbour lists (build-time helpers; they sync, never call inside a graph)
# ----------------------------------------------------------------------------


def neighbor_list_matscipy(positions, cell, pbc, cutoff: float):
    """Exactly mace.data.neighborhood.get_neighborhood (matscipy, CPU).
    Returns edge_index [2,E] int64 np, unit_shifts [E,3] float np."""
    from mace.data.neighborhood import get_neighborhood

    edge_index, _shifts, unit_shifts, _cell = get_neighborhood(
        positions=np.asarray(positions, dtype=np.float64),
        cutoff=float(cutoff),
        pbc=tuple(bool(p) for p in pbc),
        cell=np.array(cell, dtype=np.float64),
    )
    return edge_index.astype(np.int64), unit_shifts.astype(np.float64)


def neighbor_list_torch(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    pbc: Sequence[bool] = (True, True, True),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Brute-force periodic neighbour list on the positions' device.

    Returns (edge_index [2,E] int64, unit_shifts [E,3] positions.dtype) with the
    same convention as matscipy 'ijS' / MACE: sender=i, receiver=j,
    vec = pos[j] - pos[i] + S @ cell, d < cutoff, true self edges removed,
    sorted by (sender, receiver).  Positions need not be wrapped.
    O(N^2 * images) memory -- intended for small cells (<= few thousand atoms).
    """
    pos = positions.detach().to(torch.float64)
    c = cell.detach().reshape(3, 3).to(torch.float64)
    dev = pos.device
    pbc_t = torch.tensor([bool(p) for p in pbc], device=dev)
    # wrap positions into cell (periodic dims only) and remember the offsets
    frac = pos @ torch.linalg.inv(c)
    off = torch.where(pbc_t, torch.floor(frac), torch.zeros_like(frac))
    pos_w = pos - off @ c
    vol = torch.abs(torch.linalg.det(c))
    heights = torch.stack(
        [vol / torch.linalg.norm(torch.linalg.cross(c[(k + 1) % 3], c[(k + 2) % 3])) for k in range(3)]
    )
    nimg = [int(math.ceil(cutoff / float(h))) + 1 if p else 0 for h, p in zip(heights, pbc)]
    rng = [torch.arange(-n, n + 1, device=dev, dtype=torch.float64) for n in nimg]
    S = torch.cartesian_prod(*rng).reshape(-1, 3)  # [M,3]
    img = S @ c  # [M,3]
    d = pos_w[None, :, None, :] - pos_w[:, None, None, :] + img[None, None, :, :]
    d2 = (d * d).sum(-1)  # [N(i), N(j), M]
    mask = d2 < cutoff * cutoff
    n = pos.shape[0]
    zero_img = (S == 0).all(-1)
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    mask &= ~(eye[:, :, None] & zero_img[None, None, :])
    i, j, m = mask.nonzero(as_tuple=True)  # sorted lexicographically (i, j, m)
    unit = S[m] - off[j] + off[i]
    edge_index = torch.stack([i, j]).to(torch.int64)
    return edge_index, unit.to(positions.dtype)


# ----------------------------------------------------------------------------
# Input dict
# ----------------------------------------------------------------------------


def build_input_dict(
    positions: torch.Tensor,
    cell: torch.Tensor,
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    node_attrs: torch.Tensor,
    head: int = 0,
    batch: Optional[torch.Tensor] = None,
    num_graphs: int = 1,
) -> Dict[str, torch.Tensor]:
    """Minimal tensor dict for ScaleShiftMACE.forward (single or multi graph).

    positions [N,3], cell [3,3] or [G,3,3] (rows are lattice vectors),
    edge_index [2,E] (sender, receiver), unit_shifts [E,3], node_attrs [N,Z].
    All float tensors are cast to node_attrs.dtype; everything stays on
    positions.device.  No host sync.
    """
    dev = positions.device
    dt = node_attrs.dtype
    n = positions.shape[0]
    if batch is None:
        batch = torch.zeros(n, dtype=torch.int64, device=dev)
    cell3 = cell.to(dev, dt).reshape(-1, 3, 3)
    assert cell3.shape[0] == num_graphs, "cell must be [G,3,3]"
    unit_shifts = unit_shifts.to(dev, dt)
    edge_index = edge_index.to(dev, torch.int64)
    shifts = torch.einsum("ei,eij->ej", unit_shifts, cell3[batch[edge_index[0]]])
    return {
        "positions": positions.detach().to(dev, dt),
        "node_attrs": node_attrs.to(dev),
        "edge_index": edge_index,
        "unit_shifts": unit_shifts,
        "shifts": shifts,
        "cell": cell3.reshape(-1, 3),
        "batch": batch.to(dev, torch.int64),
        # only ptr.numel() is read by the model (num_graphs); values kept
        # consistent anyway (batch must be sorted). searchsorted: no host sync.
        "ptr": torch.searchsorted(
            batch.to(dev, torch.int64).contiguous(),
            torch.arange(num_graphs + 1, device=dev, dtype=torch.int64),
        ),
        "head": torch.full((num_graphs,), int(head), dtype=torch.int64, device=dev),
    }


def inputs_from_atoms(
    atoms,
    info: MaceModelInfo,
    cutoff: Optional[float] = None,
    nl: str = "matscipy",
) -> Dict[str, torch.Tensor]:
    """Convenience: ase.Atoms -> input dict (build time only)."""
    dev, dt = info.device, info.dtype
    cutoff = info.r_max if cutoff is None else cutoff
    pos = torch.tensor(atoms.get_positions(), dtype=dt, device=dev)
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=dt, device=dev)
    if nl == "matscipy":
        ei, us = neighbor_list_matscipy(atoms.get_positions(), np.array(atoms.get_cell()), atoms.get_pbc(), cutoff)
        ei = torch.from_numpy(ei).to(dev)
        us = torch.from_numpy(us).to(dev, dt)
    else:
        ei, us = neighbor_list_torch(pos, cell, cutoff, tuple(atoms.get_pbc()))
    na = one_hot_node_attrs(atoms.get_atomic_numbers(), info.z_table, dt, dev)
    return build_input_dict(pos, cell, ei, us, na, head=info.head_index)


# ----------------------------------------------------------------------------
# Fixed-capacity padding
# ----------------------------------------------------------------------------


def pad_edges_self_loop(
    edge_index: torch.Tensor,
    unit_shifts: torch.Tensor,
    e_cap: int,
    cell: torch.Tensor,
    r_max: float,
    pad_atom: int = 0,
    margin: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad to exactly e_cap edges with self-loops pad_atom->pad_atom whose shift is
    n * (longest lattice vector) with |shift| >= r_max*margin.  Vector length is
    finite and > r_max, so PolynomialCutoff gives exactly 0 (value and grad) and
    nothing is NaN.  No extra atoms or graphs needed.  (No host sync if e_cap and
    the image count are known; the norm computation below syncs once at build.)
    """
    e = edge_index.shape[1]
    if e > e_cap:
        raise ValueError(f"edges {e} > capacity {e_cap}")
    c = cell.reshape(3, 3)
    norms = torch.linalg.norm(c.to(torch.float64), dim=1)
    k = int(torch.argmax(norms))
    nrep = int(math.ceil(r_max * margin / float(norms[k])))
    pad = e_cap - e
    ei_pad = torch.full((2, pad), pad_atom, dtype=edge_index.dtype, device=edge_index.device)
    us_pad = torch.zeros(pad, 3, dtype=unit_shifts.dtype, device=unit_shifts.device)
    us_pad[:, k] = float(nrep)
    return torch.cat([edge_index, ei_pad], 1), torch.cat([unit_shifts, us_pad], 0)


def add_sink_graph(
    inputs: Dict[str, torch.Tensor],
    e_cap: int,
    r_max: float,
    n_sink: int = 1,
    sink_attr: str = "element0",
) -> Dict[str, torch.Tensor]:
    """MACE-calculator style padding (mace/data/padding_tools.py): append n_sink
    dummy atoms as a *separate graph* (batch=1) with its own cubic cell of edge
    2*r_max, and pad edges with sink->sink self loops of unit shift (1,0,0)
    (length 2*r_max).  Real graph energy/stress = out[...][0]; forces[:N]."""
    dev = inputs["positions"].device
    dt = inputs["positions"].dtype
    n = inputs["positions"].shape[0]
    e = inputs["edge_index"].shape[1]
    pad = e_cap - e
    if pad < 0:
        raise ValueError("capacity too small")
    z = inputs["node_attrs"].shape[1]
    na_s = torch.zeros(n_sink, z, dtype=dt, device=dev)
    if sink_attr == "element0":
        na_s[:, 0] = 1.0
    sink = n  # last real index + 1
    ei = torch.cat([inputs["edge_index"], torch.full((2, pad), sink, dtype=torch.int64, device=dev)], 1)
    us_pad = torch.zeros(pad, 3, dtype=dt, device=dev)
    us_pad[:, 0] = 1.0
    us = torch.cat([inputs["unit_shifts"], us_pad], 0)
    cell_s = torch.eye(3, dtype=dt, device=dev) * max(2.0 * r_max, 1.0)
    cell = torch.cat([inputs["cell"].reshape(-1, 3, 3)[:1], cell_s[None]], 0)
    pos = torch.cat([inputs["positions"].detach(), torch.zeros(n_sink, 3, dtype=dt, device=dev)], 0)
    batch = torch.cat([inputs["batch"], torch.ones(n_sink, dtype=torch.int64, device=dev)])
    na = torch.cat([inputs["node_attrs"], na_s], 0)
    out = build_input_dict(pos, cell, ei, us, na, head=int(inputs["head"][0]), batch=batch, num_graphs=2)
    return out


# ----------------------------------------------------------------------------
# Forward helpers
# ----------------------------------------------------------------------------


def mace_forward(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    compute_stress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Same model call as MACECalculator.calculate (mace/calculators/mace.py:635):
    model(batch_dict, compute_stress=True, training=False, ...).  Returns detached
    energy [G], forces [N,3], stress [G,3,3] (eV/A^3, 3x3)."""
    out = model(
        {**inputs, "positions": inputs["positions"].detach()},
        compute_force=True,
        compute_stress=compute_stress,
        training=False,
    )
    res = {"energy": out["energy"].detach(), "forces": out["forces"].detach()}
    if compute_stress and out["stress"] is not None:
        res["stress"] = out["stress"].detach()
    return res


def stress_to_voigt(s: torch.Tensor) -> torch.Tensor:
    """ase.stress.full_3x3_to_voigt_6_stress: [xx,yy,zz,yz,xz,xy]."""
    return torch.stack(
        [
            s[..., 0, 0],
            s[..., 1, 1],
            s[..., 2, 2],
            (s[..., 1, 2] + s[..., 2, 1]) / 2,
            (s[..., 0, 2] + s[..., 2, 0]) / 2,
            (s[..., 0, 1] + s[..., 1, 0]) / 2,
        ],
        -1,
    )


class CapturedMace:
    """forward + autograd.grad (forces & stress) captured in one torch.cuda.CUDAGraph.

    Static buffers: every tensor in ``static_inputs`` (positions, cell,
    unit_shifts, edge_index, ...).  Update them in place (``update``) and call
    ``replay()``; outputs are views of graph-owned memory (clone if you keep them).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        static_inputs: Dict[str, torch.Tensor],
        compute_stress: bool = True,
        warmup: int = 3,
        pool=None,
    ):
        self.model = model
        self.static = {k: v.detach().clone() for k, v in static_inputs.items()}
        self.compute_stress = compute_stress
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.out = self._run()
        torch.cuda.synchronize()

    def _run(self):
        return mace_forward(self.model, self.static, self.compute_stress)

    @torch.no_grad()
    def update(self, **tensors):
        for k, v in tensors.items():
            self.static[k].copy_(v)

    def replay(self) -> Dict[str, torch.Tensor]:
        self.graph.replay()
        return self.out

    def __call__(self, **tensors):
        self.update(**tensors)
        return self.replay()
