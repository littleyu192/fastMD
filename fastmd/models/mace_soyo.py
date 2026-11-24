# Adapted from MD main@4e246bcc6686792d867855de32a1e2f87283b6a8; BSD-3-Clause (see LICENSE).
"""MACE-SOYO behind the adapter protocol: the upstream ``MACESoyo`` model with an energy tap.

MACE-SOYO (the ``mace_soyo`` package, MIT) is a multi-head potential on the
MACE-MH-1 architecture, built from cuEquivariance modules (attention-weighted
edge weights, an equivariant RMSNorm and a heavier readout). It is optional: this
module imports it only when a model is loaded, so ``fastmd`` imports without it.
The adapter builds the upstream module with upstream's own checkpoint loader
(``mace_soyo.utils.model_checkpoint.load_model_from_checkpoint``) and calls
its own energy function, ``MACESoyo.compute_energy_from_inputs`` (the function
upstream's AOTInductor export traces); nothing of MACE-SOYO is copied here.
What the adapter declares and does:

* **Topology**: directed edges within the model's cutoff (6 A for the released
  checkpoints), no triplets, so the builders run edges only.
* **Padding**: far self-loops (``PaddingRule("far_self_loops")``). Every edge
  term of the model is multiplied by an exact cutoff, the polynomial envelope
  times ``r < range``: the tensor-product weights and the edge density at the
  model's cutoff, the ZBL pair repulsion at the sum of the two covalent radii
  (at most 5.2 A, and clamped to the cutoff where the cutoff is shorter), so a
  loop at least twice the cutoff long adds exactly nothing, to the energy and
  to its gradient; its length is never 0.
* **Capacity tiers** (``capacity_step``): MACE's (32 undirected pairs with at
  least 32 of padding, a thermal factor of 1.02): every padded edge runs the
  edge MLPs and the tensor product.
* **Energy tap**: ``node_energy`` returns the model's per-atom energies
  (``return_atomic=True``: float64 for every model dtype; the per-element E0s
  and the ZBL term included), and the engine differentiates their sum
  (``fastmd.engine.energy_derivative``).
* **Inputs**: upstream's edge is ``[receiver, sender]`` with the vector
  ``r_sender + image @ cell - r_receiver``: the engine's edge (centre,
  neighbour) as it is, so ``edge_index`` is the engine's transposed (int64) and
  the vector is the engine's edge vector, summed in the order upstream's export
  sums it (the shift ``image @ cell`` row by row, where upstream uses a batched
  matrix product: no TF32 GEMM for FP32).
* **Heads**: one head per adapter (``head=``; for the OAM checkpoint
  ``"salexmp"``, the head Matbench Discovery evaluates, by default).
* **Kernels**: upstream builds the spherical harmonics, tensor products and
  symmetric contractions with cuEquivariance's CUDA-only ``uniform_1d``
  method, so the module runs on CUDA only; every optimization profile runs it,
  there is no other kernel to select.
* **Precision**: the module runs in ``model_dtype`` (float32, upstream's
  export dtype, or float64); the E0s stay float64 as upstream keeps them.
* **Isolated atoms**: ``"model"``: the model evaluates an atom without
  neighbours itself.
* **Several GPUs** (``shard_spec``, ``sharded_energy``): the runtime's
  node-halo exchange through upstream's own multi-GPU hook (the
  ``feature_exchange`` argument of ``compute_energy_from_inputs``, which its
  LAMMPS ML-IAP export passes its exchange through): before every layer but
  the first the ghost rows of the layer's input take their owners' rows (one
  COPY of 2,048 values per ghost atom for the released checkpoints), and the
  per-atom work after each layer's message sum (the gate, the linear, the
  symmetric contraction, the residual and the readouts) runs on the owned
  slots only (the runtime's ghost-row trimming, which the shard declares).
  The edge capacity takes the model's tiers at P>1 too. IEEE FP32 by default,
  TF32 on request.

Checkpoints by name (``mace_soyo_oam``, ``mace_soyo_mptrj``) are downloaded
from the upstream release into ``$XDG_CACHE_HOME/mace_soyo`` (default
``~/.cache/mace_soyo``) and checked against a pinned sha256 (a download that
does not match is deleted); ``"mace_soyo"`` with ``model_path`` loads that
file. Loading unpickles the file (``torch.load(weights_only=False)``, as
upstream does): load only checkpoints you trust.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..adapters.protocol import (
    CapacityStep, CaptureContract, ExchangePoint, GraphBatch, PaddingRule, ShardSpec, TopologySpec,
)

_RELEASE = "https://github.com/PrincessHakuryu/MACE-SOYO/releases/download/v0.1"
# Pretrained checkpoints by name: (file of the v0.1 release, sha256 of the file, default head).
MACE_SOYO_CHECKPOINTS = {
    "mace_soyo_oam": ("mace_soyo_oam.pth", "180784bad2fdee6aec58dae2cdb0d581ad1c4f3d2272d2857bce6315ea43b82b",
                      "salexmp"),
    "mace_soyo_mptrj": ("mace_soyo_mptraj.pth", "d726daf370a92d280b75c72ca8a1551007946e00eb75a2fed6200547cd0a7de3",
                        "mptraj"),
}
_FAR_SELF_LOOPS = PaddingRule("far_self_loops")
# MACE's tiers: every padded edge costs as much as a real one (see capacity_step).
_CAPACITY_STEP = CapacityStep(u_step=32, min_pad_u=32, u_thermal_factor=1.02)
_ELEMENTS = 118  # columns of the reference-energy row (Z - 1)
# The upstream code the P>1 forward relies on (``MACESoyoAdapter.shard_spec``, ``sharded_energy``): the energy function
# with its multi-GPU hook (the exchange before every layer but the first, the readouts' gather of every row's head),
# the layers, which sum their messages and densities at the receiver and read the senders only through per-row maps
# of their input, the edges' tensor product (operand indexed by the sender, summed at the receiver), the per-row
# blocks that run on the owned slots only (the gate, the symmetric contraction, the readouts and their head masks),
# the per-row norm of the senders' rows, and the pair repulsion (summed at the receiver) and its distance transform.
# Pinned to mace-soyo 0.1.0 (sha256 of inspect.getsource); another version or source refuses P>1 with the reason.
_SHARD_PINS = {
    "mace_soyo.model.model.MACESoyo.compute_energy_from_inputs":
        "58467ebc8aea847457687f29038dabf9adb97753c4e3a3b509c06cb21b7f5cbf",
    "mace_soyo.model.model_block.EquivariantInteractionBlock.forward":
        "4fa9c252b62e308a69c832c8509e799110694e3ccd2859abe165f3acb7ec9acb",
    "mace_soyo.model.model_block.CueqConvFusionWrapper.forward":
        "66bab4c203bafb55fbf52f94507f6055448f931c148b0db7f6886b0505769e3b",
    "mace_soyo.model.model_block.CuePostGateIRMul.forward":
        "a78f2b47e245f3543a5c58d73f345d0435aa141095adefa869e210ca854cfdbc",
    "mace_soyo.model.model_block.SimpleSymmetricContraction.forward":
        "dfd10409f86ca781bd088ca62fb231f2bfd414e1259db80b89f3118984fe58ec",
    "mace_soyo.model.model_block.EquivariantRMSNorm.forward":
        "92e6813caf50fe4a59c4ed7f86e9ac52ecececc05d1b50cd3494762054f30a7d",
    "mace_soyo.model.model_block.ZBLBasis.forward": "0f1e1a323d660c5bab350f5cfa2414d657ebd4cb515341733a9be06538f4ba95",
    "mace_soyo.model.model_block.AgnesiTransform.forward":
        "1000e3ebc1342a950efb8778f7f8257cfe23bcb72689c1d6ff4bd2c8ec4c2a90",
    "mace_soyo.model.readout.NonLinearReadout.forward":
        "aa19b87e4b5b41730a2f1765c48f72797ee1a7d5c1a56919a2f14cbbff3f3c49",
    "mace_soyo.model.readout.ScalarLinear.forward": "dfaa2881922d9e6fa25945b72ffaca93928f2f3a2200e13c6d0c4051e1fd7ada",
    "mace_soyo.model.readout.mask_head": "bebfb85ed0a7846d18946285215082a723e8a0faacdb5eeceaf944d2823e45d0",
}
_SHARD_PIN_VERSIONS = ("0.1.0",)
# The packages of the library modules inside the layers and readouts (per-row maps and per-edge products of torch and
# cuEquivariance, not pinned: every other class there is upstream's and must be one of the pinned ones).
_LIBRARY_PACKAGES = ("torch", "cuequivariance", "cuequivariance_torch", "cuequivariance_ops_torch")


def is_mace_soyo_model(model, model_path: str | None = None) -> bool:
    """Whether ``load_adapter`` loads ``model`` as MACE-SOYO: a name above, or ``"mace_soyo"`` (with
    ``model_path``; without one, loading says what to pass)."""
    if not isinstance(model, (str, os.PathLike)):
        return False
    name = os.fspath(model)
    return name in MACE_SOYO_CHECKPOINTS or name == "mace_soyo"


def check_load_options(model_dtype: str | None, optimization_profile: str | None, model_path: str | None) -> None:
    """The loading options ``load_adapter`` rejects for MACE-SOYO (float32 and float64 load, every profile but
    MatRIS's runs the model's own cuEquivariance kernels)."""
    from ..adapters import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only; MACE-SOYO is written in "
                               "cuEquivariance and runs its own kernels under 'topology' and 'generic'")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_dir() -> str:
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "mace_soyo")


def _checkpoint(model, model_path: str | None) -> dict:
    """The file to load: a pinned pretrained checkpoint (downloaded once into the cache, sha256-checked), or the
    given path."""
    name = os.fspath(model)
    if name in MACE_SOYO_CHECKPOINTS and model_path is None:
        filename, expected, head = MACE_SOYO_CHECKPOINTS[name]
        path = os.path.join(_cache_dir(), filename)
        if not os.path.isfile(path):
            os.makedirs(_cache_dir(), exist_ok=True)
            partial = f"{path}.{os.getpid()}.part"
            try:
                torch.hub.download_url_to_file(f"{_RELEASE}/{filename}", partial, progress=False)
                found = _sha256(partial)
                if found != expected:
                    raise ValueError(f"MACE-SOYO checkpoint {name!r} downloaded from {_RELEASE}: sha256 {found} is "
                                     f"not the pinned {expected}")
                os.replace(partial, path)  # only a verified file enters the cache
            finally:
                if os.path.exists(partial):
                    os.remove(partial)
        found = _sha256(path)
        if found != expected:
            raise ValueError(f"MACE-SOYO checkpoint {name!r} at {path}: sha256 {found} is not the pinned {expected}")
        return {"name": name, "path": path, "sha256": found, "default_head": head}
    path = model_path if model_path is not None else name
    if path == "mace_soyo" or not os.path.isfile(path):
        raise FileNotFoundError(f"MACE-SOYO checkpoint not found: {path!r} (pass a .pth path or one of "
                                f"{sorted(MACE_SOYO_CHECKPOINTS)})")
    found = _sha256(path)
    # a copy of a pinned checkpoint keeps its default head
    heads = [head for _, sha256, head in MACE_SOYO_CHECKPOINTS.values() if sha256 == found]
    return {"name": os.path.basename(path), "path": str(path), "sha256": found,
            **({"default_head": heads[0]} if heads else {})}


def load_mace_soyo(model, *, model_path: str | None = None, device: str | None = None,
                   model_dtype: str | None = None, optimization_profile: str | None = None,
                   head: str | None = None) -> "MACESoyoAdapter":
    """A :class:`MACESoyoAdapter` of a pretrained name or checkpoint path (see :func:`is_mace_soyo_model`).

    The module is built by upstream's own loader on ``device`` (CUDA only:
    see "Kernels" above) in ``model_dtype`` (default float32, the dtype of
    upstream's export), parameters frozen, ``eval()``.
    """
    if optimization_profile == "legacy":
        raise ValueError("optimization_profile='legacy' is MatRIS's (MACE-SOYO: None, 'topology' or 'generic')")
    dtypes = {None: torch.float32, "float32": torch.float32, "float64": torch.float64}
    if model_dtype not in dtypes:
        raise ValueError(f"MACE-SOYO model_dtype must be None, 'float32' or 'float64', got {model_dtype!r}")
    dtype = dtypes[model_dtype]
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError(f"MACE-SOYO runs on CUDA only (upstream's cuEquivariance layers are CUDA kernels), "
                         f"got {device}")
    checkpoint = _checkpoint(model, model_path)
    from mace_soyo.utils.model_checkpoint import load_model_from_checkpoint

    default_dtype = torch.get_default_dtype()
    try:
        with torch.cuda.device(device):
            module = load_model_from_checkpoint(checkpoint["path"], device=device, dtype=dtype)
    finally:
        torch.set_default_dtype(default_dtype)
    return MACESoyoAdapter(module, head=head if head is not None else checkpoint.get("default_head"),
                           checkpoint=checkpoint)


def _exchange_points(module) -> tuple[ExchangePoint, ...]:
    """The P>1 exchanges of ``module``: before every layer but the first, the ghost rows of the layer's input
    (upstream's ``feature_exchange`` hook, called by the adapter's P>1 forward, in this order)."""
    return tuple(ExchangePoint(name=f"layers.{index}.x", module=None, width=int(layer.irreps_in.dim))
                 for index, layer in enumerate(module.layers) if index > 0)


def _shard_targets(module) -> tuple[str, ...]:
    """The upstream code the P>1 forward of ``module`` relies on, named from the module's own classes: its energy
    function, the forward of every class in its layers and readouts that is not a torch or cuEquivariance module, the
    readouts' head masks and, with the pair repulsion, its forward and its distance transform's. A class without a pin
    (or a model of another class) gives a pin that cannot match."""
    cls = type(module)
    targets = [f"{cls.__module__}.{cls.__qualname__}.compute_energy_from_inputs"]
    parts = [*module.layers, *module.energy_readouts]
    if getattr(module, "use_zbl", False):
        parts += [module.zbl, module.distance_transform]
    for part in parts:
        for submodule in part.modules():
            cls = type(submodule)
            if cls.__module__.split(".")[0] not in _LIBRARY_PACKAGES:
                targets.append(f"{cls.__module__}.{cls.__qualname__}.forward")
    targets.append("mace_soyo.model.readout.mask_head")
    return tuple(dict.fromkeys(targets))


def _trimmed_modules(module) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The ghost-row trimming of ``module`` at P>1 (``ShardSpec.owned_slot_inputs``, ``zero_ghost_outputs``): each
    layer's gate and residual linear and every readout run on the owned slots only, and a readout's output gets zero
    ghost rows appended, since upstream gathers each row's head from it over every row (the ghost rows' energies are
    never summed). The ghost rows receive no messages, and what a layer computes on them after its message sum is never
    read (the next layer's input takes their owners' rows, the energy the owned rows'), so the gate, the linear and the
    symmetric contraction after it, the residual and the readouts run on the owned slots."""
    layers = tuple(f"layers.{index}.{name}" for index in range(len(module.layers)) for name in ("gate", "sc_lin_x"))
    readouts = tuple(f"energy_readouts.{index}" for index in range(len(module.energy_readouts)))
    return layers + readouts, readouts


class _FeatureExchange:
    """Upstream's ``feature_exchange`` hook over the runtime's COPY of one P>1 forward (``ctx``, a ``ShardContext``):
    the n-th call is the n-th exchange point. ``x`` holds the owned slots' rows (the layer before ran its per-atom work
    on them only) or every row; the result holds every row, the ghost rows their owners' (``ctx.exchange``: the
    differentiable COPY, whose backward sums the ghost rows' gradients into their owners')."""

    def __init__(self, ctx, points: tuple[ExchangePoint, ...]):
        self.ctx, self.points, self.calls = ctx, points, 0

    def __call__(self, x: Tensor) -> Tensor:
        if self.calls >= len(self.points):
            raise RuntimeError(f"MACE-SOYO's multi-GPU forward exchanged more often than its {len(self.points)} "
                               "exchange points")
        point = self.points[self.calls]
        self.calls += 1
        rows = self.ctx.owned_rows + self.ctx.ghost_rows
        if x.shape[0] < rows:
            x = torch.cat((x, x.new_zeros((rows - x.shape[0], *x.shape[1:]))))
        return self.ctx.exchange(point, x)


@dataclass
class MACESoyoInput:
    """MACE-SOYO's input over a graph batch: the batch itself and its shape constants (every row is one system,
    graph 0, read with one head). The value-dependent tensors are built in each call."""

    batch: GraphBatch
    graph_index: Tensor  # [rows] int64 zeros
    head: Tensor  # [1] int64: the head of the system


class MACESoyoAdapter:
    """An upstream ``MACESoyo`` module as a :class:`~fastmd.adapters.protocol.ModelAdapter` with an
    :class:`~fastmd.adapters.protocol.EnergyTap` (see the module docstring).

    ``module`` is the module itself; ``head`` the head it reads; ``checkpoint``
    the loaded file (name, path, sha256).
    """

    name = "mace_soyo"
    compiled = False  # the adapter runs the upstream module eagerly (its AOTInductor export is not used)

    def __init__(self, module, *, head: str | None = None, checkpoint: dict | None = None) -> None:
        if getattr(module, "use_spin_charge", False):
            raise ValueError("MACE-SOYO with charge/spin conditioning (use_spin_charge=True) is not supported: "
                             "the adapter has no per-system charge and spin inputs")
        self.module = module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.checkpoint = {key: value for key, value in (checkpoint or {}).items() if key != "default_head"}
        heads = [str(name) for name in module.head_names]
        if head is None:
            if len(heads) > 1:
                raise ValueError(f"MACE-SOYO model has heads {heads}: pass head=")
            head = heads[0]
        if head not in heads:
            raise ValueError(f"MACE-SOYO head {head!r} is not one of {heads}")
        self.head = head
        self._head_index = heads.index(head)
        # Shape constants per (rows, device): built once and never freed, since captured graphs read them.
        self._constants: dict[tuple, tuple[Tensor, Tensor]] = {}

    # --- ModelAdapter -----------------------------------------------------

    def topology_spec(self) -> TopologySpec:
        """Directed edges within the cutoff, ``num_layers`` layers, no triplets."""
        return TopologySpec(
            cutoff=float(self.module.cutoff),
            num_interactions=int(self.module.num_layers),
            triplet_cutoff=None,
            undirected_pairs=False,
            model_dtype=str(self.module.z_emb.weight.dtype).removeprefix("torch."),
            edge_order="by_centre",  # the layers sum their messages per receiver; the order within one is free
        )

    def padding_rule(self) -> PaddingRule:
        """Far self-loops: exactly nothing for the model's exact cutoff, never a zero length."""
        return _FAR_SELF_LOOPS

    def prepare(self, batch: GraphBatch) -> MACESoyoInput:
        """The batch and its shape constants; rejects elements outside the model's embedding (host check,
        outside any graph)."""
        rows = batch.species.shape[0]
        if batch.positions is None or batch.positions.shape[0] != rows:
            raise ValueError("MACE-SOYO reads Cartesian positions: an edges-only graph batch without sink rows")
        device = batch.species.device
        key = (rows, str(device))
        if key not in self._constants:
            self._constants[key] = (torch.zeros(rows, dtype=torch.int64, device=device),
                                    torch.tensor([self._head_index], dtype=torch.int64, device=device))
        graph_index, head = self._constants[key]
        elements = int(self.module.num_elements)
        species = batch.species
        if bool(((species < 1) | (species > elements)).any()):
            raise ValueError(f"MACE-SOYO model {self.checkpoint.get('name', '')!r} supports atomic numbers 1 to "
                             f"{elements}")
        return MACESoyoInput(batch=batch, graph_index=graph_index, head=head)

    def isolated_atom_policy(self) -> str:
        """"model": the model evaluates an atom without neighbours itself."""
        return "model"

    def reference_energies(self) -> Tensor:
        """The E0 of each element for the adapter's head as a ``[1, 118]`` float64 row (column ``Z - 1``; NaN
        for elements outside the model), read-only."""
        energies = self.module.E0.detach()[:, self._head_index]
        row = torch.full((1, _ELEMENTS), float("nan"), dtype=torch.float64, device=energies.device)
        count = min(int(energies.shape[0]), _ELEMENTS)
        row[0, :count] = energies[:count].to(torch.float64)
        return row

    # --- EnergyTap ----------------------------------------------------------

    def node_energy(self, inputs: MACESoyoInput, positions: Tensor, cell: Tensor,
                    edge_shifts: Tensor | None = None) -> Tensor:
        """The model's per-atom energies ([rows] float64: E0, ZBL and the readouts of every layer) at
        ``positions`` and ``cell`` (or each edge's shift ``edge_shifts`` in its own system's cell)."""
        return self._atomic_energies(inputs, positions, cell, edge_shifts).squeeze(-1)

    def _atomic_energies(self, inputs: MACESoyoInput, positions: Tensor, cell: Tensor,
                         edge_shifts: Tensor | None = None, feature_exchange=None) -> Tensor:
        """Upstream's per-atom energies (``[rows, 1]`` float64) over ``inputs`` at ``positions`` and ``cell`` (or the
        edges' ``edge_shifts``), with upstream's multi-GPU hook ``feature_exchange`` if given."""
        batch = inputs.batch
        edges = batch.edge_index.to(torch.int64)
        centre, neighbour = edges[:, 0], edges[:, 1]
        if edge_shifts is None:
            image = batch.edge_image.to(positions.dtype)
            # row by row: no TF32 GEMM for FP32 (and exact for FP64)
            edge_shifts = image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
        # upstream's [receiver, sender] = the engine's (centre, neighbour); its vector r_j + shift - r_i
        vectors = positions[neighbour] + edge_shifts - positions[centre]
        # float64 geometry for a float32 module (Float64GeometryInput): the float64 vector cast at the module's input
        vectors = vectors.to(self.module.z_emb.weight.dtype)
        # the hook only when given: one GPU calls upstream exactly as before
        hook = {} if feature_exchange is None else {"feature_exchange": feature_exchange}
        return self.module.compute_energy_from_inputs(
            batch.species.to(torch.int64), vectors, vectors.norm(dim=-1), edges.t().contiguous(),
            inputs.graph_index, 1, inputs.head, return_atomic=True, **hook)

    # --- capability negotiation ---------------------------------------------

    @property
    def upstream(self) -> str:
        """The package providing the model and its kernels' package."""
        from importlib.metadata import version

        return f"mace-soyo {version('mace-soyo')}, cuequivariance {version('cuequivariance')}"

    def capture_contract(self) -> CaptureContract:
        """No host synchronization in a call (captured whole), an exact radial cutoff, held to upstream's own
        call within the eager noise floor (atomic scatters: not bitwise), the engine-owned derivative with the
        stress in eV/A^3, no torch.compile."""
        return CaptureContract(sync_free=True, exact_cutoff=True, accuracy_class="fp32_noise",
                               force_mode="engine_autograd", stress_unit="eV/A^3", compiled="none")

    def capacity_step(self) -> CapacityStep:
        """MACE's capacity tiers (``_CAPACITY_STEP``): every padded edge runs the edge MLPs and the tensor
        product, so a small cell takes a fine pair step."""
        return _CAPACITY_STEP

    def system_inputs(self) -> dict[str, Any]:
        """The static per-system input: the head."""
        return {"head": self.head}

    def edge_shift_input(self) -> bool:
        """True: :meth:`node_energy` takes the engine's per-edge shifts (the model reads only edge vectors; the
        stress comes from the engine's strain)."""
        return True

    def float64_geometry_input(self) -> bool:
        """True: :meth:`node_energy` takes float64 positions, cell and shifts for a float32 module (each edge's
        vector in float64, cast at the module's input; :class:`~fastmd.adapters.protocol.Float64GeometryInput`)."""
        return True

    def system_batching(self) -> str | None:
        """"union" for upstream's ``MACESoyo`` itself: its per-atom energies come from messages over edges and a
        per-atom readout (the head is one per adapter), so replicas or systems evaluated as one graph give each
        atom its own system's energy. None for any other module (a subclass may read the graph index)."""
        module = self.module
        if (type(module).__name__, type(module).__module__) != ("MACESoyo", "mace_soyo.model.model"):
            return None
        return "union"

    # --- multi-GPU (the distributed runtime's node-halo exchange) -----------

    def shard_spec(self) -> ShardSpec:
        """MACE-SOYO at P>1: each layer sums its messages and its edge density at the receiver (the edge's centre), as
        does the pair repulsion, which upstream computes once before the layers, and reads its senders only through
        per-row maps of its input (the norm, the linear to the tensor product's operand, the scalar projections of the
        edge weights), and every other per-atom computation reads its own atom's row. So the P>1 forward
        (:meth:`sharded_energy`, upstream's own multi-GPU hook) runs the per-atom work after each layer's message sum
        on the owned slots only (the runtime's ghost-row trimming, declared in ``owned_slot_inputs`` and
        ``zero_ghost_outputs``: :func:`_trimmed_modules`), and before every layer but the first (whose input is the
        species embedding, on every row) the ghost rows of the layer's input take their owners' rows: one COPY of the
        input's width per ghost atom (2,048 values for the released checkpoints, the place upstream's LAMMPS ML-IAP
        export exchanges at). Unused atom slots get hydrogen. The P>1 topology's edge
        capacity (and one CUDA rank's) takes the model's tiers (``_CAPACITY_STEP``), not the topology's 1/8 headroom:
        every padded edge runs the edge MLPs and the tensor product of both layers (6-9 % faster and about 9 % less
        memory at P=1, 2 and 4 on 10,752 RDX atoms). The head is one per adapter (``system_inputs``, part of the plan
        fingerprint), so the ranks agree on it through the rank contract. Pins: upstream's energy function and the
        forwards it relies on (``_shard_targets``); a class without a pin (or a model of another class) gives a pin
        that cannot match, so P>1 is refused with the reason. The far self-loops add nothing at P>1 as on one GPU (the
        declared exact cutoff). TF32 on request (``matmul_precision="high"``) runs at P>1: on the 1,344-atom RDX cell
        MACE-SOYO OAM's TF32 P=2 and P=4 forces, energy and stress, NVE drift and Nose-Hoover last frame stay within
        the TF32 bounds set by TF32 P=1's own run-to-run noise (five P=1 processes)."""
        from ..adapters.patches import PatchRecord

        pins = tuple(PatchRecord(name="mace_soyo.shard." + ".".join(target.rsplit(".", 2)[-2:]), target=target,
                                 distribution="mace-soyo", versions=_SHARD_PIN_VERSIONS,
                                 source_sha256=_SHARD_PINS.get(target, "0" * 64),
                                 reason="the multi-GPU forward relies on this code unchanged")
                     for target in _shard_targets(self.module))
        owned_slots, zero_ghosts = _trimmed_modules(self.module)
        return ShardSpec(exchange_points=_exchange_points(self.module), pad_species=1, source_pins=pins,
                         allow_tf32=True, capacity_step=_CAPACITY_STEP, owned_slot_inputs=owned_slots,
                         zero_ghost_outputs=zero_ghosts)

    def sharded_energy(self, inputs: MACESoyoInput, positions: Tensor, cell: Tensor, ctx) -> Tensor:
        """This rank's share of MACE-SOYO's energy at P>1 (``fastmd.adapters.protocol.ShardedTap``): the FP64 sum of
        the owned real slots' energies of upstream's own multi-GPU forward (``compute_energy_from_inputs`` with its
        ``feature_exchange`` hook) over the rank's rows (owned slots, then ghost rows).

        The edge vectors are :meth:`node_energy`'s. The embedding and each layer's norm, its linear to the tensor
        product's operand and the scalar projections run on every row (the senders' terms read the ghost rows);
        the messages, the density and the pair repulsion are summed at the receivers (all owned). The gate, the
        linear and the symmetric contraction after it, the residual linear and the readouts run on the owned slots
        only, and a readout's output is padded with zero ghost rows (the runtime's ghost-row trimming, which
        registers its hooks around this call: :func:`_trimmed_modules`), so a layer's output holds the owned slots,
        and before every layer but the first :class:`_FeatureExchange` fills the ghost rows of its input with their
        owners' rows (``ctx.exchange``). Upstream's zero anchors (the edge vectors' sum times zero added to the
        embedding, each exchanged input's sum times zero added to every atom's energy) keep every rank's backward
        running through every exchange, a rank without atoms or edges included; so every row of an exchanged input
        must be finite (an unused slot is a hydrogen atom without edges, a padded ghost slot a copy of an owned
        row)."""
        owned = ctx.owned_rows
        exchange = _FeatureExchange(ctx, _exchange_points(self.module))
        atomic = self._atomic_energies(inputs, positions, cell, feature_exchange=exchange)
        energies = atomic.squeeze(-1)[:owned]
        # A select, not a product: an unused slot's energy may be anything.
        return torch.where(ctx.owned[:owned], energies, energies.new_zeros(())).sum()
