"""TACE behind the adapter protocol: the upstream TACE model (spherical, e3nn) with an energy tap.

TACE (``tace``, installed from upstream's git at the validated commit, which is
not a PyPI release) is optional: this module imports it only when a model is
loaded, so ``fastmd`` imports without it. The adapter wraps upstream's
``TensorModel`` as its loader returns it; of TACE's code it restates one method
only (the patch below). What the adapter declares and does:

* **Checkpoints**: the TACE foundation checkpoints of the spherical family
  (``e3nnTACE``: TACE-OAM-L, TACE-OAM-RRA-Preview, TECE-OAM-RRA-1.0), by name
  from upstream's Hugging Face repository at a pinned revision, sha256-checked,
  or a ``.pt``/``.pth`` path of the same format. TACE-v1-OAM-M is not loaded
  here: its file is a state dict of tace 0.1.0's Cartesian model (``TACEV1``)
  with statistics pickled as tace 0.1.0 objects, which tace 0.2.0 cannot
  unpickle.
* **Topology**: directed edges within the model's cutoff (6 Å), no triplets, so
  the builders run edges only (``TopologySpec.triplet_cutoff`` None).
* **Padding**: far self-loops (``PaddingRule("far_self_loops")``), which the
  adapter moves onto a sink row of its own. TACE does not see a pair beyond the
  cutoff only through its envelope: TACE-OAM-L normalizes each atom's messages
  by a neighbour density summed over all of the atom's edges without the
  envelope, and the attention layers of the RRA models take a softmax over all
  of an atom's edges. A padded row of a real atom would change both. Each call
  appends one sink row (the first element of the model's table, at the
  origin) in a graph of its own and points every row at or past ``n_edges`` at
  it (sender and receiver; the row keeps its far image, so its length is at
  least twice the cutoff); the sink's energy is dropped. A padded batch without
  ``n_edges`` is refused.
* **Capacity tiers** (``capacity_step``): 32 undirected pairs with at least 32
  of padding, every capture holding 1.02 times the pair demand, as MACE's (a
  padded row runs every layer's tensor products, on the sink row).
* **Energy tap**: ``node_energy`` returns TACE's own ``node_energy`` (the
  per-element reference energy plus the scaled readout and the ZBL pair term
  per atom), and the engine differentiates its sum
  (``fastmd.engine.energy_derivative``): TACE's own force and stress autograd
  (``TensorModel.forward``) is not called; the adapter builds upstream's
  ``Graph`` from the engine's edges and calls the model's readout.
* **Inputs**: TACE gathers the sender's features at ``edge_index[0]`` and sums
  at the receiver ``edge_index[1]``, with the edge vector ``x[receiver] -
  x[sender] + edge_shifts @ lattice``. The adapter passes ``(neighbour,
  centre)`` with the shift ``-edge_image @ cell``: the engine's edge reversed,
  from the neighbour's image to the centre, summed at the centre.
* **Patch**: upstream's ``ScaleShift.forward`` counts each graph's edges with
  ``torch.bincount``, which reads the size of its result back to the host; the
  adapter installs the same forward with a counting scatter of fixed size
  (``tace.scale_shift_edge_count_on_device``). The patch and the pins of the
  code the tap calls (the readout, ``TensorModel.prepare_graph``, ``Graph``)
  are version-gated: another tace version refuses to load. Another source of
  ``ScaleShift.forward`` alone declines the patch: the model then runs
  uncaptured. The gate cannot tell the validated commit from PyPI's TACE 0.2.0,
  whose model code differs; the tests and the CI lane run the commit.
* **Kernels**: OpenEquivariance's fused gather-tensor-product-scatter of the
  interactions' Clebsch-Gordan tensor products (upstream's ``TACE_USE_OEQ``
  switch, the kernels of its accelerated calculator; the RRA models' attention
  layers keep their own) by default, e3nn (the reference) for
  ``optimization_profile="generic"`` or when OpenEquivariance is unavailable.
  Upstream's cuEquivariance switch does not build at the validated version (its
  scatter tensor product rejects the arguments the interaction passes).
* **Precision**: the module runs in its checkpoint's dtype (float32).
* **Isolated atoms**: ``"model"``: TACE evaluates an atom without neighbours
  itself (a one-atom system without edges keeps its reference energy, as
  upstream's scale and shift give it).
* **Several systems** (``system_batching`` "union", ``edge_shift_input``):
  each atom's energy depends on its own system's atoms only (messages over
  edges, the neighbour density and the attention softmax per receiving atom,
  a per-atom readout), except for one decision per graph: upstream's
  ``ScaleShift`` leaves a graph of one atom without edges unscaled and
  unshifted (its reference energy). So a batch of several systems
  (``GraphBatch.systems``) is handed to the module as one graph per system,
  plus the sink's: each occupied row in its system's graph, a slot's vacant
  rows with the sink, and ``ptr`` the prefix sums of each graph's atom count
  (upstream reads ``ptr`` for the counts only), built in each call from the
  batch's system buffers, which a slot rewrites in place. A one-atom system
  without edges then keeps its reference energy in a batch as alone, and a
  system's edge vectors come from the engine's per-edge shifts (each system
  in its own cell).
* **Several GPUs** (``shard_spec``, ``sharded_energy``): the node-halo
  exchange with the receiver edge rule. An owned atom's sums over its edges
  (messages, density, attention, ZBL) are complete on its rank; before every
  interaction but the first the ghost rows of the interaction's ``linear_up``
  output take their owners' rows (the tensor upstream's own LAMMPS path
  exchanges), through forward hooks the adapter installs for each P>1
  forward. Each rank keeps its own sink row for its padded edge rows, and
  ``ScaleShift``'s one-atom decision is made for the whole system (its atom
  count, ``ptr`` of the rank's graph), not for a rank's rows.

Loading unpickles the file (``torch.load(weights_only=False)``, as TACE does):
load only checkpoints you trust.
"""
from __future__ import annotations

import contextvars
import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..adapters.patches import PatchRecord, apply_patches, revert_patches
from ..adapters.protocol import (CapacityStep, CaptureContract, ExchangePoint, GraphBatch, KernelChoice,
                                 NegotiationError, PaddingRule, ShardSpec, TopologySpec)

# Pretrained checkpoints by name: (file in upstream's Hugging Face repository, sha256 of the file).
TACE_REPOSITORY = "xvzemin/tace-foundations"
TACE_REVISION = "aa4a36a9f6a4d68d3090c0b568a7d42b8b4ef7d9"
TACE_CHECKPOINTS = {
    "tace_oam_l": ("TACE-OAM-L.pt", "448ff14c92d0fe29b8f038e98ffad02bccfc77012d85cc29cc2e7a7a8b1e4a5a"),
    "tece_oam_rra_1_0": ("TECE-OAM-RRA-1.0.pt", "9f36562582d931347c3904f763e820edcaf5f27c3f13beb6776e49fcc7de38bb"),
    "tace_oam_rra_preview": ("TACE-OAM-RRA-Preview.pt",
                             "e2d9ede587dc7a98e860d4df6ac2157ca6469c5f83cfa8a07e8a3e59f65e7061"),
}
# The kernel site and its options, fastest first; e3nn is the reference.
KERNEL_SITE = "equivariant_products"
KERNEL_OPTIONS = ("oeq", "e3nn")
# Upstream's acceleration switches: environment flags its modules read when they are built.
_SWITCHES = ("TACE_USE_OEQ", "TACE_USE_CUE", "TACE_USE_EQT", "TACE_USE_COMPILE")
# The upstream versions the patch and the readout's use of upstream's internals were validated on.
TACE_VERSIONS = ("0.2.0",)

_FAR_SELF_LOOPS = PaddingRule("far_self_loops")
# Capacity tiers (capacity_step): every padded edge runs the layers' tensor products (on the sink row), so a small
# cell takes a fine pair step, as MACE's.
_CAPACITY_STEP = CapacityStep(u_step=32, min_pad_u=32, u_thermal_factor=1.02)
_ELEMENTS = 118  # columns of the reference-energy row (Z - 1)
# Upstream's model class the readout is called on and the class the patch replaces a method of.
_MODEL_CLASS = ("e3nnTACE", "tace.models._e3nn.tace")


# Derived from TACE 0.2.0 (commit 81f65a4c), tace/models/blocks.py (ScaleShift.forward). Copyright (c) 2025 Zemin Xu.
# Licensed under the MIT License (https://github.com/xvzemin/tace/blob/main/LICENSE.md).
# Changed: the edge count per graph is a fixed-size device scatter in place of torch.bincount (no host
# synchronization); the edge-free branch reduces to the same zero counts.
def _scale_shift_forward(self, node_energy, node_attrs, ptr, edge_index, batch, node_fidelity):
    """``tace.models.blocks.ScaleShift.forward`` with each graph's edge count summed on the device into a
    result of ``num_graphs`` rows (upstream: ``torch.bincount(..., minlength=num_graphs)``, whose result size
    the host reads back). The same values: an edge row's graph index is below ``num_graphs``."""
    if not (self.has_scale or self.has_shift):
        return node_energy

    num_graphs = ptr.numel() - 1
    num_nodes = ptr[1:] - ptr[:-1]

    if self.all_atoms:
        if self.has_scale:
            node_scale = (node_attrs * self.scale[node_fidelity]).sum(dim=-1)
            node_energy = node_energy * node_scale

        if self.has_shift:
            node_shift = (node_attrs * self.shift[node_fidelity]).sum(dim=-1)
            node_energy = node_energy + node_shift
    else:
        edge_batch = batch[edge_index[1]]
        num_edges = torch.zeros(num_graphs, dtype=torch.int64, device=node_energy.device)
        num_edges = num_edges.index_add(0, edge_batch, torch.ones_like(edge_batch))

        isolated_mask = (num_nodes == 1) & (num_edges == 0)

        if self.has_scale:
            node_scale = (node_attrs * self.scale[node_fidelity]).sum(dim=-1)
            isolated_nodes = isolated_mask[batch]
            node_scale = torch.where(
                isolated_nodes, torch.zeros_like(node_scale), node_scale
            )
            node_energy = node_energy * node_scale

        if self.has_shift:
            node_shift = (node_attrs * self.shift[node_fidelity]).sum(dim=-1)
            isolated_nodes = isolated_mask[batch]
            node_shift = torch.where(
                isolated_nodes, torch.zeros_like(node_shift), node_shift
            )
            node_energy = node_energy + node_shift

    return node_energy


SCALE_SHIFT_EDGE_COUNT = PatchRecord(
    name="tace.scale_shift_edge_count_on_device",
    target="tace.models.blocks.ScaleShift.forward",
    distribution="TACE", versions=TACE_VERSIONS,
    source_sha256="629170f5a898e8f7e1147fbc3ef7864bace96d573817dfeba61ef42087a19aff",
    reason="host synchronization: torch.bincount reads the size of its result back from the device in every "
           "energy evaluation (the scale and shift of the readout)",
    replacement=_scale_shift_forward)

# Upstream code the energy tap runs or mirrors unpatched (pins: checked, nothing installed).
_PIN_SOURCES = {
    # the readout the tap calls (node_energy: reference energy + scaled readout + ZBL)
    "tace.readout": ("tace.models._e3nn.tace.e3nnTACE.readout_fn",
                     "the energy tap reads this readout's per-atom energies"),
    # the graph the tap builds in place of this method's (edge vector, length, fidelity, atom indices)
    "tace.prepare_graph": ("tace.models.adapter.TensorModel.prepare_graph",
                           "the energy tap builds the graph this method builds, without its displacement"),
    # the fields of the graph
    "tace.graph": ("tace.models.lammps.Graph", "the energy tap builds this graph"),
}
# sha256 of inspect.getsource of each target in tace 0.2.0 at upstream commit 81f65a4c (the leaderboard's pin).
_PIN_SHA256 = {
    "tace.models.blocks.ScaleShift.forward": "629170f5a898e8f7e1147fbc3ef7864bace96d573817dfeba61ef42087a19aff",
    "tace.models._e3nn.tace.e3nnTACE.readout_fn": "9d77311b4ef99cba5efecfa39d82c35eaee94ac204abe74d046d967f8631c25e",
    "tace.models.adapter.TensorModel.prepare_graph":
        "7e48048bffc528bb297424e45ccfc61bc308a7318cdfcdb143427354d3f99e32",
    "tace.models.lammps.Graph": "c8af5020c4b91c6b19efbd1bcfe4f5a111c5ec6a05c16e18fa7dcd09925cdfa8",
}


# The code the P>1 exchange relies on (sha256 of inspect.getsource at upstream commit 81f65a4c): the layer loop, the
# node and edge embeddings (the first layer's features from the species, on every row), the interactions (each reads
# other atoms' features only through its rejector, which gathers the senders' rows of its linear_up output and sums at
# the receivers), the rejectors and the kernels they call (e3nn, OpenEquivariance, SO(2) attention), the edge updates
# (the species of an edge's two atoms only), the products (their own atom's row), the ZBL pair term (summed at the
# receiver) and upstream's ghost helpers (identities without LAMMPS data). A class outside this table gets a pin that
# cannot match: P>1 is refused with the reason.
_SHARD_PINS = {
    "tace.models._e3nn.representation.Representation.forward":
        "c0a7bb0da12371ac15a103a952993a4c192a6e5c11ab48ddcc2fdb48706ab013",
    "tace.models._e3nn.node.LinearNodeEmbedding.forward":
        "100e65f0520137eaaca865ab8b4d438a13ecd526107afd22bacb9a0d9129f33a",
    "tace.models._e3nn.edge.NonLinearEdgeEmbedding.forward":
        "f9f34ff9438631672ba0c80e284f224c506b3f6d39acfd13812688b4c8ec587e",
    "tace.models._e3nn.inter.CgtpInteraction.forward":
        "e2b732da692fda39dce6fd4253ebb9f26258962c0823e74b90bfb98287383f18",
    "tace.models._e3nn.inter.uvSO2Interaction.forward":
        "2bbe94f4d7474a7f8c8ffd0459abf390008a73bccea37b66e147e777fdced326",
    "tace.models._e3nn.fused.O3ScatterTensorProduct.forward":
        "eb9fa31d92a17d7c81c91fb6e80538f38e13eddc7943c8706dbc163ba1e5c651",
    "tace.models._e3nn.fused.uvSO2TensorProduct.forward":
        "115d2cd33e80821da7bd3bd95908d93d75d4c3fb455d8a335350b68d43d5eccd",
    "tace.models._oeq._tp_scatter.e3nnOeqScatterTensorProduct.forward":
        "f6c594164d67b7e45d3b65b8dd92af943eb258ef8110cbceaf1e7a494f390623",
    "tace.models._e3nn.edge.Element2EdgeUpdate.forward":
        "e0066f1a4c932bab88331a686bc0d5ddeec1194d636eb202f879d5fd0f397463",
    "tace.models._e3nn.prod.CgtpACE.forward":
        "7a29c9bb22269e96cdb2e24ed5571cf968d8e826ce8672d2db7f83f68d286bed",
    "tace.models.radial.ZBLBasis.forward":
        "a381094612c5fc039d387320cedda2b018b1f84c371d3d851fcf4403e12cb30a",
    "tace.models.lammps.e3nnGhostExchangeMixin.handle_lammps":
        "b869d946b5351d01b7abd646f47cf95f596606389c2df08f499dd4fad7bab79f",
    "tace.models.lammps.e3nnGhostExchangeMixin.truncate_ghosts":
        "885fed401c62f936dac85223321b544933f7d457e65fae90ce4e5048c0a340ff",
    # The ghost-row trimming (TACEAdapter._trimmed_modules): the per-row modules after each message sum.
    "tace.models.linear.e3nnLinear.forward": "dd88d7905878cf26e29fd76c410c666bbf4ce0de1320a30ec9ac7c1d1bf87384",
    "tace.models.linear.e3nnElementLinear.forward": "ebec04f238cb1b36213f6ed4a70b13cfb1db0bf03fab4047763ae43d82fc6f49",
    "tace.models._e3nn.nonlinear.O3Gate.forward": "5f8d6606d4b58e76873ab11147322e852a702158e9122385945ffb495bb5e922",
    "tace.models._e3nn.residual.SkipIdentity.forward": "1e25f1e87adbc955127dca98f8f05fd952424dbea85c8c11e9df98814a111f34",
    "tace.models._e3nn.readout.ScalarReadOut.forward": "c9dd050c7de54199c0c766c62115a90b7b73db262a9673cec4fe539beaf05f5c",
    "tace.models._e3nn.dropout.GraphDropPath.forward": "405cd43ff74f53fe2d414eb9b9e1569341a0aa19d1f38313f78c83144c5564fa",
}
_GHOST_HELPERS = ("tace.models.lammps.e3nnGhostExchangeMixin.handle_lammps",
                  "tace.models.lammps.e3nnGhostExchangeMixin.truncate_ghosts")
# The checkpoints (sha256 of the file) whose TF32 forward at P>1 was measured within TF32 at P=1's own run-to-run
# noise (the multi-GPU tier's check): TACE-OAM-L. Any other checkpoint refuses a TF32 request at P>1.
_TF32_AT_P_GT_1 = frozenset({TACE_CHECKPOINTS["tace_oam_l"][1]})


def _records() -> list[PatchRecord]:
    """The patch and the pins."""
    records = [SCALE_SHIFT_EDGE_COUNT]
    records += [PatchRecord(name=name, target=target, distribution="TACE", versions=TACE_VERSIONS,
                            source_sha256=_PIN_SHA256[target], reason=reason)
                for name, (target, reason) in _PIN_SOURCES.items()]
    return records


def _check_switches() -> None:
    """Refuse upstream's acceleration switches other than OpenEquivariance's set in the environment: they change the
    module upstream builds (``TACEAseCalc(enable_compile=True)`` sets ``TACE_USE_COMPILE`` for the process)."""
    switches = [name for name in _SWITCHES if name != "TACE_USE_OEQ" and os.environ.get(name, "0") not in ("", "0")]
    if switches:
        raise RuntimeError(f"TACE acceleration switches set in the environment ({', '.join(switches)}): the "
                           "adapter builds upstream's e3nn module (OpenEquivariance through its kernel site); unset "
                           "them")


def _check_module(module) -> None:
    """Refuse a module the energy tap does not cover: another model class than upstream's spherical ``e3nnTACE``,
    a long-range (LES) or universal-embedding energy, or a neighbour cap."""
    readout = getattr(module, "readout_fn", None)
    if (type(readout).__name__, type(readout).__module__) != _MODEL_CLASS:
        raise ValueError(f"TACE adapter runs upstream's {_MODEL_CLASS[1]}.{_MODEL_CLASS[0]} (the spherical "
                         f"family), not {type(readout).__module__}.{type(readout).__name__}")
    if getattr(readout, "les", None) is not None or hasattr(readout, "uie_readout"):
        raise ValueError("TACE models with a long-range (LES) or universal-embedding energy are not supported "
                         "(their energy is not a sum of per-atom energies over the edges)")
    if getattr(readout, "max_neighbors", None) is not None:
        raise ValueError(f"TACE models with a neighbour cap ({readout.max_neighbors}) are not supported")


def is_tace_model(model, model_path: str | None = None) -> bool:
    """Whether ``load_adapter`` loads ``model`` as TACE: a name above, or ``"tace"`` with ``model_path``."""
    if not isinstance(model, (str, os.PathLike)):
        return False
    name = os.fspath(model)
    return name in TACE_CHECKPOINTS or (name == "tace" and model_path is not None)


def check_load_options(model_dtype: str | None, optimization_profile: str | None, model_path: str | None) -> None:
    """The loading options ``load_adapter`` rejects for TACE: float64 (the checkpoints are float32) and the
    MatRIS-only profile."""
    from ..adapters import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only; TACE runs e3nn, its reference")
    if model_dtype not in (None, "float32"):
        raise NegotiationError(f"TACE runs its checkpoints' float32, not model_dtype={model_dtype!r}")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(model, model_path: str | None) -> dict:
    """The file to load: a pinned pretrained checkpoint (downloaded from upstream's repository at its pinned
    revision, sha256-checked), or the given path."""
    name = os.fspath(model)
    if name in TACE_CHECKPOINTS and model_path is None:
        from huggingface_hub import hf_hub_download

        filename, expected = TACE_CHECKPOINTS[name]
        path = hf_hub_download(repo_id=TACE_REPOSITORY, filename=filename, revision=TACE_REVISION)
        found = _sha256(path)
        if found != expected:
            raise ValueError(f"TACE checkpoint {name!r} at {path}: sha256 {found} is not the pinned {expected}")
        return {"name": name, "path": str(path), "sha256": found}
    path = model_path if model_path is not None else name
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TACE checkpoint not found: {path!r} (pass a .pt path or one of "
                                f"{sorted(TACE_CHECKPOINTS)})")
    return {"name": os.path.basename(path), "path": str(path), "sha256": _sha256(path)}


def _build(path: str, device, option: str):
    """Upstream's module of the checkpoint at ``path`` (``tace.lightning.load_tace``, strict, as ``TACEAseCalc``
    loads it), built with OpenEquivariance's fused tensor products (``TACE_USE_OEQ``, upstream's switch, set while it
    builds) for ``option`` "oeq", with e3nn's for "e3nn", and checked (:func:`_check_module`). The process default
    dtype, which upstream's loader sets, and the switch are restored afterwards (the switch is process-wide: do not
    build TACE modules from several threads at once)."""
    from tace.lightning import load_tace as upstream_load

    _check_switches()

    default_dtype = torch.get_default_dtype()
    saved = os.environ.get("TACE_USE_OEQ")
    try:
        if option == "oeq":
            if torch.device(device).type != "cuda":
                raise RuntimeError("OpenEquivariance kernels run on CUDA only")
            import openequivariance  # noqa: F401  (the kernels; ImportError: not installed)

            os.environ["TACE_USE_OEQ"] = "1"
        else:
            os.environ.pop("TACE_USE_OEQ", None)
        module = upstream_load(path, device, strict=True, use_ema=True)
    finally:
        torch.set_default_dtype(default_dtype)
        if saved is None:
            os.environ.pop("TACE_USE_OEQ", None)
        else:
            os.environ["TACE_USE_OEQ"] = saved
    _check_module(module)
    return module


def load_tace(model, *, model_path: str | None = None, device: str | None = None, model_dtype: str | None = None,
              optimization_profile: str | None = None) -> "TACEAdapter":
    """A :class:`TACEAdapter` of a pretrained name or checkpoint path (see :func:`is_tace_model`).

    Mirrors ``tace.interface.ase.TACEAseCalc``'s load: the module on ``device`` in its checkpoint's dtype
    (float32), parameters frozen, ``eval()``, built with OpenEquivariance's fused tensor products (upstream's
    ``TACE_USE_OEQ`` switch, as its accelerated calculator) unless ``optimization_profile`` is "generic" (e3nn,
    the reference) or OpenEquivariance is unavailable (e3nn, with the reason in ``kernel_reason``). Upstream's
    other switches (``TACE_USE_CUE``, ``TACE_USE_EQT``, ``TACE_USE_COMPILE``) set in the environment are refused:
    they change the module upstream builds.
    """
    check_load_options(model_dtype, optimization_profile, model_path)
    _check_switches()
    device = torch.device("cuda" if device is None and torch.cuda.is_available() else (device or "cpu"))
    checkpoint = _checkpoint(model, model_path)
    reason = ""
    if optimization_profile == "generic":
        module, kernel, reason = _build(checkpoint["path"], device, "e3nn"), "e3nn", \
            "optimization_profile='generic': e3nn, the reference"
    else:
        try:
            module, kernel = _build(checkpoint["path"], device, "oeq"), "oeq"
            _warm_up(module)
        except Exception as error:  # noqa: BLE001 - any failure keeps e3nn, with the reason (a module the adapter
            # refuses is refused again by the e3nn build)
            detail = str(error).splitlines()[0][:200] if str(error) else ""
            module, kernel = _build(checkpoint["path"], device, "e3nn"), "e3nn"
            reason = f"OpenEquivariance unavailable ({type(error).__name__}: {detail}); e3nn"
    adapter = TACEAdapter(module, checkpoint=checkpoint, kernel=kernel)
    adapter.kernel_reason = reason
    return adapter


def _warm_up(module) -> None:
    """One energy and force evaluation of ``module`` on two atoms of its first element, 2 Å apart in a 20 Å box,
    through upstream's own forward (raises if the kernels cannot run; OpenEquivariance compiles them here)."""
    readout = module.readout_fn
    device, dtype = readout.cutoff.device, readout.cutoff.dtype
    elements = int(readout.atomic_numbers.shape[0])
    node_attrs = torch.zeros((2, elements), dtype=dtype, device=device)
    node_attrs[:, 0] = 1.0
    data = {
        "positions": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype, device=device),
        "node_attrs": node_attrs,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device),
        "edge_shifts": torch.zeros((2, 3), dtype=dtype, device=device),
        "lattice": 20.0 * torch.eye(3, dtype=dtype, device=device).unsqueeze(0),
        "batch": torch.zeros(2, dtype=torch.int64, device=device),
        "ptr": torch.tensor([0, 2], dtype=torch.int64, device=device),
    }
    with torch.enable_grad():
        out = module(data)
    if not bool(torch.isfinite(out["energy"]).all()):
        raise RuntimeError("non-finite energy on the warm-up input")
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# The P>1 forward in progress in this context (:meth:`TACEAdapter.sharded_energy`): the exchange context its hooks
# act for. Another thread's (or context's) call of the same module passes through them unchanged.
_SHARDED_FORWARD: contextvars.ContextVar = contextvars.ContextVar("tace_sharded_forward", default=None)


class _GhostCopy:
    """Forward hook of one interaction's ``linear_up`` at P>1 (:meth:`TACEAdapter.sharded_energy`): inside that
    call's context, the ghost rows of its output take their owners' rows (``ctx.exchange``: one COPY of the output's
    width per ghost row, the reverse SUM backward), and the adapter's sink row after the rank's rows passes through;
    an output of the owned slots only (the layer's input trimmed to them: the shard's ghost-row trimming) gets the
    ghost rows from the COPY and a zero sink row (the sink's features are read only by its own padded edges, whose
    messages it sums itself, and its energy is never summed). Any other call of the module passes through
    unchanged."""

    __slots__ = ("point", "ctx", "rows")

    def __init__(self, point, ctx, rows: int) -> None:
        self.point, self.ctx, self.rows = point, ctx, rows

    def __call__(self, module, args, output):
        if _SHARDED_FORWARD.get() is not self.ctx:
            return None
        if output.shape[0] == self.ctx.owned_rows and self.ctx.ghost_rows:
            return torch.cat((self.ctx.exchange(self.point, output), output.new_zeros((1, *output.shape[1:]))))
        return torch.cat((self.ctx.exchange(self.point, output[: self.rows]), output[self.rows:]))


@dataclass
class TACEInput:
    """TACE's input over a graph batch: the batch itself and its shape constants (one system: the real rows are
    graph 0, the adapter's sink row, the last, graph 1; several: graph ``s`` is system ``s``'s occupied rows and the
    last graph the sink with a slot's vacant rows). The value-dependent tensors are built in each call."""

    batch: GraphBatch
    graph_index: Tensor  # [rows + 1] int64, one system: 0 for the real rows, 1 for the sink (several: per call)
    ptr: Tensor  # [3] int64, one system: (0, rows, rows + 1) (several: per call)
    node_fidelity: Tensor  # [rows + 1] int64: the model's fidelity
    atom_index: Tensor  # [rows + 1] int64: arange
    sink: Tensor  # [] int64: the sink row (rows)
    sink_attrs: Tensor  # [1, elements]: the sink's one-hot row (the first element of the model's table)
    sink_position: Tensor  # [1, 3] zeros in the model's dtype
    # [119, elements] in the model's dtype: row Z is the one-hot row of Z (NaN outside the model), so a
    # call reads the node attributes with one lookup of the int32 species
    node_attrs_by_z: Tensor
    num_graphs: int = 2  # the module's graphs: the systems (one without GraphBatch.systems) and the sink's
    # Several systems: [1] int64 holding the sink's graph (the system count) and [1] int64 holding rows + 1 (the
    # last entry of ptr); None for one system
    sink_graph: Tensor | None = None
    ptr_end: Tensor | None = None


class TACEAdapter:
    """An upstream TACE module (``TensorModel`` around ``e3nnTACE``) as a
    :class:`~fastmd.adapters.protocol.ModelAdapter` with an :class:`~fastmd.adapters.protocol.EnergyTap` (see the
    module docstring).

    ``module`` is the module itself; ``checkpoint`` names the loaded file (name, path, sha256) and ``patches``
    the patch outcomes.
    """

    name = "tace"
    compiled = False  # upstream's modules run eager (OpenEquivariance's kernels are JIT-compiled CUDA)

    def __init__(self, module, *, checkpoint: dict | None = None, kernel: str = "e3nn") -> None:
        _check_module(module)
        self.module = module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.checkpoint = dict(checkpoint or {})
        from importlib.metadata import version

        self._version = version("TACE")
        self.patches = apply_patches(_records())
        pins = [outcome for outcome in self.patches if outcome.record.pin and not outcome.applied]
        if pins:
            self.close()
            raise RuntimeError("TACE code the energy tap relies on changed: "
                               + "; ".join(f"{o.record.name}: {o.reason}" for o in pins))
        self._fidelity = int(module.get_fidelity_idx())
        self.kernel = kernel
        self.kernel_reason = ""
        self._in_use = False  # prepare() was called: the module is bound to a runtime from now on
        # The element lookups and the shape constants, per (rows, device): built once and never freed, since
        # captured graphs read them (the batch is not kept with them).
        self._constants: dict[tuple, tuple[Tensor, ...]] = {}

    # --- ModelAdapter -----------------------------------------------------

    def topology_spec(self) -> TopologySpec:
        """Directed edges within the model's cutoff, ``num_layers`` layers, no triplets."""
        readout = self.module.readout_fn
        return TopologySpec(
            cutoff=float(readout.cutoff),
            num_interactions=int(readout.num_layers),
            triplet_cutoff=None,
            undirected_pairs=False,
            model_dtype=str(readout.cutoff.dtype).removeprefix("torch."),
            edge_order="by_centre",  # TACE sums messages per receiver; the order within one is free
        )

    def padding_rule(self) -> PaddingRule:
        """Far self-loops, which each call moves onto the adapter's sink row (see the module docstring)."""
        return _FAR_SELF_LOOPS

    def prepare(self, batch: GraphBatch) -> TACEInput:
        """The batch and its shape constants; rejects elements outside the model's table (host check, outside
        any graph). A batch of several systems (``batch.systems``) gets the constants of its graph count; each
        call builds the systems' graph index and ``ptr`` from the batch's system buffers."""
        rows = batch.species.shape[0]
        if batch.positions is None or batch.positions.shape[0] != rows or batch.n_real != rows:
            raise ValueError("TACE reads Cartesian positions: an edges-only graph batch without sink rows")
        device = batch.species.device
        key = (rows, str(device))
        numbers = self.module.readout_fn.atomic_numbers.to(device=device, dtype=torch.int64)
        systems = batch.systems
        if systems is not None and systems.system_of_row.shape[0] != rows:
            raise ValueError(f"TACE: the batch's systems cover {systems.system_of_row.shape[0]} rows, not its {rows}")
        if systems is not None and (rows, str(device), systems.count) not in self._constants:
            self._constants[(rows, str(device), systems.count)] = (
                torch.tensor([systems.count], dtype=torch.int64, device=device),
                torch.tensor([rows + 1], dtype=torch.int64, device=device))
        if key not in self._constants:
            dtype = self.module.readout_fn.cutoff.dtype
            elements = numbers.shape[0]
            one_hot = torch.eye(elements, dtype=dtype, device=device)
            # NaN rows for elements outside the model: numbers changed in place after this host check (a
            # captured graph's new system) give non-finite results, never a silently wrong energy
            node_attrs_by_z = torch.full((_ELEMENTS + 1, elements), float("nan"), dtype=dtype, device=device)
            node_attrs_by_z[numbers] = one_hot
            graph_index = torch.zeros(rows + 1, dtype=torch.int64, device=device)
            graph_index[rows] = 1
            self._constants[key] = (
                graph_index, torch.tensor([0, rows, rows + 1], dtype=torch.int64, device=device),
                torch.full((rows + 1,), self._fidelity, dtype=torch.int64, device=device),
                torch.arange(rows + 1, dtype=torch.int64, device=device),
                torch.tensor(rows, dtype=torch.int64, device=device),
                one_hot[:1].clone(), torch.zeros((1, 3), dtype=dtype, device=device), node_attrs_by_z)
        (graph_index, ptr, node_fidelity, atom_index, sink, sink_attrs, sink_position,
         node_attrs_by_z) = self._constants[key]
        species = batch.species.long()
        supported = torch.zeros(_ELEMENTS + 1, dtype=torch.bool, device=device)
        supported[numbers] = True
        if bool(((species < 1) | (species > _ELEMENTS) | ~supported[species.clamp(0, _ELEMENTS)]).any()):
            raise ValueError(f"TACE model {self.checkpoint.get('name', '')!r} supports atomic numbers "
                             f"{sorted(int(z) for z in numbers.tolist())}")
        self._in_use = True
        inputs = TACEInput(batch=batch, graph_index=graph_index, ptr=ptr, node_fidelity=node_fidelity,
                           atom_index=atom_index, sink=sink, sink_attrs=sink_attrs, sink_position=sink_position,
                           node_attrs_by_z=node_attrs_by_z)
        if systems is not None:
            inputs.num_graphs = systems.count + 1
            inputs.sink_graph, inputs.ptr_end = self._constants[(rows, str(device), systems.count)]
        return inputs

    @staticmethod
    def _system_graphs(inputs: TACEInput) -> tuple[Tensor, Tensor]:
        """``(graph_index [rows + 1], ptr [count + 2])`` of a batch of several systems, from its system buffers
        (read in the call: a slot's system is rewritten in place): system ``s``'s occupied rows in graph ``s``, a
        slot's vacant rows and the sink in the last graph; ``ptr`` the prefix sums of the graphs' atom counts (each
        system's atoms, then the vacant rows and the sink), the only use upstream's ``ScaleShift`` makes of it."""
        systems = inputs.batch.systems
        system = systems.system_of_row.to(torch.int64)
        if systems.atoms is None:  # every row is occupied: the segments are the graphs
            return torch.cat((system, inputs.sink_graph)), torch.cat((systems.row_ptr, inputs.ptr_end))
        rows = system.shape[0]
        local = inputs.atom_index[:rows] - systems.row_ptr.index_select(0, system)
        occupied = local < systems.atoms.index_select(0, system)
        graph_index = torch.cat((torch.where(occupied, system, inputs.sink_graph), inputs.sink_graph))
        return graph_index, torch.cat((systems.row_ptr[:1], systems.atoms.cumsum(0), inputs.ptr_end))

    def isolated_atom_policy(self) -> str:
        """"model": TACE evaluates an atom without neighbours itself."""
        return "model"

    def reference_energies(self) -> Tensor:
        """The reference energy of each element as a ``[1, 118]`` float64 row (column ``Z - 1``; NaN for elements
        outside the model), read-only."""
        readout = self.module.readout_fn
        energies = readout.atomic_energy_layer.atomic_energy.detach()[self._fidelity]
        numbers = readout.atomic_numbers.long().to(energies.device)
        row = torch.full((1, _ELEMENTS), float("nan"), dtype=torch.float64, device=energies.device)
        row[0, numbers - 1] = energies.to(torch.float64)
        return row

    # --- EnergyTap ----------------------------------------------------------

    def node_energy(self, inputs: TACEInput, positions: Tensor, cell: Tensor,
                    edge_shifts: Tensor | None = None) -> Tensor:
        """TACE's ``node_energy`` ([rows], the model's dtype: reference energy + scaled readout + ZBL per atom) at
        ``positions`` and ``cell`` (or each edge's shift ``edge_shifts`` in its own system's cell), with TACE's
        own derivatives off; a batch of several systems as one graph per system (see the module docstring)."""
        return self._node_energy(inputs, positions, cell, edge_shifts)

    def _node_energy(self, inputs: TACEInput, positions: Tensor, cell: Tensor, edge_shifts: Tensor | None = None,
                     system_atoms: Tensor | None = None) -> Tensor:
        """:meth:`node_energy`; ``system_atoms`` (``[1]`` int64 on the device): the atoms of the whole system whose
        rows the batch holds part of (a rank's rows at P>1), for the one decision upstream makes per graph
        (``ScaleShift``'s for a graph of one atom without edges), which is the whole system's."""
        from tace.models.lammps import Graph

        batch = inputs.batch
        rows = batch.species.shape[0]
        if system_atoms is not None:
            graph_index, ptr = inputs.graph_index, torch.cat((system_atoms.new_zeros(1), system_atoms,
                                                               system_atoms + 1))
        else:
            graph_index, ptr = (inputs.graph_index, inputs.ptr) if inputs.sink_graph is None \
                else self._system_graphs(inputs)
        dtype = positions.dtype
        centre = batch.edge_index[:, 0].to(torch.int64)
        neighbour = batch.edge_index[:, 1].to(torch.int64)
        if batch.n_edges is None and batch.padding is not None:
            raise RuntimeError("a padded TACE graph batch without its edge count: the padded rows cannot be moved "
                               "onto the sink row")
        if batch.n_edges is not None:
            # padded rows (at or past the device edge count) join the sink row to itself
            real = torch.arange(centre.shape[0], dtype=torch.int64, device=centre.device) < batch.n_edges
            neighbour = torch.where(real, neighbour, inputs.sink)
            centre = torch.where(real, centre, inputs.sink)
        if edge_shifts is None:
            image = batch.edge_image.to(dtype)
            # row by row: no TF32 GEMM for FP32 (an FP64 cell keeps FP64)
            edge_shifts = image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]
        sink_position = inputs.sink_position if inputs.sink_position.dtype == dtype else inputs.sink_position.to(dtype)
        nodes = torch.cat((positions, sink_position))
        node_attrs = torch.cat((inputs.node_attrs_by_z.index_select(0, batch.species), inputs.sink_attrs))
        # TACE's edge (sender, receiver) = (neighbour, centre): x[centre] - x[neighbour] - shift, the engine's
        # edge reversed (upstream adds its shift; the negated engine shift is the same vector, exactly)
        edge_vector = nodes[centre] - nodes[neighbour] - edge_shifts
        model_dtype = inputs.node_attrs_by_z.dtype
        if dtype != model_dtype:
            # float64 geometry for a float32 module (Float64GeometryInput): the float64 vectors cast at the module's
            # input, the positions and cell it is handed in its dtype
            edge_vector, nodes, cell = edge_vector.to(model_dtype), nodes.to(model_dtype), cell.to(model_dtype)
        edge_length = (edge_vector ** 2).sum(dim=1, keepdim=True).sqrt() + 1e-9
        # one lattice per graph (the energy does not read it: upstream's direct-virial heads and long-range term do)
        lattice = cell.view(1, 3, 3).expand(inputs.num_graphs, 3, 3)
        edge_index = torch.stack((neighbour, centre))
        data = {
            "positions": nodes,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "batch": graph_index,
            "ptr": ptr,
            "lattice": lattice,
        }
        graph = Graph(lmp=False, lmp_data=None, lmp_natoms=(rows + 1, 0), num_graphs=inputs.num_graphs,
                      displacement=None, positions=nodes, edge_vector=edge_vector, edge_length=edge_length,
                      lattice=lattice, node_fidelity=inputs.node_fidelity, num_atoms_arange=inputs.atom_index,
                      dcutoff=None)
        out = self.module.readout_fn(data, graph)
        return out["node_energy"][:rows]

    # --- capability negotiation ---------------------------------------------

    @property
    def upstream(self) -> str:
        """The package providing the model."""
        from importlib.metadata import version

        oeq = f", openequivariance {version('openequivariance')}" if self.kernel == "oeq" else ""
        return f"TACE {self._version}{oeq}"

    def float64_geometry_input(self) -> bool:
        """True: :meth:`node_energy` takes float64 positions and cell for a float32 module (each edge's vector in
        float64, cast at the module's input; :class:`~fastmd.adapters.protocol.Float64GeometryInput`)."""
        return True

    def system_batching(self) -> str:
        """"union": the module (upstream's ``e3nnTACE``, the one class the adapter loads) gives each atom an energy
        from its own system's atoms only (messages, neighbour density and attention per receiving atom, a per-atom
        readout); the one decision it makes per graph, ``ScaleShift``'s for a one-atom graph without edges, is
        made per system (:meth:`node_energy` hands the module one graph per system)."""
        return "union"

    def edge_shift_input(self) -> bool:
        """True: :meth:`node_energy` takes the engine's per-edge shifts (the module reads edge vectors; its
        lattice enters no energy), so the systems of one batch may have cells of their own."""
        return True

    def capture_contract(self) -> CaptureContract:
        """No host synchronization in a call while the patch is applied and not reverted (captured whole), padding
        no real atom sees (the sink row), held to upstream's own call within the eager noise floor (atomic scatters:
        not bitwise), the engine-owned derivative with the stress in eV/Å³, no torch.compile."""
        sync_free = bool(self.patches) and all(outcome.applied and not outcome.reverted for outcome in self.patches)
        return CaptureContract(sync_free=sync_free, exact_cutoff=True, accuracy_class="fp32_noise",
                               force_mode="engine_autograd", stress_unit="eV/A^3", compiled="none")

    def capacity_step(self) -> CapacityStep:
        """TACE's capacity tiers (``_CAPACITY_STEP``): every padded edge runs the layers' tensor products (on the
        sink row), so a small cell takes a fine pair step."""
        return _CAPACITY_STEP

    def kernel_backends(self) -> tuple[KernelChoice, ...]:
        """The interactions' fused gather-tensor-product-scatter: OpenEquivariance (fastest) or e3nn (the
        reference)."""
        return (KernelChoice(site=KERNEL_SITE, options=KERNEL_OPTIONS, reference="e3nn"),)

    def select_kernel(self, site: str, option: str) -> None:
        """Install ``option`` ("oeq" or "e3nn") at the kernel site through upstream's own switch: the module is
        built again from its checkpoint with ``TACE_USE_OEQ`` set (OpenEquivariance) or unset (e3nn), and run once
        on a two-atom input (OpenEquivariance compiles its kernels on first use). All or nothing: on an error the
        module stays. The kernels are chosen before a runtime binds the module: once :meth:`prepare` has run, only
        the installed option is accepted."""
        if site != KERNEL_SITE or option not in KERNEL_OPTIONS:
            raise ValueError(f"TACE kernel site {KERNEL_SITE!r} has options {KERNEL_OPTIONS}, got {site!r}/{option!r}")
        if option == self.kernel:
            return
        if self._in_use:
            raise RuntimeError(f"TACE kernels are chosen when the model is loaded (installed: {self.kernel!r}); "
                               f"load it with the profile that selects {option!r}")
        path = self.checkpoint.get("path")
        if not path:
            raise RuntimeError("this TACE adapter was built from a module, not a checkpoint: its kernels are the "
                               "module's")
        module = _build(path, self.module.readout_fn.cutoff.device, option)
        _warm_up(module)
        self.module = module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.kernel = option
        self.kernel_reason = ""

    def system_inputs(self) -> dict[str, Any]:
        """The static per-system input: the model's fidelity."""
        return {"fidelity": self._fidelity}

    def upstream_patches(self) -> list:
        """The patch outcomes (the plan's ``patches``)."""
        return list(self.patches)

    # --- multi-GPU (the distributed runtime's node-halo exchange) -----------

    def _trimmed_modules(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The ghost-row trimming at P>1 (``ShardSpec.owned_slot_inputs``, ``zero_ghost_outputs``, one trailing sink
        row): in every interaction the per-atom stages after the message sum (``linear_down``, the gate
        ``nonlinearity``, the residuals ``resnetBB`` / ``resnetBA`` / ``resnetAB`` where present, which may read the
        atom attributes, and the stochastic depth, which reads the graph index), every product and every energy
        readout run on the owned slots; ``linear_down``'s output where the interaction divides it by its neighbour
        density (a sum over every row) and each readout's (gathered over every row) get zero ghost rows and a zero
        sink row. The ghost rows receive no messages, and what an interaction computes on them after its sum is never
        read (the next interaction's ``linear_up`` output takes their owners' rows, the energy the owned rows'). The
        first interaction's ``linear_up`` and every rejector stay on every row; a later ``linear_up`` reads the
        previous product's owned slots, and the exchange appends the ghost rows and the sink row. Nothing is declared
        for a model with another per-atom head than the energy (direct forces, stress, dipoles, charges...), which
        reads the same features on every row: it runs P>1 untrimmed."""
        readout = self.module.readout_fn
        if any("readout" in name and name != "energy_readouts" for name, _ in readout.named_children()):
            return (), ()
        prefix = "readout_fn.representation"
        representation = readout.representation
        owned, padded = [], []
        for index, layer in enumerate(representation.interactions):
            path = f"{prefix}.interactions.{index}"
            owned += [f"{path}.linear_down", f"{path}.nonlinearity"]
            owned += [f"{path}.{name}" for name in ("resnetBB", "resnetBA", "resnetAB", "stochastic_depth")
                      if hasattr(layer, name)]
            if hasattr(layer, "edge_density"):
                padded.append(f"{path}.linear_down")
        owned += [f"{prefix}.products.{index}" for index in range(len(representation.products))]
        readouts = [f"readout_fn.energy_readouts.{index}" for index in range(len(readout.energy_readouts))]
        return tuple(owned + readouts), tuple(padded + readouts)

    def _ghost_points(self) -> list[tuple[int, ExchangePoint]]:
        """The exchange before every interaction but the first: ``(layer, point)``, the point named after the
        interaction's ``linear_up``, whose output (the node features the interaction's rejector gathers at the
        senders) has ``width`` values per row."""
        interactions = self.module.readout_fn.representation.interactions
        return [(index, ExchangePoint(name=f"interactions.{index}.linear_up", module=None,
                                      width=int(layer.linear_up.irreps_out.dim)))
                for index, layer in enumerate(interactions) if index > 0]

    def shard_spec(self) -> ShardSpec:
        """TACE at P>1 on the node-halo exchange. Every interaction sums its messages at the receiver (the edge's
        centre), and reads other atoms only through its rejector, which gathers the senders' rows of the
        interaction's ``linear_up`` output (the RRA models' attention included); the neighbour density (without the
        envelope in TACE-OAM-L) and the attention's softmax are sums over the receiver's own edges, the ZBL pair term
        is summed at the receiver, the edge updates read the species of an edge's two atoms, and every other
        computation reads its own atom's row. So with the receiver edge rule an owned atom's sums are complete, and in
        every interaction but the first (whose features come from the species, on every row) the ghost rows of
        ``linear_up``'s output take their owners' rows: one COPY of that width per ghost atom and layer (2,304 values
        for TACE-OAM-L, 1,600 for the RRA models; upstream's own LAMMPS path exchanges the same tensor). The adapter
        calls the exchanges itself (:meth:`sharded_energy`): each rank's graph keeps its own sink row after its rows
        (the padded edge rows move onto it, as on one GPU), which the exchange leaves out, and ``ScaleShift``'s one
        decision per graph is the whole system's (the system's atom count, not the rank's rows). Unused atom slots get
        the first element of the model's table. Pins (``_SHARD_PINS``): the layer loop, the interaction, rejector,
        edge-update, product and ZBL classes in the module and upstream's ghost helpers; a class without a pin gives
        a pin that cannot match, so P>1 is refused (the error names the pin); a model with one interaction (no
        exchange point) or without the adapter's ScaleShift replacement installed raises NegotiationError here. TF32
        on request (``matmul_precision="high"``) runs at P>1 for the checkpoints measured (``_TF32_AT_P_GT_1``:
        TACE-OAM-L, whose TF32 P=2 and P=4 forces, energy and stress stay within the gate set by TF32 P=1's own
        run-to-run noise); any other refuses it. Capacity tiers: TACE's (every padded edge row costs a real one)."""
        from ..adapters.patches import PatchRecord

        readout = self.module.readout_fn
        representation = readout.representation
        interactions = list(representation.interactions)
        classes = ([type(representation), type(representation.node_embedding), type(representation.edge_embedding)]
                   + [type(layer) for layer in interactions] + [type(layer.rejector) for layer in interactions]
                   # the kernel a rejector delegates to (OpenEquivariance's fused gather-scatter)
                   + [type(layer.rejector.fused_tp) for layer in interactions if hasattr(layer.rejector, "fused_tp")]
                   + [type(update) for update in representation.edge_updates]
                   + [type(product) for product in representation.products])
        if hasattr(readout, "zbl"):
            classes.append(type(readout.zbl))
        # The ghost-row trimming relies on its modules reading their own atom's row (TACE's own classes; a library
        # module, e3nn's Gate, is a per-row map).
        trimmed_paths = dict.fromkeys(name for pair in self._trimmed_modules() for name in pair)
        classes += [cls for cls in (type(self.module.get_submodule(path)) for path in trimmed_paths)
                    if cls.__module__.split(".")[0] == "tace"]
        targets = list(dict.fromkeys([f"{cls.__module__}.{cls.__qualname__}.forward" for cls in classes]
                                     + list(_GHOST_HELPERS)))
        pins = tuple(PatchRecord(name=f"tace.shard.{'.'.join(target.rsplit('.', 2)[-2:])}", target=target,
                                 distribution="TACE", versions=TACE_VERSIONS,
                                 source_sha256=_SHARD_PINS.get(target, "0" * 64),
                                 reason="the multi-GPU exchange relies on this code unchanged")
                     for target in targets)
        if len(interactions) < 2:
            # Without an exchange point the runtime would not run the adapter's own forward (sharded_energy), whose
            # whole-system ScaleShift decision P>1 needs
            raise NegotiationError("TACE on several GPUs: a model with one interaction declares no exchange point, and "
                                   "P>1 runs the adapter's own forward only through one; run it on one GPU")
        if not any(outcome.record is SCALE_SHIFT_EDGE_COUNT and outcome.applied and not outcome.reverted
                   for outcome in self.patches):
            raise NegotiationError("TACE on several GPUs needs the adapter's ScaleShift.forward replacement installed: "
                                   "P>1 hands ScaleShift the whole system's atom count in ptr, which the replacement "
                                   "reads for the counts only (this ScaleShift.forward is another source, or the "
                                   "adapter was closed)")
        owned_slots, zero_ghosts = self._trimmed_modules()
        return ShardSpec(exchange_points=tuple(point for _, point in self._ghost_points()),
                         pad_species=int(readout.atomic_numbers[0]), source_pins=pins,
                         allow_tf32=self.checkpoint.get("sha256") in _TF32_AT_P_GT_1, capacity_step=_CAPACITY_STEP,
                         owned_slot_inputs=owned_slots, zero_ghost_outputs=zero_ghosts,
                         trailing_rows=1 if owned_slots else 0)

    def sharded_energy(self, inputs: TACEInput, positions: Tensor, cell: Tensor, ctx) -> Tensor:
        """This rank's share of the energy at P>1 (``ShardedTap``): :meth:`node_energy` of the rank's rows (owned
        slots, then ghost rows) with the exchange of :meth:`shard_spec` on each interaction's ``linear_up`` output
        (forward hooks for this call) and ``ScaleShift``'s decision made for the whole system (``ctx.n_total``
        atoms); the FP64 sum of the owned rows' energies (a select: an unused slot's or ghost row's energy may be
        anything)."""
        rows = inputs.batch.species.shape[0]
        interactions = self.module.readout_fn.representation.interactions
        hooks = [interactions[index].linear_up.register_forward_hook(_GhostCopy(point, ctx, rows))
                 for index, point in self._ghost_points()]
        token = _SHARDED_FORWARD.set(ctx)
        try:
            total = None if ctx.n_total is None else ctx.n_total.to(torch.int64).reshape(1)
            energies = self._node_energy(inputs, positions, cell, system_atoms=total).to(torch.float64)
        finally:
            _SHARDED_FORWARD.reset(token)
            for hook in hooks:
                hook.remove()
        return torch.where(ctx.owned, energies, energies.new_zeros(())).sum().reshape(1)

    def close(self) -> None:
        """Revert the applied patches (the adapter is not used afterwards)."""
        revert_patches(self.patches)
