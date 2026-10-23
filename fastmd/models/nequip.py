"""NequIP's native energy network over a FastMD graph.

Supported files are local ``.nequip.zip`` packages and current ``.ckpt``
training checkpoints. Loading uses NequIP's own eager loader, including its
evaluation/EMA selection; there are no downloads. Packaged Python classes
belong to a torch.package namespace, so wrappers are identified by the
upstream graph-model flag and class name rather than ``isinstance``.

Only the outer ForceStressOutput wrapper is bypassed. The original energy
network, per-type scaling/shifts and pair potential remain intact. FastMD
differentiates its atomic energies once for forces and strain derivatives.
NequIP's edges are (centre, neighbour), with the neighbour's lattice shift,
exactly as in GraphBatch. Geometry and energy shifts use float64 upstream,
even for float32 neural weights; ``geometry_dtype`` preserves that distinction.

By default radius/per-type filtering removes padding with dynamic shapes.
Explicit ``fixed_shape=True`` retains all edges and enables capture only after
checking the standard upstream polynomial-cutoff/bias-free radial network.
Its far self-loops have zero messages and ZBL energy, including derivatives.
"""
from __future__ import annotations

import hashlib
import contextlib
import io
import math
import os
from pathlib import Path
import threading

import torch
from torch import Tensor

from fastmd.adapters.protocol import CapacityStep, CaptureContract, GraphBatch, KernelChoice, PaddingRule, TopologySpec


NEQUIP_NAMES = frozenset({"nequip", "nequip-oam-l", "nequip-oam-xl", "nequip-mp-l"})
KERNEL_SITE = "tensor_product_scatter"
_PACKAGE_LOAD_LOCK = threading.RLock()


@contextlib.contextmanager
def cpu_package_load_context():
    """Map nested e3nn FX pickles to CPU during official package loading.

    PackageImporter(map_location='cpu') does not propagate its mapping into
    e3nn CodeGenMixin's pickle.loads buffers. The published OAM packages retain
    CUDA storage tags there. Scope the compatibility hook to this loading
    thread and restore it even on failure; model execution is never patched.
    """
    with _PACKAGE_LOAD_LOCK:
        original = torch.storage._load_from_bytes
        owner = threading.get_ident()

        def load_storage(data):
            if threading.get_ident() != owner:
                return original(data)
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

        torch.storage._load_from_bytes = load_storage
        try:
            yield
        finally:
            torch.storage._load_from_bytes = original


def is_nequip_model(model, model_path=None) -> bool:
    if not isinstance(model, (str, os.PathLike)):
        return False
    name = os.fspath(model).lower().replace("_", "-")
    for version in (":0.1", "-0.1"):
        if name.endswith(version):
            name = name[:-len(version)]
            break
    return name in NEQUIP_NAMES


def check_load_options(model_dtype, optimization_profile, model_path):
    from fastmd.adapters.protocol import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError("optimization_profile='legacy' is MatRIS-only")
    if model_dtype not in (None, "float32", "float64"):
        raise ValueError("NequIP model_dtype must be None, 'float32', or 'float64'")
    if model_path is None:
        raise ValueError("NequIP requires an explicit local model_path (.nequip.zip or .ckpt)")


def _apply_cueq(graph_model):
    """Use the model-discovered official modifier, including packaged classes.

    The installed TensorProductScatter class is not the class inside a
    torch.package model: calling its modifier directly can replace zero layers.
    Verify both replacement count and every original state's exact bytes while
    all tensors are still on CPU. Additional cuEq buffers are allowed.
    """
    from nequip.model import modify

    def signatures(model):
        result = {}
        for name, value in model.state_dict().items():
            if value.device.type != "cpu":
                raise ValueError("NequIP cuEq conversion must run before moving the model from CPU")
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            result[name] = (str(value.dtype), tuple(value.shape), hashlib.sha256(raw).hexdigest())
        return result

    count = sum(type(part).__name__ == "TensorProductScatter" for part in graph_model.modules())
    if count < 1 or any(type(part).__name__ == "CuEquivarianceTensorProductScatter"
                        for part in graph_model.modules()):
        raise ValueError("NequIP kernel='cueq' requires an unmodified upstream TensorProductScatter model")
    before = signatures(graph_model)
    # cuEq 0.9.1's naive scatter cannot evaluate N > 1 atoms with zero edges.
    # Keep only the original energy network's isolated-atom output as an
    # independent reference for padding validation, not a second model copy.
    energy_model = graph_model.model
    if type(energy_model).__name__ == "ForceStressOutput":
        energy_model = energy_model.func
    n_types = len(graph_model.metadata["type_names"].split())
    with torch.no_grad():
        isolated_reference = energy_model({"atom_types": torch.arange(n_types),
            "pos": torch.zeros(n_types, 3, dtype=torch.float64),
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "edge_vectors": torch.empty(0, 3, dtype=torch.float64)})["atomic_energy"].detach().clone()
    try:
        graph_model = modify(graph_model, modifiers=[{"modifier": "enable_CuEquivariance"}]).eval()
    except ImportError as error:
        raise ImportError("NequIP kernel='cueq' requires optional cuequivariance, cuequivariance-torch "
                          "and matching CUDA ops (validated with 0.9.1)") from error
    converted = sum(type(part).__name__ == "CuEquivarianceTensorProductScatter" for part in graph_model.modules())
    remaining = sum(type(part).__name__ == "TensorProductScatter" for part in graph_model.modules())
    if converted != count or remaining:
        raise ValueError(f"NequIP cuEq modifier replaced {converted}/{count} tensor products ({remaining} remain)")
    after = signatures(graph_model)
    changed = [name for name, value in before.items() if after.get(name) != value]
    if changed:
        raise ValueError(f"NequIP cuEq modifier changed saved state tensors: {changed}")
    # A plain CPU tensor attribute (not a parameter/buffer); its 89 values in
    # OAM-L survive a GraphModel deepcopy without adding state-dict entries.
    graph_model._fastmd_cueq_isolated_reference = isolated_reference
    return graph_model, {"requested": "cueq", "modifier": "enable_CuEquivariance",
                         "tensor_products_converted": converted, "original_state_tensors": len(before),
                         "original_state_tensors_unchanged": True, "added_state_tensors": len(after) - len(before)}


def load_nequip(model="nequip", *, model_path=None, device=None,
                 model_dtype=None, optimization_profile=None, fixed_shape=False, kernel="upstream"):
    """Load a trusted local NequIP package/checkpoint with the upstream loader.

    The saved neural precision is retained. Converting with ``Module.to(dtype)``
    would leave cached NequIP dtype attributes inconsistent, so an explicit
    ``model_dtype`` must match the saved precision. ``kernel='cueq'`` explicitly
    applies the official modifier on CPU, retaining saved state tensors. CUDA
    use requires the fused_tp implementation; unavailable CUDA ops never
    silently fall back. CPU evaluation uses cuEq's own naive fallback.
    """
    if kernel not in ("upstream", "cueq"):
        raise ValueError("NequIP kernel must be 'upstream' or 'cueq'")
    check_load_options(model_dtype, optimization_profile, model_path)
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NequIP model not found: {path}")
    if not (path.name.endswith(".nequip.zip") or path.suffix == ".ckpt"):
        raise ValueError("NequIP accepts local .nequip.zip packages or current .ckpt training checkpoints; "
                         "legacy deployed and compiled artifacts are not supported")
    try:
        from nequip.model.saved_models.load_utils import load_saved_model
        from nequip.utils.global_state import set_global_state
        from lightning.pytorch.utilities.seed import isolate_rng
    except ImportError as error:
        raise ImportError("NequIP requires the optional nequip==0.14.0 package") from error
    from fastmd.precision import restore_fp32_state

    old_dtype = torch.get_default_dtype()
    kernel_metadata = {"requested": kernel}
    # NequIP initialization sets the default dtype and RNGs. Keep those changes
    # out of the caller's simulation setup while following its official loader.
    try:
        with isolate_rng(include_cuda=False), restore_fp32_state(), cpu_package_load_context():
            set_global_state(allow_tf32=False)
            # set_global_state initializes dtype only on its first invocation.
            # Every load must use the upstream geometry default, including when
            # a caller restored its own default after a previous NequIP load.
            torch.set_default_dtype(torch.float64)
            graph_model = load_saved_model(str(path), compile_mode="eager").eval()
            if kernel == "cueq":
                graph_model, kernel_metadata = _apply_cueq(graph_model)
    finally:
        torch.set_default_dtype(old_dtype)
    saved_dtype = graph_model.metadata["model_dtype"]
    if model_dtype is not None and model_dtype != saved_dtype:
        raise ValueError(f"NequIP checkpoint uses {saved_dtype}; model_dtype={model_dtype!r} would change "
                         "its saved precision")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if kernel == "cueq" and target.type == "cuda":
        backends = [type(part.tp_conv.m).__name__ for part in graph_model.modules()
                    if type(part).__name__ == "CuEquivarianceTensorProductScatter"]
        if not backends or any(name != "SegmentedPolynomialFusedTP" for name in backends):
            raise RuntimeError(f"NequIP kernel='cueq' requires CUDA fused_tp in every layer; found {backends}")
    graph_model.to(device=target)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return NequIPAdapter(graph_model, fixed_shape=fixed_shape, kernel_metadata=kernel_metadata,
                         checkpoint={"name": os.fspath(model), "path": str(path),
                                                 "sha256": digest})


class NequIPAdapter:
    name = "nequip"
    compiled = False
    geometry_dtype = torch.float64

    def __init__(self, graph_model, *, checkpoint=None, fixed_shape=False, kernel_metadata=None):
        from ase.data import atomic_numbers

        if not getattr(graph_model, "is_graph_model", False) or getattr(
                graph_model, "is_compile_graph_model", False):
            raise TypeError("NequIPAdapter requires an eager upstream GraphModel")
        self.metadata = dict(graph_model.metadata)
        self.checkpoint = dict(checkpoint or {})
        self.type_names = self.metadata["type_names"].split()
        if not self.type_names or len(set(self.type_names)) != len(self.type_names):
            raise ValueError("NequIP checkpoint requires distinct chemical type names")
        if any(name not in atomic_numbers or atomic_numbers[name] < 1 for name in self.type_names):
            raise ValueError("NequIP FastMD mapping requires chemical-symbol type_names in the saved order")
        self._dtype = self.metadata["model_dtype"]
        if self._dtype not in ("float32", "float64"):
            raise ValueError(f"Unsupported NequIP model dtype: {self._dtype}")
        self._cutoff = float(self.metadata["r_max"])
        if not math.isfinite(self._cutoff) or self._cutoff <= 0:
            raise ValueError("NequIP r_max must be finite and positive")

        energy_model = graph_model.model
        if type(energy_model).__name__ == "ForceStressOutput":
            energy_model = energy_model.func
        if any(type(part).__name__ in ("ForceStressOutput", "PartialForceOutput")
               for part in energy_model.modules()):
            raise ValueError("NequIP energy tap cannot contain nested derivative wrappers")
        if "atomic_energy" not in energy_model.irreps_out:
            raise ValueError("NequIP energy network must return per-atom energies")
        self.module = energy_model.eval()
        self._layers = sum(type(part).__name__ == "ConvNetLayer" for part in self.module.modules())
        if self._layers < 1:
            raise ValueError("NequIP adapter requires a message-passing NequIP network")
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        target = next(self.module.parameters()).device
        numbers = [atomic_numbers[name] for name in self.type_names]
        self._species_indices = torch.full((max(numbers) + 1,), -1, dtype=torch.long, device=target)
        self._species_indices[numbers] = torch.arange(len(numbers), device=target)
        self._checked_species = None
        self._edge_cutoffs = None
        per_edge = self.metadata.get("per_edge_type_cutoff")
        if per_edge not in (None, ""):
            # Both packaged source versions use a flattened row-major metadata
            # string; parsing it directly avoids private upstream helper APIs.
            values = [float(value) for value in per_edge.split()]
            if len(values) != len(numbers) ** 2 or any(
                    not math.isfinite(value) or value <= 0 or value > self._cutoff for value in values):
                raise ValueError("Invalid NequIP per-edge cutoff metadata")
            self._edge_cutoffs = torch.tensor(values, device=target, dtype=self.geometry_dtype).reshape(
                len(numbers), len(numbers))
        cueq_blocks = [part for part in self.module.modules()
                       if type(part).__name__ == "CuEquivarianceTensorProductScatter"]
        registered = [type(part.tp_conv.m).__name__ for part in cueq_blocks]
        executed = [type(part.tp_conv.fallback if target.type == "cpu" else part.tp_conv.m).__name__
                    for part in cueq_blocks]
        self.kernel_metadata = dict(kernel_metadata or {})
        self.kernel_metadata.update(registered_backend_classes=registered, execution_backend_classes=executed)
        self.kernel = ("upstream" if not cueq_blocks else
                       "cueq_fused_tp" if all(name == "SegmentedPolynomialFusedTP" for name in executed)
                       else "cueq_naive" if all(name == "SegmentedPolynomialNaive" for name in executed)
                       else "cueq_unknown")
        self.kernel_reason = "saved eager NequIP energy network; dynamic radius filtering prevents capture"
        if cueq_blocks:
            self.kernel_reason += f"; official CuEquivariance modifier, execution backend {self.kernel}"
        if not isinstance(fixed_shape, bool):
            raise TypeError("NequIP fixed_shape must be a bool")
        self.fixed_shape = fixed_shape
        # Allocate this host-specified constant before any CUDA graph capture.
        self._empty_edge_vector = (torch.tensor([2 * self._cutoff, 0., 0.],
            dtype=self.geometry_dtype, device=target) if fixed_shape and cueq_blocks else None)
        self.padding_validation = None
        if fixed_shape:
            self.padding_validation = self._validate_fixed_shape(
                isolated_reference=getattr(graph_model, "_fastmd_cueq_isolated_reference", None))
            self.kernel_reason = (f"{self.kernel} NequIP network; "
                                  "validated zero-contribution far self-loop padding")

    def _validate_fixed_shape(self, *, isolated_reference=None):
        """Fail closed for networks outside the audited upstream energy pipeline.

        BesselEdgeLengthEncoding multiplies its basis by PolynomialCutoff,
        whose value and derivative vanish for x >= 1. Bias-free SiLU radial
        MLPs therefore produce zero TP weights there. ZBL multiplies by the
        same cutoff. All scatters have an explicit atom count, so adding zero
        messages cannot change the normalization or introduce host reads.
        Packaged classes have a torch.package prefix, hence the name checks.
        The numerical probe supplements these structural checks; it is not
        used to certify arbitrary/custom architectures from a few samples.
        """
        def require(condition, detail):
            if not condition:
                raise ValueError(f"NequIP fixed_shape validation failed: {detail}")

        allowed = {"SphericalHarmonicEdgeAttrs", "EdgeLengthNormalizer", "BesselEdgeLengthEncoding",
                   "ApplyFactor", "NodeTypeEmbed", "ConvNetLayer", "ScalarMLP",
                   "PerTypeScaleShift", "ZBL", "AtomwiseReduce"}
        top = list(self.module.children())
        require(type(self.module).__name__ == "SequentialGraphNetwork", "expected SequentialGraphNetwork")
        for part in top:
            require(type(part).__name__ in allowed and "nequip.nn." in type(part).__module__,
                    f"unsupported energy module {type(part).__name__}")
        encoders = [part for part in top if type(part).__name__ == "BesselEdgeLengthEncoding"]
        normalizers = [part for part in top if type(part).__name__ == "EdgeLengthNormalizer"]
        require(len(encoders) == len(normalizers) == 1, "requires one Bessel encoder and length normalizer")
        encoder, normalizer = encoders[0], normalizers[0]
        require(type(encoder.cutoff).__name__ == "PolynomialCutoff" and encoder.cutoff.p >= 2,
                "requires upstream PolynomialCutoff(p >= 2)")
        require(encoder.edge_invariant_field == "edge_embedding" and
                encoder.norm_length_field == normalizer.norm_length_field == "normed_edge_lengths",
                "unsupported radial field mapping")
        target = self._species_indices.device
        expected_recip = (self._edge_cutoffs.reciprocal().reshape(-1) if self._edge_cutoffs is not None
                          else torch.tensor(1. / self._cutoff, dtype=self.geometry_dtype, device=target))
        require(torch.equal(normalizer._rmax_recip, expected_recip), "cutoff metadata/network mismatch")
        require(bool((self._cutoff * expected_recip >= 1).all()),
                "rounded length normalization does not reach the exact global cutoff")
        blocks = [part.conv for part in top if type(part).__name__ == "ConvNetLayer"]
        require(len(blocks) == self._layers, "unsupported nested message passing")
        for block in blocks:
            require(type(block).__name__ == "InteractionBlock" and
                    type(block.tp_scatter).__name__ in ("TensorProductScatter", "CuEquivarianceTensorProductScatter") and
                    type(block.ghost_exchange).__name__ == "NoOpGhostExchangeModule",
                    "requires standard eager or official cuEq InteractionBlock/TP scatter without ghost exchange")
            if type(block.tp_scatter).__name__ == "CuEquivarianceTensorProductScatter":
                require("nequip.nn._tp_scatter_cueq" in type(block.tp_scatter).__module__ and
                        block.tp_scatter.tp_conv.method == "fused_tp", "requires official cuEq fused_tp modifier")
                require(target.type == "cpu" or type(block.tp_scatter.tp_conv.m).__name__ == "SegmentedPolynomialFusedTP",
                        "CUDA capture requires cuEq fused_tp; naive fallback is unsupported")
            radial = block.edge_mlp
            require(type(radial).__name__ == "ScalarMLPFunction" and radial.bias is False,
                    "radial MLP must be bias-free")
            require(all(part.bias is None for part in radial.modules()
                        if type(part).__name__ == "ScalarLinearLayer"), "radial linear bias is unsupported")
            require(all(type(part).__name__ in ("ScalarLinearLayer", "SiLU") for part in radial.mlp.children()),
                    "requires standard zero-preserving SiLU radial layers")
            zeros = torch.zeros(2, encoder.num_bessels, dtype=getattr(torch, self._dtype), device=target)
            with torch.no_grad():
                require(bool((radial(zeros) == 0).all()), "radial MLP does not map zero to zero")
        with torch.enable_grad():
            x = torch.tensor([1., 1.01, 2., 4.], dtype=self.geometry_dtype, device=target, requires_grad=True)
            cutoff = encoder.cutoff(x)
            gradient = torch.autograd.grad(cutoff.sum(), x)[0]
            require(bool((cutoff == 0).all() & (gradient == 0).all()), "cutoff value/gradient is nonzero")
            # Include every saved species, including the ZBL atomic-number map.
            n = len(self.type_names)
            types = torch.arange(n, device=target)
            empty_index = torch.empty(2, 0, dtype=torch.long, device=target)
            empty_vec = torch.empty(0, 3, dtype=self.geometry_dtype, device=target)
            def energy(index, vectors):
                return self.module({"atom_types": types, "edge_index": index, "edge_vectors": vectors,
                                    "pos": torch.zeros(n, 3, dtype=self.geometry_dtype, device=target)})["atomic_energy"]
            if self.kernel.startswith("cueq"):
                require(isolated_reference is not None,
                        "cuEq padding requires the upstream isolated reference from load_nequip(kernel='cueq')")
                expected = isolated_reference.to(device=target)
                require(expected.shape == (n, 1), "invalid upstream isolated reference shape")
            else:
                with torch.no_grad():
                    expected = energy(empty_index, empty_vec)
            loops = types.repeat(2)
            vectors = torch.zeros(2 * n, 3, dtype=self.geometry_dtype, device=target)
            vectors[:n, 0] = 2 * self._cutoff
            vectors[n:, 1] = 3 * self._cutoff
            vectors.requires_grad_(True)
            actual = energy(torch.stack([loops, loops]), vectors)
            gradient = torch.autograd.grad(actual.sum(), vectors)[0]
            require(torch.equal(actual, expected) and bool(torch.isfinite(actual).all()),
                    "far self-loops change isolated atom energies")
            require(bool((gradient == 0).all()), "far self-loops have nonzero edge derivatives")
        return {"validated": True, "cutoff": "PolynomialCutoff", "radial_bias": False,
                "pair_potential": "ZBL" if any(type(p).__name__ == "ZBL" for p in top) else None,
                "species_checked": n, "interaction_blocks": len(blocks), "far_edge_gradient_max": 0.0,
                "kernel": self.kernel, "isolated_reference": "upstream empty-edge energy network"}

    def topology_spec(self) -> TopologySpec:
        return TopologySpec(cutoff=self._cutoff, num_interactions=self._layers, model_dtype=self._dtype)

    def padding_rule(self) -> PaddingRule:
        return PaddingRule("far_self_loops")

    def prepare(self, batch: GraphBatch) -> GraphBatch:
        if self.fixed_shape and batch.padding is not None and batch.padding.kind != "far_self_loops":
            raise ValueError("NequIP fixed_shape requires far_self_loops padding")
        if batch.species is not self._checked_species:
            numbers = batch.species.to(torch.long)
            if bool(((numbers < 1) | (numbers >= len(self._species_indices))).any()):
                raise ValueError("Atomic number outside the NequIP checkpoint species table")
            if bool((self._species_indices[numbers] < 0).any()):
                raise ValueError("Atomic number absent from the NequIP checkpoint species table")
            self._checked_species = batch.species
        return batch

    def node_energy(self, batch: GraphBatch, positions: Tensor, cell: Tensor) -> Tensor:
        positions, cell = positions.to(self.geometry_dtype), cell.to(self.geometry_dtype)
        edge_index = batch.edge_index.T.to(torch.long).contiguous()
        shifts = batch.edge_image.to(self.geometry_dtype)
        species = self._species_indices[batch.species.to(torch.long)]
        # NequIP's unbatched with_edge_vectors_ arithmetic; centres are row 0.
        vectors = positions[edge_index[1]] - positions[edge_index[0]]
        vectors = vectors + torch.sum(shifts.unsqueeze(-1) * cell, dim=1)
        if batch.edge_vec_override is not None:
            vectors = vectors + batch.edge_vec_override.to(self.geometry_dtype)
        if not self.fixed_shape:
            lengths = torch.linalg.vector_norm(vectors, dim=-1)
            keep = lengths < self._cutoff
            if self._edge_cutoffs is not None:
                keep = keep & (lengths <= self._edge_cutoffs[species[edge_index[0]], species[edge_index[1]]])
            edge_index, shifts, vectors = edge_index[:, keep], shifts[keep], vectors[keep]
        elif self.kernel.startswith("cueq") and edge_index.shape[1] == 0:
            # Static shape branch, never a device scalar read. The upstream
            # cuEq naive implementation confuses empty output indices with an
            # unindexed reduction for N > 1 atoms. Use the already-validated
            # zero-contribution far self-loops in this otherwise empty graph.
            loops = torch.arange(positions.shape[0], device=positions.device)
            edge_index = torch.stack([loops, loops])
            shifts = torch.zeros_like(positions)
            vectors = positions * 0 + self._empty_edge_vector
        data = {"pos": positions, "cell": cell.unsqueeze(0), "atom_types": species,
                "edge_index": edge_index, "edge_cell_shift": shifts, "edge_vectors": vectors}
        return self.module(data)["atomic_energy"].reshape(-1)

    def isolated_atom_policy(self):
        return "model"

    def reference_energies(self):
        return None

    def capture_contract(self) -> CaptureContract:
        return CaptureContract(sync_free=self.fixed_shape and self.padding_validation is not None,
                               exact_cutoff=True, force_mode="engine_autograd",
                               stress_unit="eV/A^3")

    def capacity_step(self) -> CapacityStep:
        # Padded rows still execute all tensor products, so avoid the engine's
        # coarser MatRIS-oriented defaults. Explicit user capacities take priority.
        return CapacityStep(u_step=32, min_pad_u=32, u_thermal_factor=1.01)

    def kernel_backends(self) -> tuple[KernelChoice, ...]:
        """Record the explicitly loaded implementation in every engine plan.

        There is one installed option, including under the generic profile.
        Switching backends requires a fresh load with ``kernel=...`` so no
        negotiation can silently undo the user's choice or stale a capture.
        """
        return (KernelChoice(site=KERNEL_SITE, options=(self.kernel,), reference=self.kernel),)

    def select_kernel(self, site: str, option: str) -> None:
        """Accept only the current backend; negotiation never mutates the model."""
        if site != KERNEL_SITE or option != self.kernel:
            raise ValueError(f"NequIP has only the loaded {KERNEL_SITE} backend {self.kernel!r}; "
                             "load a new adapter to change kernels")

    def close(self):
        """No upstream patches or external resources to release."""
