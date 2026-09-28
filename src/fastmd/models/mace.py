"""ASE adapter for MACE-MP-0 / MACE-MPA-0 and the reb MACE kernels."""
from importlib.util import find_spec
import math

import torch

from .base import ModelBackend, ModelCapabilities


class MACEModel(ModelBackend):
    """Single-system periodic MACE inference; defaults require no graph tuning.

    ``variant='fast'`` enables the migrated Triton fusions. ``fast_cm`` also
    compiles the original GEMM-level modules; ``plain`` captures upstream MACE.
    CPU and explicitly eager calls use upstream MACE in all three cases.
    Set ``compute_stress=True`` for ASE NPT to cache energy, forces and stress
    together on every evaluation, avoiding separate force/stress calls.
    """

    capabilities = ModelCapabilities(frozenset({"energy", "forces", "stress"}),
                                     frozenset({"energy", "forces", "stress"}),
                                     cuda_graph_variable_cell=True)

    def __init__(self, *, checkpoint=None, model_name="medium-mpa-0",
                 default_dtype="float64", enable_cueq=False, head=None,
                 variant="fast", neighbor_skin=1.0, capacity_headroom=1.25,
                 compute_stress=False, **kwargs):
        super().__init__(**kwargs)
        if variant not in {"plain", "fast", "fast_cm"}:
            raise ValueError("MACE variant must be 'plain', 'fast', or 'fast_cm'")
        if not math.isfinite(neighbor_skin) or neighbor_skin <= 0:
            raise ValueError("neighbor_skin must be finite and positive")
        if not math.isfinite(capacity_headroom) or capacity_headroom < 1:
            raise ValueError("capacity_headroom must be finite and >= 1")
        from fastmd._vendor.mace_opt.tensor_batch import load_mace_model
        self.model, self.info = load_mace_model(
            str(checkpoint) if checkpoint is not None else model_name,
            device=str(self.device), default_dtype=default_dtype,
            enable_cueq=enable_cueq, head=head,
        )
        self.variant = variant if self.config.enable_fusions else "plain"
        self.neighbor_skin = float(neighbor_skin)
        self.capacity_headroom = float(capacity_headroom)
        self.compute_stress = bool(compute_stress)
        self.runner = None

    def graph_unavailable_reason(self, properties):
        reason = super().graph_unavailable_reason(properties)
        if reason:
            return reason
        if torch.version.hip is not None or find_spec("triton") is None:
            return "MACE CUDA Graph needs NVIDIA CUDA and Triton; install fastmd-mlip[mace-cuda]"
        # Padding neutrality and the fused forward are validated for these architectures.
        model = self.model
        kinds = {"RealAgnosticResidualInteractionBlock", "RealAgnosticDensityResidualInteractionBlock",
                 "RealAgnosticDensityInteractionBlock"}
        if (type(model).__name__ != "ScaleShiftMACE"
                or any(type(i).__name__ not in kinds for i in model.interactions)
                or any(getattr(p, "use_agnostic_product", False) for p in model.products)
                or not getattr(model.radial_embedding, "apply_cutoff", True)):
            return "MACE capture supports the MP-0/MPA-0 ScaleShiftMACE architectures; this checkpoint needs eager inference"
        if self.variant != "plain":
            sh, radial = model.spherical_harmonics, model.radial_embedding
            if not (sh.normalize and sh.normalization == "component" and sh._lmax == 3
                    and sh._is_range_lmax and type(radial.bessel_fn).__name__ == "BesselBasis"
                    and radial.bessel_fn.bessel_weights.numel() <= 16
                    and (not hasattr(radial, "distance_transform")
                         or type(radial.distance_transform).__name__ == "AgnesiTransform")):
                return "MACE fused edges require lmax=3, Bessel basis and optional Agnesi transform; use variant='plain' or eager"
        return None

    def _predict_eager(self, atoms, properties):
        from fastmd._vendor.mace_opt import tensor_batch as tb
        info = self.info
        inputs = tb.inputs_from_atoms(atoms, info)
        with tb.default_dtype(info.dtype):
            result = tb.mace_forward(self.model, inputs,
                                     compute_stress=self.compute_stress or "stress" in properties)
        return self._numpy_results(result)

    @staticmethod
    def _numpy_results(result):
        output = {"energy": float(result["energy"].detach().reshape(-1)[0].cpu()),
                  "forces": result["forces"].detach().cpu().numpy().copy()}
        if result.get("stress") is not None:
            output["stress"] = result["stress"].detach().reshape(3, 3).cpu().numpy().copy()
        return output

    def _predict_graph(self, atoms, properties):
        from .mace_graph import MACEGraphRunner
        if self.runner is None:
            self.runner = MACEGraphRunner(self, atoms)
        return self._numpy_results(self.runner.run(
            atoms.positions, cell=atoms.cell.array,
            compute_stress=self.compute_stress or "stress" in properties))

    def clear_graphs(self):
        self.runner = None

    def stats(self):
        return {**super().stats(), "variant": self.variant, "dtype": str(self.info.dtype),
                "compute_stress": self.compute_stress,
                "cueq": self.info.enable_cueq, "head": self.info.heads[self.info.head_index],
                "cache": self.runner.stats() if self.runner else {}}
