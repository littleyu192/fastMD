"""torch-sim ModelInterface wrapper for MatRIS."""

from __future__ import annotations

import torch
from torch import Tensor
from pymatgen.io.ase import AseAtomsAdaptor

from ..graph import RadiusGraph, atoms_to_graph_gpu
from ..graph.gpu_graph_builder import op_available
from ..model.model import MatRIS
from ..config import InferenceConfig
from ._config import (
    apply_checkpoint, configure_entrypoint, config_summary, scoped_call,
)

try:
    from torch_sim import units as _ts_units
    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState
except ImportError as exc:
    raise ImportError(
        "torch-sim is not installed. Please install torch-sim to use MatRISModel."
    ) from exc

_GPa = _ts_units.MetalUnits.pressure * 10000


class MatRISModel(ModelInterface):
    """torch-sim compatible wrapper for MatRIS.

    Converts torch-sim SimState inputs to RadiusGraph format, runs the
    MatRIS model, and returns outputs in torch-sim format.

    Output units:
        energy : eV  (total, per system)
        forces : eV/Å
        stress : eV/Å³

    Args:
        model: A :class:`~matris.model.MatRIS` instance, or the name of a
            pretrained checkpoint (e.g. ``"matris_10m_oam"``).
        compute_stress: Whether to compute the stress tensor.
        compute_magmom: Whether to compute per-atom magnetic moments.
        device: Device to run inference on. Defaults to the device of the
            loaded model parameters.
        dtype: Floating-point dtype exposed to torch-sim. Defaults to
            ``torch.float32``.
        enable_compile: Whether to compile interaction blocks.
        config: Unified inference configuration. Omission uses eager execution,
            strict isolated-atom rejection, and checkpointing off.

    Example::

        from fastmd._vendor.matris.applications.torchsim_interface import MatRISModel

        model = MatRISModel(
            "matris_10m_oam",
            compute_stress=True,
            device="cuda",
            enable_compile=True,
        )
        results = model(sim_state)   # returns {"energy", "forces", "stress"}
    """

    @configure_entrypoint("torchsim")
    def __init__(
        self,
        model: MatRIS | str = "matris_10m_oam",
        compute_stress: bool = True,
        compute_magmom: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        enable_compile: bool = True,
        config: InferenceConfig | None = None,
    ) -> None:
        """Initialize MatRIS torch-sim model.

        Args:
            enable_compile: Whether to compile interaction blocks.
            config: Unified inference settings. Only execution='eager' is
                supported here; enable_compile remains an independent option.
        """
        super().__init__()
        enable_checkpoint = self._inference_config.checkpoint_enabled
        if isinstance(model, str):
            self.model = MatRIS.load(
                model_name=model,
                device=str(device) if device is not None else None,
                enable_compile=enable_compile,
                enable_checkpoint=enable_checkpoint,
            )
        else:
            self.model = model
            self.model.enable_compile = enable_compile
            for block in self.model.interaction_block:
                block.enable_compile = enable_compile
                block.enable_checkpoint = enable_checkpoint

        apply_checkpoint(self.model, enable_checkpoint)
        self.handle_isolated_atoms = self._inference_config.handle_isolated_atoms

        self._compute_stress = compute_stress
        self._compute_magmom = compute_magmom
        self._dtype = dtype
        if device is not None:
            self._device = torch.device(device)
        else:
            try:
                self._device = next(self.model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")

        self.model = self.model.to(self._device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def config_summary(self) -> dict:
        """Requested/resolved configuration and observed execution state."""
        return config_summary(self)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @property
    def compute_forces(self) -> bool:
        return True

    @property
    def compute_magmom(self) -> bool:
        return self._compute_magmom

    @property
    def memory_scales_with(self) -> str:
        return "n_atoms_x_density"

    @scoped_call
    def _state_to_graphs(self, state: SimState) -> list[RadiusGraph]:
        """Convert a SimState (or StateDict) to a list of RadiusGraph objects.

        The cell matrix is treated as row-major (each row is a lattice vector),
        matching the ASE / pymatgen convention.
        """

        structures = state.to_structures()
        if self._device.type == "cuda" and op_available:
            atoms = [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
            graphs = atoms_to_graph_gpu(
                atoms,
                self.model.graph_converter.atom_graph_cutoff,
                self.model.graph_converter.line_graph_cutoff,
                device=str(self._device),
                check_isolated_atoms=(
                    not self.handle_isolated_atoms
                    or self.model.reference_energy is None
                ),
            )
            return graphs if isinstance(graphs, list) else [graphs]
        return [
            self.model.graph_converter(
                structure,
                check_isolated_atoms=(
                    not self.handle_isolated_atoms
                    or self.model.reference_energy is None
                ),
            ).to(str(self._device))
            for structure in structures
        ]

    @scoped_call
    def forward(self, state: SimState, **kwargs) -> dict[str, Tensor]:
        """Compute energy, forces, and optionally stress from a SimState.

        Args:
            state: A torch-sim ``SimState`` or a ``StateDict`` with keys
                ``positions``, ``cell``, ``atomic_numbers``, and
                ``system_idx``.

        Returns:
            Dictionary with:
                ``"energy"``  : Tensor of shape ``[n_systems]`` — total energy in eV.
                ``"forces"``  : Tensor of shape ``[n_atoms, 3]`` — forces in eV/Å.
                ``"stress"``  : Tensor of shape ``[n_systems, 3, 3]`` — stress in
                    eV/Å³ (only present when ``compute_stress=True``).
                ``"magmoms"`` : Tensor of shape ``[n_atoms]`` — per-atom magnetic
                    moments in μB (only present when ``compute_magmom=True``).
        """

        graphs = self._state_to_graphs(state)

        task = "ef"
        if self._compute_stress:
            task += "s"
        if self._compute_magmom:
            task += "m"

        result = self.model(
            graphs, task=task, is_training=False,
            handle_isolated_atoms=self.handle_isolated_atoms,
        )
        energy: Tensor = result["e"]
        if self.model.is_intensive:
            energy = energy * result["atoms_per_graph"].to(dtype=energy.dtype)

        output: dict[str, Tensor] = {
            "energy": energy,
            "forces": torch.concat(result["f"], dim=0),
        }

        if self._compute_stress:
            stress = result["s"]
            if isinstance(stress, Tensor):
                output["stress"] = stress * _GPa
            else:
                output["stress"] = torch.stack(list(stress), dim=0) * _GPa

        if self._compute_magmom:
            output["magmoms"] = torch.concat(result["m"], dim=0)

        return output
