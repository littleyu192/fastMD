from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes, all_properties
import numpy as np
import torch

from ..model.model import MatRIS
from ..config import InferenceConfig
from ._config import apply_checkpoint, configure_entrypoint, config_summary, scoped_call

from pymatgen.io.ase import AseAtomsAdaptor
from ..graph import RadiusGraph

from ase.optimize import (
    BFGS, BFGSLineSearch, 
    FIRE, LBFGS, 
    LBFGSLineSearch, MDMin
)

names = [
    "BFGS", "BFGSLineSearch", 
    "FIRE", "LBFGS", 
    "LBFGSLineSearch", "MDMin"
]

OPTIMIZERS = {name: globals()[name] for name in names}

class MatRISCalculator(Calculator):
    """MatRIS Calculator for ASE applications."""
    
    implemented_properties = ("energy", "forces", "stress", "magmoms")  # type: ignore
    
    @configure_entrypoint("calculator")
    def __init__(
        self,
        model_path: str = None,
        model: MatRIS | str = "matris_10m_oam",
        task: str = "efs",
        device: str = "cpu",
        enable_compile: bool = False,
        config: InferenceConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            model (MatRIS): Instance of a MatRIS model. If set to None, the default MatRIS is loaded.
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            device (str): The device to be used for predictions,
            enable_compile (bool): Whether to compile interaction blocks.
            config: Unified execution, optimization, checkpoint, isolated-atom,
                and capacity settings. Defaults to eager execution, reference
                isolated-atom handling, and checkpointing off. These settings
                also apply when passing an already constructed MatRIS model.
            stress_unit (float): the conversion factor to convert GPa(MatRIS default) to eV/A^3.
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)
        self.task=task
        self.device = device
        self.handle_isolated_atoms = self._inference_config.handle_isolated_atoms
        enable_checkpoint = self._inference_config.checkpoint_enabled
        if isinstance(model, MatRIS):
            if model_path is not None:
                raise ValueError("Pass either model=MatRIS(...) or model_path, not both")
            self.model = model.to(self.device)
            apply_checkpoint(self.model, enable_checkpoint)
        else:
            self.model = MatRIS.load(
                model_path=model_path,
                model_name=model,
                device=self.device,
                enable_compile=enable_compile,
                enable_checkpoint=enable_checkpoint,
            )
        # Inference: forces/stress are autograd w.r.t. coords/strains, never params.
        # Freezing params avoids computing unused parameter grads in the backward
        # (cuts backward work and enables the dx-only fast path of fused LN+act).
        for _p in self.model.parameters():
            _p.requires_grad_(False)
        # GPU graphs retain isolated nodes and mask their learned readouts on
        # device, avoiding the legacy per-step pymatgen neighbor-list scan.
        self._gpu_graph = (
            getattr(self.model.graph_converter, "algorithm", "legacy") == "gpu"
            and torch.cuda.is_available()
        )

        # Optional bucketed CUDA-graph replay of the (sync-free) forward+backward.
        self._graph_runner = None
        if (
            self._inference_config.execution == "model_graph"
            and self._gpu_graph
            and torch.cuda.is_available()
        ):
            from .cuda_graph import BucketedGraphRunner
            capacity = self._inference_config.capacity
            self._graph_runner = BucketedGraphRunner(
                self.model, task=self.task,
                u_step=capacity.u_step, t_step=capacity.t_step,
                n_dummy=capacity.n_dummy, min_pad_u=capacity.min_pad_u,
                warmup=capacity.warmup,
                enable_model_fusions=self._inference_config.optimization_profile != "generic",
                config=self._inference_config,
            )

        self.stress_unit = units.GPa
        key = ["atoms_per_graph", "ref_energy"]
        for t in task:
            key.append(t)
        self.key = set(key)

    def config_summary(self) -> dict:
        """Requested/resolved configuration and observed execution state."""
        return config_summary(self)

    def _get_isolated_atom_indices(self, structure) -> np.ndarray:
        center_index, _, _, _ = structure.get_neighbor_list(
            r=self.model.graph_converter.atom_graph_cutoff,
            sites=structure.sites,
            numerical_tol=1e-8,
        )
        return np.setdiff1d(
            np.arange(len(structure), dtype=int),
            np.unique(center_index),
            assume_unique=True,
        )

    def _get_atomic_reference_energy(self, atomic_numbers: np.ndarray) -> float:
        if len(atomic_numbers) == 0:
            return 0.0
        if self.model.reference_energy is None:
            raise ValueError(
                "Cannot add isolated atomic energy because this model has no "
                "reference_energy table."
            )

        atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        max_num_elements = self.model.reference_energy.fc.weight.shape[1]
        if np.any(atomic_numbers < 1) or np.any(atomic_numbers > max_num_elements):
            raise ValueError(
                f"Atomic numbers must be in [1, {max_num_elements}] to use "
                "reference atomic energies."
            )

        ref_weights = (
            self.model.reference_energy.fc.weight.detach().cpu().numpy()[0]
        )
        return float(ref_weights[atomic_numbers - 1].sum())

    def _as_numpy_prediction(self, value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        elif hasattr(value, "ndim") and value.ndim > 0:
            value = value[0]

        if hasattr(value, "cpu"):
            return value.cpu().detach().numpy()
        return np.array(value)

    @scoped_call
    def _predict_structure(self, structure, atoms=None):
        # atoms!=None takes the GPU fast path (build graph straight from ASE atoms,
        # skipping the ASE<->pymatgen round trip).
        if self._graph_runner is not None and atoms is not None:
            # Bucketed CUDA-graph replay of forward+backward (padded, dummy-sink
            # masked -> exact on the n_real real atoms).
            g = self.model.graph_converter(
                None,
                atoms=atoms,
                check_isolated_atoms=(
                    not self.handle_isolated_atoms
                    or self.model.reference_energy is None
                ),
            ).to(self.device)
            out, n_real = self._graph_runner.run(g)
            result = {}
            if "e" in self.key:
                result["e"] = float(out["e"][0])
            if "ref_energy" in self.key:
                re = out["ref_energy"]
                result["ref_energy"] = float(re[0]) if torch.is_tensor(re) else float(re)
            if "f" in self.key and "f" in out:
                result["f"] = out["f"][0][:n_real].detach().cpu().numpy()
            if "s" in self.key and "s" in out:
                result["s"] = out["s"][0].detach().cpu().numpy()
            if "m" in self.key and "m" in out:
                result["m"] = out["m"][0][:n_real].detach().cpu().numpy()
            return result

        graph = self.model.graph_converter(
            structure,
            atoms=atoms,
            check_isolated_atoms=(
                not self.handle_isolated_atoms
                or self.model.reference_energy is None
            ),
        ).to(self.device)
        graphs = [graph] if isinstance(graph, RadiusGraph) else graph

        model_prediction = self.model(
            graphs,
            task = self.task,
            is_training = False,
            handle_isolated_atoms=self.handle_isolated_atoms,
        )

        return {
            key: self._as_numpy_prediction(model_prediction[key])
            for key in self.key & set(model_prediction.keys())
        }
     
    @scoped_call
    def precapture(self, atoms, n_steps: int = 120, temperature: float = 300.0,
                   timestep: float = 1.0) -> None:
        """Warm the CUDA-graph bucket cache up front by driving a throwaway NVT
        trajectory on a COPY of `atoms`. This covers the one-time thermalization
        drift plus the equilibrium fluctuation of the edge/triplet counts, so the
        production run replays with no mid-run capture spikes. No-op unless
        execution='model_graph' is active."""
        if self._graph_runner is None:
            return
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        probe = atoms.copy()
        MaxwellBoltzmannDistribution(probe, temperature_K=temperature, force_temp=True)
        probe.calc = self
        dyn = NVTBerendsen(probe, timestep=timestep * units.fs,
                           temperature_K=temperature, taut=100 * units.fs)
        dyn.run(n_steps)
        # proactive ±1 neighbor buckets around the final state
        g = self.model.graph_converter(
            None,
            atoms=probe,
            check_isolated_atoms=(
                not self.handle_isolated_atoms
                or self.model.reference_energy is None
            ),
        ).to(self.device)
        self._graph_runner.precapture(g, proactive=True)
        print(f"CUDA-graph precapture: {self._graph_runner.captures} graphs / "
              f"{len(self._graph_runner.cache)} buckets")

    @scoped_call
    def calculate(
        self,
        atoms: Atoms,
        properties: list,
        system_changes: list,
    ) -> None:
        
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        pbc = atoms.get_pbc()
        if (not pbc[0]) or (not pbc[1]) or (not pbc[2]):
            pos = atoms.get_positions()
            cell = np.array(atoms.get_cell())
            
            pbc_x = pbc[0]
            pbc_y = pbc[1]
            pbc_z = pbc[2]
            identity = np.identity(3, dtype=float)
            max_positions = np.max(np.absolute(pos)) + 1
            
            cutoff = self.model.config["pairwise_cutoff"]
            expand = max(5, self.model.config["num_layers"])

            # Extend cell in non-periodic directions
            if not pbc_x:
                cell[0, :] = max_positions * expand * cutoff * identity[0, :]
            if not pbc_y:
                cell[1, :] = max_positions * expand * cutoff * identity[1, :]
            if not pbc_z:
                cell[2, :] = max_positions * expand * cutoff * identity[2, :]
            
            # update
            atoms.set_cell(cell, scale_atoms=False)
        
        total_atoms = len(atoms)
        active_indices = np.arange(total_atoms, dtype=int)
        isolated_indices = np.array([], dtype=int)
        isolated_ref_energy = 0.0

        use_gpu_fast_path = self._gpu_graph
        if use_gpu_fast_path:
            # Fast path: build the graph straight from ASE atoms (no ASE->pymatgen
            # ->ASE round trip). Isolated nodes stay in place; the model masks
            # learned readouts on device and retains their reference energies.
            # Without a reference table the converter rejects isolated atoms
            # strictly; connected systems still use the requested CUDA replay.
            structure = None
            pred_structure = None
            pred_atoms = atoms
        else:
            structure = AseAtomsAdaptor.get_structure(atoms)
            pred_structure = structure
            pred_atoms = None
            if self.handle_isolated_atoms:
                isolated_indices = self._get_isolated_atom_indices(structure)
                if len(isolated_indices) != 0:
                    active_indices = np.setdiff1d(
                        active_indices, isolated_indices, assume_unique=True
                    )
                    isolated_ref_energy = self._get_atomic_reference_energy(
                        atoms.get_atomic_numbers()[isolated_indices]
                    )
                    pred_structure = (
                        AseAtomsAdaptor.get_structure(atoms[active_indices])
                        if len(active_indices) != 0
                        else None
                    )

        model_predictions = {}
        forces = None
        stress = None
        magmoms = None
        ref_energy = isolated_ref_energy
        energy = isolated_ref_energy

        do_predict = (pred_structure is not None) or (pred_atoms is not None)
        if do_predict:
            model_predictions = self._predict_structure(pred_structure, atoms=pred_atoms)

            n_atoms = 1 if not self.model.is_intensive else len(active_indices)
            ref_energy += float(model_predictions["ref_energy"] * n_atoms)
            energy += float(model_predictions["e"] * n_atoms)
            forces = model_predictions.get("f", None)
            stress = model_predictions.get("s", None)
            magmoms = model_predictions.get("m", None)

            if len(isolated_indices) != 0:
                if forces is not None:
                    full_forces = np.zeros((total_atoms, 3), dtype=forces.dtype)
                    full_forces[active_indices] = forces
                    forces = full_forces
                if magmoms is not None:
                    full_magmoms = np.zeros(total_atoms, dtype=magmoms.dtype)
                    full_magmoms[active_indices] = magmoms
                    magmoms = full_magmoms
        else:
            if "f" in self.task:
                forces = np.zeros((total_atoms, 3), dtype=float)
            if "s" in self.task:
                stress = np.zeros((3, 3), dtype=float)
            if "m" in self.task:
                magmoms = np.zeros(total_atoms, dtype=float)

        if stress is not None:
            stress = stress * self.stress_unit

        self.results.update(
            ref_energy=ref_energy,
            energy=energy,  # Total Energy
            forces=forces,
            # Stress: GPa -> eV/A^3
            stress=stress,
            magmoms=magmoms,
        )


class TrajectoryObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeasObserver:
    # ref: https://github.com/CederGroupHub/chgnet

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)
                      
