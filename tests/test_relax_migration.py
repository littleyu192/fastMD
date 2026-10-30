"""The MD relaxation API in FastMD, independent of optional models and CUDA."""
from __future__ import annotations

import importlib
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.io.ase import AseAtomsAdaptor

from fastmd import InferenceConfig, StructOptimizer
from fastmd.engine import saved_recompute
from fastmd.interfaces.calculator import TrajectoryObserver


class Harmonic(Calculator):
    implemented_properties = ["energy", "forces", "stress", "magmoms"]
    device = "cpu"
    task = "efs"

    def __init__(self):
        super().__init__()
        self.closed = False

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.positions
        volume = self.atoms.get_volume()
        self.results = dict(energy=0.5 * float((positions**2).sum()) + 0.001 * volume,
                            forces=-positions.copy(), stress=np.array([0.001]*3 + [0.0]*3),
                            magmoms=np.zeros(len(self.atoms)))

    def config_summary(self):
        return {"effective": {"execution": "eager"}}

    def close(self):
        self.closed = True


class RelaxationMigrationTest(unittest.TestCase):
    @staticmethod
    def atoms():
        return Atoms("H2", positions=[[0.1, 0.2, 0.3], [0.8, 0.1, 0.0]], cell=[8]*3, pbc=True)

    def optimizer(self, **kwargs):
        calculator = Harmonic()
        with patch("fastmd.interfaces.relax.MLIPCalculator", return_value=calculator):
            optimizer = StructOptimizer(**kwargs)
        self.addCleanup(optimizer.close)
        return optimizer

    def test_source_api_matches_ase_positions_cells_and_observer(self):
        for cell in (False, True):
            for interval in (0, 1, 3):
                with self.subTest(cell=cell, interval=interval):
                    expected, actual = self.atoms(), self.atoms()
                    expected.calc = Harmonic()
                    observer = TrajectoryObserver(expected)
                    target = FrechetCellFilter(expected) if cell else expected
                    with FIRE(target, logfile=None) as reference:
                        reference.attach(observer, interval=interval)
                        reference.run(fmax=1e-12, steps=4)
                        observer()  # The MD API appends the terminal frame, even if already recorded.
                    optimizer = self.optimizer(engine="ase")
                    result = optimizer.relax(actual, relax_cell=cell, steps=4, fmax=1e-12,
                                             loginterval=interval, verbose=False, assign_magmoms=False,
                                             logfile=None)
                    self.assertEqual(set(result), {"final_structure", "trajectory"})
                    self.assertIs(actual.calc, optimizer.calculator)
                    self.assertEqual(optimizer.last_relaxation["path"], "ase")
                    np.testing.assert_array_equal(actual.positions, expected.positions)
                    np.testing.assert_array_equal(actual.cell.array, expected.cell.array)
                    np.testing.assert_allclose(result["final_structure"].cart_coords, actual.positions, atol=1e-14)
                    for name in ("energies", "forces", "stresses", "atom_positions", "cells"):
                        np.testing.assert_equal(getattr(result["trajectory"], name), getattr(observer, name))

    def test_list_cpu_fallback_preserves_order_and_writes_per_structure_records(self):
        atoms = [self.atoms(), self.atoms()]
        atoms[1].positions *= 0.5
        optimizer = self.optimizer(engine="auto")
        with tempfile.TemporaryDirectory() as directory:
            results = optimizer.relax(atoms, steps=2, relax_cell=False, verbose=False,
                                      assign_magmoms=False, save_path=f"{directory}/relax-{{index}}.pkl")
            self.assertEqual(len(results), 2)
            self.assertEqual(optimizer.last_relaxation["path"], "ase")
            self.assertEqual(optimizer.last_relaxation["gpu"], 0)
            self.assertIn("CPU".lower(), optimizer.last_relaxation["reason"].lower())
            for index, result in enumerate(results):
                with open(Path(directory) / f"relax-{index}.pkl", "rb") as handle:
                    saved = pickle.load(handle)
                np.testing.assert_array_equal(saved["atom_positions"][-1], atoms[index].positions)
                np.testing.assert_array_equal(saved["energy"], result["trajectory"].energies)

    def test_pymatgen_input_and_zero_steps(self):
        structure = AseAtomsAdaptor.get_structure(self.atoms())
        before = structure.copy()
        optimizer = self.optimizer()
        result = optimizer.relax(structure, steps=0, relax_cell=False, verbose=False, assign_magmoms=False)
        self.assertEqual(structure, before)
        np.testing.assert_allclose(result["final_structure"].cart_coords, before.cart_coords, atol=1e-14)
        self.assertEqual(len(result["trajectory"].energies), 2)

    def test_explicit_gpu_never_silently_uses_cpu(self):
        optimizer = self.optimizer(engine="gpu")
        with self.assertRaisesRegex(ValueError, "engine='gpu' cannot relax.*CUDA"):
            optimizer.relax(self.atoms(), steps=0, verbose=False)
        with self.assertRaisesRegex(ValueError, "engine must be"):
            optimizer.relax(self.atoms(), engine="invalid")
        with self.assertRaisesRegex(ValueError, "must name each structure"):
            optimizer.relax([self.atoms()], save_path="one.pkl")
        self.assertEqual(optimizer.relax([], engine="auto"), [])

    def test_gpu_failure_falls_back_from_initial_geometry(self):
        optimizer = self.optimizer(engine="auto")
        initial = self.atoms()
        with patch.object(optimizer, "_gpu_path_reason", return_value=(None, object())), \
                patch.object(optimizer, "_relax_on_gpu", return_value=(
                    [None], ["the GPU relaxation cannot start: test failure"],
                    {"construction": 0.0, "relaxation": 0.0})):
            actual = optimizer.relax(initial, steps=2, relax_cell=False, verbose=False, assign_magmoms=False)
        reference = optimizer.relax(self.atoms(), engine="ase", steps=2, relax_cell=False,
                                    verbose=False, assign_magmoms=False)
        np.testing.assert_array_equal(actual["trajectory"].atom_positions, reference["trajectory"].atom_positions)

    def test_uncompiled_scope_restores_flags_and_wrapped_modules_after_failure(self):
        optimizer = self.optimizer()
        original = SimpleNamespace(_compiled_block=True)
        wrapper = SimpleNamespace(_orig_mod=original)
        block = SimpleNamespace(enable_compile=True, _is_compiled=True, attn_block_atom_graph=wrapper)
        optimizer.calculator.adapter = SimpleNamespace(module=SimpleNamespace(interaction_block=[block]))
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with optimizer._uncompiled():
                self.assertFalse(block.enable_compile)
                self.assertFalse(block._is_compiled)
                self.assertFalse(original._compiled_block)
                self.assertIs(block.attn_block_atom_graph, original)
                raise RuntimeError("test failure")
        self.assertTrue(block.enable_compile and block._is_compiled and original._compiled_block)
        self.assertIs(block.attn_block_atom_graph, wrapper)

    def test_removed_public_paths_and_preserved_md_exports(self):
        import fastmd
        self.assertIs(importlib.import_module("fastmd.applications.relax").StructOptimizer, StructOptimizer)
        for name in ("relax", "RelaxationConfig", "RelaxationResult", "make_relaxation_calculator",
                     "relax_batch", "BatchedMLIPModel"):
            self.assertFalse(hasattr(fastmd, name), name)
        root = Path(fastmd.__file__).parent
        for relative in ("relaxation.py", "relax_cli.py", "interfaces/torchsim_batch.py",
                         "interfaces/torchsim_native.py", "interfaces/torchsim_frechet.py"):
            self.assertFalse((root / relative).exists(), relative)
        for name in ("MLIPCalculator", "MolecularDynamics", "GPUResidentMolecularDynamics",
                     "ReplicaMolecularDynamics", "MultiSystemMolecularDynamics"):
            self.assertIsNotNone(getattr(fastmd, name))
        from fastmd.execution.whole_step_cuda_graph import BatchedWholeStepGraphRunner
        self.assertIsNotNone(BatchedWholeStepGraphRunner)
        self.assertTrue((root / "interfaces/torchsim_interface.py").is_file())

    def test_probe_recompute_state_does_not_replace_running_md_state(self):
        from fastmd.config import resolve_config
        model, config = object(), resolve_config(InferenceConfig(report=False))
        key, previous = (id(model), config.fingerprint), object()
        saved_recompute._SINGLE[key] = previous
        self.addCleanup(saved_recompute._SINGLE.pop, key, None)
        with self.assertRaisesRegex(RuntimeError, "measurement failed"):
            with saved_recompute.detached_auto(model, config) as reset:
                self.assertNotIn(key, saved_recompute._SINGLE)
                saved_recompute._SINGLE[key] = object()
                reset()
                self.assertNotIn(key, saved_recompute._SINGLE)
                saved_recompute._SINGLE[key] = object()
                raise RuntimeError("measurement failed")
        self.assertIs(saved_recompute._SINGLE[key], previous)


if __name__ == "__main__":
    unittest.main()
