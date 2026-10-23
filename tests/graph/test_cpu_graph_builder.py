"""CPU graph geometry and public calculator derivatives, without model downloads."""
from __future__ import annotations

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from fastmd import MLIPCalculator
from fastmd.adapters.protocol import CaptureContract, PaddingRule, TopologySpec
from fastmd.config import InferenceConfig
from fastmd.graph.cpu_graph_builder import atoms_to_edges_cpu


class PairEnergyAdapter:
    """Smooth analytic pair potential plus species offsets, differentiated by FastMD."""

    name = "cpu_test_pair"
    compiled = False

    def __init__(self, dtype=torch.float64, geometry_dtype=None):
        self.module = torch.nn.Module()
        self.module.register_parameter("strength", torch.nn.Parameter(torch.tensor(0.25, dtype=dtype)))
        if geometry_dtype is not None:
            self.geometry_dtype = geometry_dtype
        self.last_batch = None

    def topology_spec(self):
        return TopologySpec(cutoff=2.6, num_interactions=1,
                            model_dtype="float64" if self.module.strength.dtype == torch.float64 else "float32")

    def padding_rule(self):
        return PaddingRule()

    def capture_contract(self):
        return CaptureContract(sync_free=True, exact_cutoff=True, force_mode="engine_autograd",
                               stress_unit="eV/A^3")

    def isolated_atom_policy(self):
        return "model"

    def reference_energies(self):
        return None

    def prepare(self, batch):
        if any(int(z) not in (1, 6, 8) for z in batch.species):
            raise ValueError("Unsupported test species")
        self.last_batch = batch
        return batch

    def node_energy(self, batch, positions, cell):
        centre, neighbour = batch.edge_index.long().T
        vectors = positions[neighbour] + batch.edge_image @ cell - positions[centre]
        radii = torch.linalg.vector_norm(vectors, dim=1)
        energies = 0.25 * self.module.strength * torch.clamp(2.6 - radii, min=0) ** 4
        # The zero connections define the derivative also for an isolated atom.
        base = 0.1 * batch.species.to(positions.dtype) + positions.sum(dim=1) * 0 + cell.sum() * 0
        return base.index_add(0, centre, energies)


def periodic_atoms():
    return Atoms("HCO", scaled_positions=[[0.02, 0.09, 0.12], [0.93, 0.16, 0.19], [0.38, 0.31, 0.34]],
                 cell=[[4.6, 0.0, 0.0], [0.7, 4.8, 0.0], [0.4, 0.6, 5.1]], pbc=True)


def analytic_efs(atoms):
    """Independent analytic derivatives of the directed pair sum."""
    centre, neighbour, vectors, distances = neighbor_list("ijDd", atoms, 2.6, self_interaction=False)
    gap = 2.6 - distances
    energy = 0.1 * atoms.numbers.sum() + np.sum(0.25 * 0.25 * gap**4)
    radial_derivative = -0.25 * gap**3
    gradients = (radial_derivative / distances)[:, None] * vectors
    forces = np.zeros_like(atoms.positions)
    np.add.at(forces, centre, gradients)
    np.add.at(forces, neighbour, -gradients)
    stress = np.einsum("ei,ej->ij", gradients, vectors) / atoms.get_volume()
    return energy, forces, stress


class CPUBuilderTests(unittest.TestCase):
    def test_triclinic_images_match_brute_force_and_do_not_mutate_atoms(self):
        atoms = periodic_atoms()
        atoms.positions[1] += atoms.cell[0] - 2 * atoms.cell[1]
        original = atoms.copy()
        batch = atoms_to_edges_cpu(atoms, 2.6, positions_dtype=torch.float64)
        expected = []
        positions, cell = batch.positions.numpy(), batch.cell.numpy()
        for i, j, image in product(range(len(atoms)), range(len(atoms)), product(range(-2, 3), repeat=3)):
            if i == j and image == (0, 0, 0):
                continue
            if np.linalg.norm(positions[j] + np.array(image) @ cell - positions[i]) < 2.6:
                expected.append((i, j, *image))
        actual = [tuple(row) for row in np.column_stack((batch.edge_index, batch.edge_image)).astype(int)]
        self.assertEqual(actual, sorted(expected))
        self.assertEqual(set(actual), {(j, i, -x, -y, -z) for i, j, x, y, z in actual})
        np.testing.assert_array_equal(atoms.positions, original.positions)
        np.testing.assert_array_equal(atoms.cell, original.cell)
        np.testing.assert_array_equal(batch.species, atoms.numbers)
        self.assertEqual(batch.positions.dtype, torch.float64)
        self.assertIsNone(batch.triplets)
        self.assertIsNone(batch.pair_d2u)
        self.assertIsNone(batch.padding)

    def test_partial_periodicity_cell_free_molecule_and_self_images(self):
        slab = Atoms("H2", positions=[[0.1, 0, 0], [3.9, 0, 0]], cell=[4, 0, 0], pbc=[True, False, False])
        batch = atoms_to_edges_cpu(slab, 0.5)
        np.testing.assert_array_equal(batch.edge_image[:, 1:], 0)
        np.testing.assert_array_equal(batch.edge_image[:, 0], [-1, 1])
        molecule = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
        batch = atoms_to_edges_cpu(molecule, 1.0)
        np.testing.assert_array_equal(batch.cell, np.zeros((3, 3)))
        np.testing.assert_array_equal(batch.edge_image, 0)
        one = Atoms("H", positions=[[0, 0, 0]], cell=[1, 4, 4], pbc=True)
        batch = atoms_to_edges_cpu(one, 1.1)
        self.assertEqual(batch.n_edges.item(), 2)
        np.testing.assert_array_equal(batch.edge_index, [[0, 0], [0, 0]])
        np.testing.assert_array_equal(batch.edge_image, [[-1, 0, 0], [1, 0, 0]])

    def test_isolated_and_invalid_geometry(self):
        atom = Atoms("O", positions=[[0, 0, 0]])
        batch = atoms_to_edges_cpu(atom, 2, check_isolated_atoms=False)
        self.assertEqual(batch.n_real, 1)
        self.assertEqual(tuple(batch.edge_index.shape), (0, 2))
        self.assertEqual(tuple(batch.edge_image.shape), (0, 3))
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            atoms_to_edges_cpu(atom, 2)
        for cutoff in (0, -1, np.inf, np.nan):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, "cutoff"):
                atoms_to_edges_cpu(atom, cutoff)
        with self.assertRaisesRegex(ValueError, "at least one"):
            atoms_to_edges_cpu(Atoms(), 2)
        atom.pbc = True
        with self.assertRaisesRegex(ValueError, "cell vector"):
            atoms_to_edges_cpu(atom, 2)
        atom.pbc = False
        atom.positions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            atoms_to_edges_cpu(atom, 2)


class CPUCalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def calculator(self, adapter=None, **kwargs):
        calculator = MLIPCalculator(model=adapter or PairEnergyAdapter(), device="cpu", **kwargs)
        self.addCleanup(calculator.close)
        return calculator

    def assert_analytic(self, atoms):
        expected = analytic_efs(atoms)
        np.testing.assert_allclose(atoms.get_potential_energy(), expected[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(atoms.get_forces(), expected[1], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(atoms.get_stress(voigt=False), expected[2], rtol=1e-11, atol=1e-12)

    def test_public_efs_matches_analytic_derivatives_and_cpu_plan(self):
        atoms = periodic_atoms()
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.init", side_effect=AssertionError("CPU path must not initialize CUDA")):
            atoms.calc = self.calculator(config=InferenceConfig(topology_reuse="skin"))
            self.assert_analytic(atoms)
        plan = atoms.calc.plan_record()
        self.assertEqual(plan.decision("execution").chosen, "eager")
        self.assertEqual(plan.decision("neighbor_list").chosen, "host")
        self.assertEqual(plan.decision("neighbor_list").evidence["builder"], "ase.neighbor_list")
        self.assertEqual(plan.decision("neighbor_list").evidence["geometry_dtype"], "torch.float64")
        self.assertEqual(plan.decision("neighbor_list").evidence["pbc"], [True, True, True])
        self.assertEqual(plan.decision("topology_reuse").chosen, "rebuild")
        self.assertEqual(plan.decision("padding").chosen, "none")
        self.assertIn("CPU ASE", atoms.calc._skin_off_reason)

    def test_fp32_weights_can_request_fp64_geometry_without_prior_rounding(self):
        atoms = periodic_atoms()
        atoms.positions[0, 0] += 0.000000001234567
        adapter = PairEnergyAdapter(dtype=torch.float32, geometry_dtype=torch.float64)
        atoms.calc = self.calculator(adapter)
        self.assert_analytic(atoms)
        self.assertEqual(adapter.topology_spec().model_dtype, "float32")
        self.assertEqual(adapter.module.strength.dtype, torch.float32)
        self.assertEqual(adapter.last_batch.positions.dtype, torch.float64)
        np.testing.assert_allclose(adapter.last_batch.positions, atoms.positions, rtol=0, atol=2e-15)
        self.assertGreater(np.max(np.abs(adapter.last_batch.positions.numpy() - atoms.positions.astype("float32"))), 1e-8)

    def test_rebuild_for_boundary_species_cell_and_periodicity_changes(self):
        atoms = periodic_atoms()
        adapter = PairEnergyAdapter()
        atoms.calc = self.calculator(adapter)
        self.assert_analytic(atoms)
        initial = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress(voigt=False))
        old_batch = adapter.last_batch
        atoms.positions[0] += atoms.cell[0]
        self.assert_analytic(atoms)
        self.assertIsNot(old_batch, adapter.last_batch)
        for actual, expected in zip((atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress(voigt=False)), initial):
            np.testing.assert_allclose(actual, expected, atol=2e-14)
        atoms.numbers[0] = 8
        self.assert_analytic(atoms)
        self.assertEqual(adapter.last_batch.species[0], 8)
        atoms.set_cell(atoms.cell.array @ np.diag([1.02, 0.98, 1.01]), scale_atoms=True)
        atoms.pbc = [True, False, True]
        cell_before, positions_before = atoms.cell.array.copy(), atoms.positions.copy()
        self.assert_analytic(atoms)
        np.testing.assert_array_equal(atoms.cell, cell_before)
        np.testing.assert_array_equal(atoms.positions, positions_before)
        np.testing.assert_array_equal(adapter.last_batch.edge_image[:, 1], 0)
        self.assertEqual(atoms.calc.plan_record().decision("neighbor_list").evidence["pbc"], [True, False, True])
        atoms.numbers[0] = 2
        with self.assertRaisesRegex(ValueError, "Unsupported test species"):
            atoms.get_potential_energy()

    def test_forces_and_normal_shear_stress_match_energy_finite_differences(self):
        atoms = periodic_atoms()
        atoms.calc = self.calculator()
        forces, stress, volume = atoms.get_forces(), atoms.get_stress(voigt=False), atoms.get_volume()
        step = 1e-5
        for atom, axis in ((0, 0), (1, 2)):
            energies = []
            for direction in (-1, 1):
                probe = atoms.copy()
                probe.positions[atom, axis] += direction * step
                probe.calc = atoms.calc
                energies.append(probe.get_potential_energy())
            self.assertAlmostEqual(-(energies[1] - energies[0]) / (2 * step), forces[atom, axis], delta=2e-9)
        for i, j in ((0, 0), (0, 1), (1, 2)):
            energies = []
            for direction in (-1, 1):
                deformation = np.eye(3)
                deformation[i, j] += direction * step
                probe = atoms.copy()
                probe.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
                probe.calc = atoms.calc
                energies.append(probe.get_potential_energy())
            self.assertAlmostEqual((energies[1] - energies[0]) / (2 * step * volume), stress[i, j], delta=2e-10)

    def test_isolated_atom_and_cell_free_molecule(self):
        atom = Atoms("O", positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=True)
        atom.calc = self.calculator()
        self.assertAlmostEqual(atom.get_potential_energy(), 0.8)
        np.testing.assert_array_equal(atom.get_forces(), 0)
        np.testing.assert_array_equal(atom.get_stress(), 0)
        atom.calc = self.calculator(config=InferenceConfig(isolated_atoms="error"))
        with self.assertRaisesRegex(ValueError, "isolated atom"):
            atom.get_potential_energy()
        molecule = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
        molecule.calc = self.calculator(task="ef")
        self.assertTrue(np.isfinite(molecule.get_potential_energy()))
        self.assertTrue(np.isfinite(molecule.get_forces()).all())
        np.testing.assert_array_equal(molecule.cell, np.zeros((3, 3)))
        molecule.calc = self.calculator()
        with self.assertRaisesRegex(ValueError, "stress requires a nonzero cell"):
            molecule.get_potential_energy()


if __name__ == "__main__":
    unittest.main()
