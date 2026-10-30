"""Shared slot/replica execution contracts retained by the MD relaxation migration."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from ase import units

from fastmd.adapters.protocol import GraphBatch, NegotiationError, SystemSegments, TopologySpec
from fastmd.execution.replica_batch import graph_builder_options, replica_energy_derivative
from fastmd.execution.system_batch import segment_sums, system_energy_derivative
from fastmd.execution.whole_step_cuda_graph import BatchedWholeStepGraphRunner
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import op_available


class OwnedDerivative:
    name = "owned_test"

    def __init__(self, unit="eV/A^3", energies=(2.0, 3.0)):
        self.unit = unit
        self.energies = energies
        self.calls = []

    def capture_contract(self):
        return SimpleNamespace(stress_unit=self.unit)

    def topology_spec(self):
        return TopologySpec(cutoff=5.0, num_interactions=1, triplet_cutoff=3.0)

    def extensive_energy(self, energies, atoms):
        return energies * atoms

    def evaluate(self, graphs, *, task, is_training, n_real):
        self.calls.append((task, is_training, n_real, torch.is_grad_enabled()))
        # Two sink rows after the occupied/vacant slot rows must not reach the integrator's forces.
        return {"e": torch.tensor(self.energies, requires_grad=True),
                "f": [torch.arange((n_real + 2) * 3, dtype=torch.float64).reshape(n_real + 2, 3)],
                "s": [torch.eye(3, dtype=torch.float64), 2 * torch.eye(3, dtype=torch.float64)]}


class NodeTap:
    geometry_dtype = torch.float64

    def topology_spec(self):
        return TopologySpec(cutoff=5.0, num_interactions=1, triplet_cutoff=3.0, model_dtype="float32")

    def node_energy(self, batch, positions, cell, edge_shifts):
        center, neighbor = batch.edge_index.long().unbind(dim=1)
        bonds = positions[neighbor] + edge_shifts - positions[center]
        values = bonds.square().sum(dim=1) / 4
        return positions.new_zeros(len(positions)).index_add(0, center, values)


class SegmentTap:
    name = "segments_test"

    def system_energy(self, batch, positions, cell, edge_shifts):
        node = NodeTap().node_energy(batch, positions, cell, edge_shifts)
        systems = batch.systems
        return segment_sums(node, systems.row_ptr.diff(), systems.atoms)


def batch():
    positions = torch.tensor([[0.2, 0.3, 0.4], [1.2, 0.4, 0.6], [50, 60, 70],
                              [0.3, 0.5, 0.2], [1.6, 0.8, 0.4], [90, 80, 70]], dtype=torch.float64)
    cells = torch.stack((torch.diag(torch.tensor([4., 5., 6.], dtype=torch.float64)),
                         torch.diag(torch.tensor([7., 8., 9.], dtype=torch.float64))))
    systems = SystemSegments(2, torch.tensor([0, 3, 6]), torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
                             cells, torch.tensor([2, 2]))
    return GraphBatch(positions, torch.zeros_like(positions), cells[0], torch.ones(6, dtype=torch.int32), 6,
                      torch.tensor([[0, 1], [1, 0], [3, 4], [4, 3]], dtype=torch.int32),
                      torch.tensor([[1., 0., 0.], [-1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]],
                                   dtype=torch.float64),
                      torch.tensor([4]), 5.0, None, systems=systems)


class SystemBatchContractsTest(unittest.TestCase):
    def test_model_owned_slots_convert_units_crop_sink_forces_and_enable_grad(self):
        graph = batch()
        graph.systems = SystemSegments(2, graph.systems.row_ptr, graph.systems.system_of_row,
                                       graph.systems.cells, torch.tensor([1, 2]))
        for unit, scale in (("eV/A^3", 1.0), ("GPa", units.GPa)):
            with self.subTest(unit=unit):
                adapter = OwnedDerivative(unit)
                with torch.no_grad():
                    energies, forces, stresses = system_energy_derivative(adapter, graph, graph, stress=True)
                self.assertEqual(adapter.calls, [("efs", False, 6, True)])
                torch.testing.assert_close(energies, torch.tensor([2., 6.], dtype=torch.float64))
                self.assertFalse(energies.requires_grad)
                self.assertEqual(forces.shape, (6, 3))
                torch.testing.assert_close(stresses, torch.stack((torch.eye(3), 2 * torch.eye(3))).double() * scale)
        with self.assertRaisesRegex(ValueError, "Unsupported stress unit"):
            system_energy_derivative(OwnedDerivative("invalid"), graph, graph, stress=True)

    def test_model_owned_replicas_keep_each_energy_and_crop_sink_rows(self):
        adapter = OwnedDerivative()
        with torch.no_grad():
            energies, forces = replica_energy_derivative(adapter, None, None, replicas=2, replica_atoms=3)
        self.assertEqual(adapter.calls, [("ef", False, 6, True)])
        torch.testing.assert_close(energies, torch.tensor([6., 9.], dtype=torch.float64))
        self.assertEqual(forces.shape, (6, 3))
        self.assertFalse(energies.requires_grad)

    def test_model_owned_cannot_broadcast_one_energy_over_multiple_systems(self):
        adapter = OwnedDerivative(energies=(2.0,))
        graph = batch()
        with self.assertRaisesRegex(NegotiationError, "returned 1 energies"):
            system_energy_derivative(adapter, graph, graph)
        with self.assertRaisesRegex(NegotiationError, "returned 1 energies"):
            replica_energy_derivative(adapter, None, None, replicas=2, replica_atoms=3)

    def test_segment_and_node_taps_use_each_cell_and_exclude_vacant_rows(self):
        graph = batch()
        expected_bonds = torch.stack((graph.positions[1] + graph.systems.cells[0, 0] - graph.positions[0],
                                     graph.positions[4] + graph.systems.cells[1, 0] - graph.positions[3]))
        expected_energy = expected_bonds.square().sum(dim=1) / 2
        expected_forces = torch.zeros_like(graph.positions)
        expected_forces[[0, 3]], expected_forces[[1, 4]] = expected_bonds, -expected_bonds
        volumes = torch.linalg.det(graph.systems.cells).abs()
        expected_stress = expected_bonds[:, :, None] * expected_bonds[:, None, :] / volumes[:, None, None]
        for adapter in (NodeTap(), SegmentTap()):
            with self.subTest(adapter=type(adapter).__name__), torch.no_grad():
                energies, forces, stresses = system_energy_derivative(
                    adapter, graph, graph, cell=torch.eye(3) * 97, stress=True)
                torch.testing.assert_close(energies, expected_energy)
                torch.testing.assert_close(forces, expected_forces)
                torch.testing.assert_close(stresses, expected_stress)

    def test_builder_preserves_geometry_dtype_and_model_owned_fractional_input(self):
        options = graph_builder_options(NodeTap(), 32)
        self.assertEqual(options["positions_dtype"], torch.float64)
        self.assertEqual(options["t_capacity"], 32)
        self.assertTrue(options["publish_positions"])
        options = graph_builder_options(OwnedDerivative(), 48)
        self.assertEqual(options, {"t_capacity": 48, "line_graph_cutoff": 3.0})

    def test_fastmd_batched_whole_step_runner_remains_available(self):
        self.assertTrue(callable(BatchedWholeStepGraphRunner))


@unittest.skipUnless(torch.cuda.is_available() and op_available and _fixed_builder_available,
                     "CUDA graph builders are unavailable")
class SystemLineGraphCudaTest(unittest.TestCase):
    def test_heterogeneous_line_graph_padding_and_overflow_keep_each_system_cell(self):
        from ase import Atoms
        from fastmd.engine.energy_derivative import engine_derivative
        from fastmd.execution.system_batch import SlotLayout, slot_union_builder, slot_union_demand
        from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder
        from pair_potential_adapter import SmoothPairPotential, UnionPairAdapter

        structures = [Atoms("Cu3", positions=[[0.2, 0.4, 0.3], [1.8, 0.5, 0.4], [0.5, 2.1, 0.6]],
                            cell=[[4.5, 0.2, 0], [0, 5, 0.1], [0, 0, 5.5]], pbc=True),
                      Atoms("Cu2", positions=[[0.3, 0.4, 0.5], [2.1, 0.6, 0.4]],
                            cell=[[6.5, 0.3, 0.1], [0, 7, 0.2], [0, 0, 7.5]], pbc=True)]
        layout = SlotLayout((4, 4))
        cells = torch.tensor(np.stack([atoms.cell.array for atoms in structures]), device="cuda")
        positions = torch.zeros((layout.rows, 3), device="cuda")
        numbers = torch.full((layout.rows,), 29, dtype=torch.int32, device="cuda")
        for start, atoms in zip(layout.starts, structures):
            positions[start:start + len(atoms)] = torch.as_tensor(atoms.positions, device="cuda")
        adapter = UnionPairAdapter(SmoothPairPotential(strength=0.5).cuda(), triplet_cutoff=3.3, angle=0.1)
        options = dict(layout=layout, cells=cells, atomic_numbers=numbers, atoms=[len(a) for a in structures],
                       candidate_skin=1.0, device="cuda")
        pairs, triplets = slot_union_demand(adapter, positions, **options)
        self.assertGreater(pairs, 0)
        self.assertGreater(triplets, 0)
        builder = slot_union_builder(adapter, u_capacity=pairs + 32, t_capacity=triplets + 64, **options)
        graph, status = builder.build(positions)
        self.assertFalse(bool(status.overflow.item()))
        energies, forces, stresses = system_energy_derivative(adapter, builder.batch, graph, stress=True)
        for slot, atoms in enumerate(structures):
            alone = FixedCapacityGraphBuilder(
                cell=cells[slot], atomic_numbers=torch.tensor(atoms.numbers, device="cuda"),
                pbc=torch.ones(3, dtype=torch.bool, device="cuda"), u_capacity=pairs + 32,
                atom_graph_cutoff=5.0, candidate_cutoff=6.0, padding=adapter.padding_rule(),
                prepare=adapter.prepare, device="cuda", **graph_builder_options(adapter, triplets + 64))
            single_graph, single_status = alone.build(torch.tensor(atoms.positions, device="cuda"))
            self.assertFalse(bool(single_status.overflow.item()))
            expected = engine_derivative(adapter)(alone.batch, single_graph, task="efs", n_real=len(atoms))
            torch.testing.assert_close(energies[slot], expected["e"].reshape(()), rtol=1e-5, atol=1e-6)
            start = layout.starts[slot]
            torch.testing.assert_close(forces[start:start + len(atoms)], expected["f"][0], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(stresses[slot], expected["s"][0], rtol=1e-5, atol=1e-7)
        self.assertTrue(bool((forces[~builder.occupied] == 0).all()))
        padded = builder.batch.edge_index[int(status.edge_count.item()):]
        padded_images = builder.batch.edge_image[int(status.edge_count.item()):]
        owners = builder.systems.system_of_row[padded[:, 0].long()].long()
        expected_images = builder._loop_images[owners]
        signs = torch.where(torch.arange(len(padded), device="cuda") % 2 == 0, 1, -1)
        torch.testing.assert_close(padded_images, (expected_images * signs[:, None]).to(padded_images.dtype))

        # An overflowing graph is substituted entirely by loops, also using the cell of each loop's system.
        overflow = slot_union_builder(adapter, u_capacity=1, t_capacity=1, **options)
        graph, status = overflow.build(positions)
        self.assertTrue(bool(status.overflow.item()))
        energies, forces, stresses = system_energy_derivative(adapter, overflow.batch, graph, stress=True)
        torch.testing.assert_close(energies, torch.zeros_like(energies))
        torch.testing.assert_close(forces, torch.zeros_like(forces))
        torch.testing.assert_close(stresses, torch.zeros_like(stresses))


if __name__ == "__main__":
    unittest.main()
