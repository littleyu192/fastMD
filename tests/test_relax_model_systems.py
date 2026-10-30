"""CPU checks for relaxation model unions, independent of CUDA graph builders."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest

import torch

from fastmd.adapters.negotiate import check_system_cells
from fastmd.adapters.protocol import NegotiationError, PaddingRule, SystemSegments, TopologySpec
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.models.matris.adapter import MatRISAdapter
from fastmd.models.matris.processgraph import _system_geometry
from fastmd.models.orb import ORBAdapter
from tests.test_isolated_atom_handling import make_graph, make_model


class SystemModelTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)

    def test_matris_union_matches_independent_cells_with_empty_lines_and_vacancies(self):
        torch.manual_seed(23)
        model = make_model()
        # A triangle has line-graph updates; the other system is a pair without
        # any angles. A union must not enable the pair's biased line layers.
        cell = torch.diag(torch.tensor([6.0, 7.0, 8.0]))
        first = RadiusGraph(
            graph_id=None, mp_id=None, composition="H3", atomic_number=torch.ones(3, dtype=torch.int32),
            atom_frac_coord=torch.tensor([[0., 0., 0.], [1., 0., 0.], [.4, .8, 0.]]) / cell.diag(),
            lattice=cell, neighbor_image=torch.zeros(6, 3),
            atom_graph=torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]], dtype=torch.int32),
            atom_graph_cutoff=2.0,
            line_graph=torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 0, 0],
                                     [1, 0, 2, 2, 3], [1, 2, 3, 0, 2],
                                     [2, 1, 4, 2, 5], [2, 2, 5, 1, 4]], dtype=torch.int32),
            line_graph_cutoff=1.5, directed2undirected=torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.int32),
            undirected2directed=torch.tensor([0, 1, 3], dtype=torch.int32),
            atom_target_sorted=True, line_atom_sorted=True,
        )
        second = make_graph(2, connected_pair=True)
        second.lattice = torch.diag(torch.tensor([9.0, 10.0, 11.0]))
        second.atom_frac_coord = torch.tensor([[0., 0., 0.], [1.2 / 9., 0., 0.]])
        # Two four-row slots and two sink atoms: occupied rows 0..2 and 4..5.
        union = RadiusGraph(
            graph_id=None, mp_id=None, composition="H5", atomic_number=torch.ones(10, dtype=torch.int32),
            atom_frac_coord=torch.cat((first.atom_frac_coord, torch.zeros(1, 3),
                                       second.atom_frac_coord, torch.zeros(4, 3))),
            lattice=first.lattice, neighbor_image=torch.cat((first.neighbor_image, second.neighbor_image)),
            atom_graph=torch.cat((first.atom_graph, second.atom_graph + 4)), atom_graph_cutoff=2.0,
            line_graph=first.line_graph, line_graph_cutoff=1.5,
            directed2undirected=torch.cat((first.directed2undirected, second.directed2undirected + 3)),
            undirected2directed=torch.cat((first.undirected2directed, second.undirected2directed + 6)),
            atom_target_sorted=True, line_atom_sorted=True,
        )
        union.systems = SystemSegments(
            count=2, row_ptr=torch.tensor([0, 4, 8]), system_of_row=torch.tensor([0] * 4 + [1] * 4),
            cells=torch.stack((first.lattice, second.lattice)), atoms=torch.tensor([3, 2]),
        )
        expected = [model([graph], task="efs", is_training=False) for graph in (first, second)]
        actual = model([union], task="efs", is_training=False, n_real=8)
        for index, (start, count) in enumerate(((0, 3), (4, 2))):
            torch.testing.assert_close(actual["e"][index], expected[index]["e"][0], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual["f"][0][start:start + count], expected[index]["f"][0],
                                       rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual["s"][index], expected[index]["s"][0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual["f"][0][[3, 6, 7, 8, 9]], torch.zeros(5, 3), rtol=0, atol=0)
        self.assertEqual(check_system_cells(MatRISAdapter(model)), "union")

    def test_matris_geometry_uses_each_cells_image_and_strain(self):
        graph = make_graph(2, connected_pair=True)
        graph.atom_frac_coord = torch.tensor([[.1, .2, .3], [.2, .3, .4]])
        graph.neighbor_image = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        cells = torch.stack((torch.eye(3) * 4., torch.eye(3) * 7.))
        systems = SystemSegments(count=2, row_ptr=torch.tensor([0, 1, 2]),
                                 system_of_row=torch.tensor([0, 1]), cells=cells,
                                 atoms=torch.tensor([1, 1]))
        coords, strain, _, volumes, shifts = _system_geometry(graph, systems, torch.tensor([0, 1]), True)
        torch.testing.assert_close(coords, graph.atom_frac_coord * torch.tensor([[4.], [7.]]))
        torch.testing.assert_close(shifts, torch.tensor([[4., 0., 0.], [0., 7., 0.]]))
        torch.testing.assert_close(volumes.flatten(), torch.tensor([64., 343.]))
        grad, = torch.autograd.grad(shifts[0].sum(), (strain,))
        self.assertGreater(float(grad[0].abs().sum()), 0)
        torch.testing.assert_close(grad[1], torch.zeros(3, 3), rtol=0, atol=0)

    def test_orb_pooled_readout_ignores_unoccupied_slot_rows_and_their_gradients(self):
        adapter = object.__new__(ORBAdapter)
        head = SimpleNamespace(reference=SimpleNamespace(linear=SimpleNamespace(weight=torch.arange(4.))))
        adapter.module = SimpleNamespace(heads={"energy": head}, energy_name="energy", pair_repulsion=False)
        graph = SimpleNamespace(node_features={"atomic_numbers": torch.tensor([1, 2, 3, 1, 2])},
                                senders=torch.tensor([0, 3]))
        systems = SystemSegments(count=2, row_ptr=torch.tensor([0, 3, 5]),
                                 system_of_row=torch.tensor([0, 0, 0, 1, 1]), atoms=torch.tensor([2, 1]))
        nodes = torch.tensor([[1., 2.], [3., 4.], [1e8, 1e8], [5., 6.], [1e8, 1e8]], requires_grad=True)
        sums, reference = adapter._readout_sums(graph, nodes, systems=systems)
        torch.testing.assert_close(sums, torch.tensor([[4., 6., 0.], [5., 6., 0.]], dtype=torch.float64))
        torch.testing.assert_close(reference, torch.tensor([3., 1.], dtype=torch.float64))
        grad, = torch.autograd.grad(sums.sum(), (nodes,))
        torch.testing.assert_close(grad[[2, 4]], torch.zeros(2, 2), rtol=0, atol=0)

    def test_segments_contract_requires_real_edge_shift_keyword(self):
        class Pooled:
            name = "pooled-test"

            def system_batching(self):
                return "segments"

            def edge_shift_input(self):
                return True

            def padding_rule(self):
                return PaddingRule("far_self_loops")

            def topology_spec(self):
                return TopologySpec(cutoff=5., num_interactions=1, triplet_cutoff=3.)

            def system_energy(self, inputs, positions, cell, edge_shifts=None):
                raise NotImplementedError

        self.assertEqual(check_system_cells(Pooled()), "segments")

        class FalseDeclaration(Pooled):
            def system_energy(self, inputs, positions, cell):
                raise NotImplementedError

        with self.assertRaisesRegex(NegotiationError, "takes no edge_shifts keyword"):
            check_system_cells(FalseDeclaration())


@unittest.skipUnless(importlib.util.find_spec("chgnet"), "optional CHGNet package unavailable")
class CHGNetSystemShiftTest(unittest.TestCase):
    def test_shifted_bonds_match_single_cell_for_biased_empty_and_nonempty_angles(self):
        from chgnet.model.model import CHGNet
        from pymatgen.io.ase import AseAtomsAdaptor
        from fastmd.models.chgnet import CHGNetAdapter
        from tests.adapters.test_chgnet_empty_angles import batch_of, structures

        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        model = CHGNet.load(model_name="0.2.0", use_device="cpu", verbose=False)
        adapter = CHGNetAdapter(model)
        self.addCleanup(adapter.close)
        self.assertEqual(check_system_cells(adapter), "union")
        for atoms in structures():
            batch = batch_of(model.graph_converter(AseAtomsAdaptor.get_structure(atoms)))
            inputs = adapter.prepare(batch)
            positions = batch.positions.detach().requires_grad_(True)
            expected = adapter.node_energy(inputs, positions, batch.cell)
            force, = torch.autograd.grad(expected.sum(), (positions,))
            # Deliberately inconsistent cell: a shift-aware tap must use only
            # the supplied Cartesian shifts to construct bonds and angles.
            actual = adapter.node_energy(inputs, positions, batch.cell * 1.5,
                                         edge_shifts=batch.edge_image @ batch.cell)
            shifted_force, = torch.autograd.grad(actual.sum(), (positions,))
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(shifted_force, force, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
