"""ORB upstream KNN semantics through the shared single/batch graph builder."""
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import torch
from ase import Atoms

from fastmd.graph.orb_graph import atoms_to_orb_graph, orb_graph_edges


def inputs():
    return [Atoms('CsSnI3', cell=[4.7] * 3, pbc=True,
                  scaled_positions=[[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]),
            Atoms('AcTh3', pbc=True,
                  positions=[[0., 0., 0.], [.49528581, 3.60895118, 1.11071425],
                             [3.43840883, 1.20298854, 1.11071425], [1.96684535, 2.40596746, -2.1894125]],
                  cell=[[4.9099742537, 0., -2.1894125028], [-.9762835525, 4.811934912, -2.1894125028],
                        [0., 0., 6.60025789]])]


@unittest.skipUnless(importlib.util.find_spec('orb_models'), 'optional ORB unavailable')
class ORBGraphTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision('highest')

    def test_single_cpu_graph_exactly_matches_upstream_wrap_and_knn(self):
        from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
        for cap in (20, 120):
            for atoms in inputs():
                with self.subTest(cap=cap, formula=atoms.get_chemical_formula()):
                    expected = ase_atoms_to_atom_graphs(atoms, SystemConfig(radius=6., max_num_neighbors=cap),
                                                       device='cpu', half_supercell=False)
                    actual = atoms_to_orb_graph(atoms, 6., cap, 'cpu')
                    self.assertTrue(torch.equal(actual.positions, expected.node_features['positions']))
                    self.assertTrue(torch.equal(actual.edge_index.long().T,
                                                torch.stack((expected.senders, expected.receivers))))
                    self.assertTrue(torch.equal(actual.edge_image, expected.edge_features['unit_shifts']))

    def test_nearest_k_of_the_radius_graph_is_upstreams_knn_graph_on_a_dense_crystal(self):
        """The engine's CPU radius graph of a perturbed 64-atom diamond (about 158 neighbours per atom within
        6 A), with the nearest-K selection every ORB call applies, keeps exactly upstream's kNN edges (centre,
        neighbour, image) for caps 20 and 120, and every radius edge without a cap above the largest degree."""
        from ase.build import bulk
        from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
        from fastmd.graph.cpu_graph_builder import atoms_to_edges_cpu
        from fastmd.models.orb_graph import nearest_k_mask

        atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
        atoms.positions += np.random.default_rng(16).normal(0., .1, atoms.positions.shape)
        batch = atoms_to_edges_cpu(atoms, 6.)
        senders, receivers = batch.edge_index.long().unbind(1)
        vectors = batch.positions[receivers] - batch.positions[senders] + batch.edge_image @ batch.cell
        rows = torch.arange(len(senders))

        def edges(centres, neighbours, images):
            return set(zip(centres.tolist(), neighbours.tolist(), map(tuple, images.round().long().tolist())))

        degree = int(torch.bincount(senders).max())
        self.assertGreater(int(torch.bincount(senders).min()), 120)
        for cap in (20, 120, degree + 1):
            with self.subTest(cap=cap):
                keep = nearest_k_mask(senders, vectors, rows, len(atoms), cap, 6.)
                expected = ase_atoms_to_atom_graphs(atoms, SystemConfig(radius=6., max_num_neighbors=cap),
                                                    device='cpu', half_supercell=False)
                self.assertEqual(int(keep.sum()), len(expected.senders))
                self.assertEqual(edges(senders[keep], receivers[keep], batch.edge_image[keep]),
                                 edges(expected.senders, expected.receivers, expected.edge_features['unit_shifts']))

    def test_exact_distance_ties_keep_the_same_edges_in_every_row_order(self):
        """A perfect 64-atom diamond has exact FP32 distance ties at the cap of 120 at many centres: the selection
        with the edges' neighbours and images keeps the same edges whatever the row order of the radius graph
        (the builders order their rows differently), and as many per centre as upstream's kNN graph."""
        from ase.build import bulk
        from orb_models.forcefield.atomic_system import SystemConfig, ase_atoms_to_atom_graphs
        from fastmd.graph.cpu_graph_builder import atoms_to_edges_cpu
        from fastmd.models.orb_graph import nearest_k_mask

        atoms = bulk('C', 'diamond', a=3.567, cubic=True).repeat(2)
        batch = atoms_to_edges_cpu(atoms, 6.)
        senders, receivers = batch.edge_index.long().unbind(1)
        images = batch.edge_image
        vectors = batch.positions[receivers] - batch.positions[senders] + images @ batch.cell
        distances = vectors.square().sum(-1)
        ties = sum(int(torch.sort(distances[senders == centre]).values[119]
                       == torch.sort(distances[senders == centre]).values[120]) for centre in range(len(atoms)))
        self.assertGreater(ties, 0)

        def kept(order):
            keep = nearest_k_mask(senders[order], vectors[order], torch.arange(len(order)), len(atoms), 120, 6.,
                                  receivers[order], images[order])
            return set(zip(senders[order][keep].tolist(), receivers[order][keep].tolist(),
                           map(tuple, images[order][keep].round().long().tolist())))

        expected = kept(torch.arange(len(senders)))
        for seed in range(3):
            with self.subTest(permutation=seed):
                self.assertEqual(kept(torch.randperm(len(senders), generator=torch.Generator().manual_seed(seed))),
                                 expected)
        upstream = ase_atoms_to_atom_graphs(atoms, SystemConfig(radius=6., max_num_neighbors=120), device='cpu',
                                            half_supercell=False)
        self.assertEqual(len(expected), len(upstream.senders))

    def test_odd_directed_edges_reserve_enough_model_graph_capacity(self):
        from fastmd.execution.cuda_graph import _demand, _PaddedEdgesWorkspace
        from fastmd.adapters.protocol import GraphBatch, PaddingRule
        batch = GraphBatch(positions=torch.zeros((2, 3)), frac_coords=torch.zeros((2, 3)),
                          cell=torch.eye(3) * 8, species=torch.ones(2, dtype=torch.int32), n_real=2,
                          edge_index=torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32),
                          edge_image=torch.zeros((3, 3)), n_edges=torch.tensor([3]), cutoff=6., padding=None)
        self.assertEqual(_demand(batch), (2, 2, 0))
        workspace = _PaddedEdgesWorkspace.create(batch, 2, PaddingRule('far_self_loops'), 0)
        workspace.update(batch)
        self.assertEqual(workspace.batch.edge_index.shape[0], 4)
        self.assertTrue(torch.equal(workspace.batch.edge_index[:3], batch.edge_index))
        self.assertEqual(workspace.batch.n_edges.item(), 3)

    @unittest.skipUnless(os.environ.get('ORB_CHECKPOINT_DIR'), 'set ORB_CHECKPOINT_DIR for local ORB weights')
    def test_real_single_calculator_parity_on_cpu(self):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        from fastmd import InferenceConfig, MLIPCalculator
        from fastmd.models.orb import ORBAdapter
        from fastmd.models.orb_graph import ORBGraphAdapter
        for name in ('orb-v3-direct-20-mpa', 'orb-v3-conservative-20-mpa', 'orb-v3-conservative-inf-mpa'):
            with self.subTest(checkpoint=name):
                module = pretrained.ORB_PRETRAINED_MODELS[name](
                    weights_path=str(Path(os.environ['ORB_CHECKPOINT_DIR']) / (name + '.ckpt')),
                    device='cpu', precision='float32-highest', compile=False)
                reference = ORBCalculator(module, device='cpu', half_supercell=False)
                expected = []
                for a in inputs():
                    a.calc = reference; reference.reset()
                    expected.append((a.get_potential_energy(), a.get_forces(), a.get_stress()))
                adapter = ORBAdapter(module) if 'conservative-inf' in name else ORBGraphAdapter(module)
                calculator = MLIPCalculator(model=adapter, task='efs', device='cpu', enable_compile=False,
                    config=InferenceConfig(execution='eager', optimization_profile='topology',
                                           matmul_precision='highest', triton_dot_precision='ieee', report=False))
                try:
                    for a, (energy, forces, stress) in zip(inputs(), expected):
                        a.calc = calculator; calculator.reset()
                        with patch.object(module.model, 'forward', wraps=module.model.forward) as calls:
                            actual = (a.get_potential_energy(), a.get_forces(), a.get_stress())
                        self.assertEqual(calls.call_count, 1)
                        self.assertLess(abs(actual[0] - energy) / len(a), 1e-5)
                        np.testing.assert_allclose(actual[1], forces, rtol=0, atol=1e-3)
                        np.testing.assert_allclose(actual[2], stress, rtol=0, atol=1e-4)
                finally:
                    calculator.close(); adapter.close()


if __name__ == '__main__':
    unittest.main()
