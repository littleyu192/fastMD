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
    def test_real_single_calculator_parity_and_no_generic_skin(self):
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
                    self.assertIn('upstream capped KNN', calculator._skin_static)
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
