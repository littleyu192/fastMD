"""Independent FP64 stage equations and coordinate finite differences.

The equations are the original pre-refactor formulas, not the production
shared-forward methods. No checkpoint, GPU, or distributed backend is needed.
"""
import unittest
import torch
import shared_forward_oracle as oracle
from single_gpu_fixtures import toy_model, frozen_fixture, hand_built_dispatcher


class SingleDeviceReferenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_each_attention_and_refinement_stage_output_and_vjp(self):
        """Independent pre-refactor equations check the shared layer forward."""
        for smoothed in (False, True):
            model = toy_model(smoothed=smoothed, reference=None, dtype=torch.float64)
            block = model.interaction_block[0]
            for is_atom in (False, True):
                target = torch.tensor([0, 1, 0, 2, 2, 3] if is_atom else [0, 1, 1, 2])
                source = torch.tensor([1, 0, 2, 0, 3, 2] if is_atom else [1, 0, 2, 1])
                d2u = torch.tensor([0, 0, 1, 1, 2, 2]) if is_atom else None
                graph = {"source_index": source, "target_index": target,
                         "source_bincount": torch.bincount(source, minlength=4).clamp_min(1),
                         "target_bincount": torch.bincount(target, minlength=4).clamp_min(1),
                         "num_undirected": 3, "atom_list": torch.tensor([0, 0, 2, 2]),
                         "_ops": hand_built_dispatcher()}
                nodes = torch.randn(4, 6, dtype=torch.float64)
                edges = torch.randn(3 if is_atom else 4, 6, dtype=torch.float64)
                attention = block.attn_block_atom_graph if is_atom else block.attn_block_line_graph

                # skip: the atom attention of a graph without triplets returns its input bonds (P=1 and P>1).
                for skip in ((False, True) if is_atom else (False,)):
                    def ordinary_attention(node, edge):
                        return attention(node, edge, graph, directed2undirected=d2u, skip_edge_update=skip)

                    def reference_attention(node, edge):
                        expanded = edge.repeat_interleave(2, dim=0) if is_atom else edge
                        message, source_logits, target_logits = oracle.attention_rows(attention, node[target], node[source], expanded)
                        return (oracle.attention_nodes(attention, node, message, source_logits, source, message, target_logits, target),
                                oracle.attention_edges(attention, edge, message, paired=is_atom, skip=skip))

                    self.check_outputs_and_vjp(ordinary_attention, reference_attention, (nodes, edges))
                refinement = block.refine_block_atom_graph if is_atom else block.refine_block_line_graph
                radial = torch.randn(3 if is_atom else 4, 3, dtype=torch.float64)
                atom_features = torch.randn(4, 6, dtype=torch.float64)
                for skip in (False, True):  # the atom skip: the last block under a conservative head (P=1)
                    def ordinary_refinement(node, edge, basis, atoms):
                        return refinement(node, edge, basis, graph, directed2undirected=d2u,
                                          atom_feat=None if is_atom else atoms, skip_edge_update=skip)

                    def reference_refinement(node, edge, basis, atoms):
                        expanded = edge.repeat_interleave(2, dim=0) if is_atom else edge
                        envelope = oracle.module_forward(refinement.learnable_envelope, basis)
                        weights = envelope.repeat_interleave(2, dim=0) if is_atom else envelope[source] * envelope[target]
                        weighted, edge_input = oracle.refinement_rows(refinement, node[target], node[source], expanded,
                                                                   weights, None if is_atom else atoms[graph["atom_list"]])
                        return (oracle.refinement_nodes(refinement, node, weighted, target),
                                oracle.refinement_edges(refinement, edge, edge_input, paired=is_atom, skip=skip))

                    self.check_outputs_and_vjp(ordinary_refinement, reference_refinement, (nodes, edges, radial, atom_features))

    def check_outputs_and_vjp(self, original_fn, reference_fn, inputs):
        first = tuple(value.clone().requires_grad_(True) for value in inputs)
        second = tuple(value.clone().requires_grad_(True) for value in inputs)
        original = original_fn(*first)
        actual = reference_fn(*second)
        cotangents = tuple(torch.randn_like(value) for value in original)
        for actual_value, expected_value in zip(actual, original):
            torch.testing.assert_close(actual_value, expected_value, atol=2e-12, rtol=2e-12)
        original_grad = torch.autograd.grad(original, first, grad_outputs=cotangents, allow_unused=True)
        actual_grad = torch.autograd.grad(actual, second, grad_outputs=cotangents, allow_unused=True)
        for expected_value, actual_value in zip(original_grad, actual_grad):
            if expected_value is None:
                self.assertIsNone(actual_value)
            else:
                torch.testing.assert_close(actual_value, expected_value, atol=2e-12, rtol=2e-11)

    def test_force_matches_coordinate_finite_difference(self):
        model = toy_model(smoothed=True, reference=None, dtype=torch.float64)
        model.interaction_block[-1].last_block = False
        step = 1e-5
        for periodic in (False, True):
            atoms, graph = frozen_fixture(dtype=torch.float64, periodic=periodic)
            prediction = model([graph], task="ef")
            for atom, axis in ((0, 0), (1, 1), (2, 2), (4, 0), (5, 0)):
                with self.subTest(periodic=periodic, atom=atom, axis=axis):
                    energies = []
                    for sign in (-1, 1):
                        positions = atoms.positions.clone()
                        positions[atom, axis] += sign * step
                        _, moved = frozen_fixture(dtype=torch.float64, periodic=periodic, positions=positions)
                        energies.append(model([moved], task="e")["e"][0].item() * len(positions))
                    derivative = (energies[1] - energies[0]) / (2 * step)
                    self.assertAlmostEqual(-prediction["f"][0][atom, axis].item(), derivative, delta=2e-8)


if __name__ == "__main__":
    unittest.main()
