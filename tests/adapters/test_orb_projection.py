"""ORB projection parity, VJPs and opt-in lifecycle.

Run with TRITON_INTERPRET=1 for CPU execution of the actual Triton kernels.
FASTMD_TEST_DEVICE=cuda selects a GPU when one has been allocated for testing.
"""
import copy
import importlib.metadata
import os
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

DEVICE = os.environ.get("FASTMD_TEST_DEVICE", "cpu")
try:
    ORB = importlib.metadata.version("orb-models") == "0.5.4"
except importlib.metadata.PackageNotFoundError:
    ORB = False


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(918)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)

    def test_concat_parity_all_input_gradients_and_frozen_weights(self):
        from fastmd.models.orb_ops.triton_orb_projection import projected_edge_silu
        for dtype in (torch.float32, torch.float64):
            for count, width in ((0, 7), (1, 16), (19, 33), (17, 1024)):
                with self.subTest(dtype=dtype, edges=count, output_width=width):
                    nodes = torch.randn(5, 8, dtype=dtype, device=DEVICE, requires_grad=True)
                    edges = torch.randn(count, 8, dtype=dtype, device=DEVICE, requires_grad=True)
                    weight = torch.randn(width, 24, dtype=dtype, device=DEVICE) / 5
                    bias = torch.randn(width, dtype=dtype, device=DEVICE)
                    source = torch.randint(5, (count,), device=DEVICE)
                    target = torch.randint(5, (count,), device=DEVICE)
                    expected = F.silu(F.linear(torch.cat((edges, nodes[source], nodes[target]), -1),
                                               weight, bias))
                    actual = projected_edge_silu(nodes, edges, source, target,
                        tuple(w.contiguous() for w in weight.split(8, dim=1)), bias)
                    cotangent = torch.randn_like(expected)
                    ref_grad = torch.autograd.grad(expected, (nodes, edges), cotangent)
                    actual_grad = torch.autograd.grad(actual, (nodes, edges), cotangent)
                    for observed, reference in zip((actual, *actual_grad), (expected, *ref_grad)):
                        torch.testing.assert_close(observed, reference, rtol=3e-5, atol=3e-5)

    def test_second_derivative_and_noncontiguous_fallback(self):
        from fastmd.models.orb_ops.triton_orb_projection import gather_add_silu, gather_add_silu_torch
        source = torch.tensor([2, 1, 2, 0, 1], device=DEVICE)
        target = torch.tensor([1, 0, 0, 1, 2], device=DEVICE)
        for noncontiguous in (False, True):
            values = tuple(torch.randn(rows, 14, device=DEVICE)[:, ::2].requires_grad_()
                           for rows in (5, 3, 3))
            if not noncontiguous:
                values = tuple(value.detach().contiguous().requires_grad_() for value in values)
            expected = gather_add_silu_torch(*values, source, target)
            actual = gather_add_silu(*values, source, target)
            ref_first = torch.autograd.grad(expected.square().sum(), values, create_graph=True)
            actual_first = torch.autograd.grad(actual.square().sum(), values, create_graph=True)
            ref_second = torch.autograd.grad(sum(value.sum() for value in ref_first), values)
            actual_second = torch.autograd.grad(sum(value.sum() for value in actual_first), values)
            for observed, reference in zip(actual_second, ref_second):
                torch.testing.assert_close(observed, reference, rtol=2e-5, atol=3e-5)

    def test_deterministic_policy_declines_atomics(self):
        from fastmd.models.orb_ops import triton_orb_projection as projection
        values = tuple(torch.randn(rows, 7, requires_grad=True) for rows in (5, 3, 3))
        source = torch.tensor([0, 1, 1, 2, 0])
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with mock.patch.object(projection._GatherAddSiLU, 'apply', side_effect=AssertionError('atomic')):
                result = projection.gather_add_silu(*values, source, source)
                torch.autograd.grad(result.sum(), values)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@unittest.skipUnless(ORB, "requires orb-models==0.5.4")
class ProjectionAdapterTests(unittest.TestCase):
    def model(self):
        from orb_models.forcefield.atomic_system import SystemConfig
        from orb_models.forcefield.pretrained import orb_v3_conservative_architecture
        return orb_v3_conservative_architecture(latent_dim=8, base_mlp_hidden_dim=16,
            base_mlp_depth=1, head_mlp_hidden_dim=8, head_mlp_depth=1,
            num_message_passing_steps=1, device=DEVICE,
            system_config=SystemConfig(radius=6., max_num_neighbors=20))

    def test_layer_parity_fingerprint_isolation_and_close(self):
        from fastmd.models.orb_graph import ORBGraphAdapter
        from orb_models.forcefield.gns import AttentionInteractionNetwork
        raw = self.model()
        baseline = ORBGraphAdapter(copy.deepcopy(raw))
        candidate = ORBGraphAdapter(copy.deepcopy(raw))
        self.addCleanup(baseline.close)
        self.addCleanup(candidate.close)
        baseline.enable_operator_rewrites('torch')
        candidate.enable_operator_rewrites('torch', projection_backend='triton')
        self.assertNotIn('projection_backend', baseline.system_inputs())
        self.assertEqual(candidate.system_inputs()['projection_backend'], 'triton')
        self.assertEqual(candidate.system_inputs()['operator_backend'], 'torch')
        values = (torch.randn(5, 8, device=DEVICE, requires_grad=True),
                  torch.randn(13, 8, device=DEVICE, requires_grad=True),
                  torch.rand(13, 1, device=DEVICE, requires_grad=True))
        source, target = torch.randint(5, (13,), device=DEVICE), torch.randint(5, (13,), device=DEVICE)
        expected = baseline.module.model.gnn_stacks[0](*values[:2], source, target, values[2])
        actual = candidate.module.model.gnn_stacks[0](*values[:2], source, target, values[2])
        cotangents = tuple(torch.randn_like(value) for value in expected)
        ref_grad = torch.autograd.grad(expected, values, cotangents)
        actual_grad = torch.autograd.grad(actual, values, cotangents)
        for observed, reference in zip((*actual, *actual_grad), (*expected, *ref_grad)):
            torch.testing.assert_close(observed, reference, rtol=3e-5, atol=3e-5)
        candidate.close()
        self.assertIs(candidate.module.model.gnn_stacks[0].forward.__func__,
                      AttentionInteractionNetwork.forward)
        self.assertIsNot(baseline.module.model.gnn_stacks[0].forward.__func__,
                         AttentionInteractionNetwork.forward)

    def test_invalid_option_and_failed_install_roll_back(self):
        from fastmd.models.orb_graph import ORBGraphAdapter
        from fastmd.models.orb_ops.orb_steady import install
        candidate = ORBGraphAdapter(self.model())
        self.addCleanup(candidate.close)
        for options in ({'projection_backend': 'invalid'},
                        {'projection_backend': 'triton', 'steady_options': {'prune': True}}):
            with self.assertRaises(ValueError):
                candidate.enable_operator_rewrites('torch', **options)
        candidate._base.kernel = 'inductor'
        with mock.patch.object(candidate.module.model, 'compile', side_effect=RuntimeError('compile failed')):
            with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                candidate.enable_operator_rewrites('torch', projection_backend='triton')
        self.assertIsNone(candidate._base.projection_backend)
        self.assertEqual(candidate._base.operator_backend, 'upstream')
        self.assertFalse(any(name.startswith('_fastmd_')
                             for layer in candidate.module.model.gnn_stacks for name in vars(layer)))
        candidate._base.kernel = 'eager'
        candidate.enable_operator_rewrites('torch', projection_backend='triton')
        with self.assertRaisesRegex(ValueError, 'projection_backend'):
            install(candidate, merge_node=True)


if __name__ == '__main__':
    unittest.main()
