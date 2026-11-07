"""Shared declarations must preserve optional protocol membership and instance state."""
import unittest
import torch

from fastmd.adapters.kit import declares, times_cell
from fastmd.adapters.protocol import EdgeShiftInput, PaddingRule


class KitTests(unittest.TestCase):
    def test_only_explicit_capabilities_are_installed(self):
        @declares(padding=PaddingRule('far_self_loops'))
        class Plain:
            pass

        @declares(edge_shifts=True)
        class Shifted(Plain):
            pass

        self.assertFalse(isinstance(Plain(), EdgeShiftInput))
        self.assertTrue(isinstance(Shifted(), EdgeShiftInput))
        self.assertTrue(Shifted().edge_shift_input())
        self.assertEqual(Shifted().padding_rule(), Plain().padding_rule())

    def test_duplicate_definition_fails_without_overwriting(self):
        class Adapter:
            def edge_shift_input(self):
                return False

        with self.assertRaisesRegex(TypeError, 'both declared and defined'):
            declares(edge_shifts=True)(Adapter)
        self.assertFalse(Adapter().edge_shift_input())

    def test_patch_outcomes_are_read_from_each_instance_at_call_time(self):
        @declares(patches=('first', 'second'))
        class Adapter:
            def __init__(self):
                self.first, self.second = [], []

        a, b = Adapter(), Adapter()
        a.first.append('applied')
        self.assertEqual(a.upstream_patches(), ['applied'])
        self.assertEqual(b.upstream_patches(), [])
        result = a.upstream_patches()
        a.second.append('reverted')
        self.assertEqual(result, ['applied'])
        self.assertEqual(a.upstream_patches(), ['applied', 'reverted'])

    def test_rowwise_shifts_and_cell_gradients_are_exact(self):
        torch.manual_seed(29)
        image = torch.randint(-3, 4, (7, 3)).double()
        cell = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        expected = image[:, 0:1]*cell[0] + image[:, 1:2]*cell[1] + image[:, 2:3]*cell[2]
        actual = times_cell(image, cell)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), cell)[0],
                                   image.sum(0)[:, None].expand(3, 3), rtol=0, atol=0)
