"""Shared zero-skipping SUM/gather: strict semantics and opt-in CUDA tests."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from fastmd.engine.kernels import triton_zero_skip_index as op
from test_int64_offsets import _deterministic


class ZeroSkipIndexCPUTest(unittest.TestCase):
    def test_cpu_is_not_silently_promoted_to_cuda_lowering(self):
        rows, index = torch.zeros(4, 3), torch.tensor([0, 1, 0, 1])
        self.assertIsNotNone(op.zero_skip_index_sum_reason(rows, index, 2))
        self.assertIsNotNone(op.zero_skip_index_select_reason(rows, index))
        for call in (lambda: op.zero_skip_index_sum(rows, index, 2),
                     lambda: op.zero_skip_index_select(rows, index),
                     lambda: op.zero_skip_index_add_(torch.zeros(2, 3), index, rows)):
            with self.assertRaises(ValueError):
                call()


def _seed(values, layout):
    """``values`` as a cotangent of the given memory layout (same elements)."""
    if layout == "row_strided":  # unit column stride: gathered in place
        wide = values.new_full((values.shape[0], values.shape[1] + 5), float("nan"))
        wide[:, 2:2 + values.shape[1]] = values
        return wide[:, 2:2 + values.shape[1]]
    if layout == "column_strided":  # ATen gather fallback
        return values.t().contiguous().t()
    return values


@unittest.skipUnless(torch.cuda.is_available(), "requires coordinator-assigned idle CUDA device")
class ZeroSkipIndexGPUTest(unittest.TestCase):
    def test_long_zero_destination_real_repeats_and_nonzero_sink_keep_full_sum(self):
        # Widths cover the per-element and the row-granular (vector atomic)
        # skip and several column blocks; 50,003 rows end inside a row tile.
        for width, layout in ((3, "contiguous"), (7, "row_strided"), (32, "contiguous"),
                              (32, "column_strided"), (96, "row_strided"), (128, "row_broadcast"),
                              (256, "contiguous")):
            with self.subTest(width=width, seed=layout):
                index = torch.full((50_003,), 36, dtype=torch.long, device="cuda")
                index[:257] = torch.arange(257, device="cuda") % 36
                rows = torch.zeros(len(index), width, device="cuda", requires_grad=True)
                with torch.no_grad():
                    rows[:257] = torch.arange(257 * width, device="cuda").reshape(257, width) % 13 / 16
                    rows[-3:] = torch.tensor([1., -2., 4.], device="cuda")[:, None]
                expected = rows.new_zeros(37, width).index_add(0, index, rows)
                actual = op.zero_skip_index_sum(rows, index, 37)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(actual[-1], torch.full_like(actual[-1], 3.), atol=0, rtol=0)
                if layout == "row_broadcast":  # stride (0, 1)
                    seed = (torch.arange(width, device="cuda").float() / 16).expand(37, width)
                else:
                    seed = _seed(torch.arange(37 * width, device="cuda").reshape(37, width).float()
                                 / 16, layout)
                grad, = torch.autograd.grad(actual, rows, seed)
                torch.testing.assert_close(grad, seed.index_select(0, index), atol=0, rtol=0)
                self.assertTrue(grad.is_contiguous())
                self.assertTrue(bool((grad[257:-3] != 0).any()))
        index = torch.tensor([0, 0, 1, 1, 2, 2], device="cuda")
        base = torch.tensor([[float("nan"), 0., 1.], [0., 0., 2.],
                             [float("inf"), 0., 1.], [float("-inf"), 0., 2.],
                             [float("inf"), float("-inf"), -0.], [1., 1., 0.]], device="cuda")
        for width in (3, 4, 32):  # per-element and row-granular skip
            with self.subTest(values="nan_inf_and_signed_zero", width=width):
                # NaN/Inf and nonzero rows are summed, never filtered as "zero".
                rows = base.repeat(1, 11)[:, :width].contiguous()
                expected = rows.new_zeros(3, width).index_add_(0, index, rows)
                actual = op.zero_skip_index_sum(rows, index, 3)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
                # Existing output: -0.0 survives where every addend is zero,
                # also next to nonzero addends of the same row.
                output = torch.full((3, width), -0.0, device="cuda")
                op.zero_skip_index_add_(output, index, rows)
                reference, host_rows = torch.full((3, width), -0.0), rows.cpu()
                for row, column in (host_rows != 0).nonzero().tolist():
                    reference[index.tolist()[row], column] += host_rows[row, column]
                torch.testing.assert_close(output.cpu(), reference, atol=0, rtol=0, equal_nan=True)
                finite = reference.isfinite()
                self.assertTrue(torch.equal(output.cpu().signbit()[finite], reference.signbit()[finite]))

    def test_deterministic_mode_takes_the_ordered_sum(self):
        # No atomic launch; the SUM and the select VJP equal torch's sums (up to the
        # sign of a zero) and repeat bitwise, with a nonzero sink row kept.
        for width in (7, 32):
            with self.subTest(width=width), _deterministic(atomic_kernels=[(op, "_zero_skip_index_add_kernel")]):
                index = torch.full((50_003,), 36, dtype=torch.long, device="cuda")
                index[:257] = torch.arange(257, device="cuda") % 36
                rows = torch.zeros(len(index), width, device="cuda")
                rows[:257] = torch.randn(257, width, device="cuda")
                rows[-3:] = torch.tensor([1., -2., 4.], device="cuda")[:, None]
                source = torch.randn(37, width, device="cuda", requires_grad=True)
                runs = []
                for _ in range(2):
                    summed = op.zero_skip_index_sum(rows, index, 37)
                    added = op.zero_skip_index_add_(torch.ones(37, width, device="cuda"), index, rows)
                    grad, = torch.autograd.grad(op.zero_skip_index_select(source, index), source, rows)
                    runs.append((summed, added, grad))
                expected = rows.new_zeros(37, width).index_add_(0, index, rows)
                for actual, wanted in zip(runs[0], (expected, expected + 1, expected)):
                    torch.testing.assert_close(actual, wanted, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(runs[0][0][-1], torch.full_like(runs[0][0][-1], 3.), atol=0, rtol=0)
                for again, first in zip(runs[1], runs[0]):
                    self.assertTrue(torch.equal(again.view(torch.int32), first.view(torch.int32)))

    def test_empty_zero_width_strided_fallback_and_strict_errors(self):
        empty = torch.empty(0, dtype=torch.long, device="cuda")
        for width in (0, 3, 128):
            rows = torch.empty(0, width, device="cuda", requires_grad=True)
            output = op.zero_skip_index_sum(rows, empty, 0)
            self.assertEqual(output.shape, (0, width))
            grad, = torch.autograd.grad(output.sum(), rows)
            self.assertEqual(grad.shape, rows.shape)
            self.assertEqual(op.zero_skip_index_select(rows, empty).shape, rows.shape)
        rows = torch.empty(5, 0, device="cuda")
        index = torch.arange(5, device="cuda") % 2
        self.assertEqual(op.zero_skip_index_sum(rows, index, 2).shape, (2, 0))
        self.assertEqual(op.zero_skip_index_select(rows, index).shape, (5, 0))
        # Metadata-only shape checks (the index values are all valid zeros).
        rows = torch.zeros(4, 3, device="cuda")
        index = torch.zeros(4, dtype=torch.long, device="cuda")
        self.assertIsNone(op.zero_skip_index_sum_reason(rows, index, 2))
        self.assertIsNotNone(op.zero_skip_index_sum_reason(rows, index, True))
        self.assertIsNotNone(op.zero_skip_index_sum_reason(rows, index[:2], 2))
        self.assertIsNotNone(op.zero_skip_index_sum_reason(rows, index, 0))
        self.assertIsNotNone(op.zero_skip_index_select_reason(rows[:0], index))
        self.assertIsNone(op.zero_skip_index_sum_reason(rows[:0], index[:0], 0))
        self.assertIsNone(op.zero_skip_index_select_reason(rows[:0], index[:0]))
        rows = torch.randn(5, 6, device="cuda")
        index = torch.arange(5, device="cuda")
        self.assertIsNone(op.zero_skip_index_sum_reason(rows, index, 5))
        self.assertIsNotNone(op.zero_skip_index_select_reason(rows[:, ::2], index))
        self.assertIsNotNone(op.zero_skip_index_sum_reason(rows.double(), index, 5))
        self.assertIsNotNone(op.zero_skip_index_select_reason(rows, index.int()))
        with self.assertRaises(ValueError):
            op.zero_skip_index_select(rows[:, ::2], index)
        with self.assertRaises(ValueError):
            op.zero_skip_index_add_(rows, index, rows)
        with patch.object(op, "_zero_skip_index_add_kernel") as kernel:
            kernel.__getitem__.side_effect = RuntimeError("qualified launch failed")
            with self.assertRaisesRegex(RuntimeError, "qualified launch failed"):
                op.zero_skip_index_sum(rows, index, 5)

    def test_gather_vjp_handles_duplicate_zero_cotangents_and_strided_seed(self):
        for width, layout in ((3, "column_strided"), (7, "row_strided"), (32, "column_strided"),
                              (96, "contiguous"), (128, "row_strided")):
            with self.subTest(width=width, seed=layout):
                source = torch.randn(37, width, device="cuda", requires_grad=True)
                with torch.no_grad():
                    source[5, :2] = torch.tensor([float("nan"), -0.])  # bitwise copies
                index = torch.full((50_003,), 36, dtype=torch.long, device="cuda")
                index[:257] = torch.arange(257, device="cuda") % 36
                seed = torch.zeros(len(index), width, device="cuda")
                seed[:257] = 0.5
                seed[-1] = 2.
                seed = _seed(seed, layout)
                actual = op.zero_skip_index_select(source, index)
                wanted = source.index_select(0, index)
                self.assertTrue(torch.equal(actual.view(torch.int32), wanted.view(torch.int32)))
                grad, = torch.autograd.grad(actual, source, seed)
                expected = torch.zeros_like(source).index_add_(0, index, seed)
                torch.testing.assert_close(grad, expected, atol=0, rtol=0)

    def test_cuda_graph_replay_refreshes_indices_values_and_cotangents(self):
        # Deterministic mode: its ordered SUM is captured too.
        for width, ordered in ((7, False), (32, False), (32, True)):
            with self.subTest(width=width, deterministic=ordered), _deterministic(ordered):
                self._graph_replay(width)

    def _graph_replay(self, width):
        source = torch.randn(37, width, device="cuda", requires_grad=True)
        rows = torch.zeros(8201, width, device="cuda", requires_grad=True)
        index = torch.full((8201,), 36, dtype=torch.long, device="cuda")
        index[:129] = torch.arange(129, device="cuda") % 36
        gather_seed, sum_seed = torch.zeros_like(rows), torch.ones_like(source)

        def body():
            gathered = op.zero_skip_index_select(source, index)
            summed = op.zero_skip_index_sum(rows, index, len(source))
            gs, gr = torch.autograd.grad((gathered, summed), (source, rows),
                                        grad_outputs=(gather_seed, sum_seed))
            return gathered, summed, gs, gr

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warmed = body()
                del warmed
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                output = body()
            for step in range(4):
                with torch.no_grad():
                    index[:129] = (torch.arange(129, device="cuda") + step) % 36
                    rows.zero_()
                    rows[:129] = (step + 1) / 16
                    rows[-1] = step / 8  # Nonzero sink must not be discarded.
                    gather_seed.zero_()
                    gather_seed[:129] = 0.25
                    gather_seed[-1] = step / 4
                    sum_seed.fill_(step + 1)
                    source.add_(0.125)
                graph.replay()
                torch.cuda.synchronize()
                expected = (source.index_select(0, index),
                            torch.zeros_like(source).index_add_(0, index, rows),
                            torch.zeros_like(source).index_add_(0, index, gather_seed),
                            sum_seed.index_select(0, index))
                for actual, wanted in zip(output, expected):
                    torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-4)
        finally:
            torch.cuda.synchronize()
            graph.reset()


if __name__ == "__main__":
    unittest.main()
