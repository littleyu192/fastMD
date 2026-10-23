"""Two-part segment attention for the A2 bond partials (fastmd.models.matris.op.segment_partials).

* segment_attention / paired_segment_attention(return_lse=True): out, lse and
  the VJP of both against a per-segment FP64 softmax, with an empty segment,
  a single-row segment and a skipped segment. Cotangents are nonzero
  everywhere, also on the -inf lse of the empty and skipped segments. The
  fused kernels run in FP32 and FP64 with the warps by width (the default),
  1 and 4 warps per program (MATRIS_FUSED_SEGMENT_ATTENTION_NUM_WARPS), the
  eager (CPU) path in FP64.
* merge_lse_partials: the fused kernel against the FP64 softmax over the
  union of both parts' rows and against merge_lse_partials_torch (ulp level),
  with one-sided and two-sided empty parts; forward, backward and
  create_graph. (own, peer) with first_is_own evaluated at both ends gives the
  same bits.
"""
from __future__ import annotations

from itertools import product
import unittest
from unittest.mock import patch

import torch

from fastmd.models.matris.op.segment_partials import (
    _MergeLsePartials, merge_lse_partials, merge_lse_partials_torch,
)
from fastmd.models.matris.op import triton_segment_attention
from fastmd.models.matris.op.triton_segment_attention import paired_segment_attention, segment_attention

_CUDA = torch.cuda.is_available()
_TOL = {torch.float32: 1e-5, torch.float64: 1e-12}
_INF = float("inf")


def _assert_close(test, named, tol):
    """Max error relative to the reference's largest finite magnitude; the
    non-finite entries (-inf lse of an empty part) must match exactly."""
    for name, actual, expected in named:
        with test.subTest(output=name):
            test.assertEqual(tuple(actual.shape), tuple(expected.shape))
            actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
            finite = torch.isfinite(expected)
            test.assertTrue(torch.equal(torch.isfinite(actual), finite), "non-finite pattern differs")
            test.assertTrue(torch.equal(actual[~finite], expected[~finite]))
            err = (actual - expected)[finite].abs().max() / expected[finite].abs().max()
            test.assertLess(float(err), tol)


# 1. lse from the fused segment attention -------------------------------------


def _reference_attention(logits, value, segment, count, skip):
    """Per segment softmax(logits) . value and logsumexp(logits); (0, -inf)
    for an empty or the skipped segment."""
    width = logits.shape[1]
    out, lse = [], []
    for s in range(count):
        rows = (segment == s).nonzero().flatten()
        if s == skip or rows.numel() == 0:
            out.append(logits.new_zeros(width))
            lse.append(logits.new_full((width,), -_INF))
        else:
            out.append((torch.softmax(logits[rows], 0) * value[rows]).sum(0))
            lse.append(torch.logsumexp(logits[rows], 0))
    return torch.stack(out), torch.stack(lse)


def _segments(generator, count):
    """Unsorted ids: segment 2 empty, 5 a single row, the sink (count - 1) 12 rows."""
    lengths = torch.randint(2, 7, (count,), generator=generator)
    lengths[2], lengths[5], lengths[count - 1] = 0, 1, 12
    segment = torch.repeat_interleave(torch.arange(count), lengths)
    return segment[torch.randperm(segment.numel(), generator=generator)]


class SegmentAttentionLseTest(unittest.TestCase):
    def test_warps_by_padded_width(self):
        from fastmd.models.matris.op import triton_segment_attention

        with patch.object(triton_segment_attention, "_NUM_WARPS", None):
            self.assertEqual([triton_segment_attention._num_warps(blk) for blk in (16, 32, 64, 128, 256)],
                             [1, 1, 2, 4, 4])
        with patch.object(triton_segment_attention, "_NUM_WARPS", 4):
            self.assertEqual(triton_segment_attention._num_warps(32), 4)

    def test_lse_and_vjp_match_fp64(self):
        count, width = 13, 48  # a width below its power-of-two block
        paths = [("cpu", torch.float64)] + ([("cuda", torch.float32), ("cuda", torch.float64)] if _CUDA else [])
        for (device, dtype), skip, warps in product(paths, (False, True), (None, 1, 4)):
            if device == "cpu" and warps is not None:
                continue
            generator = torch.Generator().manual_seed(7)
            seg_a = _segments(generator, count)
            seg_b = seg_a[torch.randperm(seg_a.numel(), generator=generator)]  # other rows, same sizes
            rows = seg_a.numel()
            logits_a, logits_b = (torch.randn(rows, width, generator=generator, dtype=torch.float64) * 3
                                  for _ in range(2))
            value = torch.randn(rows, width, generator=generator, dtype=torch.float64)
            seeds = [torch.randn(count, width, generator=generator, dtype=torch.float64) for _ in range(4)]
            skip_segment = count - 1 if skip else -1

            def run(inputs, segments, fn, **kwargs):
                leaves = [t.detach().to(device, dtype).requires_grad_(True) for t in inputs]
                got = fn(*leaves, *(s.to(device) for s in segments), count,
                         skip_segment=skip_segment, return_lse=True, **kwargs)
                grads = torch.autograd.grad(got, leaves, [s.to(device, dtype) for s in seeds[:len(got)]])
                return (*got, *grads)

            def reference(inputs, segments, sides):
                leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
                parts = [_reference_attention(leaves[side], leaves[-1], segments[side], count, skip_segment)
                         for side in range(sides)]
                got = [part[0] for part in parts] + [part[1] for part in parts]  # (out..., lse...)
                return (*got, *torch.autograd.grad(got, leaves, seeds[:len(got)]))

            with self.subTest(op="segment_attention", device=device, dtype=dtype, skip=skip, warps=warps), \
                    patch.object(triton_segment_attention, "_NUM_WARPS", warps):
                got = run((logits_a, value), (seg_a,), segment_attention)
                ref = reference((logits_a, value), (seg_a,), 1)
                _assert_close(self, zip(("out", "lse", "grad_logits", "grad_value"), got, ref), _TOL[dtype])
            with self.subTest(op="paired_segment_attention", device=device, dtype=dtype, skip=skip, warps=warps), \
                    patch.object(triton_segment_attention, "_NUM_WARPS", warps):
                got = run((logits_a, logits_b, value), (seg_a, seg_b), paired_segment_attention)
                ref = reference((logits_a, logits_b, value), (seg_a, seg_b), 2)
                names = ("out_a", "out_b", "lse_a", "lse_b", "grad_logits_a", "grad_logits_b", "grad_value")
                _assert_close(self, zip(names, got, ref), _TOL[dtype])


# 2. The canonical-order merge ------------------------------------------------


def _partial(logits, value, valid):
    """FP64 (lse, y) over the valid rows (last axis), packed [rows, groups*2*width];
    (-inf, 0) where no row is valid."""
    masked = logits.masked_fill(~valid, -_INF)
    y = (torch.softmax(masked, -1) * value).sum(-1)  # NaN where empty: replaced below
    y = torch.where(valid.any(-1), y, 0.0)
    return torch.stack((torch.logsumexp(masked, -1), y), 2).flatten(1)


def _merge_inputs(rows=64, groups=2, width=24, depth=5):
    """Two parts per (row, group, channel) of up to ``depth`` rows each and the
    FP64 softmax over their union. Row 0: both parts empty; 1: first empty;
    2: second empty; 3: one row each; 4-7: maxima 30 apart; 8-9: 120 apart
    (the smaller scale underflows in FP32)."""
    generator = torch.Generator().manual_seed(11)
    counts = torch.randint(0, depth + 1, (2, rows), generator=generator)
    counts[:, :4] = torch.tensor([[0, 0, 3, 1], [0, 3, 0, 1]])
    shape = (rows, groups, width, depth)
    logits = [torch.randn(shape, generator=generator, dtype=torch.float64) * 3 for _ in range(2)]
    values = [torch.randn(shape, generator=generator, dtype=torch.float64) for _ in range(2)]
    logits[1][4:8] += 30.0
    logits[0][8:10] += 120.0
    valid = [(torch.arange(depth) < count[:, None]).view(rows, 1, 1, depth) for count in counts]
    parts = [_partial(*side) for side in zip(logits, values, valid)]
    union_valid = torch.cat(valid, -1)
    union = (torch.softmax(torch.cat(logits, -1).masked_fill(~union_valid, -_INF), -1)
             * torch.cat(values, -1)).sum(-1)
    union = torch.where(union_valid.any(-1), union, 0.0).flatten(1)
    own_first = torch.rand(rows, generator=generator) < 0.5
    return parts, union, own_first, width


@unittest.skipUnless(_CUDA, "requires coordinator-assigned idle CUDA device")
class MergeLsePartialsGPUTest(unittest.TestCase):
    def test_forward_matches_union_softmax_and_torch_reference(self):
        (first, second), union, own_first, width = _merge_inputs()
        first32, second32, own_first = first.float().cuda(), second.float().cuda(), own_first.cuda()
        out = merge_lse_partials(first32.requires_grad_(True), second32, width)
        self.assertIs(type(out.grad_fn), _MergeLsePartials._backward_cls)  # the fused kernel ran
        _assert_close(self, [("vs_union_fp64", out, union)], _TOL[torch.float32])
        # Same roundings as the eager torch expression (it is bitwise equal on sm_100).
        _assert_close(self, [("vs_torch", out, merge_lse_partials_torch(first32, second32, width))], 2.0**-22)
        with self.subTest(output="both_empty"):
            self.assertTrue(torch.equal(out[0], torch.zeros_like(out[0])))
        with self.subTest(output="both_ends"):
            # End "first" passes (own, peer) = (first, second), the other end
            # (second, first): both merge (first, second), to the same bits.
            ends = (merge_lse_partials(first32, second32, width, first_is_own=own_first),
                    merge_lse_partials(second32, first32, width, first_is_own=~own_first))
            canonical = (torch.where(own_first[:, None], first32, second32),
                         torch.where(own_first[:, None], second32, first32))
            self.assertTrue(torch.equal(ends[0], ends[1]))
            self.assertTrue(torch.equal(ends[0], merge_lse_partials(*canonical, width)))
        with self.subTest(output="row_strided_group_views"):
            # One group of a two-group operand: row stride 4 * width.
            views = [merge_lse_partials(first32[:, g * 2 * width:(g + 1) * 2 * width],
                                        second32[:, g * 2 * width:(g + 1) * 2 * width], width)
                     for g in range(2)]
            self.assertTrue(torch.equal(torch.cat(views, 1), out))

    def test_backward_and_create_graph_match_fp64(self):
        (first, second), _, own_first, width = _merge_inputs()
        generator = torch.Generator().manual_seed(12)
        seed = torch.randn(first.shape[0], first.shape[1] // 2, generator=generator, dtype=torch.float64)
        weights = [torch.randn(first.shape, generator=generator, dtype=torch.float64) for _ in range(2)]

        def orders(fn, device, dtype):
            """out, first-order grads (create_graph), d/d(first, second) of sum(weights * grads)."""
            leaves = [t.to(device, dtype).requires_grad_(True) for t in (first, second)]
            out = fn(*leaves, width, first_is_own=own_first.to(device))
            grads = torch.autograd.grad(out, leaves, seed.to(device, dtype), create_graph=True)
            second_order = torch.autograd.grad(
                sum((g * w.to(device, dtype)).sum() for g, w in zip(grads, weights)), leaves)
            return (*grads, *second_order)

        # Without create_graph: the Triton VJP.
        leaves = [t.float().cuda().requires_grad_(True) for t in (first, second)]
        out = merge_lse_partials(*leaves, width, first_is_own=own_first.cuda())
        got = torch.autograd.grad(out, leaves, seed.float().cuda())
        ref = orders(merge_lse_partials_torch, "cpu", torch.float64)
        _assert_close(self, zip(("grad_first", "grad_second"), got, ref[:2]), _TOL[torch.float32])
        with self.subTest(output="empty_parts_get_zero"):
            for grad in got:  # row 0: both parts empty
                self.assertTrue(torch.equal(grad[0], torch.zeros_like(grad[0])))
            lse = [grad.view(grad.shape[0], -1, 2, width)[:, :, 0] for grad in got]
            y = [grad.view(grad.shape[0], -1, 2, width)[:, :, 1] for grad in got]
            for row, empty in ((1, 0), (2, 1)):  # one part empty: w_1 * w_2 = 0 and w_empty = 0
                for tensor in (lse[0][row], lse[1][row], y[empty][row]):
                    self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))
        # With create_graph: the differentiable torch VJP, first and second order.
        got = orders(merge_lse_partials, "cuda", torch.float32)
        names = ("grad_first", "grad_second", "second_first", "second_second")
        _assert_close(self, zip(names, got, ref), _TOL[torch.float32])


if __name__ == "__main__":
    unittest.main()
