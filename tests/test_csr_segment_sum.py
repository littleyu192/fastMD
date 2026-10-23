"""Deterministic CSR (weighted) segment sum (step 2.5a, D6): accuracy vs FP64.

Forward and VJP against an FP64 ``index_add`` reference for identity and
permutation CSRs, weighted and unweighted, with and without a skipped segment
(NaN padding rows must not leak), odd widths, empty segments, int32 indices,
row-strided value/weight, segment-column views and both VJP kernels; bitwise
determinism; the CPU path.
The int64 row-offset case (rows * width > 2**31) is opt-in with
``MATRIS_RUN_LARGE_ROW_TESTS=1``.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from fastmd.models.matris.op import triton_csr_segment_sum as csr_mod
from fastmd.models.matris.op.triton_csr_segment_sum import csr_segment_sum
from fastmd.models.matris.op.triton_segment_attention import segment_attention_csr_metadata
from matris_test_tiers import large_row_tier

_CUDA = torch.cuda.is_available()


def _reference(value, weight, segment, num_segment, skip_segment=-1):
    rows = value.double() if weight is None else weight.double() * value.double()
    if skip_segment != -1:
        rows = torch.where((segment == skip_segment)[:, None], 0.0, rows)
    return rows.new_zeros((num_segment, rows.shape[1])).index_add_(0, segment.long(), rows)


def _case(n_rows, width, num_segment, *, identity, weighted, skip, device, seed=0, strided=False):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Segments drawn from the lower 3/4 only: the upper quarter stays empty.
    segment = torch.randint(0, max(3 * num_segment // 4, 1), (n_rows,), generator=gen)
    if identity:
        segment = segment.sort().values
    base = torch.randn(n_rows, 2 * width if strided else width, generator=gen)
    value = base[:, :width] if strided else base
    weight = torch.rand(n_rows, width, generator=gen) + 0.5 if weighted else None
    if skip >= 0:
        # The skipped segment is the private sink of NaN padding rows.
        pad = torch.arange(n_rows) % 11 == 3
        segment = torch.where(pad, skip, segment)
        if identity:
            segment = segment.sort().values
            pad = segment == skip
        value[pad] = float("nan")
        if weighted:
            weight[pad] = float("nan")
    value = value.to(device)
    weight = None if weight is None else weight.to(device)
    segment = segment.to(device)
    if strided:
        # A column of an edge-index matrix, as process_graphs passes it (not copied).
        segment = torch.stack((torch.zeros_like(segment), segment), 1)[:, 1]
    return value, weight, segment


class CsrSegmentSumCPUTest(unittest.TestCase):
    def test_torch_path_matches_reference_and_zeroes_skipped_rows(self):
        value, weight, segment = _case(97, 5, 13, identity=False, weighted=True, skip=12, device="cpu")
        value.requires_grad_(True)
        weight.requires_grad_(True)
        out = csr_segment_sum(value, segment, 13, weight=weight, skip_segment=12)
        clean_v = value.detach().nan_to_num(0.0).double().requires_grad_(True)
        clean_w = weight.detach().nan_to_num(0.0).double().requires_grad_(True)
        ref = _reference(clean_v, clean_w, segment, 13, 12)
        torch.testing.assert_close(out.double(), ref, atol=1e-6, rtol=1e-6)
        gout = torch.randn_like(out)
        gv, gw = torch.autograd.grad(out, (value, weight), gout)
        rv, rw = torch.autograd.grad(ref, (clean_v, clean_w), gout.double())
        torch.testing.assert_close(gv.double(), rv, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(gw.double(), rw, atol=1e-6, rtol=1e-6)
        self.assertEqual(out[12].abs().sum().item(), 0.0)
        with self.assertRaisesRegex(ValueError, "skip_segment"):
            csr_segment_sum(value, segment, 13, skip_segment=13)


@unittest.skipUnless(_CUDA, "requires CUDA")
class CsrSegmentSumGPUTest(unittest.TestCase):
    CASES = [
        # (rows, width, segments, identity, weighted, skip, strided, int32 indices)
        (1000, 32, 64, False, True, -1, False, False),
        (1000, 32, 64, True, True, -1, False, True),
        (3001, 17, 97, False, False, 5, False, True),
        (3001, 128, 97, True, True, 96, True, False),
        (3001, 17, 97, False, True, 5, True, False),
        (4099, 32, 40, True, False, -1, True, False),      # long runs (~100 rows per segment)
        (4099, 32, 40, False, True, 0, False, False),
        (37, 32, 50, False, True, -1, False, False),       # fewer rows than segments
    ]

    def _check(self, n_rows, width, num_segment, identity, weighted, skip, strided, int32, provide_csr):
        value, weight, segment = _case(n_rows, width, num_segment, identity=identity, weighted=weighted,
                                       skip=skip, device="cuda", strided=strided)
        if int32:
            segment = segment.to(torch.int32)
        value.requires_grad_(True)
        if weighted:
            weight.requires_grad_(True)
        perm = off = None
        if provide_csr or identity:
            perm, off = segment_attention_csr_metadata(segment, num_segment)
            if int32:
                perm = perm.to(torch.int32)
            if identity:
                perm = None
        out = csr_segment_sum(value, segment, num_segment, perm, off, identity,
                              weight=weight, skip_segment=skip)
        inputs = (value,) if weight is None else (value, weight)
        clean = [x.detach().nan_to_num(0.0).double().requires_grad_(True) for x in inputs]
        ref = _reference(clean[0], clean[1] if weighted else None, segment, num_segment, skip)
        torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-5)
        gout = torch.randn_like(out)
        grads = torch.autograd.grad(out, inputs, gout)
        ref_grads = torch.autograd.grad(ref, clean, gout.double())
        for grad, ref_grad in zip(grads, ref_grads):
            self.assertTrue(torch.isfinite(grad).all())
            torch.testing.assert_close(grad.double(), ref_grad, atol=1e-6, rtol=1e-6)
        if skip >= 0:
            self.assertEqual(out[skip].abs().sum().item(), 0.0)
            pad = segment == skip
            for grad in grads:
                self.assertEqual(grad[pad].abs().sum().item(), 0.0)
        return out

    def test_matches_fp64_reference(self):
        for case in self.CASES:
            for rows_vjp in (False, True):
                with self.subTest(case=case, rows_vjp=rows_vjp), \
                        patch.object(csr_mod, "_BWD_ROWS_IDENTITY", rows_vjp), \
                        patch.object(csr_mod, "_BWD_ROWS_PERM", rows_vjp):
                    self._check(*case, provide_csr=rows_vjp)

    def test_bitwise_deterministic(self):
        n_rows, width, num_segment = 200_003, 32, 1500
        for identity in (False, True):
            value, weight, segment = _case(n_rows, width, num_segment, identity=identity, weighted=True,
                                           skip=-1, device="cuda", seed=1)
            value.requires_grad_(True)
            weight.requires_grad_(True)
            perm, off = segment_attention_csr_metadata(segment, num_segment)
            runs = []
            for _ in range(2):
                out = csr_segment_sum(value, segment, num_segment, None if identity else perm, off,
                                      identity, weight=weight)
                gout = torch.linspace(-1, 1, out.numel(), device="cuda").reshape(out.shape)
                runs.append((out, *torch.autograd.grad(out, (value, weight), gout)))
            with self.subTest(identity=identity):
                for first, second in zip(*runs):
                    self.assertTrue(torch.equal(first, second))


@large_row_tier
@unittest.skipUnless(_CUDA, "requires CUDA")
class CsrSegmentSumLargeRowTest(unittest.TestCase):
    def test_rows_times_width_past_2_31(self):
        # 2**26 + 65,539 rows x 32: element offsets pass 2**31 (fp32, ~8.6 GB per operand).
        n_rows, width, per_segment = (1 << 26) + 65_539, 32, 64
        num_segment = -(-n_rows // per_segment)
        index = torch.arange(n_rows, device="cuda")
        value = (torch.remainder(index, 1013).float() * 1e-3)[:, None].expand(n_rows, width).contiguous()
        weight = torch.full((n_rows, width), 0.5, device="cuda")
        for identity in (True, False):
            segment = index // per_segment if identity else torch.remainder(index * 7919, num_segment)
            perm, off = segment_attention_csr_metadata(segment, num_segment)
            value.requires_grad_(True)
            weight.requires_grad_(True)
            out = csr_segment_sum(value, segment, num_segment, None if identity else perm, off,
                                  identity, weight=weight)
            with torch.no_grad():
                expected = torch.zeros_like(out).index_add_(0, segment, weight * value)
            with self.subTest(identity=identity):
                torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-5)
                gout = torch.remainder(torch.arange(out.numel(), device="cuda"), 7).float().reshape(out.shape)
                gv, gw = torch.autograd.grad(out, (value, weight), gout)
                with torch.no_grad():
                    for start in range(0, n_rows, 1 << 24):
                        rows = slice(start, start + (1 << 24))
                        go = gout[segment[rows]]
                        torch.testing.assert_close(gv[rows], go * weight[rows], atol=0, rtol=0)
                        torch.testing.assert_close(gw[rows], go * value[rows], atol=0, rtol=0)
            del out, gv, gw, perm, off, segment
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
