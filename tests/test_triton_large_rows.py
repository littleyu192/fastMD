"""Opt-in large-row CUDA checks for 64-bit Triton offsets (K1).

Tier 2 (tests/matris_test_tiers.py): run with ``MATRIS_RUN_LARGE_ROW_TESTS=1``
on an otherwise idle GB300-class GPU (peak about 110 GiB). Row counts sit just below and just above the points
where a 32-bit ``rows * stride`` wraps: 2**24 for stride-128 operands (cat3
p0, merged attention logits), 2**25 for the 64-wide preactivation and cat
backward, and 2**26 for 32-wide rows (gated LN, residual add, line envelope).
Every result is compared chunk by chunk with an ATen reference, so the
reference never materializes a full-size temporary.
"""
from __future__ import annotations

import gc
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from matris_test_tiers import large_row_tier

_CHUNK = 1 << 22
# 65,539 rows past each wrap point: the pre-fix cat3 kernel silently returned
# wrong rows in (2**24, ~17.0M] and faulted beyond.
BELOW_24, ABOVE_24 = (1 << 24) - 3, (1 << 24) + 65_539
BELOW_25, ABOVE_25 = (1 << 25) - 3, (1 << 25) + 65_539
BELOW_26, ABOVE_26 = (1 << 26) - 3, (1 << 26) + 65_539


def _leaf(value):
    return value.detach().requires_grad_(True)


def _release():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class _Checker:
    """Accumulate max errors by name; report the first failing row."""

    def __init__(self, test, atol, rtol):
        self.test, self.atol, self.rtol, self.failures, self.max_err = test, atol, rtol, [], {}

    @torch.no_grad()
    def __call__(self, name, actual, expected, start):
        error = (actual.float() - expected.float()).abs()
        bound = self.atol + self.rtol * expected.float().abs()
        bad = ~(error <= bound)  # NaN counts as failure
        self.max_err[name] = max(self.max_err.get(name, 0.0), float(error.nan_to_num(float("inf")).max()))
        if bool(bad.any()):
            row = int(bad.reshape(len(bad), -1).any(1).nonzero()[0]) + start
            self.failures.append(f"{name}: first bad row {row}, max error {self.max_err[name]:.3e}")

    def finish(self, label):
        print(f"[large-rows] {label} max_abs_err={self.max_err}", flush=True)
        self.test.assertEqual(self.failures, [], label)


@large_row_tier
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LargeRowTritonTest(unittest.TestCase):
    def setUp(self):
        _release()
        torch.manual_seed(20260927)

    tearDown = setUp

    def _cat(self, parts, rows, table_rows=1 << 20, index_offset=0, hidden=32):
        from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import (
            _Cat3GatherAddSilu, _Cat4GatherAddSilu)

        n, h = 2 * hidden, hidden
        # p0: n-wide head of a [M, n + 64] merged projection (SP0 = n + 64).
        p0 = _leaf(torch.empty(rows, n + 64, device="cuda").uniform_(-1, 1)[:, :n])
        tables = [_leaf(torch.randn(table_rows, n, device="cuda")) for _ in range(parts - 1)]
        # int32 columns of a [M,5] line-graph-like table; index_a sorted.
        line = torch.randint(index_offset, table_rows, (rows, 5), dtype=torch.int32, device="cuda")
        line[:, 1] = line[:, 1].sort().values
        indices = [line[:, 1], line[:, 2], line[:, 4]][: parts - 1]
        bias = torch.randn(n, device="cuda")
        function = _Cat3GatherAddSilu if parts == 3 else _Cat4GatherAddSilu
        core, gate = function.apply(p0, *tables, *indices, bias, True)
        gcore, ggate = torch.randn_like(core), torch.randn_like(gate)
        grads = torch.autograd.grad((core, gate), (p0, *tables), (gcore, ggate))
        torch.cuda.synchronize()
        check = _Checker(self, atol=2e-5, rtol=1e-5)
        ref_tables = [torch.zeros_like(t) for t in tables]
        with torch.no_grad():
            for start in range(0, rows, _CHUNK):
                rows_slice = slice(start, min(rows, start + _CHUNK))
                pre = p0[rows_slice] + bias
                for table, index in zip(tables, indices):
                    pre = pre + table[index[rows_slice].long()]
                c, g = pre[:, :h], pre[:, h:]
                check("core", core[rows_slice], F.silu(c), start)
                check("gate", gate[rows_slice], F.silu(g), start)
                sc, sg = torch.sigmoid(c), torch.sigmoid(g)
                d = torch.cat((gcore[rows_slice] * sc * (1 + c * (1 - sc)),
                               ggate[rows_slice] * sg * (1 + g * (1 - sg))), 1)
                check("grad_p0", grads[0][rows_slice], d, start)
                for ref, index in zip(ref_tables, indices):
                    ref.index_add_(0, index[rows_slice].long(), d)
            for k, (actual, ref) in enumerate(zip(grads[1:], ref_tables), 1):
                # Atomic accumulation order differs from index_add_.
                check_tables = _Checker(self, atol=2e-4, rtol=1e-4)
                check_tables(f"grad_p{k}", actual, ref, 0)
                check.failures += check_tables.failures
                check.max_err.update(check_tables.max_err)
        check.finish(f"cat{parts} rows={rows} table_rows={table_rows}")

    def test_cat_forward_backward_around_2p24_and_2p25(self):
        for parts in (3, 4):
            for rows in (BELOW_24, ABOVE_24, BELOW_25, ABOVE_25):
                with self.subTest(parts=parts, rows=rows):
                    self._cat(parts, rows)
                    _release()

    def test_cat3_gathered_index_times_width_past_2p31(self):
        # idx * N with N = 64 wraps for table rows >= 2**25.
        self._cat(3, 1 << 16, table_rows=(1 << 25) + 4099, index_offset=(1 << 25) - 4099)
        _release()
        # D8's atom rows [32|128|128] project the node table to N = 2H = 256: wraps from 2**23 rows.
        self._cat(3, 1 << 16, table_rows=(1 << 23) + 4099, index_offset=(1 << 23) - 4099, hidden=128)

    def _paired(self, rows):
        from fastmd.models.matris.op.triton_segment_attention import (
            _eager_segment_attention, paired_segment_attention)

        dim, len_a, len_b = 32, 64, 48
        count = -(-rows // len_b)
        # Merged alpha layout: two 32-wide views of a [M,128] tensor (stride 128).
        wide = torch.empty(rows, 128, device="cuda").uniform_(-3, 3)
        logits_a, logits_b = _leaf(wide[:, 64:96]), _leaf(wide[:, 96:])
        value = _leaf(torch.randn(rows, dim, device="cuda"))
        row_ids = torch.arange(rows, device="cuda")
        seg_a, seg_b = row_ids // len_a, row_ids // len_b
        # Side a: identity CSR with the legacy int32 offsets; side b: perm CSR.
        off_a = torch.clamp(torch.arange(count + 1, device="cuda") * len_a, max=rows).int()
        out_a, out_b = paired_segment_attention(logits_a, logits_b, value, seg_a, seg_b, count,
                                                None, off_a, None, None, True, False)
        gout_a, gout_b = torch.randn_like(out_a), torch.randn_like(out_b)
        grads = torch.autograd.grad((out_a, out_b), (logits_a, logits_b, value), (gout_a, gout_b))
        torch.cuda.synchronize()
        del row_ids
        check = _Checker(self, atol=2e-5, rtol=1e-4)
        step = _CHUNK // (len_a * len_b) * (len_a * len_b)  # both segmentations align
        for start in range(0, rows, step):
            rows_slice = slice(start, min(rows, start + step))
            la, lb, v = (_leaf(x[rows_slice]) for x in (logits_a, logits_b, value))
            sa, sb = seg_a[rows_slice] - start // len_a, seg_b[rows_slice] - start // len_b
            na, nb = int(sa[-1]) + 1, int(sb[-1]) + 1
            ra = _eager_segment_attention(la, v, sa, na)
            rb = _eager_segment_attention(lb, v, sb, nb)
            a0, b0 = start // len_a, start // len_b
            check("out_a", out_a[a0:a0 + na], ra, a0)
            check("out_b", out_b[b0:b0 + nb], rb, b0)
            ref = torch.autograd.grad((ra, rb), (la, lb, v),
                                      (gout_a[a0:a0 + na], gout_b[b0:b0 + nb]))
            for name, actual, expected in zip(("grad_logits_a", "grad_logits_b", "grad_value"),
                                              grads, ref):
                check(name, actual[rows_slice], expected, start)
        check.finish(f"paired attention rows={rows}")

    def test_paired_segment_attention_forward_backward_around_2p24_and_2p25(self):
        for rows in (BELOW_24, ABOVE_24, BELOW_25, ABOVE_25):
            with self.subTest(rows=rows):
                self._paired(rows)
                _release()

    def _row_wise(self, rows):
        from fastmd.models.matris.op.triton_line_envelope import line_envelope_product
        from fastmd.models.matris.op.triton_residual_add import residual_add

        dim, base_rows = 32, 1 << 20
        core, gate, seed = (torch.randn(rows, dim, device="cuda") for _ in range(3))
        rw = torch.randn(dim, device="cuda")
        added = residual_add(core, gate, rw)
        base = _leaf(torch.randn(base_rows, dim, device="cuda"))
        src, dst = (torch.randint(0, base_rows, (rows,), dtype=torch.int32, device="cuda")
                    for _ in range(2))
        envelope = line_envelope_product(base, src, dst)
        base_grad, = torch.autograd.grad(envelope, base, seed)
        torch.cuda.synchronize()
        check = _Checker(self, atol=3e-5, rtol=1e-5)
        ref_base = torch.zeros_like(base)
        for start in range(0, rows, _CHUNK):
            rows_slice = slice(start, min(rows, start + _CHUNK))
            s = seed[rows_slice]
            with torch.no_grad():
                check("residual_add", added[rows_slice], core[rows_slice] + rw * gate[rows_slice], start)
                a, b = base[src[rows_slice].long()], base[dst[rows_slice].long()]
                check("line_envelope", envelope[rows_slice], a * b, start)
                ref_base.index_add_(0, src[rows_slice].long(), s * b)
                ref_base.index_add_(0, dst[rows_slice].long(), s * a)
        check_base = _Checker(self, atol=5e-4, rtol=1e-4)
        check_base("line_envelope_grad", base_grad, ref_base, 0)
        check.failures += check_base.failures
        check.max_err.update(check_base.max_err)
        check.finish(f"row-wise x32 rows={rows}")

    def _gated_ln(self, rows, tile_2d):
        from fastmd.models.matris.op import triton_gated_ln
        from fastmd.models.matris.op.triton_gated_ln import fused_gated_ln, fused_gated_ln_residual

        dim = 32
        core, gate = (_leaf(torch.randn(rows, dim, device="cuda")) for _ in range(2))
        cw, cb, gw, gb, rw = (torch.randn(dim, device="cuda") for _ in range(5))
        with patch.object(triton_gated_ln, "_TILE_2D", tile_2d):
            out = fused_gated_ln_residual(core, gate, cw, cb, gw, gb, core, rw)
            seed = torch.randn_like(out)
            grads = torch.autograd.grad(out, (core, gate), seed)
            plain = fused_gated_ln(core, gate, cw, cb, gw, gb)
            plain_grads = torch.autograd.grad(plain, (core, gate), seed)
        torch.cuda.synchronize()
        check = _Checker(self, atol=3e-5, rtol=1e-5)
        for start in range(0, rows, _CHUNK):
            rows_slice = slice(start, min(rows, start + _CHUNK))
            c, g = _leaf(core[rows_slice]), _leaf(gate[rows_slice])
            ref_plain = (F.silu(F.layer_norm(c, (dim,), cw, cb))
                         * torch.sigmoid(F.layer_norm(g, (dim,), gw, gb)))
            ref = ref_plain + rw * c
            s = seed[rows_slice]
            ref_grads = torch.autograd.grad(ref, (c, g), s, retain_graph=True)
            ref_plain_grads = torch.autograd.grad(ref_plain, (c, g), s)
            check("gated_ln_residual", out[rows_slice], ref.detach(), start)
            check("gated_ln", plain[rows_slice], ref_plain.detach(), start)
            for name, actual, expected in zip(("grad_core_res", "grad_gate_res", "grad_core", "grad_gate"),
                                              (*grads, *plain_grads), (*ref_grads, *ref_plain_grads)):
                check(name, actual[rows_slice], expected, start)
        check.finish(f"gated_ln x32 rows={rows} tile_2d={tile_2d}")

    def test_row_wise_x32_kernels_around_2p25_and_2p26(self):
        for rows in (BELOW_25, ABOVE_25, BELOW_26, ABOVE_26):
            with self.subTest(rows=rows):
                self._row_wise(rows)
                _release()
            # gated_ln: 1-row programs and 2-D row tiles (D7).
            for tile_2d in (False, True):
                with self.subTest(rows=rows, tile_2d=tile_2d):
                    self._gated_ln(rows, tile_2d)
                    _release()


if __name__ == "__main__":
    unittest.main()
