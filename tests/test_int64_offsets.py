"""64-bit Triton row offsets (K1): source lint and kernel parity.

CPU: a source lint over the migrated MatRIS/shared-engine/graph Triton kernels and int64 CSR
offsets. CUDA: small-shape parity of every converted kernel against ATen, using
stride-128 views, int32 strided indices and int32 CSR offsets (the in-kernel
widening paths); the atomic VJPs also on a padded tail indexing one private
sink row, with and without ``private_sink_atomics``, and in deterministic mode
(ordered VJP sums: bitwise on repeat, forward unchanged). Rows past
2**24..2**26 are checked by the opt-in ``test_triton_large_rows.py``.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import re
import unittest
from unittest.mock import MagicMock, patch
import warnings

import torch
import torch.nn.functional as F

MATRIS = Path(__file__).resolve().parents[1] / "fastmd"


class TritonOffsetSourceLintTest(unittest.TestCase):
    def test_every_row_program_id_is_widened_before_offset_math(self):
        # Axis-1 program ids tile feature columns (< D) and may stay int32. So may an axis-0 id marked
        # "int32 block index" (a block index or tile loop start, widened before the row offset math: the
        # int64 index math measured up to 30% slower in the C4 row GEMMs); the kernel must widen then.
        pattern = re.compile(r"tl\.program_id\(\s*(?:axis\s*=\s*)?0\s*\)(?!\.to\(tl\.int64\))(?!.*# int32 block index$)")
        kernels, offenders = 0, []
        # Match the upstream port's coverage; FastMD-only adapters (eSEN/ORB)
        # own different kernels and are covered by their adapter suites.
        roots = (MATRIS / "models" / "matris", MATRIS / "engine", MATRIS / "graph")
        for path in sorted(path for root in roots for path in root.rglob("*.py")):
            text = path.read_text()
            kernels += text.count("@triton.jit")
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(MATRIS.parent)}:{number}: {line.strip()}")
            if "# int32 block index" in text:
                self.assertIn(".to(tl.int64) * BM", text, path)
        self.assertGreater(kernels, 40)
        self.assertEqual(offenders, [])

    def test_csr_offsets_stay_int64(self):
        from fastmd.models.matris.op.triton_segment_attention import segment_attention_csr_metadata

        perm, off = segment_attention_csr_metadata(torch.tensor([2, 0, 2, 3], dtype=torch.int32), 4)
        self.assertEqual((perm.dtype, off.dtype), (torch.int64, torch.int64))
        self.assertEqual(off.tolist(), [0, 1, 1, 3, 4])


def _leaf(value):
    return value.detach().requires_grad_(True)


# (private sink, deterministic mode) of the atomic VJP kernels' tests.
_MODES = ((False, False), (True, False), (False, True), (True, True))


@contextmanager
def _deterministic(enabled=True, atomic_kernels=()):
    """torch.use_deterministic_algorithms(True, warn_only=True) when ``enabled``;
    the Triton kernels ``(module, name)`` of ``atomic_kernels`` must then not launch."""
    if not enabled:
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    launch = MagicMock()  # a Triton kernel's launch, a Function's apply or a launcher's call
    launch.side_effect = launch.__getitem__.side_effect = launch.apply.side_effect = AssertionError(
        "an atomic kernel launched in deterministic mode")
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with ExitStack() as stack:
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore")
            # ...but an op without a deterministic implementation would raise under the CLI's strict mode.
            warnings.filterwarnings("error", message=".*does not have a deterministic implementation.*")
            for module, name in atomic_kernels:
                stack.enter_context(patch.object(module, name, launch))
            yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def _bitwise(test, actual, expected):
    for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
        with test.subTest(output=index):
            test.assertTrue(torch.equal(a.view(torch.int32), e.view(torch.int32)))


def _grads(outputs, inputs, seeds):
    return torch.autograd.grad(outputs, inputs, seeds)


def _assert_all_close(test, actual, expected, atol=2e-5, rtol=2e-5):
    for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
        with test.subTest(output=index):
            torch.testing.assert_close(a, e, atol=atol, rtol=rtol)


def _reorder_bound(addends, index, rows):
    """How far two orders of the fp32 atomic sums ``out[index[r]] += addends[r]``
    can differ per element: 2 gamma_k sum|addends| over its k addends (Higham;
    any summation tree, so also the pre-added pairs of a sorted index)."""
    unit = torch.finfo(addends.dtype).eps / 2  # unit roundoff of the atomics
    index = index.long()
    count = torch.bincount(index, minlength=rows).double()[:, None]
    magnitude = addends.double().abs()
    magnitude = magnitude.new_zeros(rows, magnitude.shape[1]).index_add_(0, index, magnitude)
    return 2 * count * unit / (1 - count * unit) * magnitude


def _assert_same_sums(test, actual, expected, bounds, atol, rtol):
    # Same addends, another atomic order: each element may also differ by its bound.
    for index, (a, e, bound) in enumerate(zip(actual, expected, bounds, strict=True)):
        with test.subTest(output=index):
            diff, allowed = (a - e).abs(), atol + rtol * e.abs() + bound
            row, col = divmod(int((diff - allowed).argmax()), a.shape[1])  # a NaN is the argmax
            test.assertLessEqual(diff[row, col].item(), allowed[row, col].item(),
                                 f"at ({row}, {col}): {a[row, col].item()} vs {e[row, col].item()}")


@unittest.skipUnless(torch.cuda.is_available(), "requires coordinator-assigned idle CUDA device")
class Int64OffsetKernelGPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_decomposed_cat3_cat4_strided_p0_and_int32_line_graph_indices(self):
        from fastmd.engine.kernels import decomposed_indexed_cat_silu_linear as decomposed
        from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import (
            _Cat3GatherAddSilu, _Cat4GatherAddSilu)
        from fastmd.engine.kernels.sink_atomics import private_sink_atomics

        atomic = [(decomposed, "_cat3_gather_add_silu_bwd"), (decomposed, "_cat4_gather_add_silu_bwd")]

        m, e, n, pad = 1537, 211, 64, 263
        for parts, sorted_a, with_bias in ((3, True, False), (3, False, True), (4, True, True), (4, False, False)):
            # p0 is the 64-wide head of a [M,128] merged projection; the
            # indices are int32 columns of a [M,5] line-graph table. The last
            # `pad` rows are the padding of a private layout: all index the
            # sink row e - 1 (the largest, so a sorted index_a stays sorted).
            p0 = _leaf(torch.randn(m + pad, 128, device="cuda")[:, :n])
            tables = [_leaf(torch.randn(e, n, device="cuda")) for _ in range(parts - 1)]
            line = torch.randint(0, e - 1, (m + pad, 5), dtype=torch.int32, device="cuda")
            line[m:] = e - 1
            if sorted_a:
                line[:, 1] = line[:, 1].sort().values
            indices = [line[:, 1], line[:, 2], line[:, 4]][: parts - 1]
            bias = torch.randn(n, device="cuda") if with_bias else None
            function = _Cat3GatherAddSilu if parts == 3 else _Cat4GatherAddSilu
            # Padding rows get a zero cotangent (their sink atomics may be
            # skipped) or, as a control, a nonzero one (always added).
            for tail in ("zero", "nonzero"):
                pre = p0 + sum(t[i.long()] for t, i in zip(tables, indices))
                if bias is not None:
                    pre = pre + bias
                ref = (F.silu(pre[:, : n // 2]), F.silu(pre[:, n // 2:]))
                seeds = tuple(torch.randn_like(value) for value in ref)
                if tail == "zero":
                    for seed in seeds:
                        seed[m:] = 0
                expected = _grads(ref, (p0, *tables), seeds)
                grads, outputs = {}, {}
                for mode in _MODES:
                    private_sink, ordered = mode
                    with self.subTest(parts=parts, sorted_a=sorted_a, bias=with_bias, tail=tail,
                                      private_sink=private_sink, deterministic=ordered), \
                            _deterministic(ordered, atomic):
                        runs = []
                        for _ in range(2 if ordered else 1):
                            with private_sink_atomics(private_sink):
                                outputs[mode] = function.apply(p0, *tables, *indices, bias, sorted_a)
                            runs.append(_grads(outputs[mode], (p0, *tables), seeds))
                        grads[mode] = runs[0]
                        _assert_all_close(self, outputs[mode], ref, atol=1e-6)
                        _assert_all_close(self, grads[mode], expected)
                        if ordered:  # the forward kernels are unchanged; the ordered VJP repeats bitwise
                            _bitwise(self, outputs[mode], outputs[private_sink, False])
                            _bitwise(self, runs[1], runs[0])
                with self.subTest(parts=parts, sorted_a=sorted_a, tail=tail, check="same sums"):
                    # grad_p0 holds every addend of the table sums. The sink
                    # row sums `pad` of them to |sum| << sum|addends|, so the
                    # summation order alone can move it past the rtol.
                    bounds = [0.0] + [_reorder_bound(grads[False, False][0], i, e) for i in indices]
                    for mode in _MODES[1:]:
                        _assert_same_sums(self, grads[mode], grads[False, False], bounds, atol=1e-5, rtol=1e-5)

    def test_segment_attention_identity_int32_offsets_and_int32_perm(self):
        from fastmd.models.matris.op.triton_segment_attention import (
            _eager_segment_attention, paired_segment_attention, segment_attention)

        count, dim = 37, 32
        lengths = torch.randint(0, 9, (count,), device="cuda")
        lengths[5] = 0
        sorted_seg = torch.repeat_interleave(torch.arange(count, device="cuda"), lengths)
        rows = len(sorted_seg)
        off32 = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).int()
        other = torch.randint(0, count, (rows,), device="cuda")
        perm32 = torch.argsort(other).int()
        off_other32 = torch.searchsorted(other[perm32.long()], torch.arange(count + 1, device="cuda")).int()
        wide = torch.randn(rows, 128, device="cuda") * 2
        logits_a, logits_b = _leaf(wide[:, 64:96]), _leaf(wide[:, 96:])
        value = _leaf(torch.randn(rows, dim, device="cuda"))
        leaves = (logits_a, logits_b, value)
        seeds = (torch.randn(count, dim, device="cuda"), torch.randn(count, dim, device="cuda"))

        def reference():
            return (_eager_segment_attention(logits_a, value, sorted_seg, count),
                    _eager_segment_attention(logits_b, value, other, count))

        single = (segment_attention(logits_a, value, sorted_seg, count, None, off32, True),
                  segment_attention(logits_b, value, other, count, perm32, off_other32, False))
        paired = paired_segment_attention(logits_a, logits_b, value, sorted_seg, other, count,
                                          None, off32, perm32, off_other32, True, False)
        for outputs in (single, paired):
            expected = reference()
            _assert_all_close(self, outputs, expected)
            _assert_all_close(self, _grads(outputs, leaves, seeds), _grads(expected, leaves, seeds))

    def test_row_wise_layer_norm_kernels(self):
        from fastmd.models.matris.op import triton_gated_ln
        from fastmd.models.matris.op.triton_gated_ln import fused_gated_ln, fused_gated_ln_residual
        from fastmd.models.matris.op.triton_layernorm_act import ACT_SIGMOID, ACT_SILU, fused_ln_act

        rows, dim = 1029, 32
        core, gate, residual = (_leaf(torch.randn(rows, dim, device="cuda")) for _ in range(3))
        cw, cb, gw, gb, rw = (torch.randn(dim, device="cuda") for _ in range(5))

        def reference(c, g):
            return F.silu(F.layer_norm(c, (dim,), cw, cb)) * torch.sigmoid(F.layer_norm(g, (dim,), gw, gb))

        # gated_ln: 1-row programs and 2-D row tiles (D7).
        for tile_2d in (False, True):
            with self.subTest(tile_2d=tile_2d), patch.object(triton_gated_ln, "_TILE_2D", tile_2d):
                outputs = (fused_gated_ln(core, gate, cw, cb, gw, gb),
                           fused_gated_ln_residual(core, gate, cw, cb, gw, gb, residual, rw),
                           fused_ln_act(core, cw, cb, 1e-5, ACT_SILU),
                           fused_ln_act(gate, gw, gb, 1e-5, ACT_SIGMOID))
                expected = (reference(core, gate), reference(core, gate) + rw * residual,
                            F.silu(F.layer_norm(core, (dim,), cw, cb)),
                            torch.sigmoid(F.layer_norm(gate, (dim,), gw, gb)))
                _assert_all_close(self, outputs, expected)
                seeds = tuple(torch.randn_like(value) for value in outputs)
                _assert_all_close(self, _grads(outputs, (core, gate, residual), seeds),
                                  _grads(expected, (core, gate, residual), seeds))

    def test_gather_kernels_with_int32_indices(self):
        from fastmd.models.matris.op import triton_gather_cat, triton_line_envelope, triton_threebody_fourier
        from fastmd.models.matris.op.triton_directed_pair_aggregate import (
            directed_pair_average, undirected_pair_expand)
        from fastmd.models.matris.op.triton_gather_cat import aligned_gather_cat3, aligned_gather_cat4
        from fastmd.models.matris.op.triton_line_envelope import line_envelope_product
        from fastmd.models.matris.op.triton_threebody_fourier import threebody_fourier_basis
        from fastmd.engine.kernels.sink_atomics import private_sink_atomics

        atomic = [(triton_gather_cat, "_aligned_gather_cat3_bwd"), (triton_gather_cat, "_aligned_gather_cat4_bwd"),
                  (triton_line_envelope, "_line_envelope_bwd"), (triton_threebody_fourier, "_threebody_fourier_bwd")]

        rows, source_rows, dim, pad = 1031, 97, 32, 257
        # The last `pad` rows are the padding of a private layout: every index
        # is the sink row, which no real row uses.
        sink = source_rows - 1
        aligned = _leaf(torch.randn(rows + pad, dim, device="cuda"))
        table_a, table_b = (_leaf(torch.randn(source_rows, dim, device="cuda")) for _ in range(2))
        ia, ib, ic = (torch.randint(0, sink, (rows + pad,), dtype=torch.int32, device="cuda")
                      for _ in range(3))
        for index in (ia, ib, ic):
            index[rows:] = sink
        u = 211
        pair = torch.randperm(2 * u, device="cuda").reshape(u, 2).int()
        directed = _leaf(torch.randn(2 * u, dim, device="cuda"))
        undirected = _leaf(torch.randn(u, dim, device="cuda"))
        segment = torch.empty(2 * u, dtype=torch.long, device="cuda")
        segment[pair.long().reshape(-1)] = torch.arange(u, device="cuda").repeat_interleave(2)
        vectors = F.normalize(torch.randn(source_rows, 3, device="cuda"), dim=1)
        vectors[sink] = torch.tensor([0.6, 0.0, 0.0])  # padding triplets: a well-conditioned angle
        unit = _leaf(vectors)
        source = torch.where((ib == ia) & (ia != sink), (ib + 1) % sink, ib)
        freqs = torch.tensor([0.833447, 1.5861181, 2.5042067], device="cuda")
        scale = torch.pi ** 0.5

        def fourier(vectors):
            theta = (vectors[ia.long()] * vectors[source.long()]).sum(1) * (1 - 1e-6)
            angle = torch.acos(theta)[:, None] * freqs
            first = torch.full_like(angle[:, :1], 2 ** -0.5)
            return torch.cat((first, angle.sin(), angle.cos()), 1) / scale

        leaves = (aligned, table_a, table_b, directed, undirected, unit)
        # Padding rows get a zero cotangent (their sink atomics may be skipped)
        # or, as a control, a nonzero one (always added).
        for tail in ("zero", "nonzero"):
            expected = (torch.cat((aligned, table_a[ia.long()], table_a[ib.long()]), 1),
                        torch.cat((aligned, table_a[ia.long()], table_b[ib.long()], table_b[ic.long()]), 1),
                        table_a[ia.long()] * table_a[ib.long()],
                        directed.new_zeros(u, dim).index_add(0, segment, directed) * 0.5,
                        undirected[segment],
                        fourier(unit))
            seeds = tuple(torch.randn_like(value) for value in expected)
            if tail == "zero":
                for seed in seeds:
                    if seed.shape[0] == rows + pad:
                        seed[rows:] = 0
            reference = _grads(expected, leaves, seeds)
            grads, outputs = {}, {}
            for mode in _MODES:
                private_sink, ordered = mode
                with self.subTest(tail=tail, private_sink=private_sink, deterministic=ordered), \
                        _deterministic(ordered, atomic):
                    runs = []
                    for _ in range(2 if ordered else 1):
                        with private_sink_atomics(private_sink):
                            outputs[mode] = (aligned_gather_cat3(aligned, table_a, ia, ib),
                                             aligned_gather_cat4(aligned, table_a, ia, table_b, ib, ic),
                                             line_envelope_product(table_a, ia, ib),
                                             directed_pair_average(directed, pair),
                                             undirected_pair_expand(undirected, pair, 2 * u),
                                             threebody_fourier_basis(unit, ia, source, freqs, scale))
                        runs.append(_grads(outputs[mode], leaves, seeds))
                    grads[mode] = runs[0]
                    _assert_all_close(self, outputs[mode], expected, atol=1e-5, rtol=1e-4)
                    _assert_all_close(self, grads[mode], reference, atol=1e-4, rtol=1e-4)
                    if ordered:  # the forward kernels are unchanged; the ordered VJPs repeat bitwise
                        _bitwise(self, outputs[mode], outputs[private_sink, False])
                        _bitwise(self, runs[1], runs[0])
            with self.subTest(tail=tail, check="same sums"):
                for mode in _MODES[1:]:
                    _assert_all_close(self, grads[mode], grads[False, False], atol=1e-4, rtol=1e-4)

    def test_elementwise_and_segment_reduction_kernels(self):
        from fastmd.models.matris.op.triton_residual_add import residual_add
        from fastmd.engine.kernels.triton_segment_softmax import (
            _eager, fused_segment_softmax, fused_segment_softmax_atomic)
        from fastmd.models.matris.op.triton_weighted_segment_sum import weighted_segment_sum

        rows, dim, count = 1033, 32, 29
        delta, residual, weight, value, feats = (
            _leaf(torch.randn(rows, dim, device="cuda")) for _ in range(5))
        scale = torch.randn(1, dim, device="cuda")
        segment = torch.randint(0, count, (rows,), device="cuda")
        leaves = (delta, residual, weight, value, feats)
        outputs = (residual_add(delta, residual, scale),
                   weighted_segment_sum(weight, value, segment, count),
                   fused_segment_softmax(feats, segment, count),
                   fused_segment_softmax_atomic(feats, segment, count))
        expected = (delta + scale * residual,
                    value.new_zeros(count, dim).index_add(0, segment, weight * value),
                    _eager(feats, segment, count),
                    _eager(feats, segment, count))
        _assert_all_close(self, outputs, expected, atol=1e-5, rtol=1e-4)
        seeds = tuple(torch.randn_like(value) for value in outputs)
        _assert_all_close(self, _grads(outputs, leaves, seeds), _grads(expected, leaves, seeds),
                          atol=1e-5, rtol=1e-4)
        # Deterministic mode: the dispatcher declines the atomic weighted segment sum
        # (selected in the fusion scope otherwise).
        from fastmd.engine.dispatch import current_policy, dispatcher_for
        from fastmd.engine.layout import ExecutionLayout
        from fastmd.engine.policy import fusion_scope
        from fastmd.models.matris.op import triton_weighted_segment_sum

        dispatcher = dispatcher_for(current_policy(), ExecutionLayout())
        with fusion_scope():
            self.assertTrue(dispatcher.use_weighted_segment_sum(weight, value))
        with fusion_scope(), _deterministic(atomic_kernels=[(triton_weighted_segment_sum, "_WeightedSegmentSum")]):
            self.assertFalse(dispatcher.use_weighted_segment_sum(weight, value))
            runs = [dispatcher.segment_sum(value, segment, count, weight=weight) for _ in range(2)]
            _assert_all_close(self, runs[:1], expected[1:2], atol=1e-5, rtol=1e-4)
            _bitwise(self, runs[1:], runs[:1])


if __name__ == "__main__":
    unittest.main()
