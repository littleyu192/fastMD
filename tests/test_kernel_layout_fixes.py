"""Step-2.1 kernel layout contract (docs/unified_engine_design.md 2.6, 4.5, 8.1).

1. Row stride: pair average/expand and line envelope accept row-strided views
   (stride(1) == 1, stride(0) != D), e.g. ``packed[:, :D]`` of ``[R, 3D]``.
2. Pair bijection: ``undirected_pair_expand`` rejects ``num_directed != 2U``;
   ``pair_index_bijection_reason`` is the host-side admission check.
3. Paired CSR attention accepts ``skip_segment`` with segment_attention's
   semantics and still matches plain torch without it.
4. The SORTED_A pair backward is correct for any power-of-two backward row
   tile, including 1; the registry rejects non-power-of-two tiles. The row
   kernels' tiles follow the feature width (C1), a pinned option or the former
   fixed tiles under ``ResolvedConfig.scope``, each matching torch. The merged
   projection's consumers write their gradients into one buffer and its
   backward is one GEMM (C3); dead, repeated or unsupported consumers and
   CUDA-graph capture keep it correct.
5. K5: row counts are runtime arguments, so new row counts never compile
   (a JIT recompile count, i.e. a structural perf guard: opt-in perf tier,
   ``MATRIS_RUN_PERF_TIER=1``).
6. Checkpoint wrappers pass ``preserve_rng_state=self.training``, and a
   captured checkpointed eval block replays like eager.
"""
from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from matris_test_tiers import perf_tier

_CUDA = torch.cuda.is_available()
_SEED = 20260927
# Warm-up covers one class only (not 1, not divisible by 16) and the new row
# counts include 1 and multiples of 16: a plain runtime int still specializes
# on ==1 and %16==0 and would recompile for them, so only a count excluded
# from value specialization (do_not_specialize) passes.
_WARM_ROWS = (17,)
_NEW_ROWS = (1, 19, 23, 32, 48, 1001, 4103)
_BWD_BM_OPTION = "MATRIS_DECOMPOSED_INDEXED_SILU_BWD_BM"


def _leaf(value):
    return value.detach().clone().requires_grad_(True)


def _generator(offset=0):
    return torch.Generator().manual_seed(_SEED + offset)


def _pair_d2u(pair_index, num_directed):
    """directed2undirected for a pair bijection: d2u[pair_index[u, k]] = u."""
    d2u = torch.empty(num_directed, dtype=torch.long, device=pair_index.device)
    u = torch.arange(pair_index.shape[0], device=pair_index.device)
    d2u[pair_index[:, 0].long()] = u
    d2u[pair_index[:, 1].long()] = u
    return d2u


def _random_pair_index(num_undirected, *, interleaved, generator, device="cpu"):
    if interleaved:
        pair = torch.arange(2 * num_undirected).reshape(num_undirected, 2)
    else:
        pair = torch.randperm(2 * num_undirected, generator=generator).reshape(num_undirected, 2)
    return pair.to(device)


def _poison_cuda_cache(*shapes, copies=8):
    """Leave NaN-filled freed blocks in the caching allocator.

    A later ``torch.empty`` of a similar size then reuses NaN memory, so reads
    of uninitialized statistics or adjoints show up as NaN deterministically.
    """
    blocks = [torch.full(shape, float("nan"), device="cuda") for shape in shapes for _ in range(copies)]
    del blocks


def _torch_segment_attention(logits, value, segment, num_segment):
    """Plain segmented softmax attention; empty segments give 0."""
    dim = logits.shape[1]
    expanded = segment.long().unsqueeze(1).expand(-1, dim)
    peak = logits.new_full((num_segment, dim), float("-inf")).scatter_reduce(
        0, expanded, logits, reduce="amax", include_self=False)
    weight = (logits - peak[segment.long()]).exp()
    denom = logits.new_zeros((num_segment, dim)).index_add(0, segment.long(), weight)
    return logits.new_zeros((num_segment, dim)).index_add(0, segment.long(), weight / denom[segment.long()] * value)


def _csr(segment, num_segment):
    """Stable CSR metadata (permutation, int64 offsets) computed on the host."""
    segment = segment.long().cpu()
    counts = torch.bincount(segment, minlength=num_segment)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return torch.argsort(segment, stable=True), offsets


class _TritonCompileCounter:
    """Count Triton JIT specializations compiled while active (Triton 3.7).

    A chained ``triton.knobs.runtime.jit_post_compile_hook`` sees every JIT
    compile in the process, including helper kernels a fix may add; diffing
    the per-device kernel caches of the named kernels attributes them. Both
    count in-memory specializations, whether or not the disk cache is warm.
    The previous global hook is always restored.
    """

    def __init__(self, kernels):
        self.kernels = dict(kernels)
        self.compiled = []
        self.new_cache_entries = {}

    def _cache_sizes(self):
        return {name: sum(len(entry[0]) for entry in fn.device_caches.values())
                for name, fn in self.kernels.items()}

    def __enter__(self):
        from triton import knobs

        self._before = self._cache_sizes()
        self._previous = knobs.runtime.jit_post_compile_hook

        def hook(*args, **kwargs):
            info = kwargs.get("fn")
            self.compiled.append(getattr(info, "name", repr(info)))
            if self._previous is not None:
                return self._previous(*args, **kwargs)
            return None

        knobs.runtime.jit_post_compile_hook = hook
        return self

    def __exit__(self, *exc):
        from triton import knobs

        knobs.runtime.jit_post_compile_hook = self._previous
        after = self._cache_sizes()
        self.new_cache_entries = {
            name: after[name] - self._before[name] for name in after if after[name] != self._before[name]
        }
        return False

    @property
    def count(self):
        return max(len(self.compiled), sum(self.new_cache_entries.values()))

    def describe(self):
        return f"compiled={self.compiled} new_cache_entries={self.new_cache_entries}"


class _MatmulHighest(unittest.TestCase):
    """IEEE FP32 GEMMs so decomposed and concatenated projections agree."""

    def setUp(self):
        self._precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision(self._precision)


# 1. Row stride ---------------------------------------------------------------


@unittest.skipUnless(_CUDA, "CUDA is required")
class RowStrideTest(unittest.TestCase):
    """Distributed runtime passes ``packed[:, :D]`` views (stride(0) == 3D).

    Contiguous inputs and interleaved pair indices are the controls.
    """

    DIMS = (32, 128)

    @staticmethod
    def _views(dim):
        return {"contiguous": lambda t: t[:, :dim].contiguous(),
                "packed[:, :D]": lambda t: t[:, :dim], "packed[:, D:2D]": lambda t: t[:, dim:2 * dim]}

    def _check(self, name, dim, rows, run, reference, *, generator, atol=1e-6, grad_atol=1e-5, **case):
        packed = torch.randn(rows, 3 * dim, generator=generator).cuda()
        for label, view in self._views(dim).items():
            with self.subTest(op=name, dim=dim, view=label, **case):
                fused_leaf, ref_leaf = _leaf(packed), _leaf(packed)
                data = view(fused_leaf)
                self.assertEqual(data.stride(1), 1)
                self.assertEqual(data.stride(0) == dim, label == "contiguous")
                out = run(data)
                expected = reference(view(ref_leaf))
                torch.testing.assert_close(out, expected, atol=atol, rtol=atol,
                                           msg=lambda m: f"{name} forward: {m}")
                seed = torch.randn(expected.shape, generator=generator).cuda()
                grad, = torch.autograd.grad(out, fused_leaf, seed)
                ref_grad, = torch.autograd.grad(expected, ref_leaf, seed)
                torch.testing.assert_close(grad, ref_grad, atol=grad_atol, rtol=grad_atol,
                                           msg=lambda m: f"{name} backward: {m}")

    def test_directed_pair_average_row_strided_views(self):
        from fastmd.models.matris.op.triton_directed_pair_aggregate import directed_pair_average

        generator = _generator(1)
        for interleaved in (False, True):
            for dim in self.DIMS:
                u = 37
                pair = _random_pair_index(u, interleaved=interleaved, generator=generator, device="cuda")
                self._check(
                    "directed_pair_average", dim, 2 * u,
                    lambda data: directed_pair_average(data, pair),
                    lambda data: (data[pair[:, 0]] + data[pair[:, 1]]) * 0.5,
                    generator=generator, grad_atol=1e-6, interleaved=interleaved,
                )

    def test_undirected_pair_expand_row_strided_views(self):
        from fastmd.models.matris.op.triton_directed_pair_aggregate import undirected_pair_expand

        generator = _generator(2)
        for interleaved in (False, True):
            for dim in self.DIMS:
                u = 37
                pair = _random_pair_index(u, interleaved=interleaved, generator=generator, device="cuda")
                d2u = _pair_d2u(pair, 2 * u)
                self._check(
                    "undirected_pair_expand", dim, u,
                    lambda data: undirected_pair_expand(data, pair, 2 * u),
                    lambda data: torch.index_select(data, 0, d2u),
                    generator=generator, atol=0, grad_atol=1e-6, interleaved=interleaved,
                )

    def test_line_envelope_product_row_strided_views(self):
        from fastmd.models.matris.op.triton_line_envelope import line_envelope_product

        generator = _generator(3)
        for dim in self.DIMS:
            base_rows, rows = 29, 101
            source = torch.randint(0, base_rows, (rows,), generator=generator).cuda()
            target = torch.randint(0, base_rows, (rows,), generator=generator).cuda()
            self._check(
                "line_envelope_product", dim, base_rows,
                lambda base: line_envelope_product(base, source, target),
                lambda base: torch.index_select(base, 0, source) * torch.index_select(base, 0, target),
                generator=generator,
            )


# 2. Pair bijection -----------------------------------------------------------


class PairBijectionTest(unittest.TestCase):
    """Pair kernels scatter into ``torch.empty``: they need an exact bijection."""

    def _devices(self):
        return ("cpu", "cuda") if _CUDA else ("cpu",)

    def test_bijection_reason_accepts_only_bijections(self):
        from fastmd.models.matris.op.triton_directed_pair_aggregate import pair_index_bijection_reason

        generator = _generator(10)
        u = 23
        d2u = torch.arange(u).repeat_interleave(2)[torch.randperm(2 * u, generator=generator)]
        valid = {
            "random": _random_pair_index(u, interleaved=False, generator=generator),
            "interleaved": _random_pair_index(u, interleaved=True, generator=generator),
            "argsort_of_d2u": torch.argsort(d2u, stable=True).reshape(u, 2),
            "empty": torch.empty((0, 2), dtype=torch.long),
        }
        pair = torch.tensor([[0, 3], [1, 2], [5, 4]])
        invalid = {
            "duplicate": (torch.tensor([[0, 3], [1, 2], [3, 4]]), 6),
            "self_pair_duplicate": (torch.tensor([[0, 0], [1, 2], [3, 4]]), 6),
            "out_of_range_high": (torch.tensor([[0, 3], [1, 2], [6, 4]]), 6),
            "negative": (torch.tensor([[0, 3], [1, 2], [-1, 4]]), 6),
            "num_directed_too_large": (pair, 8),
            "num_directed_odd": (pair, 7),
            "num_directed_too_small": (pair, 4),
            "rank_1": (pair.reshape(-1), 6),
            "three_columns": (torch.arange(9).reshape(3, 3), 9),
            "rank_3": (pair.reshape(3, 2, 1), 6),
        }
        for device in self._devices():
            for label, pair_index in valid.items():
                for dtype in (torch.int64, torch.int32):
                    with self.subTest(device=device, case=label, dtype=dtype):
                        self.assertIsNone(pair_index_bijection_reason(
                            pair_index.to(device=device, dtype=dtype), 2 * pair_index.shape[0]))
            for label, (pair_index, num_directed) in invalid.items():
                with self.subTest(device=device, case=label):
                    reason = pair_index_bijection_reason(pair_index.to(device), num_directed)
                    self.assertIsInstance(reason, str)
                    self.assertTrue(reason)

    def test_undirected_pair_expand_rejects_num_directed_mismatch(self):
        from fastmd.models.matris.op.triton_directed_pair_aggregate import undirected_pair_expand

        u, dim = 11, 32
        pair = _random_pair_index(u, interleaved=False, generator=_generator(11))
        # Only larger outputs on CUDA: without the check a smaller one would
        # write out of bounds instead of failing this assertion.
        cases = [("cpu", 2 * u + 2), ("cpu", 2 * u + 1), ("cpu", 2 * u - 2)]
        if _CUDA:
            cases += [("cuda", 2 * u + 2), ("cuda", 2 * u + 1)]
        for device, num_directed in cases:
            with self.subTest(device=device, num_directed=num_directed):
                data = torch.randn(u, dim, device=device)
                with self.assertRaises(ValueError):
                    undirected_pair_expand(data, pair.to(device), num_directed)


# 3. Paired CSR attention SKIP_SEGMENT ---------------------------------------

class _PairedAttentionCase:
    """One segment set with an empty real segment and a trailing private sink.

    Sink rows sit in the sink on both sides; ``mixed`` rows are in the sink on
    one side only. Inputs are generated on the host, so they are identical on
    every platform; CSR metadata uses a stable host argsort.
    """

    def __init__(self, *, identity_a, sink_rows, mixed_rows, dim=32, real_segments=9,
                 real_rows=150, offset=0):
        generator = _generator(100 + offset)
        self.num_segment = real_segments + 2  # real_segments itself is empty
        self.sink = sink = self.num_segment - 1

        def real(count):
            return torch.randint(0, real_segments, (count,), generator=generator)

        def full(count):
            return torch.full((count,), sink, dtype=torch.long)

        seg_a = torch.cat((real(real_rows), full(sink_rows), full(mixed_rows), real(mixed_rows)))
        seg_b = torch.cat((real(real_rows), full(sink_rows), real(mixed_rows), full(mixed_rows)))
        rows = seg_a.numel()
        order = (torch.argsort(seg_a, stable=True) if identity_a
                 else torch.randperm(rows, generator=generator))
        seg_a, seg_b = seg_a[order], seg_b[order]
        logits_a = torch.randn(rows, dim, generator=generator) * 3
        logits_b = torch.randn(rows, dim, generator=generator) * 3
        # Large sink logits: any leak of skipped-side weights is conspicuous.
        logits_a[seg_a == sink] += 30.0
        logits_b[seg_b == sink] += 30.0
        self.identity_a = identity_a
        self.seg_a, self.seg_b = seg_a.cuda(), seg_b.cuda()
        perm_a, off_a = _csr(seg_a, self.num_segment)
        perm_b, off_b = _csr(seg_b, self.num_segment)
        self.perm_a = None if identity_a else perm_a.cuda()
        self.off_a, self.perm_b, self.off_b = off_a.cuda(), perm_b.cuda(), off_b.cuda()
        self.logits_a, self.logits_b = logits_a.cuda(), logits_b.cuda()
        self.value = torch.randn(rows, dim, generator=generator).cuda()
        self.gout_a = torch.randn(self.num_segment, dim, generator=generator).cuda()
        self.gout_b = torch.randn(self.num_segment, dim, generator=generator).cuda()

    def paired(self, **kwargs):
        from fastmd.models.matris.op.triton_segment_attention import paired_segment_attention

        leaves = [_leaf(t) for t in (self.logits_a, self.logits_b, self.value)]
        _poison_cuda_cache((self.num_segment, self.value.shape[1]), tuple(self.value.shape))
        out_a, out_b = paired_segment_attention(
            *leaves, self.seg_a, self.seg_b, self.num_segment,
            self.perm_a, self.off_a, self.perm_b, self.off_b, self.identity_a, False, **kwargs)
        grads = torch.autograd.grad((out_a, out_b), leaves, (self.gout_a, self.gout_b))
        return (out_a, out_b, *grads)

    def unpaired(self, skip_segment):
        from fastmd.models.matris.op.triton_segment_attention import segment_attention

        logits_a, logits_b, value = (_leaf(t) for t in (self.logits_a, self.logits_b, self.value))
        out_a = segment_attention(logits_a, value, self.seg_a, self.num_segment, self.perm_a, self.off_a,
                                  self.identity_a, skip_segment=skip_segment)
        out_b = segment_attention(logits_b, value, self.seg_b, self.num_segment, self.perm_b, self.off_b,
                                  False, skip_segment=skip_segment)
        grads = torch.autograd.grad((out_a, out_b), (logits_a, logits_b, value), (self.gout_a, self.gout_b))
        return (out_a, out_b, *grads)

    @staticmethod
    def default_inputs():
        return [_PairedAttentionCase(identity_a=identity, sink_rows=64, mixed_rows=6, offset=7)
                for identity in (True, False)]


_PAIRED_NAMES = ("out_a", "out_b", "grad_logits_a", "grad_logits_b", "grad_value")


@unittest.skipUnless(_CUDA, "CUDA is required")
class PairedSegmentAttentionSkipTest(unittest.TestCase):

    def _skip(self, identity_a):
        case = _PairedAttentionCase(identity_a=identity_a, sink_rows=400, mixed_rows=12)
        sink = case.sink
        result = case.paired(skip_segment=sink)
        torch.cuda.synchronize()
        expected = case.unpaired(skip_segment=sink)
        for name, tensor in zip(_PAIRED_NAMES, result):
            with self.subTest(name=name, check="finite"):
                self.assertTrue(bool(torch.isfinite(tensor).all()), f"{name} has NaN/Inf")
        out_a, out_b, grad_a, grad_b, grad_value = result
        sink_a, sink_b = case.seg_a == sink, case.seg_b == sink
        self.assertGreater(int((sink_a & sink_b).sum()), 0)
        with self.subTest(check="skipped outputs are zero"):
            self.assertTrue(bool((out_a[sink] == 0).all()) and bool((out_b[sink] == 0).all()))
        with self.subTest(check="skipped side contributes exactly zero"):
            self.assertTrue(bool((grad_a[sink_a] == 0).all()), "grad_logits_a on sink rows")
            self.assertTrue(bool((grad_b[sink_b] == 0).all()), "grad_logits_b on sink rows")
            self.assertTrue(bool((grad_value[sink_a & sink_b] == 0).all()), "grad_value on sink rows")
        for name, actual, reference in zip(_PAIRED_NAMES, result, expected):
            with self.subTest(name=name, check="matches two unpaired skip calls"):
                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4, equal_nan=False)

    def test_skip_segment(self):
        for identity_a in (True, False):
            with self.subTest(identity_a=identity_a):
                self._skip(identity_a)

    def test_default_paired_result_matches_torch(self):
        for case in _PairedAttentionCase.default_inputs():
            result = case.paired()
            # The sink is an ordinary segment here; compare with plain torch.
            logits_a, logits_b, value = (_leaf(t) for t in (case.logits_a, case.logits_b, case.value))
            ref_a = _torch_segment_attention(logits_a, value, case.seg_a, case.num_segment)
            ref_b = _torch_segment_attention(logits_b, value, case.seg_b, case.num_segment)
            grads = torch.autograd.grad((ref_a, ref_b), (logits_a, logits_b, value), (case.gout_a, case.gout_b))
            for name, actual, reference in zip(_PAIRED_NAMES, result, (ref_a, ref_b, *grads)):
                with self.subTest(identity_a=case.identity_a, name=name):
                    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=1e-4)


# 4. SORTED_A pair backward and its row tile ---------------------------------


class _DecomposedCase:
    """cat3/cat4 inputs with a sorted, heavily repeated ``index_a``. ``widths``:
    per-part input widths (D8: e.g. [32|128|128], [32|128|32|32]), else all ``width``."""

    def __init__(self, parts, *, rows=257, gathered_rows=48, width=32, hidden=32, offset=0, widths=None):
        generator = _generator(200 + offset)
        n = 2 * hidden
        widths = widths or (width,) * parts
        self.parts, self.hidden = parts, hidden

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, generator=generator) * scale).cuda()

        def index(sort=False):
            value = torch.randint(0, gathered_rows, (rows,), generator=generator)
            return (value.sort().values if sort else value).cuda()

        self.aligned = randn(rows, widths[0])
        # p0 as the head of a wider merged projection: stride(0) == 3H.
        self.merged = randn(rows, n + hidden)
        # cat3 gathers twice from one table, cat4 once from the first and twice from the second.
        self.gathered = [randn(gathered_rows, w) for w in widths[1:parts - 1]]
        self.index_a = index(sort=True)
        self.index_b = index()
        self.index_c = index() if parts == 4 else None
        self.weights = tuple(randn(n, w, scale=w ** -0.5).contiguous() for w in widths)
        self.bias = randn(n)
        self.grad_core, self.grad_gate = randn(rows, hidden), randn(rows, hidden)

    def run(self, *, index_a_sorted, bwd_bm=None, precomputed=False):
        from fastmd.engine.kernels import decomposed_indexed_cat_silu_linear as op

        aligned, merged = _leaf(self.aligned), _leaf(self.merged)
        gathered = [_leaf(t) for t in self.gathered]
        p0 = merged[:, :2 * self.hidden] if precomputed else None
        # The backward row tile is a module global read at backward time; the
        # pair-backward switch is read at forward time. Pin both over the call.
        tile = op._EPILOGUE_BWD_BM if bwd_bm is None else bwd_bm
        with mock.patch.object(op, "_SORTED_A_PAIR_BWD", True), \
                mock.patch.object(op, "_EPILOGUE_BWD_BM", tile):
            if self.parts == 3:
                result = op.decomposed_indexed_cat3_silu_linear(
                    aligned, gathered[0], self.index_a, self.index_b, self.weights, self.bias,
                    index_a_sorted, p0, strict_errors=True)
            else:
                result = op.decomposed_indexed_cat4_silu_linear(
                    aligned, gathered[0], self.index_a, gathered[1], self.index_b, self.index_c,
                    self.weights, self.bias, index_a_sorted, p0, strict_errors=True)
            assert result is not None, op.last_decomposed_indexed_cat_silu_linear_error()
            inputs = [merged if precomputed else aligned, *gathered]
            grads = torch.autograd.grad(result, inputs, (self.grad_core, self.grad_gate))
        torch.cuda.synchronize()
        return result, grads

    def reference(self, *, precomputed=False, dtype=torch.float32):
        cast = (lambda t: t.to(dtype))
        aligned, merged = _leaf(cast(self.aligned)), _leaf(cast(self.merged))
        gathered = [_leaf(cast(t)) for t in self.gathered]
        weights = [cast(w) for w in self.weights]
        pre = merged[:, :2 * self.hidden] if precomputed else aligned @ weights[0].T
        if self.parts == 3:
            pre = pre + gathered[0][self.index_a] @ weights[1].T + gathered[0][self.index_b] @ weights[2].T
        else:
            pre = (pre + gathered[0][self.index_a] @ weights[1].T + gathered[1][self.index_b] @ weights[2].T
                   + gathered[1][self.index_c] @ weights[3].T)
        pre = pre + cast(self.bias)
        result = (F.silu(pre[:, :self.hidden]), F.silu(pre[:, self.hidden:]))
        inputs = [merged if precomputed else aligned, *gathered]
        return result, torch.autograd.grad(result, inputs, (cast(self.grad_core), cast(self.grad_gate)))


@unittest.skipUnless(_CUDA, "CUDA is required")
class DecomposedSortedPairBackwardTest(_MatmulHighest):
    """Pairing rows by lane parity would double-accumulate repeats at BM=1."""

    def _check(self, parts, bwd_bms, widths=None, hidden=32):
        case = _DecomposedCase(parts, offset=parts, widths=widths, hidden=hidden)
        for precomputed in (False, True):
            ref_out, ref_grads = case.reference(precomputed=precomputed)
            if widths is not None:
                # D8: the FP32 kernels against an FP64 reference as well.
                out64, grads64 = case.reference(precomputed=precomputed, dtype=torch.float64)
                ref_out = ref_out + out64
                ref_grads = ref_grads + grads64
            _, unsorted_grads = case.run(index_a_sorted=False, precomputed=precomputed)
            for bwd_bm in bwd_bms:
                out, grads = case.run(index_a_sorted=True, bwd_bm=bwd_bm, precomputed=precomputed)
                names = ["precomputed_p0" if precomputed else "aligned"] + [
                    f"gathered_{k}" for k in range(len(grads) - 1)]
                count = len(grads)
                for k, (name, grad, unsorted_grad) in enumerate(zip(names, grads, unsorted_grads)):
                    with self.subTest(cat=parts, widths=widths, bwd_bm=bwd_bm, precomputed_p0=precomputed,
                                      grad=name):
                        torch.testing.assert_close(grad, unsorted_grad, atol=1e-4, rtol=1e-4,
                                                   msg=lambda m: f"sorted vs unsorted: {m}")
                        for ref_grad in ref_grads[k::count]:
                            torch.testing.assert_close(grad, ref_grad.to(grad.dtype), atol=1e-4, rtol=1e-4,
                                                       msg=lambda m: f"sorted vs torch {ref_grad.dtype}: {m}")
                with self.subTest(cat=parts, widths=widths, bwd_bm=bwd_bm, precomputed_p0=precomputed,
                                  check="forward"):
                    for k, actual in enumerate(out):
                        for expected in ref_out[k::len(out)]:
                            torch.testing.assert_close(actual, expected.to(actual.dtype), atol=1e-5, rtol=1e-5)

    def test_sorted_pair_backward_matches_torch(self):
        for parts in (3, 4):
            self._check(parts, (1, 2, 4, 8, 16))

    def test_mixed_width_parts_match_fp32_and_fp64(self):
        # D8: the atom rows [32|128|128] (H=128) and the line refinement [32|128|32|32] (H=32).
        self._check(3, (1, 8), widths=(32, 128, 128), hidden=128)
        self._check(4, (1, 8), widths=(32, 128, 32, 32), hidden=32)


@unittest.skipUnless(_CUDA, "CUDA is required")
class MixedWidthGatedMLPTest(_MatmulHighest):
    """D8 (step 2.5b): GatedMLP's indexed cat3/cat4 over parts of different widths split
    the frozen first linears per part; forward and input gradients match the gathered
    concatenation (FP32 and FP64 torch), and the switch off declines."""

    CASES = (("cat3", (32, 128, 128), 128), ("cat4", (32, 128, 32, 32), 32))

    def _case(self, name, widths, hidden, seed):
        import copy
        from fastmd.models.matris import functions

        generator = torch.Generator().manual_seed(seed)
        mlp = functions.GatedMLP(sum(widths), hidden, 32, norm_type="layer").cuda().eval().requires_grad_(False)
        with torch.no_grad():
            for norm in (mlp.core_norm, mlp.gate_norm):  # non-trivial affine LayerNorms
                norm.weight.copy_(1 + 0.1 * torch.randn(norm.weight.shape, generator=generator))
                norm.bias.copy_(0.1 * torch.randn(norm.bias.shape, generator=generator))
        rows, nodes = 3000, (97, 211)
        aligned = torch.randn(rows, widths[0], generator=generator).cuda()
        tables = [torch.randn(nodes[k], w, generator=generator).cuda() for k, w in enumerate(widths[1:len(widths) - 1])]
        sources = [0, 0] if name == "cat3" else [0, 1, 1]
        index = [torch.randint(0, nodes[t], (rows,), generator=generator).cuda() for t in sources]
        seed_out = torch.randn(rows, 32, generator=generator).cuda()
        return functions, copy, mlp, aligned, tables, sources, index, seed_out

    def test_mixed_width_projection_matches_fp32_and_fp64(self):
        for seed, (name, widths, hidden) in enumerate(self.CASES):
            functions, copy, mlp, aligned, tables, sources, index, seed_out = self._case(name, widths, hidden, seed)

            def fused(module, mixed):
                x, ts = _leaf(aligned), [_leaf(t) for t in tables]
                with mock.patch.object(functions, "_INDEXED_CAT_LINEAR_FUSE", True), \
                        mock.patch.object(functions, "_INDEXED_CAT_LINEAR_MIN_ROWS", 0), \
                        mock.patch.object(functions, "_MIXED_WIDTH_CAT", mixed):
                    if name == "cat3":
                        out = module.forward_aligned_gather_cat3(x, ts[0], index[0], index[1])
                    else:
                        out = module.forward_aligned_gather_cat4(x, ts[0], index[0], ts[1], index[1], index[2])
                if out is None:
                    return None
                return out, torch.autograd.grad(out, (x, *ts), seed_out)

            def reference(module, dtype):
                x, ts = _leaf(aligned.to(dtype)), [_leaf(t.to(dtype)) for t in tables]
                parts = [x] + [ts[t][i] for t, i in zip(sources, index)]
                previous = functions.set_fused_activations(False)
                try:
                    with mock.patch.object(functions, "_LN_FUSE_THRESH", 1 << 30):
                        out = module(torch.cat(parts, 1))
                finally:
                    functions.set_fused_activations(previous)
                return out, torch.autograd.grad(out, (x, *ts), seed_out.to(dtype))

            with self.subTest(case=name, check="switch off declines"):
                self.assertIsNone(fused(mlp, False))
            result = fused(mlp, True)
            with self.subTest(case=name, check="selected"):
                self.assertIsNotNone(result)
            out, grads = result
            ref32 = reference(mlp, torch.float32)
            ref64 = reference(copy.deepcopy(mlp).double(), torch.float64)
            for label, (ref_out, ref_grads) in (("fp32", ref32), ("fp64", ref64)):
                with self.subTest(case=name, reference=label, check="forward"):
                    torch.testing.assert_close(out, ref_out.float(), atol=2e-5, rtol=1e-5)
                for k, (grad, ref_grad) in enumerate(zip(grads, ref_grads)):
                    with self.subTest(case=name, reference=label, grad=k):
                        torch.testing.assert_close(grad, ref_grad.float(), atol=1e-4, rtol=1e-4)


class BackwardTileConfigTest(unittest.TestCase):
    """``tl.arange`` needs a power of two: reject other tiles at config time."""

    def _environment_without_matris(self):
        return {name: value for name, value in os.environ.items() if not name.startswith("MATRIS_")}

    def test_backward_tile_must_be_a_power_of_two(self):
        from fastmd.config import InferenceConfig, env_value, resolve_config

        for value in (1, 2, 3, 4, 6, 8, 12, 16):
            valid = value & (value - 1) == 0
            with self.subTest(path="expert_overrides", value=value):
                def resolve():
                    return resolve_config(InferenceConfig(expert_overrides={_BWD_BM_OPTION: value}, report=False))
                if valid:
                    self.assertEqual(resolve().kernel_options[_BWD_BM_OPTION], value)
                else:
                    with self.assertRaises(ValueError):
                        resolve()
            environment = self._environment_without_matris() | {_BWD_BM_OPTION: str(value)}
            with self.subTest(path="legacy environment", value=value), \
                    mock.patch.dict(os.environ, environment, clear=True):
                if valid:
                    resolved = resolve_config(InferenceConfig(optimization_profile="legacy", report=False))
                    self.assertEqual(resolved.kernel_options[_BWD_BM_OPTION], value)
                else:
                    with self.assertRaises(ValueError):
                        resolve_config(InferenceConfig(optimization_profile="legacy", report=False))
                    with self.assertRaises(ValueError):
                        env_value(_BWD_BM_OPTION)


@unittest.skipUnless(_CUDA, "CUDA is required")
class RowTileTest(_MatmulHighest):
    """C1: residual add, line envelope, directed-pair average/expand and the indexed SiLU
    forward pick their row tile by feature width (``row_tiles``); a pinned option wins, and
    MATRIS_WIDTH_ROW_TILES=0 restores the former fixed tiles. Every tile matches torch."""

    _PINNED = {"MATRIS_FUSED_RESIDUAL_ADD_BM": 2, "MATRIS_LINE_ENVELOPE_BM": 2,
               "MATRIS_DIRECTED_PAIR_AGG_BM": 2, "MATRIS_DECOMPOSED_INDEXED_SILU_BM": 1}
    # mode: (expert_overrides, {width: (row-kernel tile, indexed SiLU forward tile)}); the fixed
    # tiles are 4 (residual add, line envelope), 8 (pairs) and 2 (SiLU), see _expected. Width 7
    # (the atom smooth weight's pair expand) takes the largest tiles.
    _MODES = {
        "width": ({}, {7: (64, 16), 32: (32, 8), 64: (16, 4), 128: (8, 2)}),
        "fixed": ({"MATRIS_WIDTH_ROW_TILES": False}, None),
        "pinned": (_PINNED, {32: (2, 1)}),
    }

    @classmethod
    def _expected(cls, mode, width):
        """{(op, phase): BM}; the SiLU backward keeps its own tile (8) in every mode."""
        tiles = cls._MODES[mode][1]
        row, silu = (None, 2) if tiles is None else tiles[width]
        expected = {}
        for op in ("residual_add", "line_envelope", "pair_average", "pair_expand"):
            fixed = 8 if op.startswith("pair") else 4
            expected.update({(op, "fwd"): row or fixed, (op, "bwd"): row or fixed})
        for op in ("cat3_gather_add_silu", "cat4_gather_add_silu"):
            expected.update({(op, "fwd"): silu, (op, "bwd"): 8})
        return expected

    def _run_ops(self, width, generator):
        from fastmd.models.matris.op.triton_directed_pair_aggregate import directed_pair_average, undirected_pair_expand
        from fastmd.models.matris.op.triton_line_envelope import line_envelope_product
        from fastmd.models.matris.op.triton_residual_add import residual_add

        def cuda(*shape):
            return torch.randn(*shape, generator=generator).cuda()

        rows, u, base_rows = 101, 37, 29
        pair = _random_pair_index(u, interleaved=False, generator=generator, device="cuda")
        source = torch.randint(0, base_rows, (rows,), generator=generator).cuda()
        target = torch.randint(0, base_rows, (rows,), generator=generator).cuda()
        weight = cuda(1, width)
        cases = {
            "residual_add": ((cuda(rows, width), cuda(rows, width)), lambda d, r: residual_add(d, r, weight),
                             lambda d, r: d + weight * r),
            "line_envelope": ((cuda(base_rows, width),), lambda b: line_envelope_product(b, source, target),
                              lambda b: torch.index_select(b, 0, source) * torch.index_select(b, 0, target)),
            "pair_average": ((cuda(2 * u, width),), lambda x: directed_pair_average(x, pair),
                             lambda x: (x[pair[:, 0]] + x[pair[:, 1]]) * 0.5),
            "pair_expand": ((cuda(u, width),), lambda x: undirected_pair_expand(x, pair, 2 * u),
                            lambda x: torch.index_select(x, 0, _pair_d2u(pair, 2 * u))),
        }
        for name, (inputs, fused, reference) in cases.items():
            leaves, ref_leaves = [_leaf(t) for t in inputs], [_leaf(t) for t in inputs]
            out, expected = fused(*leaves), reference(*ref_leaves)
            seed = torch.randn(expected.shape, generator=generator).cuda()
            with self.subTest(op=name, width=width):
                torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
                for grad, ref_grad in zip(torch.autograd.grad(out, leaves, seed),
                                          torch.autograd.grad(expected, ref_leaves, seed)):
                    torch.testing.assert_close(grad, ref_grad, atol=1e-5, rtol=1e-5)
        for parts in (3, 4):
            case = _DecomposedCase(parts, width=width, hidden=width, offset=10 + parts)
            (out, grads), (ref_out, ref_grads) = case.run(index_a_sorted=True), case.reference()
            with self.subTest(op=f"indexed_cat{parts}", width=width):
                for actual, expected in zip(out + grads, ref_out + ref_grads):
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_tiles_by_width_match_torch(self):
        import warnings

        from fastmd.config import InferenceConfig, resolve_config
        from fastmd.engine.recorder import OpPlanRecorder

        generator = _generator(400)
        for mode, (overrides, tiles) in self._MODES.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolved = resolve_config(InferenceConfig(optimization_profile="topology", report=False,
                                                          expert_overrides=overrides))
            for width in tiles or (32,):
                with resolved.scope(), OpPlanRecorder(aten=False) as recording:
                    self._run_ops(width, generator)
                expected, selected = self._expected(mode, width), {}
                for entry in recording.snapshot()["sections"][0]["entries"]:
                    key = entry["op"], entry["phase"]
                    if key in expected and entry["impl"] == "triton":
                        selected.setdefault(key, set()).add(entry["params"]["BM"])
                with self.subTest(mode=mode, width=width, check="selection"):
                    self.assertEqual(selected, {key: {bm} for key, bm in expected.items()})


@unittest.skipUnless(_CUDA, "CUDA is required")
class MergedProjectionBackwardTest(_MatmulHighest):
    """C3: the cat3 epilogue (p0) and the segment attentions (alphas) write their gradients
    into the merged projection's buffer, whose backward runs one GEMM per contiguous run of
    such columns; any other gradient takes the per-split GEMMs, and without two adjacent
    registered consumers no buffer is used. Against the switch off and an FP64 torch
    reference (the switch off where no plain-torch reference exists), with the recorded path;
    the allocator holds NaN, so a read of an unwritten column fails. The switch's auto (the
    memory admission's decision) and the last block's ``grad_buffer=False`` resolve to either path."""

    ROWS, NODES, WIDTH, HIDDEN = 301, 23, 32, 32

    def setUp(self):
        super().setUp()
        generator = _generator(500)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, generator=generator) * scale).cuda()

        n = 2 * self.HIDDEN
        self.splits = (n, self.WIDTH, self.WIDTH)
        self.x = randn(self.ROWS, self.WIDTH)
        self.w_cat = randn(sum(self.splits), self.WIDTH, scale=self.WIDTH ** -0.5)
        self.nodes = randn(self.NODES, self.WIDTH)
        self.w_nodes = (randn(n, self.WIDTH, scale=self.WIDTH ** -0.5), randn(n, self.WIDTH, scale=self.WIDTH ** -0.5))
        self.bias = randn(n)
        self.index_a = torch.randint(0, self.NODES, (self.ROWS,), generator=generator).sort().values.cuda()
        self.index_b = torch.randint(0, self.NODES, (self.ROWS,), generator=generator).cuda()
        self.value = randn(self.ROWS, self.WIDTH)
        self.seeds = [randn(self.ROWS, self.HIDDEN) for _ in range(4)] + [randn(self.NODES, self.WIDTH) for _ in range(4)]
        self.grad_buffer = True

    def _loss(self, x, scenario, fp64=False):
        """Consumers of [p0 | alpha_a | alpha_b] per scenario; fp64: plain torch in float64."""
        from fastmd.engine.kernels import decomposed_indexed_cat_silu_linear as op
        from fastmd.engine.kernels.merged_frozen_projections import merged_frozen_projections
        from fastmd.models.matris.op.triton_segment_attention import paired_segment_attention, segment_attention

        cast = (lambda t: t.double()) if fp64 else (lambda t: t)
        w_cat, nodes, value, bias = cast(self.w_cat), cast(self.nodes), cast(self.value), cast(self.bias)
        seeds = [cast(t) for t in self.seeds]
        if fp64:
            p0, alpha_a, alpha_b = torch.nn.functional.linear(x, w_cat).split(self.splits, 1)
        else:
            p0, alpha_a, alpha_b = merged_frozen_projections(x, w_cat, self.splits, self.grad_buffer)

        def cat3(p0):
            if fp64:
                pre = (p0 + nodes[self.index_a] @ cast(self.w_nodes[0]).T
                       + nodes[self.index_b] @ cast(self.w_nodes[1]).T + bias)
                return F.silu(pre[:, :self.HIDDEN]), F.silu(pre[:, self.HIDDEN:])
            out = op.decomposed_indexed_cat3_silu_linear(
                x, nodes, self.index_a, self.index_b, (w_cat[:self.splits[0]], *self.w_nodes), bias,
                True, p0, strict_errors=True)
            assert out is not None, op.last_decomposed_indexed_cat_silu_linear_error()
            return out

        def attention(logits, segment, triton=True):
            if fp64 or not triton:
                return _torch_segment_attention(logits, value, segment, self.NODES)
            return segment_attention(logits, value, segment, self.NODES)

        core, gate = cat3(p0)
        loss = (core * seeds[0]).sum() + (gate * seeds[1]).sum()
        if scenario == "all":
            if fp64:
                out_a, out_b = attention(alpha_a, self.index_a), attention(alpha_b, self.index_b)
            else:
                out_a, out_b = paired_segment_attention(alpha_a, alpha_b, value, self.index_a, self.index_b,
                                                        self.NODES)
            loss = loss + (out_a * seeds[4]).sum() + (out_b * seeds[5]).sum()
        elif scenario == "skip_lse":  # the paired kernel's SKIP and LSE stores into the buffer
            out_a, out_b, lse_a, lse_b = paired_segment_attention(
                alpha_a, alpha_b, value, self.index_a, self.index_b, self.NODES, skip_segment=0, return_lse=True)
            finite = [torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse)) for lse in (lse_a, lse_b)]
            loss = (loss + (out_a * seeds[4]).sum() + (out_b * seeds[5]).sum()
                    + (finite[0] * seeds[6]).sum() + (finite[1] * seeds[7]).sum())
        elif scenario == "tail_dead":  # alpha_b unused
            loss = loss + (attention(alpha_a, self.index_a) * seeds[4]).sum()
        elif scenario == "p0_twice":  # two p0 consumers: autograd sums their gradients
            core2, gate2 = cat3(p0)
            loss = loss + (core2 * seeds[2]).sum() + (gate2 * seeds[3]).sum()
            loss = loss + (attention(alpha_a, self.index_a) * seeds[4]).sum()
            loss = loss + (attention(alpha_b, self.index_b) * seeds[5]).sum()
        elif scenario == "middle_dead_torch_tail":  # alpha_a unused; alpha_b through plain torch
            loss = loss + (attention(alpha_b, self.index_b, triton=False) * seeds[6]).sum()
        # "alphas_dead": p0 only
        return loss

    def _grads(self, x, scenario, fp64=False, passes=1):
        leaf = (x.double() if fp64 else x).detach().requires_grad_(True)
        loss = self._loss(leaf, scenario, fp64)
        return [torch.autograd.grad(loss, leaf, retain_graph=k + 1 < passes)[0] for k in range(passes)]

    # scenario: (in_buffer, GEMMs) of the single-GEMM backward; without two adjacent registered
    # consumers (alphas_dead, middle_dead_torch_tail) there is no buffer.
    PATHS = {"all": ([True, True, True], 1), "skip_lse": ([True, True, True], 1),
             "tail_dead": ([True, True, False], 1), "p0_twice": ([False, True, True], 2),
             "alphas_dead": ([False, False, False], 1), "middle_dead_torch_tail": ([False, False, False], 2)}

    def test_single_gemm_backward_matches_fallback_and_fp64(self):
        from fastmd.engine.recorder import OpPlanRecorder
        from fastmd.engine.kernels import merged_frozen_projections as merged

        for scenario, (in_buffer, gemms) in self.PATHS.items():
            reference = None
            if scenario != "skip_lse":
                reference = self._grads(self.x, scenario, fp64=True)[0].float()
            for single, row_gemm in (("off", False), ("on", False), ("off", True), ("on", True)):
                with self.subTest(scenario=scenario, single_bwd=single, row_gemm=row_gemm), \
                        mock.patch.object(merged, "_SINGLE_BWD", single), \
                        mock.patch.object(merged, "_TRITON_ROW_GEMM", row_gemm), OpPlanRecorder(aten=False) as rec:
                    _poison_cuda_cache((self.ROWS, sum(self.splits)))
                    grads = self._grads(self.x, scenario, passes=2)  # the second pass: retain_graph
                    torch.cuda.synchronize()
                    if reference is None:  # skip_lse: the switch-off kernels are the reference
                        reference = grads[0]
                    for grad in grads:
                        torch.testing.assert_close(grad, reference, atol=1e-4, rtol=1e-4)
                    paths = [(e["params"].get("in_buffer"), e["params"].get("gemms"))
                             for e in rec.snapshot()["sections"][0]["entries"]
                             if e["op"] == "merged_frozen_projections" and e["phase"] == "bwd"]
                    self.assertEqual(paths, [(in_buffer, gemms)] * 2 if single == "on" else [(None, None)] * 2)

    def test_switch_resolution(self):
        from fastmd.engine.recorder import OpPlanRecorder
        from fastmd.engine import saved_recompute
        from fastmd.engine.kernels import merged_frozen_projections as merged

        for switch, auto, bound, grad_buffer, used in (
                ("auto", True, None, True, True), ("auto", False, None, True, False),
                ("auto", False, True, True, True), ("auto", True, False, True, False),  # a bound evaluator decides
                ("on", False, None, False, False), ("off", True, True, True, False)):
            admission = None if bound is None else mock.Mock(recompute="off", single_bwd_decision=bound)
            self.grad_buffer = grad_buffer
            with self.subTest(switch=switch, auto=auto, bound=bound, grad_buffer=grad_buffer), \
                    mock.patch.object(merged, "_SINGLE_BWD", switch), \
                    mock.patch.object(saved_recompute, "_SINGLE_BWD_AUTO", auto), \
                    saved_recompute.bound_admission(admission), OpPlanRecorder(aten=False) as rec:
                self._grads(self.x, "all")
                [entry] = [e for e in rec.snapshot()["sections"][0]["entries"]
                           if e["op"] == "merged_frozen_projections" and e["phase"] == "bwd"]
                self.assertEqual(entry["params"].get("gemms"), 1 if used else None)

    def test_captured_backward_replays_like_eager(self):
        from fastmd.engine.recorder import OpPlanRecorder
        from fastmd.engine.kernels import merged_frozen_projections as merged

        static_x = self.x.clone()
        new_x = self.x.flip(0).contiguous()
        switch = mock.patch.object(merged, "_SINGLE_BWD", "on")
        switch.start()
        self.addCleanup(switch.stop)
        expected = self._grads(new_x, "all")[0]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._grads(static_x, "all")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with OpPlanRecorder(aten=False) as rec, torch.cuda.graph(graph):
            static_grad = self._grads(static_x, "all")[0]
        [entry] = [e for e in rec.snapshot()["sections"][0]["entries"]
                   if e["op"] == "merged_frozen_projections" and e["phase"] == "bwd"]
        self.assertEqual((entry["params"]["in_buffer"], entry["params"]["gemms"]), ([True, True, True], 1))
        static_x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_grad, expected, atol=1e-5, rtol=1e-5)


# C4: row GEMMs and the fused edge FFN ----------------------------------------


@unittest.skipUnless(_CUDA, "CUDA is required")
class RowGemmTest(_MatmulHighest):
    """C4: the row GEMM (every tile width, row-strided input) and the projection gradient's addmm against
    FP64 and cuBLAS; a leading or trailing column range of a product is bitwise the wide product's columns
    (the recompute recipes rely on it) and a row's result does not depend on the row count or its position
    (bond rows replicated on two ranks); the fused edge FFN forward and backward against FP64 and the
    unfused FP32 layers, also a graph-building backward; all of them replayed from a CUDA graph like eager."""

    def test_row_gemm_and_addmm_match_fp64(self):
        from fastmd.engine.kernels import triton_row_gemm as rg

        generator = _generator(700)
        cases = [(n, rows, prec) for n in sorted(rg._TILES) for rows in (301, 5003) for prec in ("tf32x3", "ieee", "tf32")]
        for n, rows, prec in cases:
            base = torch.randn(rows, 48, generator=generator).cuda()
            x = base[:, 8:40]  # row stride 48
            w = (torch.randn(n, 32, generator=generator) * 0.2).cuda()
            with self.subTest(n=n, rows=rows, precision=prec):
                self.assertIsNone(rg.row_linear_supported(x, w))
                y = rg.row_linear_forward(x, w, prec)
                reference = x.double() @ w.double().t()
                tol = 3e-3 if prec == "tf32" else 2e-5
                torch.testing.assert_close(y.double(), reference, atol=tol, rtol=tol)
                # The recompute recipes rebuild column ranges and rows alone: bitwise the wide product's.
                for cut in (64, n - 64, n // 2):
                    for start, stop in ((0, cut), (cut, n)):
                        if 0 < stop - start < n and stop - start in rg._TILES:
                            self.assertTrue(torch.equal(rg.row_linear_forward(x, w[start:stop].contiguous(), prec),
                                                        y[:, start:stop]), (start, stop))
                # A row's result depends on the row only: other row counts and offsets give the same bits.
                for offset, count in ((1, 17), (rows - 40, 40), (123, rows - 200)):
                    self.assertTrue(torch.equal(rg.row_linear_forward(x[offset:offset + count], w, prec),
                                                y[offset:offset + count]), (offset, count))
                self.assertTrue(torch.equal(rg.row_linear_forward(x.contiguous(), w, prec), y))
        for rows in (301, 5003):
            g = torch.randn(rows, 64, generator=generator).cuda()
            c = torch.randn(rows, 48, generator=generator).cuda()[:, :32]
            w = (torch.randn(64, 32, generator=generator) * 0.2).cuda()
            with self.subTest(addmm_rows=rows):
                self.assertTrue(rg.row_addmm_supported(c, g, w))
                torch.testing.assert_close(rg.row_addmm(c, g, w).double(), c.double() + g.double() @ w.double(),
                                           atol=2e-5, rtol=1e-5)

    def _ffn(self, rows, generator):
        x = torch.randn(rows, 48, generator=generator).cuda()[:, 4:36]
        w1 = (torch.randn(32, 32, generator=generator) * 0.2).cuda()
        w2 = (torch.randn(32, 32, generator=generator) * 0.2).cuda()
        seed = torch.randn(rows, 32, generator=generator).cuda()
        return x, w1, w2, seed

    def test_edge_ffn_matches_fp64_and_unfused(self):
        from fastmd.models.matris.op.triton_edge_ffn import edge_ffn_supported, fused_edge_ffn

        generator = _generator(701)
        for rows in (1, 301, 5003):
            x, w1, w2, seed = self._ffn(rows, generator)
            with self.subTest(rows=rows):
                self.assertIsNone(edge_ffn_supported(x, w1, w2))
                outs = {}
                for name, dtype in (("fused", torch.float32), ("graph", torch.float32), ("unfused", torch.float32),
                                    ("fp64", torch.float64)):
                    xs = _leaf(x.to(dtype))
                    if name in ("fused", "graph"):
                        y = fused_edge_ffn(xs, w1, w2, "tf32x3")
                    else:
                        y = torch.nn.functional.linear(torch.nn.functional.silu(
                            torch.nn.functional.linear(xs, w1.to(dtype))), w2.to(dtype))
                    # "graph": a graph-building backward takes the torch formula.
                    (grad,) = torch.autograd.grad(y, (xs,), seed.to(dtype), create_graph=name == "graph")
                    outs[name] = (y, grad)
                for name in ("fused", "graph"):
                    for got, unfused, reference in zip(outs[name], outs["unfused"], outs["fp64"]):
                        torch.testing.assert_close(got.double(), reference, atol=5e-5, rtol=1e-5)
                        torch.testing.assert_close(got, unfused, atol=5e-5, rtol=1e-5)

    def test_dot_precisions_match_fp64(self):
        """Every Triton FP32 dot site at each explicit dot precision (InferenceConfig.triton_dot_precision):
        ieee and tf32x3 at FP32 level, tf32 at TF32 level (10-bit mantissa inputs)."""
        from fastmd.models.matris.op import triton_gated_tail as gt
        from fastmd.engine.kernels import triton_row_gemm as rg
        from fastmd.models.matris.op.triton_act_linear import act_linear_pair
        from fastmd.models.matris.op.triton_edge_ffn import fused_edge_ffn
        from fastmd.precision import triton_dot_precision_scope

        generator = _generator(703)
        rows = 3001
        x = torch.randn(rows, 32, generator=generator).cuda()
        w = (torch.randn(128, 32, generator=generator) * 0.2).cuda()
        fx, w1, w2, seed = self._ffn(rows, generator)
        lc, lg = (torch.nn.Linear(32, 64).cuda().requires_grad_(False) for _ in range(2))
        tw, tb = (torch.randn(32, 32, generator=generator) * 0.2).cuda(), torch.randn(32, generator=generator).cuda()
        nw, nb = torch.rand(32, generator=generator).cuda() + 0.5, torch.randn(32, generator=generator).cuda()
        silu, d = torch.nn.functional.silu, lambda t: t.double()

        def layer_norm(value):
            return torch.nn.functional.layer_norm(value, (32,), d(nw), d(nb), 1e-5)
        for prec, tol in (("ieee", 2e-5), ("tf32x3", 2e-5), ("tf32", 3e-3)):
            with self.subTest(precision=prec):
                ffn_x = _leaf(fx)
                ffn = fused_edge_ffn(ffn_x, w1, w2, prec)
                (ffn_grad,) = torch.autograd.grad(ffn, (ffn_x,), seed)
                ffn_x64 = _leaf(d(fx))
                ffn64 = d(silu(ffn_x64 @ d(w1).t())) @ d(w2).t()
                (ffn_grad64,) = torch.autograd.grad(ffn64, (ffn_x64,), d(seed))
                with triton_dot_precision_scope(prec):
                    tail = gt.fused_gated_tail(x, fx, tw, tb, tw, tb, nw, nb, nw, nb)
                pairs = [
                    (rg.row_linear_forward(x, w, prec), d(x) @ d(w).t()),
                    (ffn, ffn64), (ffn_grad, ffn_grad64),
                    *zip(act_linear_pair(x, fx, lc, lg, silu=True, precision=prec),
                         (silu(d(x)) @ d(lc.weight).t() + d(lc.bias), silu(d(fx)) @ d(lg.weight).t() + d(lg.bias))),
                    (tail, silu(layer_norm(d(x) @ d(tw).t() + d(tb))) * torch.sigmoid(layer_norm(d(fx) @ d(tw).t() + d(tb)))),
                ]
                for got, reference in pairs:
                    torch.testing.assert_close(got.double(), reference, atol=tol, rtol=tol)

    def test_recipe_binding_and_merged_split_widths(self):
        """The merged row GEMM needs tiles for the column ranges its recipes rebuild alone (else it declines at
        every recompute level); a recipe bound to the forward's precision rebuilds bitwise from any input
        layout, and cannot silently fall back."""
        from fastmd.engine.kernels import merged_frozen_projections as mfp
        from fastmd.engine.kernels import triton_row_gemm as rg

        generator = _generator(704)
        x = torch.randn(999, 48, generator=generator).cuda()[:, :32]
        w = (torch.randn(320, 32, generator=generator) * 0.2).cuda()
        with mock.patch.object(mfp, "_TRITON_ROW_GEMM", True):
            self.assertIsNone(mfp.merged_row_gemm_decline(x, w, (256, 32, 32)))
            self.assertEqual(mfp.merged_row_gemm_decline(x, w, (192, 64, 64)), "unsupported_shape")
            self.assertIsNone(mfp.row_gemm_precision(x, w, (192, 64, 64)))
            self.assertEqual(mfp.row_gemm_precision(x, w, (256, 32, 32)), "tf32x3")
        wide = rg.row_linear_forward(x, w, "tf32x3")
        column_major = x.contiguous().t().contiguous().t()
        self.assertTrue(torch.equal(mfp.project(column_major, w[256:], "tf32x3"), wide[:, 256:]))
        self.assertTrue(torch.equal(mfp.project(x, w[256:], None), torch.nn.functional.linear(x, w[256:])))
        with self.assertRaises(RuntimeError):
            mfp.project(x, w[:192], "tf32x3")  # no tile for N = 192

    def test_wide_gemm_matches_fp64(self):
        """TX: the 3xTF32 wide GEMM (forward with bias, input gradient, accumulation into c, rows written into a
        larger output) at FP32 accuracy: its error against FP64 not above cuBLAS FP32's on the same operands, at
        every K (no growth with K, also a K tail below the 64-wide chunk) and N (64, 128-wide tiles); a row's result
        independent of the row count; the split weights follow an in-place update and a rebinding of the weight's
        storage and are released with the weight; a graph-building backward takes torch's formula; tf32 at TF32
        accuracy; widths and layouts it cannot run decline to cuBLAS."""
        import gc
        import weakref
        from fastmd.engine.kernels import triton_wide_gemm as wg

        def rms(got, ref):
            return float((got.double() - ref).square().mean().sqrt() / ref.square().mean().sqrt())

        generator = _generator(705)
        with mock.patch.object(wg, "_TRITON_WIDE_GEMM", True), mock.patch.object(wg, "_MIN_ROWS", 0):
            for (k, n), rows in [(kn, rows) for kn in ((128, 128), (128, 512), (256, 128), (512, 128), (96, 64))
                                 for rows in (77, 70001)]:
                x = torch.randn(rows, k, generator=generator).cuda()
                w = (torch.randn(n, k, generator=generator) / k ** 0.5).cuda()
                bias = torch.randn(n, generator=generator).cuda()
                seed = torch.randn(rows, n, generator=generator).cuda()
                with self.subTest(k=k, n=n, rows=rows):
                    self.assertEqual(wg.linear_precision(x, w, bias), "tf32x3")
                    xs = _leaf(x)
                    y = wg.linear(xs, w, bias)
                    self.assertIsNotNone(y.grad_fn)
                    (dx,) = torch.autograd.grad(y, (xs,), seed)
                    y64 = x.double() @ w.double().t() + bias.double()
                    dx64 = seed.double() @ w.double()
                    self.assertLessEqual(rms(y, y64), 1.5 * rms(F.linear(x, w, bias), y64) + 1e-8)
                    self.assertLessEqual(rms(dx, dx64), 1.5 * rms(seed @ w, dx64) + 1e-8)
                    c = torch.randn(rows, k, generator=generator).cuda()
                    self.assertLessEqual(rms(wg.matmul(seed, w, c), dx64 + c.double()),
                                         1.5 * rms(torch.addmm(c, seed, w), dx64 + c.double()) + 1e-8)
                    acc = c.clone()
                    self.assertIs(wg.matmul(seed, w, acc, accumulate=True), acc)  # in place, as c.addmm_
                    self.assertLessEqual(rms(acc, dx64 + c.double()),
                                         1.5 * rms(torch.addmm(c, seed, w), dx64 + c.double()) + 1e-8)
                    # Rows into a larger output (the chunked linear), decided on all rows: bitwise the same rows.
                    out = torch.empty(rows + 8, n, device="cuda")
                    wg.linear_into(out[4:4 + rows], x, w, bias, rows=rows)
                    self.assertTrue(torch.equal(out[4:4 + rows], y.detach()))
                    self.assertTrue(torch.equal(wg.linear(x[5:60].contiguous(), w, bias), y.detach()[5:60]))
                    seeds = _leaf(seed)
                    (graph_dx,) = torch.autograd.grad(wg.linear(xs, w, bias), (xs,), seeds, create_graph=True)
                    self.assertIsNotNone(graph_dx.grad_fn)  # differentiable: torch's formula, not the kernel
                    self.assertTrue(torch.equal(graph_dx, seed @ w))
                    with mock.patch.object(wg, "triton_dot_input_precision", lambda site: "tf32"):
                        self.assertTrue(1e-5 < rms(wg.linear(x, w, bias), y64) < 1e-3)
            # An in-place update and a rebinding of the storage: the cached split follows the weight.
            w.mul_(0.5)
            torch.testing.assert_close(wg.linear(x, w, bias).double(), x.double() @ w.double().t() + bias.double(),
                                       rtol=1e-5, atol=1e-5)
            w.data = torch.randn(n, k, generator=generator).cuda() / k ** 0.5
            torch.testing.assert_close(wg.linear(x, w, bias).double(), x.double() @ w.double().t() + bias.double(),
                                       rtol=1e-5, atol=1e-5)
            # The split (and its descriptors) is released with its weight.
            fresh = torch.randn(128, 128, device="cuda")
            split = wg._split(fresh, False)
            wg._b_descriptors(split, 128, True)
            hi = weakref.ref(split.hi)
            del fresh, split
            gc.collect()
            self.assertIsNone(hi())
            # Widths without a tile and unaligned operands decline (cuBLAS), they do not fail.
            x = torch.randn(301, 128, device="cuda")
            self.assertEqual(wg.wide_gemm_decline(x, torch.randn(96, 128, device="cuda"), False), "unsupported_shape")
            self.assertEqual(wg.wide_gemm_decline(x, torch.randn(128, 128, device="cuda"), False,
                                                  bias=torch.randn(129, device="cuda")[1:]), "layout")
            torch.testing.assert_close(wg.linear(x, torch.ones(96, 128, device="cuda")), x @ torch.ones(128, 96, device="cuda"))

    def test_captured_replay_matches_eager(self):
        from fastmd.engine.kernels import triton_row_gemm as rg
        from fastmd.engine.kernels import triton_wide_gemm as wg
        from fastmd.models.matris.op.triton_edge_ffn import fused_edge_ffn

        generator = _generator(702)
        rows = 4099
        x = torch.randn(rows, 32, generator=generator).cuda()
        w = (torch.randn(128, 32, generator=generator) * 0.2).cuda()
        g = torch.randn(rows, 64, generator=generator).cuda()
        wa = (torch.randn(64, 32, generator=generator) * 0.2).cuda()
        fx, w1, w2, seed = self._ffn(rows, generator)
        fx = _leaf(fx.contiguous())

        xw = torch.randn(rows, 128, generator=generator).cuda()
        ww = (torch.randn(256, 128, generator=generator) / 128 ** 0.5).cuda()
        # The dispatch: a linear forward and its input gradient (on autograd's thread), a backward matmul.
        xl = _leaf(torch.randn(rows, 128, generator=generator).cuda())
        wl, wm = ((torch.randn(128, 128, generator=generator) / 128 ** 0.5).cuda() for _ in range(2))
        gl, gm = (torch.randn(rows, 128, generator=generator).cuda() for _ in range(2))
        # A column range of the product (a recompute recipe's rebuild of part of a merged projection) is bitwise
        # the wide product's columns, also with a narrower tile.
        full = wg._launch(xw, wg._split(ww, False), "tf32x3")
        for start, size in ((0, 64), (64, 64), (128, 128), (0, 128), (64, 128)):
            part = wg._launch(xw, wg._split(ww.narrow(0, start, size), False), "tf32x3")
            self.assertTrue(torch.equal(part, full[:, start:start + size]), (start, size))

        def run():
            y = fused_edge_ffn(fx, w1, w2, "tf32x3")
            yl = wg.linear(xl, wl)
            with torch.no_grad():  # as in a Function's backward
                ym = wg.matmul(gm, wm, enabled=wg.dispatch_on())
            return (rg.row_linear_forward(x, w, "tf32x3"), rg.row_addmm(x, g, wa), y,
                    *torch.autograd.grad(y, (fx,), seed), wg._launch(xw, wg._split(ww, False), "tf32x3"),
                    yl, *torch.autograd.grad(yl, (xl,), gl), ym)

        # Thresholds (captured, eager) = (rows, 2 rows): eager runs cuBLAS and warms the splits, the capture
        # takes the 3xTF32 GEMMs.
        for name, value in (("_TRITON_WIDE_GEMM", True), ("_BAND", (rows, 2 * rows))):
            patcher = mock.patch.object(wg, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        # Above the eager threshold an eager merged product also builds its recipes' column-range splits, which a
        # later capture requires.
        wp = ww.clone()
        with mock.patch.object(wg, "_BAND", (rows, rows)):
            wg.linear(xw, wp, parts=((0, 128), (128, 128)))
        for start in (0, 128):
            self.assertIsNotNone(wg._split(wp.narrow(0, start, 128), False, build=False), start)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        from fastmd.engine.cache_provenance import StaleCaptureError, capture_retention
        graph = torch.cuda.CUDAGraph()
        with capture_retention(graph, check_replay=True), torch.cuda.graph(graph):
            static = run()
        derived = [t for group in graph._matris_derived for t in group]
        # The capture keeps every split it read alive (also if the cache is rebuilt), the input gradient's
        # (read on autograd's thread) too ...
        for weight, transposed in ((ww, False), (wl, False), (wl, True), (wm, True)):
            self.assertTrue(any(t is wg._split(weight, transposed).hi for t in derived), (weight.shape, transposed))
        for target in (x, g, fx.detach(), seed, xw, xl.detach(), gl, gm):
            target.copy_(target.flip(0).contiguous())
        with mock.patch.object(wg, "_BAND", (0, 0)):  # eager on the 3xTF32 GEMMs too (no capture decline)
            expected = run()
        graph.replay()
        torch.cuda.synchronize()
        for got, want in zip(static, expected):
            self.assertTrue(torch.equal(got, want))
        # ... and a replay after the weight changed raises instead of reading a stale split.
        ww.mul_(2.0)
        with self.assertRaises(StaleCaptureError):
            graph.replay()


# 5. K5: runtime row counts ----------------------------------------------------


@perf_tier  # JIT recompile counts: a structural perf guard
@unittest.skipUnless(_CUDA, "CUDA is required")
class RuntimeRowCountTest(_MatmulHighest):
    """After warming two row counts, new row counts must not compile."""

    def _assert_no_new_compiles(self, kernels, run):
        for rows in _WARM_ROWS:
            run(rows)
        torch.cuda.synchronize()
        for rows in _NEW_ROWS:
            with _TritonCompileCounter(kernels) as counter:
                outputs = run(rows)
                torch.cuda.synchronize()
            with self.subTest(rows=rows, check="no new compilation"):
                self.assertEqual(counter.count, 0, counter.describe())
            with self.subTest(rows=rows, check="values"):
                for name, actual, expected, atol in outputs:
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-4,
                                               msg=lambda m, name=name: f"{name} rows={rows}: {m}")

    def test_counter_detects_constexpr_row_specialization_control(self):
        import triton
        import triton.language as tl

        @triton.jit
        def constexpr_rows(x, N: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(x + offs, tl.load(x + offs, mask=offs < N) + 1.0, mask=offs < N)

        @triton.jit
        def plain_runtime_rows(x, N, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(x + offs, tl.load(x + offs, mask=offs < N) + 1.0, mask=offs < N)

        @triton.jit(do_not_specialize=["N"])
        def runtime_rows(x, N, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(x + offs, tl.load(x + offs, mask=offs < N) + 1.0, mask=offs < N)

        def launch(kernel, rows):
            x = torch.zeros(rows, device="cuda")
            kernel[(triton.cdiv(rows, 64),)](x, rows, BLOCK=64)
            return x

        # A plain runtime int is value-specialized (==1, %16==0): the counter
        # must see those compiles too, not only constexpr ones.
        for kernel, expect_compiles in ((constexpr_rows, True), (plain_runtime_rows, True),
                                        (runtime_rows, False)):
            for rows in _WARM_ROWS:
                launch(kernel, rows)
            with _TritonCompileCounter({"kernel": kernel}) as counter:
                outputs = [launch(kernel, rows) for rows in _NEW_ROWS]
                torch.cuda.synchronize()
            with self.subTest(kernel=kernel.fn.__name__):
                self.assertEqual(counter.count > 0, expect_compiles, counter.describe())
                self.assertEqual(len(counter.compiled) > 0, expect_compiles, counter.describe())
                self.assertEqual(sum(counter.new_cache_entries.values()) > 0, expect_compiles,
                                 counter.describe())
                for x in outputs:
                    self.assertTrue(bool((x == 1).all()))

    def test_row_gemm_and_edge_ffn_rows(self):
        from fastmd.models.matris.op import triton_edge_ffn as ffn
        from fastmd.engine.kernels import triton_row_gemm as rg

        w = torch.randn(128, 32, device="cuda") * 0.2
        wp = torch.randn(64, 32, device="cuda") * 0.2
        wa = torch.randn(64, 32, device="cuda") * 0.2
        w1, w2 = torch.randn(32, 32, device="cuda") * 0.2, torch.randn(32, 32, device="cuda") * 0.2

        def run(rows):
            x, g = torch.randn(rows, 32, device="cuda"), torch.randn(rows, 64, device="cuda")
            xs = _leaf(x)
            y = ffn.fused_edge_ffn(xs, w1, w2)
            seed = torch.randn_like(y)
            (grad,) = torch.autograd.grad(y, (xs,), seed)
            ref = torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x, w1)), w2)
            return [("tiles", rg.row_linear_forward(x, w), x @ w.t(), 1e-4),
                    ("persistent", rg.row_linear_forward(x, wp), x @ wp.t(), 1e-4),
                    ("addmm", rg.row_addmm(x, g, wa), x + g @ wa, 1e-4),
                    ("edge_ffn", y, ref, 1e-4), ("edge_ffn_dx_finite", grad.isfinite().all().float(),
                                                torch.ones((), device="cuda"), 0)]

        self._assert_no_new_compiles({"tiles": rg._row_gemm_tiles, "persistent": rg._row_gemm_persistent,
                                      "ffn_fwd": ffn._edge_ffn_fwd, "ffn_bwd": ffn._edge_ffn_bwd}, run)

    def test_residual_add_rows(self):
        from fastmd.models.matris.op import triton_residual_add as op

        dim = 32
        weight = torch.randn(1, dim, device="cuda")

        def run(rows):
            delta, residual = torch.randn(rows, dim, device="cuda"), torch.randn(rows, dim, device="cuda")
            fused = [_leaf(delta), _leaf(residual)]
            out = op.residual_add(*fused, weight)
            seed = torch.randn_like(out)
            grads = torch.autograd.grad(out, fused, seed)
            return [("forward", out, delta + weight * residual, 1e-6),
                    ("grad_delta", grads[0], seed, 0), ("grad_residual", grads[1], seed * weight, 1e-6)]

        self._assert_no_new_compiles({"fwd": op._residual_add_fwd, "bwd": op._residual_add_bwd}, run)

    def test_line_envelope_rows(self):
        from fastmd.models.matris.op import triton_line_envelope as op

        dim, base_rows = 32, 64
        base = torch.randn(base_rows, dim, device="cuda")

        def run(rows):
            source = torch.randint(0, base_rows, (rows,), device="cuda")
            target = torch.randint(0, base_rows, (rows,), device="cuda")
            fused, ref = _leaf(base), _leaf(base)
            out = op.line_envelope_product(fused, source, target)
            expected = ref[source] * ref[target]
            seed = torch.randn_like(out)
            return [("forward", out, expected, 1e-6),
                    ("grad_base", torch.autograd.grad(out, fused, seed)[0],
                     torch.autograd.grad(expected, ref, seed)[0], 1e-4)]

        self._assert_no_new_compiles({"fwd": op._line_envelope_fwd, "bwd": op._line_envelope_bwd}, run)

    def test_directed_pair_rows(self):
        from fastmd.models.matris.op import triton_directed_pair_aggregate as op

        dim = 32
        generator = _generator(300)

        def run(u):
            pair = _random_pair_index(u, interleaved=False, generator=generator, device="cuda")
            d2u = _pair_d2u(pair, 2 * u)
            directed, undirected = torch.randn(2 * u, dim, device="cuda"), torch.randn(u, dim, device="cuda")
            fused, ref = _leaf(directed), _leaf(directed)
            average = op.directed_pair_average(fused, pair)
            average_ref = (ref[pair[:, 0]] + ref[pair[:, 1]]) * 0.5
            seed = torch.randn_like(average)
            outputs = [("average", average, average_ref, 1e-6),
                       ("grad_average", torch.autograd.grad(average, fused, seed)[0],
                        torch.autograd.grad(average_ref, ref, seed)[0], 1e-6)]
            fused, ref = _leaf(undirected), _leaf(undirected)
            expand = op.undirected_pair_expand(fused, pair, 2 * u)
            expand_ref = ref[d2u]
            seed = torch.randn_like(expand)
            return outputs + [("expand", expand, expand_ref, 0),
                              ("grad_expand", torch.autograd.grad(expand, fused, seed)[0],
                               torch.autograd.grad(expand_ref, ref, seed)[0], 1e-6)]

        kernels = {name: getattr(op, name) for name in (
            "_directed_pair_average_fwd", "_directed_pair_average_bwd",
            "_undirected_pair_expand_fwd", "_undirected_pair_expand_bwd")}
        self._assert_no_new_compiles(kernels, run)

    def test_weighted_segment_sum_rows(self):
        from fastmd.models.matris.op import triton_weighted_segment_sum as op

        dim, segments = 32, 7

        def run(rows):
            weight, value = torch.randn(rows, dim, device="cuda"), torch.randn(rows, dim, device="cuda")
            segment = torch.randint(0, segments, (rows,), device="cuda")
            fused, ref = [_leaf(weight), _leaf(value)], [_leaf(weight), _leaf(value)]
            out = op.weighted_segment_sum(*fused, segment, segments)
            expected = torch.zeros(segments, dim, device="cuda").index_add(0, segment, ref[0] * ref[1])
            seed = torch.randn_like(out)
            grads, ref_grads = torch.autograd.grad(out, fused, seed), torch.autograd.grad(expected, ref, seed)
            return [("forward", out, expected, 1e-4), ("grad_weight", grads[0], ref_grads[0], 1e-6),
                    ("grad_value", grads[1], ref_grads[1], 1e-6)]

        self._assert_no_new_compiles({"fwd": op._weighted_segsum_fwd, "bwd": op._weighted_segsum_bwd}, run)

    def test_threebody_fourier_rows(self):
        from fastmd.models.matris.op import triton_threebody_fourier as op

        edges = 64
        unit = F.normalize(torch.randn(edges, 3, device="cuda"), dim=1)
        freqs = torch.tensor([0.833447, 1.5861181, 2.5042067], device="cuda")
        scale = 3.141592653589793 ** 0.5

        def reference(unit_vec, target, source):
            theta = (unit_vec[target] * unit_vec[source]).sum(1) * (1.0 - 1.0e-6)
            angle = torch.acos(theta)
            outer = torch.outer(angle, freqs)
            constant = torch.full_like(angle, 2 ** -0.5)[:, None]
            return torch.cat((constant, torch.sin(outer), torch.cos(outer)), 1) / scale

        def run(rows):
            target = torch.randint(0, edges, (rows,), device="cuda")
            source = (target + torch.randint(1, edges, (rows,), device="cuda")) % edges
            fused, ref = _leaf(unit), _leaf(unit)
            out = op.threebody_fourier_basis(fused, target, source, freqs, scale)
            expected = reference(ref, target, source)
            seed = torch.randn_like(out)
            grad = torch.autograd.grad(out, fused, seed)[0]
            ref_grad = torch.autograd.grad(expected, ref, seed)[0]
            return [("forward", out, expected, 1e-5),
                    ("grad_unit", grad, ref_grad, 1e-3 * (1.0 + float(ref_grad.abs().max())))]

        self._assert_no_new_compiles({"fwd": op._threebody_fourier_fwd, "bwd": op._threebody_fourier_bwd}, run)

    def _decomposed_rows(self, parts):
        from fastmd.engine.kernels import decomposed_indexed_cat_silu_linear as op

        def run(rows):
            case = _DecomposedCase(parts, rows=rows, gathered_rows=40, offset=rows)
            (core, gate), grads = case.run(index_a_sorted=True)
            (ref_core, ref_gate), ref_grads = case.reference()
            return [("core", core, ref_core, 1e-5), ("gate", gate, ref_gate, 1e-5)] + [
                (f"grad_{k}", grad, ref_grad, 1e-4) for k, (grad, ref_grad) in enumerate(zip(grads, ref_grads))]

        prefix = f"_cat{parts}_gather_add_silu_"
        kernels = {name: getattr(op, prefix + name) for name in ("fwd", "bwd")}
        self._assert_no_new_compiles(kernels, run)

    def test_decomposed_cat_rows(self):
        for parts in (3, 4):
            with self.subTest(parts=parts):
                self._decomposed_rows(parts)

    def test_paired_segment_attention_backward_rows(self):
        from fastmd.models.matris.op import triton_segment_attention as op

        dim = 32

        def run(rows):
            segments = max(2, rows // 4)
            seg_a = torch.randint(0, segments, (rows,), device="cuda")
            seg_b = torch.randint(0, segments, (rows,), device="cuda")
            tensors = [torch.randn(rows, dim, device="cuda") * s for s in (3, 3, 1)]
            fused, ref = [_leaf(t) for t in tensors], [_leaf(t) for t in tensors]
            out = op.paired_segment_attention(*fused, seg_a, seg_b, segments)
            expected = (_torch_segment_attention(ref[0], ref[2], seg_a, segments),
                        _torch_segment_attention(ref[1], ref[2], seg_b, segments))
            seeds = (torch.randn_like(out[0]), torch.randn_like(out[1]))
            grads, ref_grads = torch.autograd.grad(out, fused, seeds), torch.autograd.grad(expected, ref, seeds)
            return [("out_a", out[0], expected[0], 1e-5), ("out_b", out[1], expected[1], 1e-5)] + [
                (name, grad, ref_grad, 1e-5)
                for name, grad, ref_grad in zip(_PAIRED_NAMES[2:], grads, ref_grads)]

        kernels = {name: getattr(op, name) for name in (
            "_paired_csr_segment_attention_fwd", "_paired_segment_attention_bwd_rows")}
        self._assert_no_new_compiles(kernels, run)

    def test_zero_skip_index_add_elements(self):
        from fastmd.engine.kernels import triton_zero_skip_index as op

        width, count = 7, 13  # odd width: ELEMENTS takes both divisibility classes

        def run(rows):
            values = torch.randn(rows, width, device="cuda")
            values[::3] = 0.0
            indices = torch.randint(0, count, (rows,), device="cuda")
            summed = op.zero_skip_index_sum(values, indices, count)
            expected = torch.zeros(count, width, device="cuda").index_add(0, indices, values)
            source = _leaf(torch.randn(count, width, device="cuda"))
            gathered = op.zero_skip_index_select(source, indices)
            seed = torch.randn_like(gathered)
            grad, = torch.autograd.grad(gathered, source, seed)
            return [("sum", summed, expected, 1e-5),
                    ("gather_grad", grad, torch.zeros_like(grad).index_add(0, indices, seed), 1e-5)]

        self._assert_no_new_compiles({"index_add": op._zero_skip_index_add_kernel}, run)

    def test_decomposed_nhc_rows(self):
        from types import SimpleNamespace

        from ase import units
        from fastmd.execution import gpu_nhc
        from fastmd.engine.comm import LocalCommunicator

        kernels = {"ke2": gpu_nhc._nhc_ke2_kernel, "rows": gpu_nhc._nhc_rows_kernel,
                   "chain": gpu_nhc._nhc_chain_kernel}

        def rows_of(rows):
            generator = torch.Generator(device="cuda").manual_seed(_SEED + rows)
            q, p = (torch.randn(rows, 3, device="cuda", dtype=torch.float64, generator=generator)
                    for _ in range(2))
            m = torch.rand(rows, device="cuda", dtype=torch.float64, generator=generator) * 20 + 1
            return q, p, torch.randn(rows, 3, device="cuda", generator=generator), m

        def make(backend, comm, rows, **kw):
            # The one NHC class with capacity slots (protocol="decomposed"), float32 model forces: the
            # pre-force kick reads their float64 cache.
            q, p, f, m = rows_of(rows)
            return gpu_nhc.GPUNoseHooverChainNVT(
                m, q, p, forces=f, temperature_K=300.0, timestep=units.fs, tdamp=100 * units.fs, comm=comm,
                n_global=8192, protocol="decomposed", device="cuda", force_dtype=torch.float32,
                backend=backend, **kw), f

        def step(nhc, f):
            nhc.step_pre_force()
            nhc.step_post_force(f)

        # Warmup coverage (constexprs unique in the process): a non-root rank's
        # warmup issues no collective, and its real steps then compile nothing.
        for ke2_update, tchain in (("scaled", 3), ("recompute", 4)):
            comm = SimpleNamespace(rank=1, world_size=2, allreduce_sum_=mock.Mock())
            nhc, f = make("triton", comm, 17, tchain=tchain, tloop=4, ke2_update=ke2_update)
            nhc.warmup_compile()
            comm.allreduce_sum_.assert_not_called()
            with _TritonCompileCounter(kernels) as counter:
                step(nhc, f)
                # The P>1 in-graph halves: every capacity row, the owned-row mask (HAS_MASK kernels).
                rows = nhc.capacity
                padded = torch.zeros(rows, 3, device="cuda")
                padded[:17] = f
                nhc.step_pre_force_rows(rows)
                nhc.step_post_force_rows(padded, None, rows)
                torch.cuda.synchronize()
            with self.subTest(warmup=ke2_update):
                self.assertEqual(counter.count, 0, counter.describe())

        def run(rows):
            outputs = []
            for backend in ("triton", "torch"):
                nhc, f = make(backend, LocalCommunicator(), rows, tchain=2, tloop=3)
                step(nhc, f)
                outputs.append((nhc.positions, nhc.momenta, nhc.state.chain))
            return [(name, *pair, 1e-10) for name, pair in zip(("q", "p", "chain"), zip(*outputs))]

        self._assert_no_new_compiles(kernels, run)


# 6. Checkpoint RNG state -----------------------------------------------------


class CheckpointRngStateTest(unittest.TestCase):
    """Eval checkpointing must not save/restore RNG state (capture safety)."""

    def test_wrappers_pass_preserve_rng_state_equal_to_training(self):
        from fastmd.models.matris import interaction_block as ib

        block = ib.Interaction_Block(node_feat_dim=4, edge_feat_dim=4, three_body_feat_dim=4,
                                     num_radial=3, num_angular=3, enable_checkpoint=True)
        calls = []

        def recording_checkpoint(function, *args, **kwargs):
            calls.append(kwargs)
            return function(*args)

        def layer(*args, **kwargs):
            return "nodes", "edges"

        x = torch.zeros(2, 4)
        with mock.patch.object(ib, "checkpoint", recording_checkpoint):
            for training in (False, True):
                block.train(training)
                calls.clear()
                self.assertEqual(block.wrapper_attn_layer(
                    attn_layer=layer, node_feat=x, edge_feat=x, graph={}, use_checkpoint=True),
                    ("nodes", "edges"))
                self.assertEqual(block.wrapper_refine_layer(
                    refine_layer=layer, node_feat=x, edge_feat=x, smooth_weight=x, graph={},
                    use_checkpoint=True), ("nodes", "edges"))
                self.assertEqual(len(calls), 2)
                for wrapper, kwargs in zip(("wrapper_attn_layer", "wrapper_refine_layer"), calls):
                    with self.subTest(training=training, wrapper=wrapper):
                        self.assertIs(kwargs.get("use_reentrant"), False)
                        self.assertEqual(kwargs.get("preserve_rng_state", True), training, kwargs)

    @unittest.skipUnless(_CUDA, "CUDA is required")
    def test_checkpointed_block_capture_replay_matches_eager(self):
        from ase import Atoms
        import numpy as np

        from fastmd.engine._graph_capture import capture_graph
        from fastmd.graph import atoms_to_graph_gpu
        from fastmd.graph.gpu_graph_builder import op_available
        from fastmd.models.matris.model import MatRIS
        from fastmd.models.matris.processgraph import process_graphs

        if not op_available:
            self.skipTest("CUDA graph builder is unavailable")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), \
                contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(_SEED)
            model = MatRIS(num_layers=2, node_feat_dim=32, edge_feat_dim=32, three_body_feat_dim=32,
                           mlp_hidden_dims=(32, 32), num_radial=8, num_angular=8, max_n=3, max_l=3,
                           pairwise_cutoff=2.0, three_body_cutoff=1.5, enable_compile=False,
                           enable_checkpoint=True)
        model = model.eval().requires_grad_(False).cuda()
        # 64 atoms on a jittered 1.25 A lattice: 576 bonds, 1920 angles.
        rng = np.random.default_rng(5)
        grid = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3) * 1.25
        atoms = Atoms(numbers=rng.choice([1, 6, 8], len(grid)), cell=[5.0] * 3, pbc=True,
                      positions=grid + rng.uniform(-0.05, 0.05, grid.shape))
        graph = atoms_to_graph_gpu(atoms, atom_graph_cutoff=2.0, line_graph_cutoff=1.5,
                                   check_isolated_atoms=False)
        batch_graph = process_graphs([graph], compute_stress=False)
        with torch.no_grad():
            node_feat = model.embed_atoms(batch_graph["atomic_numbers"])
            edge_feat, smooth_weight = model.edge_embedding(graphs=batch_graph)
            threebody_feat = model.three_body_embedding(graphs=batch_graph)
        smooth_weight = {key: value.detach() for key, value in smooth_weight.items()}
        inputs = [_leaf(t) for t in (node_feat, edge_feat, threebody_feat)]
        block = model.interaction_block[0]  # not the last block: all outputs live
        self.assertTrue(block.enable_checkpoint)
        self.assertFalse(block.training)
        generator = _generator(400)
        cotangents = []

        def step():
            outputs = block(batch_graph, *inputs, smooth_weight)
            if not cotangents:
                cotangents.extend(torch.randn(o.shape, generator=generator).cuda() for o in outputs)
            grads = torch.autograd.grad(outputs, inputs, cotangents)
            return [t.detach().clone() for t in (*outputs, *grads)]

        # Deterministic mode with the default (eager) kernel set makes eager
        # bitwise reproducible, so replay must be bitwise. The model_graph
        # fused set reduces with atomics: on this graph eager-vs-eager gradients
        # already differ by ~5e-4 (values up to ~8), too noisy for this check.
        deterministic = (torch.are_deterministic_algorithms_enabled(),
                         torch.is_deterministic_algorithms_warn_only_enabled())
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            eager = step()
            noise = max(float((a - b).abs().max()) for a, b in zip(eager, step()))
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                step()
            torch.cuda.current_stream().wait_stream(side)
            captured, capture_error, cuda_graph = None, None, torch.cuda.CUDAGraph()
            try:
                with capture_graph(cuda_graph):
                    captured = step()
            except Exception as exc:
                capture_error = f"{type(exc).__name__}: {exc}"
        finally:
            torch.use_deterministic_algorithms(deterministic[0], warn_only=deterministic[1])
        torch.cuda.synchronize()

        with self.subTest(check="capture succeeds"):
            self.assertIsNone(capture_error)
        if captured is None:
            return
        for tensor in captured:
            tensor.fill_(float("nan"))
        cuda_graph.replay()
        torch.cuda.synchronize()
        names = ("node", "edge", "threebody", "grad_node", "grad_edge", "grad_threebody")
        for name, replayed, expected in zip(names, captured, eager):
            with self.subTest(check="replay matches eager", tensor=name, eager_noise=noise):
                if noise == 0.0:
                    self.assertTrue(torch.equal(replayed, expected), f"{name} replay is not bitwise eager")
                else:
                    # Atomic reductions: eager itself is not bitwise reproducible.
                    torch.testing.assert_close(replayed, expected, atol=1e-6, rtol=0)
        del cuda_graph


if __name__ == "__main__":
    unittest.main()
