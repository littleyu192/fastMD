"""OpPlanRecorder (unified engine step 2.0): recorder contract and kernel-entry records.

Tier 1 (default): on CUDA a kernel entry given a layout it does not support
declines instead of returning a wrong tensor, and a cold merged-weight cache
is not built while a CUDA graph is being captured.

Tier 3 (``MATRIS_RUN_PERF_TIER=1``, tests/matris_test_tiers.py): the recorder
guarantees the G-S3 goldens rely on. CPU tests cover cleanup after an
exception, profiler-based aten counts equal to an unrecorded run, no residue
in torch.compile programs, no recorder calls while inactive.
CUDA tests
(skipped without a GPU) drive every instrumented autograd Function forward and
backward, record declines as fallbacks, CUDA-graph capture/replay, unchanged
allocations while recording and a real RDX checkpoint forward.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import unittest
import warnings
from unittest.mock import patch

import torch
from torch import nn

from fastmd.engine import recorder as _recorder
from fastmd.engine.recorder import OpPlanRecorder, canonical_json
from matris_test_assets import RDX_1344, RDX_CHECKPOINT, test_asset, test_assets_available
from matris_test_tiers import perf_tier
from single_gpu_fixtures import frozen_fixture, toy_model

_MODULE_HOOKS = torch.nn.modules.module


def _hook_counts():
    return (len(_MODULE_HOOKS._global_forward_pre_hooks),
            len(_MODULE_HOOKS._global_forward_hooks),
            torch._C._len_torch_dispatch_stack(),
            torch.autograd._profiler_enabled())


class _Scaled(torch.autograd.Function):
    """A Function instrumented with the production idiom."""

    @staticmethod
    def forward(ctx, x, scale):
        rec = _recorder._ACTIVE
        if rec is not None:
            ctx._opsel_site = rec.site()
            rec.record("scaled", "torch", {"rows": x.shape[0], "scale": scale})
        ctx.scale = scale
        # Saved so that an early-stopping checkpoint must recompute this.
        ctx.save_for_backward(x)
        return x * scale

    @staticmethod
    def backward(ctx, grad):
        ctx.saved_tensors  # unpacking triggers a pending checkpoint recompute
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("scaled", "torch", {"rows": grad.shape[0]}, phase="bwd",
                       site=getattr(ctx, "_opsel_site", None))
        return grad * ctx.scale, None


class _ScaledBlock(nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.scale = scale

    def forward(self, x):
        return _Scaled.apply(self.linear(x), self.scale)


def _entries(snapshot, section=0):
    return snapshot["sections"][section]["entries"]


def _keys(entries):
    return [(entry["op"], entry["impl"], entry["phase"], entry["site"]) for entry in entries]


@perf_tier
class OpPlanRecorderApiTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNone(_recorder._ACTIVE)
        self.baseline_hooks = _hook_counts()

    def tearDown(self):
        self.assertIsNone(_recorder._ACTIVE)
        self.assertEqual(_hook_counts(), self.baseline_hooks)

    def test_hooks_and_dispatch_mode_are_removed_after_an_exception(self):
        model = nn.Sequential(_ScaledBlock(), _ScaledBlock())
        with self.assertRaisesRegex(ValueError, "boom"):
            with OpPlanRecorder(root=model) as rec:
                model(torch.randn(2, 3))
                raise ValueError("boom")
        self.assertEqual(len(_entries(rec.snapshot())), 2)
        # A recorder that failed can be replaced immediately.
        with OpPlanRecorder():
            pass

    def test_disabled_path_makes_no_recorder_calls(self):
        model = toy_model(smoothed=True)
        _, graph = frozen_fixture()
        calls = []
        with ExitStack() as stack:
            for name in ("record", "fallback", "site", "mark", "_collect", "_pre_hook"):
                stack.enter_context(patch.object(
                    OpPlanRecorder, name, side_effect=lambda *a, **k: calls.append(a)))
            self.assertIsNone(_recorder._ACTIVE)
            model([graph], task="ef")
        self.assertEqual(calls, [])

    def test_torch_compile_records_at_run_time_and_is_unchanged_after_exit(self):
        from torch._dynamo.utils import counters
        from fastmd.engine.dispatch import current_policy, dispatcher_for
        from fastmd.engine.layout import ExecutionLayout

        dispatcher = dispatcher_for(current_policy(), ExecutionLayout())

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.weight = nn.Parameter(torch.ones(1, 4), requires_grad=False)

            def forward(self, x):
                return dispatcher.residual(self.linear(x), x, self.weight)

        graphs, runs = [], []

        def backend(gm, example_inputs):
            graphs.append(len(gm.graph.nodes))

            def run(*args):
                runs.append(len(gm.graph.nodes))
                return gm(*args)

            return run

        torch._dynamo.reset()  # a clean slate once, never between the phases below
        compiled = torch.compile(Block(), backend=backend, dynamic=True)
        x = torch.randn(6, 4)

        def breaks():
            return sum(counters["graph_break"].values())

        compiled(x)
        compiled(x)
        # Instrumentation points constant-fold: one graph, no breaks.
        baseline = list(graphs)
        self.assertEqual((len(baseline), runs), (1, baseline * 2))
        before = breaks()
        with warnings.catch_warnings():
            # torch notes that global hooks also fire for the OptimizedModule.
            warnings.simplefilter("ignore", UserWarning)
            for aten in (False, True):
                with OpPlanRecorder(root=compiled, aten=aten) as rec:
                    compiled(x)
                # Recorded at execution (a graph break at the record), not only
                # while dynamo traced; aten=True does not make dynamo skip.
                self.assertEqual(_keys(_entries(rec.snapshot())),
                                 [("residual_add", "torch", "fwd", "Block")])
                self.assertGreater(breaks(), before)
                # After exit the recorder-free program runs again: recompiled
                # once into the same single graph, without the recording breaks.
                del graphs[:], runs[:]
                broken = breaks()
                compiled(x)
                compiled(x)
                self.assertEqual((graphs, runs, breaks()), (baseline, baseline * 2, broken))
        torch._dynamo.reset()

    def test_aten_counts_match_a_plain_profiler(self):
        """The recorder observes exactly what an unrecorded profiled run executes."""
        model = toy_model(smoothed=True)
        _, graph = frozen_fixture()
        with torch.autograd.profiler.profile(use_cpu=True) as prof:
            model([graph], task="ef")
        plain = {}
        for event in prof.kineto_results.events():
            if _recorder._OP_EVENT.match(event.name()):
                plain[event.name()] = plain.get(event.name(), 0) + 1
        with OpPlanRecorder(root=model) as rec:
            model([graph], task="ef")
        recorded = {}
        for counts in rec.snapshot()["sections"][0]["aten_counts"].values():
            for op, count in counts.items():
                recorded[op] = recorded.get(op, 0) + count
        self.assertEqual(recorded, plain)
        # Backward formulas run in place, as without a recorder (a dispatch
        # mode would switch index_add_ to the out-of-place index_add).
        self.assertIn("aten::index_add_", recorded)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA kernel entries need a GPU")
class OpPlanRecorderKernelTest(unittest.TestCase):
    """CUDA kernel entries: declines and capture (tier 1), their records (tier 3)."""

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _run(self, calls):
        """calls: {name: fn() -> output tensor(s)}; each fn runs inside its own site."""
        probes = nn.ModuleDict({name: _Probe(fn) for name, fn in calls.items()})
        with OpPlanRecorder(root=probes) as rec:
            for name in calls:
                rec.mark(name)
                outputs = probes[name]()
                outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                leaves = probes[name].leaves
                grads = torch.autograd.grad(
                    [out for out in outputs if out.requires_grad],
                    leaves,
                    [torch.randn_like(out) for out in outputs if out.requires_grad],
                    allow_unused=True,
                )
                self.assertTrue(any(grad is not None for grad in grads))
        torch.cuda.synchronize()
        snapshot = rec.snapshot()
        self.assertEqual(snapshot["invalid_params"], 0)
        return {section["label"]: section for section in snapshot["sections"]}

    def _assert_fwd_bwd(self, section, op, impl, site):
        keys = _keys(section["entries"])
        self.assertIn((op, impl, "fwd", site), keys, keys)
        self.assertIn((op, impl, "bwd", site), keys, keys)
        for entry in section["entries"]:
            if entry["op"] == op:
                self.assertTrue(entry["params"], entry)

    @perf_tier
    def test_every_instrumented_function_records_forward_and_backward(self):
        from fastmd.models.matris.op.compiled_lowerings import compiled_residual_add
        from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import (
            decomposed_indexed_cat3_silu_linear, decomposed_indexed_cat4_silu_linear,
        )
        from fastmd.models.matris.op.fuse_sigmoid_op import fused_sigmoid
        from fastmd.models.matris.op.fused_silu_op import fused_silu
        from fastmd.engine.kernels.merged_frozen_projections import (
            merged_frozen_projections, projection_with_alias,
        )
        from fastmd.models.matris.op.triton_directed_pair_aggregate import (
            directed_pair_average, undirected_pair_expand,
        )
        from fastmd.models.matris.op import triton_gated_ln
        from fastmd.models.matris.op.triton_gated_ln import fused_gated_ln, fused_gated_ln_residual
        from fastmd.models.matris.op.triton_gather_cat import aligned_gather_cat3, aligned_gather_cat4
        from fastmd.models.matris.op.triton_layernorm_act import ACT_SILU, fused_ln_act
        from fastmd.models.matris.op.triton_line_envelope import line_envelope_product
        from fastmd.models.matris.op.triton_residual_add import residual_add
        from fastmd.models.matris.op.triton_segment_attention import (
            paired_segment_attention, segment_attention, segment_attention_csr_metadata,
        )
        from fastmd.engine.kernels.triton_segment_softmax import (
            fused_segment_softmax, fused_segment_softmax_atomic,
        )
        from fastmd.models.matris.op.triton_threebody_fourier import threebody_fourier_basis
        from fastmd.models.matris.op.triton_weighted_segment_sum import weighted_segment_sum
        from fastmd.engine.kernels.triton_zero_skip_index import (
            zero_skip_index_select, zero_skip_index_sum,
        )

        dev, rows, dim, segments = self.device, 64, 32, 7

        def leaf(*shape):
            return torch.randn(*shape, device=dev, requires_grad=True)

        def frozen(*shape):
            return torch.randn(*shape, device=dev)

        def tile_1row(fn):  # 1-row gated_ln kernels for the forward only; the backward follows it
            with patch.object(triton_gated_ln, "_TILE_2D", False):
                return fn()

        segment = torch.randint(0, segments, (rows,), device=dev)
        pair = torch.randperm(rows, device=dev).reshape(rows // 2, 2)
        index_a = torch.sort(torch.randint(0, 16, (rows,), device=dev)).values
        index_b = torch.randint(0, 16, (rows,), device=dev)
        index_c = torch.randint(0, 16, (rows,), device=dev)
        sorted_segment = torch.sort(segment).values
        sorted_off = torch.searchsorted(
            sorted_segment, torch.arange(segments + 1, device=dev)).to(torch.int64)
        perm, off = segment_attention_csr_metadata(segment, segments)
        hidden = 8
        weights3 = tuple(frozen(2 * hidden, dim).contiguous() for _ in range(3))
        weights4 = tuple(frozen(2 * hidden, dim).contiguous() for _ in range(4))
        unit = torch.nn.functional.normalize(frozen(16, 3), dim=1).requires_grad_(True)
        freqs = torch.tensor([0.8, 1.6, 2.5], device=dev)
        w_cat = frozen(2 * hidden + 3 + 5, dim)

        calls = {
            "residual_add": lambda p: residual_add(p.leaf(rows, dim), p.leaf(rows, dim),
                                                   frozen(1, dim)),
            "pair_average": lambda p: directed_pair_average(p.leaf(rows, dim), pair),
            "pair_expand": lambda p: undirected_pair_expand(p.leaf(rows // 2, dim), pair, rows),
            "line_envelope": lambda p: line_envelope_product(p.leaf(16, dim), index_b, index_c),
            "weighted_segment_sum": lambda p: weighted_segment_sum(
                p.leaf(rows, dim), p.leaf(rows, dim), segment, segments),
            "segment_attention_computed": lambda p: segment_attention(
                p.leaf(rows, dim), p.leaf(rows, dim), segment, segments),
            "segment_attention_identity": lambda p: segment_attention(
                p.leaf(rows, dim), p.leaf(rows, dim), sorted_segment, segments, None,
                sorted_off, True),
            "paired_segment_attention": lambda p: paired_segment_attention(
                p.leaf(rows, dim), p.leaf(rows, dim), p.leaf(rows, dim), segment, sorted_segment,
                segments, perm, off, None, sorted_off, False, True),
            "segment_softmax_csr": lambda p: fused_segment_softmax(
                p.leaf(rows, dim), segment, segments),
            "segment_softmax_atomic": lambda p: fused_segment_softmax_atomic(
                p.leaf(rows, dim), segment, segments),
            "gated_ln": lambda p: fused_gated_ln(p.leaf(rows, dim), p.leaf(rows, dim),
                                                 frozen(dim), frozen(dim), frozen(dim),
                                                 frozen(dim)),
            "gated_ln_residual": lambda p: fused_gated_ln_residual(
                p.leaf(rows, dim), p.leaf(rows, dim), frozen(dim), frozen(dim), frozen(dim),
                frozen(dim), p.leaf(rows, dim), frozen(1, dim)),
            "gated_ln_residual_1row": lambda p: tile_1row(lambda: fused_gated_ln_residual(
                p.leaf(rows, dim), p.leaf(rows, dim), frozen(dim), frozen(dim), frozen(dim),
                frozen(dim), p.leaf(rows, dim), frozen(1, dim))),
            "layernorm_act": lambda p: fused_ln_act(p.leaf(rows, dim), frozen(dim), frozen(dim),
                                                    1e-5, ACT_SILU),
            "gather_cat3": lambda p: aligned_gather_cat3(p.leaf(rows, dim), p.leaf(16, dim),
                                                         index_a, index_b),
            "gather_cat4": lambda p: aligned_gather_cat4(
                p.leaf(rows, dim), p.leaf(16, dim), index_a, p.leaf(16, dim), index_b, index_c),
            "indexed_cat3": lambda p: decomposed_indexed_cat3_silu_linear(
                p.leaf(rows, dim), p.leaf(16, dim), index_a, index_b, weights3, None, True),
            "indexed_cat4": lambda p: decomposed_indexed_cat4_silu_linear(
                p.leaf(rows, dim), p.leaf(16, dim), index_a, p.leaf(16, dim), index_b, index_c,
                weights4, None, False, precomputed_p0=p.leaf(rows, 2 * hidden)),
            "merged_frozen_projections": lambda p: merged_frozen_projections(
                p.leaf(rows, dim), w_cat, (2 * hidden, 3, 5)),
            "projection_with_alias": lambda p: projection_with_alias(
                p.leaf(rows, dim), weights3[0]),
            "threebody_fourier_basis": lambda p: threebody_fourier_basis(
                p.use(unit), index_b, index_c, freqs, 3.141592653589793 ** 0.5),
            "zero_skip_index_sum": lambda p: zero_skip_index_sum(p.leaf(rows, dim), segment,
                                                                 segments),
            "zero_skip_index_select": lambda p: zero_skip_index_select(p.leaf(segments, dim),
                                                                       segment),
            "silu": lambda p: fused_silu(p.leaf(rows, dim)),
            "sigmoid": lambda p: fused_sigmoid(p.leaf(rows, dim)),
        }
        sections = self._run(calls)
        expected = {
            "residual_add": ("residual_add", "triton"),
            "pair_average": ("pair_average", "triton"),
            "pair_expand": ("pair_expand", "triton"),
            "line_envelope": ("line_envelope", "triton"),
            "weighted_segment_sum": ("weighted_segment_sum", "triton"),
            "segment_attention_computed": ("segment_attention", "triton_csr"),
            "segment_attention_identity": ("segment_attention", "triton_csr"),
            "paired_segment_attention": ("paired_segment_attention", "triton_csr"),
            "segment_softmax_csr": ("segment_softmax", "triton_csr"),
            "segment_softmax_atomic": ("segment_softmax", "triton_atomic"),
            "gated_ln": ("gated_ln", "triton"),
            "gated_ln_residual": ("gated_ln_residual", "triton"),
            "gated_ln_residual_1row": ("gated_ln_residual", "triton"),
            "layernorm_act": ("layernorm_act", "triton"),
            "gather_cat3": ("gather_cat3", "triton"),
            "gather_cat4": ("gather_cat4", "triton"),
            "indexed_cat3": ("cat3_gather_add_silu", "triton"),
            "indexed_cat4": ("cat4_gather_add_silu", "triton"),
            "merged_frozen_projections": ("merged_frozen_projections", "torch_linear"),
            "projection_with_alias": ("projection_with_alias", "torch_linear"),
            "threebody_fourier_basis": ("threebody_fourier_basis", "triton"),
            "silu": ("silu", "jiterator"),
            "sigmoid": ("sigmoid", "jiterator"),
        }
        for name, (op, impl) in expected.items():
            with self.subTest(call=name):
                self._assert_fwd_bwd(sections[name], op, impl, name)
        with self.subTest(call="params"):
            computed = sections["segment_attention_computed"]["entries"][0]["params"]
            identity = sections["segment_attention_identity"]["entries"][0]["params"]
            self.assertEqual((computed["csr"], computed["IDENTITY_PERM"]), ("computed", False))
            self.assertEqual((identity["csr"], identity["IDENTITY_PERM"]), ("identity", True))
            paired = sections["paired_segment_attention"]["entries"][0]["params"]
            self.assertEqual(paired["csr"], ["provided", "identity"])
            for name, tile in (("gated_ln_residual", "2d"), ("gated_ln_residual_1row", None)):
                self.assertEqual([entry["params"].get("tile") for entry in sections[name]["entries"]
                                  if entry["op"] == "gated_ln_residual"], [tile, tile], name)
            cat3 = [entry for entry in sections["indexed_cat3"]["entries"]
                    if entry["op"] == "indexed_cat3"]
            self.assertEqual(len(cat3), 1)
            self.assertEqual((cat3[0]["params"]["SORTED_A"], cat3[0]["params"]["precomputed_p0"],
                              cat3[0]["params"]["gemm_backend"]), (True, False, "torch"))
            cat4 = [entry for entry in sections["indexed_cat4"]["entries"]
                    if entry["op"] == "indexed_cat4"][0]["params"]
            self.assertEqual((cat4["SORTED_A"], cat4["precomputed_p0"]), (False, True))
            bwd = [entry for entry in sections["indexed_cat3"]["entries"]
                   if entry["op"] == "cat3_gather_add_silu" and entry["phase"] == "bwd"]
            self.assertTrue(bwd[0]["params"]["SORTED_A"])
        with self.subTest(call="zero_skip_index"):
            summed = _keys(sections["zero_skip_index_sum"]["entries"])
            self.assertIn(("zero_skip_index_add", "triton", "fwd", "zero_skip_index_sum"), summed)
            self.assertIn(("zero_skip_index_sum", "triton_row_gather", "bwd",
                           "zero_skip_index_sum"), summed)
            selected = _keys(sections["zero_skip_index_select"]["entries"])
            self.assertIn(("zero_skip_index_select", "triton_row_gather", "fwd",
                           "zero_skip_index_select"), selected)
            self.assertIn(("zero_skip_index_select", "zero_skip_index_sum", "bwd",
                           "zero_skip_index_select"), selected)
            # The VJP's own sum launch is attributed to the executing node.
            self.assertIn(("zero_skip_index_add", "triton", "bwd", "bwd:_IndexSelectBackward"),
                          selected)
        with self.subTest(call="compiled_disabled"):
            with OpPlanRecorder(aten=False) as rec:
                self.assertIsNone(compiled_residual_add(
                    frozen(rows, dim), frozen(rows, dim), frozen(1, dim)))
            self.assertEqual(rec.snapshot()["sections"][0]["fallback_counts"],
                             {"residual_add/compiled/disabled": 1})

    def _decline_inputs(self):
        """Inputs the decomposed cat3 entry does not support."""
        dev, rows, dim = self.device, 32, 16
        aligned = torch.randn(rows, 2 * dim, device=dev)[:, :dim]  # non-contiguous
        gathered = torch.randn(8, dim, device=dev)
        index = torch.randint(0, 8, (rows,), device=dev)
        weights = tuple(torch.randn(8, dim, device=dev) for _ in range(3))
        bad_weights = (weights[0], torch.randn(8, dim + 1, device=dev), weights[2])
        return aligned, gathered, index, weights, bad_weights

    def test_unsupported_layouts_decline_or_match_the_reference(self):
        """An entry given a layout it does not support returns None or the exact result."""
        from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import (
            decomposed_indexed_cat3_silu_linear,
        )

        aligned, gathered, index, weights, bad_weights = self._decline_inputs()
        # cat3: silu of the core/gate halves of A@W0.T + B[i]@W1.T + B[j]@W2.T.
        preact = (aligned @ weights[0].T + gathered[index] @ weights[1].T
                  + gathered[index] @ weights[2].T)
        hidden = weights[0].shape[0] // 2
        expected = (torch.nn.functional.silu(preact[:, :hidden]),
                    torch.nn.functional.silu(preact[:, hidden:]))
        with self.subTest(case="cat3 non-contiguous aligned rows"):
            result = decomposed_indexed_cat3_silu_linear(aligned, gathered, index, index, weights)
            if result is not None:
                for actual, reference in zip(result, expected, strict=True):
                    torch.testing.assert_close(actual, reference)
        with self.subTest(case="cat3 mismatched weight width"):
            # No correct result exists: the entry must decline.
            self.assertIsNone(decomposed_indexed_cat3_silu_linear(
                aligned.contiguous(), gathered, index, index, bad_weights))

    @perf_tier
    def test_declines_and_swallowed_exceptions_are_fallbacks(self):
        from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import (
            decomposed_indexed_cat3_silu_linear,
        )

        aligned, gathered, index, weights, bad_weights = self._decline_inputs()
        with OpPlanRecorder(aten=False) as rec:
            self.assertIsNone(decomposed_indexed_cat3_silu_linear(
                aligned, gathered, index, index, weights))
            self.assertIsNone(decomposed_indexed_cat3_silu_linear(
                aligned.contiguous(), gathered, index, index, bad_weights))
        self.assertEqual(rec.snapshot()["sections"][0]["fallback_counts"], {
            "indexed_cat3/decomposed_indexed_silu/aligned_guard": 1,
            "indexed_cat3/decomposed_indexed_silu/exception:RuntimeError": 1,
        })

    def _capture_and_replay(self, recording):
        """Capture a residual_add forward/backward plus a cold merged-weight cache
        lookup into one CUDA graph and replay it twice, optionally recorded."""
        from fastmd.engine._graph_capture import capture_graph
        from fastmd.models.matris.functions import GatedMLP
        from fastmd.models.matris.op.triton_residual_add import residual_add

        dev = self.device
        delta = torch.randn(64, 32, device=dev, requires_grad=True)
        residual = torch.randn(64, 32, device=dev, requires_grad=True)
        weight = torch.randn(1, 32, device=dev)
        mlp = GatedMLP(input_dim=96, hidden_dim=32, output_dim=32).to(dev).requires_grad_(False)
        probe = nn.ModuleDict({"step": _Probe(lambda p: residual_add(delta, residual, weight))})

        def step():
            out = probe["step"]()
            return torch.autograd.grad(out, (delta, residual), torch.ones_like(out))

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with ExitStack() as stack:
            stack.enter_context(patch("fastmd.models.matris.functions._INDEXED_CAT_LINEAR_FUSE", True))
            rec = stack.enter_context(OpPlanRecorder(root=probe)) if recording else None
            if rec is not None:
                rec.mark("capture")
            with capture_graph(graph):
                grads = step()
                cold = mlp._split_first_linear_params(20_000, 32, 3, dev, torch.float32)
            if rec is not None:
                rec.mark("replay")
            graph.replay()
            graph.replay()
            torch.cuda.synchronize()
        # residual_add(delta, residual, weight) = delta + weight * residual.
        torch.testing.assert_close(grads[1], torch.ones_like(residual) * weight)
        return mlp, cold, rec

    def test_cuda_graph_capture_does_not_build_a_cold_cache(self):
        mlp, cold, _ = self._capture_and_replay(recording=False)
        # A cold merged-weight cache declines during capture and stays unset ...
        self.assertIsNone(cold)
        self.assertIsNone(mlp._indexed_cat_linear_split_cache_key)
        self.assertIsNone(mlp._indexed_cat_linear_split_weights)
        # ... while the same lookup outside capture builds it.
        with patch("fastmd.models.matris.functions._INDEXED_CAT_LINEAR_FUSE", True):
            self.assertIsNotNone(mlp._split_first_linear_params(
                20_000, 32, 3, self.device, torch.float32))

    @perf_tier
    def test_cuda_graph_capture_records_and_replay_records_nothing(self):
        _, _, rec = self._capture_and_replay(recording=True)
        capture, replay = rec.snapshot()["sections"]
        self.assertEqual(_keys(capture["entries"]), [("residual_add", "triton", "fwd", "step"),
                                                     ("residual_add", "triton", "bwd", "step")])
        self.assertEqual(capture["fallback_counts"],
                         {"indexed_projection_weights/split_cache/cold_cache_under_capture": 1})
        self.assertTrue(capture["aten_counts"])
        self.assertEqual((replay["entries"], replay["fallbacks"], replay["aten_counts"]),
                         ([], [], {}))


class _Probe(nn.Module):
    """A named site whose forward runs a call that may create leaf inputs."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self.leaves = []

    def leaf(self, *shape):
        value = torch.randn(*shape, device="cuda", requires_grad=True)
        self.leaves.append(value)
        return value

    def use(self, value):
        self.leaves.append(value)
        return value

    def forward(self):
        self.leaves = []
        return self._fn(self)


@perf_tier
@unittest.skipUnless(torch.cuda.is_available() and test_assets_available(RDX_CHECKPOINT, RDX_1344),
                     "needs CUDA and the RDX checkpoint/structure in MATRIS_TEST_ASSETS")
class OpPlanRecorderMatrisTest(unittest.TestCase):
    def test_rdx_forward_and_forces_snapshot_is_nonempty_and_deterministic(self):
        from ase.io import read
        from fastmd.config import InferenceConfig, resolve_config
        from fastmd.models.matris.model import MatRIS

        model = MatRIS.load(model_path=str(test_asset(RDX_CHECKPOINT)), device="cuda")
        model = model.eval().requires_grad_(False)
        graph = model.graph_converter(None, atoms=read(str(test_asset(RDX_1344))),
                                      check_isolated_atoms=False).to("cuda")
        resolved = resolve_config(InferenceConfig(execution="eager",
                                                  optimization_profile="topology",
                                                  matmul_precision="highest"))
        def allocations(aten):
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            if aten is None:
                model([graph], task="ef")
            else:
                with OpPlanRecorder(root=model, aten=aten):
                    model([graph], task="ef")
            torch.cuda.synchronize()
            return torch.cuda.memory_stats()["allocation.all.allocated"] - before

        with resolved.scope():
            model([graph], task="ef")  # warm caches and kernels before recording
            texts = []
            for _ in range(2):
                with OpPlanRecorder(root=model) as rec:
                    out = model([graph], task="ef")
                    torch.cuda.synchronize()
                texts.append(canonical_json(rec.snapshot()))
            # Recording does not change what executes: the same allocator
            # traffic as an unrecorded evaluation (no out-of-place backward).
            unrecorded = allocations(None)
            self.assertEqual((allocations(True), allocations(False)), (unrecorded, unrecorded))
        self.assertEqual(texts[0], texts[1])
        self.assertTrue(torch.isfinite(out["f"][0]).all())
        snapshot = json.loads(texts[0])
        section = snapshot["sections"][0]
        self.assertEqual(snapshot["invalid_params"], 0)
        counts = section["entry_counts"]
        for key in ("paired_segment_attention/triton_csr/fwd",
                    "paired_segment_attention/triton_csr/bwd",
                    "indexed_cat3/decomposed_indexed_silu/fwd",
                    "cat3_gather_add_silu/triton/bwd",
                    "merged_frozen_projections/triton_row_gemm/fwd",
                    "line_envelope/triton/fwd", "threebody_fourier_basis/triton/bwd",
                    "gated_ln_residual/triton/fwd", "layer_checkpoint/off/fwd"):
            self.assertIn(key, counts)
        backward_sites = {entry["site"] for entry in section["entries"]
                          if entry["phase"] == "bwd"}
        self.assertIn("interaction_block.0.attn_block_line_graph", backward_sites)
        self.assertTrue(any(context.startswith("bwd:") for context in section["aten_counts"]))


if __name__ == "__main__":
    unittest.main()
