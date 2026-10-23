"""KernelPolicy reproduces the legacy/profile/expert-override kernel switches (step 2.2);
the OpDispatcher's live gates follow the policy (step 2.3).

Kernel selection (the policy against the module globals and the dispatcher's
live gates) is a structural perf guard in the opt-in perf tier
(``MATRIS_RUN_PERF_TIER=1``); the default suite compares the energies and
forces of the CSR segment-sum switch, of the mixed-width indexed projections
(D8), of the gated-LN bias fold (D13a) and of the fused narrow tail (C2).
"""

import collections
import unittest
from unittest.mock import patch
import warnings

import torch

from fastmd.config import InferenceConfig, current_config, env_value, resolve_config
from fastmd.engine import dispatch
from fastmd.engine.layout import ExecutionLayout
from fastmd.engine.policy import KernelPolicy, fusion_scope, fusion_scope_active
from fastmd.engine.recorder import OpPlanRecorder
from fastmd.models.matris import feature_embed as fe, functions as fn, interaction_block as ib
from fastmd.models.matris.op import compiled_lowerings as cl
from fastmd.engine.kernels import merged_frozen_projections as mfp, segment as seg, triton_wide_gemm as wg
from matris_test_tiers import PERF_TIER, perf_tier
from test_config import clean_environment

# (profile, environment, expert_overrides); non-legacy profiles ignore the environment.
CASES = [
    ("legacy", {}, {}),
    *[("legacy", {f"MATRIS_FUSED_{op}_EAGER": "1"}, {}) for op in (
        "WEIGHTED_SEGMENT", "GATHER_CAT", "SEGMENT_ATTENTION", "LINE_ENVELOPE",
        "THREEBODY_LINEAR", "THREEBODY_BASIS")],
    ("legacy", {f"MATRIS_FUSED_{op}": "0" for op in (
        "SEGMENT_ATTENTION", "LINE_ENVELOPE", "THREEBODY_LINEAR", "THREEBODY_BASIS", "RESIDUAL_ADD",
        "DIRECTED_PAIR_AGGREGATE", "SEGMENT_SOFTMAX", "GATED_LN_RESIDUAL")}, {}),
    ("legacy", {"MATRIS_FUSED_SEGMENT_ATTENTION": "0", "MATRIS_FUSED_SEGMENT_ATTENTION_EAGER": "1",
                "MATRIS_FUSED_THREEBODY_BASIS": "0", "MATRIS_FUSED_THREEBODY_BASIS_EAGER": "1"}, {}),
    ("legacy", {"MATRIS_FUSED_SEGMENT_ATTENTION_EAGER": "1", "MATRIS_FUSED_SEGMENT_ATTENTION_MIN_ROWS": "123",
                "MATRIS_FUSED_THREEBODY_LINEAR_MIN_ROWS": "5", "MATRIS_FUSED_THREEBODY_BASIS_MIN_ROWS": "7",
                "MATRIS_GATED_LN_FUSE_MIN_ROWS": "11", "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": "13",
                "MATRIS_FUSED_RESIDUAL_ADD_MIN_ROWS": "17", "MATRIS_FUSED_INDEXED_CAT_LINEAR": "1"}, {}),
    ("legacy", {"MATRIS_CSR_SEGMENT_SUM": "0", "MATRIS_GATED_LN_2D": "0"}, {}),
    ("legacy", {"MATRIS_MIXED_WIDTH_CAT": "0"}, {}),
    ("legacy", {"MATRIS_GATED_LN_BIAS_FOLD": "0"}, {}),
    ("legacy", {"MATRIS_FUSED_GATED_TAIL": "0"}, {}),
    ("topology", {}, {}),
    ("topology", {"MATRIS_FUSED_SEGMENT_ATTENTION": "0"}, {}),
    ("topology", {}, {"MATRIS_CSR_SEGMENT_SUM": False}),
    ("generic", {}, {}),
    ("generic", {"MATRIS_FUSED_GATHER_CAT_EAGER": "1"}, {}),
    ("legacy", {"MATRIS_FUSED_LINE_ENVELOPE_EAGER": "1"}, {"MATRIS_FUSED_LINE_ENVELOPE": False}),
    ("topology", {}, {"MATRIS_FUSED_SEGMENT_ATTENTION_EAGER": False, "MATRIS_FUSED_THREEBODY_LINEAR": False,
                      "MATRIS_FUSED_THREEBODY_BASIS_MIN_ROWS": 9, "MATRIS_GATED_LN_FUSE_MIN_ROWS": 0}),
    ("generic", {}, {"MATRIS_FUSED_WEIGHTED_SEGMENT_EAGER": True, "MATRIS_FUSED_SEGMENT_ATTENTION": True,
                     "MATRIS_FUSED_SEGMENT_ATTENTION_EAGER": True, "MATRIS_FUSED_RESIDUAL_ADD": True}),
]


def legacy_switches(scope):
    """The pre-2.2 formulas on today's module globals: {op: (enabled, min_rows)}."""
    return {
        "weighted_segment_sum": (ib._FUSED_WEIGHTED_SEGMENT_EAGER or scope, None),
        "gather_cat": (ib._FUSED_GATHER_CAT_EAGER or scope, ib.THRESHOLD_VALUE),
        "segment_attention": (ib._FUSED_SEGMENT_ATTENTION_ENABLED and (ib._FUSED_SEGMENT_ATTENTION_EAGER or scope),
                              ib._FUSED_SEGMENT_ATTENTION_MIN_ROWS),
        "line_envelope": (ib._FUSED_LINE_ENVELOPE_ENABLED and (ib._FUSED_LINE_ENVELOPE_EAGER or scope),
                          ib.THRESHOLD_VALUE),
        "threebody_linear": (fe._FUSED_THREEBODY_LINEAR_ENABLED and (fe._FUSED_THREEBODY_LINEAR_EAGER or scope),
                             fe._FUSED_THREEBODY_LINEAR_MIN_ROWS),
        "threebody_basis": (fe._FUSED_THREEBODY_BASIS_ENABLED and (fe._FUSED_THREEBODY_BASIS_EAGER or scope),
                            fe._FUSED_THREEBODY_BASIS_MIN_ROWS),
        "directed_pair_aggregate": (ib._FUSED_DIRECTED_PAIR_AGGREGATE, None),
        "directed_pair_expand": (ib._FUSED_DIRECTED_PAIR_EXPAND, None),
        "residual_add": (ib._FUSED_RESIDUAL_ADD_ENABLED, ib._FUSED_RESIDUAL_ADD_MIN_ROWS),
        "segment_softmax": (seg._FUSED_SEGMENT_SOFTMAX, None),
        # Step 2.5a: policy-only switch (no module global); a profile default since C10.
        "segment_sum_csr": (current_config().kernel_options["MATRIS_CSR_SEGMENT_SUM"], None),
        "gated_ln": (True, fn._LN_FUSE_THRESH),
        "gated_ln_residual": (fn._FUSED_GATED_LN_RESIDUAL, fn._LN_FUSE_THRESH),
        "indexed_projection": (fn._INDEXED_CAT_LINEAR_FUSE, fn._INDEXED_CAT_LINEAR_MIN_ROWS),
        "mixed_width_cat": (fn._MIXED_WIDTH_CAT, None),
        "gated_ln_bias_fold": (fn._GATED_LN_BIAS_FOLD, None),
        "gated_tail": (fn._FUSED_GATED_TAIL, fn._LN_FUSE_THRESH),
        # C4: module globals of the projections and the line-graph refinement.
        "row_gemm": (mfp._TRITON_ROW_GEMM, None),
        "edge_ffn": (ib._FUSED_EDGE_FFN, None),
        "wide_gemm": (wg._TRITON_WIDE_GEMM, wg._MIN_ROWS),
        "compiled_lowerings": (cl.COMPILED_LOWERINGS_ENABLED, None),
        "fused_activations": (fn._FUSED_ACTIVATIONS, None),
    }


def live_gates(policy, dispatcher):
    """The dispatcher's eligibility gates on CUDA tensors with exactly min_rows rows."""
    cuda = torch.device("cuda")

    def rows(op):
        return max(policy.ops[op].min_rows or 1, 1)

    n = rows("threebody_basis")
    return {
        "weighted_segment_sum": dispatcher.use_weighted_segment_sum(torch.ones(4, 2, device=cuda),
                                                                    torch.ones(4, 2, device=cuda)),
        "gather_cat": dispatcher.use_gather_cat(torch.ones(rows("gather_cat"), 2, device=cuda),
                                                torch.ones(1, 2, device=cuda)),
        "segment_attention": dispatcher.use_segment_attention(
            *[torch.ones(rows("segment_attention"), 2, device=cuda)] * 2),
        "line_envelope": dispatcher.use_line_envelope(
            torch.ones(1, 2, device=cuda), *[torch.zeros(rows("line_envelope"), device=cuda)] * 2),
        "threebody_linear": dispatcher.use_threebody_linear(
            torch.ones(rows("threebody_linear"), 2, device=cuda)),
        "threebody_basis": dispatcher.use_threebody_basis(
            torch.ones(n, 3, device=cuda), *[torch.zeros(n, dtype=torch.long, device=cuda)] * 2,
            torch.ones(3, device=cuda)),
    }


class KernelPolicyTest(unittest.TestCase):
    @perf_tier
    def test_policy_matches_the_module_globals(self):
        fingerprints = {}
        for profile, env, overrides in CASES:
            with self.subTest(profile=profile, env=env, overrides=overrides), clean_environment(**env):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    resolved = resolve_config(InferenceConfig(optimization_profile=profile,
                                                              expert_overrides=overrides, report=False))
                with resolved.scope():
                    policy = KernelPolicy.from_globals()
                    self.assertEqual(policy, KernelPolicy.from_resolved(resolved))
                    for scope in (False, True):
                        with fusion_scope(scope):
                            expected = {op: (bool(on), rows) for op, (on, rows) in legacy_switches(scope).items()}
                            self.assertEqual({op: (policy.enabled(op), policy.ops[op].min_rows)
                                              for op in expected}, expected, scope)
                            self.assertEqual(set(policy.ops), set(expected))
                            if torch.cuda.is_available():
                                # Step 2.3: the dispatcher's static and tensor eligibility.
                                gates = live_gates(policy, dispatch.OpDispatcher(policy, ExecutionLayout()))
                                self.assertEqual(gates, {op: policy.enabled(op) for op in gates}, scope)
                    # The CSR and mixed-width gates on FastMD's identity route.
                    local = dispatch.OpDispatcher(policy, ExecutionLayout())
                    self.assertEqual(local.segment_sum_csr_enabled, policy.enabled("segment_sum_csr"))
                    mixed = policy.enabled("mixed_width_cat") and policy.enabled("indexed_projection")
                    self.assertEqual(local.mixed_width_cat_enabled, mixed)
                    if torch.cuda.is_available():
                        rows = policy.ops["gather_cat"].min_rows
                        widths = [torch.ones(rows, 32, device="cuda"), torch.ones(1, 128, device="cuda"),
                                  torch.ones(1, 32, device="cuda")]
                        with fusion_scope(True):
                            self.assertEqual(dispatch.OpDispatcher(policy, ExecutionLayout()).use_gather_cat4(
                                *widths), mixed)
                fingerprints.setdefault(policy.fingerprint, []).append(policy)
        # Equal policies share a fingerprint, and every flipped switch above changes it.
        for policies in fingerprints.values():
            self.assertTrue(all(policy == policies[0] for policy in policies))
        self.assertEqual(len(fingerprints), len({str(p.as_dict()) for ps in fingerprints.values() for p in ps}))
        self.assertEqual(len(fingerprints), len(CASES) - 2)  # topology and generic ignore the env
        # strict_errors is reported next to the fingerprint of the selection table.
        strict = KernelPolicy.from_globals(strict_errors=True)
        self.assertTrue(strict.strict_errors)
        self.assertNotEqual(strict, KernelPolicy.from_globals())
        self.assertEqual(strict.fingerprint, KernelPolicy.from_globals().fingerprint)

    def test_fusion_scope_and_compat_aliases_restore(self):
        self.assertFalse(fusion_scope_active())
        with fusion_scope():
            self.assertTrue(fusion_scope_active())
            with fusion_scope(False):
                self.assertFalse(fusion_scope_active())
            self.assertTrue(fusion_scope_active())
            with self.assertRaises(RuntimeError), fusion_scope(False):
                raise RuntimeError
            self.assertTrue(fusion_scope_active())
        self.assertFalse(fusion_scope_active())

        # The deprecated set/restore aliases toggle the same scope.
        previous = ib.set_fused_graph_optimizations(True)
        previous_feature = fe.set_fused_feature_graph_optimizations(True)
        self.assertTrue(fusion_scope_active())
        fe.restore_fused_feature_graph_optimizations(previous_feature)
        ib.restore_fused_graph_optimizations(previous)
        self.assertFalse(fusion_scope_active())


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class KernelSwitchTest(unittest.TestCase):
    """Opt-in kernel switches give the E/F of the switch off."""

    def test_csr_segment_sum_switch_matches_default(self):
        """MATRIS_CSR_SEGMENT_SUM (step 2.5a D6): the segment sums run the CSR kernel,
        forward and backward, with the E/F of the switch off."""
        from fastmd.models.matris.model import MatRIS
        from op_selection_snapshots import synthetic_graph

        torch.manual_seed(20260928)
        model = MatRIS(num_layers=2, node_feat_dim=32, edge_feat_dim=32, three_body_feat_dim=32,
                       mlp_hidden_dims=(32,), norm_type="layer", reference_energy=None,
                       enable_compile=False, enable_checkpoint=False).cuda().eval()
        model.requires_grad_(False)
        graph = synthetic_graph(3000)
        for profile, extra in (("legacy", {}), ("topology", {}),
                               ("topology", {"MATRIS_FUSED_SEGMENT_ATTENTION": False})):
            runs = {}
            for csr in (False, True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    resolved = resolve_config(InferenceConfig(
                        optimization_profile=profile, report=False,
                        expert_overrides={**extra, "MATRIS_CSR_SEGMENT_SUM": csr}))
                with patch.object(ib, "THRESHOLD_VALUE", 0), resolved.scope(), \
                        OpPlanRecorder() as recording:
                    out = model([graph], task="ef")
                    torch.cuda.synchronize()
                counts = recording.snapshot()["sections"][0]["entry_counts"]
                runs[csr] = ({key for key in counts if key.startswith(
                    ("segment_sum/triton_csr/", "weighted_segment_sum/triton_csr/"))}, out["e"], out["f"])
            with self.subTest(profile=profile, **extra):
                if PERF_TIER:  # the switch off selects no CSR kernel: a kernel-selection guard
                    self.assertEqual(runs[False][0], set())
                # Non-vacuous: the switch runs the CSR kernel forward and backward.
                self.assertTrue({"weighted_segment_sum/triton_csr/fwd",
                                 "weighted_segment_sum/triton_csr/bwd"} <= runs[True][0], runs[True][0])
                torch.testing.assert_close(runs[True][1], runs[False][1], rtol=0, atol=1e-5)
                torch.testing.assert_close(runs[True][2], runs[False][2], rtol=1e-4, atol=5e-4)

    def test_mixed_width_cat_switch_matches_off(self):
        """MATRIS_MIXED_WIDTH_CAT (step 2.5b D8): the atom rows [32|64|64] and the line
        refinement [32|64|32|32] run the decomposed indexed projections instead of
        gather + cat + GEMM, with the E/F of the switch off (a different summation order)."""
        from fastmd.models.matris.model import MatRIS
        from op_selection_snapshots import synthetic_graph

        torch.manual_seed(20260929)
        model = MatRIS(num_layers=2, node_feat_dim=64, edge_feat_dim=32, three_body_feat_dim=32,
                       mlp_hidden_dims=(32,), norm_type="layer", reference_energy=None,
                       enable_compile=False, enable_checkpoint=False).cuda().eval()
        model.requires_grad_(False)
        graph = synthetic_graph(3000)
        forced = {"MATRIS_GATED_LN_FUSE_MIN_ROWS": 0, "MATRIS_FUSED_INDEXED_CAT_LINEAR": True,
                  "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0}
        for profile in ("legacy", "topology"):
            runs = {}
            for mixed in (False, True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    resolved = resolve_config(InferenceConfig(
                        optimization_profile=profile, report=False,
                        expert_overrides={**forced, "MATRIS_MIXED_WIDTH_CAT": mixed}))
                with patch.object(ib, "THRESHOLD_VALUE", 0), resolved.scope(), fusion_scope(True), \
                        OpPlanRecorder() as recording:
                    out = model([graph], task="ef")
                    torch.cuda.synchronize()
                counts = recording.snapshot()["sections"][0]["entry_counts"]
                runs[mixed] = ({key: n for key, n in counts.items() if key.startswith(
                    ("indexed_cat3/", "indexed_cat4/", "gather_cat3/", "gather_cat4/"))}, out["e"], out["f"])
            with self.subTest(profile=profile):
                # Non-vacuous: off keeps the gather-cat fallback at the three mixed-width sites of
                # both blocks; on runs the decomposed projections there (plus the equal-width line
                # attention, on in both).
                off, on = runs[False][0], runs[True][0]
                self.assertEqual(off.get("gather_cat3/torch/fwd"), 4, off)
                self.assertEqual(off.get("gather_cat4/torch/fwd"), 2, off)
                self.assertNotIn("gather_cat3/torch/fwd", on)
                self.assertNotIn("gather_cat4/torch/fwd", on)
                self.assertEqual(on.get("indexed_cat4/decomposed_indexed_silu/fwd"), 2, on)
                self.assertEqual(on.get("indexed_cat3/decomposed_indexed_silu/fwd"),
                                 off.get("indexed_cat3/decomposed_indexed_silu/fwd", 0) + 4, on)
                torch.testing.assert_close(runs[True][1], runs[False][1], rtol=0, atol=1e-5)
                torch.testing.assert_close(runs[True][2], runs[False][2], rtol=1e-4, atol=5e-4)

    def test_gated_tail_switch_matches_off(self):
        """MATRIS_FUSED_GATED_TAIL (C2, step 2.5b D13b): the GatedMLPs after the indexed projections
        run their second linears and gated LayerNorm as one kernel per direction, with the E/F of
        the switch off; the others (wider second linears here) decline."""
        from fastmd.models.matris.model import MatRIS
        from op_selection_snapshots import synthetic_graph

        torch.manual_seed(20260929)
        model = MatRIS(num_layers=2, node_feat_dim=64, edge_feat_dim=32, three_body_feat_dim=32,
                       mlp_hidden_dims=(32,), norm_type="layer", reference_energy=None,
                       enable_compile=False, enable_checkpoint=False).cuda().eval()
        model.requires_grad_(False)
        graph = synthetic_graph(3000)
        forced = {"MATRIS_GATED_LN_FUSE_MIN_ROWS": 0, "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0}
        runs = {}
        for fused in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolved = resolve_config(InferenceConfig(
                    optimization_profile="topology", report=False, matmul_precision="highest",
                    expert_overrides={**forced, "MATRIS_FUSED_GATED_TAIL": fused}))
            with patch.object(ib, "THRESHOLD_VALUE", 0), resolved.scope(), fusion_scope(True), \
                    OpPlanRecorder() as recording:
                out = model([graph], task="ef")
                torch.cuda.synchronize()
            section = recording.snapshot()["sections"][0]
            counts = {key: n for key, n in section["entry_counts"].items() if key.startswith("gated_")}
            runs[fused] = (counts, section["fallback_counts"], out["e"], out["f"])
        off, on = runs[False][0], runs[True][0]
        # Non-vacuous: on replaces gated LayerNorms forward and backward by the tail; off never runs it.
        self.assertGreater(on.get("gated_tail/triton/fwd", 0), 0, on)
        self.assertEqual(on.get("gated_tail/triton/bwd"), on["gated_tail/triton/fwd"], on)
        self.assertEqual(on["gated_tail/triton/fwd"] + on.get("gated_ln/triton/fwd", 0),
                         off["gated_ln/triton/fwd"], (on, off))
        self.assertNotIn("gated_tail/triton/fwd", off)
        self.assertGreater(runs[True][1].get("gated_tail/triton/unsupported_shape", 0), 0, runs[True][1])
        self.assertFalse([key for key in runs[False][1] if key.startswith("gated_tail/")], runs[False][1])
        torch.testing.assert_close(runs[True][2], runs[False][2], rtol=0, atol=1e-5)
        torch.testing.assert_close(runs[True][3], runs[False][3], rtol=1e-4, atol=5e-4)

    def test_row_gemm_and_edge_ffn_switches_match_off(self):
        """MATRIS_TRITON_ROW_GEMM and MATRIS_FUSED_EDGE_FFN (C4): the K = 32 projections (merged attention,
        projection with alias and its gradient, the cat3/cat4 parts) run the row GEMM and the line-graph
        edge FFN one kernel per direction, with the E/F of the switches off; at each explicit dot precision
        the sites run the resolved precision or their cuBLAS path."""
        from fastmd.models.matris.model import MatRIS
        from op_selection_snapshots import synthetic_graph

        torch.manual_seed(20260930)
        model = MatRIS(num_layers=2, node_feat_dim=64, edge_feat_dim=32, three_body_feat_dim=32,
                       mlp_hidden_dims=(32,), norm_type="layer", reference_energy=None,
                       enable_compile=False, enable_checkpoint=False).cuda().eval()
        model.requires_grad_(False)
        graph = synthetic_graph(3000)
        forced = {"MATRIS_GATED_LN_FUSE_MIN_ROWS": 0, "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0}
        runs, precisions = {}, {}
        for fused, dot in ((False, None), (True, None), (True, "ieee"), (True, "tf32")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolved = resolve_config(InferenceConfig(
                    optimization_profile="topology", report=False, matmul_precision="highest",
                    triton_dot_precision=dot,
                    expert_overrides={**forced, "MATRIS_TRITON_ROW_GEMM": fused, "MATRIS_FUSED_EDGE_FFN": fused}))
            with patch.object(ib, "THRESHOLD_VALUE", 0), resolved.scope(), fusion_scope(True), \
                    OpPlanRecorder() as recording:
                out = model([graph], task="ef")
                torch.cuda.synchronize()
            section = recording.snapshot()["sections"][0]
            key = fused if dot is None else dot
            runs[key] = (section["entry_counts"], section["fallback_counts"], out["e"], out["f"])
            precisions[key] = {}
            for e in section["entries"]:
                if e["phase"] == "fwd" and "PRECISION" in e["params"]:
                    precisions[key].setdefault(f"{e['op']}/{e['impl']}", set()).add(e["params"]["PRECISION"])
        # The dot precision option (auto here: 3xTF32 under IEEE cuBLAS). ieee: the projections and the edge
        # FFN take cuBLAS (faster than FP32 FMA dots there), the gated tail FMA dots; tf32: TF32 dots.
        self.assertIn("gated_tail/triton", precisions[True], precisions)
        self.assertEqual(precisions[True], {key: {"tf32x3"} for key in precisions[True]}, precisions)
        self.assertEqual(precisions["tf32"], {key: {"tf32"} for key in precisions[True]}, precisions)
        self.assertEqual(precisions["ieee"], {"gated_tail/triton": {"ieee"}}, precisions)
        self.assertEqual(runs["ieee"][0].get("projection_with_alias/triton_row_addmm/bwd"),
                         runs[True][0]["projection_with_alias/triton_row_addmm/bwd"])
        self.assertLessEqual({"row_gemm/triton/cublas_precision", "edge_ffn/triton/cublas_precision"},
                             set(runs["ieee"][1]), runs["ieee"][1])
        for key, tol in (("ieee", 1e-5), ("tf32", 3e-3)):
            torch.testing.assert_close(runs[key][2], runs[False][2], rtol=0, atol=tol * 10)
            torch.testing.assert_close(runs[key][3], runs[False][3], rtol=tol * 10, atol=tol * 50)
        off, on = runs[False][0], runs[True][0]
        # Non-vacuous: on runs the row GEMMs and the fused edge FFN (its residual stays the residual kernel);
        # off never does.
        self.assertGreater(on.get("merged_frozen_projections/triton_row_gemm/fwd", 0), 0, on)
        self.assertGreater(on.get("projection_with_alias/triton_row_gemm/fwd", 0), 0, on)
        self.assertGreater(on.get("projection_with_alias/triton_row_addmm/bwd", 0), 0, on)
        self.assertGreater(on.get("edge_ffn/triton/fwd", 0), 0, on)
        self.assertEqual(on.get("edge_ffn/triton/bwd"), on["edge_ffn/triton/fwd"], on)
        self.assertFalse([key for key in off if "triton_row" in key or key.startswith("edge_ffn/")], off)
        torch.testing.assert_close(runs[True][2], runs[False][2], rtol=0, atol=1e-5)
        torch.testing.assert_close(runs[True][3], runs[False][3], rtol=1e-4, atol=5e-4)

    def test_wide_gemm_switch_matches_off(self):
        """MATRIS_TRITON_WIDE_GEMM (TX) on an equal-width 64 model: the wide frozen projections run the 3xTF32
        GEMM forward and for the input gradient (also the recompute recipes at on/deep), with the E/F of the
        switch off; triton_dot_precision "ieee" keeps cuBLAS."""
        from fastmd.models.matris.model import MatRIS
        from op_selection_snapshots import synthetic_graph

        torch.manual_seed(20261002)
        model = MatRIS(num_layers=2, node_feat_dim=64, edge_feat_dim=64, three_body_feat_dim=64,
                       mlp_hidden_dims=(64,), norm_type="layer", reference_energy=None,
                       enable_compile=False, enable_checkpoint=False).cuda().eval()
        model.requires_grad_(False)
        graph = synthetic_graph(3000)
        forced = {"MATRIS_GATED_LN_FUSE_MIN_ROWS": 0, "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0,
                  "MATRIS_TRITON_WIDE_GEMM_MIN_ROWS": 0}
        runs = {}
        for name, on, dot, recompute in (("off", False, None, "off"), ("on", True, None, "off"),
                                         ("ieee", True, "ieee", "off"), ("deep", True, None, "deep")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolved = resolve_config(InferenceConfig(
                    optimization_profile="topology", report=False, matmul_precision="highest",
                    triton_dot_precision=dot,
                    expert_overrides={**forced, "MATRIS_TRITON_WIDE_GEMM": on}))
            with patch.object(ib, "THRESHOLD_VALUE", 0), resolved.scope(), fusion_scope(True), \
                    OpPlanRecorder() as recording:
                from fastmd.engine import saved_recompute
                with saved_recompute.forward_scope(recompute):
                    out = model([graph], task="ef")
                torch.cuda.synchronize()
            section = recording.snapshot()["sections"][0]
            runs[name] = (section["entry_counts"], out["e"], out["f"],
                          {e["params"]["PRECISION"] for e in section["entries"] if e["op"] == "wide_gemm"})
        on = runs["on"][0]
        self.assertEqual(runs["on"][3], {"tf32x3"})
        self.assertGreater(on.get("wide_gemm/triton/fwd", 0), 0, on)
        self.assertGreater(on.get("wide_gemm/triton/bwd", 0), 0, on)
        self.assertGreater(runs["deep"][0].get("wide_gemm/triton/bwd", 0), 0, runs["deep"][0])
        for name in ("off", "ieee"):
            self.assertFalse([key for key in runs[name][0] if key.startswith("wide_gemm/")], runs[name][0])
        for name in ("on", "ieee", "deep"):
            torch.testing.assert_close(runs[name][1], runs["off"][1], rtol=0, atol=1e-5)
            torch.testing.assert_close(runs[name][2], runs["off"][2], rtol=1e-4, atol=5e-4)

    def test_wide_gemm_routing_and_rank_uniform_thresholds(self):
        """The wide GEMM dispatch replaces a frozen nn.Linear's call only without module or global hooks (the
        op-selection recorder's own excepted), a container's only without its hooks; inside a P>1 forward
        (``rank_uniform``) the configured thresholds move above the ranks' atom-row range when inside it, and a
        Function's backward decides with its forward's thresholds."""
        layer, sequence = torch.nn.Linear(64, 64), torch.nn.Sequential(torch.nn.Linear(64, 64))
        with patch.object(wg, "_TRITON_WIDE_GEMM", False):
            self.assertFalse(fn._routed(layer))
        with patch.object(wg, "_TRITON_WIDE_GEMM", True):
            self.assertTrue(fn._routed(layer) and fn._unhooked(sequence))
            with OpPlanRecorder():
                self.assertTrue(fn._routed(layer))
            for owner, register in ((layer, layer.register_forward_hook), (sequence, sequence.register_forward_hook),
                                    (None, torch.nn.modules.module.register_module_forward_pre_hook)):
                handle = register(lambda *args: None)
                try:
                    self.assertFalse(fn._routed(layer) if owner is not sequence else fn._unhooked(sequence))
                finally:
                    handle.remove()
            self.assertFalse(fn._routed(torch.nn.Bilinear(64, 64, 64)))
            with patch.object(layer, "forward", lambda value: value):  # an instance override (wrappers)
                self.assertFalse(fn._routed(layer))
            with patch.object(wg, "_MIN_ROWS", 0):  # the P>1 scope's
                self.assertEqual(wg.dispatch_on(), (0, 0))
                nominal = int(env_value("MATRIS_TRITON_WIDE_GEMM_MIN_ROWS"))
                for low, high, expected in ((100, 200, (nominal, 4 * nominal)),
                                            (nominal - 1, nominal + 5, (nominal + 6, 4 * nominal)),
                                            (nominal, nominal + 5, (nominal, 4 * nominal)),
                                            (nominal + 1, 5 * nominal, (nominal, 5 * nominal + 1)),
                                            (nominal - 1, 5 * nominal, (5 * nominal + 1, 5 * nominal + 1))):
                    with wg.rank_uniform((low, high), nominal):
                        self.assertEqual(wg.dispatch_on(), expected)
                    self.assertEqual(wg._thresholds(), (0, 0))

    def test_gated_ln_bias_fold_switch_matches_off(self):
        """MATRIS_GATED_LN_BIAS_FOLD (step 2.5b D13a): every fused gated LayerNorm (plain,
        residual, after the indexed projections) adds the second linears' biases, with the
        E/F of the switch off (bitwise per GatedMLP; here within rounding: the model's segment
        sums are atomic)."""
        from fastmd.models.matris.model import MatRIS
        from op_selection_snapshots import synthetic_graph

        torch.manual_seed(20260929)
        model = MatRIS(num_layers=2, node_feat_dim=64, edge_feat_dim=32, three_body_feat_dim=32,
                       mlp_hidden_dims=(32,), norm_type="layer", reference_energy=None,
                       enable_compile=False, enable_checkpoint=False).cuda().eval()
        model.requires_grad_(False)
        graph = synthetic_graph(3000)
        forced = {"MATRIS_GATED_LN_FUSE_MIN_ROWS": 0, "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0}
        runs = {}
        for fold in (False, True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resolved = resolve_config(InferenceConfig(
                    optimization_profile="topology", report=False,
                    expert_overrides={**forced, "MATRIS_GATED_LN_BIAS_FOLD": fold}))
            with patch.object(ib, "THRESHOLD_VALUE", 0), resolved.scope(), fusion_scope(True), \
                    OpPlanRecorder() as recording:
                out = model([graph], task="ef")
                torch.cuda.synchronize()
            fused = collections.Counter()
            for entry in recording.snapshot()["sections"][0]["entries"]:
                if entry["op"] in ("gated_ln", "gated_ln_residual") and entry["impl"] == "triton":
                    fused[entry["op"], entry["phase"], bool(entry["params"].get("linear_bias"))] += 1
            runs[fold] = (fused, out["e"], out["f"])
        off, on = runs[False][0], runs[True][0]
        # Non-vacuous: both forms ran fused, forward and backward; on folds every one of them.
        for op in ("gated_ln", "gated_ln_residual"):
            self.assertGreater(off[op, "fwd", False], 0, off)
            self.assertGreater(off[op, "bwd", False], 0, off)
            self.assertEqual(on[op, "fwd", True], off[op, "fwd", False], on)
        self.assertEqual(sum(n for key, n in off.items() if key[2]), 0, off)
        self.assertEqual(sum(n for key, n in on.items() if key[1] == "fwd" and not key[2]), 0, on)
        torch.testing.assert_close(runs[True][1], runs[False][1], rtol=0, atol=1e-5)
        torch.testing.assert_close(runs[True][2], runs[False][2], rtol=1e-4, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
