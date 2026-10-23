"""Saved-activation recompute (``InferenceConfig.saved_tensor_recompute``) against off, and its auto ladder.

The forward is untouched and forces agree to the FP32 rounding of the rebuilt
tensors (the 3xTF32 line tails, deep's first layers by parts). On this toy the
forward itself is not run-to-run bitwise (atomic segment sums at forced row
thresholds), so both are compared within tight tolerances; the bitwise-energy
gate on the production model is checked by an A/B benchmark, not here. A small
mixed-width model (atom 64, bond and triplet 32, as the production model's
128/32) with every row threshold at 0 and a tiny chunk runs each recipe of
"on" and "deep" and the chunked first layer, for the topology and legacy
profiles, eager and under CUDA graph capture; topology with and without the
mixed-width indexed projections (D8), whose Cat3/Cat4 recipes replace the
gather-cat ones, and the gated LayerNorm's bias fold (D13a, bias-free gate
inputs; off with D8 off, as before both); with D8 the Cat3/Cat4 tails run fused
(C2) at every level, rebuilt by the fused kernel's own recipe. Plus the rejections a
comparison cannot show: checkpoint or compile with recompute, and a
create_graph backward through rebuilt tensors. The auto rules (the ladder and
the merged-projection single-GEMM backward's C3 auto) run on mocked predictions.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import numpy as np
import torch
from ase import Atoms

from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.config import CapacityConfig, InferenceConfig
from fastmd.engine import saved_recompute
from fastmd.models.matris.model import MatRIS

# Every fused kernel at any row count (the production sizes select them).
FORCED = {
    "MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0,
    "MATRIS_GATED_LN_FUSE_MIN_ROWS": 0,
    "MATRIS_FUSED_THREEBODY_LINEAR_MIN_ROWS": 0,
    "MATRIS_FUSED_THREEBODY_BASIS_MIN_ROWS": 0,
}


def make_model() -> MatRIS:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1212)
        model = MatRIS(
            num_layers=2, node_feat_dim=64, edge_feat_dim=32, three_body_feat_dim=32,
            mlp_hidden_dims=(32,), num_radial=4, num_angular=4, max_n=2, max_l=2,
            pairwise_cutoff=3.0, three_body_cutoff=2.2, norm_type="layer",
            reference_energy=None, enable_compile=False, enable_checkpoint=False,
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def make_atoms() -> Atoms:
    rng = np.random.default_rng(3)
    grid = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3) * 1.45
    positions = grid + rng.normal(scale=0.08, size=grid.shape)
    return Atoms(numbers=rng.choice([1, 6, 7, 8], size=len(grid)), positions=positions,
                 cell=[5.8] * 3, pbc=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class SavedRecomputeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        cls.atoms = make_atoms()

    @classmethod
    def tearDownClass(cls):
        torch.set_float32_matmul_precision(cls.previous_precision)

    def evaluate(self, execution: str, recompute: str, profile: str = "topology", mixed: bool = True):
        config = InferenceConfig(
            execution=execution, optimization_profile=profile, saved_tensor_recompute=recompute, report=False,
            capacity=CapacityConfig(u_step=64, t_step=512, min_pad_u=64, min_pad_t=512, warmup=1),
            expert_overrides={**FORCED, "MATRIS_MIXED_WIDTH_CAT": mixed, "MATRIS_GATED_LN_BIAS_FOLD": mixed},
        )
        with GPUResidentMolecularDynamics(self.atoms, model=make_model().cuda(), task="ef", device="cuda",
                                          ensemble="nve", timestep=0.0, config=config) as md:
            frame = md.state.positions.clone()
            results = [tuple(t.detach().double().cpu() for t in md.evaluate(frame)) for _ in range(2)]
        return results

    def test_levels_match_off(self):
        from fastmd.engine import policy
        from fastmd.models.matris import functions, interaction_block
        from fastmd.engine.kernels import decomposed_indexed_cat_silu_linear as decomposed
        # The legacy profile keeps the indexed projections off (D8 needs them).
        for profile, mixed in (("topology", True), ("topology", False), ("legacy", True)):
            for execution in ("eager", "model_graph"):
                with contextlib.ExitStack() as stack:
                    if profile == "topology" and mixed:
                        # The toy's line rows are below the cat4 gate's 60000: lower it so the
                        # line refinement [32|64|32|32] takes the D8 cat4 path as well.
                        stack.enter_context(mock.patch.object(interaction_block, "THRESHOLD_VALUE", 0))
                        stack.enter_context(mock.patch.object(policy, "_THRESHOLD_VALUE", 0))
                    stack.enter_context(mock.patch.object(saved_recompute, "CHUNK_ROWS", 256))
                    by_parts = stack.enter_context(mock.patch.object(
                        functions, "_first_linear_by_parts", wraps=functions._first_linear_by_parts))
                    cat4 = stack.enter_context(mock.patch.object(
                        decomposed, "decomposed_indexed_cat4_silu_linear",
                        wraps=decomposed.decomposed_indexed_cat4_silu_linear))
                    before = saved_recompute.stats()
                    (force_off, energy_off), _ = self.evaluate(execution, "off", profile, mixed)
                    self.assertEqual(saved_recompute.stats(), before)
                    # The line refinement [32|64|32|32] is a D8 site: decomposed exactly when on.
                    self.assertEqual(cat4.call_count > 0, profile == "topology" and mixed)
                    packed = {}
                    for level in ("on", "deep"):
                        with self.subTest(profile=profile, mixed=mixed, execution=execution, level=level):
                            before = saved_recompute.stats()
                            for force, energy in self.evaluate(execution, level, profile, mixed):
                                torch.testing.assert_close(energy, energy_off, rtol=1e-6, atol=1e-5)
                                torch.testing.assert_close(force, force_off, rtol=1e-5, atol=2e-6)
                            after = saved_recompute.stats()
                            # Recipes replaced saved tensors and were rebuilt in the backward
                            # (handles of dead last-block branches are never unpacked).
                            self.assertGreater(after["rebuilt"], before["rebuilt"])
                            packed[level] = after["packed"] - before["packed"]
                    # deep replaces the saved tensors of "on" and more; its gather-cat first layers
                    # (all three mixed-width sites without D8) were rebuilt by parts.
                    self.assertGreater(packed["on"], 0)
                    self.assertGreater(packed["deep"], packed["on"])
                    self.assertEqual(by_parts.call_count > 0, not (profile == "topology" and mixed))

    def test_deep_cat3_preact(self):
        # deep rebuilds the line attention's gather-add preact from a rebuilt p0 projection and the
        # node projections (the toy model does not take the merged-projection path that feeds it).
        from fastmd.engine.kernels import decomposed_indexed_cat_silu_linear as cat3

        generator = torch.Generator(device="cuda").manual_seed(11)
        rows, nodes, width, hidden = 20_000, 3_000, 32, 32
        weights = tuple(0.1 * torch.randn(2 * hidden, width, device="cuda", generator=generator) for _ in range(3))
        bias = 0.1 * torch.randn(2 * hidden, device="cuda", generator=generator)
        aligned = torch.randn(rows, width, device="cuda", generator=generator)
        gathered = torch.randn(nodes, width, device="cuda", generator=generator)
        index_a, index_b = (torch.randint(0, nodes, (rows,), device="cuda", generator=generator) for _ in range(2))

        def run(mode):
            x, g = aligned.clone().requires_grad_(), gathered.clone().requires_grad_()
            with saved_recompute.forward_scope(mode) as scope:
                p0 = torch.nn.functional.linear(x, weights[0])
                if scope is not None:
                    scope.register((p0,), torch.nn.functional.linear, scope.ref(x), weights[0], same_layout=False)
                core, gate = cat3.decomposed_indexed_cat3_silu_linear(
                    x, g, index_a, index_b, weights, bias, precomputed_p0=p0, strict_errors=True)
                loss = (core * gate.sin()).sum()
            del p0
            return loss.detach(), torch.autograd.grad(loss, (x, g))

        loss_off, grads_off = run("off")
        with mock.patch.object(cat3, "_cat3_preact", wraps=cat3._cat3_preact) as preact:
            loss_deep, grads_deep = run("deep")
        self.assertEqual(preact.call_count, 1)
        self.assertTrue(torch.equal(loss_deep, loss_off))
        for deep, off in zip(grads_deep, grads_off):
            torch.testing.assert_close(deep, off, rtol=1e-5, atol=1e-6)

    def test_first_linear_by_parts(self):
        # deep rebuilds a GatedMLP first layer on a gather-concatenation from its parts.
        from fastmd.models.matris.functions import _first_linear_by_parts

        generator = torch.Generator(device="cuda").manual_seed(7)
        linear = torch.nn.Linear(32 + 2 * 128, 128).cuda().requires_grad_(False)
        aligned = torch.randn(50_000, 32, device="cuda", generator=generator)
        nodes = torch.randn(1_000, 128, device="cuda", generator=generator)
        target, source = (torch.randint(0, 1_000, (50_000,), device="cuda", generator=generator) for _ in range(2))
        full = linear(torch.cat([aligned, nodes[target], nodes[source]], 1))
        torch.testing.assert_close(_first_linear_by_parts(aligned, nodes, target, source, linear.weight, linear.bias),
                                   full, rtol=1e-5, atol=1e-5)

    def test_chunked_first_layer_is_the_full_gemm(self):
        # The refinement's first layer at production row counts (widths 32 | 128 | 32 | 32 -> 32):
        # every balanced chunk runs the full GEMM's kernel, so the forward stays bitwise.
        from fastmd.models.matris.op.chunked_gather_linear import chunked_gather_linear

        generator = torch.Generator(device="cuda").manual_seed(5)
        linear = torch.nn.Linear(224, 32).cuda().requires_grad_(False)
        atoms = torch.randn(10752, 128, device="cuda", generator=generator)
        bonds = torch.randn(487112, 32, device="cuda", generator=generator)
        for rows in (1_048_577, 3_805_464, 7_610_928):
            with self.subTest(rows=rows):
                aligned = torch.randn(rows, 32, device="cuda", generator=generator)
                index = [torch.randint(0, n, (rows,), device="cuda", generator=generator)
                         for n in (10752, 487112, 487112)]
                gathered = ((atoms, index[0]), (bonds, index[1]), (bonds, index[2]))
                full = torch.nn.functional.linear(
                    torch.cat([aligned, *(value[i] for value, i in gathered)], 1), linear.weight, linear.bias)
                chunk = saved_recompute.balanced_chunk(rows, saved_recompute.CHUNK_ROWS)
                (chunked,) = chunked_gather_linear(aligned, gathered, ((linear.weight, linear.bias),), chunk)
                self.assertTrue(torch.equal(chunked, full))
                del full, chunked, aligned, index, gathered

    def test_rejected_configurations(self):
        from fastmd.config import resolve_config
        for level in ("on", "deep"):
            with self.assertRaisesRegex(ValueError, "excludes checkpoint"):
                InferenceConfig(saved_tensor_recompute=level, checkpoint="on")
            with self.assertRaisesRegex(ValueError, "warmup >= 1"):
                resolve_config(InferenceConfig(saved_tensor_recompute=level, capacity=CapacityConfig(warmup=0)))
        with self.assertRaisesRegex(ValueError, "must be None or one of"):
            InferenceConfig(saved_tensor_recompute="narrow")
        # auto is the default; it cannot turn on under checkpointing or without a warmup.
        self.assertEqual(resolve_config(InferenceConfig()).saved_tensor_recompute, "auto")
        self.assertEqual(resolve_config(InferenceConfig(checkpoint="on")).saved_tensor_recompute, "off")
        self.assertEqual(resolve_config(InferenceConfig(capacity=CapacityConfig(warmup=0))).saved_tensor_recompute,
                         "off")
        with mock.patch.dict("os.environ", {"MATRIS_SAVED_TENSOR_RECOMPUTE": "off"}):
            self.assertEqual(resolve_config(InferenceConfig()).saved_tensor_recompute, "off")
            self.assertEqual(resolve_config(InferenceConfig(saved_tensor_recompute="on")).saved_tensor_recompute, "on")
        with mock.patch.dict("os.environ", {"MATRIS_SAVED_TENSOR_RECOMPUTE": "deep"}):
            self.assertEqual(resolve_config(InferenceConfig()).saved_tensor_recompute, "deep")
            with self.assertRaisesRegex(ValueError, "excludes checkpoint"):
                resolve_config(InferenceConfig(checkpoint="on"))
        # recompute_policy: fast by default, MATRIS_RECOMPUTE_POLICY otherwise; the default leaves fingerprints.
        with self.assertRaisesRegex(ValueError, "recompute_policy must be"):
            InferenceConfig(recompute_policy="fastest")
        self.assertEqual(resolve_config(InferenceConfig()).recompute_policy, "fast")
        self.assertNotIn("recompute_policy", resolve_config(InferenceConfig()).as_dict())
        with mock.patch.dict("os.environ", {"MATRIS_RECOMPUTE_POLICY": "safe"}):
            self.assertEqual(resolve_config(InferenceConfig()).recompute_policy, "safe")
            self.assertEqual(resolve_config(InferenceConfig(recompute_policy="fast")).recompute_policy, "fast")
        with mock.patch.dict("os.environ", {"MATRIS_RECOMPUTE_POLICY": "bogus"}), \
                self.assertRaisesRegex(ValueError, "MATRIS_RECOMPUTE_POLICY"):
            resolve_config(InferenceConfig())
        model = make_model()
        for attribute in ("enable_compile", "enable_checkpoint"):
            for mode in ("narrow", "deep"):
                with self.subTest(attribute=attribute, mode=mode):
                    block = model.interaction_block[0]
                    previous = getattr(block, attribute)
                    setattr(block, attribute, True)
                    try:
                        with self.assertRaisesRegex(ValueError, "recompute requires"):
                            saved_recompute.check_model(model, mode)
                    finally:
                        setattr(block, attribute, previous)

    def test_fast_policy_keeps_off_and_falls_back(self):
        # P=1 eager, recompute_policy "fast": "off" where "safe" climbs (on expandable segments), then
        # "safe" once an evaluation leaves less than the admission margin free. Only on predictions from a
        # measured table: the toy model's own calibration is "derived" (decides as "safe", no switch).
        from fastmd.config import resolve_config
        from fastmd.engine import memory_admission
        model = make_model().cuda()
        config = resolve_config(InferenceConfig(execution="eager", optimization_profile="topology", report=False))
        fast = saved_recompute.policy_threshold("fast", "eager")
        state = saved_recompute.AutoRecompute("test", fast, memory=memory_admission.RDX_MEMORY_MODEL)
        derived = saved_recompute.AutoRecompute("derived", fast)
        total = torch.cuda.mem_get_info()[1]
        free, reserved = 100 * 2**30, torch.cuda.memory_reserved()
        budget = free + reserved - torch.cuda.memory_allocated() - memory_admission.margin_bytes(total)
        predict = lambda *a, recompute="off", **k: int({"off": 0.92, "narrow": 0.4, "deep": 0.3}[recompute] * budget)
        switched = []
        with mock.patch.object(memory_admission, "predict", predict), \
                mock.patch.object(saved_recompute, "ensure_expandable_segments", lambda: switched.append(1) or True):
            with mock.patch.object(torch.cuda, "mem_get_info", lambda *a: (free, total)):
                self.assertEqual(derived.decide_local(model, config, 1000), "narrow")
                self.assertEqual((derived.memory.calibration, derived.fast_only, switched), ("derived", False, []))
                self.assertEqual(state.decide_local(model, config, 1000), "off")
                self.assertTrue(state.fast_only)
                self.assertEqual(switched, [1])
                self.assertEqual(state.decide_local(model, config, 1000), "off")  # enough left free
            # The evaluations cached the device nearly full (same budget, less than the margin free).
            with mock.patch.object(torch.cuda, "mem_get_info", lambda *a: (2**20, total)), \
                    mock.patch.object(torch.cuda, "memory_reserved", lambda *a: reserved + free - 2**20):
                self.assertEqual(state.decide_local(model, config, 1000), "narrow")  # as "safe" decides
        self.assertTrue(state.fast_fallback)
        self.assertEqual(state.threshold, saved_recompute.AUTO_THRESHOLD)

    def test_create_graph_backward_fails(self):
        x = torch.randn(64, 32, device="cuda", requires_grad=True)
        with saved_recompute.forward_scope("narrow") as scope:
            hidden = x * 2
            scope.register((hidden,), lambda value: value * 2, scope.ref(x))
            loss = torch.nn.functional.silu(hidden).sum()  # saves hidden
        torch.testing.assert_close(torch.autograd.grad(loss, x, retain_graph=True)[0],
                                   2 * torch.ops.aten.silu_backward(torch.ones_like(x), 2 * x.detach()))
        with self.assertRaisesRegex(RuntimeError, "create_graph"):
            torch.autograd.grad(loss, x, create_graph=True)


class SavedRecomputeAutoTest(unittest.TestCase):
    """The auto rule on mocked admission predictions (CPU)."""
    GIB = 2**30

    def test_ladder_rule_and_growth(self):
        rule, gib = saved_recompute.auto_level, self.GIB

        def peaks(off, on, deep):
            return {"off": [v * gib for v in off], "narrow": [v * gib for v in on], "deep": [v * gib for v in deep]}
        budget = [100 * gib]
        self.assertEqual(rule(peaks([80], [40], [30]), budget), "off")
        self.assertEqual(rule(peaks([86], [40], [30]), budget), "narrow")  # the cheapest level that fits
        self.assertEqual(rule(peaks([200], [86], [60]), budget), "deep")
        self.assertEqual(rule(peaks([300], [150], [100]), budget), "deep")  # nothing fits: the deepest
        self.assertEqual(rule(peaks([10, 90], [5, 45], [4, 30]), budget * 2), "narrow")  # any rank
        self.assertEqual(rule({"off": [None], "narrow": [None], "deep": [None]}, [1]), "off")  # uncalibrated
        # Nothing within the threshold: the smallest predicted share, even if it is not the deepest level
        # (the P>1 model_graph "on" and "deep" lines cross at small rows).
        self.assertEqual(rule(peaks([40], [19.3], [20.6]), [20 * gib]), "narrow")
        auto = saved_recompute.AutoRecompute("test")
        self.assertEqual(auto.update(100, peaks([50], [30], [20]), budget, "first"), "off")
        self.assertEqual(auto.update(90, peaks([99], [60], [40]), budget, "smaller"), "off")  # no growth: no decision
        self.assertEqual(auto.update(120, peaks([90], [50], [35]), budget, "growth"), "narrow")
        self.assertEqual(auto.update(130, peaks([10], [5], [4]), budget, "later"), "narrow")  # never steps down
        self.assertEqual(auto.update(250, peaks([190], [95], [70]), budget, "growth"), "deep")
        self.assertEqual(auto.update(300, peaks([10], [5], [4]), budget, "later"), "deep")
        self.assertEqual([d["mode"] for d in auto.decisions], ["off", "on", "deep"])

    def test_single_bwd_auto_rule(self):
        # C3 auto: the ladder reads the predictions without C3, so its levels are those of C3 off; C3 runs
        # while the prediction with it fits the same share at that level, never back on within a level and
        # decided again after a climb: off+C3 -> off -> on+C3 -> on -> deep+C3 -> deep.
        gib = self.GIB

        def peaks(off, on, deep):
            return {"off": [off * gib], "narrow": [on * gib], "deep": [deep * gib]}
        budget = [100 * gib]
        with mock.patch.object(saved_recompute, "_SINGLE_BWD_AUTO", False):
            auto, plain = saved_recompute.AutoRecompute("c3"), saved_recompute.AutoRecompute("plain")
            for rows, without, with_c3, level, single in (
                    (100, (80, 40, 30), (84, 42, 31), "off", True),
                    (110, (84, 42, 31), (88, 44, 32), "off", False),  # C3 alone crosses 85%: it falls back first
                    (115, (82, 41, 30), (84, 42, 31), "off", False),  # never back on within a level
                    (120, (90, 45, 33), (94, 47, 34), "narrow", True),  # a climb decides it again
                    (150, (100, 84, 50), (110, 88, 52), "narrow", False),
                    (200, (200, 90, 60), (210, 95, 62), "deep", True),
                    (250, (250, 99, 84), (260, 104, 90), "deep", False),
                    (300, (100, 60, 40), (100, 60, 41), "deep", False)):
                with self.subTest(rows=rows):
                    self.assertEqual(auto.update(rows, peaks(*without), budget, "t", peaks_single_bwd=peaks(*with_c3)),
                                     level)
                    self.assertEqual(plain.update(rows, peaks(*without), budget, "t"), level)
                    self.assertEqual((auto.single_bwd, saved_recompute._SINGLE_BWD_AUTO), (single, single))
            # Uncalibrated (no admission model): off. An explicit level: C3 only, the same rule.
            none = {level: [None] for level in saved_recompute.LADDER}
            fresh = saved_recompute.AutoRecompute("uncalibrated")
            self.assertEqual(fresh.update(1, none, [0], "t", peaks_single_bwd=none), "off")
            self.assertFalse(fresh.single_bwd or saved_recompute._SINGLE_BWD_AUTO)
            fixed = saved_recompute.AutoRecompute("fixed")
            for rows, with_c3, single in ((100, (99, 99, 80), True), (120, (99, 99, 90), False),
                                          (130, (99, 99, 10), False)):
                self.assertEqual(fixed.update(rows, peaks(99, 99, 70), budget, "t", peaks_single_bwd=peaks(*with_c3),
                                              fixed="deep"), "deep")
                self.assertEqual(fixed.single_bwd, single)
            # A bound evaluator's collective decision takes precedence over the P=1 state.
            with saved_recompute.bound_admission(mock.Mock(recompute="off", single_bwd_decision=True)):
                self.assertTrue(saved_recompute.auto_single_bwd())
            self.assertFalse(saved_recompute.auto_single_bwd())

    def test_uncalibrated_architecture_never_relaxes(self):
        # Outside every measured K6 envelope (memory_admission.memory_model "derived", e.g. 12 layers at
        # 128/128/128) the ladder climbs on the derived bound and C3 stays off; for any model but RDX an
        # unpredicted level does not fit (the deepest level is taken).
        from fastmd.engine import memory_admission as ma
        gib = self.GIB
        derived = ma.MemoryModel(ma.Architecture(12, 128, 128, 128, 256, 7, 7, 4, 4, 6.0, 4.5), "derived")
        fits = {"off": [10 * gib], "narrow": [5 * gib], "deep": [4 * gib]}
        none = {level: [None] for level in saved_recompute.LADDER}
        with mock.patch.object(saved_recompute, "_SINGLE_BWD_AUTO", False):
            auto = saved_recompute.AutoRecompute("uncalibrated", memory=derived)
            self.assertEqual(auto.update(1, fits, [100 * gib], "t", peaks_single_bwd=fits), "off")
            self.assertFalse(auto.single_bwd)
            self.assertEqual(auto.update(2, {"off": [90 * gib], "narrow": [5 * gib], "deep": [4 * gib]},
                                         [100 * gib], "t", peaks_single_bwd=fits), "narrow")
            calibrated = saved_recompute.AutoRecompute("calibrated")
            self.assertEqual(calibrated.update(1, fits, [100 * gib], "t", peaks_single_bwd=fits), "off")
            self.assertTrue(calibrated.single_bwd)
            strict = saved_recompute.AutoRecompute("strict", memory=ma.MemoryModel(ma.OAM_10M, "oam_10m"))
            self.assertEqual(strict.update(1, none, [100 * gib], "t"), "deep")
            self.assertEqual(saved_recompute.AutoRecompute("rdx").update(1, none, [100 * gib], "t"), "off")

    @staticmethod
    def _predict(engine, profile, execution, rows, line_rows=0, *, checkpoint=False, recompute="off",
                 mixed_width=True, single_bwd=True, memory=None):
        return rows * {"off": 4, "narrow": 2, "deep": 1}[recompute] + (rows // 2 if single_bwd else 0)

    def test_gated_tail_decision_auto_and_explicit(self):
        # C2 (the fused narrow tail) at P=1: after the level, from the prediction with its increment, for auto
        # and for an explicit level alike; the level (the ladder) reads the predictions without it.
        from fastmd.engine import memory_admission as ma
        from op_selection_snapshots import RDX_ARCHITECTURE

        class Model:
            interaction_block = ()
            config = RDX_ARCHITECTURE  # the RDX calibration

            def parameters(self):
                return iter([mock.Mock(device=torch.device("cuda", 0), **{"element_size.return_value": 4})])
        rows, total, gib = 15_221_856, 200 * self.GIB, self.GIB
        config = mock.Mock(execution="eager", optimization_profile="topology", fingerprint="t",
                           kernel_options={"MATRIS_FUSED_GATED_TAIL": True, "MATRIS_FUSED_INDEXED_CAT_LINEAR": True,
                                           "MATRIS_MERGED_PROJECTION_SINGLE_BWD": "off"})
        # No entry has an increment since the K6 recalibration; the rule, with one.
        with mock.patch.dict(ma.GATED_TAIL_GIB_PER_MILLION_ROWS, {("single", "topology", "eager", "narrow"): 0.2}):
            base, with_tail = (ma.predict("single", "topology", "eager", rows, recompute="narrow", gated_tail=tail)
                               for tail in (False, True))
            for budget, runs in ((with_tail / 0.85 + gib, True), ((base + with_tail) / 1.7, False)):
                free = int(budget) + ma.margin_bytes(total)
                with self.subTest(budget=budget / gib), \
                        mock.patch.object(torch.cuda, "mem_get_info", return_value=(free, total)), \
                        mock.patch.object(torch.cuda, "memory_allocated", return_value=0), \
                        mock.patch.object(torch.cuda, "memory_reserved", return_value=0), \
                        mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
                        mock.patch.object(ma, "enabled", return_value=True):
                    auto, fixed = saved_recompute.AutoRecompute("auto"), saved_recompute.AutoRecompute("fixed")
                    self.assertEqual(auto.decide_local(Model(), config, rows), "narrow")
                    self.assertEqual(fixed.decide_local(Model(), config, rows, fixed="narrow"), "narrow")
                    self.assertEqual((auto.gated_tail, fixed.gated_tail), (runs, runs))
                    self.assertEqual((fixed.mode, fixed.decisions), ("off", []))  # an explicit level: C2 only
                    # A level without an increment always runs C2.
                    off = saved_recompute.AutoRecompute("off")
                    self.assertEqual(off.decide_local(Model(), config, rows, fixed="off"), "off")
                    self.assertTrue(off.gated_tail)
                    # mode_for: the entry point's explicit level, C2 decided the same way.
                    model = Model()
                    config.saved_tensor_recompute = "on"
                    with mock.patch("fastmd.config.current_config", return_value=config), \
                            mock.patch.object(saved_recompute, "_GATED_TAIL_AUTO", True):
                        self.assertEqual(saved_recompute.mode_for(model, rows), "narrow")
                        self.assertEqual(saved_recompute.gated_tail_allowed(), runs)

    def test_entry_point_decision_reads_the_model_calibration(self):
        # P=1 entry points: the recompute ladder predicts with the model's calibration (memory_admission
        # .memory_model): at 1,344 RDX atoms the RDX model stays off, the 10M OAM architecture climbs to "on",
        # and a model without a MatRIS architecture (no config, or one lacking a key) takes "deep".
        from fastmd.engine import memory_admission as ma
        from op_selection_snapshots import RDX_ARCHITECTURE

        class Model:
            interaction_block = ()

            def __init__(self, **config):
                self.config = {**RDX_ARCHITECTURE, **config}

            def parameters(self):
                return iter([mock.Mock(device=torch.device("cuda", 0), **{"element_size.return_value": 4})])
        rows, total = 1_951_244, 200 * self.GIB
        config = mock.Mock(execution="eager", optimization_profile="topology", fingerprint="t",
                           kernel_options={"MATRIS_MERGED_PROJECTION_SINGLE_BWD": "off"})
        free = int(177.5 * self.GIB) + ma.margin_bytes(total)
        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(free, total)), \
                mock.patch.object(torch.cuda, "memory_allocated", return_value=0), \
                mock.patch.object(torch.cuda, "memory_reserved", return_value=0), \
                mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
                mock.patch.object(saved_recompute, "ensure_expandable_segments", return_value=True), \
                mock.patch.object(ma, "enabled", return_value=True):
            bare, incomplete = Model(), Model()
            del bare.config, incomplete.config["norm_type"]
            for model, mode, calibration in (
                    (Model(), "off", "rdx_1m"),
                    (Model(num_layers=10, edge_feat_dim=128, three_body_feat_dim=128, three_body_cutoff=4.5),
                     "narrow", "oam_10m"),
                    (bare, "deep", "uncalibrated"), (incomplete, "deep", "uncalibrated")):
                with self.subTest(calibration=calibration, config=getattr(model, "config", None) is not None):
                    state = saved_recompute.AutoRecompute("t", saved_recompute.policy_threshold("fast", "eager"))
                    self.assertEqual(state.decide_local(model, config, rows), mode)
                    self.assertEqual(state.memory.calibration, calibration)
                    # recompute_policy "fast" only on predictions from a measured table.
                    self.assertEqual(state.decisions[-1]["threshold"],
                                     saved_recompute.AUTO_THRESHOLD if calibration == "uncalibrated" else 0.98)


    def test_fast_policy_only_on_measured_predictions(self):
        # "fast" relaxes a decision only where every prediction it reads comes from a measured table
        # (memory_admission.MemoryModel.calibrated and entry_calibrated); else it decides as "safe".
        # One prediction not from a measured table (``calibrated`` False): at most "safe"'s share.
        peaks = {"off": [90], "narrow": [50], "deep": [40]}
        for calibrated, level in ((True, "off"), (False, "narrow")):
            state = saved_recompute.AutoRecompute("t", saved_recompute.policy_threshold("fast", "eager"))
            self.assertEqual(state.update(1, peaks, [100], "t", calibrated=calibrated), level)
            self.assertEqual(state.decisions[-1]["threshold"], 0.98 if calibrated else saved_recompute.AUTO_THRESHOLD)

    def test_policy_thresholds(self):
        threshold = saved_recompute.policy_threshold
        for kind in ("eager", "model_graph", "whole_step", "single"):
            self.assertEqual(threshold("safe", kind), saved_recompute.AUTO_THRESHOLD)
        self.assertEqual([threshold("fast", kind) for kind in ("eager", "model_graph")],
                         [0.98, 0.95])
        for kind in ("whole_step", "single"):
            self.assertEqual(threshold("fast", kind), saved_recompute.AUTO_THRESHOLD)

    def test_capture_reads_the_entry_point_state_without_deciding(self):
        # A P=1 capture's byte rule and warmup count read the running level and whether "fast" still has to
        # check an uncaptured run; neither decides (no DECISIONS entry, no state change).
        model = torch.nn.Linear(1, 1)
        state = saved_recompute.AutoRecompute("test")
        state.mode, state.fast_only = "narrow", True
        decisions = len(saved_recompute.DECISIONS)
        with mock.patch.dict(saved_recompute._SINGLE, {(id(model), "t"): state}):
            for requested, level in (("auto", "narrow"), ("deep", "deep")):
                config = mock.Mock(saved_tensor_recompute=requested, fingerprint="t")
                self.assertEqual(saved_recompute.current_mode(model, config), level)
                self.assertTrue(saved_recompute.capture_check_pending(model, config))  # C3 auto keeps "fast" too
            state.fast_only = False
            self.assertFalse(saved_recompute.capture_check_pending(model, config))
            with saved_recompute.bound_admission(mock.Mock(recompute="deep")):  # an evaluator decided collectively
                self.assertEqual(saved_recompute.current_mode(model, config), "deep")
                self.assertFalse(saved_recompute.capture_check_pending(model, config))
        self.assertEqual(saved_recompute.current_mode(model, mock.Mock(saved_tensor_recompute="auto", fingerprint="t")),
                         "off")  # no state yet
        self.assertEqual((len(saved_recompute.DECISIONS), state.mode), (decisions, "narrow"))


if __name__ == "__main__":
    unittest.main()
