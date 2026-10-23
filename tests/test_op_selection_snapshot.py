"""Golden operator-selection snapshots (unified engine step 2.0, design §4.7, gate G-S3).

Kernel-selection snapshots are a structural perf guard, so every test here
belongs to the opt-in perf tier (``MATRIS_RUN_PERF_TIER=1``,
tests/matris_test_tiers.py; ``MATRIS_RUN_OP_SELECTION_SNAPSHOTS=1`` implies
it). CPU tests check that the registry and the committed goldens agree, that
the golden encoding is lossless and canonical, which differences fail the
check, and that a toy MatRIS recording is deterministic and names the generic
CPU lowerings. The CUDA tests (skipped without a GPU or the RDX assets)
regenerate configurations in fresh subprocesses with
``tests/op_selection_snapshots.py --check`` and compare them with their
goldens: a quick subset in the perf tier, the full matrix with
``MATRIS_RUN_OP_SELECTION_SNAPSHOTS=1`` (runtime depends on the selected GPU).

The harness takes /tmp/matris_gpu.lock per wave of subprocesses unless this
process already runs under it (``flock /tmp/matris_gpu.lock python -m
unittest ...``), and uses the GPUs in CUDA_VISIBLE_DEVICES (default: all).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

import torch

from matris_test_tiers import PERF_TIER
import op_selection_snapshots as snapshots

_HARNESS = Path(snapshots.__file__).resolve()
_FULL = os.environ.get("MATRIS_RUN_OP_SELECTION_SNAPSHOTS") == "1"
# Kernel-selection snapshots are a structural perf guard: the full matrix implies the tier.
_PERF_TIER = _FULL or PERF_TIER
_perf_tier = unittest.skipUnless(_PERF_TIER, "G-S3 snapshots are tier 3: set MATRIS_RUN_PERF_TIER=1")
# Single-GPU configurations (enforced by G-S3), one of each cheap kind: ragged
# eager, bucketed capture, row thresholds, zero-triplet graph, the equal-width
# fused kernels and the whole-step capture with its eager fallback. The
# full matrix contains only single-device configurations.
QUICK = ("cluster_topology_gpumd_eager", "cluster_legacy_gpumd_model_graph",
         "synth_t60001_topology_fusion", "dimers_topology_gpumd_eager",
         "eqw1344_topology_gpumd_eager", "rdx1344_topology_whole_step_fallback")


def _assets_available() -> bool:
    return all(snapshots.assets_available(name) for name in snapshots.CONFIGS)


def _lines_doc(name, lines, aten):
    sections = [{"label": "cold", "entries": lines, "fallbacks": [], "aten": aten}]
    return {"sections": sections, "summary": snapshots.summarize(sections)}


@_perf_tier
class OpSelectionRegistryTest(unittest.TestCase):
    def test_every_golden_has_a_config_and_every_config_a_golden(self):
        goldens = {path.stem for path in snapshots.GOLDEN_DIR.glob("*.json")}
        self.assertEqual(sorted(goldens - set(snapshots.CONFIGS)), [], "stale goldens")
        self.assertEqual(sorted(set(snapshots.CONFIGS) - goldens), [], "missing goldens")

    def test_goldens_are_canonical_and_within_budget(self):
        docs = {name: json.loads((snapshots.GOLDEN_DIR / f"{name}.json").read_text())
                for name in snapshots.CONFIGS}
        plain = snapshots.expand_documents(docs)
        encoded = snapshots.encode_goldens(plain)
        total = 0
        for name, doc in docs.items():
            with self.subTest(config=name):
                self.assertEqual(doc["schema"], snapshots.GOLDEN_SCHEMA)
                self.assertEqual(doc["config"], name)
                self.assertEqual(doc["spec"], snapshots.golden_spec(name))
                text = (snapshots.GOLDEN_DIR / f"{name}.json").read_text()
                # What --write produces from this content, byte for byte.
                self.assertEqual(snapshots.dumps(encoded[name]), text)
                self.assertEqual(plain[name]["summary"],
                                 snapshots.summarize(plain[name]["sections"]))
                total += len(text.encode())
        self.assertLess(total, 4_000_000)

    def test_encoding_is_lossless_and_diffs_detect_changes(self):
        base = [f"{phase} block.{layer} {op} BM=4 rows={rows} width=32"
                for phase in ("fwd", "bwd") for layer in range(4)
                for op, rows in (("residual_add/triton", 951366), ("gated_ln/triton", 60889))]
        # Another bucket: rows substituted, one checkpoint line inserted.
        bucket = [line.replace("951366", "958464").replace("60889", "61440") for line in base]
        bucket.insert(1, "fwd block.0 layer_checkpoint/on layer=attn")
        aten = {f"site{i}": "aten.mm.default=1 aten.t.default=1" for i in range(8)}
        changed = dict(aten, site3="aten.addmm.default=1 aten.t.default=1")
        plain = {"a": _lines_doc("a", base, aten), "b": _lines_doc("b", bucket, changed),
                 "c": _lines_doc("c", list(base), dict(aten))}
        encoded = snapshots.encode_documents(plain)
        self.assertEqual(snapshots.expand_documents(encoded), plain)
        self.assertTrue(snapshots.diff_documents(plain["a"], plain["b"]))
        self.assertEqual(snapshots.diff_documents(plain["a"], plain["c"]), [])

    def test_check_fails_on_selection_content_only(self):
        base = _lines_doc("a", ["fwd s op/impl rows=1"], {"s": "aten::mm=1"})
        base.update(spec={"kind": "gpumd"}, outcome="ok", resolved={"options": {"A": 1}},
                    observed={"captures": 1}, info={"seconds": 1.0})

        def check(candidate, name="rdx1344_topology_gpumd_eager", strict=False):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                failed = snapshots.compare_documents([name], {name: base}, {name: candidate},
                                                     strict=strict)
            return bool(failed), out.getvalue()

        # A new default-off kernel option or a runner counter: a note, not a failure.
        noted = dict(base, resolved={"options": {"A": 1, "B": False}},
                     observed={"captures": 2}, info={"seconds": 9.0})
        failed, text = check(noted)
        self.assertFalse(failed, text)
        self.assertTrue(check(noted, strict=True)[0])
        self.assertFalse(check(dict(base, info={"seconds": 2.0}), strict=True)[0])
        # A configuration dump is compared through the fingerprint beside it,
        # as the goldens store it (a regenerated snapshot keeps the dump).
        name = "rdx1344_topology_gpumd_eager"
        dump = dict(base, resolved={"configuration": {"A": 1}, "fingerprint": "f1"})
        stored = snapshots.golden_view(dump)
        self.assertEqual(stored["resolved"], {"fingerprint": "f1"})
        self.assertNotIn("summary", stored)
        golden = snapshots.expand_documents(snapshots.encode_documents({name: stored}))
        for candidate, strict_fails in ((dump, False), (dict(base, resolved={
                "configuration": {"A": 2}, "fingerprint": "f2"}), True)):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(snapshots.compare_documents([name], golden, {name: candidate}))
                self.assertEqual(bool(snapshots.compare_documents(
                    [name], golden, {name: candidate}, strict=True)), strict_fails)
        # A different kernel parameter, outcome or dynamo program fails.
        changed = _lines_doc("a", ["fwd s op/impl rows=2"], {"s": "aten::mm=1"})
        self.assertTrue(check(dict(base, **changed))[0])
        self.assertTrue(check(dict(base, outcome={"error": "x"}))[0])
        self.assertTrue(check(dict(base, dynamo={"stats": {"unique_graphs": 2}}))[0])
    def test_purity_compares_the_warm_and_repeat_phases(self):
        sections = [{"label": label, "entries": entries, "fallbacks": [], "aten": {}}
                    for label, entries in (("warm", []), ("warm.eval1", ["a"]),
                                           ("repeat", []), ("repeat.eval1", ["a"]))]
        self.assertEqual(snapshots.purity(sections, "warm", "repeat"),
                         {"compared": ["warm", "repeat"], "identical": True})
        sections[3] = dict(sections[3], entries=["b"])
        self.assertFalse(snapshots.purity(sections, "warm", "repeat")["identical"])


@_perf_tier
class OpSelectionCpuSmokeTest(unittest.TestCase):
    def test_toy_recording_is_deterministic_and_round_trips(self):
        from single_gpu_fixtures import frozen_fixture, toy_model

        def record():
            model = toy_model(smoothed=True)
            _, graph = frozen_fixture()
            with snapshots.Recording([model]) as recording:
                for label in ("cold", "warm"):
                    recording.phase(label)
                    model([graph], task="ef")
            return snapshots.plain_document("toy", recording.snapshot(), outcome="ok")

        first, second = record(), record()
        self.assertEqual(snapshots.dumps(first), snapshots.dumps(second))
        self.assertEqual([section["label"] for section in first["sections"]],
                         ["cold", "cold.eval1", "warm", "warm.eval1"])
        # A repeated evaluation of the same model records the same program.
        self.assertTrue(snapshots.purity(first["sections"], "cold", "warm")["identical"])
        self.assertEqual(first["recorder"]["invalid_params"], 0)
        evaluation = first["sections"][1]
        self.assertTrue(evaluation["entries"] and evaluation["aten"])
        self.assertTrue(all(re.match(r"(fwd|bwd|recompute) \S+ \S+/\S+", line)
                            for line in evaluation["entries"]))
        heads = [line.split(" ", 3)[:3] for line in evaluation["entries"]]
        selected = {tuple(choice.split("/", 1)) for _, _, choice in heads}
        # CPU: every dispatch site names its generic lowering.
        for expected in [("silu", "torch"), ("sigmoid", "torch"), ("residual_add", "torch"),
                         ("pair_average", "torch_aggregate"),
                         ("pair_expand", "torch_index_select"),
                         ("weighted_segment_sum", "torch_aggregate"),
                         ("segment_softmax", "torch"), ("gated_ln", "torch"),
                         ("gather_cat3", "torch"), ("gather_cat4", "torch"),
                         ("line_envelope", "torch_index_select_product"),
                         ("threebody_basis", "torch"), ("threebody_linear", "torch_linear"),
                         ("layer_checkpoint", "off"), ("attention_projection", "torch_linear")]:
            self.assertIn(expected, selected)
        sites = {site for _, site, _ in heads}
        self.assertIn("interaction_block.0.attn_block_atom_graph", sites)
        self.assertIn("interaction_block.1", sites)
        encoded = snapshots.encode_documents({"toy": first})
        self.assertEqual(snapshots.expand_documents(encoded)["toy"], first)


@unittest.skipUnless(_PERF_TIER and torch.cuda.is_available() and _assets_available(),
                     "perf tier (MATRIS_RUN_PERF_TIER=1); needs CUDA, the RDX checkpoint and structures")
class OpSelectionGoldenGpuTest(unittest.TestCase):
    def _check(self, selection, timeout):
        command = [sys.executable, "-B", str(_HARNESS), "--check", *selection]
        result = subprocess.run(command, cwd=str(snapshots.ROOT), capture_output=True,
                                text=True, timeout=timeout)
        self.assertEqual(result.returncode, 0,
                         f"{' '.join(command)}\n{result.stdout[-8000:]}\n{result.stderr[-4000:]}")

    def test_quick_subset_matches_goldens(self):
        self._check(["--config", *QUICK], timeout=900)

    @unittest.skipUnless(_FULL, "set MATRIS_RUN_OP_SELECTION_SNAPSHOTS=1 for the full matrix")
    def test_full_matrix_matches_goldens(self):
        self._check(["--all"], timeout=4 * 3600)


if __name__ == "__main__":
    unittest.main()
