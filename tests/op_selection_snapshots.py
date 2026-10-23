"""Golden operator-selection snapshots (unified engine step 2.0, design §4.7).

Records what every configuration of the §4.7 matrix actually executes with
:class:`fastmd.engine.recorder.OpPlanRecorder` and compares it with the
goldens under ``tests/golden/op_selection/<config>.json`` (INV-1, gate G-S3).
See ``tests/golden/op_selection/README.md`` for the matrix and the format.

Usage (run from the checkout root; the parent imports neither torch nor fastmd)::

    python -B tests/op_selection_snapshots.py --list
    python -B tests/op_selection_snapshots.py --check --all [--out-dir DIR] [--strict]
    python -B tests/op_selection_snapshots.py --check --config NAME [NAME ...]
    python -B tests/op_selection_snapshots.py --write --all [--from DIR]
    python -B tests/op_selection_snapshots.py --compare DIR_A DIR_B

``--check`` fails on a difference in what executed (see ``NOTE_KEYS`` and the
README for what is only reported); ``--compare`` requires full equality.

Every configuration runs in a fresh Python subprocess (clean module globals,
caches, Triton and dynamo state) with ``MATRIS_*`` removed from its
environment (except ``MATRIS_TEST_ASSETS``); the parent only orchestrates. Subprocesses run in waves of at
most one per GPU (``--gpus``, default: the parent's CUDA_VISIBLE_DEVICES or
every GPU); each configuration uses one GPU.

GPU lock: the parent holds ``flock(--gpu-lock)`` (default
``/tmp/matris_gpu.lock``) for the duration of each wave only, never between
waves. ``--gpu-lock auto`` (the default) skips this when an ancestor process
already holds that lock, e.g. ``flock /tmp/matris_gpu.lock python -m unittest
...``, which would otherwise deadlock; ``--gpu-lock none`` disables locking.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import difflib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from matris_test_assets import RDX_10752, RDX_1344, RDX_CHECKPOINT, test_asset, test_assets_available

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "op_selection"
STRUCTURE_DIR = GOLDEN_DIR / "structures"
# The RDX checkpoint and the two RDX crystals live outside the repository, in
# the MATRIS_TEST_ASSETS directory (tests/matris_test_assets.py); None when it
# is unset. Configurations that need a missing asset are skipped.
RDX_MODEL = test_asset(RDX_CHECKPOINT)
STRUCTURE_FILES = {
    "rdx_1344": test_asset(RDX_1344),
    "rdx_10752": test_asset(RDX_10752),
    "rdx_cluster": STRUCTURE_DIR / "rdx_cluster_r5.extxyz",
}
_STRUCTURE_ASSETS = {"rdx_1344": RDX_1344, "rdx_10752": RDX_10752}
# The one MATRIS_* variable a child keeps: it names files, not a kernel option.
_CHILD_KEEPS = ("MATRIS_TEST_ASSETS",)
GOLDEN_SCHEMA = "fastmd.op_selection_golden/1"
DEFAULT_LOCK = "/tmp/matris_gpu.lock"
PROFILES = ("legacy", "topology", "generic")

# RDX checkpoint architecture (widths 128/32/32). The equal-width model uses
# it with the MatRIS default widths 128/128/128 (also the MatRIS-10M-OAM and
# 4M-FeCOH production widths), randomly initialised with a fixed seed.
RDX_ARCHITECTURE = {
    "num_layers": 4, "node_feat_dim": 128, "edge_feat_dim": 32, "three_body_feat_dim": 32,
    "mlp_hidden_dims": [128, 256, 128], "dropout": 0.0, "use_bias": False,
    "distance_expansion": "Bessel", "three_body_expansion": "fourier", "num_radial": 7,
    "num_angular": 7, "max_l": 4, "max_n": 4, "envelope_exponent": 8,
    "graph_conv_mlp": "gatemlp", "activation_type": "silu", "norm_type": "layer",
    "pairwise_cutoff": 6, "three_body_cutoff": 4.0, "use_smoothed_for_delta_edge": False,
    "learnable_basis": True, "is_intensive": True, "is_conservation": True,
    "reference_energy": "mpa",
}
EQUAL_WIDTH_SEED = 20260928
MD_TIMESTEP_FS = 0.2      # the RDX MD driver and the G-S1 benchmark
NHC_TDAMP_FS = 20.0
TEMPERATURE_K = 300.0
BERENDSEN_TAUT_FS = 100.0
MOMENTA_SEED = 106175
# Unrecorded evaluations before recording an enable_compile configuration.
COMPILE_WARMUP = 2
# Candidate skin far below one step's displacement at 300 K: every whole-step
# window overflows, so the recovery replay also overflows and the runner
# re-executes the step eagerly in fusion scope (its eager fallback path).
FALLBACK_SKIN = 2.0e-4

# ---------------------------------------------------------------- registry

CONFIGS: dict[str, dict] = {}


def _add(name: str, covers: str, **spec) -> None:
    if name in CONFIGS:
        raise ValueError(f"duplicate op-selection config {name!r}")
    spec.setdefault("model", "rdx")
    spec.setdefault("checkpoint", "off")
    spec.setdefault("gpus", 1)
    CONFIGS[name] = {"covers": covers, **spec}


def _build_registry() -> None:
    for profile in PROFILES:
        rdx = dict(structure="rdx_1344", profile=profile)
        _add(f"rdx1344_{profile}_gpumd_eager", "GPUResidentMolecularDynamics.evaluate, eager",
             kind="gpumd", execution="eager", **rdx)
        _add(f"rdx1344_{profile}_gpumd_model_graph",
             "BucketedGraphRunner capture of 4 buckets at construction, then replay",
             kind="gpumd", execution="model_graph", **rdx)
        _add(f"rdx1344_{profile}_whole_step_nve",
             "WholeStepGraphRunner: initial eager evaluation in fusion scope, warmup, capture, replay",
             kind="whole_step", ensemble="nve", **rdx)
        _add(f"rdx1344_{profile}_calc_eager", "MLIPCalculator (ASE), eager",
             kind="calculator", execution="eager", **rdx)
        _add(f"rdx1344_{profile}_calc_model_graph",
             "MLIPCalculator: first calculate captures one bucket, then replay",
             kind="calculator", execution="model_graph", **rdx)
    for profile in ("topology", "legacy", "generic"):
        _add(f"rdx1344_{profile}_whole_step_fallback",
             "WholeStepGraphRunner eager fallback: stale candidates -> rollback, recovery replay, "
             "eager step in fusion scope",
             kind="whole_step", ensemble="nve", structure="rdx_1344", profile=profile,
             candidate_skin=FALLBACK_SKIN)
        _add(f"rdx1344_{profile}_replicas2",
             "BatchedWholeStepGraphRunner, 2 replicas, Berendsen NVT (padded batched graphs)",
             kind="replicas", replicas=2, structure="rdx_1344", profile=profile)
        _add(f"rdx1344_{profile}_replicas2_fallback",
             "BatchedWholeStepGraphRunner eager fallback: stale candidates -> rollback, recovery "
             "replay, eager step in fusion scope",
             kind="replicas", replicas=2, structure="rdx_1344", profile=profile,
             candidate_skin=FALLBACK_SKIN)
    _add("rdx1344_topology_whole_step_nhc", "WholeStepGraphRunner with the NHC NVT integrator",
         kind="whole_step", ensemble="nvt_nhc", structure="rdx_1344", profile="topology")
    for profile in PROFILES:
        _add(f"rdx1344_{profile}_gpumd_eager_compile",
             "enable_compile=True (torch.compile of the layer modules), eager",
             kind="gpumd", execution="eager", structure="rdx_1344", profile=profile, compile=True)
    for profile in PROFILES:
        _add(f"rdx1344_{profile}_gpumd_eager_ckpt_on",
             "checkpoint=on (line attention/refinement recomputation), eager",
             kind="gpumd", execution="eager", structure="rdx_1344", profile=profile,
             checkpoint="on")
    for profile in PROFILES:
        _add(f"rdx1344_{profile}_gpumd_eager_ckpt_auto",
             "checkpoint=auto (strict > 60000 triplets: on for RDX-1344), eager",
             kind="gpumd", execution="eager", structure="rdx_1344", profile=profile,
             checkpoint="auto")
    _add("rdx1344_topology_gpumd_model_graph_ckpt_on",
         "checkpoint=on with model_graph capture (accepted with a ConfigurationWarning)",
         kind="gpumd", execution="model_graph", structure="rdx_1344", profile="topology",
         checkpoint="on", may_reject=True)
    for profile in ("legacy", "topology"):
        for execution in ("eager", "model_graph"):
            _add(f"cluster_{profile}_gpumd_{execution}",
                 "small RDX cluster (all row counts below 10k/20k/60k)",
                 kind="gpumd", execution=execution, structure="rdx_cluster", profile=profile)
    for profile in ("topology", "legacy"):
        _add(f"rdx10752_{profile}_gpumd_eager", "RDX 10,752 atoms, eager",
             kind="gpumd", execution="eager", structure="rdx_10752", profile=profile)
    for profile in PROFILES:
        for execution in ("eager", "model_graph"):
            _add(f"eqw1344_{profile}_gpumd_{execution}",
                 "equal-width model (128/128/128) on RDX-1344",
                 kind="gpumd", execution=execution, structure="rdx_1344", profile=profile,
                 model="equal_width")
    _add("eqw1344_topology_whole_step_nve", "equal-width model, whole_step",
         kind="whole_step", ensemble="nve", structure="rdx_1344", profile="topology",
         model="equal_width")
    for structure, what in (("isolated", "cluster plus two isolated atoms "
                                         "(isolated_atoms='reference')"),
                            ("dimers", "well-separated dimers: zero-triplet graph")):
        for profile in ("legacy", "topology"):
            for execution in ("eager", "model_graph"):
                _add(f"{structure}_{profile}_gpumd_{execution}", what,
                     kind="gpumd", execution=execution, structure=structure, profile=profile)
    # Under "topology" the *_EAGER switches are on, so eager and fusion scope
    # select alike; under "legacy" only fusion scope enables the >= 60000-row
    # kernels. The legacy directed-edge pair around 60000 (indexed projection
    # off) shows that no atom-graph choice switches there: the aligned
    # gather-cat kernel is used on the line graph only.
    synthetic = dict(kind="synthetic", model="equal_width_1layer", checkpoint="auto")
    for profile, scope, rows in (
            ("topology", "fusion", (9999, 10000, 19999, 20000, 59999, 60000, 60001)),
            ("topology", "eager", (60000, 60001)),
            ("legacy", "fusion", (59999, 60000, 60001)),
            ("legacy", "eager", (60000, 60001))):
        for triplets in rows:
            _add(f"synth_t{triplets}_{profile}_{scope}",
                 f"one Interaction_Block, exactly {triplets} triplet rows, {scope} scope",
                 triplets=triplets, profile=profile, fusion_scope=scope == "fusion", **synthetic)
    for profile, edges_rows in (("topology", (9998, 10000, 19998, 20000)),
                                ("legacy", (59998, 60000))):
        for edges in edges_rows:
            _add(f"synth_e{edges}_{profile}_fusion",
                 f"one Interaction_Block, exactly {edges} directed-edge rows (1000 triplets), "
                 "fusion scope", edges=edges, triplets=1000, profile=profile, fusion_scope=True,
                 **synthetic)
    # The defaults (optimization_profile and execution unset: topology, auto).
    _add("rdx1344_default_calc",
         "MLIPCalculator defaults: a new system's call eager, its repeat model_graph (skin reuse)",
         kind="calculator", structure="rdx_1344", profile=None)
    _add("rdx1344_default_whole_step_nve", "GPUResidentMolecularDynamics defaults, nve: whole_step",
         kind="whole_step", ensemble="nve", structure="rdx_1344", profile=None, entry_defaults=True)


_build_registry()


def required_assets(name: str) -> list:
    """The MATRIS_TEST_ASSETS files that config ``name`` needs."""
    spec = CONFIGS[name]
    needed = [RDX_CHECKPOINT] if spec["model"] == "rdx" else []
    if spec.get("structure") in _STRUCTURE_ASSETS:
        needed.append(_STRUCTURE_ASSETS[spec["structure"]])
    return needed


def assets_available(name: str) -> bool:
    return test_assets_available(*required_assets(name))

# ------------------------------------------------ rendering and encoding
# A section has three components: "entries" and "fallbacks", one line per
# record ("<phase> <site> <op>/<impl>[ !<reason>] key=value ..."), and "aten"
# ({context: "op=count ..."}). In a golden file a component is either stored
# plainly or refers to a component recorded before it, in the same file
# ("<label>") or in a config earlier in the registry ("<config>:<label>"):
#   {"same_as": ref}                                identical content;
#   {"like": ref, "ints": {old: new}, "patch": [[start, stop, [lines]], ...]}
#       the lines of ref with integer parameter values substituted, then the
#       ranges [start, stop) of that list replaced (e.g. another capacity
#       bucket, or checkpoint recomputation lines inserted);
#   {"like": ref, "set": {context: line}, "unset": [context, ...]}  aten delta.
# Goldens are always encoded as one set (encode_documents), so a reference
# never changes what another golden means; every encoding is verified to
# expand back to the recorded content, and comparisons use expanded content.

_BARE = re.compile(r"[A-Za-z_<][A-Za-z0-9_.:<>/@+-]*\Z")
_INT = re.compile(r"(?<=[=\[,])-?\d+(?=[,\] ]|\Z)")
_LINE_KEYS = ("entries", "fallbacks")


def _value(value) -> str:
    if isinstance(value, str):
        if _BARE.match(value) and value not in ("true", "false", "null"):
            return value
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_value(item) for item in value) + "]"
    return json.dumps(value, sort_keys=True)


def entry_line(entry: dict) -> str:
    head = f"{entry['phase']} {entry['site']} {entry['op']}/{entry['impl']}"
    if "reason" in entry:
        head += f" !{entry['reason']}"
    params = entry.get("params") or {}
    if not params:
        return head
    return head + " " + " ".join(f"{key}={_value(params[key])}" for key in sorted(params))


def aten_line(counts: dict) -> str:
    return " ".join(f"{op}={counts[op]}" for op in sorted(counts))


def plain_sections(sections: list) -> list:
    """Recorder snapshot sections -> [{label, entries, fallbacks, aten}] (rendered)."""
    return [{"label": section["label"],
             "entries": [entry_line(entry) for entry in section["entries"]],
             "fallbacks": [entry_line(entry) for entry in section["fallbacks"]],
             "aten": {ctx: aten_line(counts)
                      for ctx, counts in sorted(section["aten_counts"].items())}}
            for section in sections]


def _split(line: str) -> tuple[str, str]:
    # Parameters follow "<phase> <site> <op>/<impl>" and an optional "!reason".
    parts = line.split(" ", 3)
    if len(parts) == 4 and parts[3].startswith("!"):
        reason, _, params = parts[3].partition(" ")
        return " ".join(parts[:3] + [reason]), params
    return " ".join(parts[:3]), parts[3] if len(parts) == 4 else ""


def _masked(line: str) -> str:
    head, params = _split(line)
    return head + "\x00" + _INT.sub("#", params)


def _substitute(lines: list, mapping: dict) -> list:
    if not mapping:
        return list(lines)

    def replace(match):
        return str(mapping.get(match.group(0), match.group(0)))

    out = []
    for line in lines:
        head, params = _split(line)
        out.append(head + (" " + _INT.sub(replace, params) if params else ""))
    return out


def _apply_like(base: list, encoding: dict) -> list:
    lines = _substitute(base, encoding.get("ints", {}))
    out, cursor = [], 0
    for start, stop, replacement in encoding.get("patch", []):
        out.extend(lines[cursor:start])
        out.extend(replacement)
        cursor = stop
    out.extend(lines[cursor:])
    return out


def _like_lines(base: list, base_masked: list, lines: list, masked: list) -> dict:
    """Integer substitution (majority over aligned lines) plus a line patch."""
    votes = {}
    matcher = difflib.SequenceMatcher(None, base_masked, masked, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for old, new in zip(base[i1:i2], lines[j1:j2]):
            for a, b in zip(_INT.findall(_split(old)[1]), _INT.findall(_split(new)[1])):
                count = votes.setdefault(a, {})
                count[b] = count.get(b, 0) + 1
    ints = {}
    for a, count in votes.items():
        b = min(count, key=lambda value: (-count[value], value))
        if a != b:
            ints[a] = int(b)
    substituted = _substitute(base, ints)
    matcher = difflib.SequenceMatcher(None, substituted, lines, autojunk=False)
    patch = [[i1, i2, lines[j1:j2]] for tag, i1, i2, j1, j2 in matcher.get_opcodes()
             if tag != "equal"]
    encoding = {}
    if ints:
        encoding["ints"] = {a: ints[a] for a in sorted(ints, key=int)}
    if patch:
        encoding["patch"] = patch
    return encoding


def _size(value) -> int:
    return len(json.dumps(value))


class _Index:
    """Components already encoded, for exact and similar-content references."""

    CANDIDATES = 3

    def __init__(self):
        self.order = 0
        self.lines = {key: [] for key in _LINE_KEYS}   # (order, ref, lines, masked, set)
        self.exact = {key: {} for key in _LINE_KEYS}   # tuple(lines) -> [(order, ref)]
        self.aten = []                                 # (order, ref, aten, set)
        self.aten_exact = {}

    @staticmethod
    def _ref(name, owner, label):
        return label if owner == name else f"{owner}:{label}"

    def _pick(self, hits, name):
        # Prefer the latest identical component of the same file, else the first.
        local = [hit for hit in hits if hit[1][0] == name]
        order, (owner, label) = local[-1] if local else hits[0]
        return self._ref(name, owner, label)

    def _similar(self, pool, items):
        scored = []
        for entry in pool:
            other = entry[-1]
            union = len(items | other)
            if union:
                scored.append((-len(items & other) / union, entry[0], entry))
        return [entry for _, _, entry in sorted(scored, key=lambda s: (s[0], s[1]))[
            :self.CANDIDATES]]

    def encode_lines(self, name, key, lines):
        if not lines:
            return []
        hits = self.exact[key].get(tuple(lines))
        if hits:
            return {"same_as": self._pick(hits, name)}
        masked = [_masked(line) for line in lines]
        best, best_size = None, _size(lines) // 2
        for _, (owner, label), base, base_masked, _ in self._similar(
                self.lines[key], set(masked)):
            encoding = _like_lines(base, base_masked, lines, masked)
            size = _size(encoding)
            if size < best_size:
                best, best_size = {"like": self._ref(name, owner, label), **encoding}, size
        return best if best is not None else lines

    def encode_aten(self, name, aten):
        if not aten:
            return {}
        items = frozenset(aten.items())
        hits = self.aten_exact.get(items)
        if hits:
            return {"same_as": self._pick(hits, name)}
        best, best_size = None, _size(aten) // 2
        for _, (owner, label), base, _ in self._similar(self.aten, set(items)):
            delta = {"like": self._ref(name, owner, label),
                     "set": {ctx: line for ctx, line in aten.items() if base.get(ctx) != line},
                     "unset": sorted(ctx for ctx in base if ctx not in aten)}
            size = _size(delta)
            if size < best_size:
                best, best_size = delta, size
        return best if best is not None else aten

    def add(self, name, section):
        self.order += 1
        ref = (name, section["label"])
        for key in _LINE_KEYS:
            lines = tuple(section[key])
            if lines:
                if lines not in self.exact[key]:
                    masked = [_masked(line) for line in lines]
                    self.lines[key].append((self.order, ref, list(lines), masked, set(masked)))
                self.exact[key].setdefault(lines, []).append((self.order, ref))
        if section["aten"]:
            items = frozenset(section["aten"].items())
            if items not in self.aten_exact:
                self.aten.append((self.order, ref, section["aten"], set(items)))
            self.aten_exact.setdefault(items, []).append((self.order, ref))


def encode_documents(docs: dict) -> dict:
    """{name: plain document} (in registry order) -> {name: golden document}."""
    index = _Index()
    encoded = {}
    for name, doc in docs.items():
        sections = []
        for section in doc["sections"]:
            sections.append({"label": section["label"],
                             **{key: index.encode_lines(name, key, section[key])
                                for key in _LINE_KEYS},
                             "aten": index.encode_aten(name, section["aten"])})
            index.add(name, section)
        encoded[name] = {**doc, "sections": sections}
    expanded = expand_documents(encoded)
    for name, doc in docs.items():
        if expanded[name]["sections"] != doc["sections"]:
            raise AssertionError(f"{name}: golden encoding does not round-trip")
    return encoded


def expand_documents(docs: dict) -> dict:
    """{name: golden document} -> {name: plain document}; references must resolve in ``docs``."""
    sections = {name: {section["label"]: section for section in doc.get("sections", [])}
                for name, doc in docs.items()}
    memo = {}

    def component(name, label, key):
        cached = memo.get((name, label, key))
        if cached is not None:
            return cached
        value = sections[name][label][key]

        def resolve(ref):
            owner, _, target = ref.rpartition(":")
            return component(owner or name, target, key)

        if key == "aten":
            if set(value) == {"same_as"}:
                value = resolve(value["same_as"])
            elif set(value) == {"like", "set", "unset"}:
                base = {ctx: line for ctx, line in resolve(value["like"]).items()
                        if ctx not in value["unset"]}
                base.update(value["set"])
                value = dict(sorted(base.items()))
        elif isinstance(value, dict) and "same_as" in value:
            value = resolve(value["same_as"])
        elif isinstance(value, dict):
            value = _apply_like(resolve(value["like"]), value)
        memo[name, label, key] = value
        return value

    expanded = {name: {**doc, "sections": [
        {"label": section["label"],
         **{key: list(component(name, section["label"], key)) for key in _LINE_KEYS},
         "aten": dict(component(name, section["label"], "aten"))}
        for section in doc.get("sections", [])]} for name, doc in docs.items()}
    for doc in expanded.values():
        if "summary" not in doc and "sections" in doc:  # goldens store no summary
            doc["summary"] = summarize(doc["sections"])
    return expanded


# A golden stores what the check compares, minus what is derived: ``summary``
# is recomputed from the sections by expand_documents, and a resolved
# ``configuration`` dump (ResolvedConfig.as_dict(), ~3 KB per config) is
# represented by the fingerprint recorded beside it (sha256 of the same dump
# without ``report``). Regenerated snapshots (--out-dir) keep the full dump.
FINGERPRINT_KEYS = ("fingerprint", "config_fingerprint")


def fingerprinted(value):
    """``value`` without the configuration dumps that have a fingerprint beside them."""
    if not isinstance(value, dict):
        return value
    drop = "configuration" in value and any(key in value for key in FINGERPRINT_KEYS)
    return {key: fingerprinted(item) for key, item in value.items()
            if not (drop and key == "configuration")}


def golden_view(doc: dict) -> dict:
    """The stored form of a plain document (see FINGERPRINT_KEYS)."""
    view = {key: value for key, value in doc.items() if key != "summary"}
    if "resolved" in view:
        view["resolved"] = fingerprinted(view["resolved"])
    return view


def encode_goldens(docs: dict) -> dict:
    """{name: plain document} (in registry order) -> {name: golden file document}."""
    return encode_documents({name: golden_view(doc) for name, doc in docs.items()})


def summarize(sections: list) -> dict:
    """Counts of op/impl/phase (entries) and op/impl/reason (fallbacks) over plain sections."""
    entries, fallbacks = {}, {}
    for section in sections:
        for line in section["entries"]:
            phase, _, op_impl = line.split(" ", 3)[:3]
            key = f"{op_impl}/{phase}"
            entries[key] = entries.get(key, 0) + 1
        for line in section["fallbacks"]:
            _, _, op_impl, reason = _split(line)[0].split(" ", 3)
            key = f"{op_impl}/{reason[1:]}"
            fallbacks[key] = fallbacks.get(key, 0) + 1
    return {"entries": dict(sorted(entries.items())), "fallbacks": dict(sorted(fallbacks.items()))}


def golden_spec(name: str) -> dict | None:
    """The compared spec: the registry entry without its README description."""
    spec = CONFIGS.get(name)
    return None if spec is None else {key: value for key, value in spec.items()
                                      if key != "covers"}


def plain_document(name: str, snapshot: dict, **fields) -> dict:
    """Plain (expanded) document for one recorder snapshot; ``info`` is never compared."""
    sections = plain_sections(snapshot["sections"])
    doc = {
        "schema": GOLDEN_SCHEMA, "config": name, "spec": golden_spec(name),
        "recorder": {"schema": snapshot["schema"], "aten": snapshot["aten"],
                     "invalid_params": snapshot["invalid_params"]},
        "sections": sections, "summary": summarize(sections),
        "info": {"policy_fingerprints": snapshot.get("policy_fingerprints")},
    }
    doc.update(fields)
    return doc


def dumps(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, indent=1) + "\n"


def load_documents(directory: Path, names=None) -> dict:
    """Golden-format files of ``directory`` -> plain documents (registry order first)."""
    stems = {path.stem for path in directory.glob("*.json")}
    order = [name for name in CONFIGS if name in stems] + sorted(stems - set(CONFIGS))
    docs = {name: json.loads((directory / f"{name}.json").read_text()) for name in order}
    plain = expand_documents(docs)
    return plain if names is None else {name: plain[name] for name in names if name in plain}


# ------------------------------------------------------------ comparison
# --check fails on a difference in what executed ("sections", "summary"), in
# the recording itself ("spec", "recorder", "outcome") or in the compiled
# program ("dynamo"). "resolved" and "observed" describe configuration
# resolution and runner behaviour: their differences are reported as notes
# (fatal with --strict), so e.g. a new default-off kernel option does not fail
# every golden. "resolved" is compared in its golden form: a configuration
# dump only through its fingerprint. A failed purity check (same layout and
# policy after warmup -> same selection) is always fatal.

NOTE_KEYS = frozenset(("resolved", "observed"))


def comparable(doc: dict, keys=None) -> dict:
    """The compared content of a plain document: everything except ``info``,
    with configuration dumps as their fingerprints (restricted to ``keys`` if given)."""
    return {key: fingerprinted(value) if key == "resolved" else value
            for key, value in doc.items() if key != "info" and (keys is None or key in keys)}


def recorded_only(name: str) -> bool:
    return False  # Every FastMD configuration is an enforced single-device guard.


def purity_failures(doc: dict) -> list:
    """Ranks (or ``""``) whose repeated warm evaluation selected differently."""
    purity = doc.get("purity")
    if not isinstance(purity, dict):
        return []
    if "identical" in purity:
        return [] if purity["identical"] else [""]
    return [rank for rank, value in sorted(purity.items())
            if isinstance(value, dict) and not value.get("identical", True)]


def _flat(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flat(value[key], f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def _aten_delta(before: str, after: str) -> str:
    old = dict(item.rsplit("=", 1) for item in before.split())
    new = dict(item.rsplit("=", 1) for item in after.split())
    return " ".join(f"{op}:{old.get(op, 0)}->{new.get(op, 0)}"
                    for op in sorted(set(old) | set(new)) if old.get(op) != new.get(op))


def _section_diff(label: str, old: dict, new: dict, limit: int) -> list:
    lines = []
    for key in _LINE_KEYS:
        if old[key] != new[key]:
            diff = list(difflib.unified_diff(old[key], new[key], lineterm="", n=1))[2:]
            lines.append(f"  section {label} {key} ({len(old[key])} -> {len(new[key])} lines):")
            lines.extend("    " + line for line in diff[:limit])
            if len(diff) > limit:
                lines.append(f"    ... {len(diff) - limit} more diff lines")
    if old["aten"] != new["aten"]:
        changed = [ctx for ctx in sorted(set(old["aten"]) | set(new["aten"]))
                   if old["aten"].get(ctx) != new["aten"].get(ctx)]
        lines.append(f"  section {label} aten: {len(changed)} contexts differ")
        for ctx in changed[:max(limit // 3, 1)]:
            lines.append(f"    {ctx}: "
                         + _aten_delta(old["aten"].get(ctx, ""), new["aten"].get(ctx, "")))
    return lines


def diff_documents(old: dict, new: dict, limit: int = 24, keys=None) -> list:
    """Readable first differences of two plain documents (empty if equal),
    restricted to the top-level ``keys`` if given."""
    old, new = comparable(old, keys), comparable(new, keys)
    if old == new:
        return []
    lines = []
    for key in sorted(set(old) | set(new)):
        if key == "sections" or old.get(key) == new.get(key):
            continue
        before, after = dict(_flat(old.get(key))), dict(_flat(new.get(key)))
        changed = [path for path in sorted(set(before) | set(after))
                   if before.get(path, "<absent>") != after.get(path, "<absent>")]
        lines.append(f"  {key}: {len(changed)} values differ")
        for path in changed[:limit]:
            lines.append(f"    {key}.{path}: {before.get(path, '<absent>')!r} -> "
                         f"{after.get(path, '<absent>')!r}")
    old_sections = {section["label"]: section for section in old.get("sections", [])}
    new_sections = {section["label"]: section for section in new.get("sections", [])}
    old_labels, new_labels = list(old_sections), list(new_sections)
    if old_labels != new_labels:
        lines.append("  section labels differ:")
        diff = list(difflib.unified_diff(old_labels, new_labels, lineterm="", n=1))[2:]
        lines.extend("    " + line for line in diff[:limit])
    for label in old_labels:
        if label in new_sections and old_sections[label] != new_sections[label]:
            lines.extend(_section_diff(label, old_sections[label], new_sections[label], limit))
            if len(lines) > 4 * limit:
                lines.append("  ... (further differences omitted)")
                break
    return lines or ["  documents differ (no readable difference found)"]


# ------------------------------------------------------------ orchestration

def _ppid(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return int(stream.read().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def ancestor_holds_lock(path: str) -> bool:
    """True if this process or an ancestor holds a flock(2) lock on ``path``."""
    try:
        stat = os.stat(path)
        with open("/proc/locks") as stream:
            locks = stream.read().splitlines()
    except OSError:
        return False
    device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    holders = set()
    for line in locks:
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "FLOCK" and fields[5] == device:
            holders.add(int(fields[4]))
    pid = os.getpid()
    while pid > 1:
        if pid in holders:
            return True
        pid = _ppid(pid)
    return False


@contextmanager
def _gpu_lock(path: str | None):
    if path is None:
        yield
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _visible_gpus() -> list:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [gpu for gpu in visible.split(",") if gpu.strip()]
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                 check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [str(index) for index, line in enumerate(listing.splitlines())
            if line.startswith("GPU ")]


def _child_env(gpus: list) -> dict:
    drop = ("MATRIS_", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
            "MASTER_PORT", "TORCHELASTIC_", "GROUP_RANK", "ROLE_")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(drop) or key in _CHILD_KEEPS}
    inherited = os.environ.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "tests")] + ([inherited] if inherited else []))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    return env


def _launch(name: str, gpus: list, work: Path) -> subprocess.Popen:
    spec = CONFIGS[name]
    output = work / f"{name}.json"
    child = [str(Path(__file__).resolve()), "--child", name, "--child-out", str(output)]
    command = [sys.executable, "-B", *child]
    log = open(work / "logs" / f"{name}.log", "w")
    process = subprocess.Popen(command, cwd=str(ROOT), env=_child_env(gpus), stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    process._op_selection_log = log
    process._op_selection_started = time.perf_counter()
    return process


def run_configs(names: list, work: Path, gpus: list, lock: str | None,
                timeout: float = 1200.0) -> dict:
    """Run each config in a fresh subprocess; returns {name: (ok, seconds, log)}."""
    (work / "logs").mkdir(parents=True, exist_ok=True)
    results = {}
    pending = list(names)
    for name in pending:
        if CONFIGS[name]["gpus"] > len(gpus):
            raise SystemExit(f"{name} needs {CONFIGS[name]['gpus']} GPUs; --gpus has {gpus}")
    while pending:
        wave, free = [], list(gpus)
        for name in list(pending):
            need = CONFIGS[name]["gpus"]
            if need <= len(free):
                wave.append((name, free[:need]))
                free = free[need:]
                pending.remove(name)
            if not free:
                break
        with _gpu_lock(lock):
            processes = [(name, _launch(name, devices, work)) for name, devices in wave]
            for name, process in processes:
                remaining = timeout - (time.perf_counter() - process._op_selection_started)
                try:
                    code = process.wait(timeout=max(remaining, 1.0))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, 9)
                    code = process.wait()
                    print(f"[op-selection] {name}: timed out after {timeout:.0f} s", flush=True)
                process._op_selection_log.close()
                seconds = time.perf_counter() - process._op_selection_started
                ok = code == 0
                ok = ok and (work / f"{name}.json").is_file()
                log = work / "logs" / f"{name}.log"
                results[name] = (ok, seconds, log)
                print(f"[op-selection] {name}: {'ok' if ok else f'FAILED (exit {code})'} "
                      f"{seconds:.1f} s gpus={','.join(dict(wave)[name])}", flush=True)
                if not ok:
                    tail = log.read_text(errors="replace").splitlines()[-15:]
                    print("\n".join("    | " + line for line in tail), flush=True)
    return results


def compare_documents(names: list, reference: dict, candidate: dict, limit: int = 24,
                      labels=("reference", "candidate"), strict: bool = False) -> list:
    """Names that fail the comparison (see NOTE_KEYS); prints the first differences."""
    failures = []
    for name in names:
        if name not in reference or name not in candidate:
            missing = [label for label, docs in zip(labels, (reference, candidate))
                       if name not in docs]
            print(f"[op-selection] {name}: MISSING in {', '.join(missing)}", flush=True)
            failures.append(name)
            continue
        old, new = reference[name], candidate[name]
        impure = purity_failures(new)
        if impure:
            print(f"[op-selection] {name}: PURITY FAILED ({labels[1]}) for "
                  f"{', '.join(rank or 'the process' for rank in impure)}: a repeated warm "
                  "evaluation selected differently", flush=True)
        keys = (set(old) | set(new)) - {"info"}
        if strict:
            fatal, notes = keys, set()
        elif recorded_only(name):
            fatal, notes = set(), keys
        else:
            fatal, notes = keys - NOTE_KEYS, keys & NOTE_KEYS
        lines = diff_documents(old, new, limit, fatal)
        if lines:
            print(f"[op-selection] {name}: DIFFERS ({labels[0]} -> {labels[1]})", flush=True)
            print("\n".join(lines), flush=True)
        note_lines = diff_documents(old, new, limit, notes)
        if note_lines:
            kind = "recorded only" if recorded_only(name) else "resolution/runner facts"
            print(f"[op-selection] {name}: NOTE, not fatal ({kind}; {labels[0]} -> {labels[1]})",
                  flush=True)
            print("\n".join(note_lines), flush=True)
        if impure or lines:
            failures.append(name)
    return failures


def write_goldens(regenerated: dict) -> list:
    """Re-encode the whole golden set with ``regenerated`` replacing its configs.

    Configs not regenerated keep their expanded golden content, so references
    are recomputed without changing any other golden's meaning. Goldens of
    configs no longer registered are removed. Returns the written names.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    current = load_documents(GOLDEN_DIR)
    plain = {name: regenerated.get(name, current.get(name)) for name in CONFIGS
             if name in regenerated or name in current}
    encoded = encode_goldens(plain)
    for name, doc in encoded.items():
        path = GOLDEN_DIR / f"{name}.json"
        text = dumps(doc)
        if not path.is_file() or path.read_text() != text:
            path.write_text(text)
    for stale in sorted(set(current) - set(CONFIGS)):
        (GOLDEN_DIR / f"{stale}.json").unlink()
        print(f"[op-selection] removed stale golden {stale}.json", flush=True)
    return list(encoded)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--all", action="store_true", help="every registered config")
    select.add_argument("--config", nargs="+", metavar="NAME", help="these configs")
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--write", action="store_true",
                        help=f"write regenerated snapshots to {GOLDEN_DIR}")
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare with the goldens (or --against)")
    parser.add_argument("--against", type=Path, default=GOLDEN_DIR,
                        help="reference directory for --check (default: the goldens)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("REFERENCE", "CANDIDATE"),
                        help="compare two snapshot directories without regenerating")
    parser.add_argument("--out-dir", type=Path,
                        help="keep regenerated snapshots and child logs here")
    parser.add_argument("--from", dest="from_dir", type=Path,
                        help="use snapshots regenerated earlier into this --out-dir instead "
                             "of running the configs again")
    parser.add_argument("--gpus", help="comma-separated GPU ids for the waves "
                                       "(default: CUDA_VISIBLE_DEVICES or all GPUs)")
    parser.add_argument("--gpu-lock", default="auto",
                        help=f"'auto' (default: {DEFAULT_LOCK} unless an ancestor holds it), "
                             "'none', or a lock file path")
    parser.add_argument("--timeout", type=float, default=1200.0, help="seconds per config")
    parser.add_argument("--diff-lines", type=int, default=24,
                        help="diff lines shown per section")
    parser.add_argument("--strict", action="store_true",
                        help="--check: also fail on resolved/observed differences")
    parser.add_argument("--write-fixtures", action="store_true",
                        help="regenerate the small structure fixtures from rdx_1344")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-out", type=Path, help=argparse.SUPPRESS)
    return parser, parser.parse_args(argv)


def main(argv=None) -> int:
    parser, args = _parse_args(argv)
    if args.child:
        return _child_main(args.child, args.child_out)
    if args.write_fixtures:
        if not test_assets_available(RDX_1344):
            parser.error(f"--write-fixtures needs {RDX_1344} in MATRIS_TEST_ASSETS")
        write_fixtures()
        return 0
    if args.list:
        for name, spec in CONFIGS.items():
            print(f"{name:44s} {spec['covers']}")
        print(f"{len(CONFIGS)} configs")
        return 0
    if args.compare:
        reference, candidate = (load_documents(path) for path in args.compare)
        names = args.config or list(reference)
        # Two runs of the same tree must agree on everything (determinism).
        failures = compare_documents(names, reference, candidate, args.diff_lines,
                                     tuple(str(path) for path in args.compare), strict=True)
        print(f"[op-selection] compared {len(names)}: "
              f"{'identical' if not failures else f'{len(failures)} differ: {failures}'}")
        return 1 if failures else 0
    if not (args.all or args.config):
        parser.error("choose --all or --config NAME ...")
    if not (args.write or args.check or args.out_dir):
        parser.error("choose --write, --check and/or --out-dir")
    names = list(CONFIGS) if args.all else args.config
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        parser.error(f"unknown configs {unknown}; see --list")
    skipped = [name for name in names if not assets_available(name)]
    if skipped:
        print(f"[op-selection] SKIPPED {len(skipped)} of {len(names)} configs: they need "
              "files missing from MATRIS_TEST_ASSETS "
              f"({os.environ.get('MATRIS_TEST_ASSETS') or 'unset'}); see "
              "tests/golden/op_selection/README.md", flush=True)
        names = [name for name in names if name not in skipped]
    started = time.perf_counter()
    if args.from_dir is not None:
        work = args.from_dir
        results = {name: ((work / f"{name}.json").is_file(), 0.0, None) for name in names}
    else:
        gpus = args.gpus.split(",") if args.gpus else _visible_gpus()
        if not gpus:
            parser.error("no GPU found; these snapshots need CUDA")
        if args.gpu_lock == "none":
            lock = None
        else:
            path = DEFAULT_LOCK if args.gpu_lock == "auto" else args.gpu_lock
            lock = None if ancestor_holds_lock(path) else path
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            clashes = [name for name in names if (args.out_dir / f"{name}.json").exists()]
            if clashes:
                parser.error(f"--out-dir already contains {clashes[:3]}...; use a new directory")
            work = args.out_dir
        else:
            work = Path(tempfile.mkdtemp(prefix="op_selection_"))
        print(f"[op-selection] {len(names)} configs, gpus={gpus}, lock={lock or 'none'}, "
              f"out={work}", flush=True)
        results = run_configs(names, work, gpus, lock, args.timeout)
    wall = time.perf_counter() - started
    failed = [name for name, (ok, _, _) in results.items() if not ok]
    status = 0
    if failed:
        print(f"[op-selection] {len(failed)} configs failed to run: {failed}", flush=True)
        status = 1
    done = [name for name in names if name not in failed]
    regenerated = load_documents(work, done)
    if args.check:
        differ = compare_documents(done, load_documents(args.against, done), regenerated,
                                   args.diff_lines, (str(args.against), str(work)),
                                   strict=args.strict)
        print(f"[op-selection] check: {len(done) - len(differ)}/{len(names)} pass"
              + (" (strict)" if args.strict else "")
              + (f" ({len(skipped)} skipped: assets missing)" if skipped else "")
              + (f"; fail: {differ}" if differ else ""), flush=True)
        status = status or (1 if differ else 0)
    if args.write:
        if failed:
            print("[op-selection] not writing goldens: some configs failed", flush=True)
        else:
            written = write_goldens(regenerated)
            total = sum((GOLDEN_DIR / f"{name}.json").stat().st_size for name in written)
            print(f"[op-selection] wrote {len(done)} regenerated goldens ({len(written)} "
                  f"re-encoded, {total / 1e6:.2f} MB) to {GOLDEN_DIR}", flush=True)
    print(f"[op-selection] wall {wall:.1f} s", flush=True)
    if args.out_dir is None and args.from_dir is None and status == 0:
        shutil.rmtree(work, ignore_errors=True)
    elif args.out_dir is None and args.from_dir is None:
        print(f"[op-selection] kept snapshots and logs in {work}", flush=True)
    return status


# ------------------------------------------------------------ child side
# Everything below runs in the per-config subprocess and imports torch lazily.

class Recording:
    """One recorder for a whole config; ``phase`` opens a section.

    With ``split_forwards`` a forward pre-hook on each root model opens a
    sub-section per top-level model call (``<phase>.eval<k>`` or
    ``<phase>.capture<k>`` while a CUDA graph is being captured), so warmup
    and captured evaluations are separate and identical ones deduplicate.
    """

    def __init__(self, roots=(), *, aten=True, split_forwards=True):
        from fastmd.engine.recorder import OpPlanRecorder

        roots = [root for root in roots if root is not None]
        self.recorder = OpPlanRecorder(aten=aten, root=roots or None)
        self._roots = roots if split_forwards else []
        self._handles = []
        self._phase = "default"
        self._forwards = 0

    def __enter__(self):
        self.recorder.__enter__()
        try:
            for root in self._roots:
                self._handles.append(root.register_forward_pre_hook(self._forward))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return self.recorder.__exit__(*exc)

    def phase(self, label: str) -> None:
        self._phase, self._forwards = label, 0
        self.recorder.mark(label)

    def _forward(self, module, args):
        self.split()

    def split(self) -> None:
        """Open the next per-evaluation sub-section of the current phase."""
        import torch

        self._forwards += 1
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        self.recorder.mark(f"{self._phase}.{'capture' if capturing else 'eval'}{self._forwards}")

    def snapshot(self) -> dict:
        return self.recorder.snapshot()


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _inference_config(spec: dict):
    from fastmd.config import CapacityConfig, InferenceConfig

    capacity = CapacityConfig(candidate_skin=spec["candidate_skin"]) \
        if "candidate_skin" in spec else CapacityConfig()
    return InferenceConfig(execution=spec.get("execution"), optimization_profile=spec["profile"],
                           checkpoint=spec["checkpoint"], isolated_atoms="reference",
                           capacity=capacity, report=False, matmul_precision="highest")


def _equal_width_model(num_layers: int | None = None):
    import torch
    from fastmd.models.matris.model import MatRIS

    architecture = dict(RDX_ARCHITECTURE, edge_feat_dim=128, three_body_feat_dim=128)
    if num_layers is not None:
        architecture["num_layers"] = num_layers
    torch.manual_seed(EQUAL_WIDTH_SEED)
    return MatRIS(**architecture, enable_compile=False, enable_checkpoint=False)


def _model(spec: dict):
    """G-S1 benchmark model loading: eval().requires_grad_(False) on cuda:0."""
    from fastmd.models.matris.model import MatRIS

    if spec["model"] == "rdx":
        model = MatRIS.load(model_path=str(RDX_MODEL), device="cuda",
                            enable_compile=bool(spec.get("compile", False)),
                            enable_checkpoint=False)
    elif spec["model"] == "equal_width":
        model = _equal_width_model().to("cuda")
    elif spec["model"] == "equal_width_1layer":
        model = _equal_width_model(num_layers=1).to("cuda")
    else:
        raise ValueError(spec["model"])
    return model.eval().requires_grad_(False)


def cluster_atoms():
    """Atoms of RDX-1344 within 5 A of the cell centre, in a 30 A vacuum box."""
    from ase import Atoms
    from ase.io import read
    import numpy as np

    atoms = read(str(STRUCTURE_FILES["rdx_1344"]))
    centre = atoms.cell.array.sum(axis=0) / 2
    keep = np.linalg.norm(atoms.positions - centre, axis=1) < 5.0
    positions = atoms.positions[keep] - centre + 15.0
    return Atoms(numbers=atoms.numbers[keep], positions=positions, cell=np.eye(3) * 30.0,
                 pbc=True)


def write_fixtures() -> None:
    from ase.io import write

    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    atoms = cluster_atoms()
    write(str(STRUCTURE_FILES["rdx_cluster"]), atoms, format="extxyz")
    print(f"wrote {STRUCTURE_FILES['rdx_cluster']} ({len(atoms)} atoms)")


def _structure(key: str):
    from ase import Atoms
    from ase.io import read
    import numpy as np

    if key in STRUCTURE_FILES:
        return read(str(STRUCTURE_FILES[key]))
    if key == "isolated":
        atoms = read(str(STRUCTURE_FILES["rdx_cluster"]))
        # >= 11 A from the cluster (x, y, z in [10, 20]) and 7 A between
        # the two isolated atoms' periodic images: no edge within 6 A.
        return atoms + Atoms("HN", positions=[[2.0, 2.0, 2.0], [2.0, 2.0, 25.0]])
    if key == "dimers":
        pairs = (("HH", 0.74), ("NN", 1.10), ("OO", 1.21), ("CO", 1.13),
                 ("CH", 1.09), ("NH", 1.01), ("OH", 0.97), ("CN", 1.16))
        symbols, positions = "", []
        for index, (pair, length) in enumerate(pairs):
            corner = 4.0 + 16.0 * np.array([(index >> 2) & 1, (index >> 1) & 1, index & 1])
            symbols += pair
            positions += [corner, corner + [length, 0.0, 0.0]]
        return Atoms(symbols, positions=positions, cell=np.eye(3) * 32.0, pbc=True)
    raise ValueError(key)


def _with_momenta(atoms):
    from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta

    atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), TEMPERATURE_K,
                                                seed=MOMENTA_SEED))
    return atoms


def _summary(instance) -> dict:
    """The compared part of ``config_summary()`` (no versions or environment)."""
    summary = instance.config_summary()
    effective = dict(summary["effective"])
    matmul = effective.pop("matmul_precision")
    effective["matmul"] = {"fp32_gemm": matmul["fp32_gemm"],
                           "float32_matmul_precision": matmul["float32_matmul_precision"]}
    auto = effective.get("auto_execution")
    if auto is not None:
        # execution="auto": the modes; the reasons' budgets vary with the device state.
        effective["auto_execution"] = {"mode": auto["mode"],
                                       "decisions": [d["mode"] for d in auto.get("decisions", [])]}
    return {"requested": summary["requested"], "configuration": summary["configuration"],
            "fingerprint": summary["fingerprint"], "effective": effective}


def _resolved_only(resolved) -> dict:
    return {"configuration": resolved.as_dict(), "fingerprint": resolved.fingerprint}


def _graph_sizes(graph) -> dict:
    return {"atoms": int(graph.atomic_number.shape[0]),
            "directed_edges": int(graph.atom_graph.shape[0]),
            "triplets": int(graph.line_graph.shape[0])}


def _int_stats(stats: dict | None, keys) -> dict | None:
    if stats is None:
        return None
    return {key: stats[key] for key in keys if key in stats}


def _dynamo_text(text) -> str:
    """A dynamo reason without hints, links, addresses or directories."""
    lines = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith(("Hint:", "For more details")):
            continue
        line = re.sub(r" at 0x[0-9a-fA-F]+", "", line)
        lines.append(re.sub(r"/(?:[^\s/`'\"]+/)+", "", line))
    return " | ".join(lines)


def _merged(counter, key) -> dict:
    merged = {}
    for name, count in counter.items():
        name = key(name)
        merged[name] = merged.get(name, 0) + count
    return dict(sorted(merged.items()))


def dynamo_counters() -> dict:
    """Compared ``torch._dynamo`` counters of a compiled program (design §4.3).

    Graph breaks and unimplemented reasons with counts, graphs and calls
    captured, frames, recompiles (guard failures per function), resume
    functions (graph-break continuations, without line numbers) and AOT
    compilations. Inductor and cache counters depend on on-disk caches.
    """
    from torch._dynamo.utils import counters, guard_failures

    return {
        "stats": {key: counters["stats"][key] for key in ("calls_captured", "unique_graphs")},
        "frames": dict(sorted(counters["frames"].items())),
        "graph_breaks": _merged(counters["graph_break"], _dynamo_text),
        "unimplemented": _merged(counters["unimplemented"], _dynamo_text),
        "resumes": _merged(counters["resumes"], lambda name: re.sub(r"_at_\d+\Z", "", name)),
        "recompiles": _merged({code: len(reasons) for code, reasons in guard_failures.items()},
                              lambda code: re.sub(r"_at_\d+\Z", "",
                                                  getattr(code, "co_name", str(code)))),
        "aot_autograd": {key: counters["aot_autograd"][key] for key in ("ok", "total")},
    }


def _run_gpumd(name: str, spec: dict) -> dict:
    from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics

    atoms = _structure(spec["structure"])
    whole_step = spec["kind"] == "whole_step"
    if whole_step:
        _with_momenta(atoms)
        config = _inference_config(spec if spec.get("entry_defaults") else dict(spec, execution="whole_step"))
        ensemble = spec["ensemble"]
    else:
        config = _inference_config(spec)
        ensemble = "nve"
    model = _model(spec)
    outcome = "ok"
    md = None
    fields = {}

    def construct():
        return GPUResidentMolecularDynamics(
            atoms, model=model, task="ef", device="cuda", ensemble=ensemble,
            temperature=TEMPERATURE_K, timestep=MD_TIMESTEP_FS,
            tdamp=NHC_TDAMP_FS if ensemble == "nvt_nhc" else None, config=config)

    phases = ("cold", "warm")
    compiled = spec.get("compile", False)
    if compiled:
        # enable_compile: construct and compile unrecorded, keep dynamo's
        # counters of that recorder-free program, then record two more
        # evaluations. Recording re-traces the compiled frames with a graph
        # break at every record, so aten counts would describe that program
        # (aten=False); the recorder drops those frames when it exits.
        md = construct()
        frame = md.state.positions.clone()
        for _ in range(COMPILE_WARMUP):
            md.evaluate(frame)
            _sync()
        fields["dynamo"] = dynamo_counters()
        phases = ("warm", "warm2")
    with Recording([model], aten=not compiled) as recording:
        try:
            if md is None:
                recording.phase("construct")
                md = construct()
                _sync()
                frame = md.state.positions.clone()
            for label in phases:
                recording.phase(label)
                if whole_step:
                    md.step()
                else:
                    md.evaluate(frame)
                _sync()
        except Exception as exc:  # noqa: BLE001 - snapshot an accepted rejection
            if not spec.get("may_reject"):
                raise
            outcome = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"}
    observed = {}
    resolved = None
    if md is not None:
        resolved = _summary(md)
        if whole_step:
            observed["whole_step"] = _int_stats(md.whole_step_stats(), (
                "launches", "candidate_refreshes", "fallback_windows", "recovery_replays",
                "u_capacity", "t_capacity", "capacity_recaptures", "steps_per_launch"))
        else:
            observed["model_graph"] = _int_stats(md.model_graph_stats(), (
                "captures", "cache_fit_reuses", "cached_buckets", "enable_model_fusions"))
            observed["graph"] = _graph_sizes(md.build_graph(frame))
    return plain_document(name, recording.snapshot(), resolved=resolved,
                          observed=observed, outcome=outcome, **fields)


def _run_calculator(name: str, spec: dict) -> dict:
    from ase.calculators.calculator import all_changes
    from fastmd.interfaces.calculator import MLIPCalculator

    # G-S1 benchmark "calculator": an ASE MD driver (MLIPCalculator) path
    # loads the checkpoint by path. Construction runs no model call.
    calc = MLIPCalculator(model_path=str(RDX_MODEL), task="ef", device="cuda",
                            config=_inference_config(spec))
    atoms = _structure(spec["structure"])
    with Recording([calc.model]) as recording:
        for label in ("cold", "warm"):
            recording.phase(label)
            calc.calculate(atoms, ["energy", "forces"], all_changes)
            _sync()
    # The runner of the path the calls took (topology_reuse="auto": the skin
    # runner; its eager mode captures nothing).
    runner = calc._skin if calc.topology_reuse_summary()["last_call"] == "skin" else calc._graph_runner
    observed = {"model_graph": None if runner is None else _int_stats(
        runner.stats(), ("captures", "cache_fit_reuses", "cached_buckets",
                         "enable_model_fusions")),
        "topology_reuse": calc.topology_reuse_summary()["last_call"]}
    return plain_document(name, recording.snapshot(), resolved=_summary(calc),
                          observed=observed, outcome="ok")


def _run_replicas(name: str, spec: dict) -> dict:
    import gc

    import torch
    from fastmd.engine.capacity_policy import JointCapacityPolicy
    from fastmd.execution.gpu_integrator import GPUIntegrator
    from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
    from fastmd.execution.whole_step_cuda_graph import (
        BatchedWholeStepGraphRunner, FixedCapacityOverflow,
    )
    from fastmd.config import resolve_config
    from fastmd.graph import FixedCapacityGraphBuilder, TensorGraphBuilder

    # G-S1 benchmark "replicas" construction with spec["replicas"] replicas.
    atoms = _structure(spec["structure"])
    replicas = int(spec["replicas"])
    device = torch.device("cuda", torch.cuda.current_device())
    resolved = resolve_config(_inference_config(dict(spec, execution="whole_step")),
                              default_execution="whole_step")
    model = _model(spec)
    capacity = resolved.capacity
    cell = torch.as_tensor(atoms.get_cell().array, dtype=torch.float32, device=device)
    numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=device)
    pbc = torch.as_tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    masses = torch.as_tensor(atoms.get_masses(), dtype=torch.float32, device=device)
    positions = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    cutoff = float(model.graph_converter.atom_graph_cutoff)
    line_cutoff = float(model.graph_converter.line_graph_cutoff)
    composition = atoms.get_chemical_formula()
    with resolved.scope():
        sample = TensorGraphBuilder(cell=cell, atomic_numbers=numbers, pbc=pbc,
                                    atom_graph_cutoff=cutoff, line_graph_cutoff=line_cutoff,
                                    device=device, composition=composition,
                                    check_isolated_atoms=False).build(positions)
        policy = JointCapacityPolicy(u_step=capacity.u_step, t_step=capacity.t_step,
                                     min_pad_u=capacity.min_pad_u, min_pad_t=capacity.min_pad_t)
        key = policy.start(sample.undirected2directed.shape[0], sample.line_graph.shape[0])
        del sample
    momenta = torch.stack([
        torch.as_tensor(maxwell_boltzmann_momenta(atoms.get_masses(), TEMPERATURE_K,
                                                  seed=MOMENTA_SEED + replica),
                        dtype=torch.float32, device=device)
        for replica in range(replicas)])
    integrator = GPUIntegrator(masses, dt_fs=MD_TIMESTEP_FS, ensemble="nvt",
                               temperature=TEMPERATURE_K, taut_fs=BERENDSEN_TAUT_FS,
                               fix_com=True, degrees_of_freedom=len(atoms) * 3)
    history = [[key.u_capacity, key.t_capacity]]
    with Recording([model]) as recording:
        attempt = 0
        while True:
            attempt += 1
            recording.phase("construct" if attempt == 1 else f"construct_retry{attempt}")
            builders = [FixedCapacityGraphBuilder(
                cell=cell, atomic_numbers=numbers, pbc=pbc, u_capacity=key.u_capacity,
                t_capacity=key.t_capacity, atom_graph_cutoff=cutoff, line_graph_cutoff=line_cutoff,
                candidate_cutoff=cutoff + capacity.candidate_skin, device=device,
                composition=composition, n_dummy=capacity.n_dummy) for _ in range(replicas)]
            try:
                runner = BatchedWholeStepGraphRunner(
                    model=model, builders=builders, integrator=integrator,
                    positions=positions.unsqueeze(0).expand(replicas, -1, -1).contiguous(),
                    momenta=momenta, task="ef", steps_per_launch=1, config=resolved)
                break
            except FixedCapacityOverflow as overflow:
                grown = policy.grow(key, (overflow.edge_count + 1) // 2, overflow.triplet_count)
                if grown == key:
                    raise
                key = grown
                history.append([key.u_capacity, key.t_capacity])
                del builders
                gc.collect()
        _sync()
        for label in ("cold", "warm"):
            recording.phase(label)
            runner.replay()
            _sync()
    observed = {"replicas": _int_stats(runner.stats(), (
        "launches", "candidate_refreshes", "fallback_windows", "recovery_replays",
        "batch_size", "u_capacity", "t_capacity", "steps_per_launch")),
        "capacity_history": history}
    return plain_document(name, recording.snapshot(), resolved=_resolved_only(resolved),
                          observed=observed, outcome="ok")


def synthetic_graph(num_triplets: int, num_edges: int | None = None, device="cuda"):
    """A valid periodic-helix RadiusGraph with exactly the requested row counts.

    Atom k sits on a helix (radius 1.5 A, 7 atoms per turn, rise 0.3 A) in a
    cell periodic along z; bond (a, a+o mod N) for offsets o = 1..4 has
    length <= 3.9 A (< both cutoffs), so every directed edge is "short".
    Each undirected bond has exactly two directed rows (reverse images),
    edges are sorted by centre atom and triplets are the builder's ordered
    pairs of a centre's edges, owner-sorted, truncated to ``num_triplets``.
    """
    import numpy as np
    import torch
    from fastmd.graph import RadiusGraph
    from fastmd.graph.matris_topology import matris_builder_evidence

    if num_edges is None:
        atoms = max(16, math.ceil(num_triplets / 56))
        bonds = 4 * atoms
    else:
        if num_edges % 2:
            raise ValueError("directed edges come in pairs")
        bonds = num_edges // 2
        atoms = max(16, math.ceil(bonds / 4))
    rise, radius, turn = 0.3, 1.5, 2 * math.pi / 7
    length = atoms * rise
    index = np.arange(atoms)
    cart = np.stack([radius * np.cos(turn * index), radius * np.sin(turn * index),
                     rise * index], axis=1)
    cell = np.diag([20.0, 20.0, length])
    rows = []  # (centre, neighbour, image_z, bond)
    for bond in range(bonds):
        a, offset = bond % atoms, 1 + bond // atoms
        b, image = (a + offset) % atoms, (a + offset) // atoms
        rows.append((a, b, image, bond))
        rows.append((b, a, -image, bond))
    rows.sort(key=lambda row: (row[0], row[3], row[1]))
    centre = np.array([row[0] for row in rows])
    d2u = np.array([row[3] for row in rows])
    u2d = np.zeros(bonds, dtype=np.int64)
    for position in range(len(rows) - 1, -1, -1):
        u2d[d2u[position]] = position
    triplets = []
    start = 0
    for atom in range(atoms):
        stop = start
        while stop < len(rows) and centre[stop] == atom:
            stop += 1
        for first in range(start, stop):
            for second in range(start, stop):
                if first != second:
                    triplets.append((atom, d2u[first], first, d2u[second], second))
        start = stop
        if len(triplets) >= num_triplets:
            break
    if len(triplets) < num_triplets:
        raise ValueError(f"only {len(triplets)} triplets available")
    triplets = triplets[:num_triplets]
    numbers = np.array([1, 6, 7, 8])[index % 4]

    def tensor(value, dtype):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

    return RadiusGraph(
        graph_id=None, mp_id=None, composition="synthetic",
        atomic_number=tensor(numbers, torch.int32),
        atom_frac_coord=tensor(cart @ np.linalg.inv(cell), torch.float32),
        lattice=tensor(cell, torch.float32),
        neighbor_image=tensor([[0, 0, row[2]] for row in rows], torch.float32),
        atom_graph=tensor([[row[0], row[1]] for row in rows], torch.int32),
        atom_graph_cutoff=6.0, line_graph=tensor(triplets, torch.int32).reshape(-1, 5),
        line_graph_cutoff=4.0, directed2undirected=tensor(d2u, torch.int32),
        undirected2directed=tensor(u2d, torch.int32), atom_target_sorted=True,
        line_atom_sorted=True, isolated_atom_count=tensor([0], torch.int64),
        topology_evidence=matris_builder_evidence(
            "tests.op_selection.synthetic_helix", atom_target_sorted=True,
            line_owner_sorted=True),
    )


def _run_synthetic(name: str, spec: dict) -> dict:
    from fastmd.config import resolve_config
    from fastmd.engine.policy import fusion_scope

    model = _model(spec)
    block = model.interaction_block[0]
    # A single block is also the last one; keep its line-refinement tail so
    # that every stage of the block runs at the requested row counts.
    block.last_block = False
    for layer in model.interaction_block:
        layer.enable_checkpoint = {"off": False, "on": True, "auto": None}[spec["checkpoint"]]
    graph = synthetic_graph(spec["triplets"], spec.get("edges"))
    resolved = resolve_config(_inference_config(dict(spec, execution="eager")))
    with Recording([model]) as recording, resolved.scope(), fusion_scope(spec["fusion_scope"]):
        for label in ("cold", "warm"):
            recording.phase(label)
            model([graph], task="ef")
            _sync()
    return plain_document(name, recording.snapshot(), resolved=_resolved_only(resolved),
                          observed={"graph": _graph_sizes(graph)}, outcome="ok")


def _structured(value):
    """Drop free-text descriptions (strings with spaces) from evaluator reports."""
    if isinstance(value, dict):
        return {key: _structured(item) for key, item in sorted(value.items())
                if not (isinstance(item, str) and " " in item)}
    return value


def purity(sections: list, first: str, second: str) -> dict:
    """Do the sections of phases ``first`` and ``second`` record the same plan?"""
    def phase(label):
        return [{key: section[key] for key in ("entries", "fallbacks", "aten")}
                for section in sections
                if section["label"] == label or section["label"].startswith(label + ".")]

    return {"compared": [first, second], "identical": phase(first) == phase(second)}


_RUNNERS = {"gpumd": _run_gpumd, "whole_step": _run_gpumd, "calculator": _run_calculator,
            "replicas": _run_replicas, "synthetic": _run_synthetic}


def _versions() -> dict:
    import platform

    import torch

    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {"torch": torch.__version__, "triton": triton_version, "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def _child_main(name: str, out: Path) -> int:
    import warnings

    started = time.perf_counter()
    spec = CONFIGS[name]
    import torch
    import fastmd
    from fastmd.config import ConfigurationWarning

    if Path(fastmd.__file__).resolve().parents[1] != ROOT:
        raise RuntimeError(f"imported {fastmd.__file__}, expected the checkout {ROOT}")
    leaked = sorted(key for key in os.environ
                    if key.startswith("MATRIS_") and key not in _CHILD_KEEPS)
    if leaked:
        raise RuntimeError(f"MATRIS_* environment must be empty, found {leaked}")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        doc = _RUNNERS[spec["kind"]](name, spec)
    configuration_warnings = sorted({
        f"{warning.category.__name__}: {str(warning.message).splitlines()[0]}"
        for warning in caught if issubclass(warning.category, ConfigurationWarning)})
    info = dict(doc.get("info") or {})
    info.update(_versions())
    info.update(configuration_warnings=configuration_warnings,
                seconds=round(time.perf_counter() - started, 1))
    doc["info"] = info
    doc.pop("rank", None)
    text = dumps(encode_documents({name: doc})[name])
    with out.open("x") as stream:
        stream.write(text)
    print(f"[op-selection child] {name}: {len(doc['sections'])} sections, {len(text)} bytes, "
          f"{info['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
