"""Offline single-GPU adapter validation lanes, ported from MD/tools/ci.

Reuses an existing environment: it never installs packages or downloads weights.
Tier 1 requires every adapter test to run and pass; tier 2 uses the upstream MD
benchmark presets and the included multi-model benchmark entry point.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("mace", "sevennet", "orb", "chgnet")
MARKER = "FASTMD_LANE_CHECKPOINTS="


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def check_environment(family, requirements, tier, checkpoint, check_only):
    """Run under the selected interpreter; check pins without changing it."""
    failures = []
    checked_pins = 0
    print(f"  requirements: {requirements}; sha256={sha256(requirements)}")
    for line in Path(requirements).read_text().splitlines():
        spec = line.split("#", 1)[0].strip()
        if not spec:
            continue
        requirement = spec.split()[0]
        if requirement.count("==") != 1:
            raise ValueError(f"requirements must contain exact-version pins: {requirement!r}")
        name, expected = requirement.split("==")
        checked_pins += 1
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            failures.append(f"{name}: expected {expected}, not installed")
            continue
        origin = dist.read_text("direct_url.json")
        print(f"  {name} {dist.version}: {dist.locate_file('')}" + (f"; {origin.strip()}" if origin else ""))
        if dist.version != expected:
            failures.append(f"{name}: expected {expected}, found {dist.version}")
    if not checked_pins:
        raise ValueError("requirements file has no exact-version pins")
    if failures:
        raise RuntimeError("environment does not match the requirements pins:\n  " + "\n  ".join(failures))
    import torch
    import fastmd

    if Path(fastmd.__file__).resolve().parent != ROOT / "fastmd":
        raise RuntimeError(f"fastmd imports from {fastmd.__file__}, not {ROOT}")
    print(f"  torch {torch.__version__}: {torch.__file__}; fastmd: {fastmd.__file__}")
    # Import the runtime builders and the optional family before expensive tests.
    importlib.import_module("fastmd.graph.gpu_graph_builder")
    upstream_module = {"mace": "mace.calculators", "sevennet": "sevenn.calculator",
                       "orb": "orb_models.forcefield.pretrained", "chgnet": "chgnet.model"}[family]
    importlib.import_module(upstream_module)
    if not check_only and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("the lane requires exactly one visible CUDA device")
    paths = {}
    if family == "mace":
        from mace.calculators.foundations_models import mace_mp_urls
        from mace.tools.utils import get_cache_dir
        from fastmd.models.mace import MACE_CHECKPOINTS
        names = ("mace_mp_0", "mace_mpa_0") if "2" in tier else ("mace_mp_0",)
        for name in names:
            upstream, pinned = MACE_CHECKPOINTS[name]
            filename = "".join(c for c in os.path.basename(mace_mp_urls[upstream]) if c.isalnum() or c == "_")
            paths[name] = (str(Path(get_cache_dir()) / filename), pinned)
    elif family == "sevennet":
        from sevenn.util import pretrained_name_to_path
        from fastmd.models.sevennet import SEVENNET_CHECKPOINTS
        upstream, pinned = SEVENNET_CHECKPOINTS["sevennet_0"]
        paths["sevennet_0"] = (str(pretrained_name_to_path(upstream)), pinned)
    elif family == "orb":
        from fastmd.models.orb import ORB_CHECKPOINTS
        _, url, pinned = ORB_CHECKPOINTS["orb_v3"]
        if not checkpoint:
            from cached_path import find_latest_cached
            checkpoint = find_latest_cached(url)
        if not checkpoint:
            raise RuntimeError("ORB checkpoint is not cached; provide --checkpoint FILE (no downloads)")
        paths["orb_v3"] = (str(checkpoint), pinned)
    else:
        import chgnet
        import chgnet.graph.cygraph  # ensure the compiled converter imports
        from fastmd.models.chgnet import CHGNET_CHECKPOINTS
        _, relative, pinned = CHGNET_CHECKPOINTS["chgnet_0_3_0"]
        paths["chgnet_0_3_0"] = (str(Path(chgnet.__file__).parent / relative), pinned)
    for name, (path, expected) in paths.items():
        if not Path(path).is_file():
            raise RuntimeError(f"{name}: checkpoint missing at {path}; provision it first (no downloads)")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name}: checkpoint sha256 {actual} differs from pinned {expected}: {path}")
        print(f"  {name}: {path}; sha256={actual}")
    print(MARKER + json.dumps({name: str(Path(path).resolve()) for name, (path, _) in paths.items()}), flush=True)


def run_tests(family):
    """Preserve upstream's strict no-skip / at-least-one-pass gate."""
    class Result(unittest.TextTestResult):
        passed = 0

        def addSuccess(self, test):
            super().addSuccess(test)
            self.passed += 1

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests" / "adapters"),
        pattern=f"test_{family}_adapter.py", top_level_dir=str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    for reason, count in collections.Counter(reason for _, reason in result.skipped).most_common():
        print(f"skipped {count}: {reason}")
    print(f"tier 1: {result.passed} passed, {len(result.skipped)} skipped, "
          f"{len(result.failures)} failed, {len(result.errors)} errors", flush=True)
    return 0 if result.wasSuccessful() and result.passed and not result.skipped else 1


def benchmark_cases(family):
    """The original single-card preset matrix; each case has its own process."""
    large = ["--precapture-steps", "500", "--warmup", "200"]
    small = ["--temperature", "500", "--timestep", "0.25", "--steps-per-launch", "10",
             "--precapture-steps", "2000", "--warmup", "300"]
    if family == "mace":
        return [("mace_mp_0", "float32", "large", "fp32_mace_mp_0", large)] + [
            (model, "float64", "small", "fp64_" + model, small + ["--capacity-step", "32"])
            for model in ("mace_mp_0", "mace_mpa_0")]
    model = {"sevennet": "sevennet_0", "orb": "orb_v3", "chgnet": "chgnet_0_3_0"}[family]
    if family != "chgnet":
        small += ["--capacity-step", "16" if family == "sevennet" else "32"]
    names = (model + "_large", model + "_small") if family == "sevennet" else ("large_" + model, "small_" + model)
    return [(model, "float32", "large", names[0], large), (model, "float32", "small", names[1], small)]


def validate_harness_args(extra):
    """Allow only complete public tuning options, never launcher internals.

    The benchmark itself accepts argparse abbreviations. Validating the full
    token stream here also prevents abbreviations of --out/--checkpoint and its
    internal --child path from bypassing the lane's preflight checks.
    """
    class TuningParser(argparse.ArgumentParser):
        def error(self, message):
            raise ValueError(message)

    tuning = TuningParser(add_help=False, allow_abbrev=False)
    for option in ("profile", "kernel", "matmul-precision", "frame", "ensemble", "temperature", "timestep",
                   "steps", "warmup", "capacity-step", "capacity-pad", "triplet-capacity-step",
                   "triplet-capacity-pad", "precapture-steps", "steps-per-launch", "seed", "timeout"):
        tuning.add_argument("--" + option)
    tuning.add_argument("--mode", nargs="+")
    tuning.parse_args(extra)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--python", help="existing interpreter (default: $PYTHON, else python3)")
    parser.add_argument("--venv", type=Path, help="existing environment; uses DIR/bin/python")
    parser.add_argument("--requirements", type=Path, help="exact-version pins (default: requirements-FAMILY.txt)")
    parser.add_argument("--gpu", help="one physical GPU index or UUID")
    parser.add_argument("--tier", choices=("1", "2", "1,2", "2,1"), default="1")
    parser.add_argument("--check-only", action="store_true", help="verify environment and weights without running GPU tests")
    parser.add_argument("--cpus", help="CPU affinity list for taskset -c")
    parser.add_argument("--assets", type=Path, default=os.environ.get("MATRIS_TEST_ASSETS"))
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("MATRIS_ORB_CHECKPOINT"), help="ORB pinned weights")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--large-structure", "--fp32-structure", type=Path, dest="large")
    parser.add_argument("--small-structure", "--fp64-structure", type=Path, dest="small")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--harness-args", default="", help="extra benchmark flags, parsed with shell quoting")
    args = parser.parse_args(argv)
    if args.python and args.venv:
        parser.error("choose --python or --venv, not both")
    if args.gpu is not None and (not args.gpu.strip() or "," in args.gpu or args.gpu == "-1"):
        parser.error("--gpu must select exactly one GPU")
    if not args.check_only and args.gpu is None:
        parser.error("--gpu is required for single-card validation")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    env_venv = os.environ.get(args.family.upper() + "_LANE_VENV")
    venv = args.venv or (Path(env_venv) if env_venv and not args.python else None)
    interpreter = str(venv.expanduser().resolve() / "bin" / "python") if venv else args.python or os.environ.get("PYTHON", "python3")
    interpreter = shutil.which(interpreter)
    if not interpreter:
        parser.error("selected Python interpreter does not exist; provide --python or --venv")
    requirements = args.requirements or ROOT / f"requirements-{args.family}.txt"
    if not requirements.is_file():
        parser.error(f"requirements pins are missing: {requirements}")
    if args.assets:
        args.assets = args.assets.resolve()
        if not (args.assets / "rdx_1344_1x1x1.extxyz").is_file():
            parser.error("--assets needs rdx_1344_1x1x1.extxyz")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            parser.error(f"checkpoint is missing: {args.checkpoint}")
    try:
        extra = shlex.split(args.harness_args)
        validate_harness_args(extra)
    except ValueError as exc:
        parser.error(f"--harness-args accepts complete public tuning options only: {exc}")
    harness = ROOT / "benchmarks" / "multimodel" / "run.py"
    cases = benchmark_cases(args.family)
    structures = {}
    if "2" in args.tier and not args.check_only:
        if not harness.is_file():
            parser.error(f"tier 2 requires the local benchmark harness: {harness}; restore it from the FastMD checkout")
        if not args.out:
            parser.error("tier 2 requires --out DIR")
        args.out = args.out.resolve()
        prefix = args.family.upper() + "_LANE_"
        large = args.large or os.environ.get(prefix + ("FP32_STRUCTURE" if args.family == "mace" else "LARGE_STRUCTURE"))
        small = args.small or os.environ.get(prefix + ("FP64_STRUCTURE" if args.family == "mace" else "SMALL_STRUCTURE"))
        large = large or (args.assets / "rdx_1344_1x1x1.extxyz" if args.assets else None)
        small = small or os.environ.get("MATRIS_TEST_S01_FRAMES")
        for key, value in (("large", large), ("small", small)):
            if not value or not Path(value).is_file():
                parser.error(f"tier 2 requires --{key}-structure FILE")
            structures[key] = str(Path(value).resolve())
        for _, _, _, name, _ in cases:
            output = args.out / name
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                parser.error(f"{output} already has artifacts: use a new --out")
    environment = os.environ.copy()
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
                "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        environment.pop(key, None)
    environment.update(PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / "tests"), PYTHONNOUSERSITE="1",
                       PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=args.gpu or "")
    if args.assets:
        environment["MATRIS_TEST_ASSETS"] = str(args.assets)
    affinity = ["taskset", "-c", args.cpus] if args.cpus else []
    if affinity and shutil.which("taskset") is None:
        parser.error("--cpus requires taskset")
    script = "import sys; sys.path.insert(0, sys.argv.pop(1)); from _single_gpu_lane import check_environment; check_environment(*__import__('json').loads(sys.argv[1]))"
    command = [*affinity, interpreter, "-B", "-c", script, str(Path(__file__).parent),
               json.dumps([args.family, str(requirements.resolve()), args.tier,
                           str(args.checkpoint) if args.checkpoint else None, args.check_only])]
    checked = subprocess.run(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(checked.stdout, end="", flush=True)
    if checked.returncode:
        return checked.returncode
    checkpoints = json.loads(next(line[len(MARKER):] for line in checked.stdout.splitlines() if line.startswith(MARKER)))
    if args.family == "orb":
        environment["MATRIS_ORB_CHECKPOINT"] = checkpoints["orb_v3"]
    if args.check_only:
        return 0
    if "1" in args.tier:
        script = "import sys; sys.path.insert(0, sys.argv.pop(1)); from _single_gpu_lane import run_tests; sys.exit(run_tests(sys.argv[1]))"
        result = subprocess.run([*affinity, interpreter, "-B", "-c", script, str(Path(__file__).parent), args.family],
                                cwd=ROOT, env=environment)
        if result.returncode:
            return result.returncode  # tier 2 runs only after tier 1 passed
    failed = False
    if "2" in args.tier:
        for model, dtype, structure, name, options in cases:
            # CHGNet supports its bundled model names only; passing a model_path
            # is rejected by its adapter. The packaged weights were hashed above.
            weights = [] if args.family == "chgnet" else ["--checkpoint", checkpoints[model]]
            command = [*affinity, interpreter, "-B", str(harness), "--model", model,
                       *weights, "--dtype", dtype, "--structure", structures[structure],
                       "--mode", "whole_step", "--ensemble", "nvt_nhc", "--repeats", str(args.repeats),
                       "--out", str(args.out / name), *options, *extra]
            failed |= subprocess.run(command, cwd=ROOT, env=environment).returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
