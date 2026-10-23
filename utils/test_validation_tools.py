"""CPU-only integration checks for the migrated validation launchers.

Uses a fake nvidia-smi plus real child processes; no GPU/model/dependency install.
Run: python -m unittest discover -s utils -p test_validation_tools.py -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "utils" / "run_gpu_validation.py"
spec = importlib.util.spec_from_file_location("single_gpu_lane", ROOT / "tools" / "ci" / "_single_gpu_lane.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


class GPUValidationWrapperTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        source = self.directory / "fastmd"
        source.mkdir()
        (source / "__init__.py").write_text("# fixture source\n")
        nvidia = self.directory / "nvidia-smi"
        nvidia.write_text(f"#!{sys.executable}\nimport os, sys\n"
                          "assert '--id=2' in sys.argv, sys.argv\n"
                          "if any(x.startswith('--query-gpu=') for x in sys.argv):\n"
                          "    print('2, 512, 90' if os.environ.get('MOCK_GPU_BUSY') else '2, 0, 0')\n")
        nvidia.chmod(0o755)
        self.environment = {**os.environ, "PATH": str(self.directory) + os.pathsep + os.environ["PATH"]}

    def run_wrapper(self, code, *options, name="run"):
        return subprocess.run([sys.executable, str(WRAPPER), "--cwd", str(self.directory), "--gpu", "2",
                               "--prefix", str(self.directory / name), *options, "--", sys.executable, "-c", code],
                              env=self.environment, capture_output=True, text=True, timeout=25)

    def result(self, name="run"):
        return json.loads((self.directory / (name + ".result.json")).read_text())

    def test_success_records_gpu_source_and_child_output(self):
        result = self.run_wrapper("import os; print('device=' + os.environ['CUDA_VISIBLE_DEVICES'])")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.directory / "run.log").read_text(), "device=2\n")
        self.assertTrue(self.result()["source_unchanged"])
        pre = json.loads((self.directory / "run.preflight.json").read_text())
        self.assertEqual(set(pre["manifest"]), {"fastmd/__init__.py"})

    def test_child_failure_is_preserved(self):
        result = self.run_wrapper("raise SystemExit(7)")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(self.result()["exit_code"], 7)

    def test_timeout_does_not_kill_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            result = self.run_wrapper("import time; time.sleep(30)", "--timeout", "0.1")
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertTrue(self.result()["timed_out"])
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_busy_selected_gpu_prevents_child_start(self):
        self.environment["MOCK_GPU_BUSY"] = "1"
        result = self.run_wrapper("from pathlib import Path; Path('child-started').touch()")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selected GPU is not idle", result.stderr)
        self.assertFalse((self.directory / "child-started").exists())
        self.assertFalse((self.directory / "run.preflight.json").exists())

    def test_timeout_kills_worker_even_after_group_leader_exits(self):
        worker = ("import os, signal, time; from pathlib import Path; "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  "Path('worker.pid').write_text(str(os.getpid())); time.sleep(60)")
        parent = ("import subprocess, sys, time; "
                  f"subprocess.Popen([sys.executable, '-c', {worker!r}]); time.sleep(60)")
        result = self.run_wrapper(parent, "--timeout", "0.5")
        pidfile = self.directory / "worker.pid"
        self.assertTrue(pidfile.exists(), result.stderr)
        pid = int(pidfile.read_text())
        stat = Path(f"/proc/{pid}/stat")
        try:
            self.assertEqual(result.returncode, 124, result.stderr)
            # An orphan can briefly remain a zombie until its new parent reaps
            # it; it must not remain an executing/sleeping worker after timeout.
            if stat.exists():
                self.assertEqual(stat.read_text().split(")", 1)[1].split()[0], "Z")
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_existing_artifact_is_not_overwritten(self):
        artifact = self.directory / "run.log"
        artifact.write_text("keep me")
        result = self.run_wrapper("print('overwrite')")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already has artifacts", result.stderr)
        self.assertEqual(artifact.read_text(), "keep me")

    def test_source_mutation_fails_successful_child(self):
        result = self.run_wrapper("from pathlib import Path; Path('fastmd/__init__.py').write_text('# changed')")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertEqual(self.result()["exit_code"], 125)
        self.assertFalse(self.result()["source_unchanged"])

    def test_invalid_timeout_fails_before_gpu_probe(self):
        for timeout in ("0", "nan", "inf", "-1"):
            with self.subTest(timeout=timeout):
                result = self.run_wrapper("print('no')", "--timeout", timeout)
                self.assertEqual(result.returncode, 2)
                self.assertIn("finite positive timeout", result.stderr)


class SingleGPULaneTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        (self.directory / "benchmarks" / "multimodel").mkdir(parents=True)
        (self.directory / "benchmarks" / "multimodel" / "run.py").write_text("# fixture\n")
        for family in lane.FAMILIES:
            (self.directory / f"requirements-{family}.txt").write_text("fixture==1\n")
        for size in ("large", "small"):
            (self.directory / (size + ".extxyz")).write_text("fixture")
        self.commands = []

    def fake_run(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if "check_environment" in " ".join(command):
            checkpoints = {name: "/weights/" + name for name in
                           ("mace_mp_0", "mace_mpa_0", "sevennet_0", "orb_v3", "chgnet_0_3_0")}
            return subprocess.CompletedProcess(command, 0, lane.MARKER + json.dumps(checkpoints) + "\n")
        return subprocess.CompletedProcess(command, 0)

    def run_lane(self, family, *options):
        with mock.patch.object(lane, "ROOT", self.directory), mock.patch.object(lane.subprocess, "run", side_effect=self.fake_run), \
             contextlib.redirect_stdout(io.StringIO()):
            return lane.main([family, "--python", sys.executable, "--gpu", "3", *options])

    def test_all_preset_commands_retain_model_dtype_capacity_and_checkpoint(self):
        for family in lane.FAMILIES:
            with self.subTest(family=family):
                self.commands.clear()
                result = self.run_lane(family, "--tier", "1,2", "--out", str(self.directory / family),
                    "--large-structure", str(self.directory / "large.extxyz"),
                    "--small-structure", str(self.directory / "small.extxyz"))
                self.assertEqual(result, 0)
                self.assertEqual(len(self.commands), 5 if family == "mace" else 4)
                for (command, kwargs), (model, dtype, size, name, options) in zip(self.commands[2:], lane.benchmark_cases(family)):
                    self.assertEqual(command[command.index("--model") + 1], model)
                    self.assertEqual(command[command.index("--dtype") + 1], dtype)
                    if family == "chgnet":
                        self.assertNotIn("--checkpoint", command)
                    else:
                        self.assertEqual(command[command.index("--checkpoint") + 1], "/weights/" + model)
                    self.assertEqual(command[command.index("--structure") + 1], str(self.directory / (size + ".extxyz")))
                    self.assertEqual(command[-len(options):], options)
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")

    def test_launcher_rank_and_tf32_environment_are_removed(self):
        with mock.patch.dict(os.environ, {"RANK": "2", "WORLD_SIZE": "4", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}):
            self.assertEqual(self.run_lane("chgnet"), 0)
        environment = self.commands[0][1]["env"]
        self.assertFalse({"RANK", "WORLD_SIZE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"} & environment.keys())
        self.assertEqual(environment["PYTHONPATH"], str(self.directory) + os.pathsep + str(self.directory / "tests"))

    def test_missing_harness_is_reported_before_any_child(self):
        (self.directory / "benchmarks" / "multimodel" / "run.py").unlink()
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
            self.run_lane("mace", "--tier", "2")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("local benchmark harness", errors.getvalue())
        self.assertFalse(self.commands)

    def test_multigpu_tier_and_gpu_list_are_rejected(self):
        for options in (("--tier", "m"), ("--gpu", "0,1")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.run_lane("mace", *options)
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.commands)

    def test_harness_cannot_override_verified_checkpoint(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            self.run_lane("mace", "--harness-args=--checkpoint=/unverified/file")
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.commands)

    def test_harness_abbreviations_and_internal_child_are_rejected_before_launch(self):
        for options in ("--checkpo /unverified/file", "--ou /existing/path", "--child /existing/file",
                        "--chil=/existing/file", "--struct /unverified/file", "--mod=other",
                        "--steps=10 --child=/existing/file", "--mode whole_step --ou=/existing/path"):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.run_lane("mace", "--harness-args=" + options)
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(self.commands)

    def test_full_public_tuning_options_are_allowed(self):
        lane.validate_harness_args(["--steps", "20", "--warmup=10", "--mode", "eager", "whole_step",
                                    "--capacity-step", "32", "--kernel", "e3nn"])

    def test_tier1_skip_is_failure(self):
        class Skips(unittest.TestCase):
            @unittest.skip("fixture missing capability")
            def test_skip(self):
                pass
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Skips)
        with mock.patch.object(lane.unittest.defaultTestLoader, "discover", return_value=suite), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lane.run_tests("mace"), 1)


if __name__ == "__main__":
    unittest.main()
