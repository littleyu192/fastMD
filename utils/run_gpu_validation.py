"""Run one single-GPU validation, preserving logs, source hashes and exit status.

Example: python utils/run_gpu_validation.py --gpu 0 --prefix /new/path/case -- python -m unittest ...
Existing artifacts are never overwritten. Timeout cleanup only targets the child
process group. On a shared node, select one GPU with --gpu (or a single existing
CUDA_VISIBLE_DEVICES value); other GPUs need not be idle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def gpu_state(gpu=None):
    select = [] if gpu is None else ["--id=" + gpu]
    state = subprocess.check_output(
        ["nvidia-smi", *select, "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(
        ["nvidia-smi", *select, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"], text=True)
    return {"gpus": state, "applications": apps, "time_unix": time.time()}


def source_manifest(directory):
    """Hash production Python only, excluding tests and transient artifacts."""
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((directory / "fastmd").rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        help="one physical GPU index or UUID; defaults to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--timeout", type=float, default=900.)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("a command and finite positive timeout are required")
    if args.gpu is not None and (not args.gpu.strip() or "," in args.gpu or args.gpu == "-1"):
        parser.error("--gpu / CUDA_VISIBLE_DEVICES must select exactly one GPU")
    directory = args.cwd.resolve(strict=True)
    manifest = source_manifest(directory)
    if not manifest:
        parser.error("--cwd must contain the production fastmd source tree")
    prefix = args.prefix.resolve()
    files = {key: Path(str(prefix) + suffix) for key, suffix in
             (("pre", ".preflight.json"), ("log", ".log"), ("result", ".result.json"))}
    if any(p.exists() for p in files.values()):
        parser.error("validation prefix already has artifacts")
    try:
        pre = gpu_state(args.gpu)
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"cannot inspect the selected GPU: {exc}")
    rows = pre["gpus"].strip().splitlines()
    if len(rows) != 1:
        parser.error("select exactly one GPU with --gpu or CUDA_VISIBLE_DEVICES")
    _, memory, utilization = (int(x.strip()) for x in rows[0].split(","))
    if memory > 64 or utilization > 5 or pre["applications"].strip():
        parser.error(f"selected GPU is not idle: {rows[0]}; {pre['applications'].strip()}")
    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with files["pre"].open("x") as stream:
        json.dump({**pre, "command": command, "cwd": str(directory), "manifest": manifest,
                   "environment": {k: environment.get(k) for k in
                        ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONPATH")}},
                  stream, indent=2)
    started = time.monotonic()
    timed_out = False
    interrupted = False
    with files["log"].open("x") as stream:
        try:
            process = subprocess.Popen(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
        except OSError as exc:
            stream.write(f"could not start validation: {exc}\n")
            code = 127
        else:
            print(json.dumps({"pid": process.pid, "log": str(files["log"])}), flush=True)
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                timed_out = isinstance(exc, subprocess.TimeoutExpired)
                interrupted = not timed_out
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                # The leader may exit on TERM while a worker ignores it. Waiting
                # for the leader alone does not establish that its group is gone.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                code = 124 if timed_out else 130
    unchanged = manifest == source_manifest(directory)
    try:
        post = gpu_state(args.gpu)
    except (OSError, subprocess.CalledProcessError) as exc:
        post = {"error": str(exc)}
    code = code or (0 if unchanged else 125)
    result = {"exit_code": code, "timed_out": timed_out, "interrupted": interrupted,
              "source_unchanged": unchanged, "seconds": time.monotonic() - started,
              "postflight": post}
    with files["result"].open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
