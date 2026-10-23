"""Subprocess worker: AsyncGPUMDLogger's writer thread synchronizes a frame's
D2H-copy event while GPUResidentMolecularDynamics (re)captures a CUDA graph.

Two hooks make the interleaving deterministic: the trajectory sink holds the
writer thread in write() until a capture has begun, and
torch.cuda.CUDAGraph.capture_begin holds the capture open until the writer has
called torch.cuda.Event.synchronize() on the next pending frame. In torch's
default "global" capture error mode that call fails
(cudaErrorStreamCaptureUnsupported) and invalidates the capture; the writer
thread used to die with the frame un-acknowledged, so close() waited forever.
A faulthandler watchdog turns such a hang into a traceback and exit code 1.

Usage: logger_capture_worker.py SOURCE_ROOT {whole_step,model_graph}
(fastmd must be imported from SOURCE_ROOT; only APIs that predate the capture
policy are used). Prints "OK ..." on success.
"""
from __future__ import annotations

import faulthandler
import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fastmd  # noqa: E402
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics  # noqa: E402
from fastmd.config import InferenceConfig  # noqa: E402

from md_regression_worker import system  # noqa: E402
from test_isolated_atom_handling import make_model  # noqa: E402

WAIT_S = 20.0
WATCHDOG_S = 120.0

capturing = threading.Event()  # the hooked capture has begun
synced = threading.Event()  # the writer synchronized during that capture
hook = {"armed": False, "captures": 0, "writer_syncs": 0, "writer_errors": []}

_capture_begin = torch.cuda.CUDAGraph.capture_begin
_synchronize = torch.cuda.Event.synchronize


def capture_begin(self, *args, **kwargs):
    _capture_begin(self, *args, **kwargs)
    if hook["armed"] and not capturing.is_set():
        hook["captures"] += 1
        capturing.set()
        if not synced.wait(WAIT_S):
            print("writer thread did not synchronize during the capture", flush=True)


def synchronize(self):
    during_capture = (
        capturing.is_set()
        and not synced.is_set()
        and threading.current_thread() is not threading.main_thread()
    )
    try:
        return _synchronize(self)
    except BaseException as exc:
        if during_capture:
            hook["writer_errors"].append(f"{type(exc).__name__}: {exc}".splitlines()[0])
        raise
    finally:
        if during_capture:
            hook["writer_syncs"] += 1
            synced.set()


class GatedSink:
    """Trajectory sink; once gated, its next write() waits for the capture."""

    def __init__(self) -> None:
        self.frames = 0
        self.gated = False

    def write(self, atoms) -> None:
        if self.gated:
            self.gated = False
            if not capturing.wait(WAIT_S):
                print("no capture began while the writer was held", flush=True)
        self.frames += 1

    def close(self) -> None:
        pass


def recapture(md: GPUResidentMolecularDynamics, execution: str) -> None:
    if execution == "whole_step":
        md.reset_whole_step_graph_pool()
    else:
        u_cap, t_cap = max(md.runner.cache)
        sample = md.build_graph(md.state.positions)
        md.runner.ensure(sample, (u_cap + md.runner.u_step, t_cap + md.runner.t_step))


def main(root: str, execution: str) -> int:
    source = Path(fastmd.__file__).resolve()
    assert source.is_relative_to(Path(root).resolve()), (source, root)
    faulthandler.dump_traceback_later(WATCHDOG_S, exit=True)
    torch.cuda.CUDAGraph.capture_begin = capture_begin
    torch.cuda.Event.synchronize = synchronize

    torch.manual_seed(0)
    model = make_model(num_layers=1).cuda()
    sink = GatedSink()
    md = GPUResidentMolecularDynamics(
        system(), model=model, task="ef", ensemble="nve", timestep=0.5,
        trajectory=sink, loginterval=1, logger_ring=4,
        config=InferenceConfig(execution=execution, report=False),
    )
    steps = 0
    capture_error = None
    md.run(3)
    steps += 3
    md.flush_log()
    # the writer blocks in write() on the first of these two frames, the
    # second stays pending until the capture has begun
    sink.gated = True
    md.run(2)
    steps += 2
    hook["armed"] = True
    try:
        recapture(md, execution)
    except Exception as exc:  # noqa: BLE001 - reported below
        capture_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"[:300]
        print("capture failed:", capture_error, flush=True)
    hook["armed"] = False
    if capture_error is None:
        md.run(3)
        steps += 3
    print("closing", flush=True)
    md.close()  # hung here before the fix (dead writer, frame not done)
    faulthandler.cancel_dump_traceback_later()
    expected = steps + 1  # plus the initial frame
    print(
        f"hooked_captures={hook['captures']} writer_syncs_during_capture="
        f"{hook['writer_syncs']} writer_errors={hook['writer_errors']} "
        f"frames={sink.frames}/{expected}",
        flush=True,
    )
    if (
        capture_error is None
        and hook["captures"] == 1
        and hook["writer_syncs"] == 1
        and not hook["writer_errors"]
        and sink.frames == expected
    ):
        print("OK", flush=True)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
