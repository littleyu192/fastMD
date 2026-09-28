"""CUDA-graph capture policy shared by every MatRIS capture site."""
from __future__ import annotations

import torch

# Captures run while other host threads keep using CUDA: AsyncGPUMDLogger's
# writer thread waits on its D2H-copy events (cudaEventSynchronize) while the
# MD thread (re)captures a whole-step or model graph after a capacity grow,
# tier switch or pool reset. In torch's default "global" mode CUDA rejects
# such calls from *any* thread while a capture is underway
# (cudaErrorStreamCaptureUnsupported) and invalidates the capture.
# "thread_local" keeps the prohibition of unsafe calls for the capturing
# thread, which issues all captured work and its allocations, and lets other
# threads' calls on non-captured streams/events proceed. The captured work is
# unchanged. torch.compile's CUDA-graph trees capture in this mode as well.
CAPTURE_ERROR_MODE = "thread_local"


def capture_graph(graph: torch.cuda.CUDAGraph, *, pool=None, stream=None) -> torch.cuda.graph:
    """``torch.cuda.graph(graph, pool=pool, stream=stream)`` with the MatRIS
    capture error mode."""
    return torch.cuda.graph(
        graph, pool=pool, stream=stream, capture_error_mode=CAPTURE_ERROR_MODE
    )
