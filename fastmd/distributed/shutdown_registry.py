"""The shutdown registry: what must be released before the process groups are destroyed.

A captured CUDA graph that holds NCCL work (the topology epoch's whole-step
graph in model_graph, its topology graph in eager) makes the communicator's
finalize wait for the graph's destruction, so ``destroy_process_group`` hangs
while one is alive; module hooks a strategy installs (none today) go after
the graphs that may run them. Every such object registers here (weakly: a
collected object releases its own) under one of the ordered ``PHASES``, and
``launch.shutdown()`` calls :func:`release_all` before destroying the groups:
one collection (unreachable owners release their graphs then), ``close()`` of
every live object phase by phase, then a synchronization, at which the
destroyed graphs' release callbacks (NCCL's included) run.
"""
from __future__ import annotations

import gc
import weakref

import torch

# Release order: evaluators (each releases its engine, graph runner and topology), then graphs kept alive
# elsewhere (a topology held by a closed or collected evaluator's last_graph), then hooks.
PHASES = ("evaluators", "graphs", "hooks")
_LIVE = {phase: weakref.WeakSet() for phase in PHASES}


def register(owner, phase: str) -> None:
    """Release ``owner`` (its ``close()``) in ``phase`` at shutdown unless it unregisters (an evaluator's
    ``close()``) or is collected first; a topology's ``close()`` is idempotent and runs again then."""
    _LIVE[phase].add(owner)


def unregister(owner, phase: str) -> None:
    """``owner`` released its graphs or hooks itself (its ``close()``)."""
    _LIVE[phase].discard(owner)


def live(phase: str) -> list:
    """The objects of ``phase`` not released yet."""
    return list(_LIVE[phase])


def release_all() -> None:
    """``close()`` every registered object, phase by phase (``launch.shutdown``, before the groups go)."""
    gc.collect()  # unreachable owners release their graphs here
    for phase in PHASES:
        for owner in live(phase):
            owner.close()
    if torch.cuda.is_initialized():
        # Destroyed graphs' release callbacks (NCCL's included) run at a synchronization.
        torch.cuda.synchronize()
