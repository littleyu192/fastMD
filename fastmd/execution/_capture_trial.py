"""One guarded normal bucket capture, without fitting a memory model.

The preflight estimate is deliberately labelled a heuristic, not a memory bound.
Every real capture (including growth/rebinding) rechecks current device headroom;
its measured peak is checked afterwards and OOM/capture failures use the caller's
normal eager fallback. No trial is added to the fitted-memory cache.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

import torch


class CaptureTrialRejected(RuntimeError):
    pass


def new_trial_stats():
    return {"attempts": 0, "successes": 0, "failures": 0, "budget_rejections": 0,
            "total_seconds": 0.0, "records": []}


def trial_estimate(eager_peak, *, atoms, pairs, triplets, capacity_pairs, capacity_triplets, dummy_atoms):
    """Conservative padding-scaled eager peak; admission heuristic only."""
    if eager_peak is None or not math.isfinite(eager_peak) or eager_peak <= 0:
        return None
    scale = max(1.0, capacity_pairs / max(pairs, 1),
                capacity_triplets / max(triplets, 1) if triplets else 1.0,
                (atoms + dummy_atoms) / max(atoms, 1))
    # A private graph pool retains intermediate storage; eager may reuse it.
    return math.ceil(4 * max(eager_peak, 2**20) * scale)


@contextmanager
def guarded_bucket_captures(runner, *, device, eager_peak, stats):
    """Guard only actual captures; ordinary replay does not synchronize here."""
    from fastmd.adapters.negotiate import capture_budget
    from fastmd.engine.saved_recompute import AUTO_THRESHOLD
    from fastmd.execution.cuda_graph import _demand

    original = runner._capture
    had_instance_method = "_capture" in vars(runner)

    def capture(pairs, triplets, sample):
        atoms, demand_pairs, demand_triplets = _demand(sample)
        budget = max(0, int(capture_budget(device)))
        limit = int(AUTO_THRESHOLD * budget)
        estimate = trial_estimate(eager_peak, atoms=atoms, pairs=demand_pairs, triplets=demand_triplets,
                                  capacity_pairs=pairs, capacity_triplets=triplets,
                                  dummy_atoms=runner.n_dummy)
        record = {"atoms": atoms, "capacity_edges": 2 * pairs, "capacity_triplets": triplets,
                  "eager_peak_bytes": eager_peak, "heuristic_bytes": estimate,
                  "budget_bytes": budget, "limit_bytes": limit,
                  "admission": "4x eager peak times padding ratio; heuristic, not a fitted bound"}
        stats["records"].append(record)
        del stats["records"][:-32]
        if estimate is None or estimate > limit:
            stats["budget_rejections"] += 1
            record.update(status="budget_rejected", seconds=0.0)
            raise CaptureTrialRejected("latency_trial capture skipped: insufficient current headroom for heuristic")

        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        stats["attempts"] += 1
        started = time.perf_counter()
        try:
            result = original(pairs, triplets, sample)
            torch.cuda.synchronize(device)
            measured = max(0, torch.cuda.max_memory_allocated(device) - allocated,
                           torch.cuda.max_memory_reserved(device) - reserved)
            record["peak_increment_bytes"] = measured
            if measured > limit:
                stats["budget_rejections"] += 1
                raise CaptureTrialRejected("latency_trial capture exceeded its current-memory budget; discard graph")
        except BaseException as error:
            stats["failures"] += 1
            record.update(status="failed", error=f"{type(error).__name__}: {str(error)[:200]}")
            raise
        else:
            stats["successes"] += 1
            record["status"] = "captured"
            return result
        finally:
            if "peak_increment_bytes" not in record:
                try:
                    record["peak_increment_bytes"] = max(
                        0, torch.cuda.max_memory_allocated(device) - allocated,
                        torch.cuda.max_memory_reserved(device) - reserved)
                except RuntimeError:
                    record["peak_increment_bytes"] = None
            record["seconds"] = time.perf_counter() - started
            stats["total_seconds"] += record["seconds"]

    runner._capture = capture
    try:
        yield
    finally:
        if had_instance_method:
            runner._capture = original
        else:
            del runner._capture
