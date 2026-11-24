"""Additive wall-time accounting for one rank of a measured E/F evaluation.

These are host wall intervals, not pure GPU kernel times. Prepared asynchronous
execution accounts for pending GPU work in the later completion/readout wait.
To aggregate distributed samples, choose the *same* slowest
rank by wall_ms for each sample, then average its components. Summing separate
rank maxima or component medians does not recover end-to-end wall time.
"""
from __future__ import annotations

import math


def wall_breakdown(stats: dict) -> dict[str, float]:
    """Partition one evaluation's external wall time into disjoint intervals.

    Nested control/payload/individual-stage timings, when present, describe
    parts of these intervals and must not be added again. Residuals deliberately
    remain visible (never clipped) so accounting errors can be detected.
    """
    graph = stats["graph_timing"]
    tape = stats["tape_timing"]
    halo = graph["halo_seconds"] * 1000
    local_graph = graph["build_local_seconds"] * 1000
    route_graph = graph["route_triplet_seconds"] * 1000
    route_owner = graph.get("route_owner_seconds", 0.0) * 1000
    local_forward = tape["forward_input_ms"] + tape["forward_local_ms"]
    exchange_forward = tape["forward_exchange_ms"]
    graph_run = stats.get("model_graph_ms", 0.0)
    result = {
        "preflight_ms": stats["preflight_ms"],
        "migration_ms": stats["migration_ms"],
        "halo_ms": halo,
        "local_graph_ms": local_graph,
        "triplet_metadata_ms": route_graph,
        "owner_provenance_ms": route_owner,
        "graph_checks_ms": stats["graph_ms"] - halo - local_graph - route_graph - route_owner,
        "plan_build_ms": stats["plan_build_ms"],
        "forward_local_ms": local_forward,
        "forward_exchange_ms": exchange_forward,
        "execution_metadata_ms": (stats["forward_and_plan_ms"] - stats["plan_build_ms"]
                                  - local_forward - exchange_forward - graph_run),
        "backward_local_ms": tape["backward_local_ms"],
        "backward_exchange_ms": tape["backward_exchange_ms"],
        "backward_other_ms": (stats["vjp_ms"] - tape["backward_local_ms"]
                              - tape["backward_exchange_ms"]),
        "readout_ms": stats["readout_ms"],
        "cleanup_ms": stats["cleanup_ms"],
        "return_overhead_ms": stats["wall_ms"] - stats["evaluate_wall_ms"],
    }
    if "model_graph_ms" in stats:
        result["model_graph_ms"] = graph_run
    return result


def summarize_wall_time(rank_samples: list[list[dict]]) -> dict | None:
    """Average additive components of each step's actual slowest rank."""
    if not rank_samples:
        return None
    critical = [max(sample, key=lambda rank: rank["wall_ms"]) for sample in rank_samples]
    parts = [wall_breakdown(rank) for rank in critical]
    for rank, breakdown in zip(critical, parts):
        if not math.isclose(sum(breakdown.values()), rank["wall_ms"], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("wall-time intervals do not sum to the measured evaluation")
        if any(not math.isfinite(value) or value < -1e-6 for value in breakdown.values()):
            raise ValueError("wall-time intervals overlap or contain invalid values")
    count = len(critical)
    return {
        "protocol": "per sample select rank with maximum external wall_ms; arithmetic mean of its disjoint wall intervals",
        "critical_ranks": [rank["rank"] for rank in critical],
        "mean_wall_ms": sum(rank["wall_ms"] for rank in critical) / count,
        "mean_breakdown_ms": {key: sum(part[key] for part in parts) / count for key in parts[0]},
    }
