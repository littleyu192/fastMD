"""P>1 capture coordination: one agreement per topology epoch refresh (design §5.5, §6.4; step 3.2).

``CaptureCoordinator`` owns the collective part of a CUDA-graph entry:

1. ``agree``: ONE control round per refresh (``EpochTopology.refresh``)
   carries every rank's entry state, error, capacity demand and local vote.
   It replaced the runner's three pre-3.2 rounds (closed/schedule check,
   ``graph.capacity`` demand gather, ``graph.admit``'s ~9 KB exact shape
   signatures of which only the own entry was read). The pairwise split
   consistency check (every sender's count equals its receiver's) runs on
   the gathered demands. Recapture is the OR of the votes; the capacity
   decision is a pure function of the gathered demands, so every rank takes
   the same one and NCCL sizes never depend on rank-local data.
2. Warmup that runs every collective once belongs to the runner body.
3. ``barrier``: all ranks finish warmup before any enters capture.
4. Releasing the graph before process-group destruction stays with the
   runner (``DistributedGraphRunner.close``).

The coordinator never proactively captures a neighbor bucket (P=1's bucketed
runner precaptures up to three; P>1 memory is tighter, design §6.4). It uses
only a ``gather(value, tag) -> list`` callable (``ExchangeContext._collective``:
the control rounds on the data group), so
this module imports nothing from ``fastmd.distributed``. At one rank every
round is the local value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


def check_splits(demands: Sequence[dict]) -> None:
    """Pairwise split consistency of gathered capacity demands; raises on every rank alike.

    ``demands[r]`` is rank r's ``{"has_line", "rows", "routes"}`` with
    ``routes[name] = (send_counts, receive_counts)`` indexed by peer.
    """
    names = tuple(demands[0]["routes"])
    if any(d["has_line"] != demands[0]["has_line"] or tuple(d["routes"]) != names
           for d in demands):
        raise RuntimeError("distributed capacity route/branch mismatch")
    size = len(demands)
    for sender in range(size):
        for name in names:
            sent, received = demands[sender]["routes"][name]
            if len(sent) != size or len(received) != size:
                raise RuntimeError("invalid capacity peer split length")
            for receiver in range(size):
                if sent[receiver] != demands[receiver]["routes"][name][1][sender]:
                    raise RuntimeError("capacity send/receive split mismatch")
        if "bp_slots" in names:
            # Both ends of a rank pair hold the same line slots (bond_partial).
            slots = demands[sender]["routes"]["bp_slots"][0]
            if any(slots[receiver] != demands[receiver]["routes"]["bp_slots"][0][sender]
                   for receiver in range(size)):
                raise RuntimeError("bond_partial slot capacity demand is not symmetric")


@dataclass(frozen=True)
class Agreement:
    """Replicated result of one ``agree`` round (identical on every rank)."""

    demands: tuple  # every rank's demand, rank order
    vote: bool      # some rank has no compatible entry for its own inputs


class CaptureCoordinator:
    """Rank-consistent (re)capture decisions for one resident graph entry."""

    def __init__(self, gather: Callable[[Any, str], list], *, rank: int, world_size: int):
        self._gather = gather
        self.rank, self.world_size = rank, world_size
        self.rounds = 0

    def agree(self, entry: tuple, demand: Any, vote: bool, error: str | None) -> Agreement:
        """ONE round: ``entry`` (generation, has graph, ...) must match on all ranks.

        ``error`` (this rank's rejection or None) is raised on every rank in
        ``ExchangeContext.collective_check``'s form before anything else is
        read; a rank with an error may send any demand.
        """
        self.rounds += 1
        gathered = self._gather((entry, error, demand, bool(vote)), "capture.agree")
        failures = [f"rank {rank}: {item[1]}" for rank, item in enumerate(gathered) if item[1]]
        if failures:
            raise RuntimeError("distributed validation failed: " + "; ".join(failures))
        if any(item[0] != gathered[0][0] for item in gathered):
            raise RuntimeError("model_graph entry/generation differs across ranks")
        demands = tuple(item[2] for item in gathered)
        check_splits(demands)
        return Agreement(demands, any(item[3] for item in gathered))

    def barrier(self, generation: int) -> None:
        """All ranks finished side-stream warmup; capture may start."""
        self.rounds += 1
        self._gather(generation, "capture.ready")
