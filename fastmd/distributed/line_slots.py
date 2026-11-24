"""Line slots: the model-independent part of a line-graph partial exchange at P>1.

A model with a line graph (its nodes are the bonds; a triplet joins two bonds
of one center within the line-graph cutoff) keeps every bond with an owned
end on its rank, so a boundary bond (ends owned by two ranks) has a replica
on both, while each triplet stays with the owner of its center. A sum over a
boundary bond's triplets is therefore partial at each end. A line slot is a
boundary bond shorter than the line-graph cutoff (the only kind that can have
triplets): both ends send the slot's partial to the other and merge the same
two values, so the replicas stay equal.

The runtime owns what does not depend on the model:

- slot detection (``find_line_slots``): a rank's boundary bonds and slots in
  (peer, canonical key) order, the slot order at both ends, with per-peer
  counts and order-independent key checksums; and the admission
  (``slot_admission_error``): a boundary bond a local triplet uses is a slot;
- the symmetry check of the gathered per-peer sets (``asymmetric_peers``,
  ``symmetry_error``): both ends of a rank pair hold the same bonds;
- the binding (``bind_slots``): my k-th slot for peer p is p's (start of its
  slots for me) + k among the slots p sends;
- the SUM merge (``merge_sum``): own + peer partial, one IEEE add per slot,
  the same bits at both ends because the add commutes;
- the per-step slot-hit symmetry (``pair_hit_entries``, ``asymmetric_pairs``):
  both ends of a rank pair resolve the same slots at every step of the
  topology epoch (``fastmd.distributed.epoch``, its finite flag round).

The per-peer slot capacities of a padded layout
(``fastmd.distributed.capacity``) are the runtime's too.

A strategy keeps the rest: the cutoff and lengths that decide "short", the
collective round that gathers the per-peer sets (with its own entries), its
row layout and exchange plans (what a slot's partial travels with), merges
other than the SUM (a softmax partial's log-sum-exp merge needs the
canonical rank order), its padded layout and its runtime hooks.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

# Absolute (Angstrom) margin on the line-graph cutoff for line slots: the
# neighbor builder decides "short" per directed row, the slot set from the
# canonical length; the margin makes the slots a superset (checked).
LINE_SLOT_MARGIN = 1e-3


def _key_checksums(keys: Tensor, peers: Tensor, world: int) -> list[int]:
    """Order-independent per-peer checksum of canonical bond keys (int64 wrap)."""
    mix = (keys[:, 0] * 1000003 + keys[:, 1] * 999983 + keys[:, 2] * 7919
           + keys[:, 3] * 104729 + keys[:, 4] * 1299709)
    mix = mix * 6364136223846793005 + 1442695040888963407
    return torch.zeros(world, dtype=torch.int64, device=keys.device).index_add_(0, peers, mix).tolist()


def _peer_order(rows: Tensor, peer: Tensor) -> Tensor:
    """Ascending ``rows`` of the sorted canonical keys, stably sorted by peer:
    (peer, canonical key) order, the slot order at both ends."""
    return rows[torch.argsort(peer[rows], stable=True)]


class LineSlots(NamedTuple):
    """One rank's boundary bonds and line slots, both in (peer, canonical key) order.

    The rows index the rank's canonical keys; ``*_peer`` is the other end's
    rank of each row. The last four fields are lists of ``world`` ints, per
    peer: what both ends of a rank pair must agree on.
    """
    boundary_rows: Tensor
    boundary_peer: Tensor
    slots: Tensor
    slot_peer: Tensor
    boundary_counts: list
    boundary_checks: list
    slot_counts: list
    slot_checks: list


def find_line_slots(canonical: Tensor, boundary: Tensor, short: Tensor, peer: Tensor, world: int) -> LineSlots:
    """Slot detection: the boundary bonds and the line slots (boundary and short) of one rank.

    ``canonical``: the rank's bond keys (target ID, source ID, image), sorted
    unique rows (``device_graph.unique_rows``), so ascending rows are key
    order; ``boundary`` and ``short`` mark its rows, ``peer`` is each row's
    other rank. Local; the caller gathers the per-peer lists in its own round.
    """
    boundary_rows = _peer_order(torch.where(boundary)[0], peer)
    boundary_peer = peer[boundary_rows]
    slots = _peer_order(torch.where(boundary & short)[0], peer)
    slot_peer = peer[slots].contiguous()
    boundary_counts = torch.bincount(boundary_peer, minlength=world).tolist()
    slot_counts = torch.bincount(slot_peer, minlength=world).tolist()
    boundary_checks = _key_checksums(canonical[boundary_rows], boundary_peer, world)
    slot_checks = _key_checksums(canonical[slots], slot_peer, world)
    return LineSlots(boundary_rows, boundary_peer, slots, slot_peer,
                     boundary_counts, boundary_checks, slot_counts, slot_checks)


def slot_admission_error(boundary: Tensor, short: Tensor, used_bonds: Tensor) -> str | None:
    """A boundary bond that a local triplet uses (``used_bonds``: rows) must be a line slot."""
    if bool((boundary[used_bonds] & ~short[used_bonds]).any()):
        return "a boundary bond with triplets is not a line slot (raise LINE_SLOT_MARGIN)"
    return None


def asymmetric_peers(gathered, own, rank: int) -> list[int]:
    """Peers whose per-peer lists for this rank differ from this rank's for them.

    ``own`` is this rank's (boundary counts, boundary checksums, slot counts,
    slot checksums), ``gathered[p]`` rank p's as gathered: the counts and key
    checksums detect, with high probability, a bond that only one end decided
    is a boundary bond or a slot.
    """
    return [p for p in range(len(gathered)) if p != rank and any(
        theirs[rank] != mine[p] for theirs, mine in zip(gathered[p], own))]


def symmetry_error(label: str, asymmetric) -> str | None:
    """The collective rejection (``ExchangeContext.collective_check``) of ``asymmetric_peers``;
    ``label``: the strategy's name, the message prefix."""
    return (f"{label}: boundary bond sets differ from peers {asymmetric} "
            "(asymmetric cutoff decision on a boundary bond)" if asymmetric else None)


def bind_slots(peer: Tensor, counts, gathered_counts, rank: int, device) -> Tensor:
    """Slot binding: my k-th slot for peer p is p's (start of its slots for me) + k.

    ``peer``: each slot's peer in (peer, canonical key) order; ``counts``: my
    slot count per peer; ``gathered_counts[p]``: rank p's. Returns each slot's
    row among the slots its peer sends (the peer's slots in its own (peer,
    key) order, so those for lower ranks come first): the row to request from
    that peer.
    """
    starts = torch.tensor([sum(gathered_counts[p][:rank]) for p in range(len(gathered_counts))],
                          dtype=torch.int64, device=device)
    local_counts = torch.tensor(counts, dtype=torch.int64, device=device)
    within = torch.arange(peer.numel(), device=device) - (local_counts.cumsum(0) - local_counts)[peer]
    return (starts[peer] + within).contiguous()


def merge_sum(partial: Tensor, slots: Tensor, received: Tensor) -> Tensor:
    """The complete sum on the slots: own partial + the peer's (``received``, slot order), one IEEE add each."""
    return partial.index_add(0, slots, received)


# Per-rank-pair slot-hit symmetry rides in a MAX flag round for up to this many
# ranks (2 entries per unordered pair; the round holds 56).
PAIR_CHECK_MAX_RANKS = 6
NO_PAIR = -(1 << 62)


def pair_hit_entries(slot_peer: Tensor, slot_rows: Tensor, rank: int, world: int, device, *,
                     has_line: bool = True) -> Tensor:
    """[+hits, -hits] per unordered rank pair (i < j), this rank's own pairs only (device; capturable).

    A slot is resolved this step where ``slot_rows >= 0``; ``slot_peer`` is
    its other end. Both ends of a slot resolve it with the same decision on
    the same coordinates, so the two ranks of a pair must count the same
    resolved slots; after the MAX round, max(+) == -max(-) for every pair
    (``asymmetric_pairs``). Pairs of other ranks hold ``NO_PAIR``, and so do
    all without a line graph.
    """
    if world == 1 or world > PAIR_CHECK_MAX_RANKS:
        return torch.empty(0, dtype=torch.int64, device=device)
    entries = torch.full((world * (world - 1),), NO_PAIR, dtype=torch.int64, device=device)
    if not has_line:
        return entries
    hits = torch.zeros(world, dtype=torch.int64, device=device).index_add_(
        0, slot_peer, (slot_rows >= 0).long())
    for peer in range(world):
        if peer == rank:
            continue
        i, j = min(rank, peer), max(rank, peer)
        pair = i * world - i * (i + 1) // 2 + (j - i - 1)
        entries[2 * pair] = hits[peer]
        entries[2 * pair + 1] = -hits[peer]
    return entries


def asymmetric_pairs(agreed, world_size) -> list:
    """Rank pairs whose agreed slot-hit counts differ (``pair_hit_entries`` after the MAX round)."""
    if world_size == 1 or world_size > PAIR_CHECK_MAX_RANKS:
        return []
    pairs = [(i, j) for i in range(world_size) for j in range(i + 1, world_size)]
    agreed = list(agreed)[:2 * len(pairs)]
    return [pair for index, pair in enumerate(pairs[:len(agreed) // 2])
            if agreed[2 * index] != NO_PAIR and agreed[2 * index] != -agreed[2 * index + 1]]
