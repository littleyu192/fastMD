"""Coordinated capacities of the topology epoch's padded layout (eager and model_graph).

Padding never represents a real atom: producer sinks, requested padding rows,
safe geometry and the readout real mask isolate it in both primal and VJP.
The capacity policy is ``TieredCapacity`` (step 3.2), planned at every epoch
refresh from the gathered demands (``EpochTopology.refresh``); the padded
layout itself is ``fastmd.models.matris.distributed.bond_partial.pad_bond_partial``. Step
4.2 removed the pre-3.2 policy, the rows exchange's padding and the rebuild
topology's per-evaluation padding (``pad_prepared``).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import math

from fastmd.engine.capacity_policy import CapacityKey, CapturePolicy, JointCapacityPolicy, tier as _tier
from fastmd.config import CapacityConfig

# The tier knobs are CapturePolicy's (the single-GPU tier cache's MATRIS_TIER_*
# options, read through env_value). Growth band of the P>1 model_graph tiers
# (CapturePolicy.graph_growth_headroom, MATRIS_TIER_GROWTH_HEADROOM): an
# overflow grows every bucket to the tier of its window high-water mark plus
# max(sigma * std, this fraction of it). P=1 grows to the bare tier and relies
# on its tier cache and whole-step transactions; a P>1 recapture is collective
# (~300 ms) and the bare quantum is ~0.1-0.4% of the per-rank triplets at
# 10,752 atoms (design §6.4).


class TieredCapacity:
    """P>1 capacity: the single-GPU ``JointCapacityPolicy`` tiers plus hysteresis (step 3.2).

    The layout interface of ``pad_bond_partial`` (``rows``, ``peers``,
    ``generation``, ``local``, ``splits``), with no collective of its own:
    ``plan`` is a pure function of every rank's demand, gathered by
    ``CaptureCoordinator.agree``, so every rank installs the same layout. Tier
    inputs are reduced collectively:

    * demand: the global MAX over ranks for each row bucket (n, u, i, t), so
      every rank shares one row key; each (sender, receiver) payload split
      keeps its own bucket. A bulk-synchronous step waits for the largest
      rank, whose padded work and graph pool set the step time and the
      memory ceiling either way, and one shared key cannot be overflowed by
      a smaller rank alone;
    * free memory: the MIN over ranks, as ``MemoryAdmission.check`` (every
      rank's budget) before each later capture;
    * time: no P>1 decision reads a timing in step 3.2 (``execution="auto"``
      chooses by row thresholds), so there is nothing to reduce.

    Tiers: u, i, t use ``JointCapacityPolicy.key_for``'s rule (u_step /
    min_pad_u, t_step / min_pad_t), n uses n_dummy for both, peer splits
    per-peer scaled quanta (line slots u_step / min_pad_u, ghost rows
    n_dummy). An overflow (demand >= capacity in
    any bucket; the strict sink row) grows EVERY bucket to the tier of its
    window high-water mark plus a band of max(sigma * std,
    ``_TIER_GROWTH_HEADROOM`` of it), never below its current capacity.
    Shrinking follows the single-GPU hysteresis
    (``GPUResidentMolecularDynamics._maybe_switch_down_tier``): every
    ``_TIER_CHECK_INTERVAL`` observations, once the window is full and
    ``_TIER_DOWN_COOLDOWN`` observations passed since the last growth, the
    tier of p95 + sigma * std must be strictly smaller in some bucket, no
    larger in any, and still hold the window maximum plus the minimum
    padding. The window is cleared on every change. There is no tier cache
    and no precapture at P>1: one resident entry.
    """

    def __init__(self, context, config=None, *, growth_headroom=None, capture_policy=None):
        self.context = context
        self.config = CapacityConfig() if config is None else config
        if not isinstance(self.config, CapacityConfig):
            raise TypeError("capacity must be CapacityConfig")
        # Window, sigma, check interval, cooldown, bands and warmups: one
        # CapturePolicy, built once per evaluator (EpochTopology).
        self.capture_policy = (CapturePolicy.from_config(self.config, distributed=True) if capture_policy is None
                               else capture_policy)
        # The growth band's floor (fraction of the high-water mark); the eager
        # topology epoch uses a tighter one (its growth costs no recapture).
        self.growth_headroom = (self.capture_policy.graph_growth_headroom if growth_headroom is None
                                else float(growth_headroom))
        c = self.config
        # The single-GPU policy object: u/t keys, capacity history, overflow
        # events and padding statistics use its own methods.
        self.policy = JointCapacityPolicy(u_step=c.u_step, t_step=c.t_step,
                                          min_pad_u=c.min_pad_u, min_pad_t=c.min_pad_t)
        self.rows = None
        self.peers = {}
        self.generation = 0
        self._buckets = None
        # Installed capacity of every bucket ever used, including the routes
        # of the other line branch, so a branch flip back does not regrow.
        self._retained = {}
        self._window = deque(maxlen=self.capture_policy.hysteresis_window)
        self.observations = self._last_growth = self._since_check = 0
        self.growths = self.shrinks = 0
        # Every input of plan() and of the captures besides the gathered
        # demands: compared across ranks in each CaptureCoordinator round
        # (per-process config/env; a rank with another warmup count would pair
        # its warmup collectives with another rank's capture).
        self.fingerprint = (tuple(asdict(c).items()), self.capture_policy.fingerprint, self.growth_headroom)

    def set_growth_headroom(self, growth_headroom=None) -> None:
        """Change the growth band's floor (every rank at the same point; part of the fingerprint)."""
        self.growth_headroom = (self.capture_policy.graph_growth_headroom if growth_headroom is None
                                else float(growth_headroom))
        self.fingerprint = self.fingerprint[:-1] + (self.growth_headroom,)

    def _layout(self, demands):
        """Bucket order and (step, min_pad) of each: rows, then every peer split."""
        c, size = self.config, len(demands)
        buckets = [(("rows", "n"), c.n_dummy, c.n_dummy), (("rows", "u"), c.u_step, c.min_pad_u),
                   (("rows", "i"), c.u_step, c.min_pad_u), (("rows", "t"), c.t_step, c.min_pad_t)]
        for name in demands[0]["routes"]:
            if name == "bp_slots":
                step, pad = max(1, c.u_step // size), max(1, c.min_pad_u // size)
            else:  # bp_ghosts
                step, pad = max(1, c.n_dummy // size), max(1, c.n_dummy // size)
            buckets.extend(((name, s, r), step, pad) for s in range(size) for r in range(size))
        return tuple(buckets)

    @staticmethod
    def _reduce(demands, buckets):
        """Global MAX of each row bucket over ranks; each peer split as sent."""
        rows = {key: max(d["rows"][key] for d in demands) for key in ("n", "u", "i", "t")}
        return tuple(rows[key[1]] if key[0] == "rows" else demands[key[1]]["routes"][key[0]][0][key[2]]
                     for key, _, _ in buckets)

    def _current(self):
        rows = self.rows[0]
        return tuple(rows[key[1]] if key[0] == "rows" else self.peers[key[0]][key[1]][key[2]]
                     for key, _, _ in self._buckets)

    def _install(self, capacities, demands):
        size = len(demands)
        values = dict(zip((key for key, _, _ in self._buckets), capacities))
        row = {key: values[("rows", key)] for key in ("n", "u", "i", "t")}
        peers = {name: [[values[(name, s, r)] for r in range(size)] for s in range(size)]
                 for name in demands[0]["routes"]}
        changed = self.rows is None or row != self.rows[0] or peers != self.peers
        if changed:
            self.rows, self.peers = [dict(row) for _ in range(size)], peers
            self._retained.update(values)
            self.generation += 1
            self.policy.capacity_history.append(CapacityKey(row["u"], row["t"]))
            self._window.clear()
        return changed

    def plan(self, demands):
        """Install the agreed layout for the gathered ``demands``; True if it changed.

        Pure in the replicated demands and this object's (replicated) state:
        call it on every rank with the same gathered list.
        """
        buckets = self._layout(demands)
        flipped = self._buckets is not None and buckets != self._buckets
        if flipped:
            # The global line branch (any rank owns a triplet) changed the
            # route set: a growth event from the retained capacities (0 for
            # a route never used). The window's columns change meaning, so it
            # restarts.
            self._window.clear()
        self._buckets = buckets
        demand = self._reduce(demands, buckets)
        self._window.append(demand)
        self.observations += 1
        self._since_check += 1
        rows = dict(zip(("n", "u", "i", "t"), demand))
        if flipped:
            return self._grow(tuple(self._retained.get(key, 0) for key, _, _ in buckets), demands)
        if self.rows is not None:
            current = self._current()
            if any(value >= capacity for value, capacity in zip(demand, current)):
                self.policy.record_overflow(rows["u"], rows["t"])
                return self._grow(current, demands)
            if self._since_check >= self.capture_policy.check_interval:
                self._since_check = 0
                return self._maybe_shrink(current, demands)
            return False
        return self._grow(None, demands)

    def _grow(self, current, demands):
        columns = tuple(zip(*self._window))
        grown = []
        for i, (_, step, pad) in enumerate(self._buckets):
            values = columns[i]
            high = max(values)
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            band = max(math.ceil(self.capture_policy.hysteresis_sigma * std), int(high * self.growth_headroom))
            tier = _tier(high + band, step, pad)
            grown.append(tier if current is None else max(tier, current[i]))
        self.growths += 1
        self._last_growth = self.observations
        return self._install(grown, demands)

    def _maybe_shrink(self, current, demands):
        if len(self._window) < self._window.maxlen:
            return False
        if self.observations - self._last_growth < self.capture_policy.down_cooldown:
            return False
        candidate = []
        for i, (_, step, pad) in enumerate(self._buckets):
            values = sorted(column[i] for column in self._window)
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            tier = _tier(values[int(0.95 * (len(values) - 1))] + int(self.capture_policy.hysteresis_sigma * std),
                         step, pad)
            # Never above the current tier; the window maximum must still fit
            # with the minimum padding (at least one sink row).
            if tier > current[i] or tier < values[-1] + max(pad, 1):
                return False
            candidate.append(tier)
        if tuple(candidate) == current:
            return False
        self.shrinks += 1
        return self._install(candidate, demands)

    def local(self):
        if self.rows is None:
            raise RuntimeError("capacity has not been observed")
        return self.rows[self.context.rank]

    def splits(self, name):
        rank = self.context.rank
        table = self.peers[name]
        return tuple(table[rank]), tuple(row[rank] for row in table)

    def node_rows(self):
        """(lowest, highest) local atom rows over the ranks: the owned capacity plus the received ghost blocks
        (the only row count of the layout that differs between ranks; ``triton_wide_gemm.rank_uniform``)."""
        table = self.peers["bp_ghosts"]
        rows = [self.rows[rank]["n"] + sum(row[rank] for row in table) for rank in range(len(self.rows))]
        return min(rows), max(rows)

    def stats(self):
        return {"policy": "tiers", "generation": self.generation, "observations": self.observations,
                "growths": self.growths, "shrinks": self.shrinks, "window": len(self._window),
                "window_size": self._window.maxlen, "check_interval": self.capture_policy.check_interval,
                "cooldown": self.capture_policy.down_cooldown, "sigma": self.capture_policy.hysteresis_sigma,
                "growth_headroom": self.growth_headroom}


def _pad(value, rows, fill=0):
    if rows < len(value):
        raise ValueError("capacity overflow before padded copy")
    result = value.new_full((rows, *value.shape[1:]), fill)
    result[:len(value)].copy_(value)
    return result
