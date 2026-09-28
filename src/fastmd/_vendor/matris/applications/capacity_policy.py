"""Joint edge/triplet capacity planning for fixed-shape CUDA Graphs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class CapacityKey:
    u_capacity: int
    t_capacity: int


class JointCapacityPolicy:
    """Track and plan correlated undirected-edge/triplet capacities.

    Runtime growth is monotonic to avoid graph churn. ``recommend`` performs a
    small offline greedy facility search over observed joint keys and is used by
    the paper artifact to quantify the padding-vs-capture tradeoff.
    """

    def __init__(
        self,
        *,
        u_step: int = 512,
        t_step: int = 8192,
        min_pad_u: int = 256,
        min_pad_t: int = 4096,
        u_cost_weight: float = 2.0,
        t_cost_weight: float = 1.0,
    ) -> None:
        self.u_step = int(u_step)
        self.t_step = int(t_step)
        self.min_pad_u = int(min_pad_u)
        self.min_pad_t = int(min_pad_t)
        self.u_cost_weight = float(u_cost_weight)
        self.t_cost_weight = float(t_cost_weight)
        self.capacity_history: list[CapacityKey] = []
        self.observations: list[tuple[int, int, CapacityKey]] = []
        self.overflow_events: list[tuple[int, int]] = []

    def key_for(self, u: int, t: int) -> CapacityKey:
        u_capacity = ((int(u) // self.u_step) + 1) * self.u_step
        if u_capacity - int(u) < self.min_pad_u:
            u_capacity += self.u_step
        t_capacity = ((int(t) // self.t_step) + 1) * self.t_step
        if t_capacity - int(t) < self.min_pad_t:
            t_capacity += self.t_step
        return CapacityKey(u_capacity, t_capacity)

    def start(self, u: int, t: int) -> CapacityKey:
        key = self.key_for(u, t)
        self.capacity_history.append(key)
        return key

    def grow(self, current: CapacityKey, u: int, t: int) -> CapacityKey:
        required = self.key_for(u, t)
        grown = CapacityKey(
            max(current.u_capacity, required.u_capacity),
            max(current.t_capacity, required.t_capacity),
        )
        if grown != current:
            self.capacity_history.append(grown)
        return grown

    def observe(self, u: int, t: int, capacity: CapacityKey) -> None:
        self.observations.append((int(u), int(t), capacity))

    def record_overflow(self, u: int, t: int) -> None:
        self.overflow_events.append((int(u), int(t)))

    def reset_observations(self) -> None:
        self.observations.clear()
        self.overflow_events.clear()

    def _assignment_cost(
        self,
        point: tuple[int, int],
        buckets: list[CapacityKey],
    ) -> float:
        u, t = point
        costs = [
            self.u_cost_weight * (key.u_capacity - u)
            + self.t_cost_weight * (key.t_capacity - t)
            for key in buckets
            if u <= key.u_capacity and t <= key.t_capacity
        ]
        return min(costs) if costs else float("inf")

    def recommend(self, max_buckets: int = 4) -> list[CapacityKey]:
        if not self.observations or max_buckets <= 0:
            return []
        points = [(u, t) for u, t, _ in self.observations]
        full = self.key_for(
            max(u for u, _ in points),
            max(t for _, t in points),
        )
        candidates = sorted({self.key_for(u, t) for u, t in points})
        selected = [full]
        while len(selected) < max_buckets:
            current_cost = sum(self._assignment_cost(point, selected) for point in points)
            best_key = None
            best_cost = current_cost
            for candidate in candidates:
                if candidate in selected:
                    continue
                trial = selected + [candidate]
                cost = sum(self._assignment_cost(point, trial) for point in points)
                if cost < best_cost:
                    best_key = candidate
                    best_cost = cost
            if best_key is None:
                break
            selected.append(best_key)
        return sorted(selected)

    def stats(self) -> dict[str, object]:
        if not self.observations:
            return {
                "observations": 0,
                "overflow_events": len(self.overflow_events),
                "capacity_history": [
                    (key.u_capacity, key.t_capacity) for key in self.capacity_history
                ],
                "recommended_keys": [],
            }
        u_padding = [
            (key.u_capacity - u) / max(u, 1)
            for u, _, key in self.observations
        ]
        t_padding = [
            (key.t_capacity - t) / max(t, 1)
            for _, t, key in self.observations
        ]
        return {
            "observations": len(self.observations),
            "overflow_events": len(self.overflow_events),
            "capacity_history": [
                (key.u_capacity, key.t_capacity) for key in self.capacity_history
            ],
            "u_padding_mean": sum(u_padding) / len(u_padding),
            "u_padding_max": max(u_padding),
            "t_padding_mean": sum(t_padding) / len(t_padding),
            "t_padding_max": max(t_padding),
            "recommended_keys": [
                (key.u_capacity, key.t_capacity) for key in self.recommend()
            ],
        }
