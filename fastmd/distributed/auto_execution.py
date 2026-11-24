"""Size-based choice between distributed eager and model_graph.

Small shards are host bound: the captured graph removes thousands of eager
launches, while asynchronous eager hides device work under them. Larger
shards are device bound, where the graph's capacity padding (and host
prepare serialized before replay) costs more than the launches it saves.
The input is the per-rank triplet row count, the dominant work and the padded
dimension; the slowest rank sets the step time.

Measured on 4x GB300 (RDX, native, FP32 highest; median ms, 2 and 4 ranks):
248k rows/rank model_graph 100.8 vs eager 123.6; from 494k to 3.8M rows/rank
eager is 3-13% faster (2 ranks at 494k: 3.1%) and reserves 7-12% less memory.
The crossover lies between 248k and 494k (interpolated ~420-470k). The
default threshold sits below it: when both are close, eager also avoids graph
capture/recapture and the graph pool.

One CUDA rank on the single-GPU engine has its own threshold: without exchange
or prepare work its crossover lies much higher. Single GPU, RDX, topology,
FP32 highest (BASE0 G-S1, 1 GB300): 1,344 atoms (951k triplet rows)
model_graph 63.2 vs eager 65.2 ms (legacy 71.6 vs 84.1); 10,752 atoms (7.6M
rows) 448.6 vs 448.9 ms, where eager reserves ~98 instead of ~143 GiB.
The legacy profile's eager is host bound at every measured size: model_graph
is 11.6-15.6% faster from 951k to 7.6M rows (M1 regression study), so with memory
admission on it prefers model_graph whenever that fits.

At P>1 the Stage 0 result above reversed on the m1 code with the native
backend: model_graph is faster at 1.9M-15.2M rows per rank with bond_partial
(10.5-11.5% at 1.9M-7.6M with the rows exchange removed in step 4.2). So under
memory admission the P>1 evaluator with the native backend also prefers
model_graph whenever it fits. The torch_reference backend keeps the
threshold: its padded model_graph replays are 2.2-4.4x slower than eager
at 1.9M rows per rank (P=4, 10,752 atoms, GB200).

Memory admission (memory_admission) passes the modes that fit; a mode that
does not fit is never chosen, and model_graph is not proposed again until the
rows fall below the hysteresis band under the rows it was rejected at.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AutoExecution:
    """Choose model_graph below and eager at or above ``eager_min_triplet_rows``.

    The count compared is the maximum over ranks of owned triplet rows, so
    every rank takes the same decision. The first evaluation decides; a later
    switch needs the count to leave the band threshold * (1 -/+ hysteresis) on
    the other side, so MD fluctuations near the threshold never flip the mode.
    """

    eager_min_triplet_rows: int = 400_000
    hysteresis: float = 0.15
    # The P=1 single-GPU engine's threshold on its own triplet rows (see above).
    single_rank_eager_min_triplet_rows: int = 4_000_000

    def __post_init__(self):
        for name in ("eager_min_triplet_rows", "single_rank_eager_min_triplet_rows"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.hysteresis, bool) or not isinstance(self.hysteresis, (int, float))
                or not math.isfinite(self.hysteresis) or not 0 <= self.hysteresis < 1):
            raise ValueError("hysteresis must be a finite number in [0, 1)")

    def choose(self, max_rows: int, current: str | None, *, single_rank: bool = False) -> str:
        """Pure decision; ``current=None`` is the first evaluation."""
        threshold = self.single_rank_eager_min_triplet_rows if single_rank else self.eager_min_triplet_rows
        if current is None:
            return "eager" if max_rows >= threshold else "model_graph"
        if current == "model_graph" and max_rows >= threshold * (1 + self.hysteresis):
            return "eager"
        if current == "eager" and max_rows < threshold * (1 - self.hysteresis):
            return "model_graph"
        return current


class AutoExecutionState:
    """Current choice and its decision record for one evaluator.

    Deterministic in the gathered counts; ranks evaluating in lockstep keep
    identical states, which ``observe`` verifies from the same gathered list.
    """

    def __init__(self, policy: AutoExecution, *, device_type: str,
                 single_rank: bool = False, fixed: str | None = None, prefer_model_graph: bool = False):
        self.policy = policy
        self.mode = None
        self.last_max_rows = None
        self.decisions = []
        # Legacy single engine, or a P>1 evaluator with the native backend,
        # under memory admission: model_graph whenever it fits.
        self.prefer_model_graph = prefer_model_graph
        # Rows at which admission last rejected model_graph.
        self.memory_bound_rows = None
        # ``fixed``: the caller pins eager and records why.
        self.single_rank = single_rank
        self.fixed = fixed
        if fixed is None and device_type != "cuda":
            self.fixed = "model_graph requires CUDA; eager"
        if self.fixed is not None:
            self.mode = "eager"
            self.decisions.append({"epoch": None, "mode": "eager", "max_rank_triplet_rows": None,
                                   "reason": self.fixed})

    @property
    def state(self):
        """Small comparable token gathered with the per-step counts."""
        return self.mode, len(self.decisions)

    def propose(self, rows: int) -> str:
        """The mode ``observe`` would choose at ``rows`` before memory admission."""
        if self.fixed is not None:
            return self.mode
        chosen = ("model_graph" if self.prefer_model_graph else
                  self.policy.choose(rows, self.mode, single_rank=self.single_rank))
        bound = self.memory_bound_rows
        if chosen == "model_graph" and bound is not None and rows >= bound * (1 - self.policy.hysteresis):
            return "eager"
        return chosen

    def observe(self, epoch: int, gathered: list, fits: set | None = None) -> bool:
        """Decide from every rank's (atoms, triplet rows, state); True if mode changed.

        ``fits``: the modes memory admission admitted at this evaluation (None: not checked).
        """
        if any(item[2] != gathered[0][2] for item in gathered):
            raise RuntimeError("auto execution state differs across ranks: "
                               f"{[item[2] for item in gathered]}")
        rows = max(item[1] for item in gathered)
        self.last_max_rows = rows
        if self.fixed is not None:
            return False
        chosen = self.propose(rows)
        memory = None
        if fits is not None and chosen == "model_graph" and "model_graph" not in fits:
            self.memory_bound_rows = rows
            chosen, memory = "eager", "model_graph does not fit device memory (memory admission)"
        elif fits is not None and chosen == "eager" and "eager" not in fits:
            chosen, memory = "model_graph", "eager does not fit device memory (memory admission)"
        if chosen == self.mode:
            return False
        threshold = self.threshold
        if memory is not None:
            reason = f"max per-rank triplet rows {rows}: {memory}"
        elif self.mode is None and self.prefer_model_graph and chosen == "model_graph":
            reason = f"max per-rank triplet rows {rows}: model_graph preferred and it fits"
        elif self.mode is None:
            relation = ">=" if chosen == "eager" else "<"
            reason = f"max per-rank triplet rows {rows} {relation} threshold {threshold}"
        elif self.memory_bound_rows is not None:
            bound = self.memory_bound_rows * (1 - self.policy.hysteresis)
            reason = (f"max per-rank triplet rows {rows} fell below {bound:.0f} (model_graph memory bound): "
                      f"{self.mode} -> {chosen}")
            self.memory_bound_rows = None
        else:
            bound = threshold * (1 + self.policy.hysteresis if chosen == "eager"
                                 else 1 - self.policy.hysteresis)
            reason = (f"max per-rank triplet rows {rows} crossed hysteresis bound {bound:.0f}: "
                      f"{self.mode} -> {chosen}")
        self.decisions.append({"epoch": epoch, "mode": chosen, "max_rank_triplet_rows": rows,
                               "reason": reason})
        self.mode = chosen
        return True

    @property
    def threshold(self) -> int:
        return (self.policy.single_rank_eager_min_triplet_rows if self.single_rank
                else self.policy.eager_min_triplet_rows)

    def summary(self) -> dict:
        return {"policy": asdict(self.policy), "mode": self.mode or "undecided",
                "threshold_triplet_rows": None if self.prefer_model_graph else self.threshold,
                "prefer_model_graph": self.prefer_model_graph, "memory_bound_rows": self.memory_bound_rows,
                "last_max_rank_triplet_rows": self.last_max_rows,
                "switches": max(0, len(self.decisions) - 1), "decisions": list(self.decisions),
                "input": ("this rank's triplet rows (P=1 single-GPU engine)" if self.single_rank else
                          "max over ranks of owned triplet rows, gathered with the per-step atom count"),
                "reason": self.decisions[-1]["reason"] if self.decisions else "decided at the first evaluation"}
