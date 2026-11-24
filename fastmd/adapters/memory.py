"""Measured memory model of a model without a K6 calibration.

K6 (``fastmd.engine.memory_admission``) predicts MatRIS peaks from tables measured per
architecture; any other model is "uncalibrated" there, so every automatic memory choice takes its safe
side and execution auto stays eager. A :class:`MeasuredMemoryModel` measures the model instead: the
peak CUDA bytes of one eager step and of one capture at three or more capacity points, fitted per kind
by least squares to

    bytes = a + b * edges            (+ c * triplets when the topology has triplets)

The fit is valid with at least three distinct capacity points (and one more than its unknowns), with
triplet/edge ratios spanning at least ``min_ratio_span`` when triplets are fitted (else the two costs
cannot be told apart), with costs that are not negative beyond the noise (a negative per-edge or
per-triplet cost whose effect over the measured range stays within the residual bound is taken as 0),
and with every residual within the residual bound ``max(residual_floor, residual_fraction * largest
peak)``. Its bound at a capacity is the fit plus twice the largest residual (at least
``headroom_floor``): the headroom comes from the measured fit, not from a fixed factor. It predicts
nothing beyond ``max_extrapolation`` times the largest measured edge (or triplet) capacity, and an
invalid fit predicts only the capacities it measured (the largest peak measured at or below them; its
bound adds that headroom), so auto keeps the safe side elsewhere. The model is valid when its capture fit is (the
eager fit is recorded, not required). MatRIS stays on K6.

Peaks measured at one system's atoms can be linear in pieces: the peak is that of whichever part of
the evaluation holds the most memory, and with the atoms fixed a part dominated by per-atom buffers
holds it at few edges per atom, a part dominated by per-edge buffers at many (a steeper line above the
crossover; e.g. MACE-SOYO at about 25 edges per atom). Without triplets, a kind whose peaks are not
linear over all its capacities is therefore fitted over the largest ones (the smallest capacity
dropped one at a time while at least three remain); such a fit predicts nothing below its smallest
fitted capacity, where the peaks are not on its line.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
from typing import Callable, Iterable, Mapping, Sequence

KINDS = ("eager", "capture")
MIB = 1 << 20


@dataclass(frozen=True)
class MemoryPoint:
    """One measurement: the peak bytes above the allocated baseline of one ``kind`` run at a capacity."""

    kind: str
    edges: int
    triplets: int
    peak_bytes: int


@dataclass(frozen=True)
class MemoryFit:
    kind: str
    a: float  # bytes
    b: float  # bytes per edge
    c: float  # bytes per triplet (0 without triplets)
    max_residual: float | None  # bytes, over the fitted points (None: not fitted)
    residual_bound: float  # bytes
    max_edges: int  # the largest measured edge capacity
    max_triplets: int  # the largest measured triplet capacity
    valid: bool
    reason: str
    dropped: int = 0  # the smallest capacities left out of the fit (their peaks are not on its line)
    min_edges: int = 0  # the smallest fitted edge capacity (a fit that dropped points predicts nothing below)

    def predict(self, edges: int, triplets: int = 0) -> float:
        return max(self.a + self.b * edges + self.c * triplets, 0.0)


class MeasuredMemoryModel:
    """Peaks of one model measured at capacity points and fitted linearly per kind (see the module doc).

    ``triplets``: whether the topology has triplets (a ``c * triplets`` term). Points come from
    :meth:`measure` (CUDA) or :meth:`from_points`.
    """

    calibration = "measured"

    def __init__(self, points: Iterable[MemoryPoint], *, triplets: bool, residual_floor: float = 4 * MIB,
                 residual_fraction: float = 0.02, max_extrapolation: float = 2.0, min_ratio_span: float = 1.5,
                 headroom_floor: float = 2 * MIB) -> None:
        self.points = tuple(points)
        self.triplets = bool(triplets)
        self.residual_floor = float(residual_floor)
        self.residual_fraction = float(residual_fraction)
        self.max_extrapolation = float(max_extrapolation)
        self.min_ratio_span = float(min_ratio_span)
        self.headroom_floor = float(headroom_floor)
        self._failure = None  # why the measurement failed (``failed``)
        unknown = {point.kind for point in self.points} - set(KINDS)
        if unknown:
            raise ValueError(f"memory point kinds must be in {KINDS}, got {sorted(unknown)}")
        self.fits = {kind: self._fit(kind) for kind in KINDS if any(p.kind == kind for p in self.points)}

    @classmethod
    def from_points(cls, points: Iterable[MemoryPoint], *, triplets: bool, **bounds) -> "MeasuredMemoryModel":
        return cls(points, triplets=triplets, **bounds)

    @classmethod
    def failed(cls, reason: str, *, triplets: bool) -> "MeasuredMemoryModel":
        """A measurement that failed (``reason``, e.g. a CUDA OOM): no points, invalid, predicts nothing."""
        model = cls((), triplets=triplets)
        model._failure = str(reason)
        return model

    @classmethod
    def measure(cls, runs: Mapping[str, Callable], capacities: Sequence, device, *, triplets: bool,
                warmup: bool = True, metric: str = "allocated", **bounds) -> "MeasuredMemoryModel":
        """Measure ``runs`` (kind -> a callable that runs one step of that kind at a capacity and returns
        the (edges, triplets) capacity it held) at each capacity (an edge count, or an (edges, triplets)
        pair for a topology with triplets, whose ratios must vary): the peak allocated bytes above
        the allocated baseline before the call, after a collection and ``empty_cache`` (``metric="reserved"``:
        the peak reserved bytes above the reserved baseline, what runs out on the device). Every point must
        start from the same state: a capture's run captures its graph and releases it before returning
        (a captured graph that stays alive keeps state later captures share, e.g. the first capture's
        cuBLAS workspace, which then counts in one point only: the peaks are not linear and the fit is
        invalid). ``warmup``: each run first runs once, unmeasured, at the smallest capacity (its
        one-time state then exists before the measurement)."""
        import torch
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MeasuredMemoryModel.measure needs a CUDA device")
        if metric not in ("allocated", "reserved"):
            raise ValueError(f"metric must be 'allocated' or 'reserved', got {metric!r}")
        current, peak = ((torch.cuda.memory_allocated, torch.cuda.max_memory_allocated) if metric == "allocated"
                         else (torch.cuda.memory_reserved, torch.cuda.max_memory_reserved))
        points = []
        convert = lambda capacity: (tuple(int(value) for value in capacity) if isinstance(capacity, (tuple, list))
                                    else int(capacity))
        for kind, run in runs.items():
            if warmup:
                run(convert(min(capacities, key=lambda c: c[0] if isinstance(c, (tuple, list)) else c)))
            for capacity in capacities:
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                baseline = current(device)
                torch.cuda.reset_peak_memory_stats(device)
                edges, held_triplets = run(convert(capacity))
                torch.cuda.synchronize(device)
                points.append(MemoryPoint(kind, int(edges), int(held_triplets), int(peak(device) - baseline)))
        return cls(points, triplets=triplets, **bounds)

    @staticmethod
    def _ratios(points) -> list[float]:
        return [p.triplets / p.edges for p in points if p.edges > 0]

    def _fit(self, kind: str) -> MemoryFit:
        points = [point for point in self.points if point.kind == kind]
        fit = self._fit_points(kind, points)
        if self.triplets or fit.max_residual is None or fit.max_residual <= fit.residual_bound:
            return fit
        # Not linear over every capacity (edges only): fit the largest capacities (see the module doc).
        capacities = sorted({point.edges for point in points})
        for dropped in range(1, len(capacities) - 2):
            upper = [point for point in points if point.edges >= capacities[dropped]]
            refit = self._fit_points(kind, upper)
            if refit.valid:
                return replace(refit, dropped=dropped, reason=(
                    f"{kind}: not linear over all {len(capacities)} capacities (max residual "
                    f"{fit.max_residual / MIB:.2f} MiB > bound {fit.residual_bound / MIB:.2f} MiB); over the largest "
                    f"{len(capacities) - dropped} ({capacities[dropped]} to {capacities[-1]} edges): max residual "
                    f"{refit.max_residual / MIB:.2f} MiB <= {refit.residual_bound / MIB:.2f} MiB"))
        return fit

    def _fit_points(self, kind: str, points: list) -> MemoryFit:
        import numpy as np
        columns = 3 if self.triplets else 2
        x = np.array([[1.0, p.edges] + ([p.triplets] if self.triplets else []) for p in points], dtype=np.float64)
        y = np.array([p.peak_bytes for p in points], dtype=np.float64)
        distinct = len({(p.edges, p.triplets if self.triplets else 0) for p in points})
        max_edges = max((p.edges for p in points), default=0)
        max_triplets = max((p.triplets for p in points), default=0) if self.triplets else 0
        min_edges = min((p.edges for p in points), default=0)
        bound = max(self.residual_floor, self.residual_fraction * (float(y.max()) if len(y) else 0.0))
        need = max(3, columns + 1)
        invalid = lambda why: MemoryFit(kind, 0.0, 0.0, 0.0, None, bound, max_edges, max_triplets, False,
                                        f"{kind}: {why}", min_edges=min_edges)
        if distinct < need:
            return invalid(f"{distinct} distinct capacity points, at least {need} needed")
        ratios = self._ratios(points)
        if self.triplets and not (ratios and max(ratios) > 0 and (
                min(ratios) == 0 or max(ratios) >= self.min_ratio_span * min(ratios))):
            return invalid(f"triplet/edge ratios span less than {self.min_ratio_span}x: the per-edge and "
                           "per-triplet costs cannot be told apart")
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        if np.linalg.matrix_rank(x) < columns:
            return invalid("edges and triplets do not vary independently")
        residual = float(np.abs(x @ coefficients - y).max())
        a, b = float(coefficients[0]), float(coefficients[1])
        c = float(coefficients[2]) if self.triplets else 0.0
        # A negative cost whose effect over the measured range stays within the residual bound is noise
        # (taken as 0, which only raises the predictions); a larger one means the peaks are not linear.
        problems = [why for ok, why in ((b * max_edges >= -bound, f"negative bytes per edge {b:.1f}"),
                                        (c * max_triplets >= -bound, f"negative bytes per triplet {c:.1f}"),
                                        (residual <= bound, f"max residual {residual / MIB:.2f} MiB > bound "
                                                            f"{bound / MIB:.2f} MiB (not linear)")) if not ok]
        reason = f"{kind}: " + ("; ".join(problems) if problems else
                                f"{len(points)} points, max residual {residual / MIB:.2f} MiB <= {bound / MIB:.2f} MiB")
        return MemoryFit(kind, a, max(b, 0.0), max(c, 0.0), residual, bound, max_edges, max_triplets, not problems,
                         reason, min_edges=min_edges)

    @property
    def valid(self) -> bool:
        """The capture kind was measured and its fit is valid: what the admission bounds. The eager fit is
        recorded (``fits``, ``reason``), not required: no bound uses it, and an eager evaluation's peaks can
        vary between runs where the captured ones do not (a compiled model's eager path)."""
        return "capture" in self.fits and self.fits["capture"].valid

    @property
    def reason(self) -> str:
        if self._failure is not None:
            return self._failure
        if "capture" not in self.fits:
            return "no capture points measured"
        return "; ".join(fit.reason for fit in self.fits.values())

    def predict(self, kind: str, edges: int, triplets: int = 0) -> int | None:
        """The fitted peak bytes, or None (no fit of ``kind``, beyond the extrapolation range, or below the
        fitted range of a fit over the largest capacities). A fit that is not valid predicts only the
        capacities it measured: the largest peak measured at or below them (nothing is extrapolated from
        peaks that are not linear)."""
        fit = self.fits.get(kind)
        if fit is None:
            return None
        if not fit.valid:
            # A measured capacity: the largest peak measured at or below it (a capture may hold a little less
            # than the capacity decided at, and the peaks of an invalid fit need not grow with the edges).
            points = [point for point in self.points if point.kind == kind]
            if not any(point.edges == edges and (not self.triplets or point.triplets == triplets) for point in points):
                return None
            return max(point.peak_bytes for point in points
                       if point.edges <= edges and (not self.triplets or point.triplets <= triplets))
        if edges > self.max_extrapolation * max(fit.max_edges, 1) or (fit.dropped and edges < fit.min_edges):
            return None
        if self.triplets:
            # Two costs fitted on a few ratios: predict only within the measured triplet/edge ratios.
            ratios = self._ratios([p for p in self.points if p.kind == kind])
            if triplets > self.max_extrapolation * max(fit.max_triplets, 1) or not (
                    min(ratios) <= (triplets / edges if edges > 0 else float("inf")) <= max(ratios)):
                return None
        return int(round(fit.predict(edges, triplets)))

    def headroom(self, kind: str) -> int:
        """Twice the fit's largest residual (that of the fit over every point where it is not valid), at
        least ``headroom_floor`` bytes (the floor alone for a fit without residuals: too few points)."""
        if kind not in self.fits:
            raise ValueError(f"no {kind!r} points were measured (measured: {sorted(self.fits)})")
        return int(max(2 * (self.fits[kind].max_residual or 0.0), self.headroom_floor))

    def bound(self, kind: str, edges: int, triplets: int = 0) -> int | None:
        """The prediction plus the measured headroom (:meth:`headroom`), or None."""
        predicted = self.predict(kind, edges, triplets)
        if predicted is None:
            return None
        return predicted + self.headroom(kind)

    def max_edges(self, kind: str, budget: float, triplets_per_edge: float = 0.0) -> int | None:
        """The largest edge capacity whose bound fits ``budget`` bytes (triplets growing with the
        edges at ``triplets_per_edge``), within the extrapolation range; 0 where even the headroom does not
        fit; None without a valid fit, or where it would lie below the fitted range of a fit over the largest
        capacities."""
        fit = self.fits.get(kind)
        if fit is None or not fit.valid:
            return None
        per_edge = fit.b + fit.c * triplets_per_edge
        limit = int(self.max_extrapolation * max(fit.max_edges, 1))
        if self.triplets and triplets_per_edge > 0:
            limit = min(limit, int(self.max_extrapolation * max(fit.max_triplets, 1) / triplets_per_edge))
        if budget < self.headroom(kind):
            return 0  # even an empty capture's bound (the headroom) does not fit
        fixed = fit.a + self.headroom(kind)
        if per_edge <= 0:
            largest = limit if fixed <= budget else 0
        else:
            largest = max(0, min(int((budget - fixed) // per_edge), limit))
        if fit.dropped and largest < fit.min_edges:
            return None  # nothing in the fitted range fits, and below it the fit predicts nothing
        return largest

    def summary(self) -> dict:
        return {"calibration": self.calibration, "valid": self.valid, "reason": self.reason,
                "triplets": self.triplets,
                "fits": {kind: {**asdict(fit), "max_residual_mib": None if fit.max_residual is None
                                else fit.max_residual / MIB} for kind, fit in self.fits.items()},
                "points": [asdict(point) for point in self.points]}
