"""The memory predictor of a model on the node-halo exchange (``ShardSpec``): one per rank, measured on its device.

A rank of the node-halo exchange runs the model on its rows (the owned slots,
then the ghost rows) and edges (the shard's edge rule), so the peak of one
step is that of the model on that many rows and edges, plus the exchange's
buffers. :class:`NodeHaloMemoryPredictor` measures the model alone on the
rank's device (its energy tap and backward; at P>1 for a model with its own
layer loop, the rank's share with every exchange the identity; no
collective): the peak CUDA bytes above the allocated baseline of one eager
step (the warm-up) and of one captured step (that warm-up, then the capture
into a private pool, as the evaluator captures) at the four (edges, rows) points of a 2 x 2
grid at a quarter and a half of the rank's capacities, on synthetic batches
(the shapes set the memory), fitted per kind by least squares to

    bytes = a + b * edges + c * rows

The row term is the ghost term: ghost rows add rows without edges, which a fit
over edges alone cannot tell apart. The prediction at the rank's capacities
extrapolates the fit by at most ``MAX_EXTRAPOLATION`` (the largest probe is
half of the capacities, so capacities grown by more than a quarter measure
again), adds twice the fit's
largest residual (at least ``HEADROOM_FLOOR``) and the exchange's buffers,
counted from the declared widths (per exchange point the rows sent and
received, the concatenated input and their gradients). A fit with a residual
above ``max(RESIDUAL_FLOOR, RESIDUAL_FRACTION x the largest peak)`` or a
negative cost beyond it predicts nothing: the automatic choices then take the
safe side (eager).

The predictor implements ``fastmd.engine.memory_admission.MemoryPredictor``
(one level, "off": the models of a declared shard have no recompute levels);
the generic ``MemoryAdmission`` gathers its predictions and every rank's
available memory and decides on the same data on every rank.
"""
from __future__ import annotations

import contextlib
import gc
import math

import torch

KINDS = {"eager": "eager", "model_graph": "capture"}  # execution mode -> measured kind
MAX_EXTRAPOLATION = 2.5
RESIDUAL_FLOOR = 4 << 20
RESIDUAL_FRACTION = 0.02
HEADROOM_FLOOR = 16 << 20
_MIB = 1 << 20


def exchange_bytes(points, *, rows: int, ghost_rows: int, send_rows: int, owned_slots: int, itemsize: int) -> int:
    """The buffers of one step's exchanges beyond the model's own: per declared point the rows sent and received,
    the concatenated input and the owners' sums with their gradients, and the coordinates' COPY."""
    per_row = sum(point.width for point in points)
    rows_moved = 2 * send_rows + 2 * ghost_rows + 2 * rows + owned_slots + 1
    return int(itemsize * (per_row * rows_moved + 6 * (send_rows + ghost_rows + rows)))


def synthetic_batch(adapter, *, rows: int, edges: int, centres: int, cell: torch.Tensor, species: int,
                    dtype: torch.dtype, device):
    """A graph batch of ``rows`` rows and ``edges`` edges (centres among the first ``centres`` rows, neighbours any
    other row, image 0) at random coordinates in a cube of half the cutoff's edge, so that every edge is shorter than
    the cutoff: the shapes of a rank's batch, for a memory probe."""
    from fastmd.adapters.protocol import GraphBatch

    rows, centres = max(2, int(rows)), max(1, min(int(centres), int(rows)))
    cutoff = float(adapter.topology_spec().cutoff)
    generator = torch.Generator(device="cpu").manual_seed(rows * 7919 + edges)
    cartesian = torch.rand((rows, 3), generator=generator, dtype=torch.float64) * (0.5 * cutoff)
    cell = cell.to(torch.float64).cpu()
    fractions = cartesian @ torch.linalg.inv(cell)
    positions = cartesian.to(device=device, dtype=dtype)
    index = torch.arange(edges, dtype=torch.int64)
    centre = index % centres
    neighbour = (centre + 1 + torch.div(index, centres, rounding_mode="floor") % (rows - 1)) % rows
    return GraphBatch(positions=positions, frac_coords=fractions.to(device=device, dtype=dtype),
                      cell=cell.to(device=device, dtype=dtype),
                      species=torch.full((rows,), int(species), dtype=torch.int32, device=device), n_real=rows,
                      edge_index=torch.stack((centre, neighbour), dim=1).to(device=device, dtype=torch.int32),
                      edge_image=torch.zeros((edges, 3), dtype=torch.float32, device=device), n_edges=None,
                      cutoff=float(adapter.topology_spec().cutoff), padding=None)


def _step(adapter, inputs, batch, derivative=None):
    """The model's part of one evaluation: the energy tap (per atom, or of the whole system: what one CUDA rank
    runs) and its backward to the coordinates; with ``derivative`` (P>1) for a model whose adapter runs its own
    layer loop (``derivative.tap``), its share of the energy with every row owned and every exchange and global
    sum the identity (no collective)."""
    if derivative is not None and derivative.tap:
        from .node_halo import ExchangeHooks

        rows, device = batch.positions.shape[0], batch.positions.device
        owned = torch.ones(rows, dtype=torch.bool, device=device)
        exchanges = derivative.context(None, owned, torch.full((), float(rows), dtype=torch.float64, device=device),
                                       torch.ones((), dtype=torch.bool, device=device), count_only=True)
        with torch.enable_grad(), ExchangeHooks.exchanging(exchanges):
            leaf = batch.positions.detach().requires_grad_(True)
            energy = derivative.local_energy(inputs, leaf, batch.cell, owned, exchanges)
            (grad,) = torch.autograd.grad(energy, (leaf,))
        return energy.detach(), grad
    from fastmd.adapters.protocol import EnergyTap

    with torch.enable_grad():
        leaf = batch.positions.detach().requires_grad_(True)
        tap = adapter.node_energy if isinstance(adapter, EnergyTap) else adapter.system_energy
        energy = tap(inputs, leaf, batch.cell).to(torch.float64).sum()
        (grad,) = torch.autograd.grad(energy, (leaf,))
    return energy.detach(), grad


def _peaks(adapter, inputs, batch, device, capture: bool, stream, derivative=None) -> tuple[int, int | None]:
    """The evaluator's procedure from an allocated baseline: a warm-up step on the side ``stream`` (the eager peak),
    then (``capture``) a capture into a private pool, released at return (the capture's peak: the larger of both)."""
    from fastmd.engine._graph_capture import capture_graph

    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        _step(adapter, inputs, batch, derivative)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    eager = int(torch.cuda.max_memory_allocated(device) - baseline)
    if not capture:
        return eager, None
    torch.cuda.empty_cache()
    graph = torch.cuda.CUDAGraph()
    # Keep the probe on its warmed stream, away from other graph owners'
    # default capture stream and their pool-owned cuBLAS workspaces.
    with capture_graph(graph, pool=torch.cuda.graph_pool_handle(), stream=stream, check_replay=False):
        _step(adapter, inputs, batch, derivative)
    torch.cuda.synchronize(device)
    del graph
    return eager, int(torch.cuda.max_memory_allocated(device) - baseline)


class NodeHaloMemoryPredictor:
    """One rank's ``MemoryPredictor`` of a model on the node-halo exchange (see the module docstring).

    :meth:`measure` probes the model at a rank's capacities (``extents``: ``rows``, ``edges``, ``ghost_rows``,
    ``send_rows``, ``owned_slots``); ``predict(level, mode, extents)`` is the bound of one step of ``mode``
    ("eager" or "model_graph") at ``extents`` in bytes above the allocated baseline, or None.
    """

    slack = 0.0  # the bound carries the fit's measured headroom instead
    strict = True

    def __init__(self, points: list[tuple[str, int, int, int]], *, model: str, exchange_points, itemsize: int,
                 failure: str | None = None):
        self.points = list(points)  # (kind, edges, rows, peak bytes)
        self.model, self.exchange_points, self.itemsize = model, tuple(exchange_points), int(itemsize)
        self.failure = failure
        self.fits = {kind: self._fit(kind) for kind in sorted({point[0] for point in self.points})}
        self.decisions: list = []

    @classmethod
    def measure(cls, adapter, spec, extents: dict, *, cell: torch.Tensor, device, dtype: torch.dtype,
                kinds=("eager", "capture"), stream=None, single: bool = False) -> "NodeHaloMemoryPredictor":
        """Probe ``adapter`` on this rank's device at the 2 x 2 grid of a quarter and a half of ``extents``' edges
        and rows (``kinds``: "eager" and/or "capture"; ``stream``: the side stream of the warm-ups; ``single``: one
        CUDA rank, which runs the model's energy tap, not its own layer loop), after one unmeasured run at the
        smallest point (one-time state then exists before the measurement). Local: no collective; a probe that fails
        gives a predictor that predicts nothing."""
        from .node_halo import ShardedEnergyDerivative

        name = getattr(adapter, "name", type(adapter).__name__)
        points, failure = [], None
        derivative = None if single else ShardedEnergyDerivative(adapter, spec)
        edges, rows = int(extents["edges"]), int(extents["rows"])
        small_edges, small_rows = max(64, edges // 4), max(16, rows // 4)
        grid = [(e, r) for e in (small_edges, max(2 * small_edges, edges // 2))
                for r in (small_rows, max(2 * small_rows, rows // 2))]
        centres = max(1, int(extents.get("owned_slots", rows)))
        try:
            stream = torch.cuda.Stream(device=device) if stream is None else stream
            for index, (probe_edges, probe_rows) in enumerate([grid[0]] + grid):
                batch = synthetic_batch(adapter, rows=probe_rows, edges=probe_edges,
                                        centres=max(1, centres * probe_rows // max(rows, 1)), cell=cell,
                                        species=spec.pad_species, dtype=dtype, device=device)
                eager, captured = _peaks(adapter, adapter.prepare(batch), batch, device, "capture" in kinds, stream,
                                         derivative)
                if index:  # (the first run is the unmeasured warm-up)
                    points += [(kind, probe_edges, probe_rows, peak) for kind, peak in (("eager", eager),
                                                                                         ("capture", captured))
                               if kind in kinds]
                del batch
        except Exception as error:  # noqa: BLE001 - a probe never raises (its peers would wait in the admission)
            failure = f"memory probe failed: {type(error).__name__}: {(str(error).splitlines() or [''])[0][:200]}"
        try:
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - as above (a sticky CUDA error raises again in the step)
            failure = failure or f"memory probe failed: {type(error).__name__}: {str(error)[:200]}"
        return cls(points, model=name, exchange_points=spec.exchange_points,
                   itemsize=torch.finfo(dtype).bits // 8, failure=failure)

    def _fit(self, kind: str) -> dict:
        import numpy as np

        points = [point for point in self.points if point[0] == kind]
        x = np.array([[1.0, edges, rows] for _, edges, rows, _ in points], dtype=np.float64)
        y = np.array([peak for *_, peak in points], dtype=np.float64)
        bound = max(RESIDUAL_FLOOR, RESIDUAL_FRACTION * (float(y.max()) if len(y) else 0.0))
        fit = {"kind": kind, "points": len(points), "max_edges": max((p[1] for p in points), default=0),
               "max_rows": max((p[2] for p in points), default=0), "residual_bound": bound, "valid": False}
        if len({(p[1], p[2]) for p in points}) < 4 or np.linalg.matrix_rank(x) < 3:
            return {**fit, "reason": f"{kind}: {len(points)} probe points, four distinct ones needed"}
        (a, b, c), *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = float(np.abs(x @ np.array([a, b, c]) - y).max())
        problems = [why for ok, why in ((b * fit["max_edges"] >= -bound, f"negative bytes per edge {b:.1f}"),
                                        (c * fit["max_rows"] >= -bound, f"negative bytes per row {c:.1f}"),
                                        (residual <= bound, f"max residual {residual / _MIB:.2f} MiB > "
                                                            f"{bound / _MIB:.2f} MiB (not linear)")) if not ok]
        return {**fit, "a": float(a), "b": max(float(b), 0.0), "c": max(float(c), 0.0), "max_residual": residual,
                "valid": not problems, "reason": f"{kind}: " + ("; ".join(problems) if problems else
                                                               f"{len(points)} points, max residual "
                                                               f"{residual / _MIB:.2f} MiB")}

    def covers(self, extents: dict, kinds=("eager", "capture")) -> bool:
        """Whether ``kinds`` were measured over a range that reaches ``extents`` (within its extrapolation): else
        the caller measures again. An invalid fit over such a range stays: it predicts nothing there."""
        return all((fit := self.fits.get(kind)) is not None
                   and extents["edges"] <= MAX_EXTRAPOLATION * fit["max_edges"]
                   and extents["rows"] <= MAX_EXTRAPOLATION * fit["max_rows"] for kind in kinds)

    # --- MemoryPredictor ----------------------------------------------------------------------------------

    @property
    def label(self) -> str:
        return f"node-halo exchange of {self.model!r}, measured on this device"

    def levels(self) -> tuple:
        return ("off",)

    def calibrated(self, modes=("eager", "model_graph")) -> bool:
        return all((fit := self.fits.get(KINDS[mode])) is not None and fit["valid"] for mode in modes)

    def predict(self, level: str, mode: str, extents: dict) -> int | None:
        fit = self.fits.get(KINDS[mode])
        if fit is None or not fit["valid"] or extents["edges"] > MAX_EXTRAPOLATION * fit["max_edges"] \
                or extents["rows"] > MAX_EXTRAPOLATION * fit["max_rows"]:
            return None
        model = fit["a"] + fit["b"] * extents["edges"] + fit["c"] * extents["rows"]
        headroom = max(2 * fit["max_residual"], HEADROOM_FLOOR)
        exchange = exchange_bytes(self.exchange_points, rows=extents["rows"], ghost_rows=extents["ghost_rows"],
                                  send_rows=extents["send_rows"], owned_slots=extents["owned_slots"],
                                  itemsize=self.itemsize) if extents["ghost_rows"] else 0
        return int(math.ceil(max(model, 0.0) + headroom + exchange))

    def predictions(self, modes, extents: dict) -> dict:
        return {"by_level": {"off": {mode: self.predict("off", mode, extents) for mode in modes}}}

    def decide(self, gathered: list[dict], budgets: list[int], *, modes, requested: str | None, event: str) -> str:
        return "off"

    def peaks(self, entry: dict, level: str) -> dict:
        return entry["by_level"][level]

    def record(self, modes, *, event: str, requested: str, fits: list, shared_rows_modes: list, by_rank: list[dict],
               calibrated: bool) -> dict:
        return {"event": event, "requested": requested, "fits": fits, "by_rank": by_rank, "calibrated": calibrated}

    def describe(self, entry: dict) -> str:
        return f"{entry['rows']:,} rows ({entry['ghost_rows']:,} ghost), {entry['edges']:,} edge rows"

    def summary(self) -> dict:
        return {"predictor": "node_halo_measured", "model": self.model, "failure": self.failure,
                "fits": {kind: dict(fit) for kind, fit in self.fits.items()},
                "points": [{"kind": kind, "edges": edges, "rows": rows, "peak_bytes": peak}
                           for kind, edges, rows, peak in self.points],
                "model_form": "peak above the allocated baseline = a + b * edges + c * rows (fitted on this "
                              "device) + 2 x max residual + the exchange buffers"}

    def bind(self):
        return contextlib.nullcontext()
