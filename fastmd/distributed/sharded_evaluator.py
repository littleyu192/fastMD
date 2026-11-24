"""The distributed evaluator of a declared shard (``ShardSpec``): MACE, MACE-SOYO, SevenNet, ORB, CHGNet and TACE on several GPUs.

:class:`ShardedEvaluator` has the API of ``DistributedMatRISEvaluator``
(``distribute``, ``evaluate`` returning a ``DistributedPrediction``,
``gather``, ``close``, ``config_summary``), so the one MD driver
(``GPUResidentMolecularDynamics.from_evaluator``) runs it unchanged. It runs
the generic node-halo exchange (``fastmd.distributed.node_halo``) of the
adapter's ``ShardSpec``:

* P>1 on CUDA: the node-halo topology (``NodeHaloTopology``: owned and ghost
  rows of one edges-only builder with the shard's edge rule, refreshed on the
  device when the skin is used up), the declared exchanges as hooks (or called
  by the adapter's own layer loop, ``ShardedTap``), the sharded derivative.
  ``execution="model_graph"`` captures the evaluation step per layout in one
  CUDA graph: the ghost coordinates' COPY, the builder's exact filter, the
  model with its exchanges, the backward with their reverse SUMs, the
  energy's sum over the ranks and the step flags (NCCL inside the graph);
  ``"eager"`` runs the same step uncaptured. Under model_graph the MD driver
  (``from_evaluator``) runs each step through ``md_step``: the integrator's
  halves inside the same captured step, the refresh at the committed
  positions before it, a discarded step rolled back (as
  ``DistributedMatRISEvaluator.md_step``).
* One CUDA rank: the single-GPU engine's parts, unchanged (a single-system
  fixed-capacity builder and the engine-owned ``EnergyDerivative``, captured
  likewise for ``"model_graph"``); no hook is installed.
* CPU (Gloo, any P): the exact reference topology, rebuilt every evaluation
  (the FP64 exactness tests).
* A line-graph shard (``message_flow="line_graph"``, CHGNet) runs the
  line-partial exchange (``fastmd.distributed.line_partial``) instead of the
  node halo: edges with an owned end, triplets at owned centres, the line
  slots' partials merged at the declared pair_merge points; its memory
  predictor adds the triplets (``LineGraphMemoryPredictor``).

The forces of a P>1 evaluation are the owned atoms'; the energy is the FP64
total and the stress (``evaluate(state, stress=True)``) the whole system's,
the same on every rank. The geometry (``InferenceConfig.geometry_dtype``;
"float64" by default for a float32 model whose adapter declares float64 geometry
input, at every rank count; "float32" opts out; ``geometry_dtype``) keeps the
positions, the cell, the ghost coordinates' COPY, the edge vectors, the
strain, the forces and the stress in float64, the model's arithmetic in its
own dtype; the domain decomposition, the ownership and the halo read the
float32 rounding of the coordinates, and on CUDA so do the neighbour search
and the edge decision (as on one GPU; the CPU ranks' exact reference
topologies decide their edges in float64). Each rank negotiates its plan with the entry point
"distributed" (``fastmd.adapters.negotiate``) and the ranks agree it (the
kernels, then the rank contract); ``plan_record()`` reports it, with a
``distributed`` block at P>1. At each new layout a memory admission predicts
every rank's peak (``NodeHaloMemoryPredictor``; ``LineGraphMemoryPredictor``
for a line graph), decides execution auto
(model_graph where it fits on every rank, else eager) and allocates the
predicted peak once on every rank before any collective of the step.
"""
from __future__ import annotations

import copy
import dataclasses
import gc
import hashlib
import json
import struct
import time
import traceback
import warnings

import torch
import torch.distributed as dist
from torch import Tensor

from fastmd.adapters.protocol import (
    EnergyTap, NegotiationError, ShardDeclaration, ShardedTap, ShardSpecDeclaration, SystemEnergyTap,
)
from fastmd.config import CapacityConfig
from fastmd.precision import deterministic_mode, matmul_precision_scope, precision_state
from . import shutdown_registry
from .config import ParallelConfig
from .evaluator import DistributedMatRISEvaluator, DistributedPrediction, collective_check
from .exchange import ExchangeContext
from .node_halo import (
    EDGE_RULES, ExchangeHooks, NodeHaloReferenceTopology, NodeHaloStrategy, ShardedEnergyDerivative, agree_exchanges,
    count_exchanges, summed_stress,
)
from .line_partial import (
    LineExchangeHooks, LineGraphMemoryPredictor, LinePartialDerivative, LinePartialStrategy, LineReferenceTopology,
    agree_line_exchanges, count_line_exchanges, line_exchange_schedule,
)
from .node_halo_memory import KINDS, NodeHaloMemoryPredictor
from .node_halo_topology import TIER_ALIGN, TIER_HEADROOM, _tier, edge_tier
from .partition import AtomState, DomainPartition
from .rank_contract import agree_rank_contract

# Step flags, MAX over the ranks: an edge overflow, used-up candidates (skin or candidate buffer), a
# non-finite energy or force, a rank whose input is not the refreshed state (a refresh on every rank); for a model
# with a neighbour cap, the most neighbours of an atom in the builds so far; for a line-graph shard, line slots that
# did not cover the step (also in the stale flag).
_OVERFLOW, _STALE, _NONFINITE, _NEW_STATE, _NEIGHBORS = 0, 1, 2, 3, 4
_SLOTS = 4
# TorchScript profiling and its optimized spherical-harmonic forward/backward
# kernels may initialize on distinct calls. Use the engine's standard three
# preparation runs, with the same count on all ranks, before CUDA capture.
_CAPTURE_WARMUPS = CapacityConfig().warmup


def exchange_schedule(spec, dtype: torch.dtype, ghost_rows: list[int], flags: int = 4,
                      coordinate_dtype: torch.dtype | None = None) -> dict:
    """The P>1 step's exchanges in order (``exchange_schedule``: name, kind, values per row, dtype, and each rank's
    forward and backward bytes from its ``ghost_rows``) and their sums per rank (``bytes_per_step``): the builder's
    coordinate COPY, the derivative's differentiable one (both in ``coordinate_dtype``, the geometry's: default
    ``dtype``) and each declared feature COPY (their reverse SUMs in the backward), each global sum of a pooled
    readout (FP64, its backward the sum of the gradients), then the FP64 sum of the energy (and strain gradients) and
    the step flags' MAX (``flags`` values)."""
    coordinate_dtype = dtype if coordinate_dtype is None else coordinate_dtype
    stages = [("coordinates.builder", 3, False, coordinate_dtype), ("coordinates", 3, True, coordinate_dtype)] + [
        (point.name, point.width, True, dtype) for point in spec.exchange_points]
    copies = [{"name": name, "kind": "copy", "width": width, "dtype": str(stage_dtype).removeprefix("torch."),
               "forward_bytes": [ghost * width * (torch.finfo(stage_dtype).bits // 8) for ghost in ghost_rows],
               "backward_bytes": [ghost * width * (torch.finfo(stage_dtype).bits // 8) if reverse else 0
                                  for ghost in ghost_rows]}
              for name, width, reverse, stage_dtype in stages]
    sums = [{"name": point.name, "kind": "global_sum", "width": point.width, "dtype": "float64",
             "forward_bytes": [8 * point.width] * len(ghost_rows),
             "backward_bytes": [8 * point.width] * len(ghost_rows)} for point in spec.global_reductions]
    reductions = [{"name": "energy_and_strain", "kind": "all_reduce_sum", "width": "1 (10 with the stress)",
                   "dtype": "float64"},
                  {"name": "step_flags", "kind": "all_reduce_max", "width": flags, "dtype": "int64"}]
    return {"exchange_schedule": copies + sums + reductions,
            "bytes_per_step": {direction: [sum(stage[f"{direction}_bytes"][rank] for stage in copies + sums)
                                           for rank in range(len(ghost_rows))]
                               for direction in ("forward", "backward")}}


def dry_run_record(adapter, config, *, atoms, world_size: int, device=None, parallel: ParallelConfig | None = None):
    """The plan :class:`ShardedEvaluator` would record for ``atoms`` (ASE) at ``world_size`` ranks, from one process
    without ranks or evaluations (``python -m fastmd.explain --world-size N``): one rank's negotiation, and at P>1
    the decomposition simulated here, labelled as estimates (``distributed["estimate"]``): the domain grid, each
    rank's owned atoms, its ghost rows within the halo (on CUDA with the node-halo topology's capacity tiers of a
    first refresh; the CPU reference topology has none), and its edges at ``atoms``' positions with the edge
    capacity (CUDA: the tier of their MAX); for a line-graph shard the edges with an owned end, the triplets of the
    owned centres and the line slots (with the triplet capacity and the sink rows). Memory is not estimated: each
    rank measures its own at the first layout."""
    import numpy as np
    from ase.neighborlist import neighbor_list
    from fastmd.adapters.negotiate import negotiate

    spec = declared_shard(adapter)
    parallel = ParallelConfig() if parallel is None else parallel
    unsupported = unsupported_settings(config, parallel)
    if unsupported:
        raise ValueError(f"ShardedEvaluator does not run {', '.join(unsupported)}")
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = next(adapter.module.parameters()).dtype
    check_device(device, dtype, world_size)
    pbc = tuple(bool(axis) for axis in atoms.get_pbc())
    skin = float(parallel.topology_skin)
    record = negotiate(adapter, config, entry="distributed", device=device, pbc=pbc, world_size=world_size,
                       shard=spec, skin=skin)
    if record.distributed is None:
        return record
    cutoff = adapter.topology_spec().cutoff
    geometry = plan_geometry(record, dtype)
    # The run's decomposition and owners: the cell and the positions in the model's dtype, as the evaluator, its
    # distribute() and its migrations decide them (on float64 geometry too: on the float32 rounding; an FP64 cell may
    # break the "auto" grid's ties differently: the face spacings of a rotated or a hexagonal cell round differently in
    # another dtype).
    partition = model_partition(atoms.cell.array, dtype, world_size, parallel.domain_grid, pbc)
    if world_size > 1:
        partition.check_image_reach(cutoff + skin)
    positions = torch.as_tensor(atoms.positions, dtype=dtype)
    owner = partition.owner_of(positions)
    matrix = []  # matrix[sender][receiver]: the sender's atoms within the receiver's halo
    for sender in range(world_size):
        rows = torch.nonzero(owner == sender).reshape(-1)
        targets = partition.halo_destinations(positions[rows], cutoff + skin) if rows.numel() else []
        matrix.append([0 if receiver == sender or not targets else int(targets[receiver].numel())
                       for receiver in range(world_size)])
    owned = [int((owner == rank).sum()) for rank in range(world_size)]
    centre, neighbour = neighbor_list("ij", atoms, cutoff)
    centre_owner, neighbour_owner = owner.numpy()[centre], owner.numpy()[neighbour]
    # The edges of the owned centres ("receiver"), or every edge with an owned end ("either").
    either = EDGE_RULES[spec.message_flow] == "either"
    edges = [int(((centre_owner == rank) | (either & (neighbour_owner == rank))).sum()) for rank in range(world_size)]
    ghost_atoms = [sum(matrix[sender][rank] for sender in range(world_size)) for rank in range(world_size)]
    line = spec.message_flow == "line_graph"
    if line:
        from .line_slots import LINE_SLOT_MARGIN

        # The triplets of the owned centres (ordered pairs of their bonds within the line-graph cutoff) and the line
        # slots (boundary bonds within the slots' reach); the edges are those with an owned end (EDGE_RULES).
        first, second, lengths = neighbor_list("ijd", atoms, cutoff)
        own = owner.numpy()
        boundary = own[first] != own[second]
        bonds = np.bincount(first[lengths < float(adapter.topology_spec().triplet_cutoff)], minlength=len(atoms))
        triplets = [int((bonds * (bonds - 1))[own == rank].sum()) for rank in range(world_size)]
        reach = spec.line_slot_cutoff + (skin if device.type == "cuda" else 0.0) + LINE_SLOT_MARGIN
        slot = boundary & (lengths < reach)
        # [rank][peer]: the line slots of each rank pair (each bond once per end), and of every rank.
        slot_matrix = [[int((slot & (own[first] == rank) & (own[second] == peer)).sum()) for peer in range(world_size)]
                       for rank in range(world_size)]
        line_slots = [sum(row) for row in slot_matrix]
    if device.type == "cuda":
        tier = lambda count: _tier(count, 0, TIER_HEADROOM, TIER_ALIGN)
        slots = [max(1, tier(max(owned)))] * world_size
        ghosts = [sum(tier(matrix[sender][rank]) for sender in range(world_size)) for rank in range(world_size)]
        # A line graph keeps a padded pair (edges below the capacity) and 64 sink rows; its triplets at least one.
        capacities = [2 * (tier(max(edges) // 2 + 1) if line else edge_tier(-(-max(edges) // 2), 0, spec.capacity_step))
                      ] * world_size
        sinks, triplet_capacity = 64, [tier(max(max(triplets), 1)) if line else 0] * world_size
        slot_capacity = [sum(tier(count) for count in row) for row in slot_matrix] if line else []
    else:  # the CPU reference topology: exact rows and edges (one unused slot on a rank without atoms)
        slots, ghosts, capacities = [max(1, count) for count in owned], ghost_atoms, edges
        # A line graph: one sink row, its padded pair and padded triplet.
        sinks = 1
        triplet_capacity = [count + 1 for count in triplets] if line else [0] * world_size
        if line:
            capacities = [count + 2 for count in edges]
            slot_capacity = list(line_slots)
    block = record.distributed
    block.update(estimate=True, grid=list(partition.domain_grid),
                 **(line_exchange_schedule(spec, dtype, ghosts, slot_capacity, coordinate_dtype=geometry) if line
                    else exchange_schedule(spec, dtype, ghosts,
                                           flags=4 + (adapter.topology_spec().max_neighbors is not None),
                                           coordinate_dtype=geometry)))
    block["per_rank"] = [{"rank": rank, "owned": owned[rank], "owned_slots": slots[rank],
                          "rows": slots[rank] + ghosts[rank] + (sinks if line else 0), "ghost_rows": ghosts[rank],
                          "ghost_atoms": ghost_atoms[rank], "edges": edges[rank], "edge_capacity": capacities[rank],
                          **({"triplets": triplets[rank], "triplet_capacity": triplet_capacity[rank],
                              "line_slots": line_slots[rank], "slot_capacity": slot_capacity[rank]} if line else {})}
                         for rank in range(world_size)]
    record.memory = {**record.memory, "note": "a dry run measures nothing: each rank measures its own peak at the "
                     "first layout" if device.type == "cuda" else record.memory.get("note", "")}
    return record


def declared_shard(adapter):
    """The adapter's ``ShardSpec``, or ``NegotiationError`` where the node-halo exchange cannot run it: no declared
    shard, an exchange strategy of its own, no energy tap (for exchanges the adapter calls itself: no layer loop of
    its own, ``ShardedTap``), isolated atoms not evaluated by the model."""
    name = getattr(adapter, "name", type(adapter).__name__)
    spec = adapter.shard_spec() if isinstance(adapter, ShardSpecDeclaration) else None
    if spec is None:
        raise NegotiationError(f"model {name!r} declares no ShardSpec: it does not run on the generic node-halo "
                               "exchange")
    if isinstance(adapter, ShardDeclaration) and adapter.custom_strategy() is not None:
        raise NegotiationError(f"model {name!r} has its own exchange strategy: use DistributedMatRISEvaluator")
    if spec.adapter_called:
        if not isinstance(adapter, ShardedTap):
            raise NegotiationError(f"model {name!r}: its ShardSpec has exchanges the adapter calls itself (a pooled "
                                   "readout, or an exchange point without a module), which needs its own layer loop "
                                   "(ShardedTap.sharded_energy)")
        if not isinstance(adapter, (EnergyTap, SystemEnergyTap)):
            raise NegotiationError(f"model {name!r}: one CUDA rank runs the single-GPU engine, which needs an energy "
                                   "tap (node_energy or system_energy)")
    elif not isinstance(adapter, EnergyTap):
        raise NegotiationError(f"model {name!r}: a per-atom readout needs an energy tap (node_energy)")
    if adapter.isolated_atom_policy() != "model":
        raise NegotiationError(f"model {name!r}: the node-halo exchange needs isolated atoms evaluated by the model "
                               "(isolated_atom_policy 'model')")
    if spec.message_flow == "line_graph":
        line = adapter.topology_spec().triplet_cutoff
        # The line slots must reach every bond with triplets (else a boundary bond's sum would stay partial).
        if line is None or spec.line_slot_cutoff < float(line):
            raise NegotiationError(f"model {name!r}: a line-graph shard needs a line graph (TopologySpec."
                                   f"triplet_cutoff {line}) no longer than its line_slot_cutoff "
                                   f"({spec.line_slot_cutoff:g} Å)")
    return spec


def check_device(device: torch.device, dtype: torch.dtype, world_size: int) -> None:
    """P>1 on CUDA runs float32 models (on float32 or float64 geometry, ``InferenceConfig.geometry_dtype``)."""
    if device.type == "cuda" and world_size > 1 and dtype != torch.float32:
        raise ValueError("P>1 on CUDA runs float32 models (float64 geometry for them: "
                         "InferenceConfig.geometry_dtype='float64')")


def plan_geometry(plan, dtype: torch.dtype) -> torch.dtype:
    """The geometry dtype of a negotiated plan: its "geometry" decision (an explicit ``geometry_dtype``, or the
    distributed entry point's default), else the model's ``dtype``."""
    decision = next((decision for decision in plan.decisions if decision.id == "geometry"), None)
    return dtype if decision is None else {"float32": torch.float32, "float64": torch.float64}[decision.chosen]


def model_partition(cell, dtype: torch.dtype, world_size: int, domain_grid, pbc) -> DomainPartition:
    """Build previews and evaluators from the same model-dtype cell on the CPU."""
    return DomainPartition(torch.as_tensor(cell, dtype=dtype, device="cpu"), world_size, domain_grid, pbc)


def _counted(count) -> tuple[dict, dict]:
    """``count()`` (the counting forward: the declared exchanges' calls and shapes), or a failure in its place,
    which the agreement then reports on every rank (a forward that raises on one rank only would leave the others
    in the agreement's collective)."""
    try:
        return count()
    except Exception as exc:  # noqa: BLE001 - agreed on every rank by the caller, never raised alone
        return {f"the counting forward failed: {type(exc).__name__}: {str(exc)[:200]}": 1}, {}


def unsupported_settings(config, parallel: ParallelConfig) -> list[str]:
    """The settings of ``config`` (an ``InferenceConfig`` or None) and ``parallel`` the node-halo exchange does not
    run, each as "name=value": rejected rather than ignored. Honoured: ``execution``, ``optimization_profile``
    (the kernels), ``matmul_precision``, ``geometry_dtype`` (negotiated), ``ParallelConfig.domain_grid`` and
    ``topology_skin``; accepted without
    effect: ``saved_tensor_recompute`` (recorded off: the sharded derivative runs no recompute), ``report`` (no
    report is printed) and ``topology_engine`` (both values run the node-halo topology); ``payload_transport``
    "auto" or "all_to_all" (the COPY is one all_to_all of fixed per-peer blocks)."""
    found = []
    defaults = ParallelConfig()
    for name in ("local_backend", "auto_execution", "exchange_self_rows", "debug_replay_check"):
        if getattr(parallel, name) != getattr(defaults, name):
            found.append(f"ParallelConfig.{name}={getattr(parallel, name)!r}")
    if parallel.payload_transport not in ("auto", "all_to_all"):
        found.append(f"ParallelConfig.payload_transport={parallel.payload_transport!r}")
    if config is not None:
        for name, value in (("checkpoint", "off"), ("isolated_atoms", "reference"), ("triton_dot_precision", None),
                            ("recompute_policy", None)):
            if getattr(config, name) != value:
                found.append(f"InferenceConfig.{name}={getattr(config, name)!r}")
        if config.topology_reuse == "off":
            found.append("InferenceConfig.topology_reuse='off' (the topology always reuses its candidates within "
                         "the skin)")
        if config.capacity != CapacityConfig():
            found.append("InferenceConfig.capacity (the evaluator sizes its own capacities; its skin is "
                         "ParallelConfig.topology_skin)")
        if config.expert_overrides:
            found.append(f"InferenceConfig.expert_overrides {sorted(config.expert_overrides)}")
    return found


def model_fingerprint(module) -> str:
    """sha256 of the module's parameters and buffers (names, dtypes, shapes, bytes): the rank contract's model."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b"")
    return digest.hexdigest()


class _SingleSystem:
    """One CUDA rank: the single-GPU engine's builder and derivative (no ghost rows, no hook); its edge capacity in
    the tiers of the P>1 topology (``capacity_step``: the shard's declared ones, else the topology's).
    ``positions_dtype`` float64: float64 geometry of a float32 model (``cell`` then the float64 cell)."""

    def __init__(self, adapter, cell: Tensor, *, skin: float, device, capacity_step=None,
                 positions_dtype: torch.dtype | None = None):
        from fastmd.engine.energy_derivative import EnergyDerivative, edges_only_builder_options

        self.adapter, self.device, self.skin = adapter, device, float(skin)
        self.capacity_step = capacity_step
        self.spec = adapter.topology_spec()
        # A model with a line graph: the single-system builder's line graph, positions published (its leaf).
        self.line = self.spec.triplet_cutoff is not None
        self.options = {"publish_positions": True} if self.line else edges_only_builder_options(adapter)
        if positions_dtype is not None:
            self.options = {**self.options, "positions_dtype": positions_dtype}
        self.dtype = self.options.get("positions_dtype", torch.float32)
        self.cell = cell
        self.derivative = EnergyDerivative(adapter)
        # The builds' running maximum of an atom's edges, for a model with a neighbour cap (a fifth step flag).
        self.neighbor_count = (None if self.spec.max_neighbors is None
                               else torch.zeros(1, dtype=torch.int64, device=device))
        if self.neighbor_count is not None:
            self.options = {**self.options, "max_neighbor_count": self.neighbor_count}
        self.builder = None
        self.u_capacity = self.t_capacity = 0
        self.n_owned = 0
        self.layouts = 0
        self._numbers = None

    def refresh(self, state: AtomState) -> AtomState:
        from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder

        state = state.sorted()
        positions = state.positions.detach().to(device=self.device, dtype=self.dtype)
        numbers = state.numbers.to(device=self.device)
        if self.builder is None or self._numbers is None or not torch.equal(numbers, self._numbers):
            self.builder, self.u_capacity = None, 0
        while True:
            if self.builder is None:
                self.builder = FixedCapacityGraphBuilder(
                    cell=self.cell.to(self.dtype), atomic_numbers=numbers, u_capacity=max(1, self.u_capacity),
                    t_capacity=max(1, self.t_capacity) if self.line else 0, atom_graph_cutoff=self.spec.cutoff,
                    line_graph_cutoff=float(self.spec.triplet_cutoff) if self.line else None,
                    candidate_cutoff=self.spec.cutoff + self.skin, padding=self.adapter.padding_rule(),
                    prepare=self.adapter.prepare, device=self.device, **self.options)
                self.positions = torch.zeros((len(numbers), 3), dtype=self.dtype, device=self.device)
                self.layouts += 1
            self.positions.copy_(positions)
            self.builder.refresh_candidates(self.positions)
            _, status = self.builder.build(self.positions, rebuild_candidates=False)
            edges = int(status.edge_count.item())
            if not bool(status.candidate_capacity_valid.item()):
                raise RuntimeError(f"a candidate row exceeds the builder's neighbor buffer ({self.builder.buffer_max})")
            if not self.line and edges <= self.builder.e_capacity:
                break
            triplets = int(status.triplet_count.item()) if self.line else 0
            # A line graph keeps a padded pair (its padded triplets use one).
            if self.line and edges < self.builder.e_capacity and triplets <= self.builder.t_capacity:
                break
            self.u_capacity = (_tier(-(-edges // 2) + 1, self.u_capacity, 0.125, 32) if self.line
                               else edge_tier(-(-edges // 2), self.u_capacity, self.capacity_step))
            if self.line:
                self.t_capacity = _tier(max(triplets, 1), self.t_capacity, 0.125, 32)
            self.builder = None
        self._numbers = numbers
        self.n_owned = len(state.ids)
        return state

    def set_positions(self, positions: Tensor) -> None:
        self.positions.copy_(positions)

    def step(self, stress: bool = False):
        """One evaluation at the written positions (device only): ``(energy [1], forces [n, 3], flags [4],
        stress [3, 3] or None)``."""
        with torch.no_grad():
            graph, status = self.builder.build(self.positions, rebuild_candidates=False)
        out = self.derivative(self.builder.batch, graph, task="efs" if stress else "ef", n_real=self.n_owned)
        energy, forces = out["e"], out["f"][0][: self.n_owned]
        virial = out["s"][0] if stress else None
        finite = torch.isfinite(energy).all() & torch.isfinite(forces).all()
        if stress:
            finite = finite & torch.isfinite(virial).all()
        flags = torch.cat((status.overflow.reshape(-1)[:1].long(), (~status.candidate_valid.reshape(-1)[:1]).long(),
                           (~finite).long().reshape(1), torch.zeros(1, dtype=torch.long, device=energy.device),
                           *(() if self.neighbor_count is None else (self.neighbor_count,))))
        return energy, forces, flags, virial

    @property
    def key(self):
        return (self.layouts, id(self.builder))

    def extents(self) -> dict:
        """The sizes of the current builder (the memory admission's extents; no ghost rows)."""
        extents = {"owned": self.n_owned, "owned_slots": self.n_owned, "rows": int(self.positions.shape[0]),
                   "ghost_rows": 0, "send_rows": 0, "edges": int(self.builder.e_capacity)}
        if self.line:
            extents.update(triplets=int(self.builder.t_capacity), slots=0)
        return extents

    def close(self):
        self.builder = None


class ShardedEvaluator:
    """Spatial decomposition of one periodic system for a model with a ``ShardSpec`` (see the module docstring).

    ``model``: the adapter (``fastmd.adapters.load_adapter``), with an energy
    tap (``node_energy``; a model that calls its exchanges itself: its own
    layer loop, ``sharded_energy``, and an energy tap of either kind for one
    CUDA rank), isolated atoms evaluated by the model itself, and a
    ``ShardSpec`` (``shard_spec()``). All methods that communicate must be
    called by every rank in the same order. ``evaluate(state, stress=False)``
    returns the owned atoms' forces of the (possibly migrated) state it
    returns, and with ``stress`` the stress; ``config.execution``: "auto"
    (default: model_graph where every rank's predicted capture peak fits),
    "model_graph" (one captured evaluation step per layout) or "eager".
    Settings the node-halo exchange does not run are rejected
    (:func:`unsupported_settings`). ``geometry_dtype`` and ``geometry_cell``:
    the geometry the plan runs (see the module docstring) and its cell (the
    caller's, in float64 on float64 geometry; else the partition's).
    """

    # The scatter of the initial structure or of a restart and the diagnostic gather, as DistributedMatRISEvaluator's
    # (they read the context, partition, dtype and device only).
    distribute = DistributedMatRISEvaluator.distribute
    distribute_restart = DistributedMatRISEvaluator.distribute_restart
    gather = DistributedMatRISEvaluator.gather

    _md_graph = None  # (set in __init__; a class default for a close() after a failed construction)
    # Refreshes one in-graph MD step may take (a look-ahead or first one, then redos), as DistributedMatRISEvaluator's.
    _MD_STEP_REFRESHES = 4
    # The in-graph MD step's look-ahead: refresh before the step whose largest displacement since the refresh,
    # extrapolated by this many times the last step's increase, would exceed half the skin (a violation is still
    # caught after the step and redone). One step: an epoch then ends one step before the out-of-graph path's, which
    # refreshes at the step that leaves the skin (EpochTopology's three steps made the epochs about a quarter as long
    # at 1,500 K).
    _MD_LOOKAHEAD_STEPS = 1
    # After a step that stays stale right after its refresh (its own drift exceeds half the skin) and runs outside the
    # graph, this many more steps run outside before the in-graph step is tried again (its refresh and capture).
    _MD_FALLBACK_STEPS = 16

    def __init__(self, model, cell, *, pbc=(True, True, True), config=None, parallel: ParallelConfig | None = None,
                 context: ExchangeContext | None = None):
        from fastmd.adapters import as_adapter

        self.context = ExchangeContext() if context is None else context
        if not isinstance(self.context, ExchangeContext):
            raise TypeError("context must be an ExchangeContext")
        self._engine = None
        self._graphs = {}
        self.hooks = None
        error = None
        try:
            self.adapter = adapter = as_adapter(model)
            name = getattr(adapter, "name", type(adapter).__name__)
            self.spec = spec = declared_shard(adapter)
            self._line = spec.message_flow == "line_graph"
            self.parallel = ParallelConfig() if parallel is None else parallel
            if not isinstance(self.parallel, ParallelConfig):
                raise TypeError("parallel must be ParallelConfig")
            unsupported = unsupported_settings(config, self.parallel)
            if unsupported:
                raise ValueError(f"ShardedEvaluator does not run {', '.join(unsupported)} (the node-halo exchange "
                                 "has no such option): leave them at their defaults")
            module = adapter.module
            parameter = next(module.parameters())
            self.device, self.dtype = parameter.device, parameter.dtype
            world = self.context.world_size
            self.single = world == 1 and self.device.type == "cuda"
            check_device(self.device, self.dtype, world)
            if self.device.type == "cuda":
                if self.context.backend == "gloo":
                    raise ValueError("the Gloo data backend requires a CPU model")
            elif self.context.backend == "nccl":
                raise ValueError("NCCL requires a CUDA model")
            self.partition = model_partition(cell, self.dtype, world, self.parallel.domain_grid, pbc)
            self.skin = float(self.parallel.topology_skin)
            self.cutoff = float(adapter.topology_spec().cutoff)
            if world > 1:
                self.partition.check_image_reach(self.cutoff + self.skin)
            self._config, self._pbc = config, tuple(bool(axis) for axis in pbc)
            from fastmd.engine import memory_admission
            self._admission_on = self.device.type == "cuda" and memory_admission.enabled()
            # Each rank's plan (no collective), then agreed: the kernels below, everything else in the contract.
            plan = self._negotiate(None)
            self.pins = [] if self.single else spec.check_pins(name)
        except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
            error = str(exc)
        collective_check(self.context, error)
        self.plan = self._agree_kernels(plan)
        # The geometry (InferenceConfig.geometry_dtype, else the distributed entry point's default; the plan's
        # "geometry" decision): float64 geometry of a float32 model keeps the positions, the cell (the caller's, in
        # float64), the coordinate COPY, the edge vectors, the strain, the forces and the stress in float64; the
        # partition, the ownership and the halo selection stay in the model's dtype, and on CUDA the neighbour search
        # too (the CPU reference topologies decide their edges in float64).
        self.geometry_dtype = plan_geometry(self.plan, self.dtype)
        self._float64_geometry = self.geometry_dtype != self.dtype
        self.geometry_cell = (torch.as_tensor(cell, dtype=self.geometry_dtype, device="cpu").detach().clone()
                              if self._float64_geometry else self.partition.cell)
        execution = self.plan.decision("execution")
        self.execution = execution.chosen
        self.execution_auto = execution.evidence.get("requested") == "auto"
        # The modes auto admits at each layout: model_graph only where the negotiation found the model capturable.
        self._auto_modes = ("model_graph", "eager") if execution.chosen == "model_graph" else ("eager",)
        if not self._admission_on and self.device.type == "cuda":
            self.plan.memory = {**self.plan.memory, "calibration": "off",
                                "note": "MATRIS_MEMORY_ADMISSION=off: no memory admission, no trial allocation"}
            self.plan.decisions = [dataclasses.replace(d, evidence={**d.evidence, "memory": "admission off"})
                                   if d.id == "execution" else d for d in self.plan.decisions]
        self.matmul_precision = self.plan.matmul_precision or "highest"
        # After the kernels' installation, which may replace modules of the model (the hooks go on those).
        self.model = self.adapter.module
        self.fingerprint = model_fingerprint(self.model)
        contract = {
            "adapter": self.adapter.name, "kernel": getattr(self.adapter, "kernel", None),
            "model": self.fingerprint, "plan": self.plan.fingerprint, "shard_spec": self.spec.digest(),
            "pins": [(outcome.record.name, outcome.record.source_sha256) for outcome in self.pins],
            "kernels": {d.id: d.chosen for d in self.plan.decisions if d.id.startswith("kernels.")},
            "cell": self.partition.cell.tolist(), "grid": list(self.partition.domain_grid), "dtype": str(self.dtype),
            "execution": self.execution, "execution_auto": self.execution_auto, "skin": self.skin,
            "pbc": list(self._pbc), "memory_admission": self._admission_on, "matmul_precision": self.matmul_precision,
            "precision": {key: self.plan.precision.get(key) for key in ("chosen", "cublas", "device")},
            "engine": self._engine_name(),
            "capture_warmups": _CAPTURE_WARMUPS,
            "deterministic": deterministic_mode(),
            **({"geometry_dtype": str(self.geometry_dtype), "geometry_cell": self.geometry_cell.tolist()}
               if self._float64_geometry else {}),
        }
        agree_rank_contract(self.context, contract)
        self.rank_contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str)
                                                   .encode()).hexdigest()
        error = None
        self.derivative = None  # one CUDA rank: the single-GPU engine's own (_SingleSystem)
        try:
            hooked = tuple(point for point in self.spec.exchange_points if point.module is not None)
            if not self.single and hooked:
                hooks = LineExchangeHooks if self._line else ExchangeHooks
                self.hooks = hooks.acquire(self.model, hooked)
            if not self.single:
                # The CPU reference's (a node-halo derivative resolves the declared ghost-row trimming's modules: a
                # name that matches none refuses the evaluator on every rank, as does the CUDA strategy's own).
                self.derivative = (LinePartialDerivative(self.adapter, self.spec.exchange_points) if self._line
                                   else ShardedEnergyDerivative(self.adapter, self.spec))
            # Float64 geometry of a float32 model: the engine's coordinates in float64, over the float64 cell.
            geometry = ({"positions_dtype": self.geometry_dtype, "cell": self.geometry_cell}
                        if self._float64_geometry else {})
            if self.single:
                self._engine = _SingleSystem(self.adapter, self.geometry_cell, skin=self.skin, device=self.device,
                                             capacity_step=self.spec.capacity_step,
                                             positions_dtype=geometry.get("positions_dtype"))
            elif self.device.type == "cuda":
                strategy = LinePartialStrategy if self._line else NodeHaloStrategy
                self._engine = strategy(self.adapter, self.spec, self.partition, self.context, skin=self.skin,
                                        device=self.device, **geometry)
            elif self._line:
                self._engine = LineReferenceTopology(self.partition, self.context, cutoff=self.cutoff,
                                                     line_cutoff=float(self.adapter.topology_spec().triplet_cutoff),
                                                     slot_cutoff=self.spec.line_slot_cutoff,
                                                     dtype=self.geometry_dtype, pad_species=self.spec.pad_species,
                                                     cell=geometry.get("cell"))
            else:
                self._engine = NodeHaloReferenceTopology(self.partition, self.context, cutoff=self.cutoff,
                                                         dtype=self.geometry_dtype, pad_species=self.spec.pad_species,
                                                         edges=EDGE_RULES[self.spec.message_flow],
                                                         cell=geometry.get("cell"))
        except (ValueError, TypeError, RuntimeError) as exc:
            error = str(exc)
        try:
            collective_check(self.context, error)
        except BaseException:
            if self.hooks is not None:
                self.hooks.release()
                self.hooks = None
            raise
        self.epoch = 0
        self.refreshes = 0
        self.line_slot_refreshes = 0  # refreshes a line-graph shard's slot flag asked for (none within the skin)
        self.closed = False
        self._ids = self._numbers = None
        self.reference_energy = None
        self._key = None
        self._graphs = {}  # stress (bool) -> (the captured step of the current layout, its outputs): one at most
        # The in-graph MD step (md_step): (its key, graph, outputs) or None; the captured graphs are one at most in
        # all (this or one of _graphs). The MD driver's hook it serves, the refresh count at the hook's install, the
        # stress once asked for, the last agreed displacement since the refresh and the look-ahead's verdict.
        self._md_graph = self._md = self._md_refreshes = self._md_displacement = None
        self._md_stress = self._md_ahead = False
        self._md_outside = 0  # steps still to run outside after a fallback (_MD_FALLBACK_STEPS)
        self.md_steps = self.md_redos = self.md_overflows = 0  # in-graph MD steps, their redos, the overflows'
        self._pool = None
        self._agreed_exchanges = self.single
        self.natoms = 0
        self.last_stats = {}
        self._predictor = self._admission = None
        self._stream = None  # the side stream of the warm-ups and of the memory probe (one: no new cuBLAS workspaces)
        self.predicted_peak = None  # this rank's admitted prediction (bytes above the allocated baseline) or None
        # The most neighbours of an atom in the evaluations so far (all ranks), for a model with a neighbour cap.
        self.max_neighbors = self.adapter.topology_spec().max_neighbors
        self._neighbor_count = 0
        self._neighbor_cap_warned = False
        shutdown_registry.register(self, "evaluators")

    # --- negotiation (collective where it agrees) -----------------------------------------------------------

    def _negotiate(self, agreed):
        """This rank's plan (``negotiate(entry="distributed")``; local), the kernels from ``agreed`` on."""
        from fastmd.adapters.negotiate import negotiate

        return negotiate(self.adapter, self._config, entry="distributed", device=self.device, pbc=self._pbc,
                         world_size=self.context.world_size, shard=self.spec, skin=self.skin, agreed_kernels=agreed)

    def _agree_kernels(self, plan):
        """``plan`` once every rank chose the same option at every kernel site (collective). Otherwise every rank
        starts each site's chain at the latest of the ranks' choices, past every option a rank failed to install,
        and negotiates again: an option that fails on one rank falls back on every rank (a round per disagreement,
        bounded by the chains)."""
        while self.context.world_size > 1:
            local = {d.id: (d.chosen, list(d.chain), sorted((d.evidence or {}).get("failed", {})))
                     for d in plan.decisions if d.id.startswith("kernels.")}
            gathered = self.context.all_gather_object(local)
            if any(set(other) != set(local) for other in gathered):
                raise NegotiationError("the ranks' models declare different kernel sites")
            if all(other[site][0] == chosen for other in gathered for site, (chosen, _, _) in local.items()):
                return plan
            agreed = {}
            for site, (_, chain, _) in local.items():
                failed = {option for other in gathered for option in other[site][2]}
                start = max(chain.index(other[site][0]) if other[site][0] in chain else len(chain) - 1
                            for other in gathered)
                options = [option for option in chain[start:] if option not in failed] or chain[-1:]
                agreed[site.removeprefix("kernels.")] = options[0]
            error = None
            try:
                plan = self._negotiate(agreed)
            except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
                error = str(exc)
            collective_check(self.context, error)
        return plan

    # --- evaluation ---------------------------------------------------------------------------------------

    def evaluate(self, state: AtomState, *, stress: bool = False) -> DistributedPrediction:
        """Energy and the owned atoms' forces of ``state`` (collective), and with ``stress`` (the same on every
        rank; a cell periodic in all three directions) the stress of the whole system (``prediction.stress``:
        ``[3, 3]`` in eV/Å³, the same on every rank). Returns the state the forces belong to: ``state`` itself, or
        the migrated state after a refresh (a new object)."""
        if self.closed:
            raise RuntimeError("ShardedEvaluator: evaluate after close() (its hooks are removed)")
        if not isinstance(stress, bool):
            raise TypeError("ShardedEvaluator.evaluate: stress must be a bool")
        if stress and not all(self._pbc):
            raise ValueError("ShardedEvaluator.evaluate: the stress needs a cell periodic in all three directions")
        with matmul_precision_scope(self.matmul_precision):
            if self.device.type == "cpu":
                return self._evaluate_reference(state, stress)
            return self._evaluate_cuda(state, stress)

    def _refresh(self, state: AtomState, floor: dict | None = None) -> AtomState:
        started = time.perf_counter()
        engine = self._engine
        # (floor: the in-graph MD step's capacity floor, P>1 on CUDA only.)
        state = engine.refresh(state) if floor is None else engine.refresh(state, floor)
        self.refreshes += 1
        self._totals(state)
        if engine.key != self._key:
            # New buffers (the same event on every rank): the captured step of the old ones goes, and the new
            # layout's memory is admitted before any collective of its steps (a failed admission keeps the old key:
            # the next evaluation is admitted again).
            self._release_graph()
            self._admit()
            self._key = engine.key
        if not self._agreed_exchanges:
            # Before the first forward that exchanges (collective).
            if self._line:
                engine.agree(*_counted(engine.count))
                self._agreed_exchanges = True
            else:
                agree_exchanges(self.context, self.spec, *_counted(engine.count), engine.layout.rows)
                self._agreed_exchanges = self._trimming_checked(engine.layout.ghost_rows)
        self._ids, self._numbers = state.ids, state.numbers
        self.last_stats["refresh_ms"] = (time.perf_counter() - started) * 1000
        return state

    def _trimming_checked(self, ghost_rows: int) -> bool:
        """Whether the counting forward that just ran (and was agreed) checked the declared ghost-row trimming
        (collective where trimming is declared): always without one; with one, once a rank had ghost rows (a rank
        without them trims nothing, so until then the counting forward runs again at each refresh, before a step in
        which a wrong declaration would raise on the ranks with ghost rows only)."""
        if not (self.spec.owned_slot_inputs or self.spec.zero_ghost_outputs):
            return True
        if self.context.world_size == 1:
            return ghost_rows > 0
        return any(self.context.all_gather_object(ghost_rows > 0))

    def _admit(self) -> None:
        """The memory admission of a new layout (collective; CUDA). This rank's predicted peaks of the candidate
        modes (its ``NodeHaloMemoryPredictor``, measured on its device at the first layout and again where the
        capacities outgrow the measured range) are gathered with every rank's available memory
        (``MemoryAdmission``: every rank decides on the same data; a requested mode that does not fit raises
        ``MemoryAdmissionError`` on every rank). Under auto the mode is model_graph where it fits on every rank,
        else eager. Then every rank allocates the chosen mode's predicted peak once, and the outcomes are gathered
        in one round: a rank that cannot allocate it fails the group with a ``NegotiationError`` here, before
        any collective of the step (an out-of-memory error inside a step's collectives would leave the other ranks
        waiting); so does a rank whose memory probe ran out of memory. ``MATRIS_MEMORY_ADMISSION=off`` (read at
        construction, agreed in the rank contract): no admission, no trial. Side effects: the probe resets the
        device's peak memory statistics, and the trial raises ``max_memory_allocated`` to the baseline plus the
        prediction."""
        from fastmd.engine import memory_admission

        if not self._admission_on:
            return
        extents = self._engine.extents()
        modes = self._auto_modes if self.execution_auto else (self.execution,)
        kinds = tuple(KINDS[mode] for mode in modes)
        if self._predictor is None or not self._predictor.covers(extents, kinds):
            # A line graph's predictor adds the triplets (its dominant term).
            predictor = LineGraphMemoryPredictor if "triplets" in extents else NodeHaloMemoryPredictor
            self._predictor = predictor.measure(self.adapter, self.spec, extents, cell=self.partition.cell,
                                                device=self.device, dtype=self.dtype, kinds=kinds,
                                                stream=self._side_stream(), single=self.single,
                                                geometry_dtype=self.geometry_dtype if self._float64_geometry else None)
            if self._admission is None:
                self._admission = memory_admission.MemoryAdmission(device=self.device, predictor=self._predictor)
            self._admission.predictor = self._predictor
        fits = self._admission.check(modes, event=f"layout {self._engine.key[0]}",
                                     requested=None if self.execution_auto else self.execution,
                                     context=self.context if self.context.world_size > 1 else None, extents=extents)
        mode = ("model_graph" if "model_graph" in fits else "eager") if self.execution_auto else self.execution
        predicted = self._predictor.predict("off", mode, extents)
        error = None
        if self._predictor.failure is not None and "OutOfMemoryError" in self._predictor.failure:
            # A rank whose probe ran out of memory would run out of memory again inside the step's collectives.
            error = self._predictor.failure
        elif predicted is not None:
            torch.cuda.empty_cache()  # the released graphs' pool
            try:
                trial = torch.empty(predicted, dtype=torch.uint8, device=self.device)
                del trial
            except RuntimeError as exc:  # out of memory (or any failure here: agreed below, never raised alone)
                error = (f"the trial allocation of this rank's predicted peak ({predicted / 2**30:.2f} GiB, {mode}) "
                         f"failed: {str(exc).splitlines()[0][:200]}")
            torch.cuda.empty_cache()
        errors = self.context.all_gather_object(error) if self.context.world_size > 1 else [error]
        if any(errors):
            raise NegotiationError("memory admission: " + "; ".join(f"rank {rank}: {message}" for rank, message
                                                                    in enumerate(errors) if message))
        if mode != self.execution:
            self._release_graph()
            self.execution = mode
        self.predicted_peak = predicted
        self.plan.decisions = [self._admitted(decision, mode, fits) if decision.id == "execution" else decision
                               for decision in self.plan.decisions]

    def _admitted(self, decision, mode: str, fits) -> object:
        """The plan's execution decision after an admission: the mode, why, and every rank's numbers (all from the
        gathered data: the same on every rank)."""
        by_rank = self._admission.decisions[-1]["by_rank"]
        evidence = {**decision.evidence, "memory": [
            {"rank": rank, "predicted_gib": {key: None if value is None else round(value / 2**30, 3)
                                             for key, value in entry["predicted"].items()},
             "budget_gib": round((entry["available"] - entry["margin"]) / 2**30, 3)}
            for rank, entry in enumerate(by_rank)], "fits": sorted(fits)}
        unpredicted = [rank for rank, entry in enumerate(by_rank) if entry["predicted"].get(mode) is None]
        if not self.execution_auto:
            reason = "explicit request; " + (f"no prediction on rank(s) {unpredicted}: admitted unchecked"
                                             if unpredicted else "admitted on every rank")
        elif mode == "model_graph":
            reason = "auto: capturable, and every rank's predicted capture peak fits its budget"
        elif self._auto_modes == ("eager",):
            reason = decision.reason
        else:
            missing = [rank for rank, entry in enumerate(by_rank) if entry["predicted"].get("model_graph") is None]
            reason = ("auto: " + (f"no capture prediction on rank(s) {missing}" if missing
                                  else "a rank's predicted capture peak does not fit its budget") + ": eager")
        return dataclasses.replace(decision, chosen=mode, reason=reason, evidence=evidence)

    def _side_stream(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device)
        return self._stream

    def _totals(self, state: AtomState) -> None:
        """The global atom count and reference energy (the sum of the adapter's per-element reference energies,
        as the single-GPU engine reports it; 0 without a table) of a refreshed state (collective)."""
        table = self.adapter.reference_energies()
        device = state.ids.device
        reference = (torch.zeros((), dtype=torch.float64, device=device) if table is None else
                     table[0].to(device)[state.numbers.long() - 1].to(torch.float64).sum())
        totals = torch.stack((torch.tensor(float(len(state.ids)), dtype=torch.float64, device=device), reference))
        if self.context.world_size > 1:
            dist.all_reduce(totals, group=self.context.group)
        natoms, reference = totals.cpu().tolist()
        self.natoms = int(natoms)
        self.reference_energy = torch.tensor(reference, dtype=torch.float32, device=device)

    def _evaluate_cuda(self, state: AtomState, stress: bool) -> DistributedPrediction:
        entered = time.perf_counter()
        self.last_stats = {}
        engine = self._engine
        refreshed = False
        # The refreshed state's own tensors (the MD driver hands them back), else a new state: a refresh. A rank
        # decides it locally, so at P>1 the decision rides in the step's flags (MAX over the ranks): a rank with a
        # new state runs the step at its old positions and every rank then refreshes.
        new = (self._ids is None or state.ids is not self._ids or state.numbers is not self._numbers
               or state.positions.shape != (engine.n_owned, 3) or state.positions.device != self.device)
        if self._ids is None or (new and self.single):
            state, refreshed = self._refresh(state), True
        elif not new:
            engine.set_positions(state.positions.detach())
        # Judge P>1 by this step's neighbours, including when an old layout runs before a refresh.
        counter = None if self.single else getattr(engine, "neighbor_count", None)
        if not self.single:
            engine.new_state.fill_(int(new and not refreshed))
            if counter is not None:
                counter.zero_()
        energy, forces, flags, virial = self._run(stress)
        verdict = flags.cpu().tolist()
        # A line-graph shard's fifth flag: its line slots did not cover the step (also in the stale flag).
        slots = self._line and len(verdict) > _SLOTS and bool(verdict[_SLOTS])  # (one CUDA rank: four flags)
        if not refreshed and (verdict[_OVERFLOW] or verdict[_STALE] or verdict[_NEW_STATE]):
            self.line_slot_refreshes += int(slots)
            # Every rank sees the same reduced flags: all refresh at the current positions and redo the step. The
            # outputs of a captured step live in its graph's pool: drop them before the refresh, which may replace the
            # graph, so that a new capture into the pool finds none of its blocks in use.
            del energy, forces, flags, virial
            state, refreshed = self._refresh(state), True
            if not self.single:
                engine.new_state.fill_(0)
                if counter is not None:
                    counter.zero_()
            energy, forces, flags, virial = self._run(stress)
            verdict = flags.cpu().tolist()
        if verdict[_OVERFLOW] or verdict[_STALE] or verdict[_NEW_STATE]:
            raise RuntimeError(f"node-halo step flags after a refresh: overflow {verdict[_OVERFLOW]}, "
                               f"stale candidates {verdict[_STALE]}" + (
                                   ", line slots not covering the short boundary bonds" if self._line
                                   and len(verdict) > _SLOTS and verdict[_SLOTS] else ""))
        if verdict[_NONFINITE]:
            raise ValueError("distributed evaluation: non-finite energy or forces (on some rank)")
        if self.max_neighbors is not None:
            # The reduced flags describe this evaluation; the host maximum is only for reporting.
            largest = int(verdict[_NEIGHBORS])
            self._neighbor_count = max(self._neighbor_count, largest)
            self._refuse_neighbor_cap(largest)
            self.neighbor_cap_summary()  # warns once when an atom exceeds the model's neighbour cap
        n = engine.n_owned
        energy = energy.detach().reshape(()).clone()
        forces = forces[:n].detach().clone()
        self.epoch += 1
        self.last_stats.update({"evaluate_ms": (time.perf_counter() - entered) * 1000, "refreshed": refreshed,
                                "refreshes": self.refreshes, "captured": bool(self._graphs),
                                "communication": dict(self.context.comm.stats) if hasattr(self.context.comm, "stats")
                                else {}})
        return self._prediction(state, forces, energy, None if virial is None else virial.detach().clone())

    def _prediction(self, state, forces, energy, stress=None) -> DistributedPrediction:
        return DistributedPrediction(state, forces, energy, energy / max(self.natoms, 1), self.reference_energy,
                                     self.natoms, self.epoch, dict(self.last_stats), stress)

    def _run(self, stress: bool):
        """The step at the written positions: eager, or the captured graph of the current layout. One graph per
        layout, as admitted: the step with the stress once the stress was asked for (it returns the energy and
        forces too; its strain adds a few rows of work), else the one without."""
        engine = self._engine
        if self.execution == "eager":
            return engine.step(stress)
        if stress and False in self._graphs:
            self._release_graph()
        key = stress or True in self._graphs
        if key not in self._graphs:
            self._capture(key)
        graph, (energy, forces, flags, virial) = self._graphs[key]
        graph.replay()
        return energy, forces, flags, virial if stress else None

    def _capture(self, stress: bool) -> None:
        """Capture the step (after a warm-up on a side stream; both collective). A capture that fails on one rank
        (out of memory, a capture invalidated by another thread) fails on every rank before any replay."""
        from fastmd.engine._graph_capture import capture_graph

        engine = self._engine
        started = time.perf_counter()
        if self._md_graph is not None:
            self._release_graph()  # one captured graph in all: the in-graph MD step's goes
        torch.cuda.empty_cache()
        stream = self._side_stream()
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(_CAPTURE_WARMUPS):
                engine.step(stress)  # collective: the warm-up count is part of the rank contract
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        gc.collect()
        graph, outputs, error = torch.cuda.CUDAGraph(), None, None
        try:
            # Reuse this evaluator's warmed stream. A shared default capture
            # stream can make cuBLAS workspaces belong to another graph's pool.
            with capture_graph(graph, pool=self._pool, stream=stream, check_replay=False):
                outputs = engine.step(stress)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            # capture_end may mask the operator that invalidated the stream.
            # Preserve the complete exception chain before the collective
            # turns each rank's failure into a validation error.
            error = f"capture failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        self.context.collective_check(error)
        self._graphs[stress] = (graph, outputs)
        self.last_stats["capture_ms"] = (time.perf_counter() - started) * 1000

    def _release_graph(self) -> None:
        if self._graphs or self._md_graph is not None:
            # A queued replay may still read the graph's private allocations.
            torch.cuda.synchronize(self.device)
            self._graphs = {}
            self._md_graph = None
            gc.collect()

    # --- the in-graph MD step ----------------------------------------------------------------------------------

    def md_step(self, md, *, stress: bool = False) -> DistributedPrediction | None:
        """One MD step with the integrator inside the captured step (collective; P>1 on CUDA, model_graph).

        ``md`` is the evaluator-driven MD driver's hook
        (``GPUResidentMolecularDynamics.from_evaluator``, the hook of
        ``DistributedMatRISEvaluator.md_step``): ``refresh_state()`` (the
        committed state q_n, its payload also carrying the cached forces
        F(q_n)), ``install(state, rows)``, ``capture_key()``, the captured
        halves ``pre(rows)`` / ``post(forces, energy, rows)`` / ``flags(rows)``
        on the topology's owned slots, and ``restore()`` / ``state()``.

        A step is one replay of a graph that runs the integrator's pre-force
        half into the owned slots, the evaluation step of ``evaluate`` (ghost
        COPY, exact filter, the model with its exchanges, the backward, the
        energy's sum) and the post-force half, then the step flags' MAX (NCCL
        inside the graph), and one download of the flags. A refresh runs
        BEFORE the step, at the committed positions q_n (the first step, a
        refresh by ``evaluate`` since, the look-ahead: the agreed displacement
        since the refresh, extrapolated, would exceed half the skin): the
        migration with the cached forces, the topology, the memory admission
        of a new layout, the integrator's reorder into slots of the owned
        capacity, then a capture where the key (the layout, the hook's
        integrator buffers, the stress) changed. A step that must be discarded
        (a capacity overflow, candidates the skin no longer covers, line slots
        that do not cover it) restores the integrator to q_n (the backup the
        graph took) and is redone after a refresh there; an overflow grows the
        capacities to at least the discarded step's own counts (the capacity
        floor), retried while that agreed capacity grows. Non-finite step
        positions, energy, forces or integrator state, and (with several
        ranks) an atom with more neighbours than the model's cap in the step's
        build (the captured step counts from zero), restore the integrator,
        then raise on every rank.

        Returns None where this evaluator does not run the captured P>1 step
        (one CUDA rank, the CPU reference, eager; also after execution="auto"
        chose eager at this step's refresh) and when the step leaves its
        candidates stale right after a refresh at q_n (its own drift exceeds
        half the skin; the integrator is restored): the driver then runs the
        step with the integrator outside, around ``evaluate`` (which refreshes
        at q_{n+1}), and the next ``_MD_FALLBACK_STEPS`` steps too. ``stress``
        (the same on every rank): the prediction carries the stress; from then
        on the captured step computes it.
        """
        if self.closed:
            raise RuntimeError("ShardedEvaluator: md_step after close() (its hooks are removed)")
        if not isinstance(stress, bool):
            raise TypeError("ShardedEvaluator.md_step: stress must be a bool")
        if self.single or self.device.type != "cuda" or self.execution != "model_graph":
            return None
        if stress and not all(self._pbc):
            raise ValueError("ShardedEvaluator.md_step: the stress needs a cell periodic in all three directions")
        if self._md_outside:  # after a step that ran outside (agreed: the same count on every rank)
            self._md_outside -= 1
            return None
        with matmul_precision_scope(self.matmul_precision):
            return self._md_step(md, stress)

    def _md_step(self, md, stress: bool) -> DistributedPrediction | None:
        entered = time.perf_counter()
        self.last_stats = {}
        engine = self._engine
        floor = overflowed = None
        attempts = refreshes = 0
        refresh_ms = capture_ms = 0.0
        while True:
            attempts += 1
            refreshed = (attempts > 1 or md is not self._md or self._md_ahead or self.refreshes != self._md_refreshes)
            if refreshed:
                refreshes += 1
                if refreshes > self._MD_STEP_REFRESHES:
                    raise RuntimeError(f"ShardedEvaluator: the in-graph MD step was discarded after "
                                       f"{self._MD_STEP_REFRESHES} refreshes")
                self._md_refresh(md, floor)
                refresh_ms += self.last_stats.get("refresh_ms", 0.0)
                if overflowed is not None and not any(
                        grown > before for grown, before in zip(engine.topology.capacities(), overflowed)):
                    # (The capacities are the tiers of the MAX demand over the ranks: the same verdict on every rank.)
                    raise RuntimeError("ShardedEvaluator: the in-graph MD step still overflows after a capacity "
                                       "refresh")
                if self.execution != "model_graph":  # execution="auto" chose eager at this refresh
                    return None
            # The stress once asked for (the same on every rank), then on every later capture.
            restressed = stress and not self._md_stress
            self._md_stress = self._md_stress or stress
            key = (engine.key, md.capture_key(), self._md_stress)
            current = self._md_graph is not None and self._md_graph[0] == key
            if refreshed or restressed or self._md_graph is None:
                # Agreed events (a refresh, the stress, a released graph): every rank captures if any rank's key
                # changed (its own integrator buffers may have grown), the warm-up and capture being collective.
                capture = any(self.context._collective(not current, "sharded.md_capture")
                              if self.context.world_size > 1 else [not current])
                if capture:
                    self._capture_md(md, key)
                    capture_ms += self.last_stats.get("capture_ms", 0.0)
            elif not current:
                raise RuntimeError("ShardedEvaluator: the in-graph MD step's buffers changed outside a refresh")
            _, graph, (energy, forces, flags, virial) = self._md_graph
            graph.replay()
            verdict = self._md_verdict(flags)
            capacity, bits, bad_input, md_nonfinite = verdict[-4:]
            if bad_input:
                md.restore()
                raise ValueError("distributed evaluation: non-finite MD positions (on some rank)")
            if capacity or verdict[_OVERFLOW] or verdict[_STALE]:
                # Discarded: the state before the step, and a refresh at it (every rank: the flags are agreed). The
                # step's outputs live in its graph's pool and the refresh may replace the graph: drop them (and the
                # graph) first, so that its release frees the pool before the memory admission and the capture.
                del graph, energy, forces, flags, virial
                md.restore()
                self.md_redos += 1
                self.line_slot_refreshes += int(self._line and bool(verdict[_SLOTS]))
                if capacity:
                    # This rank's counts of the discarded step: a floor for the refresh at q_n.
                    self.md_overflows += 1
                    floor, overflowed = engine.topology.demand(), engine.topology.capacities()
                    continue
                if refreshed:
                    # Stale right after a refresh at q_n: the step's own drift exceeds half the skin, which another
                    # refresh at q_n cannot help. The driver runs this step outside (evaluate refreshes at q_{n+1}),
                    # and the next _MD_FALLBACK_STEPS steps too (each return here costs two captures).
                    self._md_outside = self._MD_FALLBACK_STEPS
                    # (The steps outside move the atoms unobserved: unless a refresh comes first, as it does when
                    # evaluate refreshes them, the next in-graph step extrapolates no growth.)
                    self._md_displacement = None
                    self.last_stats.update({"in_graph": False, "attempts": attempts, "md_redos": self.md_redos})
                    return None
                floor = overflowed = None
                continue
            if verdict[_NONFINITE] or md_nonfinite:
                md.restore()
                raise ValueError("distributed evaluation: non-finite energy, forces, MD positions or momenta (on some "
                                 "rank)")
            break
        if self.max_neighbors is not None:
            largest = int(verdict[_NEIGHBORS])  # this step's builds (the captured step zeroes the count first)
            self._neighbor_count = max(self._neighbor_count, largest)
            try:
                self._refuse_neighbor_cap(largest)  # on every rank, from the same flags
            except RuntimeError:
                md.restore()  # the state before the refused step, as for the other refusals
                raise
            self.neighbor_cap_summary()  # warns once when an atom exceeds the model's neighbour cap
        displacement = struct.unpack("<f", struct.pack("<i", int(bits)))[0]
        previous = self._md_displacement
        growth = 0.0 if previous is None else max(displacement - previous, 0.0)
        self._md_displacement = displacement
        self._md_ahead = 2.0 * (displacement + self._MD_LOOKAHEAD_STEPS * growth) > self.skin
        n = engine.n_owned
        energy = energy.detach().reshape(()).clone()
        forces = forces[:n].detach().clone()
        virial = virial.detach().clone() if stress else None
        self.epoch += 1
        self.md_steps += 1
        self.last_stats.update({
            "evaluate_ms": (time.perf_counter() - entered) * 1000, "refreshed": refreshes > 0,
            "refreshes": self.refreshes, "captured": True, "in_graph": True, "attempts": attempts,
            "md_redos": self.md_redos, "md_overflows": self.md_overflows, "max_displacement": displacement,
            "refresh_ms": refresh_ms,
            "capture_ms": capture_ms,
            "communication": dict(self.context.comm.stats) if hasattr(self.context.comm, "stats") else {}})
        return self._prediction(md.state(), forces, energy, virial)

    @staticmethod
    def _md_verdict(flags: Tensor) -> list:
        """The agreed flags of an in-graph MD step on the host (its one synchronization)."""
        return flags.cpu().tolist()

    def _md_refresh(self, md, floor) -> None:
        """The in-graph MD step's refresh at the committed state q_n (collective): the migration of the atoms with
        their cached forces, the topology (``floor``: this rank's counts of a step that overflowed), the admission of
        a new layout, and the integrator's rows reordered into slots of the topology's owned capacity."""
        state = self._refresh(md.refresh_state(), floor)
        engine = self._engine
        engine.new_state.fill_(0)
        md.install(state, engine.topology.n_cap)
        self._md, self._md_refreshes = md, self.refreshes
        self._md_displacement, self._md_ahead = 0.0, False

    def _capture_md(self, md, key) -> None:
        """Capture the in-graph MD step (``key``: the layout, the hook's integrator buffers, the stress) on the side
        stream after a warm-up there, which the integrator's restore then undoes (both collective). The plain
        step's graph goes first (one captured graph in all). A capture that fails on one rank fails on every rank."""
        from fastmd.engine._graph_capture import capture_graph

        engine, stress = self._engine, key[2]
        started = time.perf_counter()
        self._release_graph()
        torch.cuda.empty_cache()
        stream = self._side_stream()
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(_CAPTURE_WARMUPS):
                engine.step(stress, md=md)  # agreed count, as for the plain step's capture
                md.restore()  # every warm-up starts at the same committed state
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        gc.collect()
        graph, outputs, error = torch.cuda.CUDAGraph(), None, None
        try:
            # On the warm-up's (this evaluator's own) stream, as the plain step's capture (see _capture).
            with capture_graph(graph, pool=self._pool, stream=stream, check_replay=False):
                outputs = engine.step(stress, md=md)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            error = f"capture failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        self.context.collective_check(error)
        self._md_graph = (key, graph, outputs)
        self.last_stats["capture_ms"] = (time.perf_counter() - started) * 1000

    def _evaluate_reference(self, state: AtomState, stress: bool) -> DistributedPrediction:
        """CPU: migrate, rebuild the exact rows and edges, one eager forward and backward (collective)."""
        entered = time.perf_counter()
        from .node_halo import _prepared

        topology = self._engine
        state = topology.refresh(state)
        self.refreshes += 1
        self._totals(state)
        batch = topology.batch
        inputs, error = _prepared(self.adapter, batch)
        if error is None and self.max_neighbors is not None and self.context.world_size > 1:
            # An owned centre has its complete neighbourhood on its rank. Ghost centres need not.
            centres = batch.edge_index[:, 0].long()
            centres = centres[centres < topology.n_owned]
            largest = int(torch.bincount(centres).max()) if len(centres) else 0
            try:
                self._refuse_neighbor_cap(largest)
            except RuntimeError as exc:
                error = str(exc)
        collective_check(self.context, error)
        n_total = torch.tensor(float(self.natoms), dtype=torch.float64)
        first = torch.tensor(self.context.rank == 0)
        if not self._agreed_exchanges:
            if self._line:
                fired, shapes = _counted(lambda: count_line_exchanges(self.adapter, inputs, batch.positions,
                                                                      batch.cell, topology.layout,
                                                                      self.spec.exchange_points))
                agree_line_exchanges(self.context, self.spec, fired, shapes, batch.positions.shape[0],
                                     topology.layout.pairs)
            else:
                fired, shapes = _counted(lambda: count_exchanges(self.derivative, inputs, batch.positions, batch.cell,
                                                                 topology.layout, topology.owned_mask, n_total, first))
                agree_exchanges(self.context, self.spec, fired, shapes, topology.layout.rows)
            self._agreed_exchanges = self._line or self._trimming_checked(topology.layout.ghost_rows)
        totals, forces = self.derivative(inputs, batch.positions, batch.cell, topology.owned_mask, topology.layout,
                                         stress=stress, n_total=n_total, first=first)
        if self.context.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=self.context.group)
        forces = forces[: topology.n_owned].detach()
        collective_check(self.context, None if bool(torch.isfinite(totals).all() and torch.isfinite(forces).all())
                         else "non-finite energy or forces")
        self._ids = state.ids
        self.epoch += 1
        self.last_stats = {"evaluate_ms": (time.perf_counter() - entered) * 1000, **topology.stats}
        return self._prediction(state, forces, totals[0].detach().clone(),
                                summed_stress(totals, batch.cell, forces.dtype))

    # --- reports and release ------------------------------------------------------------------------------

    def plan_record(self, op_plan=None):
        """The engine plan record of this evaluator (``fastmd.adapters.entry_plan_record``; ``op_plan`` linked on a
        copy): :meth:`running_plan`."""
        from fastmd.adapters import entry_plan_record

        return entry_plan_record(self, op_plan=op_plan)

    def _engine_name(self) -> str:
        """What runs the evaluations: "single" (one CUDA rank), "node_halo" or "line_partial" (P>1 on CUDA, by the
        shard's message flow), "reference" (CPU ranks)."""
        if self.single:
            return "single"
        if self.device.type != "cuda":
            return "reference"
        return "line_partial" if self._line else "node_halo"

    def slot_coverage(self) -> dict:
        """This rank's line slots at the last step of a line-graph shard at P>1 on CUDA (diagnostics and tests): the
        short boundary bonds at its owned centres and how many are resolved slots, its slots and how many resolved
        (``LineHaloTopology.slot_coverage``; host reads)."""
        if not isinstance(self._engine, LinePartialStrategy):
            raise ValueError("slot_coverage: the line-partial exchange of P>1 on CUDA only")
        return self._engine.slot_coverage()

    def edge_keys(self) -> Tensor:
        """This rank's edges of the last evaluation as ``[E, 5]`` int64 rows (centre ID, neighbour ID, integer image),
        on the device (diagnostics and tests; P>1 on CUDA: the node-halo topology's last build)."""
        if not isinstance(self._engine, (NodeHaloStrategy, LinePartialStrategy)):
            raise ValueError("edge_keys: the node-halo topology of P>1 on CUDA only")
        return self._engine.topology.edge_keys()

    def running_plan(self):
        """The negotiated plan (``EnginePlanRecord``, entry "distributed") with what this evaluator runs now: the
        admitted execution and the memory admission's summary; at P>1 the ``distributed`` block gains the domain
        grid, the exchange schedule and its bytes per step, every rank's sizes and memory at the last admission
        (``per_rank``: the same on every rank) and the rank contract's sha256."""
        record = copy.deepcopy(self.plan)
        if self._admission is not None and self._admission_on:
            record.memory = {**self._admission.summary(), "calibration": "measured per rank",
                             "note": record.memory.get("note", "")}
        if record.distributed is None:
            return record
        block = record.distributed
        block["grid"] = list(self.partition.domain_grid)
        block["rank_contract_sha256"] = self.rank_contract_sha256
        by_rank = self._admission.decisions[-1]["by_rank"] if self._admission and self._admission.decisions else []
        ghost_rows = [entry["ghost_rows"] for entry in by_rank]
        block.update(line_exchange_schedule(self.spec, self.dtype, ghost_rows, [entry["slots"] for entry in by_rank],
                                            coordinate_dtype=self.geometry_dtype)
                     if self._line else exchange_schedule(self.spec, self.dtype, ghost_rows,
                                                          flags=4 + (self.max_neighbors is not None),
                                                          coordinate_dtype=self.geometry_dtype))
        block["per_rank"] = [
            {"rank": rank, "owned": entry["owned"], "owned_slots": entry["owned_slots"], "rows": entry["rows"],
             "ghost_rows": entry["ghost_rows"], "edge_capacity": entry["edges"],
             **({"triplet_capacity": entry["triplets"], "line_slots": entry["line_slots"],
                 "slot_capacity": entry["slots"]} if self._line else {}),
             "predicted_peak_gib": {mode: None if value is None else round(value / 2**30, 3)
                                    for mode, value in entry["predicted"].items()},
             "budget_gib": round((entry["available"] - entry["margin"]) / 2**30, 3)}
            for rank, entry in enumerate(by_rank)]
        return record

    def config_summary(self) -> dict:
        layout = getattr(self._engine, "layout", None)
        if callable(layout):
            layout = None
        ghost_rows = 0 if layout is None else layout.ghost_rows
        features = sum(point.width for point in self.spec.exchange_points if point.kind == "copy")
        # A line-graph shard's line slots: their partials at each pair_merge point.
        slot_values = getattr(layout, "slots", 0) * sum(point.width for point in self.spec.exchange_points
                                                        if point.kind == "pair_merge")
        with matmul_precision_scope(self.matmul_precision):
            state = precision_state()
        return {
            "rank": self.context.rank, "world_size": self.context.world_size, "adapter": self.adapter.name,
            "kernel": getattr(self.adapter, "kernel", None),
            "engine": self._engine_name(),
            "execution": self.execution, "domain_grid": list(self.partition.domain_grid), "skin": self.skin,
            "cutoff": self.cutoff, "dtype": str(self.dtype), "geometry_dtype": str(self.geometry_dtype),
            "matmul_precision": self.matmul_precision,
            "precision_state": state, "shard_spec": self.spec.digest(),
            "exchange_points": [(point.name, point.module, point.width) for point in self.spec.exchange_points],
            "source_pins": [outcome.as_dict() for outcome in self.pins],
            "ghost_rows": ghost_rows,
            # Forward bytes of this rank's ghost rows (and line slots) per evaluation (the backward moves those of the
            # differentiable COPYs and the slots again): the raw and wrapped coordinates in the geometry's dtype, the
            # features in the model's.
            "exchange_bytes_per_step": ((ghost_rows * features + slot_values) * (torch.finfo(self.dtype).bits // 8)
                                        + ghost_rows * 6 * (torch.finfo(self.geometry_dtype).bits // 8)),
            "refreshes": self.refreshes, "captured": bool(self._graphs) or self._md_graph is not None,
            "plan_fingerprint": self.plan.fingerprint, "execution_auto": self.execution_auto,
            "kernels": {d.id: d.chosen for d in self.plan.decisions if d.id.startswith("kernels.")},
            "predicted_peak_bytes": self.predicted_peak,
            "memory_baseline_bytes": None if self._admission is None else self._admission.baseline,
            "rank_contract_sha256": self.rank_contract_sha256,
            **({"global_reductions": [(point.name, point.width) for point in self.spec.global_reductions]}
               if self.spec.global_reductions else {}),
            # the modules the declared ghost-row trimming resolved to (a class: each submodule of it), where declared
            **({"ghost_row_trimming": self._trim_paths()} if self.spec.owned_slot_inputs or self.spec.zero_ghost_outputs
               else {}),
            **({"neighbor_cap": self.neighbor_cap_summary()} if self.max_neighbors is not None
               and self.device.type == "cuda" else {}),
            **({"line_slots": getattr(getattr(self._engine, "topology", self._engine), "stats", {}).get("line_slots", 0),
                "slot_capacity": getattr(layout, "slots", 0), "line_slot_cutoff": self.spec.line_slot_cutoff,
                "line_slot_refreshes": self.line_slot_refreshes} if self._line else {}),
        }

    def _trim_paths(self) -> dict | None:
        """The declared ghost-row trimming's module paths, or None where it is not installed (one CUDA rank)."""
        derivative = getattr(self._engine, "derivative", None) if self.device.type == "cuda" else self.derivative
        trim = getattr(derivative, "trim", None)
        return None if trim is None else trim.paths

    def neighbor_cap_summary(self) -> dict | None:
        """For a model that keeps at most ``max_neighbors`` neighbours per atom in its own graph
        (``TopologySpec.max_neighbors``): the cap, the most neighbours within the cutoff an atom had in this
        evaluator's evaluations (on any rank: the step flags' MAX over the ranks) and whether that exceeded the
        cap, in which case this warns once. An adapter that selects nearest neighbours (ORB) selects on one
        rank and refuses over-cap evaluations at P>1; other adapters evaluate every radius pair. None without
        a cap, and on CPU ranks (their reference topology counts owned centres only for the refusal)."""
        if self.max_neighbors is None or self.device.type != "cuda":
            return None
        largest = self._neighbor_count
        if largest > self.max_neighbors and not self._neighbor_cap_warned:
            self._neighbor_cap_warned = True
            kept = ("the engine evaluated every pair" if not getattr(self.adapter, "selects_nearest_neighbors", False)
                    else f"the model kept the nearest {self.max_neighbors}" if self.single
                    else "those evaluations were refused (more than one rank)")
            warnings.warn(f"model {self.adapter.name!r} keeps at most {self.max_neighbors} neighbours per atom in its "
                          f"own graph; an atom had {largest} within the {self.cutoff:g} A cutoff, and {kept}",
                          RuntimeWarning, stacklevel=2)
        return {"max_neighbors": self.max_neighbors, "largest": largest, "exceeded": largest > self.max_neighbors}

    def _refuse_neighbor_cap(self, largest: int) -> None:
        """Refuse a P>1 evaluation whose radius graph differs from the model's nearest-neighbour graph.

        A ghost centre's neighbourhood is incomplete on this rank, so it cannot select that centre's nearest K.
        CUDA callers pass the global maximum; CPU callers propagate any refusal through collective_check.
        """
        if (self.context.world_size > 1 and self.max_neighbors is not None and largest > self.max_neighbors
                and getattr(self.adapter, "selects_nearest_neighbors", False)):
            raise RuntimeError(f"model {self.adapter.name!r} keeps the nearest {self.max_neighbors} neighbours of "
                               f"each atom; an atom had {largest} within the {self.cutoff:g} A cutoff, which a rank "
                               "cannot select for its ghost atoms: evaluate this system on one rank")

    def close(self) -> None:
        """Release the captured step (its NCCL work) first, then the topology and the hooks (idempotent)."""
        shutdown_registry.unregister(self, "evaluators")
        self.closed = True
        if self.device.type == "cuda":
            self._release_graph()
        if self._engine is not None and hasattr(self._engine, "close"):
            self._engine.close()
        if self.hooks is not None:
            self.hooks.release()
            self.hooks = None
