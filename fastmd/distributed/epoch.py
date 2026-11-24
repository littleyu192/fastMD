"""Skin-epoch topology for P>1 bond_partial (unified step 3.4, ``topology_engine="epoch"``).

Once per epoch (outside any CUDA graph, collective, host synchronization
allowed): migrate the atoms, build the historical device shard and prepare it
with the model's exchange strategy (``fastmd.distributed.strategy``; MatRIS's
``prepare_bond_partial``) with every cutoff widened by the skin (halo and bond
candidates rc + s, line slots r3 + s), agree the capacities, and refresh the
candidate list of a single-GPU ``FixedCapacityGraphBuilder`` whose rows are
this rank's [owned slots | per-peer ghost slots] (``GhostSpec``). The ghost
set, every exchange plan and the prevalidated schedule are frozen for the
epoch in the padded fixed-split layout of the strategy's ``pad``
(``pad_bond_partial``).

Every step (device only, no host synchronization): the owned coordinates go
into the builder rows, one COPY of the frozen ghost binding fills the ghost
rows, the builder filters the candidates exactly (d < rc and d < r3; the
triplets are enumerated exactly), and the padded bond_partial tensors are
derived in place from its buffers. Line slots are the epoch's superset
(every boundary bond within r3 + s at the epoch): a slot whose bond is not
short this step carries two empty partials, which merge to the value the bond
has without a slot (``merge_lse_partials``), and a slot whose bond does not
exist reads the sink bond. The rows are exact because no pair farther apart
than rc + s at the epoch comes within rc while every atom moves at most s/2
(the epoch ends when twice the global maximum displacement exceeds s).

Design and exactness argument: the task's DESIGN.md (3.4) and
docs/unified_engine_design.md sections 6.3-6.5.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time
from types import MappingProxyType

import torch

from fastmd._nvtx import nvtx_range, nvtx_ranged
from fastmd.graph.geometry import small_matmul
from fastmd.precision import order_independent

from . import shutdown_registry
from .line_slots import NO_PAIR, PAIR_CHECK_MAX_RANKS, asymmetric_pairs, pair_hit_entries
from .segment_layout import _csr

# Halo and candidate margin over rc + s and the slot cutoff (Angstrom): the
# epoch shard decides its bonds with torch arithmetic, the builder with its
# kernel; the margin keeps the epoch sets supersets of the builder's.
EPOCH_MARGIN = 1e-3
# A slot of the epoch superset that is a bond at the epoch must be one of the
# builder's edges unless it lies this close to the cutoff (a last-bit
# difference between the two distance expressions).
CUTOFF_TIE = 1e-4


def effective_skin(lengths, cutoff: float, skin: float, *, explicit: bool) -> float:
    """The skin a P>4 epoch runs with: more than 4 ranks keep one periodic image per axis.

    ``lengths`` are the reciprocal-plane spacings (lattice lengths for an
    orthogonal cell). ``skin`` if the halo ``cutoff + skin`` fits every spacing (the rule of
    ``DomainPartition.check_halo``). Otherwise ``topology_engine="auto"``
    (``explicit`` False) clamps it to ``min(lengths) - cutoff - EPOCH_MARGIN``
    while that is positive (the halo of the epoch's shard adds the margin);
    an explicit "epoch" or a box too thin for any skin is rejected. Pure in
    the agreed geometry, so every rank decides alike. (Until step 4.2 "auto"
    fell back to the rebuild topology here.)
    """
    lengths = [float(length) for length in lengths]
    halo = cutoff + skin
    if all(halo <= length for length in lengths):
        return skin
    shortest = min(lengths)
    clamped = shortest - cutoff - EPOCH_MARGIN
    if not explicit and clamped > 0:
        return clamped
    axis = next(axis for axis, length in enumerate(lengths) if halo > length)
    raise ValueError(
        f"halo width {halo:g} A (cutoff + skin) exceeds the box face spacing {lengths[axis]:g} A along axis "
        f"{'xyz'[axis]}: the distributed engine allows one periodic image per axis; use a supercell with "
        f"every face spacing >= {halo:g} A" + (f" or a topology skin of at most {clamped:g} A (--topology-skin)"
                                          if clamped > 0 else ""))


def step_verdict(tail, world_size) -> tuple[bool, list]:
    """(redo, fatal findings) of an agreed ``[isolated, *STEP_FLAGS, pair entries...]`` vector.

    A capacity overflow on any rank discards the step: every rank grows its
    capacities and evaluates the same positions again, so none of the step's
    other flags is a verdict. The overflowed rank's graph is truncated (it
    resolves no line slot; its triplet and incidence rows are cut): judging it
    reported false findings, e.g. in hot MD one rank's triplets past its
    capacity made its line-slot hits 0 against its peers' ("hits differ").
    Otherwise the findings are the fatal step flags and the rank pairs whose
    line-slot hit counts differ (the caller checks isolated atoms, entry 0).
    """
    tail = list(tail)
    head = 1 + len(EpochTopology.STEP_FLAGS)
    if tail[1]:
        return True, []
    fatal = [name for name, value in zip(EpochTopology.STEP_FLAGS[1:], tail[2:head]) if value]
    asymmetric = asymmetric_pairs(tail[head:], world_size)
    if asymmetric:
        fatal.append(f"line slot hits differ between rank pairs {asymmetric}")
    return False, fatal


def _float_bits(value):
    """Order-preserving int64 key of a nonnegative float32 device scalar (MAX-reducible)."""
    return value.reshape(1).to(torch.float32).view(torch.int32).to(torch.int64)


class EpochTopology:
    """The epoch plan and the per-step device topology of one evaluator.

    ``refresh(state, ...)`` is collective and returns the migrated state;
    ``step(state)`` fills the padded preparation in place and returns it;
    ``admission_flags`` / ``step_flags`` are the device entries the
    evaluator folds into its two per-step flag rounds.
    """

    def __init__(self, model, partition, context, *, skin: float, handle_isolated_atoms: bool,
                 capacity_config=None, eager: bool = True, capture_policy=None, strategy):
        from fastmd.engine.capacity_policy import CapturePolicy
        from fastmd.config import CapacityConfig
        from fastmd.engine.capture import CaptureCoordinator
        from .capacity import TieredCapacity

        if not (math.isfinite(skin) and skin > 0):
            raise ValueError("topology_skin must be finite and positive")
        self.model, self.partition, self.context = model, partition, context
        # The model's exchange strategy (fastmd.distributed.strategy): prepares and pads the epoch superset.
        self.strategy = strategy
        self.skin = float(skin)
        # prepare_bond_partial's ``active``: isolated owned atoms are inactive
        # only when they are handled by the reference table.
        self.handle_isolated = bool(handle_isolated_atoms) and model.reference_energy is not None
        self.rc = float(model.config["pairwise_cutoff"])
        self.r3 = float(model.config["three_body_cutoff"])
        # The capacity and capture policy of this evaluator (tiers, bands,
        # warmups), built once outside any forward scope; its fingerprint is
        # agreed at every refresh. Eager growth band floor (growth_headroom,
        # 1/128): an eager capacity growth costs one redone step, not a
        # recapture, so the padded rows (activation memory) stay tight.
        capacity_config = capacity_config or CapacityConfig()
        self.capture_policy = (CapturePolicy.from_config(capacity_config, distributed=True) if capture_policy is None
                               else capture_policy)
        self.capacity = TieredCapacity(context, capacity_config, capture_policy=self.capture_policy,
                                       growth_headroom=self.capture_policy.growth_headroom if eager else None)
        self.eager = bool(eager)
        # Its eager topology graphs hold NCCL work, and a topology can outlive its
        # evaluator (``DistributedMatRISEvaluator.last_graph``): released at shutdown.
        shutdown_registry.register(self, "graphs")
        self.slot_check_pending = False
        self._topology_graph = self._topology_graph_key = None
        self.coordinator = CaptureCoordinator(context._collective, rank=context.rank,
                                              world_size=context.world_size)
        self.prepared = None      # the padded, epoch-frozen preparation (tensors are static buffers)
        self.builder = None
        self._builder_key = None
        self.epochs = self.steps = self.overflows = 0
        self._fresh = False
        self._buffers = self._buffers_generation = None
        # Line slots of the installed layout (``_slot_keys``; None without a line graph).
        self._slot_peer = self._slot_rows = None
        # Part of every capture key: bumped when an interrupted refresh leaves
        # the persistent buffers undefined (``invalidate``).
        self._allocations = 0
        self.isolated_is_error = False
        # The last agreed displacement (whole-step path's look-ahead refresh).
        self.last_displacement = None
        self.refresh_ahead = False
        self.stats = {}

    # --- per-step flags -------------------------------------------------

    def admission_flags(self, state, device):
        """[atoms (IDs, order or numbers) changed, float bits of this rank's max owned displacement].

        Never raises (it runs before the admission round): an unusable state
        reads as changed, and the preflight reports it.
        """
        stale = torch.ones(2, dtype=torch.int64, device=device)
        stale[1] = 0
        if self.prepared is None:
            return stale
        try:
            ids, numbers, positions = state.ids, state.numbers, state.positions.detach()
            reference = self._reference[:self.n_owned]
            if (ids.shape != self._ids.shape or numbers.shape != self._numbers.shape
                    or positions.shape != reference.shape or positions.device != device
                    or positions.dtype != reference.dtype):
                return stale
            changed = torch.zeros((), dtype=torch.bool, device=device)
            if ids is not self._ids:
                changed = changed | (ids.to(device) != self._ids).any()
            if numbers is not self._numbers:
                changed = changed | (numbers.to(device) != self._numbers).any()
            if self.n_owned:
                # Minimum image (the builder's rule): cell-wrapped inputs do not end the epoch.
                delta = positions - reference
                delta = delta - small_matmul(torch.round(small_matmul(delta, self._inverse)), self._cell)
                displacement = torch.linalg.vector_norm(delta, dim=1).max()
            else:
                displacement = torch.zeros((), dtype=torch.float32, device=device)
            return torch.cat((changed.reshape(1).long(), _float_bits(displacement)))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return stale

    # Look-ahead of the whole-step path: refresh before the step whose
    # displacement, extrapolated by this many times the last step's increase,
    # would exceed half the skin (a violation is caught after the step anyway
    # and redone; the look-ahead only makes that rare).
    LOOKAHEAD_STEPS = 3

    def observe_displacement(self, displacement: float) -> None:
        """Record an agreed step displacement; set ``refresh_ahead`` for the next step."""
        previous = self.last_displacement
        growth = 0.0 if previous is None else max(displacement - previous, 0.0)
        self.last_displacement = displacement
        self.refresh_ahead = 2.0 * (displacement + self.LOOKAHEAD_STEPS * growth) > self.skin

    def needs_refresh(self, agreed) -> tuple[bool, float]:
        """From the agreed MAX of ``admission_flags``: (refresh, global max displacement)."""
        displacement = float(torch.tensor([int(agreed[1])], dtype=torch.int32).view(torch.float32)[0])
        stale = self.prepared is None or bool(agreed[0]) or 2.0 * displacement > self.skin
        return stale, displacement

    # Entries of step_flags(): capacity overflow (grow and redo), stale or
    # truncated candidates, unpaired directed rows, a boundary bond with local
    # triplets and no slot, received ghost coordinates that differ from the
    # builder's (the last four are fatal).
    STEP_FLAGS = ("capacity", "stale_candidates", "invalid_pairs", "slot_missed", "ghost_mismatch")

    def step_flags(self, result):
        """This step's device predicates (``STEP_FLAGS``), after the forward of ``result``.

        A model_graph replay (a ``GraphResult``) has no autograd backend: its
        ghost coordinates are the captured COPY of the same owners' rows (not
        rechecked per replay).
        """
        n_cap = self._layout["n_cap"]
        backend = getattr(result, "backend", None)
        if backend is None:
            return torch.cat((self._flags, torch.zeros(1, dtype=torch.int64, device=self._flags.device),
                          self._pair_counts()))
        try:
            received = backend.exchange_output("coordinates")
        except RuntimeError:
            raise RuntimeError("topology epoch: the step has no ghost coordinate exchange") from None
        mismatch = (received.detach().view(torch.int32)
                    != self.coords[n_cap:, 3:].contiguous().view(torch.int32)).any().reshape(1).long()
        return torch.cat((self._flags, mismatch, self._pair_counts()))

    def whole_flags(self, energy, forces, result):
        """The flag vector of one whole step (computed inside the captured graph; MAX-agreed after it).

        ``[isolated, capacity, stale, invalid, slot_missed, ghost_mismatch,
        pair counts..., displacement bits, non-finite E/F, non-finite input]``
        (``whole_fields``): the step flags of the finite round, the
        displacement of the step (the epoch decision is made after the step:
        a violated skin redoes it after a refresh) and the preflight's
        finite-coordinate predicate.
        """
        device = self._flags.device
        isolated = (self.builder._isolated_atom_count.reshape(1).long().clamp(max=1) if self.isolated_is_error
                    else torch.zeros(1, dtype=torch.int64, device=device))
        nonfinite = ~(torch.isfinite(energy).all() & torch.isfinite(forces).all())
        bad_input = ~torch.isfinite(self._pos_in).all()
        return torch.cat((isolated, self.step_flags(result), _float_bits(self._disp),
                          nonfinite.reshape(1).long(), bad_input.reshape(1).long()))

    def whole_fields(self, agreed) -> dict:
        """Named entries of an agreed ``whole_flags`` vector."""
        agreed = list(agreed)
        head = len(self.STEP_FLAGS) + 1
        pairs = 0 if (self.context.world_size == 1 or self.context.world_size > PAIR_CHECK_MAX_RANKS) \
            else self.context.world_size * (self.context.world_size - 1)
        fields = {"isolated": agreed[0], **dict(zip(self.STEP_FLAGS, agreed[1:head])),
                  "pairs": agreed[head:head + pairs]}
        tail = agreed[head + pairs:]
        fields["displacement"] = float(torch.tensor([int(tail[0])], dtype=torch.int32).view(torch.float32)[0])
        fields["nonfinite"], fields["bad_input"] = tail[1], tail[2]
        # The in-graph MD step's integrator flag (non-finite positions or momenta), appended by the runner.
        fields["md_nonfinite"] = tail[3] if len(tail) > 3 else 0
        return fields

    # Per-rank-pair slot-hit symmetry (``fastmd.distributed.line_slots``) rides in the finite MAX round.
    PAIR_CHECK_MAX_RANKS = PAIR_CHECK_MAX_RANKS
    _NO_PAIR = NO_PAIR

    def _pair_counts(self):
        """[+hits, -hits] per unordered rank pair of this rank's resolved slots (``pair_hit_entries``)."""
        return pair_hit_entries(self._slot_peer, self._slot_rows, self.context.rank, self.context.world_size,
                                self._flags.device, has_line=self._layout["has_line"])

    def pair_asymmetry(self, agreed) -> list:
        """Rank pairs whose agreed slot-hit counts differ (``_pair_counts`` after the MAX round)."""
        return asymmetric_pairs(agreed, self.context.world_size)

    # --- epoch ------------------------------------------------------------

    @torch.no_grad()
    @nvtx_ranged("EpochTopology.refresh")
    def refresh(self, state, *, epoch, prepare_kwargs, validate, floor=None):
        """Migrate, prepare the epoch superset, agree capacities, refresh the builder. Collective.

        ``validate(prepared)`` is the evaluator's run-boundary admission of the
        superset preparation (``validate_prepared``). ``floor``: this rank's
        row demand at least these ``{"u", "i", "t"}`` counts (the in-graph MD
        step refreshes at the committed positions, so an overflow of the step
        after them grows the capacities to the overflowed step's untruncated
        counts, ``untruncated_rows``). Returns the migrated state.
        """
        from .graph import build_distributed_graph
        from .partition import migrate_atoms

        started = time.perf_counter()
        context = self.context
        state = migrate_atoms(state, self.partition, context)
        migrated = time.perf_counter()
        shard = build_distributed_graph(state, self.partition, context,
                                        self.rc + self.skin + EPOCH_MARGIN,
                                        self.r3 + self.skin + EPOCH_MARGIN, triplets=False)
        graphed = time.perf_counter()
        superset = self.strategy.prepare(self.model, shard, context, epoch=epoch,
                                         slot_cutoff=self.r3 + self.skin + EPOCH_MARGIN,
                                         **prepare_kwargs)
        validate(superset)
        prepared_at = time.perf_counter()
        topology = shard.device_topology
        t = superset.tensors
        device = t["owned_positions"].device
        n_owned = t["owned_positions"].shape[0]
        # Rows of the superset's local atoms (owned by ID, then ghosts by owner, ID).
        ids = topology["atom_ids"]
        owned_ids = ids[topology["owned_atom_indices"]]
        error = None
        if owned_ids.shape != state.ids.shape or not bool((owned_ids == state.ids).all()):
            error = "topology epoch: the migrated state is not in owned-row order"
        demand = self._demand(superset, shard)
        if floor:
            rows = demand["rows"]
            for name, count in floor.items():
                if name in rows and (name == "u" or demand["has_line"]):
                    rows[name] = max(rows[name], int(count))
        agreed = self.coordinator.agree((self.capacity.generation, self.capacity.fingerprint),
                                        None if error is not None else demand, False, error)
        self.natoms = sum(item["rows"]["n"] for item in agreed.demands)
        # Every rank's (atoms, triplets) at this refresh: execution="auto" chooses from them.
        self.rank_rows = [(item["rows"]["n"], item["rows"].get("t", 0)) for item in agreed.demands]
        if not self.natoms:
            raise ValueError("distributed preflight failed: " + "; ".join(
                f"rank {rank}: empty global system" for rank in range(context.world_size)))
        first = self.capacity.rows is None
        self.capacity_changed = self.capacity.plan(list(agreed.demands)) or first
        self.demand_rows = dict(demand["rows"]) if demand is not None else {}
        padded = self.strategy.pad(superset, self.capacity, topology=False)
        self._install(padded, superset, shard, state)
        from .graph_runner import structure_signature
        # model_graph entry key: agreed capacities and the (rank-uniform) structure.
        self.graph_key = (self.capacity.generation, structure_signature(self.prepared), self._builder_key)
        self.n_owned = n_owned
        self._ids, self._numbers = state.ids, state.numbers
        self._cell = self.builder.cell
        self._inverse = self.builder.inv_cell
        self.epochs += 1
        finished = time.perf_counter()
        self.stats = {
            "epochs": self.epochs, "capacity_generation": self.capacity.generation,
            "refresh_ms": (finished - started) * 1000, "migration_ms": (migrated - started) * 1000,
            "superset_graph_ms": (graphed - migrated) * 1000,
            "superset_prepare_ms": (prepared_at - graphed) * 1000,
            "superset": {key: superset.stats[key] for key in (
                "owned_atoms", "ghost_atoms", "local_bonds", "boundary_bonds", "line_slots",
                "local_triplets", "has_line") if key in superset.stats},
            "rows": dict(self.capacity.local()),
        }
        return state

    def _demand(self, superset, shard):
        """Capacity demand of the epoch: the exact topology at the epoch positions, superset routes.

        Rows (n, u, i, t) count the bonds (canonical length < rc), the owned
        short rows of centers with at least two, and the triplets at the
        epoch positions; the ghost and slot splits are the superset's.
        """
        topology = shard.device_topology
        t, plans = superset.tensors, superset.plans
        device = t["owned_positions"].device
        cell = t["cell"]
        from .device_neighbors import _canonical_lengths
        canonical = topology["canonical_edges"]
        ids = topology["atom_ids"]
        target = torch.searchsorted(ids, canonical[:, 0].contiguous())
        source = torch.searchsorted(ids, canonical[:, 1].contiguous())
        lengths = _canonical_lengths(shard.positions, cell, target, source, canonical[:, 2:])
        bonds = int((lengths < self.rc).sum())
        center = topology["center_owned_target"]
        row_vectors = (shard.positions[topology["center_source"]] - shard.positions[topology["center_target"]])
        from fastmd.graph.geometry import small_matmul
        row_vectors = row_vectors + small_matmul(topology["center_images"].to(cell.dtype), cell)
        short = row_vectors.norm(dim=1) < self.r3
        counts = torch.zeros(len(t["owned_positions"]), dtype=torch.int64, device=device)
        counts.index_add_(0, center[short], torch.ones_like(center[short]))
        incidences = int(counts[counts >= 2].sum())
        triplets = int((counts * (counts - 1)).sum())
        ghosts = plans["ghosts"]
        routes = {"bp_ghosts": (tuple(ghosts.send_counts), tuple(ghosts.receive_counts))}
        if superset.has_line:
            slots = plans["line_partials"]
            routes["bp_slots"] = (tuple(slots.send_counts), tuple(slots.receive_counts))
        rows = {"n": len(t["owned_positions"]), "u": bonds, "i": incidences if superset.has_line else 0,
                "t": triplets if superset.has_line else 0}
        return {"has_line": superset.has_line, "rows": rows, "routes": routes}

    def _install(self, padded, superset, shard, state):
        """The epoch layout, builder, candidates and slot keys, in persistent buffers.

        Everything the per-step program reads lives in buffers allocated per
        layout key (capacities, ghost rows, slots, candidate capacity) and is
        rewritten in place at every refresh with the same key, so one captured
        whole-step graph (``DistributedGraphRunner.run_whole``) serves every
        epoch of that key: route indices, Z, slot keys, the candidate list,
        the reference coordinates and the builder's candidates are copies.
        """
        from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder, GhostSpec

        t = padded.tensors
        device = t["owned_positions"].device
        rows = self.capacity.local()
        n_cap, u_cap = rows["n"], rows["u"]
        has_line = padded.has_line
        i_cap, t_cap = (rows["i"], rows["t"]) if has_line else (1, 1)
        ghost_rows = t["ghost_atomic_numbers"].shape[0]
        local_rows = n_cap + ghost_rows
        slot_total = sum(self.capacity.splits("bp_slots")[0]) if has_line else 0
        layout = (n_cap, ghost_rows, u_cap, i_cap, t_cap, has_line, slot_total)
        if self._builder_key is None or self._builder_key[:5] != layout[:5]:
            self._topology_graph = self._topology_graph_key = None
            self.builder = None
            self._buffers = None
            torch.cuda.empty_cache()
            self.builder = FixedCapacityGraphBuilder(
                cell=t["cell"], atomic_numbers=torch.ones(local_rows, dtype=torch.int64, device=device),
                u_capacity=u_cap, t_capacity=t_cap, atom_graph_cutoff=self.rc,
                line_graph_cutoff=self.r3, candidate_cutoff=self.rc + self.skin + EPOCH_MARGIN,
                device=device, ghosts=GhostSpec(n_owned=n_cap))
            self.positions = torch.zeros((local_rows, 3), dtype=torch.float32, device=device)
            # [continuous builder coordinates | the owner's wrapped model coordinates] per row.
            self.coords = torch.zeros((local_rows, 6), dtype=torch.float32, device=device)
            # The caller's owned coordinates (rows past the real owned atoms stay 0) and the epoch reference.
            self._pos_in = torch.zeros((n_cap, 3), dtype=torch.float32, device=device)
            self._reference = torch.zeros((n_cap, 3), dtype=torch.float32, device=device)
            self._n_owned_t = torch.zeros((), dtype=torch.int64, device=device)
            self._wrap_cell = torch.zeros((3, 3), dtype=torch.float32, device=device)
            self._wrap_inverse = torch.zeros((3, 3), dtype=torch.float32, device=device)
            self._disp = torch.zeros((), dtype=torch.float32, device=device)
            self._k = torch.zeros((local_rows, 3), dtype=torch.int64, device=device)
            self._flags = torch.zeros(4, dtype=torch.int64, device=device)
            self._line_flag = torch.zeros(1, dtype=torch.int32, device=device)
            self._counts = torch.zeros(2, dtype=torch.int64, device=device)
            self._candidate_capacity = 0
        builder = self.builder
        # Epoch contents of the builder rows (= the model's local atoms).
        st = superset.tensors
        owned = st["owned_positions"].shape[0]
        topology = shard.device_topology
        ids = topology["atom_ids"]
        ghost_ids_real = ids[self._ghost_old(superset, shard)]
        receive = self.capacity.splits("bp_ghosts")[1]
        real_receive = superset.plans["ghosts"].receive_counts
        offsets = [sum(receive[:p]) for p in range(len(receive))]
        ghost_position = torch.cat([torch.arange(o, o + c, device=device) for o, c in zip(offsets, real_receive)]
                                   or [torch.empty(0, dtype=torch.int64, device=device)])
        numbers = torch.ones(local_rows, dtype=torch.int64, device=device)
        numbers[:owned] = st["atomic_numbers"]
        numbers[n_cap + ghost_position] = st["ghost_atomic_numbers"]
        atom_ids = torch.full((local_rows,), -1, dtype=torch.int64, device=device)
        atom_ids[:owned] = ids[topology["owned_atom_indices"]]
        atom_ids[n_cap + ghost_position] = ghost_ids_real
        active = torch.cat((torch.arange(owned, device=device), n_cap + ghost_position))
        builder.set_atoms(numbers, atom_ids, active, owned)
        self.ghost_position = ghost_position
        self._n_owned_t.fill_(owned)
        # In place: a captured graph reads these (a new tensor per refresh would
        # leave a replay reading freed memory).
        self._wrap_cell.copy_(self.partition.cell.to(device=device, dtype=torch.float32))
        self._wrap_inverse.copy_(torch.linalg.inv(self._wrap_cell))

        # Buffers of the padded preparation: epoch constants copied into the
        # persistent tensors, topology buffers (filled every step) allocated once.
        zeros = lambda *shape, dtype=torch.int64: torch.zeros(shape, dtype=dtype, device=device)
        # A new capacity generation (rows or per-peer splits) or line branch
        # changes route and slot shapes: new buffers (and a new capture key).
        generation = (self.capacity.generation, has_line)
        fresh = self._buffers is None or self._buffers_generation != generation
        self._buffers_generation = generation
        if fresh:
            bonds = u_cap
            buffers = {name: value.clone() for name, value in t.items()}
            buffers.update({
                "bond_valid_mask": zeros(bonds, dtype=torch.bool), "bond_target": zeros(bonds),
                "bond_source": zeros(bonds), "bond_images": zeros(bonds, 3, dtype=torch.float32),
                "atom_target_index": zeros(2 * bonds), "atom_source_index": zeros(2 * bonds),
                "owned_positions": zeros(n_cap, 3, dtype=torch.float32), "active": zeros(n_cap, dtype=torch.bool),
                "atom_target_csr_perm": zeros(2 * bonds), "atom_target_csr_off": zeros(local_rows + 1),
                "atom_target_count": zeros(local_rows), "atom_source_csr_perm": zeros(2 * bonds),
                "atom_source_csr_off": zeros(local_rows + 1), "atom_source_count": zeros(local_rows),
            })
            if has_line:
                buffers.update({
                    "incidence_valid_mask": zeros(i_cap, dtype=torch.bool), "incidence_target": zeros(i_cap),
                    "incidence_source": zeros(i_cap), "incidence_images": zeros(i_cap, 3, dtype=torch.float32),
                    "triplet_left_geometry": zeros(t_cap), "triplet_right_geometry": zeros(t_cap),
                    "line_target_index": zeros(t_cap), "line_source_index": zeros(t_cap),
                    "line_atom_index": zeros(t_cap),
                    "line_target_csr_perm": zeros(t_cap), "line_target_csr_off": zeros(bonds + 1),
                    "line_target_count": zeros(bonds), "line_source_csr_perm": zeros(t_cap),
                    "line_source_csr_off": zeros(bonds + 1), "line_source_count": zeros(bonds),
                })
                buffers["line_active"] = torch.ones((), dtype=torch.bool, device=device)
            self._buffers = buffers
            self.prepared = replace(padded, tensors=MappingProxyType(buffers))
        else:
            buffers = self._buffers
            for name, value in t.items():
                if name not in ("owned_positions", "active", "bp_slots"):
                    buffers[name].copy_(value)
            # Route indices of the new epoch into the persistent bindings (same
            # capacities, so the same shapes; the plans' split counts are caps).
            copied = set()
            for (name, target), (fresh_name, source) in zip(self.prepared.schedule.exchanges,
                                                            padded.schedule.exchanges):
                if name != fresh_name:
                    raise RuntimeError("topology epoch: the exchange schedule changed within a layout")
                if id(target.indices) not in copied:
                    for target_index, source_index in zip(target.indices, source.indices):
                        target_index.copy_(source_index)
                    copied.add(id(target.indices))
            self.prepared = replace(self.prepared, stats=padded.stats)
        # Builder input rows at the epoch: raw owned positions, raw ghost copies.
        self.coords.zero_()
        self._reference.zero_()
        self._reference[:owned] = state.positions.detach().to(torch.float32)
        self._pos_in.copy_(self._reference)
        self._fill()
        builder.refresh_candidates(self.positions)
        truncated = not bool(builder._candidate_capacity_valid.item())
        self.context.collective_check(
            f"topology epoch: a candidate row exceeds the builder's neighbor buffer ({builder.buffer_max})"
            if truncated else None)
        if has_line:
            self._slot_keys(padded, superset, shard, fresh)
        self._layout = dict(n_cap=n_cap, u_cap=u_cap, i_cap=i_cap, t_cap=t_cap, local_rows=local_rows,
                            n_owned=owned, has_line=has_line)
        # Compact candidate list of the epoch (the predicate runs on it only),
        # padded to a capacity: padding entries read slot 0 and write the trash slot.
        width = builder.buffer_max
        valid = torch.arange(width, device=device)[None, :] < builder.num_neighbors.long()[:, None]
        candidates = torch.nonzero(valid.reshape(-1)).reshape(-1)
        count = candidates.numel()
        # The capacity is part of the capture key: it must be the same on every
        # rank, or one rank would recapture (collectives in its warmup) while the
        # others replay. One gather of the local counts, at the refresh only.
        largest = max(self.context.all_gather_object(count)) if self.context.world_size > 1 else count
        if largest > self._candidate_capacity or self._candidate_capacity == 0:
            # (A new layout resets the capacity; a system without candidates still gets the buffers.)
            self._candidate_capacity = max(1024, -(-largest * 9 // 8 // 1024) * 1024)
            self._cand_slots = torch.zeros(self._candidate_capacity, dtype=torch.int64, device=device)
            self._cand_target = torch.zeros_like(self._cand_slots)
            self._edge_full = torch.zeros(local_rows * width + 1, dtype=torch.bool, device=device)
            self._short_full = torch.zeros_like(self._edge_full)
        self._cand_slots.zero_()
        self._cand_slots[:count] = candidates
        self._cand_target.fill_(local_rows * width)
        self._cand_target[:count] = candidates
        self._edge_full.zero_()
        self._short_full.zero_()
        self._builder_key = layout + (self._candidate_capacity, self._allocations)
        self.slot_check_pending = has_line
        if self.eager:
            # First step at the epoch positions: also the slot-key admission.
            # (model_graph: the refresh step's replay derives it; the evaluator
            # then calls check_slots.)
            self._derive()
            self._fresh = True
            self.check_slots()
            stale = self._topology_graph_key != self._builder_key + (self._buffers_generation,)
            if self.context.world_size > 1:
                # Collective (COPY and all_reduce inside): capture on every rank or none.
                stale = any(self.context.all_gather_object(stale))
            if stale:
                self._capture_topology()

    @staticmethod
    def _ghost_old(superset, shard):
        """The superset's ghost rows in the shard's coordinate table (prepare_bond_partial's order)."""
        topology = shard.device_topology
        owned_old = topology["owned_atom_indices"]
        ids = topology["atom_ids"]
        canonical = topology["canonical_edges"]
        count = len(ids)
        used = torch.zeros(count, dtype=torch.bool, device=ids.device)
        used.index_fill_(0, torch.searchsorted(ids, canonical[:, 0].contiguous()), True)
        used.index_fill_(0, torch.searchsorted(ids, canonical[:, 1].contiguous()), True)
        used.index_fill_(0, owned_old, False)
        ghost_old = torch.where(used)[0]
        return ghost_old[torch.argsort(shard.atom_owner_ranks[ghost_old], stable=True)]

    def _slot_keys(self, padded, superset, shard, fresh):
        """Every padded slot's directed builder row at the epoch: (center row, neighbor row, image)."""
        st = superset.tensors
        device = st["owned_positions"].device
        n_cap = self.capacity.local()["n"]
        owned = st["owned_positions"].shape[0]
        slots = st["bp_slots"]
        # Superset local index -> builder row.
        local = torch.cat((torch.arange(owned, device=device), n_cap + self.ghost_position))
        target = local[st["bond_target"][slots]]
        source = local[st["bond_source"][slots]]
        # The superset's images are relative to the producer-wrapped epoch
        # coordinates (partition.wrap_positions), the builder's to its own
        # reference wrap: shift each end by the integer difference of the two
        # wraps of its raw coordinates (exact by rounding, whatever the ulp of
        # either wrap near a cell face).
        topology = shard.device_topology
        ids = topology["atom_ids"]
        canonical = topology["canonical_edges"][slots]
        old_target = torch.searchsorted(ids, canonical[:, 0].contiguous())
        old_source = torch.searchsorted(ids, canonical[:, 1].contiguous())
        inverse = torch.linalg.inv(st["cell"].double().cpu()).to(device)

        def frame_shift(builder_rows, old_rows):
            raw = self.positions[builder_rows].double()
            producer = shard.positions[old_rows].double()
            producer_wraps = torch.round(small_matmul(raw - producer, inverse)).long()
            return self.builder.reference_wraps[builder_rows].long() - producer_wraps

        images = (canonical[:, 2:].long() + frame_shift(source, old_source)
                  - frame_shift(target, old_target))
        send = self.capacity.splits("bp_slots")[0]
        real = superset.plans["line_partials"].send_counts
        offsets = [sum(send[:p]) for p in range(len(send))]
        position = torch.cat([torch.arange(o, o + c, device=device) for o, c in zip(offsets, real)]
                             or [torch.empty(0, dtype=torch.int64, device=device)])
        total = sum(send)
        if fresh:
            # Peer of every padded slot (blocks in peer order), for the per-pair hit counts.
            self._slot_peer = torch.repeat_interleave(
                torch.arange(len(send), device=device), torch.tensor(send, device=device), output_size=total)
            self._slot_valid = torch.zeros(total, dtype=torch.bool, device=device)
            self._slot_center = torch.zeros(total, dtype=torch.int64, device=device)
            self._slot_neighbor = torch.zeros(total, dtype=torch.int64, device=device)
            self._slot_image = torch.zeros((total, 3), dtype=torch.int64, device=device)
            self._slot_rows = torch.full((total,), -1, dtype=torch.int64, device=device)
        self._slot_valid.zero_()
        self._slot_center.zero_()
        self._slot_neighbor.zero_()
        self._slot_image.zero_()
        self._slot_valid[position] = True
        self._slot_center[position] = target
        self._slot_neighbor[position] = source
        self._slot_image[position] = images
        self._slot_length = torch.full((total,), float("inf"), dtype=torch.float32, device=device)
        from .device_neighbors import _canonical_lengths
        canonical = topology["canonical_edges"]
        rows_t = torch.searchsorted(ids, canonical[:, 0].contiguous())
        rows_s = torch.searchsorted(ids, canonical[:, 1].contiguous())
        lengths = _canonical_lengths(shard.positions, st["cell"], rows_t, rows_s, canonical[:, 2:])
        # prepare_bond_partial's bonds are the shard's canonical rows in order.
        self._slot_length[position] = lengths[slots].float()

    def _capture_topology(self):
        """Capture the eager path's per-step topology (fill, COPY, filter, derive) on the persistent buffers.

        Collective (the ghost COPY and the RG-20 all_reduce are inside); every
        rank captures at the same refresh (the layout key is agreed). Replays
        read ``_pos_in`` and rewrite the same buffers, so eager steps launch
        one graph instead of ~150 small kernels.
        """
        device = self._pos_in.device
        self._topology_graph = None
        saved = self._pos_in.clone()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad():
            with torch.cuda.stream(stream):
                self._fill()
                self._derive()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            self.context.collective_check(None)
            graph = torch.cuda.CUDAGraph()
            from .graph_runner import _capture_graph
            with _capture_graph(graph, stream=stream):
                self._fill()
                self._derive()
        torch.cuda.synchronize(device)
        self._pos_in.copy_(saved)
        self._topology_graph = graph
        self._topology_graph_key = self._builder_key + (self._buffers_generation,)

    def invalidate(self):
        """After an exception inside a refresh or an epoch step: no epoch.

        The persistent buffers may be half rewritten (or re-created under a
        captured graph). The next evaluation refreshes with a new builder, new
        buffers and new capture keys, so no graph captured before replays.
        """
        self.close()
        self.prepared = None
        self.graph_key = None
        self.builder = None
        self._builder_key = None
        self._buffers = self._buffers_generation = None
        self._fresh = self.slot_check_pending = self.refresh_ahead = False
        self._allocations += 1

    def close(self):
        """Release the captured topology graph (captured NCCL work) before process-group destruction.

        NCCL's communicator finalize waits for every graph holding its work to
        be destroyed: a live graph hangs ``destroy_process_group``. The
        destruction's release callback runs at a device synchronization.
        """
        self.release_graph(synchronize=True)

    def set_eager(self, eager):
        """The evaluator's execution mode (execution="auto" may switch at a capacity-change refresh).

        Eager captures its own topology graph at the next refresh (until then
        its steps derive without it); model_graph captures the whole step.
        """
        eager = bool(eager)
        if eager == self.eager:
            return
        self.eager = eager
        self.capacity.set_growth_headroom(self.capture_policy.growth_headroom if eager else None)
        self.release_graph(synchronize=True)
        self._fresh = False

    def release_graph(self, synchronize=False):
        """Destroy the captured topology graph (the next eager refresh captures again).

        Without ``synchronize`` also safe from a garbage-collection finalizer,
        which may run while another graph is being captured.
        """
        graph, self._topology_graph, self._topology_graph_key = self._topology_graph, None, None
        if graph is not None:
            graph.reset()
            del graph
            if synchronize:
                torch.cuda.synchronize(self._pos_in.device)

    @torch.no_grad()
    def derive_at_epoch(self):
        """model_graph, the in-graph MD step: the eager derive at the refresh positions, then the slot admission.

        The captured step derives at the positions after its integrator
        half, never at the epoch positions the slot keys were checked
        against (``_check_slots``, CUTOFF_TIE): this eager derive of the
        reference rows (``_install`` filled them) gives ``check_slots`` the
        epoch graph. Collective (the RG-20 all_reduce, the admission).
        """
        self._derive()
        self.check_slots()

    def untruncated_rows(self) -> dict:
        """This rank's untruncated row counts of the last derive (one download; after an overflowed step).

        Bonds from the builder's edge offsets and, with the line graph, the
        incidences and triplets of the owned centers from its short-row
        counts: neither is clamped to a capacity (``_derive``'s incidences and
        the status counts are), so they are the demand that step had.
        """
        b, lay = self.builder, self._layout
        rows = {"u": int(b.edge_offsets[-1]) // 2}
        if lay["has_line"]:
            counts = b.short_counts[:lay["n_owned"]].long()
            rows["i"] = int(counts[counts >= 2].sum())
            rows["t"] = int((counts * (counts - 1)).sum())
        return rows

    def check_slots(self):
        """The slot-key admission of a refresh, after the first derive at the epoch positions. Collective."""
        if self.slot_check_pending:
            self.slot_check_pending = False
            self._check_slots()

    def _check_slots(self):
        """Collective admission of the slot keys at the epoch positions.

        Every slot whose bond exists at the epoch (length < rc, away from the
        cutoff by more than CUTOFF_TIE) must resolve to a builder edge: a
        frame or key mismatch would otherwise drop its partial silently.
        """
        found = self._slot_rows >= 0
        must = self._slot_valid & (self._slot_length < self.rc - CUTOFF_TIE)
        may = self._slot_valid & (self._slot_length < self.rc + CUTOFF_TIE)
        bad = int(((must & ~found) | (found & ~may)).sum())
        self.context.collective_check(
            None if bad == 0 else f"topology epoch: {bad} line slots do not resolve to builder edges")

    def _fill(self):
        """Builder and model coordinates of every local row from ``_pos_in``, then the ghost COPY.

        Owned slots (rows past the real owned atoms stay 0 in ``_pos_in`` and
        the reference). Columns 0-2: continuous coordinates for the builder
        (the epoch reference plus the minimum-image displacement: a caller's
        cell-wrapped coordinate is no jump). Columns 3-5: the model's
        coordinates, the caller's coordinates wrapped into the cell by the
        full shard's producer wrap (same bits as its halo coordinates, no
        reconstruction roundoff). Ghost rows receive both from their owners
        in one COPY of the frozen ghost binding (6 wide), so the model's ghost
        coordinates are the owners' bits. Device only, fixed shapes (captured
        in the whole-step graph); ``_disp`` is the largest owned displacement.
        """
        cell, inverse = self.builder.cell, self.builder.inv_cell
        n_cap = self._pos_in.shape[0]
        delta = self._pos_in - self._reference
        delta = delta - small_matmul(torch.round(small_matmul(delta, inverse)), cell)
        self._disp.copy_(torch.linalg.vector_norm(delta, dim=1).max())
        self.coords[:n_cap, :3].copy_(self._reference + delta)
        # The producer wrap of the full shard's halo (DomainPartition.wrap_positions,
        # with its device inverse computed once: the same bits, no per-step sync).
        fractional = torch.remainder(small_matmul(self._pos_in, self._wrap_inverse), 1.0)
        self.coords[:n_cap, 3:].copy_(small_matmul(fractional, self._wrap_cell))
        # A forward-only COPY: no private padded reverse-SUM lowering.
        binding = replace(self.prepared.schedule.exchanges[0][1], trailing_shape=(6,),
                          _private_reverse_sum=None)
        received = binding.plan._communicate(self.coords[:n_cap], reverse=False, _prepared=binding)
        self.coords[n_cap:].copy_(received)
        self.positions.copy_(self.coords[:, :3])

    def load_positions(self, positions) -> bool:
        """Copy the caller's owned coordinates into the static input; False if they cannot be used."""
        n = self._layout["n_owned"]
        if (positions.shape != (n, 3) or positions.dtype != torch.float32
                or positions.device != self._pos_in.device):
            return False
        self._pos_in[:n].copy_(positions.detach())
        return True

    # --- step -------------------------------------------------------------

    @torch.no_grad()
    @nvtx_ranged("EpochTopology.step")
    def step(self, state):
        """Fill the padded preparation for ``state`` (same atoms as the epoch); device only."""
        if self._fresh:
            # The refresh derived this step's topology already (the migrated
            # state is the refresh positions).
            self._fresh = False
            self.steps += 1
            return self.prepared
        if not self.load_positions(state.positions):
            raise ValueError("topology epoch: the state's positions do not match the epoch")
        if self._topology_graph is not None:
            # Eager: the topology part (fill, COPY, exact filter, derivation) is
            # one captured graph per layout key, on the persistent buffers.
            with nvtx_range("cudaGraphLaunch"):
                self._topology_graph.replay()
        else:
            self._fill()
            self._derive()
        self.steps += 1
        return self.prepared

    def _predicate(self, b):
        """Every candidate slot's (edge, short) decision with the full shard's exact arithmetic.

        The full shard (``build_distributed_graph``, also the CPU reference
        topology) decides a bond by the norm of its canonical row (lower
        global ID first) and a short row by its own norm, on the
        producer-wrapped coordinates (``neighbor_candidates`` and
        ``_canonical_lengths``, element-wise torch expressions). The same
        expressions on the same model coordinates and images give the same
        bits, so the epoch topology equals the full shard's even at a cutoff tie,
        and the two ranks holding a boundary bond decide it identically.
        Candidate images are converted from the builder's frame (reference
        shift plus wrap change) to the model frame by the integer offsets k.
        """
        rows, width = b.n_real, b.buffer_max
        model = self.coords[:, 3:]
        # w_model = w_builder - k cell per row (integer, exact by rounding).
        self._k.copy_(torch.round(small_matmul(b.wrapped - model, b.inv_cell)))
        k = self._k
        slots = self._cand_slots  # the epoch's candidate slots (row * width + slot), padded
        # Padding entries read slot 0, which may be unfilled (neighbor index
        # n_real): clamp into range; their decisions go to the trash slot.
        source = b.neighbor_matrix.reshape(-1)[slots].long().clamp_(max=rows - 1)
        center = torch.div(slots, width, rounding_mode="floor")
        offset = b.wrap_delta.long() + k
        image = b.neighbor_shifts.reshape(-1, 3)[slots].long() + offset[source] - offset[center]
        cell = b.cell
        displacement = model[source] - model[center]
        displacement = displacement + small_matmul(
            image.float().unsqueeze(-2), cell.expand(len(center), 3, 3)).squeeze(-2)
        row_length = displacement.norm(dim=1)
        gid = b.atom_ids
        sx, sy, sz = image[:, 0], image[:, 1], image[:, 2]
        positive = (sx > 0) | ((sx == 0) & ((sy > 0) | ((sy == 0) & (sz > 0))))
        flip = (gid[center] > gid[source]) | ((gid[center] == gid[source]) & positive)
        first = torch.where(flip, source, center)
        second = torch.where(flip, center, source)
        canonical_image = torch.where(flip[:, None], -image, image)
        canonical = model[second] - model[first]
        canonical = canonical + small_matmul(
            canonical_image.to(model.dtype).unsqueeze(-2), cell.expand(len(first), 3, 3)).squeeze(-2)
        edge = (canonical.norm(dim=1) < self.rc) & ((center != source) | (image != 0).any(dim=1))
        short = edge & (row_length < self.r3)
        # Slots outside the candidate list stay False for the whole epoch;
        # padding entries of the list land on the trash slot past the end.
        with order_independent():
            self._edge_full[self._cand_target] = edge
            self._short_full[self._cand_target] = short
        return (self._edge_full[:rows * width].view(rows, width),
                self._short_full[:rows * width].view(rows, width))

    def _derive(self):
        """The builder's exact graph, then the padded bond_partial tensors (in place, fixed shapes).

        Every derived index is bounded by its capacity even when the builder
        overflowed (the step is then discarded): padding and overflow rows
        read the sink atom, bond and incidence.
        """
        b, buf, lay = self.builder, self._buffers, self._layout
        n_cap, u_cap, local_rows = lay["n_cap"], lay["u_cap"], lay["local_rows"]
        graph, status = b.build(self.positions, rebuild_candidates=False, predicate=self._predicate)
        device = self.positions.device
        sink = n_cap - 1
        edges = status.edge_count[0].clamp(max=b.e_capacity)
        real_u = edges // 2
        overflow = status.overflow[0]
        capacity = (status.edge_count[0] >= b.e_capacity) | (status.triplet_count[0] > b.t_capacity)
        stale = ~status.candidate_valid[0]
        invalid = status.invalid_pairs[0]

        def model_row(rows):
            rows = rows.long()
            return torch.where((rows >= 0) & (rows < local_rows), rows, torch.full_like(rows, sink))

        center, neighbor = b.center.long(), b.neighbor.long()
        safe = lambda rows: rows.clamp(min=0, max=local_rows - 1)
        # Integer offset k of each row between the builder's wrapped frame and
        # the model's (w_model = w_builder - k cell): the model image of a row
        # c -> n is the builder's plus k[n] - k[c] (exact; rows beyond the
        # local atoms are padding).
        model = self.coords[:, 3:]
        k = self._k.to(torch.int32)
        images = b.images + k[safe(neighbor)] - k[safe(center)]
        # Bonds: the canonical directed row of each undirected pair (lower
        # global ID first, as the full shard's canonical keys), so both ranks
        # holding a boundary bond evaluate its geometry identically.
        u_ids = b.u_ids
        first = b.undirected2directed.long().clamp(min=0, max=b.e_capacity - 1)
        second = b.pair_row[first].clamp(min=0, max=b.e_capacity - 1)
        gid_c, gid_n = b.atom_ids[safe(center[first])], b.atom_ids[safe(neighbor[first])]
        img = images[first]
        positive = (img[:, 0] > 0) | ((img[:, 0] == 0) & ((img[:, 1] > 0) | ((img[:, 1] == 0) & (img[:, 2] > 0))))
        flip = (gid_c > gid_n) | ((gid_c == gid_n) & positive)
        canonical = torch.where(flip, second, first)
        valid_u = (u_ids < real_u) & ~overflow
        bond_target = torch.where(valid_u, model_row(center[canonical]), sink)
        bond_source = torch.where(valid_u, model_row(neighbor[canonical]), sink)
        buf["bond_valid_mask"].copy_(valid_u)
        buf["bond_target"].copy_(bond_target)
        buf["bond_source"].copy_(bond_source)
        buf["bond_images"].copy_(torch.where(valid_u[:, None], images[canonical].float(), 0.0))
        buf["atom_target_index"].copy_(torch.stack((bond_target, bond_source), dim=1).reshape(-1))
        buf["atom_source_index"].copy_(torch.stack((bond_source, bond_target), dim=1).reshape(-1))
        buf["owned_positions"].copy_(model[:n_cap])
        real = torch.arange(n_cap, device=device) < self._n_owned_t
        if self.handle_isolated:
            buf["active"].copy_(real & (b.edge_counts[:n_cap] > 0))
        else:
            buf["active"].copy_(real)
        from fastmd.engine.kernels.segment import segment_count
        for prefix, segment in (("atom_target", buf["atom_target_index"]),
                                ("atom_source", buf["atom_source_index"])):
            perm, off = _csr(segment, local_rows)
            buf[f"{prefix}_csr_perm"].copy_(perm)
            buf[f"{prefix}_csr_off"].copy_(off)
            with order_independent():  # integer counts
                buf[f"{prefix}_count"].copy_(segment_count(segment, local_rows))
        missed = torch.zeros((), dtype=torch.bool, device=device)
        if lay["has_line"]:
            i_cap, t_cap = lay["i_cap"], lay["t_cap"]
            real_e = b.edge_ids < edges
            short = real_e & b.short_flags & (b.short_counts[safe(center)] >= 2) & (center < n_cap)
            incidence = torch.cumsum(short.long(), dim=0) - 1
            incidences = short.long().sum()
            # A padding incidence (the sink of the padding triplets) must remain.
            capacity = capacity | (incidences >= i_cap)
            overflow = overflow | (incidences >= i_cap)
            fits = short & (incidence < i_cap - 1) & ~overflow
            slot = torch.where(fits, incidence, i_cap)
            staged = torch.full((i_cap + 1,), sink, dtype=torch.int64, device=device)
            with order_independent():  # every duplicate slot is the dropped trash row i_cap
                buf["incidence_target"].copy_(staged.scatter(0, slot, model_row(center))[:i_cap])
                buf["incidence_source"].copy_(staged.scatter(0, slot, model_row(neighbor))[:i_cap])
                image_rows = torch.zeros((i_cap + 1, 3), dtype=torch.float32, device=device)
                # Rows that are no incidence all land on the dropped trash row i_cap.
                buf["incidence_images"].copy_(image_rows.index_put((slot,), images.float())[:i_cap])
            buf["incidence_valid_mask"].copy_((torch.arange(i_cap, device=device) < incidences) & ~overflow)
            lg = b.line_graph.long()
            triplets = status.triplet_count[0].clamp(max=t_cap)
            valid_t = (b.triplet_ids < triplets) & ~overflow
            left = incidence[lg[:, 2].clamp(min=0, max=b.e_capacity - 1)]
            right = incidence[lg[:, 4].clamp(min=0, max=b.e_capacity - 1)]
            valid_t = (valid_t & (lg[:, 0] < n_cap) & (left >= 0) & (left < i_cap - 1)
                       & (right >= 0) & (right < i_cap - 1))
            buf["line_target_index"].copy_(torch.where(valid_t, lg[:, 1], u_cap - 1))
            buf["line_source_index"].copy_(torch.where(valid_t, lg[:, 3], u_cap - 1))
            buf["line_atom_index"].copy_(torch.where(valid_t, lg[:, 0], sink))
            buf["triplet_left_geometry"].copy_(torch.where(valid_t, left, i_cap - 1))
            buf["triplet_right_geometry"].copy_(torch.where(valid_t, right, i_cap - 1))
            for prefix, segment in (("line_target", buf["line_target_index"]),
                                    ("line_source", buf["line_source_index"])):
                perm, off = _csr(segment, u_cap)
                buf[f"{prefix}_csr_perm"].copy_(perm)
                buf[f"{prefix}_csr_off"].copy_(off)
                with order_independent():  # integer counts
                    buf[f"{prefix}_count"].copy_(segment_count(segment, u_cap))
            buf["line_active"].copy_(valid_t.any())
            if self.context.world_size > 1:
                # RG-20: the line layers' result is used iff some rank has a real
                # triplet (the single-GPU _has_real_line_graph of the whole system).
                self._line_flag.copy_(buf["line_active"].reshape(1))
                torch.distributed.all_reduce(self._line_flag, op=torch.distributed.ReduceOp.MAX,
                                             group=self.context.group)
                buf["line_active"].copy_(self._line_flag[0] > 0)
            # Slots -> this step's bond rows (the sink bond when the pair is no bond now).
            rows = b.edge_rows(self._slot_center, self._slot_neighbor, self._slot_image)
            self._slot_rows.copy_(torch.where(self._slot_valid & ~overflow, rows, -1))
            slot_u = b.directed2undirected.long()[self._slot_rows.clamp(min=0)].clamp(max=u_cap - 1)
            buf["bp_slots"].copy_(torch.where(self._slot_rows >= 0, slot_u, u_cap - 1))
            # Always-on coverage check: a real boundary bond with local
            # triplets must be some slot's bond (else its peer partial is lost).
            boundary = valid_u & ((bond_target < n_cap) != (bond_source < n_cap))
            used = (buf["line_target_count"] + buf["line_source_count"]) > 0
            hit = torch.zeros(u_cap, dtype=torch.bool, device=device)
            hit.index_fill_(0, buf["bp_slots"], True)
            missed = (boundary & used & ~hit).any() & ~overflow
        self._flags.copy_(torch.stack((capacity | (overflow & ~stale & ~invalid), stale, invalid, missed)).long())
        self._counts.copy_(torch.stack((real_u, status.triplet_count[0])))

    def step_counts(self) -> dict:
        """This step's real local bonds and triplets (one small download; after the step's flag round)."""
        bonds, triplets = self._counts.tolist()
        return {"owned_bonds": bonds, "owned_triplets": triplets if self._layout["has_line"] else 0,
                "ghost_atoms": int(self.ghost_position.numel())}




class EpochGraphView:
    """Diagnostic key view of one epoch step (``gather_topology`` / ``freeze_topology`` input).

    The keys of the full shard: owned atom IDs, both directed rows of every
    bond whose canonical first atom (lower ID) is owned here, and the ordered
    triplets of owned centers as (center ID, left row, right row). Images are
    those of the builder's wrapped coordinates. Host tuples, built on access
    only (small systems, oracle and tests).
    """

    def __init__(self, topology):
        b, lay = topology.builder, topology._layout
        n_cap, local_rows = lay["n_cap"], lay["local_rows"]
        edges = int(b.edge_offsets[-1].clamp(max=b.e_capacity))
        ids = b.atom_ids.cpu()
        center = b.center[:edges].long().cpu()
        neighbor = b.neighbor[:edges].long().cpu()
        # Images in the model frame (the producer-wrapped coordinates, as the full shard's keys).
        k = topology._k.cpu()
        safe = lambda rows: rows.clamp(max=local_rows - 1)
        images = b.images[:edges].long().cpu() + k[safe(neighbor)] - k[safe(center)]
        if edges and int(center.max()) >= local_rows:
            raise RuntimeError("topology epoch: the last step overflowed; no key view")
        keys = [(int(ids[c]), int(ids[n]), *map(int, s)) for c, n, s in
                zip(center.tolist(), neighbor.tolist(), images.tolist())]
        owned_first = [(key if key[0] < key[1] or (key[0] == key[1] and key[2:] <= tuple(-v for v in key[2:]))
                        else (key[1], key[0], *(-v for v in key[2:]))) for key in keys]
        owned_ids = set(int(i) for i in ids[:lay["n_owned"]].tolist())
        self.owned_atom_ids = tuple(sorted(owned_ids))
        self.directed_keys = tuple(sorted(key for key, first in zip(keys, owned_first) if first[0] in owned_ids))
        if lay["has_line"]:
            triplets = int(b.pair_offsets[-1].clamp(max=b.t_capacity))
            rows = b.line_graph[:triplets].long().cpu().tolist()
            self.owned_triplet_keys = tuple(sorted((int(ids[row[0]]), keys[row[2]], keys[row[4]]) for row in rows))
        else:
            self.owned_triplet_keys = ()
        self.atom_cutoff, self.line_cutoff = topology.rc, topology.r3
        self.atom_ids = tuple(int(i) for i in ids.tolist() if i >= 0)
