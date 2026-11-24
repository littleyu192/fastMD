"""The symmetric boundary-bond partial exchange (step A2, ``exchange_scheme="bond_partial"``).

Every rank keeps all bonds with an owned end (its local bonds); a boundary bond
(ends owned by two ranks) is replicated on both. The local atoms are the owned
atoms followed by their ghosts (the other ends of local bonds). Triplets and
their states stay on the owner of their center. Per interaction block
(``fastmd.models.matris.distributed.runtime.BondPartialRuntime``):

- line attention: rows and the local segment softmax over the local triplets;
  for every line slot (a short boundary bond, the only kind with triplets) the
  local partial (log-sum-exp and softmax-weighted mean per side) goes to the
  other end; both ends merge the same two partials in canonical order (the
  lower rank's first) and update every local bond;
- atom attention: the single-GPU layer on the local graph, with the block
  input's ghost rows; owned segments are complete;
- line refinement: the local target-segment sum; a slot's partial goes to the
  other end, which adds it (one IEEE add);
- atom refinement: as atom attention, with the ghost rows of its input.

Each ghost COPY rides on the preceding line-partial exchange (one fused
binding, two stages per block) and block 0 computes its ghost features
locally, so a step has 1 + 2L exchanges (the rows exchange, removed in step
4.2, had 2 + 8L). No bond or triplet row is routed and no bond provenance is
needed; the slot binding (boundary bonds in (peer, canonical key) order at
both ends) is a function of the topology. Slot detection and admission, the
symmetry checks, the binding and the SUM merge are the runtime's
(``fastmd.distributed.line_slots``); this module keeps the bond-row layout,
the fused exchanges, the attention partials' log-sum-exp merge and the padded
layout. The merge and why it is exact are in
``fastmd.models.matris.op.segment_partials``; the user-level description is
docs/distributed_inference.md.

``MATRIS_A2_CHECK_REPLICAS`` (diagnostic, agreed across ranks): ``1`` compares
both replicas of every boundary bond bitwise after each bond update and
reports the mismatches in the step stats; ``strict`` also checks the slot keys
exactly and fails the evaluation on any replica mismatch.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
import time
from types import MappingProxyType
import weakref

import torch

from fastmd.graph.geometry import small_matmul
from fastmd.models.matris.op.segment_partials import merge_lse_partials
from fastmd.precision import fp32_gemm_mode, gemm_precision_key

from fastmd.distributed.exchange import ExchangePlan
from fastmd.distributed.execution import (
    DistributedModelGuard, ExecutionResult, PreparedExecution, _compact_row_map, _model_stamp,
    _plan_stamps, _shard_stamp, _tensor_stamp, grad_backend, model_program_stamp,
)
from fastmd.distributed.line_slots import (
    LINE_SLOT_MARGIN, asymmetric_peers, bind_slots, find_line_slots, merge_sum, slot_admission_error,
    symmetry_error,
)
from fastmd.distributed.segment_layout import _csr
from fastmd.distributed.schedule import PrevalidatedSchedule

CHECK_REPLICAS_ENV = "MATRIS_A2_CHECK_REPLICAS"
# The tensor key that marks a bond_partial preparation (it survives every copy
# of the prepared tensors, unlike ``stats``).
MARKER = "bp_slots"
# Fewest bond rows and local atom rows of a rank's batches: cuBLAS chooses a
# small-M GEMM kernel below this (bond rows, and the atom-table GEMM of the
# decomposed indexed projection), and the two replicas of a boundary bond on a
# tiny and a large rank would differ in the last bits.
MIN_BOND_ROWS = MIN_ATOM_ROWS = 64


def _check_mode(value) -> str:
    mode = os.environ.get(CHECK_REPLICAS_ENV, "0") if value is None else value
    if mode is True:
        mode = "1"
    if mode in (False, None):
        mode = "0"
    if mode not in ("0", "1", "strict"):
        raise ValueError(f"{CHECK_REPLICAS_ENV} must be 0, 1 or strict")
    return mode


def _widths(model):
    """(atom feature F, line-attention message D_a, line-refinement message D_r)."""
    block = model.interaction_block[0]
    return (block.attn_block_atom_graph.node_res_weight.shape[-1],
            block.attn_block_line_graph.edge_res_weight.shape[-1],
            block.refine_block_line_graph.edge_res_weight.shape[-1])


def _fused_requests(ghost_owners, ghost_rows, slot_peers, slot_rows, owned_counts, atoms, partial):
    """Requests of one fused binding: ghost rows of ``atoms`` units, then slot rows of ``partial`` units.

    A producer's owned array is its owned atom rows (``atoms`` units each)
    followed by its sent slot rows (``partial`` units each); a unit is the
    exchanged width. ``owned_counts`` holds every rank's owned atom count.
    """
    device = ghost_owners.device
    units = torch.arange(atoms, device=device)
    owners = [ghost_owners.repeat_interleave(atoms)]
    rows = [(ghost_rows[:, None] * atoms + units).reshape(-1)]
    units = torch.arange(partial, device=device)
    base = owned_counts[slot_peers] * atoms + slot_rows * partial
    owners.append(slot_peers.repeat_interleave(partial))
    rows.append((base[:, None] + units).reshape(-1))
    return torch.cat(owners), torch.cat(rows)


def prepare_bond_partial(model, shard, context, *, epoch: int | None = None,
                         skip_dead_threebody_tail: bool | None = None,
                         handle_isolated_atoms: bool = True,
                         local_backend: str = "torch_reference",
                         expected_model_program=None,
                         exchange_self_rows: str = "local",
                         model_guard: DistributedModelGuard | None = None,
                         check_replicas=None, slot_cutoff: float | None = None) -> PreparedExecution:
    """Local graph, slot binding and the prevalidated fused schedule of one evaluation.

    Admission: the model stamp or guard precheck, epoch, local backend and
    precision agreement, and one binding round; the result is a
    ``PreparedExecution`` (fastmd.distributed.execution) that
    ``validate_prepared`` and ``run_prepared`` consume. Collective; all ranks
    prepare the same epoch. The only P>1 preparation since step 4.2.
    ``slot_cutoff`` (default: the model's three-body cutoff) bounds the line
    slots; the topology epoch (``fastmd.distributed.epoch``) prepares a
    superset shard with the skin added.
    """
    started = time.perf_counter()
    rank, world = context.rank, context.world_size
    error = None
    model_state = None
    try:
        if epoch is not None and not (type(epoch) is int and epoch >= 0):
            raise ValueError("epoch must be a nonnegative integer or None")
        if local_backend not in ("torch_reference", "native"):
            raise ValueError("local_backend must be 'torch_reference' or 'native'")
        if exchange_self_rows not in ("nccl", "local"):
            raise ValueError("exchange_self_rows must be 'nccl' or 'local'")
        if model_guard is not None:
            if expected_model_program is not None:
                raise ValueError("model_guard carries the admitted program; do not pass expected_model_program")
            if not isinstance(model_guard, DistributedModelGuard) or model_guard.model is not model:
                raise ValueError("model_guard must be a DistributedModelGuard of this model")
            guard_error = model_guard.precheck_error()
            if guard_error is not None:
                raise ValueError(guard_error)
            model_state = model_guard.stamp
        else:
            model_state = _model_stamp(model)
            if expected_model_program is not None and model_program_stamp(model_state) != expected_model_program:
                raise ValueError("model program was mutated after evaluator construction")
        if getattr(shard, "device_topology", None) is None:
            raise ValueError("exchange_scheme='bond_partial' requires the device-topology shard")
        if getattr(shard, "atom_owner_ranks", None) is None or getattr(shard, "atom_owner_rows", None) is None:
            raise ValueError("exchange_scheme='bond_partial' requires atom producer rank/row metadata")
        check = _check_mode(check_replicas)
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        error = str(exc)
        check = "0"
    precision = gemm_precision_key()
    metadata = context._collective((error, epoch, local_backend, precision), "bond_partial.prepare")
    context._raise_errors([entry[0] for entry in metadata])
    if any(entry[1:] != metadata[0][1:] for entry in metadata):
        raise RuntimeError("prepared execution epoch, local backend or matmul precision differs across ranks")
    if skip_dead_threebody_tail is None:
        from fastmd.models.matris.interaction_block import _SKIP_DEAD_THREEBODY_TAIL
        skip_dead_threebody_tail = _SKIP_DEAD_THREEBODY_TAIL

    topology = shard.device_topology
    device, dtype = shard.positions.device, shard.positions.dtype
    atom_count = len(shard.atom_ids)
    owned_old = topology["owned_atom_indices"]
    owned_count = owned_old.numel()
    canonical = topology["canonical_edges"]
    bond_count = canonical.shape[0]
    ids = topology["atom_ids"]
    owners_old = shard.atom_owner_ranks
    triplets = shard.triplet_edge_index
    triplet_count = 0 if triplets is None else int(triplets.shape[0])
    # A triplet-free superset shard (topology epoch) carries only its count.
    triplet_count_only = (getattr(shard, "device_graph_stats", None) or {}).get("triplet_count_only")
    if triplet_count_only is not None:
        triplet_count = int(triplet_count_only)
    center_count = topology["center_target"].numel()

    # Local atoms: owned (sorted by ID, = the owner row order) then ghosts.
    old_target = torch.searchsorted(ids, canonical[:, 0].contiguous())
    old_source = torch.searchsorted(ids, canonical[:, 1].contiguous())
    used = torch.zeros(atom_count, dtype=torch.bool, device=device)
    used.index_fill_(0, old_target, True)
    used.index_fill_(0, old_source, True)
    used.index_fill_(0, owned_old, False)
    ghost_old = torch.where(used)[0]
    # Ghosts grouped by owner (then ID): each peer's ghost rows are one
    # contiguous block, which the padded layout extends in place.
    ghost_old = ghost_old[torch.argsort(owners_old[ghost_old], stable=True)]
    ghost_count = ghost_old.numel()
    local_index = torch.full((atom_count,), -1, dtype=torch.int64, device=device)
    local_index[owned_old] = torch.arange(owned_count, device=device)
    local_index[ghost_old] = owned_count + torch.arange(ghost_count, device=device)
    bond_target, bond_source = local_index[old_target], local_index[old_source]
    target_owner, source_owner = owners_old[old_target], owners_old[old_source]
    foreign = (target_owner != rank).long() + (source_owner != rank).long()
    boundary = foreign == 1
    peer = torch.where(target_owner != rank, target_owner, source_owner)
    cell = shard.cell.to(device=device, dtype=dtype)
    images = canonical[:, 2:].to(dtype)
    with torch.no_grad():
        # Both ends evaluate the canonical row with the same elementwise
        # arithmetic on bitwise-equal halo coordinates (see _canonical_lengths).
        from fastmd.distributed.device_neighbors import _canonical_lengths
        lengths = _canonical_lengths(shard.positions, cell, old_target, old_source,
                                     canonical[:, 2:])
    if slot_cutoff is None:
        slot_cutoff = float(model.config["three_body_cutoff"])
    short = lengths < float(slot_cutoff) + LINE_SLOT_MARGIN
    local_error = None
    invalid = [(foreign == 2).any()]
    if triplet_count and triplet_count_only is None:
        invalid.append(((triplets < 0) | (triplets >= center_count)).any())
    if bool(torch.stack(invalid).any()):
        local_error = ("a local bond has no owned end" if bool(invalid[0])
                       else "triplet_edge_index contains an out-of-bounds incidence")
    elif triplet_count and triplet_count_only is None:
        used_bonds = topology["center_canonical"][triplets.reshape(-1)]
        local_error = slot_admission_error(boundary, short, used_bonds)

    # Boundary bonds and line slots in (peer, canonical key) order, with the
    # per-peer counts and key checksums.
    (boundary_rows, boundary_peer, slots, slot_peer, boundary_counts, boundary_checks, slot_counts,
     slot_checks) = find_line_slots(canonical, boundary, short, peer, world)
    # One round: errors, the global line switch, the check mode, owned counts
    # and the symmetry of the boundary bond and slot sets (count and key
    # checksum per peer pair).
    gathered = context._collective(
        (local_error, triplet_count > 0, check, owned_count, boundary_counts, boundary_checks,
         slot_counts, slot_checks, (bond_count, owned_count + ghost_count)), "bond_partial.topology")
    context._raise_errors([entry[0] for entry in gathered])
    if len({entry[2] for entry in gathered}) != 1:
        raise RuntimeError(f"{CHECK_REPLICAS_ENV} differs across ranks")
    asymmetric = asymmetric_peers([entry[4:8] for entry in gathered],
                                  (boundary_counts, boundary_checks, slot_counts, slot_checks), rank)
    context.collective_check(symmetry_error("bond_partial", asymmetric))
    has_line = any(entry[1] for entry in gathered)
    # Line slots exist only with a global line graph; without one they stay empty.
    if not has_line:
        slots = slots[:0]
        slot_peer = slot_peer[:0]
        slot_counts = [0] * world
    # Rank-uniform: every rank runs the padded layout when one rank is tiny.
    min_row_padding = any(entry[8][0] < MIN_BOND_ROWS or entry[8][1] < MIN_ATOM_ROWS for entry in gathered)
    owned_counts = torch.tensor([entry[3] for entry in gathered], dtype=torch.int64, device=device)

    # Slot binding: my k-th slot for peer p is p's (start of its slots for me) + k.
    slot_total = slots.numel()
    slot_rows = bind_slots(slot_peer, slot_counts, [entry[6] for entry in gathered], rank, device)
    ghost_owners = owners_old[ghost_old]
    ghost_rows = shard.atom_owner_rows[ghost_old]
    atoms_width, attention_width, refinement_width = _widths(model)
    attention_partial = 4 * attention_width
    units = {"attention": math.gcd(atoms_width, attention_partial),
             "refinement": math.gcd(atoms_width, refinement_width)}
    # (plan key, spec) in batch order.
    specs = [("ghosts", (owned_count, ghost_owners, ghost_rows, "bond_partial.ghosts"))]
    if has_line:
        specs.append(("line_partials", (slot_total, slot_peer, slot_rows, "bond_partial.line_partials")))
        for name, width in (("attention", attention_partial), ("refinement", refinement_width)):
            if name == "attention" and len(model.interaction_block) == 1:
                continue  # block 0's line attention sends partials only: never bound
            unit = units[name]
            atoms, partial = atoms_width // unit, width // unit
            owners, rows = _fused_requests(ghost_owners, ghost_rows, slot_peer, slot_rows, owned_counts,
                                           atoms, partial)
            specs.append((f"line_{name}", (owned_count * atoms + slot_total * partial, owners, rows,
                                           f"bond_partial.line_{name}")))
    if check != "0":
        # Both replicas of every boundary bond: the slot binding over the boundary bonds.
        check_rows = bind_slots(boundary_peer, boundary_counts, [entry[4] for entry in gathered], rank, device)
        specs.append(("check", (boundary_rows.numel(), boundary_peer, check_rows, "bond_partial.replica_check")))
    plan_started = time.perf_counter()
    built = dict(zip((key for key, _ in specs),
                     ExchangePlan.from_owner_rows_batch([spec for _, spec in specs], context=context)))
    check_plan = built.pop("check", None)
    plans = built
    plan_ms = (time.perf_counter() - plan_started) * 1000
    if check == "strict" and has_line:
        # Exact slot keys (one int64 COPY): counts and checksums above only
        # detect, with high probability, a differing slot set.
        peer_keys = plans["line_partials"].forward(canonical[slots].contiguous())
        context.collective_check(None if bool((peer_keys == canonical[slots]).all())
                                 else "bond_partial: slot keys differ from the peer's")

    # Tensors of the local graph.
    owned_positions = shard.positions[owned_old]
    active = torch.zeros(owned_count, dtype=torch.bool, device=device)
    active.index_fill_(0, topology["center_owned_target"], True)
    if not handle_isolated_atoms or model.reference_energy is None:
        active = torch.ones_like(active)
    atom_target = torch.stack((bond_target, bond_source), dim=1).reshape(-1)
    atom_source = torch.stack((bond_source, bond_target), dim=1).reshape(-1)
    local_atoms = owned_count + ghost_count
    tensors = {
        "owned_positions": owned_positions,
        "atomic_numbers": shard.atomic_numbers[owned_old].long(),
        "ghost_atomic_numbers": shard.atomic_numbers[ghost_old].long(),
        "active": active, "cell": cell,
        "bond_target": bond_target, "bond_source": bond_source, "bond_images": images,
        "atom_target_index": atom_target, "atom_source_index": atom_source,
        "directed2undirected": torch.arange(bond_count, device=device).repeat_interleave(2),
        MARKER: slots,
        # The own partial is the canonical first operand iff this rank is the lower one.
        "bp_slot_first": slot_peer > rank,
    }
    csr_layouts = [("atom_target", atom_target, local_atoms), ("atom_source", atom_source, local_atoms)]
    if has_line:
        if triplets is None or not triplet_count or triplet_count_only is not None:
            triplets = torch.empty((0, 2), dtype=torch.int64, device=device)
        incidence_rows = topology["triplet_incidence_rows"]
        incidence = _compact_row_map(incidence_rows, center_count)[triplets]
        line_target = topology["center_canonical"][triplets[:, 0]]
        line_source = topology["center_canonical"][triplets[:, 1]]
        tensors.update({
            "incidence_target": local_index[topology["center_target"][incidence_rows]],
            "incidence_source": local_index[topology["center_source"][incidence_rows]],
            "incidence_images": topology["center_images"][incidence_rows].to(dtype),
            "triplet_left_geometry": incidence[:, 0], "triplet_right_geometry": incidence[:, 1],
            "line_target_index": line_target, "line_source_index": line_source,
            "line_atom_index": topology["center_owned_target"][triplets[:, 0]],
        })
        csr_layouts += [("line_target", line_target, bond_count), ("line_source", line_source, bond_count)]
        if bool(((incidence < 0).any() | (tensors["incidence_source"] < 0).any())):
            local_error = "a triplet incidence is not in the compact incidence set"
    from fastmd.models.matris.functions import segment_count
    for prefix, segment, count in csr_layouts:
        tensors[f"{prefix}_csr_perm"], tensors[f"{prefix}_csr_off"] = _csr(segment, count)
        tensors[f"{prefix}_count"] = segment_count(segment, count)
    if check != "0":
        tensors["bp_check_bonds"] = boundary_rows
    context.collective_check(local_error if has_line else None)

    # The prevalidated schedule (forward order; the reverse mirrors it).
    exchanges = [("coordinates", plans["ghosts"], (3,), dtype, device)]
    for index in range(len(model.interaction_block)):
        prefix = f"block_{index}"
        if has_line:
            if index == 0:
                exchanges.append((f"{prefix}.line_attention.partials", plans["line_partials"],
                                  (attention_partial,), dtype, device))
            else:
                exchanges.append((f"{prefix}.line_attention.partials", plans["line_attention"],
                                  (units["attention"],), dtype, device))
            exchanges.append((f"{prefix}.line_refinement.partials", plans["line_refinement"],
                              (units["refinement"],), dtype, device))
        else:
            if index:
                exchanges.append((f"{prefix}.atom_attention.ghosts", plans["ghosts"], (atoms_width,),
                                  dtype, device))
            exchanges.append((f"{prefix}.atom_refinement.ghosts", plans["ghosts"], (atoms_width,),
                              dtype, device))
    from fastmd.engine.dispatch import current_policy
    from fastmd.distributed.config import distributed_resolved
    with distributed_resolved(local_backend).scope():
        policy = current_policy().fingerprint
    generation = getattr(context, "_prepared_execution_generation", 0) + 1
    binding = (epoch, generation, bool(skip_dead_threebody_tail), local_backend, bool(handle_isolated_atoms),
               exchange_self_rows, "bond_partial", has_line, policy)
    bindings = context._collective(binding, "execution.prepared")
    if any(item != bindings[0] for item in bindings):
        raise RuntimeError("prepared generation, epoch or execution flags differ across ranks")
    context._prepared_execution_generation = generation
    schedule = PrevalidatedSchedule.prepare(
        context, exchanges,
        signature=("bond_partial", has_line, bool(skip_dead_threebody_tail), local_backend, precision,
                   tuple(bool(block.last_block) for block in model.interaction_block)),
        self_rows=exchange_self_rows)

    remote = [p for p in range(world) if p != rank]
    element = torch.finfo(dtype).bits // 8
    sent = {name: sum(plan.send_counts[p] for p in remote) * shape[0] * element
            for name, plan, shape, _, _ in exchanges}
    received = sum(plan.receive_counts[p] * shape[0] * element for _, plan, shape, _, _ in exchanges
                   for p in remote)
    stats = {
        "exchange_scheme": "bond_partial", "local_backend": local_backend, "kernel_policy": policy,
        "owned_atoms": owned_count, "ghost_atoms": ghost_count, "local_atoms": local_atoms,
        "local_bonds": bond_count, "boundary_bonds": boundary_rows.numel(),
        "line_slots": slot_total, "line_slots_per_peer": slot_counts,
        "short_bonds": int(short.sum()), "local_triplets": triplet_count, "has_line": has_line,
        "exchange_stages": len(exchanges), "exchanges_per_block": 2 if has_line else 2,
        "check_replicas": check,
        # Remote payload this rank sends per E/F: the forward exchanges and the
        # mirrored reverse transfer of the backward (receive counts reversed).
        "forward_sent_bytes": sent, "forward_sent_bytes_total": sum(sent.values()),
        "backward_sent_bytes_total": received,
        "plan_build_ms": plan_ms, "prepared_generation": generation,
        "compact_triplet_indices": True, "device_entity_indices": True, "owned_triplets": triplet_count,
        "min_row_padding": min_row_padding,
        "exchange_self_rows": exchange_self_rows,
        "model_guard": ("identity precheck in prepare; one full scan per evaluation"
                        if model_guard is not None else "full scans at prepare and run boundary"),
    }
    stats["exchange_bytes_per_step"] = stats["forward_sent_bytes_total"] + stats["backward_sent_bytes_total"]
    frozen = MappingProxyType(tensors)
    from .strategy import BOND_PARTIAL
    prepared = PreparedExecution(
        model, shard, context, epoch, generation, has_line, bool(skip_dead_threebody_tail), local_backend,
        frozen, MappingProxyType(plans), stats, _shard_stamp(shard), model_state,
        (has_line, bool(skip_dead_threebody_tail), local_backend, precision),
        tuple((name, _tensor_stamp(tensor)) for name, tensor in tensors.items()), _plan_stamps(plans),
        schedule=schedule, _model_guard=model_guard, strategy=BOND_PARTIAL)
    # Diagnostic only (outside the schedule and the stamps).
    object.__setattr__(prepared, "_bp_check_plan", check_plan)
    stats["prepare_ms"] = (time.perf_counter() - started) * 1000
    context._prepared_execution_ref = weakref.ref(prepared)
    return prepared


class BondPartialRoutes:
    """The ``routes`` of a ``BondPartialRuntime``: local graph, bindings, partial arithmetic.

    One object per evaluation. It also carries the forward's pending ghost
    rows between the layer hooks: a fused exchange posted by a line layer
    delivers the ghost rows the next atom layer consumes.
    """

    def __init__(self, prepared, dispatcher, initial_ghosts, check=None):
        tensors, plans = prepared.tensors, prepared.plans
        self.prepared, self.dispatcher, self.plans = prepared, dispatcher, plans
        self.has_line = prepared.has_line
        self.block_count = len(prepared.model.interaction_block)
        self.owned_atoms = tensors["owned_positions"].shape[0]
        self.ghost_atoms = tensors["ghost_atomic_numbers"].shape[0]
        self.local_atoms = self.owned_atoms + self.ghost_atoms
        self.slots = tensors[MARKER]
        self.slot_first = tensors["bp_slot_first"]
        self.initial_ghosts = initial_ghosts
        self._check = check
        atoms_width, attention_width, refinement_width = _widths(prepared.model)
        self.attention_width = attention_width
        self.widths = {"attention": 4 * attention_width, "refinement": refinement_width}
        self.atoms_width = atoms_width
        self.units = {name: math.gcd(atoms_width, width) for name, width in self.widths.items()}
        self._pending, self._fused, self._block_input = {}, {}, None
        # Saved-tensor recompute: line edges run inside the rows stages (runtime.py).
        self.fold_line_edges = False
        bonds = tensors["bond_target"].numel()
        self.atom_graph = {
            "target_index": tensors["atom_target_index"], "source_index": tensors["atom_source_index"],
            "num_undirected": bonds,
            "_ops": dispatcher, "_topology_graph_kind": "atom",
            "source_bincount": tensors["atom_source_count"], "target_bincount": tensors["atom_target_count"],
            "_directed2undirected_pair_index": torch.arange(
                2 * bonds, device=self.slots.device).reshape(bonds, 2),
        }
        self.line_graph = {}
        if self.has_line:
            self.line_graph = {
                "target_index": tensors["line_target_index"], "source_index": tensors["line_source_index"],
                "atom_list": tensors["line_atom_index"], "_ops": dispatcher, "_topology_graph_kind": "line",
                "source_bincount": tensors["line_source_count"],
                "target_bincount": tensors["line_target_count"],
            }
        # Padded layout: the sink atom (last owned row) and sink bond are skipped segments.
        padded = "real_atom_mask" in tensors
        skips = {"atom": self.owned_atoms - 1 if padded else -1, "line": bonds - 1 if padded else -1}
        for graph, kind in ((self.atom_graph, "atom"), (self.line_graph, "line")):
            if not graph:
                continue
            for side in ("source", "target"):
                csr = (tensors[f"{kind}_{side}_csr_perm"], tensors[f"{kind}_{side}_csr_off"], skips[kind])
                graph[f"_owner_{side}_csr"] = graph[f"_owner_{side}_reduce_csr"] = csr
        self.batch_graph = {"atom_graph_dict": self.atom_graph, "line_graph_dict": self.line_graph,
                            "directed2undirected": tensors["directed2undirected"], "_ops": dispatcher}
        if self.has_line and "line_active" in tensors:
            # Topology epoch (RG-20): the line layers run on the epoch's
            # structure; their result is used iff some rank has a real triplet
            # this step (collective device flag, like the single-GPU
            # _has_real_line_graph of one system).
            self.batch_graph["_has_real_line_graph"] = tensors["line_active"]
        dispatcher.layout.check_required_keys("atom", self.atom_graph)
        if self.has_line:
            dispatcher.layout.check_required_keys("line", self.line_graph)

    # --- exchange bindings and the fused ghost rows ---

    def binding(self, index: int, layer: str):
        if layer == "attention" and index == 0:
            return self.plans["line_partials"]
        return self.plans[f"line_{layer}"]

    def fused_nodes(self, index: int, layer: str):
        """The owned atom value whose ghost rows the line ``layer``'s exchange carries."""
        if layer == "attention":
            if index == 0:
                return None  # block 0: ghost features are local
            if self._block_input is None:
                raise RuntimeError("bond_partial: block input is unknown")
            return self._block_input
        raise ValueError(layer)

    def pack(self, atoms, sent, index: int, layer: str):
        """The send buffer of a line exchange: owned atom rows, then slot partials."""
        if atoms is None:
            return sent
        unit = self.units[layer]
        return torch.cat((atoms.reshape(-1, unit), sent.reshape(-1, unit)))

    def post_ghosts(self, index: int, layer: str, received):
        fused = not (layer == "attention" and index == 0)
        if not fused:
            return
        rows = self.ghost_atoms * (self.atoms_width // self.units[layer])
        width = self.atoms_width

        def unpack(tensor):
            return tensor[:rows].reshape(-1, width)

        self._pending[(index, "atom_" + layer)] = (received, unpack)

    def _peer(self, received, index: int, layer: str):
        if layer == "attention" and index == 0:
            return received
        rows = self.ghost_atoms * (self.atoms_width // self.units[layer])
        return received[rows:].reshape(-1, self.widths[layer])

    def ghosts(self, view, nodes, layer: str):
        """(value, unpack) giving the ghost rows of ``nodes`` for the atom ``layer`` of ``view``."""
        key = (view.index, "atom_" + layer)
        if layer == "attention":
            if view.index and self.has_line and nodes is not self._block_input:
                raise RuntimeError("bond_partial: atom attention input is not the block input")
        pending = self._pending.pop(key, None)
        if pending is not None:
            return pending
        if view.index == 0 and layer == "attention":
            return self.initial_ghosts, _identity
        if self.has_line:
            raise RuntimeError(f"bond_partial: no ghost rows were exchanged for {view.prefix}")
        value = view.backend.exchange(nodes, self.plans["ghosts"], name=f"{view.prefix}.ghosts")
        return value, _identity

    def block_output(self, index: int, value):
        """The atom refinement's owned output: the next block's input."""
        self._block_input = value

    def set_block_input(self, value):
        self._block_input = value

    # --- partial arithmetic (called inside local stages) ---

    def attention_partials(self, target_lse, target, source_lse, source):
        """[slots, 4 D] = (lse, y) of the target side, then of the source side."""
        slots = self.slots
        return torch.cat((target_lse.index_select(0, slots), target.index_select(0, slots),
                          source_lse.index_select(0, slots), source.index_select(0, slots)), dim=1)

    def merge_attention(self, target, source, sent, received, index: int):
        """Every local bond's (target, source) attention after merging the slot partials."""
        # One canonical-order merge of both sides: (own, peer) per slot, the
        # lower rank's partial first ([slots, 2 D] = target, then source).
        width = self.attention_width
        merged = merge_lse_partials(sent, self._peer(received, index, "attention"), width,
                                    first_is_own=self.slot_first)
        return (target.index_copy(0, self.slots, merged[:, :width]),
                source.index_copy(0, self.slots, merged[:, width:]))

    def merge_refinement(self, partial, received, index: int):
        """The complete target-segment sum: own + peer on the slots (one IEEE add each)."""
        return merge_sum(partial, self.slots, self._peer(received, index, "refinement"))

    def reduce_target(self, values, weight, graph, count: int):
        """``refinement_nodes``' target-segment reduction (dispatcher branch)."""
        dispatcher = graph["_ops"]
        if dispatcher.segment_sum_csr_enabled:
            return dispatcher.segment_sum_graph(values, graph["target_index"], count, graph, "target",
                                                weight=weight, bin_count=graph["target_bincount"])
        return dispatcher.segment_sum(values, graph["target_index"], count, weight=weight,
                                      bin_count=graph["target_bincount"])

    def check(self, label: str, value) -> None:
        if self._check is not None:
            self._check(label, value)


def _identity(tensor):
    return tensor


def _replica_checker(prepared, records):
    """Bitwise comparison of both replicas of every boundary bond (diagnostic)."""
    plan = getattr(prepared, "_bp_check_plan", None)
    if plan is None:
        return None
    rows = prepared.tensors["bp_check_bonds"]

    def check(label, value):
        with torch.no_grad():
            mine = value.tensor.index_select(0, rows).contiguous()
            theirs = plan.forward(mine)
            same = (mine.view(torch.int32) == theirs.view(torch.int32)).all(dim=1)
            difference = (mine - theirs).abs()
            scale = torch.maximum(mine.abs(), theirs.abs()).clamp_min(1e-30)
            records.append({
                "stage": label, "slots": int(rows.numel()), "mismatch_rows": int((~same).sum()),
                "max_abs": float(difference.max()) if rows.numel() else 0.0,
                "max_rel": float((difference / scale).max()) if rows.numel() else 0.0,
            })

    return check


def run_bond_partial_forward(model, context, prepared, dispatcher, *, stress: bool = False) -> ExecutionResult:
    """Eager E/F forward on the autograd backend (``backend.backward(energy)`` gives forces).

    ``stress``: a zero ``[3, 3]`` strain leaf (``ExecutionResult.strain``) deforms every local row, owned
    and ghost, after the coordinates' COPY (which carries the unstrained rows), and this rank's cell, as
    MatRIS's single-GPU graph strains its coordinates and lattice: each row of a rank is strained once, by
    that rank's leaf, and the strain gradients summed over the ranks are the system's (``global_stress``).
    """
    from .runtime import BondPartialRuntime

    tensors, plans, has_line = prepared.tensors, prepared.plans, prepared.has_line
    backend = grad_backend(prepared)
    records = []
    check = _replica_checker(prepared, records)
    positions = backend.input(tensors["owned_positions"], name="owned_positions")
    ghost_positions = backend.exchange(positions, plans["ghosts"], name="coordinates")
    cell = tensors["cell"]
    strain = (backend.input(cell.new_zeros((3, 3)), name="strain") if stress else None)

    def geometry(owned, ghosts, *strained):
        coordinates = torch.cat((owned, ghosts))
        lattice = cell
        if strained:
            symmetric = 0.5 * (strained[0] + strained[0].transpose(0, 1))
            coordinates = coordinates + small_matmul(coordinates, symmetric)
            lattice = cell + small_matmul(cell, symmetric)

        def vector(prefix):
            value = (coordinates.index_select(0, tensors[f"{prefix}_target"])
                     - (coordinates.index_select(0, tensors[f"{prefix}_source"])
                        + small_matmul(tensors[f"{prefix}_images"], lattice)))
            mask = tensors.get(f"{prefix}_valid_mask")
            if mask is not None:
                # Padded layout: padding rows get a finite unit geometry before
                # norm/division (zeroing an undefined gradient is not enough).
                value = torch.where(mask[:, None], value, tensors["padding_unit_vector"])
            return value

        bonds = vector("bond").norm(dim=1)
        if not has_line:
            return bonds
        incidence = vector("incidence")
        return bonds, incidence / incidence.norm(dim=1)[:, None]

    geometry_values = backend.local(geometry, positions, ghost_positions, *(() if strain is None else (strain,)),
                                    name="geometry")
    lengths = geometry_values[0] if has_line else geometry_values
    # Block 0's ghost features are a function of Z only; no exchange. One
    # batch over all local atoms, so that a ghost's row is computed as on its
    # owner (a separate small batch could take another GEMM kernel).
    owned = tensors["atomic_numbers"].shape[0]
    embedded = model.embed_atoms(torch.cat((tensors["atomic_numbers"], tensors["ghost_atomic_numbers"])))
    h = backend.input(embedded[:owned], name="initial_atom_features")
    ghosts = backend.input(embedded[owned:], name="initial_ghost_features")
    b, atom_rbf, line_rbf = backend.local(model.edge_embedding.forward_lengths, lengths, name="bond_embedding")
    if check is not None:
        check("bond_embedding", b)
    q = None
    if has_line:
        q = backend.local(lambda unit: model.three_body_embedding.forward_indexed(
            unit, tensors["triplet_left_geometry"], tensors["triplet_right_geometry"], dispatcher=dispatcher),
            geometry_values[1], name="triplet_embedding")
    from fastmd.engine import saved_recompute
    recompute = getattr(model, "saved_recompute", "off")
    if recompute == "off":
        # saved_tensor_recompute="auto": the evaluator's collective admission decided.
        recompute = saved_recompute.current_distributed_mode()
    routes = BondPartialRoutes(prepared, dispatcher, ghosts, check)
    routes.fold_line_edges = recompute != "off"
    routes.set_block_input(h)
    runtime = BondPartialRuntime(context.comm, backend, routes)
    saved_recompute.check_model(model, recompute, runtime=True)
    # Saved-tensor recompute: recipes are registered and packed inside the rank-local stages.
    with saved_recompute.forward_scope(recompute):
        h, b, q = model.forward_features(routes.batch_graph, h, b, q,
                                         {"atom graph": atom_rbf, "line graph": line_rbf}, runtime=runtime)
    energy = backend.energy_anchor(backend.local(lambda nodes: model.atomic_energy(
        nodes, tensors["atomic_numbers"], tensors["active"], real_mask=tensors.get("real_atom_mask")),
        h, name="owned_energy"))
    stats = dict(prepared.stats)
    stats["forward_implementation"] = "shared_model"
    stats["dispatcher"] = {"policy": dispatcher.policy.fingerprint, "layout": dispatcher.layout.fingerprint,
                           "strict_errors": dispatcher.strict_errors, "backend": dispatcher.backend}
    summary = {key: prepared.stats[key] for key in (
        "owned_atoms", "ghost_atoms", "local_bonds", "boundary_bonds", "line_slots", "line_slots_per_peer",
        "short_bonds", "local_triplets", "exchange_stages", "exchanges_per_block", "forward_sent_bytes",
        "forward_sent_bytes_total", "backward_sent_bytes_total", "exchange_bytes_per_step", "check_replicas")
        if key in prepared.stats}
    if check is not None:
        summary["replicas"] = records
        summary["replica_mismatch_rows_total"] = sum(record["mismatch_rows"] for record in records)
        if prepared.stats.get("check_replicas") == "strict":
            context.collective_check(
                None if not summary["replica_mismatch_rows_total"] else
                f"bond_partial: {summary['replica_mismatch_rows_total']} boundary bond replica rows differ")
    stats["bond_partial"] = summary
    return ExecutionResult(backend, energy, positions, stats, strain=strain)


# --- the padded fixed-shape layout (the topology epoch's TieredCapacity; minimum_rows) ---

def _offsets(counts):
    return [sum(counts[:peer]) for peer in range(len(counts))]


def _blocks(offsets, counts, device):
    """Concatenated ``offsets[p] + arange(counts[p])`` over peers p."""
    return torch.cat([torch.arange(offset, offset + count, device=device, dtype=torch.int64)
                      for offset, count in zip(offsets, counts)]
                     or [torch.empty(0, dtype=torch.int64, device=device)])


def _units(index, width):
    """Each row index r expanded to its ``width`` unit rows r * width + c."""
    return (index[:, None] * width + torch.arange(width, device=index.device)).reshape(-1)


def pad_bond_partial(prepared, capacity, *, keep_check=False, topology=True):
    """The private padded view of a consumed bond_partial preparation.

    Not a new public PreparedExecution: it must not be passed through
    run_prepared. All real rows retain their order; communication padding
    gets unique receive destinations, preserving the exact COPY/reverse-SUM
    law.

    Fixed shapes from the installed capacity: owned atoms ``n`` (padding rows
    after the real ones; the last is the owned sink), per-peer ghost blocks
    (real ghosts then padding, fed by the peer's owned sink), bonds ``u``,
    incidences ``i`` and triplets ``t`` (padding rows reference the sink atom,
    bond and incidence, with masked geometry), and per-peer line slot blocks
    (padding slots send and receive the sink bond's partial). Every exchange
    moves [ghost block | slot block] per peer, so the ghost and slot regions
    of a received buffer have fixed extents. Sinks are the skipped segments of
    the attention and reduction CSRs, and the real-atom mask excludes padding
    from the energy. Local; no communication. The boundary-replica checker
    runs only on the CPU reference topology (``keep_check``: ``minimum_rows``);
    the topology epoch's layout cannot run it, and its strict mode is
    rejected here rather than silently dropped.
    ``topology=False`` (topology epoch): only the atom rows, the per-peer
    ghost and slot blocks and the padded routes; the bond, incidence and
    triplet tensors, their CSRs and the slot bonds are the caller's (every
    slot reads the sink bond here).
    """
    from fastmd.distributed.capacity import _pad

    if not prepared._consumed or prepared.schedule is None:
        raise ValueError("padding requires a consumed, boundary-admitted prepared execution")
    if not keep_check and prepared.stats.get("check_replicas") == "strict":
        # Rank-uniform (the mode is agreed in prepare): every rank raises here.
        raise ValueError(f"{CHECK_REPLICAS_ENV}=strict requires the CPU reference topology (the topology "
                         "epoch's padded layout cannot run the checker)")
    context, t, plans = prepared.context, prepared.tensors, prepared.plans
    device = t["owned_positions"].device
    rows = capacity.local()
    n, u = rows["n"], rows["u"]
    owned, ghosts_real = len(t["owned_positions"]), len(t["ghost_atomic_numbers"])
    ghost_send, ghost_receive = capacity.splits("bp_ghosts")
    real_send, real_receive = plans["ghosts"].send_counts, plans["ghosts"].receive_counts
    ghost_count = sum(ghost_receive)
    ghost_offsets = _offsets(ghost_receive)
    # Real ghost j of peer p (ghosts are grouped by owner) -> ghost_offsets[p] + j.
    ghost_position = _blocks(ghost_offsets, real_receive, device)
    local = torch.cat((torch.arange(owned, device=device), n + ghost_position))
    sink_atom = n - 1

    def remap(index, rows_, fill):
        return _pad(local[index], rows_, fill)

    ghost_numbers = torch.ones(ghost_count, dtype=t["ghost_atomic_numbers"].dtype, device=device)
    ghost_numbers[ghost_position] = t["ghost_atomic_numbers"]
    tensors = {
        "owned_positions": _pad(t["owned_positions"], n),
        "atomic_numbers": _pad(t["atomic_numbers"], n, 1),
        "ghost_atomic_numbers": ghost_numbers,
        "active": _pad(t["active"], n), "cell": t["cell"].clone(),
        "padding_unit_vector": torch.eye(1, 3, dtype=t["owned_positions"].dtype, device=device).reshape(3),
        "real_atom_mask": _pad(torch.ones_like(t["active"], dtype=torch.bool), n),
        "directed2undirected": torch.arange(u, device=device).repeat_interleave(2),
    }
    local_atoms = n + ghost_count
    csr_layouts = []
    if topology:
        bond_target = remap(t["bond_target"], u, sink_atom)
        bond_source = remap(t["bond_source"], u, sink_atom)
        tensors.update({
            "bond_valid_mask": _pad(torch.ones_like(t["bond_target"], dtype=torch.bool), u),
            "bond_target": bond_target, "bond_source": bond_source,
            "bond_images": _pad(t["bond_images"], u),
            "atom_target_index": torch.stack((bond_target, bond_source), dim=1).reshape(-1),
            "atom_source_index": torch.stack((bond_source, bond_target), dim=1).reshape(-1),
        })
        csr_layouts = [("atom_target", tensors["atom_target_index"], local_atoms),
                       ("atom_source", tensors["atom_source_index"], local_atoms)]
    slot_send = slot_receive = ()
    if prepared.has_line:
        i, nt = rows["i"], rows["t"]
        slot_send, slot_receive = capacity.splits("bp_slots")
        if tuple(slot_send) != tuple(slot_receive):
            raise ValueError("bond_partial slot capacities must be symmetric per rank pair")
        slot_offsets = _offsets(slot_send)
        slot_position = _blocks(slot_offsets, plans["line_partials"].send_counts, device)
        slots = torch.full((sum(slot_send),), u - 1, dtype=torch.int64, device=device)
        if topology:
            slots[slot_position] = t[MARKER]
        first = torch.zeros(sum(slot_send), dtype=torch.bool, device=device)
        first[slot_position] = t["bp_slot_first"]
        tensors.update({MARKER: slots, "bp_slot_first": first})
        if topology:
            tensors.update({
                "incidence_valid_mask": _pad(torch.ones_like(t["incidence_target"], dtype=torch.bool), i),
                "incidence_target": remap(t["incidence_target"], i, sink_atom),
                "incidence_source": remap(t["incidence_source"], i, sink_atom),
                "incidence_images": _pad(t["incidence_images"], i),
                "triplet_left_geometry": _pad(t["triplet_left_geometry"], nt, i - 1),
                "triplet_right_geometry": _pad(t["triplet_right_geometry"], nt, i - 1),
                "line_target_index": _pad(t["line_target_index"], nt, u - 1),
                "line_source_index": _pad(t["line_source_index"], nt, u - 1),
                "line_atom_index": _pad(t["line_atom_index"], nt, sink_atom),
            })
            csr_layouts += [("line_target", tensors["line_target_index"], u),
                            ("line_source", tensors["line_source_index"], u)]
    else:
        tensors.update({MARKER: t[MARKER], "bp_slot_first": t["bp_slot_first"]})
    from fastmd.models.matris.functions import segment_count
    for prefix, segment, count in csr_layouts:
        tensors[f"{prefix}_csr_perm"], tensors[f"{prefix}_csr_off"] = _csr(segment, count)
        tensors[f"{prefix}_count"] = segment_count(segment, count)

    # Padded routes: per peer [ghost block | slot block], real rows first.
    bindings = {id(binding.plan): binding for _, binding in prepared.schedule.exchanges}
    real_ghost_sends = bindings[id(plans["ghosts"])].indices[0].split(list(real_send))
    ghost_sends = [torch.cat((rows_, rows_.new_full((cap - count,), sink_atom)))
                   for rows_, cap, count in zip(real_ghost_sends, ghost_send, real_send)]
    ghost_receives = [torch.arange(offset, offset + cap, device=device)
                      for offset, cap in zip(ghost_offsets, ghost_receive)]
    slot_total = sum(slot_send)
    slot_blocks = ([torch.arange(offset, offset + cap, device=device)
                    for offset, cap in zip(_offsets(slot_send), slot_send)] if prepared.has_line else [])
    atoms_width, attention_width, refinement_width = _widths(prepared.model)

    def route(atoms, partial):
        """(owned rows, requested rows, send, receive, send counts, receive counts) of one binding."""
        send, receive, send_counts, receive_counts = [], [], [], []
        for peer in range(context.world_size):
            parts_send, parts_receive = [], []
            if atoms:
                parts_send.append(_units(ghost_sends[peer], atoms))
                parts_receive.append(_units(ghost_receives[peer], atoms))
            if partial:
                parts_send.append(n * atoms + _units(slot_blocks[peer], partial))
                parts_receive.append(ghost_count * atoms + _units(slot_blocks[peer], partial))
            send.append(torch.cat(parts_send))
            receive.append(torch.cat(parts_receive))
            send_counts.append(send[-1].numel())
            receive_counts.append(receive[-1].numel())
        return (n * atoms + slot_total * partial, ghost_count * atoms + slot_total * partial,
                torch.cat(send), torch.cat(receive), tuple(send_counts), tuple(receive_counts))

    shapes = {"ghosts": (1, 0)}
    if prepared.has_line:
        shapes["line_partials"] = (0, 1)
        for name, width in (("attention", 4 * attention_width), ("refinement", refinement_width)):
            unit = math.gcd(atoms_width, width)
            shapes[f"line_{name}"] = (atoms_width // unit, width // unit)
    padded_plans, indices = {}, {}
    for name, plan in plans.items():
        owned_rows, requested_rows, send, receive, send_counts, receive_counts = route(*shapes[name])
        padded_plans[name] = replace(plan, owned_keys=range(owned_rows), requested_keys=range(requested_rows),
                                     send_indices=send, receive_indices=receive, send_counts=send_counts,
                                     receive_counts=receive_counts, _device_indices={})
        indices[id(plan)] = (send, receive)
    lookup = {id(plan): padded_plans[name] for name, plan in plans.items()}
    exchanges = []
    for name, binding in prepared.schedule.exchanges:
        plan, route_indices = lookup[id(binding.plan)], indices[id(binding.plan)]
        padded = replace(binding, plan=plan, indices=route_indices, _identity=binding._plan_identity(plan),
                         _index_versions=tuple(x._version for x in route_indices))
        if prepared.local_backend == "native" and fp32_gemm_mode() == "ieee":
            from fastmd.distributed.exchange import _bind_private_padded_reverse_sum
            padded = _bind_private_padded_reverse_sum(padded)
        exchanges.append((name, padded))
    schedule = replace(prepared.schedule, exchanges=tuple(exchanges), stats={})
    if keep_check and "bp_check_bonds" in t:
        tensors["bp_check_bonds"] = t["bp_check_bonds"]
    stats = {**prepared.stats, "capacity": {"generation": capacity.generation, "rows": dict(rows),
             "ghost_rows": ghost_count, "line_slots": slot_total, "real_owned_atoms": owned,
             "real_ghost_atoms": ghosts_real,
             # The agreed layout's atom-row range over the ranks (None: per-rank exact sizes).
             "node_rows": capacity.node_rows() if hasattr(capacity, "node_rows") else None,
             "padding": "owned/bond/incidence/triplet sinks; per-peer ghost and slot blocks; real-atom mask"}}
    padded_prepared = replace(prepared, shard=None, tensors=MappingProxyType(tensors),
                              plans=MappingProxyType(padded_plans), schedule=schedule, stats=stats,
                              _shard_stamp=(), _tensor_stamps=(), _plan_stamps=())
    if keep_check:
        object.__setattr__(padded_prepared, "_bp_check_plan", getattr(prepared, "_bp_check_plan", None))
    return padded_prepared


class _MinimumRows:
    """The capacity of ``minimum_rows``: exact sizes plus one sink row, at least
    ``MIN_BOND_ROWS`` bonds and ``MIN_ATOM_ROWS`` local atoms (the owned block
    grows); every per-peer ghost and slot block keeps its real size."""

    generation = 0

    def __init__(self, prepared):
        t, plans = prepared.tensors, prepared.plans
        ghost_rows = len(t["ghost_atomic_numbers"])
        self._rows = {"n": max(len(t["owned_positions"]) + 1, MIN_ATOM_ROWS - ghost_rows),
                      "u": max(len(t["bond_target"]) + 1, MIN_BOND_ROWS), "i": 0, "t": 0}
        ghosts = plans["ghosts"]
        self._splits = {"bp_ghosts": (tuple(ghosts.send_counts), tuple(ghosts.receive_counts))}
        if prepared.has_line:
            self._rows.update(i=len(t["incidence_target"]) + 1, t=len(t["triplet_left_geometry"]) + 1)
            slots = plans["line_partials"]
            self._splits["bp_slots"] = (tuple(slots.send_counts), tuple(slots.receive_counts))

    def local(self):
        return self._rows

    def splits(self, name):
        return self._splits[name]


def minimum_rows(prepared):
    """Eager preparation in the padded layout when some rank has fewer than
    ``MIN_BOND_ROWS`` bonds or ``MIN_ATOM_ROWS`` local atoms
    (``stats["min_row_padding"]``, agreed in prepare): its bond and atom row
    batches then reach the batch-invariant GEMM range."""
    return pad_bond_partial(prepared, _MinimumRows(prepared), keep_check=True)
