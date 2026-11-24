"""Edges-only P>1 topology of a node-halo exchange: owned and ghost rows of one fixed-capacity builder.

A model whose layers read only node features across an edge (no triplets, no
line graph) runs at P>1 on [owned | ghost] rows: every atom within the halo
(cutoff + skin) of the rank's domain is a ghost row holding its owner's
coordinates, and each edge carries its periodic image. This module keeps
those rows in one ``FixedCapacityGraphBuilder`` with edges-only ghost rows
(``GhostSpec.edges``): each rank's builder decides its edges with the
single-system arithmetic on the single-system coordinates, so the union of
the ranks' edges is the single-system edge set and every row holds the bits
a single-system build gives its atom. No model runs here.

``refresh(state)`` (collective, outside any CUDA graph; device work and a few
small host reads, no per-atom host work): migrate the atoms to their spatial
owners, select every owned atom's halo peers on the device
(``DomainPartition.halo_mask``), size the rank-uniform layout from one
all_gather of the counts, send each ghost row's ID and atomic number once,
fill the ghost rows with the per-step coordinate COPY, refresh the builder's
candidates on the device, and agree the edge capacity (the edges at the
refresh positions, MAX over the ranks).

``step(positions)`` (device only, fixed shapes): the owned coordinates go into
the owned rows, one all_to_all of fixed per-peer blocks copies them into the
peers' ghost rows, and the builder filters the candidates exactly. Each rank's
status flags an edge overflow or a skin violation of the rows it holds; the
caller reduces the flags over the ranks (MAX) and then refreshes on every rank.
"""
from __future__ import annotations

import math
import time

import torch
import torch.distributed as dist

from .partition import AtomState, DomainPartition, migrate_atoms


# The capacity tiers of a refresh that outgrows them: the count plus this headroom, aligned (rank-uniform).
TIER_HEADROOM, TIER_ALIGN = 0.125, 32
# Edges in a model's declared tiers (CapacityStep): the growth of a capacity that the demand outgrows (the first
# capacity is the model's tier of the demand).
EDGE_GROWTH_HEADROOM = 0.08


def _tier(count: int, current: int, headroom: float, align: int) -> int:
    """The capacity of ``count`` rows: ``current`` while it fits, else ``count`` plus headroom, aligned."""
    if count <= current:
        return current
    return -(-math.ceil(count * (1.0 + headroom)) // align) * align


def edge_tier(pairs: int, current: int, step=None) -> int:
    """The undirected-pair capacity of ``pairs`` (edge rows: twice that): ``current`` while it fits; else without
    ``step`` the count plus the topology's headroom (1/8), aligned to 32; with a model's ``step``
    (``CapacityStep``) its tier: the demand times the larger of its thermal factor and, past a first capacity
    (``current`` > 0), ``1 + EDGE_GROWTH_HEADROOM``, plus its minimum padding, rounded up to its pair step."""
    if step is None:
        return _tier(pairs, current, TIER_HEADROOM, TIER_ALIGN)
    if pairs <= current:
        return current
    factor = max(float(step.u_thermal_factor), 1.0 + (EDGE_GROWTH_HEADROOM if current else 0.0))
    return -(-(math.ceil(pairs * factor) + int(step.min_pad_u)) // int(step.u_step)) * int(step.u_step)


class NodeHaloTopology:
    """The [owned | ghost] rows and edges of one rank (see the module docstring).

    ``edges``: the builder's edge rule, ``"receiver"`` (the edges of owned
    centres, each edge on one rank) or ``"either"`` (the edges with an owned
    end; a boundary edge on both ranks). ``padding``: the padded edge rows
    (the builder's ``PaddingRule``; self-loops land on real owned rows).
    ``pad_species``: the atomic number of the unused slots.
    ``max_neighbor_count`` (a ``[1]`` int64 device tensor, or None): the
    builder folds each build's largest per-row edge count into it (the
    builder's option). Capacities (owned slots, each sender's block of ghost
    slots, edges) are rank-uniform and grow by ``headroom`` when a refresh
    exceeds them; they never shrink. ``capacity_step`` (a model's
    ``CapacityStep``, or None): the edges' tiers (:func:`edge_tier`) in place of
    the headroom. ``n_total``: the atoms of the whole system at the last
    refresh.

    Rows: owned slots ``[0, n_cap)`` (the migrated state's atoms, in its ID
    order, then unused slots), then one block of ghost slots per sending
    rank in rank order (its real ghosts first). ``builder.owned_mask`` and
    ``builder.real_mask`` mark the real owned and the real rows.
    """

    def __init__(self, partition: DomainPartition, context, *, cutoff: float, skin: float,
                 edges: str = "receiver", padding=None, pad_species: int = 1, n_dummy: int = 64,
                 headroom: float = TIER_HEADROOM, align: int = TIER_ALIGN, max_neighbor_count=None,
                 capacity_step=None):
        if edges not in ("receiver", "either"):
            raise ValueError(f"edges must be 'receiver' or 'either', got {edges!r}")
        if not (math.isfinite(cutoff) and cutoff > 0 and math.isfinite(skin) and skin > 0):
            raise ValueError("cutoff and skin must be finite and positive")
        if partition.world_size != context.world_size:
            raise ValueError("partition and process-group world sizes differ")
        if not 0 <= headroom or align < 1:
            raise ValueError("headroom must be nonnegative and align positive")
        self.partition, self.context = partition, context
        self.cutoff, self.skin, self.edges = float(cutoff), float(skin), edges
        self.halo = self.cutoff + self.skin
        self.padding, self.pad_species, self.n_dummy = padding, int(pad_species), int(n_dummy)
        self.headroom, self.align = float(headroom), int(align)
        self.max_neighbor_count, self.capacity_step = max_neighbor_count, capacity_step
        self.builder = None
        self.n_cap = self.u_capacity = self.n_owned = self.n_total = 0
        self.blocks = None  # [sender][receiver] ghost-slot capacities, the same on every rank
        self._layout = self._builder_key = None
        self.refreshes = self.layouts = self.builders = 0
        self.stats = {}

    # --- refresh (collective, outside any CUDA graph) ------------------------

    @torch.no_grad()
    def refresh(self, state: AtomState) -> AtomState:
        """Migrate ``state``, select and exchange the ghost rows, refresh the candidates. Collective.

        Returns the migrated state (owned atoms in ID order, the order of the
        owned rows and of ``step``'s positions). The input check is collective;
        a failure on one rank after it (an allocation, say) leaves its peers in
        a collective, as in the topology epoch's refresh: fatal for the group.
        """
        started = time.perf_counter()
        context, partition = self.context, self.partition
        error = None
        if not (isinstance(state.positions, torch.Tensor) and state.positions.is_cuda
                and state.positions.dtype == torch.float32):
            error = "node-halo topology: the positions must be a float32 CUDA tensor"
        if hasattr(context, "collective_check"):
            context.collective_check(error)
        elif error is not None:
            raise ValueError(error)
        state = migrate_atoms(state, partition, context)
        migrated = time.perf_counter()
        positions = state.positions.detach()
        device = positions.device
        world, rank = context.world_size, context.rank
        n = len(state.ids)
        # Every owned atom's halo peers (distance to their domain boxes), on the device.
        if world > 1:
            mask = partition.halo_mask(positions, self.halo, rank)
        else:
            mask = torch.zeros((1, n), dtype=torch.bool, device=device)
        counts = torch.cat((torch.tensor([n], dtype=torch.int64, device=device), mask.sum(1)))
        gathered = torch.empty((world, world + 1), dtype=torch.int64, device=device)
        if world > 1:
            dist.all_gather_into_tensor(gathered, counts, group=context.group)
        else:
            gathered.copy_(counts[None])
        table = gathered.cpu().tolist()  # [owned atoms, sent rows per peer] of every rank
        owned_counts = [row[0] for row in table]
        self.n_total = sum(owned_counts)
        matrix = [row[1:] for row in table]  # matrix[sender][receiver]
        # Rank-uniform capacities (every rank derives them from the same table and history).
        n_cap = max(1, _tier(max(owned_counts), self.n_cap, self.headroom, self.align))
        old = self.blocks or [[0] * world for _ in range(world)]
        blocks = [[_tier(matrix[s][r], old[s][r], self.headroom, self.align) for r in range(world)]
                  for s in range(world)]
        send_splits = list(blocks[rank])
        recv_splits = [blocks[s][rank] for s in range(world)]
        send_counts = list(matrix[rank])
        recv_counts = [matrix[s][rank] for s in range(world)]
        ghost_rows = sum(recv_splits)
        # Rows sent to each peer, in (peer, row) order: the order of the records and of the COPY blocks.
        flat = torch.nonzero(mask.reshape(-1)).reshape(-1)
        send_rows = flat % max(n, 1)
        peer = torch.div(flat, max(n, 1), rounding_mode="floor")
        records = torch.stack((state.ids.to(device=device, dtype=torch.int64)[send_rows],
                               state.numbers.to(device=device, dtype=torch.int64)[send_rows]), dim=1)
        incoming = records.new_empty((sum(recv_counts), 2))
        if world > 1:
            context.all_to_all_single(incoming, records.contiguous(), recv_counts, send_counts)
        selected = time.perf_counter()

        # The layout: owned slots, then each sender's block of ghost slots (real ghosts first).
        self.n_cap, self.blocks = n_cap, blocks
        rows = n_cap + ghost_rows
        layout = (n_cap, tuple(map(tuple, blocks)))
        if self._layout != layout:
            # New capacities (the same event on every rank): new buffers.
            self._layout = layout
            self.layouts += 1
            self._rows = torch.zeros((rows, 3), dtype=torch.float32, device=device)
            self._send_index = torch.zeros(sum(send_splits), dtype=torch.int64, device=device)
            self._reverse_index = torch.zeros(sum(send_splits), dtype=torch.int64, device=device)
            self._send = torch.zeros((sum(send_splits), 3), dtype=torch.float32, device=device)
        self._builder_for(device)
        recv_offsets = [sum(recv_splits[:s]) for s in range(world)]
        ghost_slots = torch.cat([torch.arange(o, o + c, device=device) for o, c in zip(recv_offsets, recv_counts)]
                                or [torch.empty(0, dtype=torch.int64, device=device)])
        send_offsets = torch.tensor([sum(send_splits[:p]) for p in range(world)], dtype=torch.int64, device=device)
        peer_starts = torch.tensor([sum(send_counts[:p]) for p in range(world)], dtype=torch.int64, device=device)
        within = torch.arange(flat.numel(), device=device) - peer_starts[peer]
        # Padded send slots read owned row 0 (its receivers' slots are unused rows).
        self._send_index.zero_()
        self._send_index[send_offsets[peer] + within] = send_rows
        # The same slots for a reverse SUM, padded slots at the discarded row n_cap (HaloLayout).
        self._reverse_index.fill_(n_cap)
        self._reverse_index[send_offsets[peer] + within] = send_rows
        self._send_splits, self._recv_splits = send_splits, recv_splits
        atom_ids = torch.full((rows,), -1, dtype=torch.int64, device=device)
        numbers = torch.full((rows,), self.pad_species, dtype=torch.int64, device=device)
        atom_ids[:n] = state.ids.to(device=device, dtype=torch.int64)
        numbers[:n] = state.numbers.to(device=device, dtype=torch.int64)
        atom_ids[n_cap + ghost_slots] = incoming[:, 0]
        numbers[n_cap + ghost_slots] = incoming[:, 1]
        active = torch.cat((torch.arange(n, device=device), n_cap + ghost_slots))
        self.n_owned = n
        self.ghost_slots = ghost_slots
        self._rows.zero_()
        self._rows[:n].copy_(positions)
        edges = self._install(numbers, atom_ids, active, n)
        if edges > self.builder.e_capacity:
            # More edges than rows (the first refresh, or growth): a builder with the agreed capacity.
            self.u_capacity = (_tier(-(-edges // 2), self.u_capacity, self.headroom, self.align)
                               if self.capacity_step is None else
                               edge_tier(-(-edges // 2), self.u_capacity, self.capacity_step))
            self._builder_for(device)
            edges = self._install(numbers, atom_ids, active, n)
        finished = time.perf_counter()
        self.refreshes += 1
        self.stats = {
            "refresh_ms": (finished - started) * 1000, "migration_ms": (migrated - started) * 1000,
            "halo_ms": (selected - migrated) * 1000, "rows_ms": (finished - selected) * 1000,
            "owned_atoms": n, "ghost_atoms": int(ghost_slots.numel()), "rows": rows, "max_edges": edges,
            "layouts": self.layouts, "builders": self.builders,
        }
        return state

    def _builder_for(self, device):
        """The builder of the current rows and edge capacity (a new one when either changed)."""
        from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder, GhostSpec

        rows = self._rows.shape[0]
        key = (self.n_cap, rows, self.u_capacity)
        if self._builder_key != key:
            self.builder = None
            self.builder = FixedCapacityGraphBuilder(
                cell=self.partition.cell.to(device=device, dtype=torch.float32),
                atomic_numbers=torch.full((rows,), self.pad_species, dtype=torch.int64, device=device),
                u_capacity=max(1, self.u_capacity), t_capacity=0, atom_graph_cutoff=self.cutoff,
                line_graph_cutoff=None, candidate_cutoff=self.halo, n_dummy=self.n_dummy, device=device,
                padding=self.padding, ghosts=GhostSpec(n_owned=self.n_cap, edges=self.edges),
                max_neighbor_count=self.max_neighbor_count)
            self._builder_key = key
            self.builders += 1
        return self.builder

    def _install(self, numbers, atom_ids, active, owned) -> int:
        """The epoch's rows in the builder, its candidates, and the agreed edge count at the refresh positions."""
        builder = self.builder
        builder.set_atoms(numbers, atom_ids, active, owned)
        self._copy_ghosts()
        builder.refresh_candidates(self._rows)
        _, status = builder.build(self._rows, rebuild_candidates=False)
        count = status.edge_count.clone()
        bad = (~status.candidate_capacity_valid).long().reshape(1)
        flags = torch.cat((count, bad))
        if self.context.world_size > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self.context.group)
        edges, truncated = flags.tolist()
        if truncated:
            raise RuntimeError(f"node-halo topology: a candidate row exceeds the builder's neighbor buffer "
                               f"({builder.buffer_max})")
        return int(edges)

    # --- step (device only) ---------------------------------------------------

    def _copy_ghosts(self):
        """The owners' rows into the ghost slots: one all_to_all of the fixed per-peer blocks.

        Directly on the data group, not through the context's
        ``all_to_all_single`` (which may use grouped P2P and keeps host-side
        counters outside a capture): the same collective, eager or captured.
        """
        if self.context.world_size == 1:
            return
        torch.index_select(self._rows, 0, self._send_index, out=self._send)
        dist.all_to_all_single(self._rows[self.n_cap:], self._send, output_split_sizes=self._recv_splits,
                               input_split_sizes=self._send_splits, group=self.context.group)

    @torch.no_grad()
    def step(self, positions: torch.Tensor):
        """This step's graph at the owned ``positions`` (``[n_owned, 3]``, the refreshed state's order).

        Returns the builder's ``(graph, status)``. Device only, fixed shapes.
        The caller may hand in cell-wrapped coordinates (the builder removes
        the lattice jump). Every rank must call it (the COPY is collective);
        the input checks are local, so a rank that raises here leaves its
        peers in the COPY: the caller agrees the step first.
        """
        self.set_positions(positions)
        return self.build_rows()

    def set_positions(self, positions: torch.Tensor) -> None:
        """Write the owned ``positions`` (``[n_owned, 3]``, the refreshed state's order) into the owned rows."""
        if self.builder is None:
            raise RuntimeError("node-halo topology: refresh before the first step")
        if positions.shape != (self.n_owned, 3) or positions.dtype != torch.float32 \
                or positions.device != self._rows.device:
            raise ValueError("node-halo topology: the positions do not match the refreshed owned atoms")
        self._rows[:self.n_owned].copy_(positions)

    def build_rows(self):
        """The graph at the owned rows as written (``set_positions``): the ghost COPY and the builder's
        exact filter. Device only, fixed shapes, collective: what a captured step replays (its shapes
        depend on the layout, not on the owned count)."""
        self._copy_ghosts()
        return self.builder.build(self._rows, rebuild_candidates=False)

    def layout(self):
        """The ghost rows' COPY layout of the current refresh (``fastmd.distributed.node_halo.HaloLayout``)."""
        from .node_halo import HaloLayout

        return HaloLayout(n_cap=self.n_cap, send_index=self._send_index, reverse_index=self._reverse_index,
                          send_splits=self._send_splits, recv_splits=self._recv_splits, group=self.context.group,
                          world_size=self.context.world_size)

    # --- diagnostics ------------------------------------------------------------

    def edge_keys(self) -> torch.Tensor:
        """[E, 5] int64 (centre ID, neighbour ID, image) of the last build's edges (device; diagnostics, tests)."""
        builder = self.builder
        edges = int(builder.edge_offsets[-1].clamp(max=builder.e_capacity))
        center, neighbor = builder.center[:edges].long(), builder.neighbor[:edges].long()
        return torch.cat((builder.atom_ids[center, None], builder.atom_ids[neighbor, None],
                          builder.images[:edges].long()), dim=1)

    def close(self):
        """Drop the builder and the layout keys (the next refresh builds them anew; nothing here holds captured work)."""
        self.builder = None
        self._layout = self._builder_key = None
