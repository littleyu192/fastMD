"""The line-partial exchange of a model with a line graph that declares its shard (``ShardSpec``, "line_graph").

A model whose bond updates sum the messages of the triplets at both ends of a
bond (CHGNet's ``BondConv``) runs at P>1 on [owned | ghost | sink] rows: a
rank keeps every edge with an owned end (a boundary bond on the ranks of both
its ends, in both directions), the triplets of its owned centres and the line
graph's padding (sink atoms; padded triplets keep every rank's line graph
non-empty, so no rank skips a layer and its exchanges). A bond's sum over its
triplets is therefore partial at a boundary. A line slot is a boundary bond
shorter than the shard's ``line_slot_cutoff`` plus the skin (the slots of a
topology then cover every bond with triplets until its refresh). At each
declared ``pair_merge`` point (the input of the layer that consumes the sum,
CHGNet: ``bond_conv_layers.t.mlp_out``) both ends of every line slot send
their partial to the other and add the other's: one IEEE add, the same bits at
both ends. A COPY point fused into it (``ExchangePoint.fuse_into``: CHGNet's
next atom features, final before the bond update) rides in the same
collective: its tensor is taken where the model first passes it
(``fuse_from``) and the received ghost rows replace the ghost rows at the
COPY's own module. In FP64 a rank's sum is the single-system evaluation's to
rounding; the node halo alone is not (the bond sums would miss the far end).

The runtime parts of :mod:`fastmd.distributed.line_slots` (slot detection,
the per-peer symmetry checks, the per-step slot-hit symmetry) are shared with
MatRIS's own bond-partial strategy; the rows, edges and ghost COPY are the
node-halo topology's (:mod:`fastmd.distributed.node_halo_topology`) with a line
graph, and the derivative is the node halo's (owned leaf, ghost coordinates
through one differentiable COPY, FP64 sum of the owned real rows' energies).

- CUDA: :class:`LineHaloTopology` (the fixed-capacity builder with
  ``GhostSpec(edges="either")``: single-system arithmetic, triplets at owned
  centres; edge, pair and triplet capacities and the per-rank-pair slot capacities
  rank-uniform) and :class:`LinePartialStrategy` (``NodeHaloStrategy``'s API).
- CPU (Gloo): :class:`LineReferenceTopology`, exact and rebuilt every
  evaluation (the FP64 exactness tests).
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import Tensor

from .device_keys import canonicalize_edges, lexsort_rows
from .line_slots import LINE_SLOT_MARGIN, NO_PAIR, asymmetric_peers, find_line_slots, pair_hit_entries, \
    symmetry_error
from .node_halo import (
    _ACTIVE, ExchangeHooks, HaloLayout, NodeHaloReferenceTopology, _prepared, halo_copy, md_step_flags, summed_stress,
)
from .node_halo_memory import NodeHaloMemoryPredictor
from .node_halo_topology import TIER_ALIGN, TIER_HEADROOM, NodeHaloTopology, _tier
from .partition import AtomState, DomainPartition

_LABEL = "line-partial exchange"


# --- the exchanges of one forward -------------------------------------------------------------------------


class LineLayout:
    """The exchanges of one rank's rows at one refresh: the ghost COPY (``atoms``, a ``HaloLayout``) and the line
    slots, in fixed per-peer blocks.

    ``slot_splits[peer]``: the line slots shared with each peer (a rank-uniform capacity, the same at both ends of
    a rank pair; a padded slot carries zeros); my k-th slot for a peer is that peer's k-th slot for me (both ends
    order their slots by (peer, canonical bond key)). ``slot_pairs`` (``[slots]`` int64): each slot's row among
    the line graph's ``pairs`` (undirected bonds) at this step, ``pairs`` for a padded or absent one (a discarded
    row). ``rows``: the rows the ghost COPY covers (owned slots and ghost rows; sink rows follow). A fused
    collective sends each peer its block of COPY rows, then its block of slot partials.
    """

    def __init__(self, atoms: HaloLayout, slot_splits, slot_pairs: Tensor, pairs: int, positions: dict):
        self.atoms = atoms
        self.n_cap, self.rows = atoms.n_cap, atoms.rows
        self.slot_splits = [int(v) for v in slot_splits]
        self.slots = sum(self.slot_splits)
        self.slot_pairs, self.pairs = slot_pairs, int(pairs)
        self.send_fused = [a + s for a, s in zip(atoms.send_splits, self.slot_splits)]
        self.recv_fused = [a + s for a, s in zip(atoms.recv_splits, self.slot_splits)]
        self._positions = positions

    @property
    def ghost_rows(self) -> int:
        return self.atoms.ghost_rows

    @staticmethod
    def positions(atoms: HaloLayout, slot_splits, device) -> dict:
        """The rows of the fused buffers (static for one set of splits): where each COPY row and slot sits in what a
        rank sends (``send_atoms``, ``send_slots``) and receives (``recv_atoms``, ``recv_slots``), with and without
        the COPY rows (``*_alone``: an unfused pair_merge)."""
        def rows(blocks_first, blocks_second):
            first, second, start = [], [], 0
            for a, s in zip(blocks_first, blocks_second):
                first.append(torch.arange(start, start + a, dtype=torch.int64))
                second.append(torch.arange(start + a, start + a + s, dtype=torch.int64))
                start += a + s
            empty = torch.zeros(0, dtype=torch.int64)
            return (torch.cat(first or [empty]).to(device), torch.cat(second or [empty]).to(device))

        send_atoms, send_slots = rows(atoms.send_splits, slot_splits)
        recv_atoms, recv_slots = rows(atoms.recv_splits, slot_splits)
        _, alone = rows([0] * len(slot_splits), slot_splits)
        return {"send_atoms": send_atoms, "send_slots": send_slots, "recv_atoms": recv_atoms,
                "recv_slots": recv_slots, "slots_alone": alone}

    def _all_to_all(self, output: Tensor, input: Tensor, output_splits, input_splits) -> None:
        atoms = self.atoms
        if atoms.world_size > 1:
            dist.all_to_all_single(output, input, output_split_sizes=output_splits, input_split_sizes=input_splits,
                                   group=atoms.group)
        else:
            output.copy_(input)

    def forward(self, owned: Tensor | None, partial: Tensor) -> tuple[Tensor | None, Tensor]:
        """The ghost rows of ``owned`` (``[n_cap, W]``; None: no COPY rows) and the merged ``partial`` (``[pairs, W]``:
        each line slot's row plus the peer's partial of the same bond)."""
        p, atoms = self._positions, self.atoms
        fused = owned is not None
        width = partial.shape[1:]
        send = self.send_fused if fused else self.slot_splits
        recv = self.recv_fused if fused else self.slot_splits
        sent = partial.new_zeros((sum(send), *width))
        slot_rows = p["send_slots"] if fused else p["slots_alone"]
        extended = torch.cat((partial, partial.new_zeros((1, *width))))
        sent.index_copy_(0, slot_rows, extended.index_select(0, self.slot_pairs))
        if fused:
            sent.index_copy_(0, p["send_atoms"], owned.index_select(0, atoms.send_index))
        received = partial.new_empty((sum(recv), *width))
        self._all_to_all(received, sent, recv, send)
        ghosts = received.index_select(0, p["recv_atoms"]) if fused else None
        peer = received.index_select(0, p["recv_slots"] if fused else p["slots_alone"])
        merged = extended.index_add(0, self.slot_pairs, peer)[: self.pairs]
        return ghosts, merged

    def backward(self, grad_ghosts: Tensor | None, grad_merged: Tensor, fused: bool) -> tuple[Tensor | None, Tensor]:
        """The adjoint of :meth:`forward`: the owners' sums of the ghost rows' gradients (the COPY's reverse SUM) and
        the partial's gradient (its own plus the peer's gradient of each slot's merged row)."""
        p, atoms = self._positions, self.atoms
        width = grad_merged.shape[1:]
        send = self.recv_fused if fused else self.slot_splits
        recv = self.send_fused if fused else self.slot_splits
        sent = grad_merged.new_zeros((sum(send), *width))
        extended = torch.cat((grad_merged, grad_merged.new_zeros((1, *width))))
        sent.index_copy_(0, p["recv_slots"] if fused else p["slots_alone"], extended.index_select(0, self.slot_pairs))
        if fused:
            sent.index_copy_(0, p["recv_atoms"], grad_ghosts)
        received = grad_merged.new_empty((sum(recv), *width))
        self._all_to_all(received, sent, recv, send)
        grad_partial = extended.index_add(0, self.slot_pairs, received.index_select(
            0, p["send_slots"] if fused else p["slots_alone"]))[: self.pairs]
        grad_owned = None
        if fused:
            owned = grad_merged.new_zeros((atoms.n_cap + 1, *width))
            owned.index_add_(0, atoms.reverse_index, received.index_select(0, p["send_atoms"]))
            grad_owned = owned[: atoms.n_cap]
        return grad_owned, grad_partial


class _LineExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, partial, layout):
        if torch._C._current_graph_task_id() != -1:
            raise RuntimeError(f"{_LABEL}: forward inside a backward (a recompute) would re-issue its collective")
        ctx.layout, ctx.fused = layout, owned is not None
        ctx.like = (partial.shape[1:], partial.dtype, partial.device)
        ctx.set_materialize_grads(False)
        ghosts, merged = layout.forward(owned, partial)
        return (merged,) if ghosts is None else (ghosts, merged)

    @staticmethod
    def backward(ctx, *grads):
        layout = ctx.layout
        trailing, dtype, device = ctx.like
        grad_ghosts, grad_merged = grads if ctx.fused else (None, grads[0])
        # An unused output: zeros, so that the collective still runs on every rank.
        if grad_merged is None:
            grad_merged = torch.zeros((layout.pairs, *trailing), dtype=dtype, device=device)
        if ctx.fused and grad_ghosts is None:
            grad_ghosts = torch.zeros((layout.atoms.ghost_rows, *trailing), dtype=dtype, device=device)
        grad_owned, grad_partial = layout.backward(grad_ghosts, grad_merged, ctx.fused)
        return grad_owned, grad_partial, None


@torch._dynamo.disable
def line_exchange(owned: Tensor | None, partial: Tensor, layout: LineLayout):
    """``(ghost rows of owned or None, merged partial)``: one collective (differentiable, see :class:`LineLayout`)."""
    out = _LineExchange.apply(owned, partial, layout)
    return (None, out[0]) if owned is None else out


def expected_calls(points) -> dict:
    """What one forward must call: each declared point once, and each fused COPY's source once."""
    expected = {point.name: 1 for point in points}
    expected.update({f"{point.name}.source": 1 for point in points if point.fuse_into is not None})
    return expected


class _LineExchanges:
    """The exchanges of one P>1 forward of a line-graph shard, which the declared hooks call (``count_only``: count
    and record shapes, change nothing)."""

    def __init__(self, layout: LineLayout | None, points, *, count_only: bool = False):
        self.layout, self.count_only = layout, count_only
        self.fired: dict[str, int] = {}
        self.shapes: dict[str, tuple] = {}
        self._fused = {point.fuse_into: point for point in points if point.fuse_into is not None}
        self._sources: dict[str, Tensor] = {}
        self._received: dict[str, tuple[Tensor, Tensor]] = {}

    def _record(self, name: str, tensor) -> bool:
        self.fired[name] = self.fired.get(name, 0) + 1
        if not isinstance(tensor, Tensor):
            self.shapes[name] = (type(tensor).__name__,)
            if self.count_only:
                return False
            raise TypeError(f"exchange point {name!r}: the hooked value is a {type(tensor).__name__}, not a tensor")
        self.shapes[name] = ((int(tensor.shape[0]), math.prod(tensor.shape[1:])) if tensor.ndim else (-1, -1))
        return not self.count_only

    def source(self, point, tensor) -> None:
        """The fused COPY ``point``'s tensor where the model first passes it (its ``fuse_from``)."""
        if self._record(f"{point.name}.source", tensor):
            self._sources[point.name] = tensor

    def exchange(self, point, tensor):
        if not self._record(point.name, tensor):
            return tensor
        layout = self.layout
        if point.kind == "pair_merge":
            if tensor.ndim == 0 or tensor.shape[0] != layout.pairs:
                raise ValueError(f"exchange point {point.name!r}: {tuple(tensor.shape)} rows, the line graph has "
                                 f"{layout.pairs} bonds")
            copy = self._fused.get(point.name)
            source = None if copy is None else self._sources.pop(copy.name, None)
            if copy is not None and source is None:
                raise RuntimeError(f"exchange point {point.name!r}: the fused COPY {copy.name!r} saw no tensor at "
                                   f"{copy.fuse_from}")
            ghosts, merged = line_exchange(None if source is None else source[: layout.n_cap], tensor, layout)
            if copy is not None:
                self._received[copy.name] = (source, ghosts)
            return merged
        if tensor.ndim == 0 or tensor.shape[0] < layout.rows:
            raise ValueError(f"exchange point {point.name!r}: {tuple(tensor.shape)} rows, the layout has {layout.rows}")
        if point.fuse_into is None:
            ghosts = halo_copy(tensor[: layout.n_cap], layout.atoms)
        else:
            source, ghosts = self._received.pop(point.name, (None, None))
            if source is not tensor:
                # The fused collective sent the tensor seen at fuse_from: another tensor here was not exchanged.
                raise RuntimeError(f"exchange point {point.name!r}: its tensor is not the one passed at "
                                   f"{point.fuse_from} (the fused COPY needs the same tensor object)")
        return torch.cat((tensor[: layout.n_cap], ghosts, tensor[layout.rows:]))


class _SourceHook:
    """Forward pre-hook (with kwargs) at a fused COPY's ``fuse_from``: hands the input to the forward's exchanges;
    holds the point and its module's id only (deepcopy-safe, as the node halo's hooks)."""

    __slots__ = ("point", "target")

    def __init__(self, point, target):
        self.point, self.target = point, id(target)

    def __call__(self, module, args, kwargs):
        exchanges = _ACTIVE.get()
        if exchanges is None or id(module) != self.target or not hasattr(exchanges, "source"):
            return None
        arg = self.point.fuse_from[1]
        if isinstance(arg, str):
            value = kwargs.get(arg)
        else:
            value = args[arg] if -len(args) <= arg < len(args) else None
        exchanges.source(self.point, value)
        return None


class LineExchangeHooks(ExchangeHooks):
    """:class:`~fastmd.distributed.node_halo.ExchangeHooks` plus the source hook of every fused COPY (at its
    ``fuse_from``); the same lifetime, sharing and pass-through outside the strategy's forward."""

    def __init__(self, module, points):
        super().__init__(module, points)
        try:
            for point in self.points:
                if point.fuse_from is None:
                    continue
                path = point.fuse_from[0]
                try:
                    target = module.get_submodule(path)
                except AttributeError as error:
                    raise ValueError(f"exchange point {point.name!r}: no module {path!r} in the model") from error
                self._handles.append(target.register_forward_pre_hook(_SourceHook(point, target), with_kwargs=True))
        except BaseException:
            self.remove()
            raise


class LinePartialDerivative:
    """The engine-owned derivative at P>1 over a line-graph layout (``ShardedEnergyDerivative``'s arithmetic).

    ``derivative(inputs, rows, cell, owned_mask, layout, stress=False)``:
    ``rows`` are the batch's coordinates (owned slots, ghost rows, then sink rows,
    which stay constant), ``layout`` the :class:`LineLayout`. The owned rows are
    the leaf (strained before the coordinates' COPY for the stress), the ghost
    rows their COPY; the declared points exchange through the hooks; returns
    ``(totals, forces [n_cap, 3])`` as ``ShardedEnergyDerivative``.
    """

    def __init__(self, adapter, points):
        self.adapter, self.points = adapter, tuple(points)
        self.expected = expected_calls(self.points)

    def __call__(self, inputs, rows: Tensor, cell: Tensor, owned_mask: Tensor, layout: LineLayout, *,
                 stress: bool = False, n_total=None, first=None):
        # (n_total, first: the node halo's inputs of a pooled readout; a line-graph shard's readout is per atom.)
        from fastmd.engine.energy_derivative import _times

        exchanges = _LineExchanges(layout, self.points)
        with torch.enable_grad(), ExchangeHooks.exchanging(exchanges):
            leaf = rows[: layout.n_cap].detach().requires_grad_(True)
            owned, leaves = leaf, (leaf,)
            if stress:
                strain = torch.zeros((3, 3), dtype=leaf.dtype, device=leaf.device, requires_grad=True)
                symmetric = 0.5 * (strain + strain.transpose(0, 1))
                owned = leaf + _times(leaf, symmetric)
                cell = cell + _times(cell, symmetric)
                leaves = (leaf, strain)
            positions = torch.cat((owned, halo_copy(owned, layout.atoms), rows[layout.rows:].detach()))
            energies = self.adapter.node_energy(inputs, positions, cell)
            if exchanges.fired != self.expected:
                # Agreed before the first exchanging forward: a difference here is the same on every rank.
                raise RuntimeError(f"{_LABEL}: declared exchanges called {exchanges.fired}, expected {self.expected} "
                                   "(closed evaluator or removed hooks?)")
            energies = energies.to(torch.float64)
            local = torch.where(owned_mask, energies, energies.new_zeros(())).sum()
            with torch.autograd.set_multithreading_enabled(False):
                grads = torch.autograd.grad(local, leaves)
        totals = local.detach().reshape(1)
        if stress:
            totals = torch.cat((totals, grads[1].to(torch.float64).reshape(9)))
        return totals, -grads[0]


def count_line_exchanges(adapter, inputs, rows: Tensor, cell: Tensor, layout: LineLayout, points) -> tuple[dict, dict]:
    """One forward without exchanges (no gradient) that counts the declared hooks' calls and shapes (as the node
    halo's ``count_exchanges``)."""
    exchanges = _LineExchanges(layout, points, count_only=True)
    with torch.no_grad(), ExchangeHooks.exchanging(exchanges):
        adapter.node_energy(inputs, rows.detach(), cell)
    return exchanges.fired, exchanges.shapes


def agree_line_exchanges(context, spec, fired: dict, shapes: dict, rows: int, pairs: int) -> None:
    """Every rank's counting forward called each declared point (and each fused COPY's source) once, on ``rows`` rows
    (COPY) or ``pairs`` rows (pair_merge) of the declared width (collective; an error on any rank raises on every
    rank)."""
    expected = expected_calls(spec.exchange_points)
    error = None
    if fired != expected:
        error = f"declared exchanges called {fired}, expected once each ({sorted(expected)})"
    else:
        for point in spec.exchange_points:
            names = [(point.name, pairs if point.kind == "pair_merge" else rows)]
            if point.fuse_into is not None:
                names.append((f"{point.name}.source", rows))
            for name, count in names:
                if shapes.get(name) != (count, point.width):
                    error = f"exchange point {name!r}: rows and width {shapes.get(name)}, expected ({count}, " \
                            f"{point.width})"
                    break
            if error:
                break
    context.collective_check(error)


# --- slots ---------------------------------------------------------------------------------------------------


def boundary_slots(ids: Tensor, centre: Tensor, neighbour: Tensor, images: Tensor, lengths: Tensor, peer: Tensor,
                   short_below: float, world: int):
    """The boundary bonds of one rank (``centre`` owned, ``neighbour`` a ghost row of rank ``peer``; one directed
    edge per bond) in canonical (lower ID first) key order and their line slots (shorter than ``short_below``) in
    (peer, key) order: ``(order, slots)`` with ``order`` the edges' permutation and ``slots`` the
    :class:`~fastmd.distributed.line_slots.LineSlots` over the sorted keys."""
    keys = torch.stack((ids[centre], ids[neighbour], images[:, 0], images[:, 1], images[:, 2]), dim=1).long()
    canonical, _ = canonicalize_edges(keys)
    order = lexsort_rows(canonical)
    canonical = canonical[order]
    found = find_line_slots(canonical, torch.ones_like(order, dtype=torch.bool), lengths[order] < short_below,
                            peer[order], world)
    return order, found


def agree_slots(context, found, label: str = _LABEL) -> list[list[int]]:
    """The per-peer boundary and slot sets agree at both ends of every rank pair (collective; counts and key
    checksums): returns the slot counts ``[sender][receiver]`` of every rank."""
    own = (found.boundary_counts, found.boundary_checks, found.slot_counts, found.slot_checks)
    gathered = context.all_gather_object(own) if context.world_size > 1 else [own]
    context.collective_check(symmetry_error(label, asymmetric_peers(gathered, own, context.rank)))
    return [list(entry[2]) for entry in gathered]


# --- CUDA ------------------------------------------------------------------------------------------------------


def _itself(batch):
    return batch


class LineHaloTopology(NodeHaloTopology):
    """The [owned | ghost | sink] rows, edges, pairs, triplets and line slots of one rank (CUDA, P>1).

    The node-halo topology with a line graph: the builder keeps the edges with
    an owned end (``GhostSpec(edges="either")``, the single-system arithmetic)
    and the triplets of owned centres, padded with sink atoms; its edge, pair
    and triplet capacities are rank-uniform (MAX over the ranks at the refresh
    positions, grown by the headroom; the triplet capacity at least 1). After a
    refresh, the line slots: the boundary bonds shorter than ``slot_cutoff`` +
    skin + ``LINE_SLOT_MARGIN`` at the refresh positions, agreed at both ends
    (counts and key checksums), in per-peer blocks of rank-uniform capacities.
    ``resolve_slots()`` (device, every step) finds each slot's pair at the
    step's build and flags a short boundary bond that is not a slot or a slot
    that is no bond (both impossible within the skin).
    """

    def __init__(self, partition: DomainPartition, context, *, cutoff: float, line_cutoff: float,
                 slot_cutoff: float, skin: float, pad_species: int = 1, n_dummy: int = 64,
                 headroom: float = TIER_HEADROOM, align: int = TIER_ALIGN, positions_dtype: torch.dtype = torch.float32,
                 cell=None):
        from fastmd.adapters.protocol import PaddingRule

        super().__init__(partition, context, cutoff=cutoff, skin=skin, edges="either",
                         padding=PaddingRule("sink_atoms"), pad_species=pad_species, n_dummy=n_dummy,
                         headroom=headroom, align=align, positions_dtype=positions_dtype, cell=cell)
        if not 0 < line_cutoff <= cutoff or not 0 < slot_cutoff:
            raise ValueError("the line-graph and slot cutoffs must be positive, the line graph's within the cutoff")
        self.line_cutoff, self.slot_cutoff = float(line_cutoff), float(slot_cutoff)
        self.t_capacity = 0
        self.slot_blocks = None  # [sender][receiver] slot capacities, symmetric and the same on every rank
        self._slot_layout = None
        self.slot_layouts = 0
        self.slot_counts = []

    def _builder_for(self, device):
        from fastmd.graph.fixed_capacity_builder import FixedCapacityGraphBuilder, GhostSpec

        rows = self._rows.shape[0]
        key = (self.n_cap, rows, self.u_capacity, self.t_capacity)
        if self._builder_key != key:
            self.builder = None
            self.builder = FixedCapacityGraphBuilder(
                **self._builder_cell(device),
                atomic_numbers=torch.full((rows,), self.pad_species, dtype=torch.int64, device=device),
                u_capacity=max(1, self.u_capacity), t_capacity=max(1, self.t_capacity),
                atom_graph_cutoff=self.cutoff, line_graph_cutoff=self.line_cutoff, candidate_cutoff=self.halo,
                n_dummy=self.n_dummy, device=device, padding=self.padding,
                ghosts=GhostSpec(n_owned=self.n_cap, edges="either"), publish_positions=True, prepare=_itself)
            self._builder_key = key
            self.builders += 1
        return self.builder

    def _install(self, numbers, atom_ids, active, owned, floor=None) -> int:
        """The rows in the builder, its candidates, and capacities that hold the refresh positions' edges, pairs and
        triplets on every rank (MAX over the ranks; a new builder until they fit), and at least this rank's
        ``floor`` (``demand``). Returns the agreed edge count."""
        while True:
            builder = self.builder
            builder.set_atoms(numbers, atom_ids, active, owned)
            self._copy_ghosts()
            builder.refresh_candidates(self._rows)
            _, status = builder.build(self._rows, rebuild_candidates=False)
            flags = torch.cat((status.edge_count.long().reshape(1), status.triplet_count.long().reshape(1),
                               (~status.candidate_capacity_valid).long().reshape(1),
                               status.invalid_pairs.long().reshape(1)))
            if floor:
                flags[:2] = torch.maximum(flags[:2], torch.tensor(
                    [int(floor.get("edges", 0)), int(floor.get("triplets", 0))], dtype=flags.dtype,
                    device=flags.device))
            if self.context.world_size > 1:
                dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self.context.group)
            edges, triplets, truncated, unpaired = flags.tolist()
            if truncated:
                raise RuntimeError(f"line-graph topology: a candidate row exceeds the builder's neighbor buffer "
                                   f"({builder.buffer_max})")
            # A padded pair must remain (the padded triplets use one): edges < e_capacity.
            if edges < builder.e_capacity and triplets <= builder.t_capacity:
                if unpaired:  # (an overflowing build truncates its edges: judged only once they fit)
                    raise RuntimeError("line-graph topology: an edge without its reverse at the refresh")
                return int(edges)
            self.u_capacity = _tier(edges // 2 + 1, self.u_capacity, self.headroom, self.align)
            self.t_capacity = _tier(max(triplets, 1), self.t_capacity, self.headroom, self.align)
            self._builder_for(self._rows.device)

    @torch.no_grad()
    def refresh(self, state: AtomState, floor: dict | None = None) -> AtomState:
        state = super().refresh(state, floor)
        self._slots()
        return state

    def demand(self) -> dict:
        """This rank's edges and triplets of the last build, untruncated (``refresh``'s ``floor``; host reads)."""
        builder = self.builder
        return {"edges": int(builder.edge_offsets[-1]), "triplets": int(builder.pair_offsets[-1])}

    def capacities(self) -> tuple:
        """The agreed capacities a step can outgrow (the edges and the triplets)."""
        return (int(self.builder.e_capacity), int(self.builder.t_capacity))

    def _slots(self) -> None:
        """The line slots of the refresh (collective): detection at the refresh build, the agreement of both ends,
        the rank-uniform per-rank-pair capacities and this rank's slot buffers (in place while the capacities
        hold)."""
        builder, context = self.builder, self.context
        world, rank = context.world_size, context.rank
        device = self._rows.device
        edges = int(builder.edge_offsets[-1])
        centre, neighbour = builder.center[:edges].long(), builder.neighbor[:edges].long()
        n_real = builder.n_real
        boundary = torch.where((centre < self.n_cap) & (neighbour >= self.n_cap) & (neighbour < n_real))[0]
        centre, neighbour = centre[boundary], neighbour[boundary]
        images = builder.images[boundary].long()
        lengths = builder.distances[boundary]
        starts = torch.tensor([self.n_cap + sum(self._recv_splits[:s]) for s in range(world)], dtype=torch.int64,
                              device=device)
        peer = torch.searchsorted(starts, neighbour, right=True) - 1
        order, found = boundary_slots(builder.atom_ids, centre, neighbour, images, lengths, peer,
                                      self.slot_cutoff + self.skin + LINE_SLOT_MARGIN, world)
        counts = agree_slots(context, found)
        old = self.slot_blocks or [[0] * world for _ in range(world)]
        blocks = [[_tier(counts[s][r], old[s][r], self.headroom, self.align) for r in range(world)]
                  for s in range(world)]
        self.slot_blocks, self.slot_counts = blocks, counts
        splits = list(blocks[rank])
        layout = (tuple(map(tuple, blocks)), tuple(self._send_splits), tuple(self._recv_splits))
        if self._slot_layout != layout:
            # New capacities (the same event on every rank): new buffers and fused positions.
            self._slot_layout = layout
            self.slot_layouts += 1
            total = sum(splits)
            self.slot_centre = torch.zeros(total, dtype=torch.int64, device=device)
            self.slot_neighbour = torch.zeros(total, dtype=torch.int64, device=device)
            self.slot_images = torch.zeros((total, 3), dtype=torch.int64, device=device)
            self.slot_real = torch.zeros(total, dtype=torch.bool, device=device)
            self.slot_peer = torch.zeros(total, dtype=torch.int64, device=device)
            self.slot_pairs = torch.zeros(total, dtype=torch.int64, device=device)
            self._covered = torch.zeros(builder.u_capacity + 1, dtype=torch.bool, device=device)
            self._positions = LineLayout.positions(HaloLayout(
                n_cap=self.n_cap, send_index=self._send_index, reverse_index=self._reverse_index,
                send_splits=self._send_splits, recv_splits=self._recv_splits, group=context.group,
                world_size=world), splits, device)
        if self._covered.shape[0] != builder.u_capacity + 1:
            self._covered = torch.zeros(builder.u_capacity + 1, dtype=torch.bool, device=device)
        # Slot k of peer p's block <- this rank's k-th slot for p, (peer, key) order.
        rows = order[found.slots]
        slot_peer = found.slot_peer.to(device)
        offsets = torch.tensor([sum(splits[:p]) for p in range(world)], dtype=torch.int64, device=device)
        local = torch.tensor(found.slot_counts, dtype=torch.int64, device=device)
        within = torch.arange(rows.numel(), device=device) - (local.cumsum(0) - local)[slot_peer]
        where = offsets[slot_peer] + within
        self.slot_real.zero_()
        self.slot_centre.zero_()
        self.slot_neighbour.zero_()
        self.slot_images.zero_()
        self.slot_peer.copy_(torch.repeat_interleave(torch.arange(world, device=device),
                                                     torch.tensor(splits, device=device), output_size=sum(splits)))
        self.slot_real[where] = True
        self.slot_centre[where] = centre[rows]
        self.slot_neighbour[where] = neighbour[rows]
        self.slot_images[where] = images[rows]
        self.slot_pairs.fill_(builder.u_capacity)
        self.stats.update({"line_slots": int(rows.numel()), "line_slots_per_peer": list(found.slot_counts),
                           "boundary_bonds": int(boundary.numel()), "slot_capacity": sum(splits),
                           "slot_layouts": self.slot_layouts, "triplet_capacity": builder.t_capacity,
                           "pair_capacity": builder.u_capacity})

    def line_layout(self) -> LineLayout:
        """The exchanges of the current refresh (persistent buffers: what a captured step reads)."""
        return LineLayout(self.layout(), self.slot_blocks[self.context.rank], self.slot_pairs,
                          self.builder.u_capacity, self._positions)

    def resolve_slots(self) -> tuple[Tensor, Tensor]:
        """Each slot's pair at this step's build (into ``slot_pairs``; device only, fixed shapes) and the step's flags:
        ``(bad [1] bool, pair_hits)``: ``bad`` a short boundary bond at an owned centre that is not a resolved slot,
        or a slot without its bond; ``pair_hits`` the per-rank-pair resolved counts for the MAX round
        (``line_slots.pair_hit_entries``)."""
        builder = self.builder
        rows = builder.edge_rows(self.slot_centre, self.slot_neighbour, self.slot_images)
        rows = torch.where(self.slot_real, rows, torch.full_like(rows, -1))
        found = rows >= 0
        pair_of = builder.directed2undirected.long()
        self.slot_pairs.copy_(torch.where(found, pair_of[rows.clamp(min=0)],
                                          torch.full_like(rows, builder.u_capacity)))
        self._covered.zero_()
        self._covered.index_fill_(0, self.slot_pairs, True)
        edge_ids = torch.arange(builder.e_capacity, device=rows.device)
        centre, neighbour = builder.center.long(), builder.neighbor.long()
        short_boundary = ((edge_ids < builder.edge_offsets[-1]) & builder.short_flags & (centre < self.n_cap)
                          & (neighbour >= self.n_cap) & (neighbour < builder.n_real))
        uncovered = (short_boundary & ~self._covered[pair_of.clamp(max=builder.u_capacity)]).any()
        bad = (uncovered | (self.slot_real & ~found).any()).reshape(1)
        hits = pair_hit_entries(self.slot_peer, rows, self.context.rank, self.context.world_size, rows.device)
        return bad, hits

    def slot_coverage(self) -> dict:
        """The last step's short boundary bonds at owned centres and how many are resolved slots (host; diagnostics
        and tests)."""
        builder = self.builder
        edges = int(builder.edge_offsets[-1])
        centre, neighbour = builder.center[:edges].long(), builder.neighbor[:edges].long()
        short = builder.short_flags[:edges] & (centre < self.n_cap) & (neighbour >= self.n_cap) & \
            (neighbour < builder.n_real)
        pairs = builder.directed2undirected[:edges].long()[short]
        covered = self._covered[pairs]
        return {"short_boundary_bonds": int(short.sum()), "covered": int(covered.sum()),
                "slots": int(self.slot_real.sum()), "resolved": int((self.slot_real
                                                                    & (self.slot_pairs < builder.u_capacity)).sum())}

    def close(self):
        super().close()
        self._slot_layout = None

    def edge_keys(self) -> Tensor:
        """[E, 5] int64 (centre ID, neighbour ID, image) of the last build's edges at owned centres (each edge of the
        system on exactly one rank; the ghost centres' edges are their owners'), on the device."""
        keys = super().edge_keys()
        builder = self.builder
        edges = int(builder.edge_offsets[-1].clamp(max=builder.e_capacity))
        return keys[builder.center[:edges].long() < self.n_cap]


class LinePartialStrategy:
    """The line-partial exchange of one rank at P>1 on CUDA (``NodeHaloStrategy``'s API: ``refresh``,
    ``set_positions``, ``step``, ``count``, ``extents``, ``key``, ``close``). ``step``'s first four flags are the
    node halo's, the slots' folded into the stale one (a refresh and a redo); the fifth is the slots' own (a short
    boundary bond that is not a resolved slot, a slot without its bond, or a rank pair whose ends resolved different
    counts), which the evaluator counts. ``step(stress, md)`` and ``refresh(state, floor)``: the in-graph MD step and
    the capacity floor, as the node halo's (the MD step's four flags after the fifth). ``positions_dtype`` and
    ``cell``: the geometry, as the node halo's."""

    def __init__(self, adapter, spec, partition, context, *, skin: float, device,
                 positions_dtype: torch.dtype = torch.float32, cell=None):
        topology = adapter.topology_spec()
        self.adapter, self.spec, self.context = adapter, spec, context
        self.topology = LineHaloTopology(partition, context, cutoff=topology.cutoff,
                                         line_cutoff=topology.triplet_cutoff, slot_cutoff=spec.line_slot_cutoff,
                                         skin=skin, pad_species=spec.pad_species, positions_dtype=positions_dtype,
                                         cell=cell)
        self.derivative = LinePartialDerivative(adapter, spec.exchange_points)
        self.inputs = self.layout = None
        self.n_owned = 0
        self.new_state = torch.zeros(1, dtype=torch.long, device=device)

    def refresh(self, state: AtomState, floor: dict | None = None) -> AtomState:
        topology = self.topology
        state = topology.refresh(state, floor)
        self.layout = topology.line_layout()
        self.inputs, error = _prepared(self.adapter, topology.builder.batch)
        self.context.collective_check(error)
        self.n_owned = topology.n_owned
        return state

    def set_positions(self, positions: Tensor) -> None:
        self.topology.set_positions(positions)

    def step(self, stress: bool = False, md=None):
        topology = self.topology
        builder = topology.builder
        with torch.no_grad():
            if md is not None:
                positions = md.pre(topology.n_cap)
                topology.load_owned_slots(positions)
            _, status = topology.build_rows()
            slot_flag, hits = topology.resolve_slots()
        batch = builder.batch
        totals, forces = self.derivative(self.inputs, batch.positions, batch.cell, builder.owned_mask, self.layout,
                                         stress=stress)
        flags = torch.cat((status.overflow.reshape(-1)[:1].long(), (~status.candidate_valid.reshape(-1)[:1]).long(),
                           (~(torch.isfinite(totals).all() & torch.isfinite(forces).all())).long().reshape(1),
                           self.new_state, slot_flag.long(), hits))
        group = self.context.group
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=group)
        if md is not None:
            flags = torch.cat((flags, md_step_flags(md, totals, forces, status, positions)))
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
        slots = flags[4:5]
        if hits.numel():
            # A rank pair whose ends resolved different slot counts: the slots do not cover the step either.
            agreed = flags[5:5 + hits.numel()].reshape(-1, 2)
            asymmetric = ((agreed[:, 0] != -agreed[:, 1]) & (agreed[:, 0] != NO_PAIR)).any()
            slots = slots | asymmetric.long().reshape(1)
        # The node halo's four flags, the slots' folded into the stale one (a refresh and a redo), then the slots'
        # (and the in-graph MD step's four).
        md_flags = () if md is None else (flags[5 + hits.numel():],)
        flags = torch.cat((flags[:1], flags[1:2] | slots, flags[2:4], slots, *md_flags))
        return totals[:1], forces, flags, summed_stress(totals, batch.cell, forces.dtype)

    def count(self):
        batch = self.topology.builder.batch
        return count_line_exchanges(self.adapter, self.inputs, batch.positions, batch.cell, self.layout,
                                    self.spec.exchange_points)

    def agree(self, fired: dict, shapes: dict) -> None:
        agree_line_exchanges(self.context, self.spec, fired, shapes, self.topology.builder.batch.positions.shape[0],
                             self.layout.pairs)

    def extents(self) -> dict:
        """This rank's sizes of the current layout (the memory admission's extents)."""
        layout, builder = self.layout, self.topology.builder
        return {"owned": self.n_owned, "owned_slots": layout.n_cap, "rows": int(builder.batch.positions.shape[0]),
                "ghost_rows": layout.atoms.ghost_rows, "send_rows": int(layout.atoms.send_index.shape[0]),
                "edges": int(builder.e_capacity), "triplets": int(builder.t_capacity), "slots": layout.slots,
                "line_slots": int(self.topology.stats.get("line_slots", 0)),
                "pair_merges": sum(point.kind == "pair_merge" for point in self.spec.exchange_points)}

    def slot_coverage(self) -> dict:
        return self.topology.slot_coverage()

    @property
    def key(self):
        topology = self.topology
        return (topology.layouts, topology.builders, topology.slot_layouts, id(topology.builder))

    def close(self):
        self.topology.close()


# --- the CPU reference topology (Gloo; exact, rebuilt at every evaluation) -------------------------------


class LineReferenceTopology(NodeHaloReferenceTopology):
    """The line-graph rows, edges, pairs, triplets and line slots of one rank on the CPU: exact, rebuilt at every
    evaluation (Gloo; the FP64 exactness tests).

    The node-halo reference rows and the edges with an owned end (one decision
    per bond: an owned centre's edge to a ghost row, of two owned rows the lower
    ID's direction; then every bond's reverse), their pairs,
    the triplets of the owned centres (ordered pairs of two distinct edges of
    the centre shorter than ``line_cutoff``), one sink row with one padded pair
    and one padded triplet (every rank's line graph is non-empty, so every rank
    runs every layer and exchange), and the line slots (boundary bonds shorter
    than ``slot_cutoff`` + ``LINE_SLOT_MARGIN``) in exact per-peer blocks.
    """

    def __init__(self, partition: DomainPartition, context, *, cutoff: float, line_cutoff: float,
                 slot_cutoff: float, dtype: torch.dtype, pad_species: int = 1, cell=None):
        super().__init__(partition, context, cutoff=cutoff, dtype=dtype, pad_species=pad_species, cell=cell)
        self.line_cutoff, self.slot_cutoff = float(line_cutoff), float(slot_cutoff)

    def refresh(self, state: AtomState) -> AtomState:
        from fastmd.adapters.protocol import GraphBatch, PaddingRule

        state = super().refresh(state)
        context, world = self.context, self.context.world_size
        base, halo = self.batch, self.layout
        rows, cell, n = base.positions, base.cell, self.n_owned
        n_rows = rows.shape[0]
        ids = self.atom_ids
        first_centre, first_neighbour = base.edge_index[:, 0].long(), base.edge_index[:, 1].long()
        first_images = base.edge_image.long()
        # One decision per bond (never a reverse looked up, which a cutoff tie could miss on one side): the owned
        # centres' edges to ghost rows, and of an edge between two owned rows the direction from the lower ID (an
        # atom's bond to its own image: the positive image); every bond's other direction is its reverse.
        x, y, z = first_images[:, 0], first_images[:, 1], first_images[:, 2]
        positive = (x > 0) | ((x == 0) & ((y > 0) | ((y == 0) & (z > 0))))
        centre_id, neighbour_id = ids[first_centre], ids[first_neighbour]
        first = (first_neighbour >= halo.n_cap) | (centre_id < neighbour_id) | ((centre_id == neighbour_id) & positive)
        first_centre, first_neighbour, first_images = first_centre[first], first_neighbour[first], first_images[first]
        pairs = int(first_centre.numel())
        centre = torch.cat((first_centre, first_neighbour))
        neighbour = torch.cat((first_neighbour, first_centre))
        images = torch.cat((first_images, -first_images))
        bond = torch.arange(pairs).repeat(2)
        # Every edge in (centre, neighbour, image) order; the pairs numbered by their first row, which leads them.
        keys = torch.stack((centre, neighbour, images[:, 0], images[:, 1], images[:, 2]), dim=1)
        order = lexsort_rows(keys)
        centre, neighbour, images, bond = centre[order], neighbour[order], images[order], bond[order]
        edges = int(centre.numel())
        leading_row = torch.full((pairs,), edges, dtype=torch.int64).scatter_reduce(
            0, bond, torch.arange(edges), reduce="amin")
        u2d, by_row = torch.sort(leading_row)
        number = torch.empty(pairs, dtype=torch.int64)
        number[by_row] = torch.arange(pairs)
        d2u = number[bond]
        vectors = rows[neighbour] + images.to(rows.dtype) @ cell.to(rows.dtype) - rows[centre]
        lengths = torch.linalg.vector_norm(vectors, dim=1)
        # Triplets at the owned centres: ordered pairs of two distinct short edges of the centre.
        short = (centre < n) & (lengths < self.line_cutoff)
        triplets = []
        for atom in range(n):
            mine = torch.where(short & (centre == atom))[0].tolist()
            triplets += [(atom, int(d2u[a]), a, int(d2u[b]), b) for a in mine for b in mine if a != b]
        # Padding: one sink row, one padded pair (two opposite far edges between the sink and itself) and one padded
        # triplet on it.
        sink = n_rows
        lattice = cell.to(torch.float64)
        far = max(1, math.ceil(2.0 * self.cutoff / max(float(lattice.sum(0).norm()), 1e-6)) + 1)
        edge_index = torch.cat((torch.stack((centre, neighbour), dim=1),
                                torch.tensor([[sink, sink], [sink, sink]]))).to(torch.int32)
        edge_image = torch.cat((images, torch.tensor([[far] * 3, [-far] * 3]))).to(torch.float32)
        d2u = torch.cat((d2u, torch.tensor([pairs, pairs])))
        u2d = torch.cat((u2d, torch.tensor([edges])))
        triplets.append((sink, pairs, edges, pairs, edges))
        positions = torch.cat((rows, rows.new_zeros((1, 3))))
        species = torch.cat((base.species, torch.tensor([self.pad_species], dtype=base.species.dtype)))
        self.batch = GraphBatch(positions=positions, frac_coords=positions.new_zeros(positions.shape), cell=cell,
                                species=species, n_real=int(positions.shape[0]), edge_index=edge_index,
                                edge_image=edge_image, n_edges=None, cutoff=self.cutoff,
                                padding=PaddingRule("sink_atoms"), triplet_cutoff=self.line_cutoff,
                                pair_d2u=d2u.to(torch.int32), pair_u2d=u2d.to(torch.int32),
                                triplets=torch.tensor(triplets, dtype=torch.int32))
        self.owned_mask = torch.cat((self.owned_mask, torch.zeros(1, dtype=torch.bool)))
        # The line slots: boundary bonds (owned centre, ghost neighbour) shorter than the slot cutoff (+ margin).
        ghost_edges = torch.where((centre < n) & (neighbour >= halo.n_cap))[0]
        starts = torch.tensor([halo.n_cap + sum(halo.recv_splits[:s]) for s in range(world)], dtype=torch.int64)
        peer = torch.searchsorted(starts, neighbour[ghost_edges], right=True) - 1
        slot_order, found = boundary_slots(self.atom_ids, centre[ghost_edges], neighbour[ghost_edges],
                                           images[ghost_edges], lengths[ghost_edges], peer,
                                           self.slot_cutoff + LINE_SLOT_MARGIN, world)
        agree_slots(context, found)
        slot_edges = ghost_edges[slot_order[found.slots]]
        slot_pairs = d2u[slot_edges]
        # Every boundary bond with triplets (shorter than the line-graph cutoff) is a slot (collective).
        uncovered = (lengths[ghost_edges] < self.line_cutoff).sum() - (lengths[slot_edges] < self.line_cutoff).sum()
        context.collective_check(f"{_LABEL}: {int(uncovered)} boundary bonds with triplets are not line slots"
                                 if int(uncovered) else None)
        positions_ = LineLayout.positions(halo, found.slot_counts, torch.device("cpu"))
        self.layout = LineLayout(halo, found.slot_counts, slot_pairs, pairs + 1, positions_)
        self.stats.update({"edges": edges, "pairs": pairs, "triplets": len(triplets) - 1,
                           "line_slots": int(slot_edges.numel()), "boundary_bonds": int(ghost_edges.numel())})
        return state


# --- memory -----------------------------------------------------------------------------------------------------


def synthetic_line_batch(adapter, *, rows: int, edges: int, triplets: int, centres: int, cell: Tensor, species: int,
                         dtype: torch.dtype, device, geometry_dtype: torch.dtype | None = None):
    """A graph batch of ``rows`` rows, ``edges`` edge rows (pairs of opposite edges, centres among the first
    ``centres`` rows), their pairs and ``triplets`` triplet rows, at random coordinates in a cube of half the
    line-graph cutoff's edge: the shapes of a rank's line-graph batch, for a memory probe. Its positions and cell are
    in ``geometry_dtype`` (the run's geometry: default ``dtype``)."""
    from fastmd.adapters.protocol import GraphBatch, PaddingRule

    topology = adapter.topology_spec()
    rows, centres = max(2, int(rows)), max(1, min(int(centres), int(rows)))
    pairs, triplets = max(1, int(edges) // 2), max(1, int(triplets))
    generator = torch.Generator(device="cpu").manual_seed(rows * 7919 + pairs * 31 + triplets)
    cartesian = torch.rand((rows, 3), generator=generator, dtype=torch.float64) * (0.5 * float(topology.triplet_cutoff))
    cell = cell.to(torch.float64).cpu()
    index = torch.arange(pairs, dtype=torch.int64)
    centre = index % centres
    neighbour = (centre + 1 + torch.div(index, centres, rounding_mode="floor") % (rows - 1)) % rows
    edge_index = torch.stack((torch.stack((centre, neighbour), 1), torch.stack((neighbour, centre), 1)), 1)
    edge_index = edge_index.reshape(-1, 2)
    d2u = torch.arange(2 * pairs, dtype=torch.int64) // 2
    rows_t = torch.arange(triplets, dtype=torch.int64)
    first = (2 * (rows_t % pairs)) % (2 * pairs)
    second = (first + 2) % (2 * pairs)
    line = torch.stack((edge_index[first, 0], d2u[first], first, d2u[second], second), dim=1)
    geometry_dtype = dtype if geometry_dtype is None else geometry_dtype
    return GraphBatch(positions=cartesian.to(device=device, dtype=geometry_dtype),
                      frac_coords=(cartesian @ torch.linalg.inv(cell)).to(device=device, dtype=dtype),
                      cell=cell.to(device=device, dtype=geometry_dtype),
                      species=torch.full((rows,), int(species), dtype=torch.int32, device=device), n_real=rows,
                      edge_index=edge_index.to(device=device, dtype=torch.int32),
                      edge_image=torch.zeros((2 * pairs, 3), dtype=torch.float32, device=device), n_edges=None,
                      cutoff=float(topology.cutoff), padding=PaddingRule("sink_atoms"),
                      triplet_cutoff=float(topology.triplet_cutoff), pair_d2u=d2u.to(device=device, dtype=torch.int32),
                      pair_u2d=(2 * index).to(device=device, dtype=torch.int32),
                      triplets=line.to(device=device, dtype=torch.int32))


def line_exchange_bytes(points, *, rows: int, ghost_rows: int, send_rows: int, owned_slots: int, pairs: int,
                        slots: int, itemsize: int, coordinate_itemsize: int | None = None) -> int:
    """The buffers of one step's exchanges beyond the model's own: the COPY points' as the node halo's
    (``node_halo_memory.exchange_bytes``, where the rank has ghost rows; the coordinates' items of
    ``coordinate_itemsize`` bytes) and per pair_merge point, ghost rows or not, the slot rows sent, received and
    gathered both ways and the extended and merged partials of every bond (forward and backward)."""
    from .node_halo_memory import exchange_bytes

    copies = [point for point in points if point.kind == "copy"]
    merges = sum(point.width for point in points if point.kind == "pair_merge")
    return (exchange_bytes(copies, rows=rows, ghost_rows=ghost_rows, send_rows=send_rows, owned_slots=owned_slots,
                           itemsize=itemsize, coordinate_itemsize=coordinate_itemsize) if ghost_rows else 0) + \
        itemsize * merges * (8 * slots + 4 * (pairs + 1))


class LineGraphMemoryPredictor(NodeHaloMemoryPredictor):
    """One rank's ``MemoryPredictor`` of a model on the line-partial exchange: the node halo's
    (:class:`~fastmd.distributed.node_halo_memory.NodeHaloMemoryPredictor`) with a triplet term, the dominant one
    of a line-graph model:

        bytes = a + b * edges + c * rows + d * triplets

    measured on the rank's device at the 2 x 2 x 2 grid of a quarter and a half of its edge, row and triplet
    capacities (synthetic line-graph batches; the first point once more, unmeasured, before), plus the exchange
    buffers of its COPY rows and line slots. One rank (the single-system builder) measures the same way."""

    def __init__(self, points, *, model: str, exchange_points, itemsize: int, failure: str | None = None,
                 coordinate_itemsize: int | None = None):
        super().__init__([], model=model, exchange_points=exchange_points, itemsize=itemsize, failure=failure,
                         coordinate_itemsize=coordinate_itemsize)
        self.points = list(points)  # (kind, edges, rows, triplets, peak bytes)
        self.fits = {kind: self._fit(kind) for kind in sorted({point[0] for point in self.points})}

    @classmethod
    def measure(cls, adapter, spec, extents: dict, *, cell: Tensor, device, dtype: torch.dtype,
                kinds=("eager", "capture"), stream=None, single: bool = False,
                geometry_dtype: torch.dtype | None = None) -> "LineGraphMemoryPredictor":
        """Probe ``adapter`` on this rank's device (local: no collective; a probe that fails predicts nothing). The
        model's per-atom energy tap at every point (``single``, one CUDA rank, runs the same tap), with positions and
        cell in ``geometry_dtype`` (the run's geometry: default ``dtype``)."""
        import gc

        from .node_halo_memory import _peaks

        name = getattr(adapter, "name", type(adapter).__name__)
        points, failure = [], None
        edges, rows, triplets = int(extents["edges"]), int(extents["rows"]), int(extents["triplets"])
        axes = [(max(low, value // 4), max(2 * max(low, value // 4), value // 2))
                for value, low in ((edges, 64), (rows, 16), (triplets, 64))]
        grid = [(e, r, t) for e in axes[0] for r in axes[1] for t in axes[2]]
        centres = max(1, int(extents.get("owned_slots", rows)))
        try:
            stream = torch.cuda.Stream(device=device) if stream is None else stream
            for index, (probe_edges, probe_rows, probe_triplets) in enumerate([grid[0]] + grid):
                batch = synthetic_line_batch(adapter, rows=probe_rows, edges=probe_edges, triplets=probe_triplets,
                                             centres=max(1, centres * probe_rows // max(rows, 1)), cell=cell,
                                             species=spec.pad_species, dtype=dtype, device=device,
                                             geometry_dtype=geometry_dtype)
                eager, captured = _peaks(adapter, adapter.prepare(batch), batch, device, "capture" in kinds, stream)
                if index:  # (the first run is the unmeasured warm-up)
                    points += [(kind, probe_edges, probe_rows, probe_triplets, peak)
                               for kind, peak in (("eager", eager), ("capture", captured)) if kind in kinds]
                del batch
        except Exception as error:  # noqa: BLE001 - a probe never raises (its peers would wait in the admission)
            failure = f"memory probe failed: {type(error).__name__}: {(str(error).splitlines() or [''])[0][:200]}"
        try:
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - as above
            failure = failure or f"memory probe failed: {type(error).__name__}: {str(error)[:200]}"
        return cls(points, model=name, exchange_points=spec.exchange_points,
                   itemsize=torch.finfo(dtype).bits // 8, failure=failure,
                   coordinate_itemsize=None if geometry_dtype is None else torch.finfo(geometry_dtype).bits // 8)

    def _fit(self, kind: str) -> dict:
        import numpy as np

        from .node_halo_memory import _MIB, RESIDUAL_FLOOR, RESIDUAL_FRACTION

        points = [point for point in self.points if point[0] == kind]
        x = np.array([[1.0, edges, rows, triplets] for _, edges, rows, triplets, _ in points], dtype=np.float64)
        y = np.array([peak for *_, peak in points], dtype=np.float64)
        bound = max(RESIDUAL_FLOOR, RESIDUAL_FRACTION * (float(y.max()) if len(y) else 0.0))
        fit = {"kind": kind, "points": len(points), "max_edges": max((p[1] for p in points), default=0),
               "max_rows": max((p[2] for p in points), default=0),
               "max_triplets": max((p[3] for p in points), default=0), "residual_bound": bound, "valid": False}
        if len({p[1:4] for p in points}) < 8 or np.linalg.matrix_rank(x) < 4:
            return {**fit, "reason": f"{kind}: {len(points)} probe points, eight distinct ones needed"}
        (a, b, c, d), *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = float(np.abs(x @ np.array([a, b, c, d]) - y).max())
        problems = [why for ok, why in ((b * fit["max_edges"] >= -bound, f"negative bytes per edge {b:.1f}"),
                                        (c * fit["max_rows"] >= -bound, f"negative bytes per row {c:.1f}"),
                                        (d * fit["max_triplets"] >= -bound, f"negative bytes per triplet {d:.1f}"),
                                        (residual <= bound, f"max residual {residual / _MIB:.2f} MiB > "
                                                            f"{bound / _MIB:.2f} MiB (not linear)")) if not ok]
        return {**fit, "a": float(a), "b": max(float(b), 0.0), "c": max(float(c), 0.0), "d": max(float(d), 0.0),
                "max_residual": residual, "valid": not problems,
                "reason": f"{kind}: " + ("; ".join(problems) if problems else
                                         f"{len(points)} points, max residual {residual / _MIB:.2f} MiB")}

    def _reaches(self, fit: dict, extents: dict) -> bool:
        from .node_halo_memory import MAX_EXTRAPOLATION

        return (extents["edges"] <= MAX_EXTRAPOLATION * fit["max_edges"]
                and extents["rows"] <= MAX_EXTRAPOLATION * fit["max_rows"]
                and extents["triplets"] <= MAX_EXTRAPOLATION * fit["max_triplets"])

    def covers(self, extents: dict, kinds=("eager", "capture")) -> bool:
        return all((fit := self.fits.get(kind)) is not None and self._reaches(fit, extents) for kind in kinds)

    @property
    def label(self) -> str:
        return f"line-partial exchange of {self.model!r}, measured on this device"

    def predict(self, level: str, mode: str, extents: dict) -> int | None:
        from .node_halo_memory import HEADROOM_FLOOR, KINDS

        fit = self.fits.get(KINDS[mode])
        if fit is None or not fit["valid"] or not self._reaches(fit, extents):
            return None
        model = fit["a"] + fit["b"] * extents["edges"] + fit["c"] * extents["rows"] + fit["d"] * extents["triplets"]
        headroom = max(2 * fit["max_residual"], HEADROOM_FLOOR)
        # The exchange buffers of a P>1 rank (one that runs pair_merge collectives; every rank does, ghost rows or not).
        exchange = line_exchange_bytes(self.exchange_points, rows=extents["rows"], ghost_rows=extents["ghost_rows"],
                                       send_rows=extents["send_rows"], owned_slots=extents["owned_slots"],
                                       pairs=extents["edges"] // 2, slots=extents.get("slots", 0),
                                       itemsize=self.itemsize, coordinate_itemsize=self.coordinate_itemsize
                                       ) if extents.get("pair_merges") else 0
        return int(math.ceil(max(model, 0.0) + headroom + exchange))

    def describe(self, entry: dict) -> str:
        return (f"{entry['rows']:,} rows ({entry['ghost_rows']:,} ghost), {entry['edges']:,} edge rows, "
                f"{entry.get('triplets', 0):,} triplet rows")

    def summary(self) -> dict:
        return {"predictor": "line_partial_measured", "model": self.model, "failure": self.failure,
                "fits": {kind: dict(fit) for kind, fit in self.fits.items()},
                "points": [{"kind": kind, "edges": edges, "rows": rows, "triplets": triplets, "peak_bytes": peak}
                           for kind, edges, rows, triplets, peak in self.points],
                "model_form": "peak above the allocated baseline = a + b * edges + c * rows + d * triplets (fitted "
                              "on this device) + max(2 x max residual, 16 MiB) + the exchange buffers"}


def line_exchange_schedule(spec, dtype: torch.dtype, ghost_rows: list[int], slots: list[int],
                           coordinate_dtype: torch.dtype | None = None) -> dict:
    """The P>1 step's exchanges of a line-graph shard in order (as the node halo's ``exchange_schedule``): the
    builder's and the derivative's coordinate COPYs (in ``coordinate_dtype``, the geometry's: default ``dtype``),
    then per pair_merge point one collective carrying its line slots' partials (``slots`` of each rank) and the ghost
    rows of the COPY fused into it (backward: the same, reversed), an unfused COPY on its own, then the energy's sum
    and the step flags' MAX."""
    itemsize = torch.finfo(dtype).bits // 8
    name = str(dtype).removeprefix("torch.")
    coordinate_dtype = dtype if coordinate_dtype is None else coordinate_dtype
    coordinate_itemsize = torch.finfo(coordinate_dtype).bits // 8
    coordinate_name = str(coordinate_dtype).removeprefix("torch.")
    fused = {point.fuse_into: point for point in spec.exchange_points if point.fuse_into is not None}
    stages = [{"name": "coordinates.builder", "kind": "copy", "width": 3, "dtype": coordinate_name,
               "forward_bytes": [ghost * 3 * coordinate_itemsize for ghost in ghost_rows],
               "backward_bytes": [0] * len(ghost_rows)},
              {"name": "coordinates", "kind": "copy", "width": 3, "dtype": coordinate_name,
               "forward_bytes": [ghost * 3 * coordinate_itemsize for ghost in ghost_rows],
               "backward_bytes": [ghost * 3 * coordinate_itemsize for ghost in ghost_rows]}]
    for point in spec.exchange_points:
        if point.fuse_into is not None:
            continue
        copy = fused.get(point.name)
        if point.kind == "pair_merge":
            per_rank = [(slot * point.width + (ghost * copy.width if copy is not None else 0)) * itemsize
                        for ghost, slot in zip(ghost_rows, slots)]
            stage = {"name": point.name, "kind": "pair_merge", "width": point.width, "dtype": name,
                     "fused_copy": None if copy is None else {"name": copy.name, "width": copy.width}}
        else:
            per_rank = [ghost * point.width * itemsize for ghost in ghost_rows]
            stage = {"name": point.name, "kind": "copy", "width": point.width, "dtype": name}
        stages.append({**stage, "forward_bytes": per_rank, "backward_bytes": list(per_rank)})
    reductions = [{"name": "energy_and_strain", "kind": "all_reduce_sum", "width": "1 (10 with the stress)",
                   "dtype": "float64"},
                  {"name": "step_flags", "kind": "all_reduce_max",
                   "width": "5 (the node halo's 4 and the line slots'; + 2 per rank pair: the slots' resolved counts, "
                            "up to six ranks)", "dtype": "int64"}]
    return {"exchange_schedule": stages + reductions,
            "bytes_per_step": {direction: [sum(stage[f"{direction}_bytes"][rank] for stage in stages)
                                           for rank in range(len(ghost_rows))] for direction in ("forward", "backward")}}
