"""Canonical owned-center graph construction and frozen-topology test adapters.

Production construction sees only owned atoms and the coordinate halo.  The
explicit frozen/full-graph adapters below are for small correctness oracles.
The rows exchange's triplet routing and bond provenance were removed in step
4.2; the bond_partial exchange reads neither.
"""
from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field, replace
import itertools
import math
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist

from fastmd.graph.geometry import small_matmul
from .partition import AtomState, DomainPartition, exchange_halo, _collective_check

DirectedKey = tuple[int, int, int, int, int]
BondKey = DirectedKey
TripletKey = tuple[int, DirectedKey, DirectedKey]


def reverse_directed(key: DirectedKey) -> DirectedKey:
    return key[1], key[0], -key[2], -key[3], -key[4]


def canonical_bond(key: DirectedKey) -> BondKey:
    return min(key, reverse_directed(key))


@dataclass(frozen=True, eq=False)
class TensorKeyView(Sequence):
    """Lazy diagnostic atom IDs or directed/bond full keys, never hot-path CPU."""

    values: Tensor

    @property
    def tensors(self):
        return (self.values,)

    def __len__(self):
        return len(self.values)

    def _materialize(self):
        rows = self.values.detach().cpu().tolist()
        return tuple(rows) if self.values.ndim == 1 else tuple(tuple(row) for row in rows)

    def __iter__(self):
        return iter(self._materialize())

    def __getitem__(self, index):
        return self._materialize()[index]

    def __eq__(self, other):
        return self._materialize() == tuple(other) if isinstance(other, Sequence) else NotImplemented

    def __reduce__(self):
        return tuple, (self._materialize(),)


@dataclass(frozen=True, eq=False)
class AtomOwnerView(MappingABC):
    """Compatibility-only owner mapping backed by sorted device atom tables."""

    ids: Tensor
    owners: Tensor

    @property
    def tensors(self):
        return self.ids, self.owners

    def __len__(self):
        return len(self.ids)

    def _materialize(self):
        return dict(zip(self.ids.detach().cpu().tolist(), self.owners.detach().cpu().tolist()))

    def __iter__(self):
        return iter(self._materialize())

    def __getitem__(self, key):
        if not isinstance(key, int) or key < 0 or key > torch.iinfo(torch.int64).max:
            raise KeyError(key)
        row = int(torch.searchsorted(self.ids, self.ids.new_tensor(key)).item())
        if row == len(self.ids) or int(self.ids[row].item()) != key:
            raise KeyError(key)
        return int(self.owners[row].item())

    def __reduce__(self):
        return dict, (self._materialize(),)


@dataclass(frozen=True, eq=False)
class TripletKeyView(Sequence[TripletKey]):
    """Diagnostic-only immutable view of device-resident ordered triplets.

    ``len``/``bool`` and ``tensors`` never download or create Python triplet
    objects. Explicit iteration, indexing, comparison or serialization is a
    diagnostic request and materializes a CPU tuple. Production execution must
    consume the backing integer tensors instead. No materialization cache is
    retained, so diagnostics cannot become stale after an invalid buffer edit.
    """

    edge_keys: Sequence[DirectedKey] | None = None
    edge_index: Tensor | None = None
    packet: Tensor | None = None

    def __post_init__(self):
        owned = self.edge_keys is not None and self.edge_index is not None and self.packet is None
        incoming = self.edge_keys is None and self.edge_index is None and self.packet is not None
        if not (owned or incoming):
            raise ValueError("triplet view requires either edge keys/indices or an incoming packet")
        tensor, width = (self.edge_index, 2) if owned else (self.packet, 14)
        if tensor.ndim != 2 or tensor.shape[1] != width or tensor.dtype != torch.int64:
            raise ValueError("triplet view requires an int64 matrix with the expected width")

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return (self.edge_index,) if self.edge_index is not None else (self.packet,)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def _materialize(self) -> tuple[TripletKey, ...]:
        if self.edge_index is not None:
            edges = tuple(self.edge_keys)
            return tuple((edges[left][0], edges[left], edges[right])
                         for left, right in self.edge_index.detach().cpu().tolist())
        return tuple((row[0], (row[0], row[1], *row[2:5]),
                      (row[0], row[5], *row[6:9]))
                     for row in self.packet[:, 1:10].detach().cpu().tolist())

    def __iter__(self):
        return iter(self._materialize())

    def __getitem__(self, index):
        return self._materialize()[index]

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return self._materialize() == tuple(other)
        return NotImplemented

    def __reduce__(self):
        # Explicit diagnostic object collectives must never pickle CUDA tensors
        # and deserialize them on a different rank/device.
        return tuple, (self._materialize(),)


@dataclass
class LocalGraphShard:
    rank: int
    atom_ids: Sequence[int]
    positions: Tensor
    atomic_numbers: Tensor
    cell: Tensor
    atom_owners: Mapping[int, int]
    owned_atom_ids: Sequence[int]
    center_directed_keys: Sequence[DirectedKey]
    owned_bond_keys: Sequence[BondKey]
    owned_triplet_keys: Sequence[TripletKey]
    directed_keys: Sequence[DirectedKey]
    required_atom_ids: Sequence[int]
    required_bond_keys: Sequence[BondKey]
    # The routed line rows of the rows exchange (removed in step 4.2), and
    # their incidence and producer provenance: unset on every shard built
    # now; the fields stay defaulted for one version.
    line_triplet_keys: Sequence[TripletKey] = ()
    # Production triplets reference rows of sorted center_directed_keys; they
    # stay on this device. The independent frozen/local oracle retains Python
    # keys and CPU metadata.
    triplet_edge_index: Tensor | None = None
    line_source_index: Tensor | None = None
    line_target_index: Tensor | None = None
    line_owner_ranks: Tensor | None = None
    line_owner_rows: Tensor | None = None
    # Evaluation-local producer provenance of the local atoms, aligned with
    # atom_ids (the ghost COPY's owner rows); on the coordinate device in
    # production shards. The center bond arrays (the rows exchange's bond
    # provenance) are unset.
    atom_owner_ranks: Tensor | None = None
    atom_owner_rows: Tensor | None = None
    center_bond_owner_ranks: Tensor | None = None
    center_bond_owner_rows: Tensor | None = None
    device_topology: Mapping[str, Tensor] | None = None
    device_graph_stats: dict[str, Any] = field(default_factory=dict)
    atom_cutoff: float = 0.0
    line_cutoff: float = 0.0
    graph_timings: dict[str, float] = field(default_factory=dict)

    @property
    def triplet_left_bond_keys(self) -> tuple[BondKey, ...]:
        return tuple(canonical_bond(q[1]) for q in self.owned_triplet_keys)

    @property
    def triplet_right_bond_keys(self) -> tuple[BondKey, ...]:
        return tuple(canonical_bond(q[2]) for q in self.owned_triplet_keys)

    @property
    def owned_atom_indices(self) -> Tensor:
        if self.device_topology is not None:
            return self.device_topology["owned_atom_indices"]
        rows = {key: row for row, key in enumerate(self.atom_ids)}
        return torch.tensor([rows[key] for key in self.owned_atom_ids],
                            dtype=torch.long, device=self.positions.device)

    @property
    def active_atom_mask(self) -> Tensor:
        if self.device_topology is not None:
            return torch.bincount(self.device_topology["center_owned_target"],
                                  minlength=len(self.owned_atom_ids)) > 0
        active = {key[0] for key in self.center_directed_keys}
        return torch.tensor([key in active for key in self.owned_atom_ids],
                            dtype=torch.bool, device=self.positions.device)

    @property
    def owned_active_mask(self) -> Tensor:
        return self.active_atom_mask

    @property
    def owned_center_directed_keys(self) -> tuple[DirectedKey, ...]:
        return self.center_directed_keys

    def bond_owner(self, key: BondKey) -> int:
        return self.atom_owners[canonical_bond(key)[0]]


def _make_shard(coordinates: AtomState, owned_ids: Iterable[int], partition: DomainPartition,
                rank: int, directed: Iterable[DirectedKey], triplets: Iterable[TripletKey],
                atom_cutoff: float, line_cutoff: float) -> LocalGraphShard:
    coordinates = coordinates.sorted()
    ids = tuple(coordinates.ids.detach().cpu().tolist())
    owner_values = coordinates.payload.get("_owner")
    if owner_values is None:
        owner_values = partition.owner_of(coordinates.positions)
    owners = dict(zip(ids, owner_values.detach().cpu().tolist()))
    producer_rows = coordinates.payload.get("_owner_row")
    atom_owner_ranks = atom_owner_rows = None
    if producer_rows is not None:
        if coordinates.payload.get("_owner") is None:
            raise ValueError("atom producer rows require sender owner provenance")
        for name, value in (("owner", owner_values), ("owner_row", producer_rows)):
            if value.shape != (len(ids),) or value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"atom {name} provenance must be one integer per atom")
        atom_owner_ranks = owner_values.detach().to(device="cpu", dtype=torch.long).clone()
        atom_owner_rows = producer_rows.detach().to(device="cpu", dtype=torch.long).clone()
        if ((atom_owner_ranks < 0).any() or (atom_owner_ranks >= partition.world_size).any()
                or (atom_owner_rows < 0).any()):
            raise ValueError("atom producer provenance is out of range")
    owned_ids = tuple(sorted(owned_ids))
    directed = tuple(sorted(directed))
    triplets = tuple(sorted(triplets))
    # A directed incidence participates in many triplets. Canonicalize once per
    # edge rather than repeatedly rebuilding/reversing keys for every line row.
    canonical = {key: canonical_bond(key) for key in directed}
    owned_bonds = tuple(sorted({bond for bond in canonical.values() if owners[bond[0]] == rank}))
    triplet_edges = {key for q in triplets for key in (q[1], q[2])}
    required_bonds = tuple(sorted({canonical[key] for key in triplet_edges}))
    needed_atoms = set(owned_ids)
    for key in owned_bonds:
        needed_atoms.update(key[:2])
    for key in triplet_edges:
        needed_atoms.update(key[:2])
    return LocalGraphShard(
        rank=rank, atom_ids=ids, positions=coordinates.positions.detach().clone(),
        atomic_numbers=coordinates.numbers.to(coordinates.positions.device).clone(),
        cell=partition.cell.to(coordinates.positions).clone(), atom_owners=owners,
        owned_atom_ids=owned_ids, center_directed_keys=directed,
        owned_bond_keys=owned_bonds, owned_triplet_keys=triplets,
        directed_keys=tuple(key for bond in owned_bonds for key in (bond, reverse_directed(bond))),
        required_atom_ids=tuple(sorted(needed_atoms)), required_bond_keys=required_bonds,
        atom_owner_ranks=atom_owner_ranks, atom_owner_rows=atom_owner_rows,
        atom_cutoff=atom_cutoff, line_cutoff=line_cutoff)


def _cpu_candidates(owned: AtomState, coordinates: AtomState, partition: DomainPartition,
                    rank: int, atom_cutoff: float):
    """(ID-sorted coordinates, their ID tuple, owned IDs, candidate rows) of the CPU search.

    Candidate rows are (center, neighbor, sx, sy, sz) in sorted coordinate rows,
    every owned-center pair within ``atom_cutoff`` plus a conservative margin.
    """
    owned.validate()
    coordinates.validate()
    if not (partition.owner_of(owned.positions) == rank).all():
        raise ValueError("owned atoms do not belong to this spatial rank")
    coordinates = coordinates.sorted()
    ids = tuple(coordinates.ids.detach().cpu().tolist())
    index = {key: row for row, key in enumerate(ids)}
    owned_ids = tuple(sorted(owned.ids.detach().cpu().tolist()))
    if not set(owned_ids).issubset(index):
        raise ValueError("coordinate halo must contain every owned atom")
    positions = coordinates.positions.detach().cpu()
    cell = partition.cell.to(positions).cpu()
    # CPU discovery only supplies conservative candidates. The final predicate
    # runs on the coordinate device: a CPU/CUDA norm difference of one ULP at
    # either cutoff changes the discrete graph, not just numerical E/F error.
    # Model geometry is recomputed differentiably from shard.positions.
    lengths = partition.face_spacings.numpy()
    cartesian = positions.double().numpy()
    lattice = cell.double().numpy()
    margin = max(1e-6, 32 * torch.finfo(positions.dtype).eps
                 * max(float(partition.lengths.max()), atom_cutoff))
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(cartesian)
    except ImportError:
        tree = None
    if partition.orthogonal:
        reach = [math.floor(atom_cutoff / spacing) + 1 for spacing in lengths]
    else:
        # FP32 wrapping of a skew cell can leave rows just outside [0, 1).
        # Bound images by the rows' actual spread and the padded search radius.
        spread = (np.maximum(np.ptp(cartesian @ np.linalg.inv(lattice), axis=0), 1.0) if len(cartesian)
                  else np.ones(3))
        reach = [math.floor((atom_cutoff + margin) / spacing + float(width))
                 for spacing, width in zip(lengths, spread)]
    shifts = itertools.product(*(range(-k, k + 1) for k in reach))
    candidate_rows: list[tuple[int, int, int, int, int]] = []
    centers = [index[key] for key in owned_ids]
    for image in shifts:
        targets = cartesian[centers] - np.asarray(image) @ lattice
        if tree is None:
            neighbors = [np.flatnonzero(np.linalg.norm(cartesian - point, axis=1) <= atom_cutoff + margin)
                         for point in targets]
        else:
            neighbors = tree.query_ball_point(targets, atom_cutoff + margin)
        for center, neighbor_rows in zip(centers, neighbors):
            for neighbor in neighbor_rows:
                if center == neighbor and image == (0, 0, 0):
                    continue
                candidate_rows.append((center, int(neighbor), *image))
    return coordinates, ids, owned_ids, candidate_rows


def build_local_graph(owned: AtomState, coordinates: AtomState, partition: DomainPartition,
                      rank: int, atom_cutoff: float, line_cutoff: float) -> LocalGraphShard:
    """Build just owned-center stars and triplets using owned+halo coordinates.

    The independent CPU key oracle (Python keys; the CLI ``--oracle`` and the
    tests' single-device graphs). Production shards come from
    ``build_distributed_graph``.
    """
    for name, value in (("atom_cutoff", atom_cutoff), ("line_cutoff", line_cutoff)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    coordinates, ids, owned_ids, candidate_rows = _cpu_candidates(
        owned, coordinates, partition, rank, atom_cutoff)
    positions = coordinates.positions.detach().cpu()
    cell = partition.cell.to(positions).cpu()
    directed: list[DirectedKey] = []
    short_by_center: dict[int, list[DirectedKey]] = {key: [] for key in owned_ids}
    if candidate_rows:
        candidates = torch.tensor(candidate_rows, dtype=torch.long)
        if coordinates.positions.is_cuda:
            # Match neighbor_list_nvidia's dtype, operand ordering, batched
            # image transform and norm on CUDA. CPU distances are not an oracle
            # for near-cutoff FP32 GPU edges, even with the same '<' predicate.
            device_positions = coordinates.positions.detach()
            device_candidates = candidates.to(device_positions.device)
            device_cell = partition.cell.to(device_positions).expand(len(candidates), 3, 3)
            images = device_candidates[:, 2:].float()
            displacement = device_positions[device_candidates[:, 1]] - device_positions[device_candidates[:, 0]]
            displacement = displacement + small_matmul(images.unsqueeze(-2), device_cell).squeeze(-2)
        else:
            # Keep CPU FP64 frozen-topology/finite-difference fixtures unchanged.
            displacement = positions[candidates[:, 1]] - positions[candidates[:, 0]]
            displacement = displacement + candidates[:, 2:].to(positions.dtype) @ cell
        distances = displacement.norm(dim=1)
        cutoff_masks = torch.stack((distances < atom_cutoff, distances < line_cutoff), dim=1).cpu()
        selected = torch.where(cutoff_masks[:, 0])[0]
        # Avoid O(E) Python calls into scalar BoolTensor indexing/conversion.
        short_masks = cutoff_masks[selected, 1].tolist()
        for row, is_short in zip(selected.tolist(), short_masks):
            center, neighbor, sx, sy, sz = candidate_rows[row]
            key = (ids[center], ids[neighbor], sx, sy, sz)
            directed.append(key)
            if is_short:
                short_by_center[key[0]].append(key)
    triplets = [(center, left, right) for center, star in short_by_center.items()
                for ordered in (sorted(star),)
                for left in ordered for right in ordered if left != right]
    return _make_shard(coordinates, owned_ids, partition, rank, directed, triplets,
                       atom_cutoff, line_cutoff)


def build_cpu_bond_partial_graph(owned: AtomState, coordinates: AtomState, partition: DomainPartition,
                                 rank: int, atom_cutoff: float, line_cutoff: float) -> LocalGraphShard:
    """The device-topology shard of the bond_partial exchange from the CPU search.

    The CPU candidates (``_cpu_candidates``) take the canonical-key cutoff
    decision of ``neighbor_candidates`` and go through the same pure-tensor
    assembly as the CUDA path (``assemble_device_shard``).
    """
    from .device_graph import assemble_device_shard
    from .device_neighbors import _canonical_lengths

    for name, value in (("atom_cutoff", atom_cutoff), ("line_cutoff", line_cutoff)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    coordinates, _, _, candidate_rows = _cpu_candidates(owned, coordinates, partition, rank, atom_cutoff)
    positions = coordinates.positions.detach()
    cell = partition.cell.to(positions)
    candidates = torch.tensor(candidate_rows, dtype=torch.long).reshape(-1, 5)
    center, source, images = candidates[:, 0], candidates[:, 1], candidates[:, 2:]
    displacement = positions[source] - positions[center] + images.to(positions.dtype) @ cell
    distances = displacement.norm(dim=1)
    keep = _canonical_lengths(positions, cell, center, source, images) < atom_cutoff
    return assemble_device_shard(owned, coordinates, partition, rank, center[keep], source[keep],
                                 images[keep], distances[keep], atom_cutoff, line_cutoff)


def build_distributed_graph(state: AtomState, partition: DomainPartition, context: Any,
                            atom_cutoff: float, line_cutoff: float, *,
                            triplets: bool = True) -> LocalGraphShard:
    """After migration, build an exact shard of the bond_partial exchange without gathering global positions.

    Each bond is decided from its canonical key at both ends
    (``neighbor_candidates``; on CPU the same decision on the CPU search,
    ``build_cpu_bond_partial_graph``), and the device-topology shard is built
    on either device. ``triplets=False`` (the topology epoch's superset) only
    counts the triplets.
    """
    # Wall-clock subphases, with only the synchronization already required by
    # construction/communication. The evaluator synchronizes the total graph
    # phase; these subdivisions do not add synchronization to the hot path.
    started = time.perf_counter()
    coordinates = exchange_halo(state, partition, context, atom_cutoff)
    halo_done = time.perf_counter()
    error, shard = None, None
    try:
        if state.positions.is_cuda:
            from .device_graph import build_device_local_graph
            shard = build_device_local_graph(state, coordinates, partition, context,
                                              atom_cutoff, line_cutoff, triplets=triplets)
        else:
            shard = build_cpu_bond_partial_graph(state, coordinates, partition, context.rank,
                                                 atom_cutoff, line_cutoff)
    except (ValueError, TypeError, RuntimeError) as exc:
        error = str(exc)
    _collective_check(context, error)
    assert shard is not None
    local_done = time.perf_counter()
    # The route phases (bond provenance, triplet routing) left with the rows exchange: 0.
    return replace(shard, graph_timings={"halo_seconds": halo_done - started,
                                         "build_local_seconds": local_done - halo_done,
                                         "route_owner_seconds": 0.0, "route_triplet_seconds": 0.0})


@dataclass(frozen=True)
class FrozenTopology:
    """Small-oracle discrete topology; never contains frozen distances/angles."""
    atom_ids: tuple[int, ...]
    directed_keys: tuple[DirectedKey, ...]
    triplet_keys: tuple[TripletKey, ...]
    atom_cutoff: float
    line_cutoff: float

    def validate(self) -> None:
        atoms, edges = set(self.atom_ids), set(self.directed_keys)
        if len(atoms) != len(self.atom_ids) or len(edges) != len(self.directed_keys):
            raise ValueError("frozen topology has duplicate atoms or directed incidences")
        for edge in edges:
            if not set(edge[:2]).issubset(atoms) or reverse_directed(edge) not in edges:
                raise ValueError("frozen topology requires known atoms and complete reverse incidences")
        if len(set(self.triplet_keys)) != len(self.triplet_keys):
            raise ValueError("duplicate ordered triplets")
        for center, left, right in self.triplet_keys:
            if left not in edges or right not in edges or left == right or left[0] != center or right[0] != center:
                raise ValueError("invalid frozen ordered triplet")


def freeze_topology(shards: Iterable[LocalGraphShard]) -> FrozenTopology:
    """Test-only union; production must not gather full E/T to a GPU."""
    shards = tuple(shards)
    if not shards:
        raise ValueError("at least one shard is required")
    result = FrozenTopology(tuple(sorted(key for shard in shards for key in shard.owned_atom_ids)),
                            tuple(sorted(key for shard in shards for key in shard.directed_keys)),
                            tuple(sorted(key for shard in shards for key in shard.owned_triplet_keys)),
                            shards[0].atom_cutoff, shards[0].line_cutoff)
    result.validate()
    return result


def gather_topology(shard: LocalGraphShard, context: Any, root: int = 0) -> FrozenTopology | None:
    """Diagnostic-only CPU gather for a small single-GPU graph oracle.

    All ranks must call this explicitly; never used by the normal E/F loop.
    Tensor coordinates, features and activations are not gathered.
    """
    valid_root = isinstance(root, int) and not isinstance(root, bool) and 0 <= root < context.world_size
    _collective_check(context, None if valid_root else "invalid topology gather root")
    roots = context.all_gather_object(root)
    _collective_check(context, None if all(value == root for value in roots)
                      else "topology gather root differs across ranks")
    item = (tuple(shard.owned_atom_ids), tuple(shard.directed_keys), tuple(shard.owned_triplet_keys),
            shard.atom_cutoff, shard.line_cutoff)
    if context.world_size == 1:
        gathered = [item]
    else:
        gathered = [None] * context.world_size if context.rank == root else None
        group = context.control_group
        destination = dist.get_global_rank(group, root)
        dist.gather_object(item, object_gather_list=gathered, dst=destination, group=group)
    result, error = None, None
    if context.rank == root:
        try:
            if any(values[3:] != item[3:] for values in gathered):
                raise ValueError("topology cutoff settings differ across ranks")
            result = FrozenTopology(
                tuple(sorted(key for values in gathered for key in values[0])),
                tuple(sorted(key for values in gathered for key in values[1])),
                tuple(sorted(key for values in gathered for key in values[2])),
                shard.atom_cutoff, shard.line_cutoff)
            result.validate()
        except (ValueError, TypeError) as exc:
            error = str(exc)
    _collective_check(context, error)
    return result


def frozen_topology_from_radius_graph(graph: Any, atom_ids: Iterable[int] | None = None) -> FrozenTopology:
    """Normalize a single existing GPU/CPU RadiusGraph without rebuilding it."""
    ids = tuple(range(len(graph.atomic_number))) if atom_ids is None else tuple(atom_ids)
    if len(ids) != len(graph.atomic_number):
        raise ValueError("one global ID per graph atom is required")
    images = graph.neighbor_image.detach().cpu()
    if not torch.equal(images, images.round()):
        raise ValueError("periodic images must be integer-valued")
    edges = tuple((ids[i], ids[j], *image) for (i, j), image in
                  zip(graph.atom_graph.detach().cpu().tolist(), images.long().tolist()))
    triplets = tuple((ids[row[0]], edges[row[2]], edges[row[4]])
                     for row in graph.line_graph.detach().cpu().reshape(-1, 5).tolist())
    result = FrozenTopology(tuple(sorted(ids)), tuple(sorted(edges)), tuple(sorted(triplets)),
                            float(graph.atom_graph_cutoff), float(graph.line_graph_cutoff))
    result.validate()
    return result


def radius_graph_from_frozen(topology: FrozenTopology, atoms: AtomState, cell: Tensor) -> Any:
    """Single-graph oracle adapter; recompute continuous geometry in the model."""
    from fastmd.graph.radiusgraph import RadiusGraph
    topology.validate()
    atoms = atoms.sorted()
    ids = atoms.ids.detach().cpu().tolist()
    if tuple(ids) != tuple(sorted(topology.atom_ids)):
        raise ValueError("atom IDs do not match frozen topology")
    rows = {key: row for row, key in enumerate(ids)}
    edges = tuple(sorted(topology.directed_keys))
    edge_rows = {key: row for row, key in enumerate(edges)}
    bonds = tuple(sorted({canonical_bond(key) for key in edges}))
    bond_rows = {key: row for row, key in enumerate(bonds)}
    device = atoms.positions.device
    def integer(values: Any, width: int | None = None) -> Tensor:
        result = torch.tensor(values, device=device, dtype=torch.int64)
        return result if width is None else result.reshape(-1, width)
    cell = cell.to(atoms.positions)
    return RadiusGraph(graph_id=None, mp_id=None, composition=None,
        atomic_number=atoms.numbers.to(device),
        atom_frac_coord=small_matmul(atoms.positions, torch.linalg.inv(cell)),
        lattice=cell, neighbor_image=integer([key[2:] for key in edges], 3).to(atoms.positions.dtype),
        atom_graph=integer([(rows[key[0]], rows[key[1]]) for key in edges], 2),
        atom_graph_cutoff=topology.atom_cutoff, line_graph_cutoff=topology.line_cutoff,
        line_graph=integer([(rows[q[0]], bond_rows[canonical_bond(q[1])], edge_rows[q[1]],
                             bond_rows[canonical_bond(q[2])], edge_rows[q[2]])
                            for q in sorted(topology.triplet_keys)], 5),
        directed2undirected=integer([bond_rows[canonical_bond(key)] for key in edges]),
        undirected2directed=integer([edge_rows[key] for key in bonds]),
        atom_target_sorted=True, line_atom_sorted=True)
