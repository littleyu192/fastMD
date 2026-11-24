"""Full-key device topology of the bond_partial exchange.

All N/E/T topology arrays remain on the coordinate device. Python key views
exist solely for explicit diagnostics. The CPU topology builder remains an
independent oracle; no global coordinate, bond-key or triplet gather is used.
The rows exchange's bond producer-row resolution was removed in step 4.2.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import torch

from .device_keys import canonicalize_edges, lexsort_rows, unique_rows
from .partition import _collective_check


def _triplet_indices(center_owned_target, distances, owned_count, line_cutoff):
    device = distances.device
    short = torch.where(distances < line_cutoff)[0]
    counts = torch.bincount(center_owned_target[short], minlength=owned_count)
    pairs = counts * (counts - 1)
    total = int(pairs.sum().item())
    participating = short[counts[center_owned_target[short]] >= 2]
    if not total:
        return torch.empty((0, 2), dtype=torch.int64, device=device), participating
    offsets, pair_offsets = counts.cumsum(0) - counts, pairs.cumsum(0) - pairs
    centers = torch.repeat_interleave(torch.arange(owned_count, device=device), pairs,
                                      output_size=total)
    within = torch.arange(total, device=device) - pair_offsets[centers]
    width = counts[centers] - 1
    left = torch.div(within, width, rounding_mode="floor")
    right = within.remainder(width)
    right = right + (right >= left).long()
    base = offsets[centers]
    return torch.stack((short[base + left], short[base + right]), dim=1), participating


def assemble_device_shard(owned, coordinates, partition, rank, target, source, images,
                          distances, atom_cutoff, line_cutoff, *, neighbor_stats=None, triplets=True):
    """Pure-tensor assembly, also callable with CPU oracle edges in unit tests.

    Inputs are exact accepted owned-center candidates. Halo atom rows must be
    sorted by stable ID, and sender owner/row provenance must be present.
    Candidate generation is deliberately separate to permit independent tests.
    """
    from .graph import AtomOwnerView, LocalGraphShard, TensorKeyView, TripletKeyView

    device = coordinates.positions.device
    ids = coordinates.ids.to(device=device, dtype=torch.int64).clone()
    own_ids = owned.ids.to(device=device, dtype=torch.int64).sort().values
    own_indices = torch.searchsorted(ids, own_ids)
    atom_owners = coordinates.payload["_owner"].to(device=device, dtype=torch.int64).clone()
    atom_rows = coordinates.payload["_owner_row"].to(device=device, dtype=torch.int64).clone()
    edges = torch.cat((ids[target, None], ids[source, None], images.long()), dim=1)
    order = lexsort_rows(edges)
    edges, target, source, images, distances = (value[order] for value in
                                                (edges, target, source, images, distances))
    center_owned = torch.searchsorted(own_ids, ids[target])
    canonical, _ = canonicalize_edges(edges)
    canonical, center_canonical = unique_rows(canonical)
    canonical_atom_rows = torch.searchsorted(ids, canonical[:, 0].contiguous())
    canonical_owners = atom_owners[canonical_atom_rows]
    owned_canonical = torch.where(canonical_owners == rank)[0]
    bonds = canonical[owned_canonical]
    triplet_count_only = None
    if triplets:
        pairs, participating = _triplet_indices(center_owned, distances, len(own_ids), line_cutoff)
    else:
        # Topology epoch superset (fastmd.distributed.epoch): only whether the
        # rank has a triplet within line_cutoff; the triplets themselves are
        # enumerated exactly every step on the device.
        short = torch.where(distances < line_cutoff)[0]
        counts = torch.bincount(center_owned[short], minlength=len(own_ids))
        triplet_count_only = int((counts * (counts - 1)).sum().item())
        pairs = torch.empty((0, 2), dtype=torch.int64, device=device)
        participating = short[:0]
    used_canonical = torch.unique(center_canonical[participating], sorted=True)
    required_bonds = canonical[used_canonical]
    required_atoms = torch.unique(torch.cat((own_ids, bonds[:, :2].reshape(-1),
                                            edges[participating, :2].reshape(-1))), sorted=True)
    reversed_bonds = torch.cat((bonds[:, 1:2], bonds[:, 0:1], -bonds[:, 2:]), dim=1)
    directed = torch.stack((bonds, reversed_bonds), dim=1).reshape(-1, 5)
    topology = {
        "atom_ids": ids, "owned_atom_ids": own_ids, "owned_atom_indices": own_indices,
        "center_target": target.long(), "center_source": source.long(),
        "center_images": images.long(), "center_owned_target": center_owned,
        "center_canonical": center_canonical, "canonical_edges": canonical,
        # This sorted, unique row set was already computed while expanding the
        # center CSR. Keep it in this evaluation's immutable topology so the
        # preparation builds its compact map without sorting 2*T indices again.
        "triplet_incidence_rows": participating,
    }
    center_view = TensorKeyView(edges)
    return LocalGraphShard(
        rank=rank, atom_ids=TensorKeyView(ids), positions=coordinates.positions.detach().clone(),
        atomic_numbers=coordinates.numbers.to(device).clone(), cell=partition.cell.to(coordinates.positions).clone(),
        atom_owners=AtomOwnerView(ids, atom_owners), owned_atom_ids=TensorKeyView(own_ids), center_directed_keys=center_view,
        owned_bond_keys=TensorKeyView(bonds), owned_triplet_keys=TripletKeyView(edge_keys=center_view, edge_index=pairs),
        directed_keys=TensorKeyView(directed), required_atom_ids=TensorKeyView(required_atoms),
        required_bond_keys=TensorKeyView(required_bonds), triplet_edge_index=pairs,
        atom_owner_ranks=atom_owners, atom_owner_rows=atom_rows,
        device_topology=MappingProxyType(topology),
        device_graph_stats={**(neighbor_stats or {}),
                            **({} if triplet_count_only is None else {"triplet_count_only": triplet_count_only})},
        atom_cutoff=atom_cutoff, line_cutoff=line_cutoff)


def build_device_local_graph(owned, coordinates, partition, context, atom_cutoff, line_cutoff, *,
                             triplets=True):
    """The CUDA device-topology shard (canonical-key cutoff, ``neighbor_candidates``)."""
    from .device_neighbors import neighbor_candidates

    # Neighbor discovery contains collectives (overflow retry). Admit every
    # rank before entering it, rather than letting one malformed rank skip to
    # the outer graph error collective while peers are still inside discovery.
    error = None
    try:
        for name, value in (("atom_cutoff", atom_cutoff), ("line_cutoff", line_cutoff)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if coordinates.positions.dtype != torch.float32:
            raise ValueError("production device topology requires FP32 coordinates")
        ids = coordinates.ids
        owners, rows = coordinates.payload["_owner"], coordinates.payload["_owner_row"]
        if ids.device != coordinates.positions.device or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("device topology requires integer coordinate IDs on the coordinate device")
        for name, value in (("owner", owners), ("owner_row", rows)):
            if value.shape != ids.shape or value.dtype != torch.int64 or value.device != ids.device:
                raise ValueError(f"coordinate {name} provenance must be aligned device int64")
        own_ids = owned.ids.to(ids.device).sort().values
        own_indices = torch.searchsorted(ids, own_ids)
        present = (ids[own_indices.clamp(max=len(ids) - 1)] == own_ids).all() if len(ids) else own_ids.new_tensor(not len(own_ids), dtype=torch.bool)
        own_provenance = ((owners[own_indices.clamp(max=len(ids) - 1)] == context.rank).all()
                          if len(ids) else present)
        flags = torch.stack(((ids[1:] > ids[:-1]).all(), present,
                             ((owners >= 0) & (owners < context.world_size)).all(),
                             (rows >= 0).all(), own_provenance)).cpu().tolist()
        if not all(flags):
            raise ValueError("coordinate IDs/provenance must be sorted, valid, and contain every owned atom")
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
        error = str(exc)
    _collective_check(context, error)
    cell = partition.cell.to(coordinates.positions)
    # Broaden only candidate discovery. Final CUDA norm and strict predicate
    # are unchanged, including when Warp's squared-distance arithmetic differs.
    margin = max(1e-6, 32 * torch.finfo(coordinates.positions.dtype).eps
                 * max(float(partition.lengths.max()), atom_cutoff))
    target, source, images, distances, stats = neighbor_candidates(
        coordinates.positions, cell, coordinates.payload["_owner"], context.rank,
        atom_cutoff, context, candidate_margin=margin)
    return assemble_device_shard(owned, coordinates, partition, context.rank,
                                  target, source, images, distances, atom_cutoff, line_cutoff,
                                  neighbor_stats=stats, triplets=triplets)
