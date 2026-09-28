from __future__ import annotations

from fastmd._vendor.matris.config import env_value


import numpy as np
import torch
from torch import Tensor

from .._nvtx import nvtx_range
from .geometry import small_matmul
from .matris_topology import matris_builder_evidence
from .radiusgraph import RadiusGraph
from .validation import raise_if_isolated_atoms

try:
    from nvalchemiops.neighborlist.neighbor_utils import estimate_max_neighbors
    from nvalchemiops.neighborlist.neighborlist import neighbor_list as _nvidia_nl

    op_available = True
except ImportError:
    op_available = False

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    triton = None
    tl = None
    _triton_available = False

# Static upper bound on periodic-image shift magnitude, used to size sort keys
# without a .item() D2H sync. |shift| <= MAX_IMAGE holds whenever the smallest
# cell dimension >= cutoff / MAX_IMAGE, i.e. always for physical cells.
MAX_IMAGE = 10

_GPU_BUILDER_EVIDENCE = matris_builder_evidence(
    "matris.gpu_radius_graph",
    atom_target_sorted=True,
    line_owner_sorted=True,
)


def _env_flag(name: str, default: bool) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "off", "no"}


_LINE_GRAPH_TRITON = _env_flag("MATRIS_LINE_GRAPH_TRITON", True)
_LINE_GRAPH_TRITON_BLOCK = int(env_value("MATRIS_LINE_GRAPH_TRITON_BLOCK", "256"))
# Elide the redundant _build_d2u key sort and _build_line_graph stable sort
# when the edge stream is already ordered by construction (single-system path;
# neighbor_list_nvidia argsorts by (batch, center, neighbor, image), and
# batch is a monotone function of the graph-contiguous center index, so both
# downstream key orders are postconditions of that argsort).
_PRESORTED_BUILD = _env_flag("MATRIS_PRESORTED_BUILD", True)


def _cuda_device(device: torch.device | str) -> torch.device:
    device = torch.device(device)
    if device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(device)
    return device


@torch.no_grad()
def neighbor_list_nvidia(
    positions: Tensor,
    cell: Tensor,
    cutoff: float,
    batch_idx: Tensor,
    pbc: Tensor,
    device: torch.device | str = "cuda",
    return_num_neighbors: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    with nvtx_range("neighbor_list_nvidia"):
        device = _cuda_device(device)
        positions = positions.to(device)
        cell = cell.to(device)
        batch_idx = batch_idx.to(device=device, dtype=torch.int32)
        pbc = pbc.to(device=device, dtype=torch.bool)

        n_atoms = positions.shape[0]
        buffer_max = estimate_max_neighbors(cutoff=cutoff, safety_factor=2.0)
        neighbor_matrix = torch.full(
            (n_atoms, buffer_max),
            n_atoms,
            dtype=torch.int32,
            device=device,
        )
        neighbor_shifts = torch.zeros(
            (n_atoms, buffer_max, 3),
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(n_atoms, dtype=torch.int32, device=device)

        _nvidia_nl(
            positions=positions,
            cutoff=cutoff + 1e-6,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            method="batch_cell_list",
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            num_neighbors=num_neighbors,
            half_fill=False,
        )

        atom_idx = torch.arange(n_atoms, device=device).unsqueeze(1)
        neigh_idx = torch.arange(buffer_max, device=device).unsqueeze(0)
        valid = neigh_idx < num_neighbors.unsqueeze(1)
        center = atom_idx.expand(-1, buffer_max)[valid].long()
        neighbor = neighbor_matrix[valid].long()
        images = neighbor_shifts[valid]

        edge_batch = batch_idx[center].long()
        diff = positions[neighbor] - positions[center]
        diff = diff + small_matmul(
            images.float().unsqueeze(-2), cell[edge_batch]
        ).squeeze(-2)
        dist = diff.norm(dim=1)
        mask = dist < cutoff
        center = center[mask]
        neighbor = neighbor[mask]
        images = images[mask]
        dist = dist[mask]

        img = images.long()
        # Static image bound avoids a .item() D2H sync. |shift| <= MAX_IMAGE holds for
        # any cell whose smallest dimension >= cutoff/MAX_IMAGE (always true physically).
        max_img = MAX_IMAGE
        radix = 2 * max_img + 1
        order = (
            batch_idx[center].long() * n_atoms * n_atoms * radix**3
            + center * n_atoms * radix**3
            + neighbor * radix**3
            + (img[:, 0] + max_img) * radix * radix
            + (img[:, 1] + max_img) * radix
            + (img[:, 2] + max_img)
        ).argsort()
        result = center[order], neighbor[order], images[order], dist[order]
        if return_num_neighbors:
            return (*result, num_neighbors)
        return result


def _build_d2u(
    center: Tensor,
    neighbor: Tensor,
    images: Tensor,
    n_atoms: int,
    presorted: bool = False,
) -> tuple[Tensor, Tensor]:
    with nvtx_range("_build_d2u"):
        device = center.device
        n_edges = center.shape[0]
        if n_edges == 0:
            # A valid all-isolated graph has no reverse-edge pairs. Shape-based
            # dispatch is host-known and does not synchronize the device.
            empty = torch.empty(0, dtype=torch.int32, device=device)
            return empty, empty.clone()
        img = images.long()
        max_img = MAX_IMAGE  # static bound -> no .item() D2H sync
        radix = 2 * max_img + 1
        volume = radix * radix * radix
        forward_key = (
            (center * n_atoms + neighbor) * volume
            + (img[:, 0] + max_img) * radix * radix
            + (img[:, 1] + max_img) * radix
            + (img[:, 2] + max_img)
        )
        reverse_key = (
            (neighbor * n_atoms + center) * volume
            + (-img[:, 0] + max_img) * radix * radix
            + (-img[:, 1] + max_img) * radix
            + (-img[:, 2] + max_img)
        )
        if presorted:
            # forward_key order is a postcondition of the neighbor-list
            # argsort (same lexicographic components; the omitted batch term
            # is monotone in center), so the sort is the identity.
            pair_idx = torch.searchsorted(forward_key, reverse_key).clamp(
                max=n_edges - 1
            )
            sorted_key = forward_key
        else:
            sorted_key, sort_idx = forward_key.sort()
            pair_idx = sort_idx[
                torch.searchsorted(sorted_key, reverse_key).clamp(
                    max=n_edges - 1
                )
            ]
        edge_ids = torch.arange(n_edges, device=device)
        uvals, directed2undirected = torch.unique(
            torch.minimum(edge_ids, pair_idx),
            return_inverse=True,
        )
        undirected2directed = torch.full(
            (uvals.shape[0],),  # = num unique undirected edges; shape read, no .item() D2H sync
            n_edges,
            dtype=torch.int64,
            device=device,
        )
        undirected2directed.scatter_reduce_(
            0,
            directed2undirected.long(),
            edge_ids,
            reduce="amin",
            include_self=False,
        )
        return directed2undirected.int(), undirected2directed.int()


if _triton_available:

    @triton.jit
    def _count_isolated_sorted_kernel(
        center,
        batch_idx,
        isolated_atom_counts,
        n_edges,
        SEARCH_STEPS: tl.constexpr,
    ):
        atom = tl.program_id(0)
        lo = atom * 0
        hi = lo + n_edges
        for _ in tl.static_range(0, SEARCH_STEPS):
            mid = (lo + hi) // 2
            in_bounds = mid < n_edges
            value = tl.load(center + mid, mask=in_bounds, other=-1)
            move_right = in_bounds & (value < atom)
            lo = tl.where(move_right, mid + 1, lo)
            hi = tl.where(move_right, hi, mid)

        value = tl.load(center + lo, mask=lo < n_edges, other=-1)
        graph_idx = tl.load(batch_idx + atom)
        tl.atomic_add(
            isolated_atom_counts + graph_idx,
            1,
            mask=value != atom,
        )


    @triton.jit
    def _line_graph_status_kernel(
        pair_counts,
        atom_neighbor_counts,
        status,
        N_ATOMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        atoms = tl.arange(0, BLOCK)
        mask = atoms < N_ATOMS
        pairs = tl.load(pair_counts + atoms, mask=mask, other=0)
        neighbors = tl.load(atom_neighbor_counts + atoms, mask=mask, other=0)
        tl.store(status, tl.sum(pairs, axis=0))
        tl.store(
            status + 1,
            tl.sum((mask & (neighbors == 0)).to(tl.int64), axis=0),
        )
        tl.store(status + 2, tl.sum(neighbors.to(tl.int64), axis=0))

    @triton.jit
    def _line_graph_materialize_kernel(
        counts,
        offsets,
        pair_offsets,
        short_de,
        short_ude,
        out,
        TOTAL_PAIRS,
        N_ATOMS: tl.constexpr,
        LOG_N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < TOTAL_PAIRS

        lo = tl.full((BLOCK,), 0, dtype=tl.int64)
        hi = tl.full((BLOCK,), N_ATOMS - 1, dtype=tl.int64)
        for _ in tl.static_range(0, LOG_N):
            mid = (lo + hi + 1) // 2
            mid_offsets = tl.load(pair_offsets + mid)
            take_upper = mid_offsets <= rows
            lo = tl.where(take_upper, mid, lo)
            hi = tl.where(take_upper, hi, mid - 1)

        atom = lo
        count = tl.load(counts + atom)
        atom_pair_offset = tl.load(pair_offsets + atom)
        local_idx = rows - atom_pair_offset
        denom = tl.maximum(count - 1, 1)
        first_in_group = local_idx // denom
        second_raw = local_idx - first_in_group * denom
        second_in_group = second_raw + (second_raw >= first_in_group)
        atom_offset = tl.load(offsets + atom)
        first_pos = atom_offset + first_in_group
        second_pos = atom_offset + second_in_group

        first_ude = tl.load(short_ude + first_pos, mask=mask, other=0)
        first_de = tl.load(short_de + first_pos, mask=mask, other=0)
        second_ude = tl.load(short_ude + second_pos, mask=mask, other=0)
        second_de = tl.load(short_de + second_pos, mask=mask, other=0)
        base = rows * 5
        tl.store(out + base + 0, atom, mask=mask)
        tl.store(out + base + 1, first_ude, mask=mask)
        tl.store(out + base + 2, first_de, mask=mask)
        tl.store(out + base + 3, second_ude, mask=mask)
        tl.store(out + base + 4, second_de, mask=mask)

else:
    _count_isolated_sorted_kernel = None
    _line_graph_status_kernel = None
    _line_graph_materialize_kernel = None


def _count_isolated_atoms(
    center: Tensor,
    batch_idx: Tensor,
    *,
    n_atoms: int,
    n_graphs: int,
) -> Tensor:
    isolated_atom_counts = torch.zeros(
        n_graphs,
        dtype=torch.int32,
        device=center.device,
    )
    if n_atoms == 0:
        return isolated_atom_counts
    if _triton_available and center.is_cuda:
        # ``neighbor_list_nvidia`` sorts by batch and global center atom. A
        # fixed-width lower bound avoids bincount + compare + reduction kernels.
        _count_isolated_sorted_kernel[(n_atoms,)](
            center,
            batch_idx,
            isolated_atom_counts,
            center.shape[0],
            SEARCH_STEPS=32,
            num_warps=1,
        )
        return isolated_atom_counts

    neighbor_counts = torch.bincount(center, minlength=n_atoms)
    for graph_idx in range(n_graphs):
        atom_ids = torch.where(batch_idx == graph_idx)[0]
        isolated_atom_counts[graph_idx] = (neighbor_counts[atom_ids] == 0).sum()
    return isolated_atom_counts


def _line_graph_status(
    pair_counts: Tensor,
    atom_neighbor_counts: Tensor,
) -> Tensor:
    n_atoms = int(pair_counts.shape[0])
    if (
        _triton_available
        and pair_counts.is_cuda
        and n_atoms <= 65_536
    ):
        status = torch.empty(3, dtype=torch.int64, device=pair_counts.device)
        block = triton.next_power_of_2(max(n_atoms, 1))
        _line_graph_status_kernel[(1,)](
            pair_counts,
            atom_neighbor_counts,
            status,
            N_ATOMS=n_atoms,
            BLOCK=block,
            num_warps=8,
        )
        return status
    return torch.stack(
        [
            pair_counts.sum(),
            (atom_neighbor_counts == 0).sum(),
            atom_neighbor_counts.sum(),
        ]
    )


def _exact_isolated_count(center: Tensor, n_atoms: int) -> tuple[Tensor, int]:
    count = (
        torch.bincount(center, minlength=n_atoms) == 0
    ).sum().reshape(1)
    return count, int(count.item())


def _materialize_line_graph_triton(
    counts: Tensor,
    offsets: Tensor,
    pair_offsets: Tensor,
    short_de: Tensor,
    short_ude: Tensor,
    total_pairs: int,
    n_atoms: int,
) -> Tensor | None:
    if not (_LINE_GRAPH_TRITON and _triton_available and short_de.is_cuda):
        return None
    with nvtx_range("_build_line_graph_materialize"):
        out = torch.empty((total_pairs, 5), dtype=torch.int32, device=short_de.device)
        block = _LINE_GRAPH_TRITON_BLOCK
        grid = (triton.cdiv(total_pairs, block),)
        _line_graph_materialize_kernel[grid](
            counts,
            offsets,
            pair_offsets,
            short_de,
            short_ude,
            out,
            TOTAL_PAIRS=total_pairs,
            N_ATOMS=n_atoms,
            LOG_N=max(1, (n_atoms - 1).bit_length()),
            BLOCK=block,
            num_warps=4,
        )
        return out


def _build_line_graph(
    center: Tensor,
    directed2undirected: Tensor,
    dist: Tensor,
    n_atoms: int,
    cutoff: float,
    atom_neighbor_counts: Tensor | None = None,
    presorted: bool = False,
) -> Tensor | tuple[Tensor, Tensor, int]:
    with nvtx_range("_build_line_graph"):
        device = center.device
        short_de = torch.where(dist < cutoff)[0]
        if short_de.numel() == 0:
            empty = torch.empty((0, 5), dtype=torch.int32, device=device)
            if atom_neighbor_counts is None:
                return empty
            if center.shape[0] == 0:
                isolated = torch.full((1,), n_atoms, dtype=torch.int64, device=device)
                return empty, isolated, n_atoms
            isolated, isolated_count = _exact_isolated_count(center, n_atoms)
            return empty, isolated, isolated_count

        if presorted:
            # center is non-decreasing (neighbor-list argsort postcondition)
            # and short_de is an ascending index subset, so the stable sort
            # is the identity permutation.
            sorted_center = center[short_de]
        else:
            sorted_center, order = center[short_de].sort(stable=True)
            short_de = short_de[order]
        short_ude = directed2undirected[short_de].long()
        counts = torch.bincount(sorted_center.int(), minlength=n_atoms).long()
        pair_counts = counts * (counts - 1)
        isolated_atom_counts = None
        isolated_count = 0
        if atom_neighbor_counts is None:
            total_pairs = int(pair_counts.sum().item())
        else:
            graph_status = _line_graph_status(pair_counts, atom_neighbor_counts)
            total_pairs, candidate_isolated, candidate_edges = (
                int(value) for value in graph_status.tolist()
            )
            if candidate_edges == center.shape[0]:
                isolated_atom_counts = graph_status[1:2]
                isolated_count = candidate_isolated
            else:
                isolated_atom_counts, isolated_count = _exact_isolated_count(
                    center,
                    n_atoms,
                )
        if total_pairs == 0:
            empty = torch.empty((0, 5), dtype=torch.int32, device=device)
            if isolated_atom_counts is None:
                return empty
            return empty, isolated_atom_counts, isolated_count

        offsets = torch.zeros(n_atoms, dtype=torch.long, device=device)
        offsets[1:] = counts[:-1].cumsum(0)
        pair_offsets = torch.zeros(n_atoms, dtype=torch.long, device=device)
        pair_offsets[1:] = pair_counts[:-1].cumsum(0)
        triton_line_graph = _materialize_line_graph_triton(
            counts,
            offsets,
            pair_offsets,
            short_de,
            short_ude,
            total_pairs,
            n_atoms,
        )
        if triton_line_graph is not None:
            if isolated_atom_counts is None:
                return triton_line_graph
            return triton_line_graph, isolated_atom_counts, isolated_count

        pair_atom = torch.repeat_interleave(torch.arange(n_atoms, device=device), pair_counts)
        local_idx = torch.arange(total_pairs, device=device) - pair_offsets[pair_atom]
        group_size = counts[pair_atom]
        first_in_group = local_idx // (group_size - 1)
        second_raw = local_idx % (group_size - 1)
        second_in_group = second_raw + (second_raw >= first_in_group).long()
        first_pos = offsets[pair_atom] + first_in_group
        second_pos = offsets[pair_atom] + second_in_group

        line_graph = torch.stack(
            [
                pair_atom.int(),
                short_ude[first_pos].int(),
                short_de[first_pos].int(),
                short_ude[second_pos].int(),
                short_de[second_pos].int(),
            ],
            dim=1,
        )
        if isolated_atom_counts is None:
            return line_graph
        return line_graph, isolated_atom_counts, isolated_count


def _extract_atoms(atoms_list, device: torch.device):
    atomic_numbers = []
    frac_coords = []
    positions = []
    cells = []
    pbc = []
    batch_idx = []
    atom_offsets = [0]
    compositions = []

    # Pack one or more ASE Atoms objects into a single batched CUDA payload.
    for graph_idx, atoms in enumerate(atoms_list):
        cell = torch.from_numpy(atoms.get_cell().array.astype(np.float32)).to(device)
        pos = torch.from_numpy(atoms.get_positions().astype(np.float32)).to(device)
        frac = small_matmul(pos, torch.linalg.inv(cell)) % 1.0
        n_atoms = len(atoms)
        atomic_numbers.append(
            torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=device)
        )
        frac_coords.append(frac)
        positions.append(small_matmul(frac, cell))
        cells.append(cell)
        pbc.append(torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device))
        batch_idx.append(torch.full((n_atoms,), graph_idx, dtype=torch.int32, device=device))
        atom_offsets.append(atom_offsets[-1] + n_atoms)
        compositions.append(atoms.get_chemical_formula())

    return {
        "atomic_numbers": torch.cat(atomic_numbers),
        "frac_coords": torch.cat(frac_coords),
        "positions": torch.cat(positions),
        "cells": torch.stack(cells),
        "pbc": torch.stack(pbc),
        "batch_idx": torch.cat(batch_idx),
        "atom_offsets": atom_offsets,
        "compositions": compositions,
    }


def _extract_tensor_system(
    positions: Tensor,
    cell: Tensor,
    atomic_numbers: Tensor,
    pbc: Tensor | None,
    device: torch.device,
    composition: str | None = None,
):
    positions = positions.to(device=device, dtype=torch.float32)
    cell = cell.to(device=device, dtype=torch.float32)
    atomic_numbers = atomic_numbers.to(device=device, dtype=torch.int32)
    if pbc is None:
        pbc = torch.ones(3, dtype=torch.bool, device=device)
    else:
        pbc = pbc.to(device=device, dtype=torch.bool)

    if cell.dim() != 2 or tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {tuple(cell.shape)}")
    if positions.dim() != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {tuple(positions.shape)}")
    if atomic_numbers.dim() != 1 or atomic_numbers.shape[0] != positions.shape[0]:
        raise ValueError(
            "atomic_numbers must be a 1D tensor with the same length as positions"
        )
    if pbc.dim() != 1 or pbc.shape[0] != 3:
        raise ValueError(f"pbc must have shape (3,), got {tuple(pbc.shape)}")

    frac = small_matmul(positions, torch.linalg.inv(cell)) % 1.0
    n_atoms = int(positions.shape[0])
    return {
        "atomic_numbers": atomic_numbers,
        "frac_coords": frac,
        "positions": small_matmul(frac, cell),
        "cells": cell.unsqueeze(0),
        "pbc": pbc.unsqueeze(0),
        "batch_idx": torch.zeros(n_atoms, dtype=torch.int32, device=device),
        "atom_offsets": [0, n_atoms],
        "compositions": [composition or ""],
    }


def _split_graphs(
    payload: dict[str, Tensor | list],
    center: Tensor,
    neighbor: Tensor,
    images: Tensor,
    directed2undirected: Tensor,
    undirected2directed: Tensor,
    line_graph: Tensor,
    isolated_atom_counts: Tensor,
    atom_graph_cutoff: float,
    line_graph_cutoff: float,
) -> list[RadiusGraph]:
    # Single-system fast path (the MD case): atom_start==0 and all edges/triplets
    # belong to graph 0, so global indices ARE local — skip the per-graph
    # where/unique/searchsorted/boolean-mask remapping (~7 D2H syncs/step).
    if len(payload["atom_offsets"]) - 1 == 1:
        return [
            RadiusGraph(
                atomic_number=payload["atomic_numbers"],
                atom_frac_coord=payload["frac_coords"],
                atom_graph=torch.stack([center, neighbor], dim=1).int(),
                neighbor_image=images.float(),
                directed2undirected=directed2undirected.int(),
                undirected2directed=undirected2directed.int(),
                line_graph=line_graph.int(),
                lattice=payload["cells"][0],
                graph_id=None,
                mp_id=None,
                composition=payload["compositions"][0],
                atom_graph_cutoff=atom_graph_cutoff,
                line_graph_cutoff=line_graph_cutoff,
                atom_target_sorted=True,
                line_atom_sorted=True,
                isolated_atom_count=isolated_atom_counts[0:1],
                topology_evidence=_GPU_BUILDER_EVIDENCE,
            )
        ]

    graphs = []
    for graph_idx in range(len(payload["atom_offsets"]) - 1):
        atom_start = payload["atom_offsets"][graph_idx]
        atom_end = payload["atom_offsets"][graph_idx + 1]
        edge_ids = torch.where(payload["batch_idx"][center] == graph_idx)[0].long()
        local_d2u_global = directed2undirected[edge_ids].long()
        unique_ude_global, local_d2u = torch.unique(
            local_d2u_global,
            sorted=True,
            return_inverse=True,
        )
        local_line_graph = line_graph[
            (line_graph[:, 0] >= atom_start) & (line_graph[:, 0] < atom_end)
        ].clone().long()
        if local_line_graph.numel() != 0:
            local_line_graph[:, 0] -= atom_start
            local_line_graph[:, 1] = torch.searchsorted(
                unique_ude_global,
                local_line_graph[:, 1].contiguous(),
            )
            local_line_graph[:, 2] = torch.searchsorted(
                edge_ids,
                local_line_graph[:, 2].contiguous(),
            )
            local_line_graph[:, 3] = torch.searchsorted(
                unique_ude_global,
                local_line_graph[:, 3].contiguous(),
            )
            local_line_graph[:, 4] = torch.searchsorted(
                edge_ids,
                local_line_graph[:, 4].contiguous(),
            )

        graphs.append(
            RadiusGraph(
                atomic_number=payload["atomic_numbers"][atom_start:atom_end],
                atom_frac_coord=payload["frac_coords"][atom_start:atom_end],
                atom_graph=torch.stack(
                    [center[edge_ids] - atom_start, neighbor[edge_ids] - atom_start],
                    dim=1,
                ).int(),
                neighbor_image=images[edge_ids].float(),
                directed2undirected=local_d2u.int(),
                undirected2directed=torch.searchsorted(
                    edge_ids,
                    undirected2directed[unique_ude_global].long(),
                ).int(),
                line_graph=local_line_graph.int(),
                lattice=payload["cells"][graph_idx],
                graph_id=None,
                mp_id=None,
                composition=payload["compositions"][graph_idx],
                atom_graph_cutoff=atom_graph_cutoff,
                line_graph_cutoff=line_graph_cutoff,
                atom_target_sorted=True,
                line_atom_sorted=True,
                isolated_atom_count=isolated_atom_counts[
                    graph_idx : graph_idx + 1
                ],
                topology_evidence=_GPU_BUILDER_EVIDENCE,
            )
        )
    return graphs


def _build_graphs_from_payload(
    payload: dict[str, Tensor | list],
    atom_graph_cutoff: float,
    line_graph_cutoff: float,
    device: torch.device,
    *,
    check_isolated_atoms: bool = True,
) -> list[RadiusGraph]:
    n_atoms = int(payload["atomic_numbers"].shape[0])
    n_graphs = len(payload["atom_offsets"]) - 1
    atom_neighbor_counts = None
    if n_graphs == 1:
        center, neighbor, images, dist, atom_neighbor_counts = neighbor_list_nvidia(
            payload["positions"],
            payload["cells"],
            atom_graph_cutoff,
            payload["batch_idx"],
            payload["pbc"],
            device=device,
            return_num_neighbors=True,
        )
        if check_isolated_atoms and center.numel() == 0:
            raise_if_isolated_atoms(n_atoms)
        isolated_atom_counts = None
    else:
        center, neighbor, images, dist = neighbor_list_nvidia(
            payload["positions"],
            payload["cells"],
            atom_graph_cutoff,
            payload["batch_idx"],
            payload["pbc"],
            device=device,
        )
        isolated_atom_counts = _count_isolated_atoms(
            center,
            payload["batch_idx"],
            n_atoms=n_atoms,
            n_graphs=n_graphs,
        )
        if check_isolated_atoms:
            raise_if_isolated_atoms(isolated_atom_counts)
    presorted = _PRESORTED_BUILD and n_graphs == 1
    directed2undirected, undirected2directed = _build_d2u(
        center,
        neighbor,
        images,
        n_atoms,
        presorted=presorted,
    )
    line_graph_result = _build_line_graph(
        center,
        directed2undirected,
        dist,
        n_atoms,
        line_graph_cutoff,
        atom_neighbor_counts=atom_neighbor_counts,
        presorted=presorted,
    )
    if n_graphs != 1:
        line_graph = line_graph_result
    else:
        line_graph, isolated_atom_counts, isolated_count = line_graph_result
        if check_isolated_atoms:
            raise_if_isolated_atoms(isolated_count)
    return _split_graphs(
        payload,
        center,
        neighbor,
        images,
        directed2undirected,
        undirected2directed,
        line_graph,
        isolated_atom_counts,
        atom_graph_cutoff,
        line_graph_cutoff,
    )


class TensorGraphBuilder:
    """Cached single-system tensor graph builder for fixed-cell GPU MD."""

    def __init__(
        self,
        *,
        cell: Tensor,
        atomic_numbers: Tensor,
        pbc: Tensor | None = None,
        atom_graph_cutoff: float = 6.0,
        line_graph_cutoff: float = 4.5,
        device: torch.device | str = "cuda",
        composition: str | None = None,
        check_isolated_atoms: bool = True,
    ) -> None:
        self.device = _cuda_device(device)
        self.cell = cell.to(device=self.device, dtype=torch.float32)
        self.inv_cell = torch.linalg.inv(self.cell)
        self.cells = self.cell.unsqueeze(0)
        self.atomic_numbers = atomic_numbers.to(device=self.device, dtype=torch.int32)
        if pbc is None:
            pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        else:
            pbc = pbc.to(device=self.device, dtype=torch.bool)
        self.pbc = pbc.unsqueeze(0)
        self.n_atoms = int(self.atomic_numbers.shape[0])
        self.batch_idx = torch.zeros(self.n_atoms, dtype=torch.int32, device=self.device)
        self.atom_offsets = [0, self.n_atoms]
        self.compositions = [composition or ""]
        self.atom_graph_cutoff = float(atom_graph_cutoff)
        self.line_graph_cutoff = float(line_graph_cutoff)
        self.check_isolated_atoms = bool(check_isolated_atoms)

        if self.cell.dim() != 2 or tuple(self.cell.shape) != (3, 3):
            raise ValueError(f"cell must have shape (3, 3), got {tuple(self.cell.shape)}")
        if self.pbc.dim() != 2 or tuple(self.pbc.shape) != (1, 3):
            raise ValueError(f"pbc must have shape (3,), got {tuple(self.pbc.squeeze(0).shape)}")
        if self.atomic_numbers.dim() != 1:
            raise ValueError("atomic_numbers must be a 1D tensor")

    @torch.no_grad()
    def build(self, positions: Tensor) -> RadiusGraph:
        with nvtx_range("tensors_to_graph_gpu"):
            positions = positions.to(device=self.device, dtype=torch.float32)
            if positions.dim() != 2 or positions.shape[1] != 3:
                raise ValueError(
                    f"positions must have shape (N, 3), got {tuple(positions.shape)}"
                )
            if positions.shape[0] != self.n_atoms:
                raise ValueError(
                    "positions must have the same length as atomic_numbers "
                    f"({positions.shape[0]} != {self.n_atoms})"
                )
            frac = small_matmul(positions, self.inv_cell) % 1.0
            payload = {
                "atomic_numbers": self.atomic_numbers,
                "frac_coords": frac,
                "positions": small_matmul(frac, self.cell),
                "cells": self.cells,
                "pbc": self.pbc,
                "batch_idx": self.batch_idx,
                "atom_offsets": self.atom_offsets,
                "compositions": self.compositions,
            }
            return _build_graphs_from_payload(
                payload,
                self.atom_graph_cutoff,
                self.line_graph_cutoff,
                self.device,
                check_isolated_atoms=self.check_isolated_atoms,
            )[0]

    __call__ = build


@torch.no_grad()
def tensors_to_graph_gpu(
    positions: Tensor,
    cell: Tensor,
    atomic_numbers: Tensor,
    pbc: Tensor | None = None,
    atom_graph_cutoff: float = 6.0,
    line_graph_cutoff: float = 4.5,
    device: torch.device | str = "cuda",
    composition: str | None = None,
    check_isolated_atoms: bool = True,
) -> RadiusGraph:
    """Build a single-system RadiusGraph directly from CUDA/CPU tensors.

    This is the GPU-resident MD entry point: callers can keep positions, cell,
    and atomic numbers as tensors and avoid the per-step ASE -> NumPy -> CUDA
    packing performed by :func:`atoms_to_graph_gpu`.
    """
    with nvtx_range("tensors_to_graph_gpu"):
        device = _cuda_device(device)
        payload = _extract_tensor_system(
            positions=positions,
            cell=cell,
            atomic_numbers=atomic_numbers,
            pbc=pbc,
            device=device,
            composition=composition,
        )
        return _build_graphs_from_payload(
            payload,
            atom_graph_cutoff,
            line_graph_cutoff,
            device,
            check_isolated_atoms=check_isolated_atoms,
        )[0]


@torch.no_grad()
def atoms_to_graph_gpu(
    atoms,
    atom_graph_cutoff: float = 6.0,
    line_graph_cutoff: float = 4.5,
    device: torch.device | str = "cuda",
    check_isolated_atoms: bool = True,
) -> RadiusGraph | list[RadiusGraph]:
    device = _cuda_device(device)
    batched = not hasattr(atoms, "get_positions")
    atoms_list = list(atoms) if batched else [atoms]
    payload = _extract_atoms(atoms_list, device)

    graphs = _build_graphs_from_payload(
        payload,
        atom_graph_cutoff,
        line_graph_cutoff,
        device,
        check_isolated_atoms=check_isolated_atoms,
    )
    return graphs if batched else graphs[0]
