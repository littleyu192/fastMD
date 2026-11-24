"""GPU cell-list candidates with collective overflow retry and exact filtering.

Reuse the nvalchemiops matrix API (``fastmd.graph.neighbor_list_backend``),
also used underneath TorchSim's Alchemiops backend. It queries every local
owned+halo center; only owned-center rows are retained. No global coordinates
or dense N squared matrix are constructed. Final topology uses MatRIS's
original CUDA math, not Warp's candidate-distance predicate.
"""
from __future__ import annotations

import torch

from fastmd.graph.geometry import small_matmul


def neighbor_candidates(positions, cell, owner_ranks, rank, cutoff, context, *,
                        candidate_margin=1e-6, initial_capacity=None):
    """Owned-center directed rows within ``cutoff`` (strict), with their lengths.

    ``positions`` rows are sorted by stable atom ID. The cutoff decision is
    canonical (the bond_partial exchange, K4/R11): a row exists iff the length
    of its bond's *canonical* row (``canonicalize_edges``) is below the
    cutoff. The two ends of a boundary bond then decide from bitwise-equal
    coordinates with the same arithmetic and agree on the bond; per-row
    lengths (returned, used for the three-body cutoff) are unchanged. (The
    rows exchange's per-row decision was removed with it in step 4.2.)
    """
    from fastmd.graph.neighbor_list_backend import estimate_max_neighbors, neighbor_list, require

    require("GPU neighbor candidates need the nvalchemiops neighbor list")

    device, count = positions.device, len(positions)
    if not positions.is_cuda:
        raise ValueError("device neighbor candidates require CUDA")
    capacity = (max(16, estimate_max_neighbors(cutoff + candidate_margin, safety_factor=2.0))
                if initial_capacity is None else int(initial_capacity))
    if capacity <= 0:
        raise ValueError("neighbor capacity must be positive")
    queried = count if bool((owner_ranks == rank).any()) else 0
    batch = torch.zeros(count, device=device, dtype=torch.int32)
    pbc = torch.ones((1, 3), device=device, dtype=torch.bool)
    attempts = 0
    while True:
        attempts += 1
        error, maximum = None, 0
        try:
            matrix = torch.full((queried, capacity), count, dtype=torch.int32, device=device)
            images = torch.zeros((queried, capacity, 3), dtype=torch.int32, device=device)
            counts = torch.zeros(queried, dtype=torch.int32, device=device)
            if queried:
                neighbor_list(positions=positions, cutoff=cutoff + candidate_margin,
                              cell=cell.unsqueeze(0), pbc=pbc, batch_idx=batch,
                              method="batch_cell_list", half_fill=False,
                              neighbor_matrix=matrix, neighbor_matrix_shifts=images,
                              num_neighbors=counts, return_neighbor_list=False)
                # The installed kernel increments the count even when a slot
                # overflows; its writes are bounds guarded. Never truncate.
                maximum = int(counts.max().item())
        except Exception as exc:
            error = f"GPU neighbor candidates: {type(exc).__name__}: {exc}"
        states = context.all_gather_object((error, maximum > capacity))
        context._raise_errors([entry[0] for entry in states])
        if not any(entry[1] for entry in states):
            break
        if maximum > capacity:
            capacity = max(capacity * 2, maximum)
    if not queried:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty.reshape(0, 1).expand(0, 3), positions.new_empty(0), {
            "candidate_centers": 0, "owned_centers": 0, "neighbor_capacity": capacity,
            "neighbor_attempts": attempts, "owned_candidate_edges": 0,
        }
    valid = ((torch.arange(capacity, device=device)[None, :] < counts[:, None])
             & (owner_ranks[:, None] == rank))
    center, slots = torch.where(valid)
    source, shifts = matrix[center, slots].long(), images[center, slots].long()
    candidate_count = len(center)
    displacement = positions[source] - positions[center]
    # neighbor_list_nvidia's image transform: small_matmul never runs in TF32;
    # a batched cuBLAS product may, depending on shape, under allow_tf32.
    displacement = displacement + small_matmul(
        shifts.float().unsqueeze(-2), cell.expand(len(center), 3, 3)).squeeze(-2)
    distances = displacement.norm(dim=1)
    keep = _canonical_lengths(positions, cell, center, source, shifts) < cutoff
    keep = keep & ((center != source) | (shifts != 0).any(dim=1))
    return center[keep], source[keep], shifts[keep], distances[keep], {
        "candidate_centers": queried, "owned_centers": int((owner_ranks == rank).sum().item()),
        "neighbor_capacity": capacity, "neighbor_attempts": attempts,
        "owned_candidate_edges": candidate_count,
    }


def _canonical_lengths(positions, cell, center, source, shifts):
    """Length of each row's canonical row, identical at both ends of its bond.

    Row (c, n, S) has key (id_c, id_n, S); IDs are sorted with the rows, so
    comparing row indices compares IDs. The canonical row is the reverse
    (n, c, -S) iff id_c > id_n or (id_c == id_n and S is lexicographically
    positive), as in ``canonicalize_edges``. Its length is evaluated with the
    per-row expression of ``neighbor_candidates`` (a canonical row keeps its
    own decision); the row-wise norm and cell product are batch-invariant, so
    both ends compute the same bits from the same coordinates.
    """
    sx, sy, sz = shifts[:, 0], shifts[:, 1], shifts[:, 2]
    positive = (sx > 0) | ((sx == 0) & ((sy > 0) | ((sy == 0) & (sz > 0))))
    flip = (center > source) | ((center == source) & positive)
    first = torch.where(flip, source, center)
    second = torch.where(flip, center, source)
    image = torch.where(flip[:, None], -shifts, shifts)
    displacement = positions[second] - positions[first]
    displacement = displacement + small_matmul(
        image.to(positions.dtype).unsqueeze(-2), cell.expand(len(first), 3, 3)).squeeze(-2)
    return displacement.norm(dim=1)
