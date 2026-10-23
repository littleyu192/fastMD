"""ORB's upstream periodic candidate and KNN semantics for single or batch use.

The full-supercell path deliberately uses upstream candidate construction and
neighbor selection. A generic radius list followed by stable nearest-K sorting
differs at capped distance ties and at the upstream supercell boundary.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def orb_wrap_geometry(positions, cells, owners, dtype=torch.float32):
    """Wrap original FP64 geometry before converting to the neural graph dtype."""
    cells64 = cells.double()
    frac64 = torch.linalg.solve(cells64.transpose(1, 2)[owners],
                                positions.double().unsqueeze(-1)).squeeze(-1).remainder(1)
    wrapped64 = torch.bmm(frac64[:, None], cells64[owners]).squeeze(1)
    return wrapped64.to(dtype), frac64.to(dtype), cells64.to(dtype)


@torch.no_grad()
def orb_graph_edges(positions, cells, counts, cutoff, cap, max_candidates=4_000_000,
                    *, edge_method='knn_brute_force'):
    """Return centre, neighbor, image and upstream edge vector tensors.

    Graph construction loops over cells, but all GPU coordinate work stays on
    device. Callers concatenate these graphs for one neural network forward.
    GPU KNN uses upstream FP64 cdist/topk and image-major candidate ordering;
    CPU calculators can explicitly retain upstream scipy KNN instead.
    """
    from orb_models.forcefield.featurization_utilities import (
        construct_minimal_supercell, compute_supercell_neighbors)

    if edge_method not in {'knn_brute_force', 'knn_scipy'}:
        raise ValueError('ORB graph helper supports knn_brute_force or knn_scipy')
    if edge_method == 'knn_scipy' and positions.device.type != 'cpu':
        raise ValueError('ORB scipy graph construction must be on CPU')
    if max_candidates is not None and (type(max_candidates) is not int or max_candidates <= 0):
        raise ValueError('max_candidates must be positive or None')
    centers, neighbors, images, vectors = [], [], [], []
    start = 0
    for system, count in enumerate(counts.tolist()):
        if count <= 0:
            raise ValueError('ORB graph systems must be nonempty')
        local = positions[start:start + count]
        supercell, offsets = construct_minimal_supercell(local, cells[system], cutoff)
        supercell = supercell.transpose(0, 1).reshape(-1, 3)
        candidates = count * len(supercell)
        if max_candidates is not None and candidates > max_candidates:
            raise ValueError(f'ORB batch graph system {system} needs {candidates} KNN candidates; limit is {max_candidates}')
        center, receiver, vector, _ = compute_supercell_neighbors(
            central_cell_positions=local, supercell_positions=supercell,
            radius=cutoff, max_num_neighbors=cap, edge_method=edge_method, n_workers=1)
        centers.append(center + start)
        neighbors.append(receiver.remainder(count) + start)
        images.append(offsets[receiver.div(count, rounding_mode='floor')])
        vectors.append(vector)
        start += count
    if start != len(positions) or len(counts) != len(cells):
        raise ValueError('ORB graph counts do not match positions and cells')
    return tuple(torch.cat(values) for values in (centers, neighbors, images, vectors))


@torch.no_grad()
def atoms_to_orb_graph(atoms, cutoff, max_neighbors, device, *,
                       positions_dtype=torch.float32, check_isolated_atoms=True):
    """ASE single-system input with the same wrap/KNN choice as ORBCalculator.

    Wrapping the original host FP64 input before device transfer matches the
    official ASE converter. Only graph selection changes; the borrowed FastMD
    adapter, installed neural optimizations and model-graph execution survive.
    """
    from orb_models.forcefield.featurization_utilities import map_to_pbc_cell
    from fastmd.adapters.protocol import GraphBatch
    from fastmd.graph.validation import raise_if_isolated_atoms

    if not len(atoms) or not bool(atoms.pbc.all()):
        raise ValueError('ORB shared periodic graph requires nonempty fully periodic atoms')
    cell64 = torch.as_tensor(atoms.cell.array, dtype=torch.float64)
    given64 = torch.as_tensor(atoms.positions, dtype=torch.float64)
    if not bool(torch.isfinite(given64).all() & torch.isfinite(cell64).all()
                & (torch.linalg.det(cell64).abs() > 0)):
        raise ValueError('ORB graph requires finite coordinates and a nonsingular cell')
    wrapped64 = map_to_pbc_cell(given64, cell64)
    fractions64 = torch.linalg.solve(cell64.T, given64.T).T.remainder(1)
    positions = wrapped64.to(device=device, dtype=positions_dtype)
    cell = cell64.to(device=device, dtype=positions_dtype)
    counts = torch.tensor([len(atoms)], dtype=torch.long, device=device)
    method = 'knn_scipy' if positions.device.type == 'cpu' else 'knn_brute_force'
    center, neighbor, images, _ = orb_graph_edges(positions, cell[None], counts,
        cutoff, max_neighbors, max_candidates=None, edge_method=method)
    if check_isolated_atoms:
        degree = torch.bincount(center, minlength=len(atoms))
        raise_if_isolated_atoms(int((degree == 0).sum().item()))
    return GraphBatch(positions=positions, frac_coords=fractions64.to(device=device, dtype=positions_dtype),
        cell=cell, species=torch.as_tensor(atoms.numbers, dtype=torch.int32, device=device), n_real=len(atoms),
        edge_index=torch.stack((center, neighbor), dim=1).int(), edge_image=images.to(positions_dtype),
        n_edges=torch.tensor([len(center)], dtype=torch.long, device=device), cutoff=float(cutoff),
        padding=None, composition=atoms.get_chemical_formula())
