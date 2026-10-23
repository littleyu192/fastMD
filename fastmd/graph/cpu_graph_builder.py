"""Ragged, edges-only CPU graphs from ASE's periodic neighbor list.

This path rebuilds the exact radius graph on every call. It neither imports a
GPU neighbor-list implementation nor pads edges or invents non-periodic cells.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from ase.neighborlist import neighbor_list

from fastmd.adapters.protocol import GraphBatch
from fastmd.graph.validation import raise_if_isolated_atoms


def atoms_to_edges_cpu(atoms, atom_graph_cutoff: float, *, check_isolated_atoms: bool = True,
                       positions_dtype: torch.dtype = torch.float32) -> GraphBatch:
    """Build directed ``(centre, neighbour)`` edges, retaining every atom.

    The displacement is ``positions[j] + image @ cell - positions[i]``.
    Periodic coordinates are wrapped on a copy before finding the corresponding
    images; non-periodic directions, the caller's atoms, and its cell stay intact.
    ``positions_dtype`` independently selects geometric precision, including for
    models whose FP32 neural weights consume FP64 geometry.
    """
    if not math.isfinite(atom_graph_cutoff) or atom_graph_cutoff <= 0:
        raise ValueError("The CPU neighbor cutoff must be finite and positive")
    if positions_dtype not in (torch.float32, torch.float64):
        raise ValueError("CPU graph positions_dtype must be torch.float32 or torch.float64")
    if not len(atoms):
        raise ValueError("The CPU neighbor graph requires at least one atom")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError("The CPU neighbor graph requires finite positions and cell")
    # ASE needs a nonzero lattice vector for each periodic direction. For an
    # entirely non-periodic molecule it can work with an unspecified cell.
    if np.any(atoms.pbc & ~np.any(atoms.cell.array != 0, axis=1)):
        raise ValueError("Every periodic direction requires a nonzero cell vector")
    wrapped = atoms.copy()
    wrapped.wrap(eps=0.0)
    centre, neighbour, image = neighbor_list("ijS", wrapped, atom_graph_cutoff, self_interaction=False)
    image = np.asarray(image).reshape(-1, 3)
    # ASE groups by centre, but does not promise neighbour/image order.
    order = np.lexsort((image[:, 2], image[:, 1], image[:, 0], neighbour, centre))
    edges = np.stack((centre[order], neighbour[order]), axis=1)
    image = image[order]
    if check_isolated_atoms:
        raise_if_isolated_atoms(np.count_nonzero(np.bincount(centre, minlength=len(atoms)) == 0))
    return GraphBatch(
        positions=torch.tensor(wrapped.positions, dtype=positions_dtype, device="cpu"),
        frac_coords=torch.tensor(wrapped.get_scaled_positions(wrap=False), dtype=positions_dtype, device="cpu"),
        cell=torch.tensor(wrapped.cell.array, dtype=positions_dtype, device="cpu"),
        species=torch.tensor(wrapped.numbers, dtype=torch.int32, device="cpu"),
        n_real=len(wrapped),
        edge_index=torch.tensor(edges, dtype=torch.int32, device="cpu"),
        edge_image=torch.tensor(image, dtype=positions_dtype, device="cpu"),
        n_edges=torch.tensor([len(edges)], dtype=torch.int64, device="cpu"),
        cutoff=float(atom_graph_cutoff), padding=None, composition=wrapped.get_chemical_formula(),
    )
