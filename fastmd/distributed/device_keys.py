"""Exact device-side identities for distributed periodic graph entities.

Rows contain ``(target stable ID, source stable ID, sx, sy, sz)``. Sorting
compares full signed int64 columns; IDs never pass through a floating-point or
bounded mixed-radix encoding. All data-dependent work stays on the tensor
device. Dynamic unique output sizes may still require a scalar shape sync.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _validate_keys(keys: Tensor) -> None:
    if (not isinstance(keys, Tensor) or keys.layout != torch.strided
            or keys.ndim != 2 or keys.shape[1] != 5 or keys.dtype != torch.int64):
        raise TypeError("entity keys must be a strided int64 [N, 5] tensor")


def lexsort_rows(keys: Tensor) -> Tensor:
    """Return a stable lexicographic permutation without packing the keys."""
    _validate_keys(keys)
    order = torch.arange(keys.shape[0], device=keys.device)
    if keys.shape[0] < 2:
        return order
    for column in range(4, -1, -1):
        order = order[torch.argsort(keys[order, column], stable=True)]
    return order


def _group_boundaries(sorted_keys: Tensor) -> Tensor:
    return torch.cat((torch.ones(1, dtype=torch.bool, device=sorted_keys.device),
                      (sorted_keys[1:] != sorted_keys[:-1]).any(dim=1)))


def unique_rows(keys: Tensor) -> tuple[Tensor, Tensor]:
    """Sorted unique keys and the original rows' exact inverse mapping."""
    order = lexsort_rows(keys)
    if keys.shape[0] == 0:
        return keys.clone(), order
    sorted_keys = keys[order]
    boundary = _group_boundaries(sorted_keys)
    groups = boundary.long().cumsum(0) - 1
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, groups)
    return sorted_keys[boundary], inverse


def canonicalize_edges(keys: Tensor) -> tuple[Tensor, Tensor]:
    """Canonical keys and 0/1 reverse orientation of each directed incidence.

    Image components must admit their additive inverse in int64 (neighbor
    kernels supply int32 images). No bound such as MAX_IMAGE=10 is assumed.
    Stable IDs can span the complete nonnegative int64 range.
    """
    _validate_keys(keys)
    first, second = keys[:, 0], keys[:, 1]
    sx, sy, sz = keys[:, 2], keys[:, 3], keys[:, 4]
    positive_image = (sx > 0) | ((sx == 0) & ((sy > 0) | ((sy == 0) & (sz > 0))))
    reverse = (first > second) | ((first == second) & positive_image)
    reversed_keys = torch.cat((keys[:, 1:2], keys[:, 0:1], -keys[:, 2:]), dim=1)
    return torch.where(reverse[:, None], reversed_keys, keys), reverse.long()
