"""Owner segment layouts (CSR), prepared once outside the model body.

The bond_partial preparation and the topology epoch describe each owner
segment of their local rows (atom and line, both sides, including the padded
layouts' sink rows) by a stable permutation and offsets. They are ordinary
prepared tensors: version guarded, copied into graph-owned storage and
refreshed on every new topology epoch like all other indices.
"""
from __future__ import annotations

import torch


def _csr(segment, count):
    # Fixed output shapes, no GPU max/item/bincount-result admission. All
    # segment bounds were already checked when preparing the owner topology.
    # Stable ordering also keeps tie reduction order reproducible across a
    # fresh rebuild with identical content.
    segment = segment.to(dtype=torch.int64)
    permutation = torch.argsort(segment, stable=True)
    ordered = segment.index_select(0, permutation)
    offsets = torch.searchsorted(
        ordered, torch.arange(count + 1, device=segment.device, dtype=torch.int64))
    return permutation, offsets
