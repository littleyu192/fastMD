"""Merged frozen projections: one wide GEMM feeding several consumers.

Several attention-layer reads of the same feature tensor are frozen no-bias
linear projections (the decomposed-cat3 p0 block plus the source/target
attention weight linears). Executing them as one wide GEMM and returning
narrow views collapses N autograd consumers of the input into one Function,
removing N-1 gradient-accumulation ``add_`` kernels per layer in backward
(the InputBuffer sums that dominate the measured [rows,128] add family).

Constraints honored (see md_perf_analysis/nsys/s3_20260716/S3_PLAN.md):
- single wide mm (bmm/expand historically regressed the graphed path);
- outputs are row-strided views with stride(1)==1 — every consumer must
  accept that (segment attention does; the cat3 epilogue takes SP0);
- frozen weights only; the caller gates on requires_grad and caches the
  concatenated weight outside CUDA-graph capture;
- dead consumers (e.g. layer-4 branches) yield grad=None, which backward
  must skip without materializing zeros.
"""

from __future__ import annotations

import torch
from torch.autograd import Function


class _MergedFrozenProjections(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_cat: torch.Tensor,
        split0: int,
        split1: int,
        split2: int,
    ):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(w_cat)
        ctx.splits = (split0, split1, split2)
        wide = torch.nn.functional.linear(x, w_cat)
        out0 = wide.narrow(1, 0, split0)
        out1 = wide.narrow(1, split0, split1)
        out2 = wide.narrow(1, split0 + split1, split2)
        return out0, out1, out2

    @staticmethod
    def backward(ctx, *grads):
        (w_cat,) = ctx.saved_tensors
        grad_x = None
        offset = 0
        for grad, size in zip(grads, ctx.splits):
            w_part = w_cat.narrow(0, offset, size)
            offset += size
            if grad is None:
                continue
            if grad_x is None:
                grad_x = torch.mm(grad, w_part)
            else:
                grad_x.addmm_(grad, w_part)
        return grad_x, None, None, None, None


def merged_frozen_projections(
    x: torch.Tensor,
    w_cat: torch.Tensor,
    splits: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute [x@W0.T | x@W1.T | x@W2.T] with one GEMM; views returned."""
    return _MergedFrozenProjections.apply(x, w_cat, *splits)


class _ProjectionWithAlias(Function):
    """One frozen projection plus an identity alias of the input.

    Routing a second consumer (e.g. a residual read) through the alias makes
    this Function the input's single autograd consumer, so the two gradient
    contributions meet here and fold into one addmm instead of a separate
    InputBuffer add_ kernel.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w0: torch.Tensor):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(w0)
        return torch.nn.functional.linear(x, w0), x.view_as(x)

    @staticmethod
    def backward(ctx, grad_p0, grad_alias):
        (w0,) = ctx.saved_tensors
        if grad_p0 is None:
            return grad_alias, None
        if grad_alias is None:
            return torch.mm(grad_p0, w0), None
        return torch.addmm(grad_alias, grad_p0, w0), None


def projection_with_alias(
    x: torch.Tensor,
    w0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x @ w0.T, alias-of-x) with a fused-accumulation backward."""
    return _ProjectionWithAlias.apply(x, w0)
