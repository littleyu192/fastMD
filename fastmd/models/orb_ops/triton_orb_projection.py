"""ORB's decomposed first edge projection with a gather-add-SiLU epilogue.

Dense source projections retain PyTorch's GEMM/precision policy. Only the two
indexed additions and the following SiLU are fused; the VJP reduces projected
edge gradients to source nodes before the dense backward GEMMs. This is the
same reuse exploited by MatRIS's indexed concat decomposition, without its
core/gate split (ORB has a single 1024-wide activation).

CPU, FP64 and deterministic execution use differentiable Torch operations.
TRITON_INTERPRET=1 permits exact-kernel unit tests without a GPU.
"""
from __future__ import annotations

import os

import torch
from torch.nn import functional as F
import triton
import triton.language as tl


def gather_add_silu_torch(edge, sent, received, senders, receivers):
    return F.silu(edge + sent[senders] + received[receivers])


@triton.jit
def _forward(P0, P1, P2, SRC, DST, OUT, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    source = tl.load(SRC + row).to(tl.int64)
    target = tl.load(DST + row).to(tl.int64)
    value = tl.load(P0 + row * D + col, col < D, other=0.)
    value += tl.load(P1 + source * D + col, col < D, other=0.)
    value += tl.load(P2 + target * D + col, col < D, other=0.)
    tl.store(OUT + row * D + col, value * tl.sigmoid(value), col < D)


@triton.jit
def _backward(P0, P1, P2, SRC, DST, GRAD, G0, G1, G2,
              D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, B)
    source = tl.load(SRC + row).to(tl.int64)
    target = tl.load(DST + row).to(tl.int64)
    value = tl.load(P0 + row * D + col, col < D, other=0.)
    value += tl.load(P1 + source * D + col, col < D, other=0.)
    value += tl.load(P2 + target * D + col, col < D, other=0.)
    grad = tl.load(GRAD + row * D + col, col < D, other=0.)
    sigmoid = tl.sigmoid(value)
    grad = grad * sigmoid * (1. + value * (1. - sigmoid))
    tl.store(G0 + row * D + col, grad, col < D)
    tl.atomic_add(G1 + source * D + col, grad, col < D, sem="relaxed")
    tl.atomic_add(G2 + target * D + col, grad, col < D, sem="relaxed")


class _GatherAddSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, sent, received, senders, receivers):
        ctx.save_for_backward(edge, sent, received, senders, receivers)
        rows, width = edge.shape
        result = torch.empty_like(edge)
        if rows:
            _forward[(rows,)](edge, sent, received, senders, receivers, result,
                              width, triton.next_power_of_2(width),
                              enable_fp_fusion=False)
        return result

    @staticmethod
    def backward(ctx, grad):
        edge, sent, received, senders, receivers = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Keep the VJP differentiable for Hessians and force derivatives.
            value = edge + sent[senders] + received[receivers]
            sigmoid = value.sigmoid()
            derivative = grad * sigmoid * (1. + value * (1. - sigmoid))
            return (derivative,
                    torch.zeros_like(sent).index_add(0, senders, derivative),
                    torch.zeros_like(received).index_add(0, receivers, derivative),
                    None, None)
        g0 = torch.empty_like(edge)
        g1, g2 = torch.zeros_like(sent), torch.zeros_like(received)
        rows, width = edge.shape
        if rows:
            _backward[(rows,)](edge, sent, received, senders, receivers,
                               grad.contiguous(), g0, g1, g2, width,
                               triton.next_power_of_2(width), enable_fp_fusion=False)
        return g0, g1, g2, None, None


def gather_add_silu(edge, sent, received, senders, receivers):
    """Fuse a single activation, falling back for unsupported input policies."""
    from fastmd.precision import deterministic_mode

    tensors = (edge, sent, received)
    use_triton = (edge.dtype == torch.float32
                  and (edge.is_cuda or os.environ.get("TRITON_INTERPRET") == "1")
                  and all(value.dtype == edge.dtype and value.device == edge.device
                          and value.is_contiguous() for value in tensors)
                  and senders.is_contiguous() and receivers.is_contiguous()
                  and not torch.are_deterministic_algorithms_enabled()
                  and not deterministic_mode())
    if use_triton:
        return _GatherAddSiLU.apply(edge, sent, received, senders, receivers)
    return gather_add_silu_torch(edge, sent, received, senders, receivers)


def projected_edge_silu(nodes, edges, senders, receivers, weights, bias):
    """Project each source once, then fuse the per-edge epilogue and its VJP."""
    edge_weight, send_weight, receive_weight = weights
    return gather_add_silu(F.linear(edges, edge_weight, bias),
                           F.linear(nodes, send_weight), F.linear(nodes, receive_weight),
                           senders, receivers)
