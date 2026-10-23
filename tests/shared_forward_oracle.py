"""Pre-refactor stage equations retained only as an independent test oracle.

Production owns one forward implementation in the model layers. This module is
deliberately not imported by production, and must not delegate its stage math
back to the shared model methods under test: frozen modules are lowered here to
plain ``torch.nn.functional`` calls (no fused kernel, any dtype incl. FP64).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fastmd.models.matris.functions import FusedSiLU, FusedSigmoid, GatedMLP, MLP


def module_forward(module: nn.Module | None, value: Tensor) -> Tensor:
    """Apply a frozen model module (eval mode) without its own forward."""
    if module is None or isinstance(module, (nn.Identity, nn.Dropout)):
        if isinstance(module, nn.Dropout) and module.training and module.p:
            raise ValueError("the oracle requires dropout in eval mode")
        return value
    if isinstance(module, GatedMLP):
        core = module_forward(module.core_norm, module_forward(module.mlp_core, value))
        gate = module_forward(module.gate_norm, module_forward(module.mlp_gate, value))
        return module_forward(module.activation_func, core) * module_forward(module.activation_gate, gate)
    if isinstance(module, MLP):
        if module.use_fp16:
            raise TypeError("the oracle does not model MLP.use_fp16")
        return module_forward(module.layers, value)
    if isinstance(module, nn.Sequential):
        for child in module:
            value = module_forward(child, value)
        return value
    if isinstance(module, nn.Linear):
        return F.linear(value, module.weight, module.bias)
    if isinstance(module, nn.LayerNorm):
        return F.layer_norm(value, module.normalized_shape, module.weight, module.bias, module.eps)
    if isinstance(module, nn.RMSNorm):
        return F.rms_norm(value, module.normalized_shape, module.weight, module.eps)
    if isinstance(module, nn.BatchNorm1d):
        if module.training or not module.track_running_stats:
            raise ValueError("the oracle requires BatchNorm eval with fixed running statistics")
        return F.batch_norm(value, module.running_mean, module.running_var,
                            module.weight, module.bias, False, module.momentum, module.eps)
    if isinstance(module, (FusedSiLU, nn.SiLU)):
        return F.silu(value)
    if isinstance(module, (FusedSigmoid, nn.Sigmoid)):
        return torch.sigmoid(value)
    raise TypeError(f"module not modeled by the oracle: {type(module).__qualname__}")


def segment_sum(rows: Tensor, indices: Tensor, count: int) -> Tensor:
    return rows.new_zeros((count, rows.shape[1])).index_add(0, indices.long(), rows)


def segment_attention(logits: Tensor, rows: Tensor, indices: Tensor, count: int) -> Tensor:
    """Channel-wise softmax over a complete owner segment; an empty segment is 0."""
    expanded = indices.long()[:, None].expand_as(logits)
    maxima = logits.new_full((count, logits.shape[1]), -torch.inf)
    maxima = maxima.scatter_reduce(0, expanded, logits, reduce="amax", include_self=False)
    exponential = torch.exp(logits - maxima[indices])
    denominator = segment_sum(exponential, indices, count)
    return segment_sum(exponential / denominator[indices] * rows, indices, count)


def attention_rows(layer, target, source, edge):
    message = module_forward(layer.edge_nonlinear_update, torch.cat((edge, target, source), dim=1))
    return (message, module_forward(layer.source_weight_linear, edge),
            module_forward(layer.target_weight_linear, edge))


def attention_nodes(layer, nodes, source_rows, source_logits, source_index,
                    target_rows, target_logits, target_index):
    source = segment_attention(source_logits, source_rows, source_index, len(nodes))
    target = segment_attention(target_logits, target_rows, target_index, len(nodes))
    return (module_forward(layer.node_nonlinear_update, torch.cat((nodes, target, source), dim=1))
            + nodes * layer.node_res_weight)


def attention_edges(layer, edges, messages, *, paired, skip=False):
    if skip:
        return edges
    if paired:
        messages = messages.reshape(len(edges), 2, messages.shape[-1]).mean(dim=1)
    return messages + edges * layer.edge_res_weight


def refinement_rows(layer, target, source, edge, weights, center=None):
    parts = (edge, target, source) if center is None else (edge, center, target, source)
    raw = module_forward(layer.edge_nonlinear_update, torch.cat(parts, dim=1))
    weighted = weights * raw
    edge_input = weighted if layer.graph_type == "atom graph" and layer.use_smoothed_for_delta_edge else raw
    return weighted, edge_input


def refinement_nodes(layer, nodes, weighted_rows, target_index):
    total = segment_sum(weighted_rows, target_index, len(nodes))
    return module_forward(layer.node_FFN, total) + nodes * layer.node_res_weight


def refinement_edges(layer, edges, edge_input, *, paired, skip=False):
    if skip:
        return edges
    delta = module_forward(layer.edge_FFN, edge_input)
    if paired:
        delta = delta.reshape(len(edges), 2, delta.shape[-1]).mean(dim=1)
    return delta + edges * layer.edge_res_weight
