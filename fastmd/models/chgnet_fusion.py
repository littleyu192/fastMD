"""Opt-in CHGNet gather/concat/Linear decomposition with frozen inference weights.

Project atom and bond tables before gathering their repeated rows. Core and gate
weights share each projection. Hidden SiLU layers can use MatRIS's fused indexed
epilogue; an AngleUpdate without a hidden layer must keep its output normalization
before its SiLU, so that case uses the Torch decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fastmd.adapters.patches import PatchRecord, apply_patches, revert_patches
from fastmd.engine import recorder as _recorder


_HASHES = {
    "AtomConv": "d161d41a6152c6d7f7db04ec197561c80563a479fbf9bafd88ab2359c8171665",
    "BondConv": "163796057c24c12e05ba0bab624d32d8aecf0a41d6c2d9fafd3d7d3a7a000acc",
    "AngleUpdate": "4abb95fb3e2a25aa9b0fff4c38e080933637a3b4ff44580c5e868c78597dc10a",
    "GatedMLP": "bf9d7ac77d5973aa61afc2ac1ac1e02e17310c159741ef638820e6c8c11d7f3d",
    "MLP": "10451e01956bfa4c408338ca0b773acf7931fa8b1d42016a557f97d41e42833b",
}


def _record(name, cls, *, replacement=None):
    where = "layers" if cls in ("AtomConv", "BondConv", "AngleUpdate") else "functions"
    return PatchRecord(name=f"chgnet.operators.{name}", target=f"chgnet.model.{where}.{cls}.forward",
                       distribution="chgnet", versions=("0.4.2",), source_sha256=_HASHES[cls],
                       reason="opt-in indexed first Linear with frozen weights and input-gradient parity",
                       replacement=replacement, per_instance=replacement is not None)


def _first(mlp):
    layers = list(mlp.layers)
    index = 0
    while index < len(layers) and type(layers[index]) is nn.Dropout and not layers[index].training:
        index += 1
    if index == len(layers) or type(layers[index]) is not nn.Linear:
        raise ValueError("CHGNet indexed projection requires Linear after inference-only Dropout")
    return index, layers[index]


def _first_order_only(gradient):
    if torch.is_grad_enabled():
        raise RuntimeError("CHGNet Triton indexed projections support first derivatives only; "
                           "select operator backend='torch' for create_graph=True or Hessians")
    return gradient


class _Projection:
    def __init__(self, gated, widths, backend):
        from chgnet.model.functions import GatedMLP, MLP

        if type(gated) is not GatedMLP or any(type(mlp) is not MLP for mlp in (gated.mlp_core, gated.mlp_gate)):
            raise ValueError("CHGNet indexed projections require the source-pinned GatedMLP and MLP classes")
        self.gated, self.widths, self.backend = gated, tuple(widths), backend
        self.core_index, self.core = _first(gated.mlp_core)
        self.gate_index, self.gate = _first(gated.mlp_gate)
        if self.core.in_features != sum(widths) or self.gate.in_features != sum(widths):
            raise ValueError("CHGNet indexed projection feature widths do not match first Linear")
        if self.core.out_features != self.gate.out_features:
            raise ValueError("CHGNet indexed core and gate first Linear must have equal output widths")
        if any(p.requires_grad for p in gated.parameters()) or gated.training:
            raise ValueError("CHGNet operator rewrites require frozen inference weights")
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in gated.modules()):
            raise ValueError("CHGNet operator rewrites cannot bypass user hooks on the gated MLP")
        self.hidden_silu = all(
            len(mlp.layers) > index + 1 and type(mlp.layers[index + 1]) is nn.SiLU
            for mlp, index in ((gated.mlp_core, self.core_index), (gated.mlp_gate, self.gate_index)))
        self.key = None
        self._refresh()

    def _refresh(self):
        if any(m.training for m in self.gated.modules()) or any(p.requires_grad for p in self.gated.parameters()):
            raise RuntimeError("CHGNet operator rewrites require frozen inference weights; close the adapter "
                               "to restore upstream layers before training")
        sources = (self.core.weight, self.gate.weight, self.core.bias, self.gate.bias)
        key = tuple(None if p is None else (p.data_ptr(), p._version, p.device, p.dtype) for p in sources)
        if key == self.key:
            return
        if self.core.weight.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CHGNet projection weights changed after warm-up; warm up before capture")
        with torch.no_grad():
            self.weights = tuple(torch.cat((a, b), dim=0).contiguous() for a, b in zip(
                self.core.weight.detach().split(self.widths, dim=1),
                self.gate.weight.detach().split(self.widths, dim=1)))
            a, b = self.core.bias, self.gate.bias
            if a is None and b is None:
                self.bias = None
            else:
                self.bias = torch.cat((a.detach() if a is not None else self.core.weight.new_zeros(
                    self.core.out_features), b.detach() if b is not None else self.gate.weight.new_zeros(
                    self.gate.out_features))).contiguous()
        self.key = key

    def __call__(self, tables, indices, kind):
        self._refresh()
        fused = None
        rows = indices[0].shape[0]
        source_work = sum(table.shape[0] * width for table, width in zip(tables, self.widths))
        gathered_work = rows * sum(self.widths)
        if source_work >= gathered_work:
            # A sparse or padded angle graph may reference only a few of the
            # bond table's rows. Projecting every bond then costs more than the
            # original small gathered GEMM. This decision uses static shapes.
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("chgnet_indexed_projection", "upstream", {
                    "kind": kind, "rows": rows, "reason": "no_projection_reuse",
                    "source_rows": [t.shape[0] for t in tables], "widths": self.widths})
            return self.gated(torch.cat(tuple(table if index is None else table.index_select(0, index)
                                              for table, index in zip(tables, indices)), dim=1))
        if (self.backend == "triton" and all(t.is_cuda and t.dtype == torch.float32 for t in tables)
                and not torch.is_autocast_enabled("cuda") and self.hidden_silu and rows):
            from fastmd.engine.kernels.decomposed_indexed_cat_silu_linear import (
                decomposed_indexed_cat3_silu_linear, decomposed_indexed_cat4_silu_linear)
            if kind == "atom":
                # Upstream order is [centre atom | bond | neighbour atom]. The
                # MatRIS interface takes [aligned | gathered a | gathered b].
                p0 = F.linear(tables[1], self.weights[1]).index_select(0, indices[1])
                fused = decomposed_indexed_cat3_silu_linear(
                    p0, tables[0], indices[0], indices[2],
                    (self.weights[1], self.weights[0], self.weights[2]), self.bias,
                    precomputed_p0=p0, strict_errors=True)
            else:
                # Upstream [bond i | bond j | angle | centre atom].
                fused = decomposed_indexed_cat4_silu_linear(
                    tables[2], tables[3], indices[3], tables[0], indices[0], indices[1],
                    (self.weights[2], self.weights[3], self.weights[0], self.weights[1]),
                    self.bias, strict_errors=True)
        skip = 2 if fused is not None else 1
        if fused is not None:
            # MatRIS's custom epilogue VJP is first-order. Refuse a requested
            # differentiable VJP before it can silently discard curvature.
            for value in fused:
                if value.requires_grad:
                    value.register_hook(_first_order_only)
        if fused is None:
            projected = []
            for table, weight, index in zip(tables, self.weights, indices):
                value = F.linear(table, weight)
                projected.append(value if index is None else value.index_select(0, index))
            value = projected[0]
            for part in projected[1:]:
                value = value + part
            if self.bias is not None:
                value = value + self.bias
            fused = value.split(self.core.out_features, dim=1)
        core, gate = fused
        for layer in list(self.gated.mlp_core.layers)[self.core_index + skip:]:
            core = layer(core)
        for layer in list(self.gated.mlp_gate.layers)[self.gate_index + skip:]:
            gate = layer(gate)
        if self.gated.norm is not None:
            core, gate = self.gated.bn1(core), self.gated.bn2(gate)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("chgnet_indexed_projection", "triton" if skip == 2 else "torch",
                       {"kind": kind, "rows": core.shape[0], "source_rows": [t.shape[0] for t in tables],
                        "widths": self.widths, "hidden_silu": self.hidden_silu})
        return self.gated.activation(core) * self.gated.sigmoid(gate)


# These layer tails reproduce chgnet 0.4.2 chgnet/model/layers.py (Modified BSD,
# Copyright 2023 The Regents of the University of California through LBNL and
# University of California, Berkeley). Only the gathered first projection changes.
def _atom(self, atom_feas, bond_feas, bond_weights, atom_graph, directed2undirected):
    from chgnet.model.layers import aggregate

    messages = self._fastmd_chgnet_projection(
        (atom_feas, bond_feas, atom_feas),
        (atom_graph[:, 0], directed2undirected, atom_graph[:, 1]), "atom")
    messages = messages * bond_weights.index_select(0, directed2undirected)
    result = aggregate(messages, atom_graph[:, 0], average=False, num_owner=len(atom_feas))
    if self.use_mlp_out:
        result = self.mlp_out(result)
    if self.resnet:
        result = result + atom_feas
    return self.atom_norm(result) if self.atom_norm is not None else result


def _bond_message(self, atom_feas, bond_feas, angle_feas, bond_graph, kind):
    return self._fastmd_chgnet_projection(
        (bond_feas, bond_feas, angle_feas, atom_feas),
        (bond_graph[:, 1], bond_graph[:, 2], None, bond_graph[:, 0]), kind)


def _bond(self, atom_feas, bond_feas, bond_weights, angle_feas, bond_graph):
    from chgnet.model.layers import aggregate

    messages = _bond_message(self, atom_feas, bond_feas, angle_feas, bond_graph, "bond")
    messages = messages * bond_weights.index_select(0, bond_graph[:, 1])
    messages = messages * bond_weights.index_select(0, bond_graph[:, 2])
    result = aggregate(messages, bond_graph[:, 1], average=False, num_owner=len(bond_feas))
    if self.use_mlp_out:
        result = self.mlp_out(result)
    if self.resnet:
        result = result + bond_feas
    return self.bond_norm(result) if self.bond_norm is not None else result


def _angle(self, atom_feas, bond_feas, angle_feas, bond_graph):
    result = _bond_message(self, atom_feas, bond_feas, angle_feas, bond_graph, "angle")
    if self.resnet:
        result = result + angle_feas
    return self.angle_norm(result) if self.angle_norm is not None else result


@dataclass
class Installation:
    outcomes: list = field(default_factory=list)
    auxiliary: list = field(default_factory=list)
    restored: bool = False

    def restore(self):
        if self.restored:
            return
        revert_patches(self.outcomes)
        for layer, owned, value in reversed(self.auxiliary):
            if owned:
                layer._fastmd_chgnet_projection = value
            else:
                del layer._fastmd_chgnet_projection
        self.restored = True


def install(module, backend):
    """Install only on this model's layer instances, rolling back on any refusal."""
    from chgnet.model.layers import AtomConv, BondConv, AngleUpdate

    result = Installation()
    try:
        result.outcomes.extend(apply_patches([_record("gated_mlp", "GatedMLP"), _record("mlp", "MLP")]))
        if not all(outcome.applied for outcome in result.outcomes):
            raise RuntimeError("CHGNet operator rewrite MLP source pins were declined")
        for name, layer in module.named_modules():
            if type(layer) not in (AtomConv, BondConv, AngleUpdate):
                continue
            atom_dim = module.atom_embedding.embedding.embedding_dim
            bond_dim = module.bond_embedding.out_features
            if type(layer) is AtomConv:
                widths, gated, implementation = (atom_dim, bond_dim, atom_dim), layer.twoBody_atom, _atom
            else:
                angle_dim = module.angle_embedding.out_features
                widths, gated = (bond_dim, bond_dim, angle_dim, atom_dim), layer.twoBody_bond
                implementation = _bond if type(layer) is BondConv else _angle
            projection = _Projection(gated, widths, backend)
            # A model's ordinary state_dict and original weight names are retained.
            result.auxiliary.append((layer, "_fastmd_chgnet_projection" in vars(layer),
                                     vars(layer).get("_fastmd_chgnet_projection")))
            layer._fastmd_chgnet_projection = projection
            added = apply_patches([_record(name, type(layer).__name__,
                                          replacement=MethodType(implementation, layer))], instance=layer)
            result.outcomes.extend(added)
            if not all(outcome.applied for outcome in added):
                raise RuntimeError(f"CHGNet operator rewrite refused: {[o.reason for o in added]}")
    except BaseException:
        result.restore()
        raise
    return result
