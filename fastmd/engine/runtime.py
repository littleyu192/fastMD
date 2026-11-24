"""The model's runtime hooks, design layer L0<->L6 (docs/unified_engine_design.md §3.2, §5.7 T1).

``EngineRuntime`` is the one implementation of the eight hooks the interaction
layers call (``attention_rows/route/nodes/edges`` and
``refinement_rows/route/nodes/edges``, called from
``matris/models/matris/interaction_block.py``), plus ``for_block``/``for_layer``,
``select_line_nodes`` (block level) and ``local``. It is parameterized by a ``Communicator`` (``fastmd.engine.comm``),
a ``GradBackend`` (L3) and the row routes of the layout:

* identity routes (P=1, ``routes=None``, no backend): a hook calls the layer
  method directly and inserts no stage or exchange (INV-6). Only the layer
  level is used: ``MatRIS.forward`` and ``Interaction_Block.forward`` keep
  ``runtime=None`` at block level, so the single-GPU control flow
  (torch.compile, checkpoint, the ``torch.where`` line-graph selection) is
  unchanged, and the layers take ``LOCAL_RUNTIME`` where they took the
  ``LOCAL_FORWARD_RUNTIME`` singleton. The compiled layers trace these hooks,
  so the identity branch keeps that singleton's bodies and call depth, and
  holds no closure (see ``fastmd.engine.dispatch`` on torch.compile).
* exchange routes (P>1, ``BondPartialRuntime`` in MatRIS's multi-GPU package,
  ``fastmd.models.matris.distributed.runtime``): every local stage runs
  through ``backend.local`` and every row transport through
  ``backend.exchange``, in the fixed stage and exchange order of the prepared
  schedule (G-D4). Since step 4.2 the P>1 backend is
  ``fastmd.engine.grad.AutogradBackend`` (one autograd backward; the
  StageTape is gone) and the routes are the bond_partial exchange's
  (``fastmd.models.matris.distributed.bond_partial.BondPartialRoutes``). The rows
  exchange's routed hooks of this class (and its routes object) were
  removed with it: the base class keeps the identity branches, the local
  stage and the line-graph selection that ``BondPartialRuntime`` inherits.

The pre-3.1b runtimes and their ``MATRIS_RUNTIME_ENGINE=legacy`` rollback
were removed in step 4.2; for one version setting the variable to anything
but ``unified`` is an error at import.

This module must not import ``fastmd.models.matris`` or ``fastmd.distributed``.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from fastmd.engine.comm import LocalCommunicator

if os.environ.get("MATRIS_RUNTIME_ENGINE", "unified") != "unified":
    raise ValueError("MATRIS_RUNTIME_ENGINE: the legacy runtimes were removed in step 4.2; "
                     f"only 'unified' remains, got {os.environ['MATRIS_RUNTIME_ENGINE']!r}")

# The model's forward entry points: the layer methods these hooks call and the model methods the forwards call
# (MatRIS's admission and the run boundary's model stamp in fastmd.distributed.execution read them).
# Shared forward entrypoints are executable model state, just like forward.
# Reject instance replacements at public admission and stamp internal callers'
# bindings so prepare/run cannot execute two different model programs.
# Every method of a stock MatRIS module class that a forward or its recomputation reaches through an attribute of
# the module (an instance attribute would win) is listed, at P=1 and P>1 (the P>1 hooks call the layers'
# attention_node_update/refinement_node_update; the gated MLP's fused paths call its private helpers);
# tests/test_shared_forward_guards.py checks that every method of those classes is listed or known not to be one.
MODEL_FORWARD_METHODS = frozenset({
    "forward", "forward_features", "embed_atoms", "normalize_readout",
    "atomic_energy", "atomic_energy_rows", "atomic_energies", "_graph_sum", "_per_atom_reference",
    "forward_lengths", "forward_indexed",
    "forward_vectors", "forward_basis", "attention_rows", "attention_nodes",
    "attention_edges", "attention_node_update", "refinement_rows", "refinement_payload",
    "refinement_nodes", "refinement_node_update", "refinement_edges", "_fused_edge_ffn", "wrapper_attn_layer",
    "wrapper_refine_layer", "_attn_layer_call", "_refine_layer_call", "_forward", "_normalize_real_counts",
    "_can_fuse_indexed_cat_first_linear", "_first_linear_modules",
    "_split_first_linear_params", "aligned_first_linear_block",
    "forward_aligned_gather_cat3", "forward_aligned_gather_cat4", "forward_gathered_chunks", "forward_with_residual",
    "_run_after_first", "_run_after_first_activation", "_finish_gate", "_folded_biases", "_gated_tail",
    "_second_layers", "_second_linears", "_register_gate_inputs", "_register_gate_output",
    "_merged_attention_projection", "_combined_swish_linear_weight",
})


@runtime_checkable
class GradBackend(Protocol):
    """Differentiation boundaries of one forward (design §3.2 L3).

    ``input`` registers a leaf, ``local`` runs a rank-local stage on values of
    this backend, ``exchange`` moves rows along a binding and ``energy_anchor``
    ties the energy to every exchange of the forward; the owner differentiates
    that seed once (``AutogradBackend.backward``). The P>1 implementation is
    ``fastmd.engine.grad.AutogradBackend`` (steps 4.1/4.2).
    """

    def input(self, x, name: str = ""): ...

    def local(self, fn: Callable, *values, name: str = ""): ...

    def exchange(self, value, binding, name: str = ""): ...

    def energy_anchor(self, energy): ...


class EngineRuntime:
    """Runtime hooks of the shared model forward, at root, block or layer level.

    ``for_block(i, block)`` and ``for_layer(name)`` return views bound to one
    interaction block / layer; the eight hooks are called on layer views,
    ``select_line_nodes`` on block views. A view is built once per block and
    layer of an evaluation (as the pre-3.1b block/layer runtimes were).
    """

    def __init__(self, comm: Any = None, backend: GradBackend | None = None, routes: Any = None):
        if (backend is None) != (routes is None):
            raise ValueError("exchange routes need a GradBackend; identity routes take none")
        # ``comm``: the context's Communicator (Local at P=1, DomainCommunicator at P>1).
        self.comm, self.backend, self.routes = comm, backend, routes
        self.index = self.block = self.name = self.line_layer = None

    def _view(self, index, block, name) -> EngineRuntime:
        # Plain attribute stores: touching ``__dict__`` would materialize it
        # and slow every later attribute read of the view (CPython 3.12).
        view = object.__new__(type(self))
        view.comm, view.backend, view.routes = self.comm, self.backend, self.routes
        view.index, view.block, view.name, view.line_layer = index, block, name, None
        return view

    def for_block(self, index: int, block) -> EngineRuntime:
        return self._view(index, block, None)

    def for_layer(self, name: str) -> EngineRuntime:
        view = self._view(self.index, self.block, name)
        routes = self.routes
        if routes is not None:
            raise TypeError("P>1 exchange routes are BondPartialRuntime's")
        return view

    def local(self, fn: Callable, *args, name: str = ""):
        """A rank-local stage: ``fn(*args)`` itself with identity routes."""
        if self.routes is None:
            return fn(*args)
        return self.backend.local(fn, *args, name=name)

    def select_line_nodes(self, new, old, mask):
        """Block level: the line-graph result where ``mask``, else the input edge
        features (the P=1 ``torch.where`` of ``Interaction_Block.forward``). With
        exchange routes ``mask`` must be collectively admitted (INV-7)."""
        if self.routes is None:
            return torch.where(mask, new, old)
        return self.backend.local(lambda values, original: torch.where(mask, values, original),
                                  new, old, name=f"block_{self.index}.{self.line_layer}.select")

    # --- the eight layer hooks; the identity branch keeps the pre-3.1b local runtime's bodies ---

    def attention_rows(self, layer, nodes, edges, graph, directed2undirected):
        if self.routes is None:
            return layer.attention_rows(nodes, edges, graph, directed2undirected)
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")

    def attention_route(self, rows, graph):
        if self.routes is None:
            return rows
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")

    def attention_nodes(self, layer, nodes, rows, graph):
        if self.routes is None:
            message, source_logits, target_logits = rows
            return layer.attention_nodes(
                nodes, message, source_logits, graph["source_index"],
                message, target_logits, graph["target_index"], graph,
            )
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")

    def attention_edges(self, layer, edges, rows, graph, directed2undirected):
        if self.routes is None:
            return layer.attention_edges(edges, rows[0], graph, directed2undirected)
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")

    def refinement_rows(self, layer, nodes, edges, radial, graph,
                        directed2undirected, atom_feat):
        if self.routes is None:
            raw, weight, residual = layer.refinement_rows(
                nodes, edges, radial, graph, directed2undirected, atom_feat,
            )
            if layer.graph_type == "atom graph" and layer.use_smoothed_for_delta_edge:
                weighted, edge_input = layer.refinement_payload(raw, weight)
                return weighted, None, edge_input, residual
            # Keep weight separate so the local fused weighted-segment reduction
            # never needs to materialize the full weighted message tensor.
            return raw, weight, raw, residual
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")

    def refinement_route(self, rows, graph):
        if self.routes is None:
            return rows
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")

    def refinement_nodes(self, layer, nodes, rows, graph):
        if self.routes is None:
            return layer.refinement_nodes(
                nodes, rows[0], rows[1], graph["target_index"], graph,
            )
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")

    def refinement_edges(self, layer, edges, rows, graph,
                         directed2undirected, skip_edge_update):
        if self.routes is None:
            # A merged projection may return a residual alias whose shared VJP
            # eliminates a separate accumulation.  Keep that alias alive here.
            residual = edges if skip_edge_update else rows[3]
            return layer.refinement_edges(
                residual, rows[2], graph, directed2undirected, skip_edge_update,
            )
        raise TypeError("P>1 exchange routes are BondPartialRuntime's")


# P=1: the layer-level singleton (identity routes); blocks keep runtime=None.
LOCAL_RUNTIME = EngineRuntime(LocalCommunicator())
