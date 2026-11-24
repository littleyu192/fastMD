"""MatRIS's multi-GPU runtime hooks (``BondPartialRuntime``).

The P>1 implementation of the hooks MatRIS's interaction layers call
(``fastmd.engine.runtime.EngineRuntime``, design layer L0<->L6 of
docs/unified_engine_design.md), on the routes of the boundary-bond partial
exchange (``fastmd.models.matris.distributed.bond_partial.BondPartialRoutes``),
which builds one per evaluation (``run_bond_partial_forward``).
"""
from __future__ import annotations

import torch

from fastmd.engine.runtime import EngineRuntime


class BondPartialRuntime(EngineRuntime):
    """P>1 hooks of the symmetric boundary-bond partial exchange (A2, design Q1).

    ``routes`` is a ``fastmd.models.matris.distributed.bond_partial.BondPartialRoutes``: the
    local graph (every bond with an owned end, both directed rows of each;
    the local atoms are the owned atoms followed by their ghosts), the
    exchange bindings and the partial arithmetic. Stages run through
    ``backend.local`` and transports through ``backend.exchange``, in the
    order of the prepared schedule (G-D4):

    * atom layers: the single-GPU layer methods on the local graph, with the
      block input's ghost rows appended to the owned node rows. Owned
      segments are complete; the node update runs on the owned rows only and
      every local bond is updated (both replicas of a boundary bond).
    * line layers: rows and the local segment reduction over the local
      triplets; each line slot's local partial goes to the other end of its
      bond in the exchange that also carries the next atom layer's ghost rows
      (two stages per block), and both ends merge the same two partials in
      canonical order before updating every local bond.

    Block 0's ghost features are computed locally (``routes.initial_ghosts``).
    """

    def for_layer(self, name: str) -> BondPartialRuntime:
        view = self._view(self.index, self.block, name)
        if name.startswith("line_"):
            # Names the stage of the next select_line_nodes on this block (RG-20).
            self.line_layer = name
        view.prefix = f"block_{self.index}.{name}"
        view.line = name.startswith("line_")
        view.final = self.index + 1 == self.routes.block_count
        return view

    def attention_rows(self, layer, nodes, edges, graph, directed2undirected):
        routes = self.routes
        if self.line:
            fold = routes.fold_line_edges

            def compute(bonds, triplets, owned=None):
                message, source_logits, target_logits = layer.attention_rows(bonds, triplets, graph)
                source, target, source_lse, target_lse = graph["_ops"].segment_attention(
                    source_logits, message, graph["source_index"],
                    target_logits, message, graph["target_index"], bonds.shape[0], graph,
                    return_lse=True)
                sent = routes.attention_partials(target_lse, target, source_lse, source)
                # Folded (saved-tensor recompute): the triplet update leaves this stage
                # instead of the message, which a stage output would pin.
                first = layer.attention_edges(triplets, message, graph) if fold else message
                return first, target, source, sent, routes.pack(owned, sent, self.index, "attention")

            owned = routes.fused_nodes(self.index, "attention")
            args = (nodes, edges) if owned is None else (nodes, edges, owned)
            return self.backend.local(compute, *args, name=f"{self.prefix}.rows")
        ghosts, unpack = routes.ghosts(self, nodes, "attention")

        def compute(owned, halo, bonds):
            return layer.attention_rows(torch.cat((owned, unpack(halo))), bonds, graph,
                                        directed2undirected)

        return self.backend.local(compute, nodes, ghosts, edges, name=f"{self.prefix}.rows")

    def attention_route(self, rows, graph):
        if not self.line:
            return rows
        received = self.backend.exchange(rows[4], self.routes.binding(self.index, "attention"),
                                         name=f"{self.prefix}.partials")
        self.routes.post_ghosts(self.index, "attention", received)
        return rows[1], rows[2], rows[3], received

    def attention_nodes(self, layer, nodes, routed, graph):
        routes = self.routes
        if self.line:
            def compute(bonds, target, source, sent, received):
                target, source = routes.merge_attention(target, source, sent, received, self.index)
                return layer.attention_node_update(bonds, target, source, graph)

            value = self.backend.local(compute, nodes, *routed, name=f"{self.prefix}.nodes")
            routes.check(self.prefix, value)
            return value

        def compute(owned, message, source_logits, target_logits):
            source, target = graph["_ops"].segment_attention(
                source_logits, message, graph["source_index"],
                target_logits, message, graph["target_index"], routes.local_atoms, graph)
            count = owned.shape[0]
            return layer.attention_node_update(owned, target[:count], source[:count], graph)

        return self.backend.local(compute, nodes, *routed, name=f"{self.prefix}.nodes")

    def attention_edges(self, layer, edges, rows, graph, directed2undirected):
        if self.line:
            if self.routes.fold_line_edges:
                return rows[0]
            return self.backend.local(
                lambda triplets, message: layer.attention_edges(triplets, message, graph),
                edges, rows[0], name=f"{self.prefix}.edges")
        if not self.routes.has_line:
            # Without a line graph the atom refinement reads the block input
            # bonds (Interaction_Block.forward): this update is dead.
            return edges
        value = self.backend.local(
            lambda bonds, message: layer.attention_edges(bonds, message, graph, directed2undirected),
            edges, rows[0], name=f"{self.prefix}.edges")
        self.routes.check(self.prefix, value)
        return value

    def refinement_rows(self, layer, nodes, edges, radial, graph,
                        directed2undirected, atom_feat):
        routes = self.routes
        if self.line:
            final, fold = self.final, routes.fold_line_edges

            def compute(bonds, triplets, basis, atoms):
                raw, weight, residual = layer.refinement_rows(bonds, triplets, basis, graph, None, atoms)
                partial = routes.reduce_target(raw, weight, graph, bonds.shape[0])
                sent = partial.index_select(0, routes.slots)
                packed = routes.pack(atoms, sent, self.index, "refinement")
                # The last block's q'' is dead (distributed E/F has no consumer).
                if final:
                    return partial, packed
                if fold:
                    # Folded (saved-tensor recompute): q'' instead of the pinned raw rows and residual.
                    return partial, packed, layer.refinement_edges(residual, raw, graph)
                return partial, packed, raw, residual

            return self.backend.local(compute, nodes, edges, radial, atom_feat,
                                      name=f"{self.prefix}.rows")
        ghosts, unpack = routes.ghosts(self, nodes, "refinement")
        smoothed = layer.use_smoothed_for_delta_edge

        def compute(owned, halo, bonds, basis):
            raw, weight, residual = layer.refinement_rows(
                torch.cat((owned, unpack(halo))), bonds, basis, graph, directed2undirected, None)
            if smoothed:
                weighted, edge_input = layer.refinement_payload(raw, weight)
                return weighted, edge_input, residual
            return raw, weight, residual

        return self.backend.local(compute, nodes, ghosts, edges, radial, name=f"{self.prefix}.rows")

    def refinement_route(self, rows, graph):
        if not self.line:
            return rows
        received = self.backend.exchange(rows[1], self.routes.binding(self.index, "refinement"),
                                         name=f"{self.prefix}.partials")
        self.routes.post_ghosts(self.index, "refinement", received)
        return rows[0], received

    def refinement_nodes(self, layer, nodes, routed, graph):
        routes = self.routes
        if self.line:
            def compute(bonds, partial, received):
                return layer.refinement_node_update(
                    bonds, routes.merge_refinement(partial, received, self.index), graph)

            value = self.backend.local(compute, nodes, *routed, name=f"{self.prefix}.nodes")
            routes.check(self.prefix, value)
            return value
        smoothed = layer.use_smoothed_for_delta_edge

        def compute(owned, values, weight=None):
            reduced = routes.reduce_target(values, weight, graph, routes.local_atoms)
            return layer.refinement_node_update(owned, reduced[:owned.shape[0]], graph)

        args = (nodes, routed[0]) if smoothed else (nodes, routed[0], routed[1])
        value = self.backend.local(compute, *args, name=f"{self.prefix}.nodes")
        routes.block_output(self.index, value)
        return value

    def refinement_edges(self, layer, edges, rows, graph,
                         directed2undirected, skip_edge_update):
        # Distributed E/F has no consumer for the last block's b and q.
        if self.final:
            return edges
        if self.line:
            if self.routes.fold_line_edges:
                return rows[2]
            return self.backend.local(
                lambda residual, raw: layer.refinement_edges(residual, raw, graph),
                rows[3], rows[2], name=f"{self.prefix}.edges")
        value = self.backend.local(
            lambda residual, edge_input: layer.refinement_edges(
                residual, edge_input, graph, directed2undirected),
            rows[2], rows[1] if layer.use_smoothed_for_delta_edge else rows[0],
            name=f"{self.prefix}.edges")
        self.routes.check(self.prefix, value)
        return value
