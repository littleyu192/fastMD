"""Operator dispatcher, design layer L1 (docs/unified_engine_design.md §3.2, §4.3).

``OpDispatcher(KernelPolicy, ExecutionLayout)`` chooses, per model site, the
fused kernel or the generic torch lowering that the P=1 model selected inline
before step 2.3, in the same priority order and with the same recorder
records and run-time fallbacks. Selection has two parts (§4.3):

* static eligibility depends only on the policy, the layout and the fusion
  scope. The scope-independent part is computed once per dispatcher (plain
  attributes); dispatchers are cached per (policy fingerprint, layout
  fingerprint, strict_errors, backend) in a bounded process-wide LRU, so the
  cache key of a static decision is (site, policy, layout, fusion scope) and
  never contains a shape. The fusion scope itself is read live, last, where
  the inline gates read it (``fusion_scope_active()``), so checkpoint
  recomputation and dynamo guards see the runner's scope and the compiled
  program guards the same globals.
* tensor eligibility (row count >= ``min_rows``, device, rank, width,
  ``requires_grad``) is checked per call from host metadata only: no
  synchronization, no allocation.

At P>1 (step 2.4) the distributed runtime attaches a ``DistributedOpDispatcher``
built from the layout its preparer declares (``ExecutionLayout.from_bond_partial``:
route "exchange", interleaved pairs, private sink, prepared owner CSR, shared
value rows) with ``strict_errors=True`` (INV-4) and the local backend
as cache-key component; the same model sites then select the single-GPU fused
kernels on the distributed layouts. Step 2.6 removed the P>1 ``ops`` backends
(``NativeBackend`` / ``torch_reference`` modules) and the ``dispatch_engine``
rollback switch: this dispatcher is the only operator engine.

A kernel entry may still return None (a cold cache under capture, an entry
guard). At P=1 that keeps today's fallback and the decline is recorded by the
entry itself, exactly as before.

Deterministic mode (``fastmd.precision.deterministic_mode``): the atomic
weighted segment sum declines here (its forward adds with atomics); the
kernels whose VJP adds with atomics stay selected and order that sum
themselves (``triton_zero_skip_index.ordered_index_add_``), so their forward
values are unchanged.

``process_graphs`` attaches the dispatcher to the batch, atom and line graph
dicts under ``"_ops"``; the model sites select only through it (step 2.6
removed their inline P=1 branches), so a graph dict without one is an error.

torch.compile (``enable_compile``) traces these methods inside the compiled
layer modules, and G-S3 requires the compiled program (graph breaks, frames,
recompiles, resume functions) to stay identical. Hence:

* a method replaced an inline helper (removed in step 2.6) at the helper's
  call depth and kept its parameter names, body and gate order (the ``use_*``
  predicates mirror the former ``_use_fused_*`` helpers): with other names
  dynamo unified a size symbol of ``pair_expand`` differently when the
  recording pass split that frame, and
  Inductor failed to codegen. Where the former code called kernels inline and
  a callee can graph-break (the line envelope's compiled lowering), the site
  keeps the inline calls under the dispatcher's predicate (``use_line_envelope``);
* calls into modules whose activations graph-break (``GatedMLP``, ``MLP``, the
  indexed/merged projections) stay at the model site. For them the
  dispatcher supplies the policy (``use_gather_cat``, ``sorted_vjp``, the
  ``dispatcher=`` argument of ``GatedMLP.forward_with_residual``), not a
  wrapper frame. ``use_threebody_basis`` / ``use_threebody_linear`` do the
  same for the (uncompiled) three-body embedding.

Transition notes: the policy is ``KernelPolicy.from_globals()`` (step 2.2),
snapshotted when the graph is processed and cached per raw value of every
input ``from_globals`` reads; the model's module globals remain the source of
truth. Gates that live inside modules (``GatedMLP._finish_gate``'s gated
LayerNorm, the indexed-projection and merged-projection weight caches, the
compiled lowerings) keep their own guards; the dispatcher decides whether a
site tries them. This module must not import ``fastmd.models.matris`` at import time.
"""
from __future__ import annotations

from collections import OrderedDict
import os
import sys

import torch

from fastmd.engine import policy as _policy, recorder as _recorder
from fastmd.engine.layout import ExecutionLayout
from fastmd.engine.policy import KernelPolicy
from fastmd.precision import deterministic_mode

# Step 2.6 removed the legacy operator engine and its rollback switch; the
# former default "unified" is still accepted and has no effect.
if os.environ.get("MATRIS_DISPATCH_ENGINE", "unified") != "unified":
    raise ValueError("MATRIS_DISPATCH_ENGINE was removed in step 2.6: the OpDispatcher is the "
                     "only operator engine (no legacy rollback); unset MATRIS_DISPATCH_ENGINE")

_CACHE_SIZE = 256
_DISPATCHERS: OrderedDict[tuple, OpDispatcher] = OrderedDict()
_POLICIES: OrderedDict[tuple, KernelPolicy] = OrderedDict()
# Row thresholds per dispatcher key, read through this module global: dynamo
# specializes ints from global sources like the legacy module-global
# thresholds, while ints reached from a frame-local object (the graph dict's
# dispatcher) would become symbolic under torch.compile(dynamic=True). Never
# evicted: a dispatcher can outlive its LRU slot in a live graph dict.
_MIN_ROWS: dict[str, dict[str, int]] = {}
_MISSING = object()
# (name, bindings) of every option KernelPolicy.from_globals reads (model and kernel).
_POLICY_OPTIONS: tuple | None = None


def _policy_key() -> tuple | None:
    """Every input of ``KernelPolicy.from_globals()``, cheaply; None if unhashable.

    A loaded module's global, else what ``env_value`` would return: the active
    ``ResolvedConfig``'s kernel option (by value, so the fresh but equal
    configurations of ``plain_torch_config`` share an entry), else the
    environment.
    """
    global _POLICY_OPTIONS
    from fastmd.config import OPTION_REGISTRY, current_config

    if _POLICY_OPTIONS is None:
        _POLICY_OPTIONS = tuple((name, spec.bindings) for name, spec in OPTION_REGISTRY.items()
                                if spec.category in ("model", "kernel"))
    config = current_config()
    scoped = None if config is None else config.kernel_options
    values = []
    for name, bindings in _POLICY_OPTIONS:
        value = _MISSING
        for module_name, attribute in bindings:
            module = sys.modules.get(module_name)
            if module is not None:
                value = getattr(module, attribute, _MISSING)
                if value is not _MISSING:
                    break
        if value is _MISSING:
            value = (os.environ.get(name) if scoped is None or name not in scoped
                     else ("scoped", scoped[name]))
        values.append(value)
    block = sys.modules.get("fastmd.models.matris.interaction_block")
    functions = sys.modules.get("fastmd.models.matris.functions")
    values.append(getattr(block, "THRESHOLD_VALUE", None))
    values.append(getattr(functions, "_FUSED_ACTIVATIONS", None))
    key = tuple(values)
    try:
        hash(key)
    except TypeError:
        return None
    return key


def current_policy() -> KernelPolicy:
    """``KernelPolicy.from_globals()``, cached per raw input values."""
    key = _policy_key()
    if key is None:
        return KernelPolicy.from_globals()
    policy = _POLICIES.get(key)
    if policy is not None:
        _POLICIES.move_to_end(key)
        return policy
    policy = KernelPolicy.from_globals()
    _POLICIES[key] = policy
    if len(_POLICIES) > _CACHE_SIZE:
        _POLICIES.popitem(last=False)
    return policy


def dispatcher_for(policy: KernelPolicy, layout: ExecutionLayout, *,
                   strict_errors: bool = False, backend: str | None = None) -> OpDispatcher:
    """The cached dispatcher of (policy, layout, strict_errors, backend)."""
    key = (policy.fingerprint, layout.fingerprint, bool(strict_errors), backend)
    dispatcher = _DISPATCHERS.get(key)
    if dispatcher is not None:
        _DISPATCHERS.move_to_end(key)
        return dispatcher
    cls = DistributedOpDispatcher if layout.route == "exchange" else OpDispatcher
    dispatcher = cls(policy, layout, strict_errors=strict_errors, backend=backend)
    _DISPATCHERS[key] = dispatcher
    if len(_DISPATCHERS) > _CACHE_SIZE:
        _DISPATCHERS.popitem(last=False)
    return dispatcher


def attach_dispatcher(batch_graph: dict, plan) -> OpDispatcher:
    """Attach the P=1 dispatcher of ``plan`` to a processed batch.

    The policy is fixed for this graph dict (also for checkpoint recomputation
    of its layers); callers that change kernel switches process the graph
    again. The model sites have no other operator path (step 2.6): a policy
    or layout that cannot be built (e.g. a malformed ``MATRIS_*`` value) is an
    error here, not a silent fallback.
    """
    try:
        dispatcher = dispatcher_for(current_policy(), ExecutionLayout.from_plan(plan))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot build the operator dispatcher of this graph: {exc}") from exc
    batch_graph["_ops"] = dispatcher
    atom_graph, line_graph = batch_graph["atom_graph_dict"], batch_graph["line_graph_dict"]
    atom_graph["_ops"] = dispatcher
    line_graph["_ops"] = dispatcher
    if dispatcher.segment_sum_csr_enabled:
        # D6: the side CSRs exist before the compiled layers run (see prepare_segment_csr).
        dispatcher.prepare_segment_csr(atom_graph, atom_graph["num_segment"])
        if "target_index" in line_graph:
            dispatcher.prepare_segment_csr(line_graph, atom_graph["num_undirected"])
    return dispatcher


class OpDispatcher:
    """The P=1 operator selection of the shared model (design §3.2 L1).

    Static eligibility is held in plain attributes (dynamo treats them as
    constants of the guarded dispatcher object). A plain switch is one bit; an
    op with an EAGER/fusion-scope pair keeps ``*_enabled`` (the switch) and
    ``*_eager``, and its gate reads the live fusion scope last, after the
    tensor checks, exactly where the legacy gate read it, so the compiled
    program guards the same globals. Row thresholds (inclusive) live in
    ``_MIN_ROWS[self.key]`` (see there).
    """

    def __init__(self, policy: KernelPolicy, layout: ExecutionLayout, *,
                 strict_errors: bool = False, backend: str | None = None):
        self.policy, self.layout = policy, layout
        self.strict_errors, self.backend = bool(strict_errors), backend
        self.key = f"{policy.fingerprint}/{layout.fingerprint}/{int(self.strict_errors)}/{backend}"
        ops = policy.ops
        _MIN_ROWS[self.key] = {op: ops[op].min_rows for op in (
            "residual_add", "gated_ln_residual", "gather_cat", "segment_attention",
            "line_envelope", "threebody_basis", "threebody_linear")}
        # "interleaved" is the P>1 pair layout (2u, 2u+1); both are bijective with E == 2U.
        paired = layout.pair_layout in ("pair_index", "interleaved")
        # Plain switches (the same value in and out of the fusion scope).
        self.pair_expand_enabled = ops["directed_pair_expand"].enabled_eager and paired
        self.pair_average_enabled = ops["directed_pair_aggregate"].enabled_eager and paired
        self.residual_add_enabled = ops["residual_add"].enabled_eager
        self.segment_softmax_enabled = ops["segment_softmax"].enabled_eager
        self.gated_ln_residual_enabled = ops["gated_ln_residual"].enabled_eager
        # D6 (step 2.5a): P=1 layouts only, where ``segment_csr`` reads the P=1
        # CSR; a P>1 dispatcher enables it together with its owner-CSR override.
        self.segment_sum_csr_enabled = ops["segment_sum_csr"].enabled_eager and layout.route == "identity"
        # D8 (step 2.5b): indexed projections over parts of different widths (they need the
        # indexed projection itself: nothing else can take mixed-width parts).
        self.mixed_width_cat_enabled = (ops["mixed_width_cat"].enabled_eager
                                        and ops["indexed_projection"].enabled_eager)
        # EAGER-or-fusion-scope ops: enabled and (eager or fusion scope).
        self.weighted_segment_sum_eager = ops["weighted_segment_sum"].enabled_eager
        self.gather_cat_eager = ops["gather_cat"].enabled_eager
        self.segment_attention_enabled = ops["segment_attention"].enabled_fusion_scope
        self.segment_attention_eager = ops["segment_attention"].enabled_eager
        self.line_envelope_enabled = ops["line_envelope"].enabled_fusion_scope
        self.line_envelope_eager = ops["line_envelope"].enabled_eager
        self.threebody_basis_enabled = ops["threebody_basis"].enabled_fusion_scope
        self.threebody_basis_eager = ops["threebody_basis"].enabled_eager
        self.threebody_linear_enabled = ops["threebody_linear"].enabled_fusion_scope
        self.threebody_linear_eager = ops["threebody_linear"].enabled_eager
        attention = dict(ops["segment_attention"].tiles)
        self.reuse_csr = bool(attention["MATRIS_FUSED_SEGMENT_ATTENTION_REUSE_CSR"])
        self.paired_attention = bool(attention["MATRIS_FUSED_PAIRED_SEGMENT_ATTENTION"])
        sorted_target = bool(attention["MATRIS_FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR"])
        # Layout: certified contracts.
        self.identity_csr = {kind: sorted_target and kind in layout.identity_csr
                             for kind in ("atom", "line")}
        self.line_incidence_direct = layout.incidence in ("strong", "bounded")
        self.compiled_envelope = layout.compiled_envelope
        self.sorted_by = {"atom": "atom_indexed_affine" in layout.sorted_by,
                          "line": "line_indexed_affine" in layout.sorted_by}
        # Executor lowering policy handed to the indexed projections (None: the
        # module's own switches, P=1); the P>1 subclass supplies a strict one.
        self.projection_config = None

    def __repr__(self) -> str:
        return (f"OpDispatcher(policy={self.policy.fingerprint}, layout={self.layout.fingerprint}, "
                f"strict_errors={self.strict_errors}, backend={self.backend})")

    # ------------------------------------------------------------------ pairs

    def pair_expand(self, data, directed2undirected, graph):
        """``data[d2u]`` for undirected rows [U, D] -> directed rows [2U, D]."""
        num_undirected = graph["num_undirected"]
        if (
            self.pair_expand_enabled
            and data.is_cuda
            and directed2undirected.is_cuda
            and data.dim() == 2
            and data.shape[0] == num_undirected
            and directed2undirected.numel() == 2 * num_undirected
        ):
            from fastmd.models.matris.op.triton_directed_pair_aggregate import undirected_pair_expand

            return undirected_pair_expand(
                data,
                _directed_pair_index(graph, directed2undirected),
                directed2undirected.numel(),
            )
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("pair_expand", "torch_index_select", {
                "pairs": num_undirected, "width": data.shape[-1], "rows": directed2undirected.numel(),
            })
        return torch.index_select(data, 0, directed2undirected)

    def pair_average(self, data, directed2undirected, graph):
        """Mean of the two directed rows of every pair: [2U, D] -> [U, D]."""
        num_undirected = graph["num_undirected"]
        if (
            self.pair_average_enabled
            and data.is_cuda
            and directed2undirected.is_cuda
            and data.dim() == 2
            and directed2undirected.numel() == 2 * num_undirected
        ):
            from fastmd.models.matris.op.triton_directed_pair_aggregate import directed_pair_average

            return directed_pair_average(data, _directed_pair_index(graph, directed2undirected))
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("pair_average", "torch_aggregate", {
                "pairs": num_undirected, "width": data.shape[-1],
            })
        from fastmd.engine.kernels.segment import aggregate

        return aggregate(
            data=data,
            segment=directed2undirected,
            bin_count=None,
            average=True,
            num_segment=num_undirected,
        )

    # --------------------------------------------------------------- residual

    def residual(self, delta, residual, weight):
        """``delta + weight * residual``: Triton residual add, compiled lowering, torch."""
        if (
            self.residual_add_enabled
            and delta.is_cuda
            and residual.is_cuda
            and weight.is_cuda
            and delta.dim() == 2
            and residual.shape == delta.shape
            and delta.shape[0] >= _MIN_ROWS[self.key]["residual_add"]
            and delta.shape[1] == weight.numel()
            and delta.dtype == residual.dtype == weight.dtype
            and not weight.requires_grad
        ):
            from fastmd.models.matris.op.triton_residual_add import residual_add

            return residual_add(delta, residual, weight)
        from fastmd.models.matris.op.compiled_lowerings import compiled_residual_add

        compiled_out = compiled_residual_add(delta, residual, weight)
        if compiled_out is not None:
            return compiled_out
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("residual_add", "torch", {"rows": delta.shape[0], "width": delta.shape[-1]})
        return delta + weight * residual

    def gated_ln_residual(self) -> tuple[bool, int]:
        """(enabled, min_rows) of ``GatedMLP.forward_with_residual``'s fused gated-LN residual."""
        return self.gated_ln_residual_enabled, _MIN_ROWS[self.key]["gated_ln_residual"]

    # ---------------------------------------------------------- segment ops

    def use_weighted_segment_sum(self, weight, value) -> bool:
        if not (weight.is_cuda and value.is_cuda and weight.shape == value.shape):
            return False
        if deterministic_mode():  # its forward adds with atomics; aggregate is ordered then
            return False
        return self.weighted_segment_sum_eager or _policy.fusion_scope_active()

    def use_csr_segment_sum(self, value, segment, weight) -> bool:
        """Tensor eligibility of the deterministic CSR segment sum (host metadata only;
        the policy switch ``segment_sum_csr_enabled`` is checked by the caller)."""
        if not (value.is_cuda and segment.is_cuda and value.dim() == 2):
            return False
        if weight is not None and not (weight.is_cuda and weight.shape == value.shape
                                       and weight.dtype == value.dtype):
            return False
        return segment.dim() == 1 and segment.numel() == value.shape[0]

    def prepare_segment_csr(self, graph, num_segment):
        """Build the CSR of both sides of ``graph`` for ``segment_sum_graph``, outside
        the compiled layers (``attach_dispatcher``), so that compiled code only reads
        existing keys: no graph-dict insertion or graph-kind read in a traced frame.

        P=1: the segment-attention CSR (same cache keys, shared with the fused
        attention), the identity CSR for a certified sorted target, no skipped
        segment; stored ready as ``graph[f"_{side}_segment_sum_csr"]``.
        """
        kind = "atom" if graph.get("_topology_graph_kind") == "atom" else "line"
        for side in ("source", "target"):
            identity = (side == "target" and self.identity_csr[kind]
                        and (f"_{side}_segment_attention_counts" in graph
                             or f"_{side}_segment_attention_offsets" in graph))
            perm, off, identity = _segment_attention_csr_metadata(
                graph, side, graph[f"{side}_index"], num_segment, identity)
            graph[f"_{side}_segment_sum_csr"] = (perm, off, identity, -1)

    def segment_csr(self, graph, side, segment, num_segment):
        """(perm, int64 offsets, identity, skip_segment) of ``side`` (``"source"`` /
        ``"target"``) of ``graph``, whose rows ``segment`` indexes
        (``graph[f"{side}_index"]``).

        P=1: the tuple ``prepare_segment_csr`` stored (built here if missing).
        ``DistributedOpDispatcher`` overrides this with the prepared owner CSR
        (the private sink as skipped segment) and may return None (no CSR:
        ``segment_sum_graph`` falls back to ``segment_sum``).
        """
        csr = graph.get(f"_{side}_segment_sum_csr")
        if csr is None:
            self.prepare_segment_csr(graph, num_segment)
            csr = graph[f"_{side}_segment_sum_csr"]
        return csr

    def segment_sum_graph(self, value, segment, num_segment, graph, side, *, weight=None,
                          bin_count=None):
        """``segment_sum`` over ``side`` (``"source"`` / ``"target"``) of ``graph``:
        the deterministic CSR segment sum (step 2.5a, D6) on the side's CSR when
        the ``segment_sum_csr`` switch is on and the tensors are eligible, else
        ``segment_sum``.

        Call sites take this entry only when ``segment_sum_csr_enabled`` (a
        constant of the dispatcher): with the switch off they call
        ``segment_sum`` exactly as before (same compiled programs), and a
        subclass that overrides ``segment_sum`` keeps its signature.
        """
        if self.segment_sum_csr_enabled and self.use_csr_segment_sum(value, segment, weight):
            csr = self.segment_csr(graph, side, segment, num_segment)
            if csr is not None:
                from fastmd.models.matris.op.triton_csr_segment_sum import csr_segment_sum

                perm, off, identity, skip_segment = csr
                return csr_segment_sum(value, segment, num_segment, perm, off, identity,
                                       weight=weight, skip_segment=skip_segment)
        return self.segment_sum(value, segment, num_segment, weight=weight, bin_count=bin_count)

    def segment_sum(self, value, segment, num_segment, *, weight=None, bin_count=None):
        """``sum_{i in segment s} [weight_i *] value_i`` for ``num_segment`` segments.

        Weighted: the Triton weighted segment sum (atomic) when eligible, else the
        generic aggregate of ``weight * value``. Names mirror the former helper
        ``_weighted_segment_sum_or_aggregate`` (see ``pair_expand``). The CSR
        segment sum (D6) is ``segment_sum_graph``.
        """
        from fastmd.engine.kernels.segment import aggregate

        if weight is None:
            rec = _recorder._ACTIVE
            if rec is not None:
                rec.record("segment_sum", "torch_aggregate", {
                    "rows": value.shape[0], "width": value.shape[-1], "segments": num_segment,
                })
            return aggregate(value, segment, bin_count, average=False, num_segment=num_segment)
        if self.use_weighted_segment_sum(weight, value):
            from fastmd.models.matris.op.triton_weighted_segment_sum import weighted_segment_sum

            return weighted_segment_sum(weight, value, segment, num_segment)
        rec = _recorder._ACTIVE
        if rec is not None:
            rec.record("weighted_segment_sum", "torch_aggregate", {
                "rows": value.shape[0], "width": value.shape[-1], "segments": num_segment,
            })
        from fastmd.engine import saved_recompute
        data = weight * value
        scope = saved_recompute.active()
        if scope is not None:
            scope.register_product(data, weight, value)
        return aggregate(
            data=data,
            segment=segment,
            bin_count=bin_count,
            average=False,
            num_segment=num_segment,
        )

    def use_segment_attention(self, logits, value) -> bool:
        if not self.segment_attention_enabled:
            return False
        if not (logits.is_cuda and value.is_cuda and logits.shape == value.shape):
            return False
        if logits.dim() != 2 or value.dim() != 2:
            return False
        if logits.shape[0] < _MIN_ROWS[self.key]["segment_attention"]:
            return False
        return self.segment_attention_eager or _policy.fusion_scope_active()

    def segment_attention(self, source_alpha_0, source_rows, source_node_index,
                          target_alpha_0, target_rows, target_node_index, num_segment, graph):
        """Channel-wise segment softmax attention of both sides of one graph.

        Fused (CSR) when eligible, paired when both sides share their value
        rows (tensor identity); else segment softmax plus weighted sums.
        Names mirror the former inline ``attention_nodes`` body (see ``pair_expand``).
        """
        if self.use_segment_attention(source_alpha_0, source_rows):
            from fastmd.models.matris.op.triton_segment_attention import segment_attention, paired_segment_attention

            if self.reuse_csr:
                target_identity_csr = self.identity_csr[
                    "atom" if graph.get("_topology_graph_kind") == "atom" else "line"]
                source_perm, source_off, source_identity = _segment_attention_csr_metadata(
                    graph,
                    "source",
                    source_node_index,
                    num_segment,
                    False,
                )
                target_perm, target_off, target_identity = _segment_attention_csr_metadata(
                    graph,
                    "target",
                    target_node_index,
                    num_segment,
                    target_identity_csr,
                )
            else:
                source_perm = source_off = target_perm = target_off = None
                source_identity = target_identity = False

            if self.paired_attention and source_rows is target_rows:
                return paired_segment_attention(
                    source_alpha_0,
                    target_alpha_0,
                    source_rows,
                    source_node_index,
                    target_node_index,
                    num_segment,
                    source_perm,
                    source_off,
                    target_perm,
                    target_off,
                    source_identity,
                    target_identity,
                )
            attn_source_feat = segment_attention(
                source_alpha_0,
                source_rows,
                source_node_index,
                num_segment,
                source_perm,
                source_off,
                source_identity,
            )
            attn_target_feat = segment_attention(
                target_alpha_0,
                target_rows,
                target_node_index,
                num_segment,
                target_perm,
                target_off,
                target_identity,
            )
            return attn_source_feat, attn_target_feat
        from fastmd.engine.kernels.segment import Dimwise_softmax

        source_alpha = Dimwise_softmax(source_alpha_0, source_node_index, num_segment,
                                       fused=self.segment_softmax_enabled)
        target_alpha = Dimwise_softmax(target_alpha_0, target_node_index, num_segment,
                                       fused=self.segment_softmax_enabled)
        # C10: compiled layers keep their traced segment sums (is_compiling first: no guard).
        if not torch.compiler.is_compiling() and self.segment_sum_csr_enabled:
            return (
                self.segment_sum_graph(source_rows, source_node_index, num_segment, graph, "source",
                                       weight=source_alpha, bin_count=graph['source_bincount']),
                self.segment_sum_graph(target_rows, target_node_index, num_segment, graph, "target",
                                       weight=target_alpha, bin_count=graph['target_bincount']),
            )
        attn_source_feat = self.segment_sum(
            source_rows, source_node_index, num_segment, weight=source_alpha,
            bin_count=graph['source_bincount'],
        )
        attn_target_feat = self.segment_sum(
            target_rows, target_node_index, num_segment, weight=target_alpha,
            bin_count=graph['target_bincount'],
        )
        return attn_source_feat, attn_target_feat

    # ------------------------------------------------------------ gathers

    def gather(self, value, indices):
        """``value[indices]`` (rows)."""
        return torch.index_select(value, 0, indices)

    def use_gather_cat(self, aligned, gathered) -> bool:
        """Whether the fused 3-part gather-cat (and indexed projection path) applies."""
        if not (aligned.is_cuda and gathered.is_cuda and aligned.dim() == 2 and gathered.dim() == 2):
            return False
        if aligned.shape[1] != gathered.shape[1] or aligned.shape[0] < _MIN_ROWS[self.key]["gather_cat"]:
            return False
        return self.gather_cat_eager or _policy.fusion_scope_active()

    def use_gather_cat4(self, aligned, gathered_a, gathered_b) -> bool:
        """Whether the fused 4-part gather-cat (and indexed projection path) applies.

        Parts of different widths (D8, ``mixed_width_cat``) qualify the indexed
        projection only (the aligned gather-cat kernel needs equal widths), under
        the same row threshold and eager/fusion-scope rule."""
        if (self.use_gather_cat(aligned, gathered_a)
                and gathered_b.is_cuda and gathered_b.dim() == 2 and aligned.shape[1] == gathered_b.shape[1]):
            return True
        if not self.mixed_width_cat_enabled or torch.compiler.is_compiling():  # see functions._mixed_width_allowed
            return False
        if not (aligned.is_cuda and gathered_a.is_cuda and gathered_b.is_cuda
                and aligned.dim() == 2 and gathered_a.dim() == 2 and gathered_b.dim() == 2):
            return False
        if aligned.shape[1] == gathered_a.shape[1] == gathered_b.shape[1]:
            return False
        if aligned.shape[0] < _MIN_ROWS[self.key]["gather_cat"]:
            return False
        return self.gather_cat_eager or _policy.fusion_scope_active()

    def gather_cat(self, aligned, gathered, indices):
        """Fused ``cat([aligned, gathered_k[indices_k]...], 1)`` (3 or 4 parts)."""
        if len(gathered) == 1:
            from fastmd.models.matris.op.triton_gather_cat import aligned_gather_cat3

            return aligned_gather_cat3(aligned, gathered[0], indices[0], indices[1])
        from fastmd.models.matris.op.triton_gather_cat import aligned_gather_cat4

        return aligned_gather_cat4(aligned, gathered[0], indices[0], gathered[1],
                                   indices[1], indices[2])

    def sorted_vjp(self, relation: str) -> bool:
        """Whether the indexed projection over ``relation`` (``"atom"``/``"line"``)
        may assume sorted index rows in its VJP (layout ``sorted_by``, DSP-07)."""
        return self.sorted_by[relation]

    def use_line_envelope(self, base, source, target) -> bool:
        if not self.line_envelope_enabled:
            return False
        if not (base.is_cuda and source.is_cuda and target.is_cuda):
            return False
        if base.dim() != 2 or source.numel() != target.numel():
            return False
        if source.numel() < _MIN_ROWS[self.key]["line_envelope"]:
            return False
        return self.line_envelope_eager or _policy.fusion_scope_active()

    # ---------------------------------------------------------- three-body

    def use_threebody_linear(self, basis) -> bool:
        """Whether the combined frozen two-linear projection may be used."""
        if not self.threebody_linear_enabled:
            return False
        if not (basis.is_cuda and basis.dim() == 2):
            return False
        if basis.shape[0] < _MIN_ROWS[self.key]["threebody_linear"]:
            return False
        return self.threebody_linear_eager or _policy.fusion_scope_active()

    def use_threebody_basis(self, unit_vec, target_index, source_index, freqs) -> bool:
        """Whether the indexed Fourier basis kernel replaces the torch basis."""
        if not self.threebody_basis_enabled:
            return False
        if not (unit_vec.is_cuda and target_index.is_cuda and source_index.is_cuda and freqs.is_cuda):
            return False
        if unit_vec.dim() != 2 or unit_vec.shape[1] != 3:
            return False
        if (target_index.numel() != source_index.numel()
                or target_index.numel() < _MIN_ROWS[self.key]["threebody_basis"]):
            return False
        if freqs.requires_grad or freqs.numel() != 3:
            return False
        return self.threebody_basis_eager or _policy.fusion_scope_active()


class DistributedOpDispatcher(OpDispatcher):
    """The P>1 selection (route "exchange", design §4.4 "P>1 需要"): the P=1
    rules on a distributed layout, plus what that layout declares.

    * ``strict_errors`` (INV-4): an admitted fused implementation that fails
      raises instead of silently returning None; the indexed projections get
      a strict ``projection_config`` (the policy's backend and row threshold).
    * pairs: interleaved (E == 2U with padding) qualifies the pair kernels.
    * private sink (padded layout): row gathers and unweighted row sums use
      the zero-skipping kernels, whose atomics omit the zero addends of the
      padding rows that all target the one sink (deterministic mode: ordered
      sums); the atomic weighted segment sum is not selected there.
    * the deterministic CSR segment sum (step 2.5a D6, policy switch
      ``segment_sum_csr``) reduces the refinement rows over the owner CSR of
      the local rows (``segment_csr``), skipping the private sink.
    * segment attention reads the prepared owner CSR (``_owner_{side}_csr``,
      int64, with the private sink as skipped segment) and never sorts in the
      model body; paired when the layout shares the kind's value rows and
      both sides are the same tensor.
    """

    def __init__(self, policy: KernelPolicy, layout: ExecutionLayout, *,
                 strict_errors: bool = True, backend: str | None = None):
        super().__init__(policy, layout, strict_errors=strict_errors, backend=backend)
        self.private_sink = layout.sink == "private_single"
        self.owner_csr = {kind: kind in layout.owner_csr for kind in ("atom", "line")}
        self.shared_value_rows = {kind: kind in layout.shared_value_rows for kind in ("atom", "line")}
        # D6: the owner reductions have the prepared owner CSR (segment_csr).
        self.segment_sum_csr_enabled = policy.ops["segment_sum_csr"].enabled_eager and bool(layout.owner_csr)
        projection = policy.ops["indexed_projection"]
        if self.strict_errors and projection.enabled_eager:
            from fastmd.engine.kernels.segment import IndexedProjectionConfig

            self.projection_config = IndexedProjectionConfig(
                enabled=True, backend=projection.backend, min_rows=projection.min_rows,
                strict_errors=True, mixed_width=self.mixed_width_cat_enabled)

    def gather(self, value, indices):
        if self.private_sink:
            from fastmd.engine.kernels.triton_zero_skip_index import (
                zero_skip_index_select, zero_skip_index_select_reason,
            )

            if zero_skip_index_select_reason(value, indices) is None:
                return zero_skip_index_select(value, indices)
        return torch.index_select(value, 0, indices)

    def use_weighted_segment_sum(self, weight, value) -> bool:
        if self.private_sink:
            return False
        return super().use_weighted_segment_sum(weight, value)

    def segment_sum(self, value, segment, num_segment, *, weight=None, bin_count=None):
        if weight is None and self.private_sink:
            from fastmd.engine.kernels.triton_zero_skip_index import (
                zero_skip_index_sum, zero_skip_index_sum_reason,
            )

            if zero_skip_index_sum_reason(value, segment, num_segment) is None:
                return zero_skip_index_sum(value, segment, num_segment)
        return super().segment_sum(value, segment, num_segment, weight=weight, bin_count=bin_count)

    def segment_csr(self, graph, side, segment, num_segment):
        """The owner CSR of this rank's local rows
        (``graph["_owner_{side}_reduce_csr"]`` = (perm, int64 offsets, skip),
        the private sink as skipped segment), as (perm, off, False, skip); None
        without one.
        """
        csr = graph.get(f"_owner_{side}_reduce_csr")
        if csr is None or csr[1].numel() != num_segment + 1:
            return None
        perm, off, skip = csr
        return perm, off, False, skip

    def segment_attention(self, source_alpha_0, source_rows, source_node_index,
                          target_alpha_0, target_rows, target_node_index, num_segment, graph,
                          *, return_lse=False):
        """``return_lse`` (the A2 bond-partial line attention) also returns each
        side's per-segment log-sum-exp of the logits (-inf where empty):
        ``(source, target, source_lse, target_lse)``, from the fused kernels'
        own forward pass when they run."""
        kind = "atom" if graph.get("_topology_graph_kind") == "atom" else "line"
        source_csr = graph.get("_owner_source_csr") if self.owner_csr[kind] else None
        target_csr = graph.get("_owner_target_csr") if self.owner_csr[kind] else None
        if (source_csr is not None and target_csr is not None
                and self.use_segment_attention(source_alpha_0, source_rows)):
            from fastmd.models.matris.op.triton_segment_attention import (
                paired_segment_attention, segment_attention,
            )

            source_perm, source_off, source_skip = source_csr
            target_perm, target_off, target_skip = target_csr
            if (self.paired_attention and self.shared_value_rows[kind]
                    and source_rows is target_rows and source_skip == target_skip):
                return paired_segment_attention(
                    source_alpha_0, target_alpha_0, source_rows, source_node_index,
                    target_node_index, num_segment, source_perm, source_off,
                    target_perm, target_off, False, False, skip_segment=source_skip,
                    **({"return_lse": True} if return_lse else {}),
                )
            if return_lse:
                source, source_lse = segment_attention(
                    source_alpha_0, source_rows, source_node_index, num_segment,
                    source_perm, source_off, False, skip_segment=source_skip, return_lse=True)
                target, target_lse = segment_attention(
                    target_alpha_0, target_rows, target_node_index, num_segment,
                    target_perm, target_off, False, skip_segment=target_skip, return_lse=True)
                return source, target, source_lse, target_lse
            return (
                segment_attention(source_alpha_0, source_rows, source_node_index, num_segment,
                                  source_perm, source_off, False, skip_segment=source_skip),
                segment_attention(target_alpha_0, target_rows, target_node_index, num_segment,
                                  target_perm, target_off, False, skip_segment=target_skip),
            )
        from fastmd.engine.kernels.segment import Dimwise_softmax

        source_alpha = Dimwise_softmax(source_alpha_0, source_node_index, num_segment,
                                       fused=self.segment_softmax_enabled)
        target_alpha = Dimwise_softmax(target_alpha_0, target_node_index, num_segment,
                                       fused=self.segment_softmax_enabled)
        result = (
            self.segment_sum(source_rows, source_node_index, num_segment, weight=source_alpha,
                             bin_count=graph["source_bincount"]),
            self.segment_sum(target_rows, target_node_index, num_segment, weight=target_alpha,
                             bin_count=graph["target_bincount"]),
        )
        if not return_lse:
            return result
        from fastmd.engine.kernels.segment import segment_logsumexp

        return (*result, segment_logsumexp(source_alpha_0, source_node_index, num_segment),
                segment_logsumexp(target_alpha_0, target_node_index, num_segment))


def _segment_attention_csr_metadata(graph, key_prefix, segment, num_segment, identity_perm):
    """(perm, offsets, identity) of one attention side, cached in the graph.

    Same keys, values and names as the former ``interaction_block`` helper; the
    identity decision (policy switch and certified layout) is the caller's.
    """
    perm_key = f"_{key_prefix}_segment_attention_perm"
    off_key = f"_{key_prefix}_segment_attention_offsets"
    count_key = f"_{key_prefix}_segment_attention_counts"
    if identity_perm:
        off = graph.get(off_key)
        if off is None:
            counts = graph[count_key]
            # int64: identity-CSR rows index stride-128 merged alpha views.
            off = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int64)
            graph[off_key] = off
        return None, off, True

    perm = graph.get(perm_key)
    off = graph.get(off_key)
    if perm is None or off is None:
        from fastmd.models.matris.op.triton_segment_attention import segment_attention_csr_metadata

        perm, off = segment_attention_csr_metadata(segment, num_segment)
        graph[perm_key] = perm
        graph[off_key] = off
    return perm, off, False


def _directed_pair_index(graph: dict, directed2undirected):
    """[U, 2] directed rows of every pair, cached in the graph.

    Same body and names as the former ``interaction_block._directed_pair_index``: under
    torch.compile(dynamic=True) a different form of this frame changed which
    symbol of U survives unification and tripped an Inductor codegen assertion.
    """
    pair_key = "_directed2undirected_pair_index"
    pair_index = graph.get(pair_key)
    if pair_index is None:
        num_undirected = graph["num_undirected"]
        pair_index = torch.argsort(directed2undirected).reshape(num_undirected, 2)
        graph[pair_key] = pair_index.contiguous()
    return graph[pair_key]
