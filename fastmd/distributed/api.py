"""Model-neutral construction of the single-system distributed evaluator."""
from __future__ import annotations


def DistributedEvaluator(model, cell, *, pbc=(True, True, True), config=None, parallel=None, context=None):
    """Construct the distributed evaluator declared by a FastMD model adapter.

    MatRIS uses its boundary-bond strategy; CHGNet uses the generic line-partial
    evaluator, and MACE, SevenNet and ORB use the generic node-halo evaluator.
    The returned evaluator exposes ``distribute``, ``evaluate``, ``gather``,
    ``distribute_restart`` and ``close`` and works with
    ``GPUResidentMolecularDynamics.from_evaluator``.
    """
    from fastmd.adapters import as_adapter
    from fastmd.adapters.protocol import ShardDeclaration
    from fastmd.distributed.evaluator import DistributedMatRISEvaluator, collective_check
    from fastmd.distributed.exchange import ExchangeContext
    from fastmd.distributed.sharded_evaluator import ShardedEvaluator

    context = ExchangeContext() if context is None else context
    if not isinstance(context, ExchangeContext):
        raise TypeError("context must be an ExchangeContext")
    error, adapter, custom = None, None, False
    try:
        adapter = as_adapter(model)
        custom = isinstance(adapter, ShardDeclaration) and adapter.custom_strategy() is not None
    except (ValueError, TypeError, RuntimeError) as exc:
        error = str(exc)
    collective_check(context, error)
    kinds = context.all_gather_object(custom)
    if any(kind != kinds[0] for kind in kinds):
        raise ValueError("distributed evaluator strategies differ between ranks")
    evaluator = DistributedMatRISEvaluator if custom else ShardedEvaluator
    return evaluator(adapter, cell, pbc=pbc, config=config, parallel=parallel, context=context)
