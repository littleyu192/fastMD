"""Model-neutral construction of the single-system distributed evaluator."""
from __future__ import annotations

import os


def DistributedEvaluator(model, cell, *, pbc=(True, True, True), config=None, parallel=None, context=None):
    """Construct the distributed evaluator declared by a FastMD model adapter.

    Pass a loaded FastMD adapter (for example, from ``load_adapter``) or an
    uncompiled MatRIS instance. For other models, load them with ``load_adapter``
    and pass the returned adapter, not a name, path or bare third-party module.

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
    from fastmd.models import DEFAULT, _row

    context = ExchangeContext() if context is None else context
    if not isinstance(context, ExchangeContext):
        raise TypeError("context must be an ExchangeContext")
    error, adapter, custom = None, None, False
    try:
        adapter = as_adapter(model)
        # Only MatRIS may use the default wrapper; compiled MatRIS keeps its evaluator's admission error.
        if adapter is not model and not _row(DEFAULT).function("claims")(getattr(model, "_orig_mod", model), None):
            given = (f"the model name/path {os.fspath(model)!r}" if isinstance(model, (str, os.PathLike))
                     else f"a {type(model).__name__} object")
            raise TypeError(
                f"DistributedEvaluator takes a FastMD model adapter or a MatRIS instance, not {given}: "
                "load the model first and pass its adapter, e.g. fastmd.adapters.load_adapter(name, "
                "device=...) or load_adapter(family, model_path=..., device=...)")
        custom = isinstance(adapter, ShardDeclaration) and adapter.custom_strategy() is not None
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
        error = str(exc)
    collective_check(context, error)
    kinds = context.all_gather_object(custom)
    if any(kind != kinds[0] for kind in kinds):
        raise ValueError("distributed evaluator strategies differ between ranks")
    evaluator = DistributedMatRISEvaluator if custom else ShardedEvaluator
    return evaluator(adapter, cell, pbc=pbc, config=config, parallel=parallel, context=context)
