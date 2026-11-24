"""The exchange-strategy protocol: what the distributed runtime asks of a model's multi-GPU (P>1) code.

The runtime (this package) partitions the system, migrates atoms, builds the
halo shard, agrees capacities, runs the control plane and the exchange
schedule, and captures the whole step. What depends on the model (its local
graph and line slots, its exchanges, its forward and the scopes its kernels
need, its memory model) is an ``ExchangeStrategy`` object the model's adapter
provides: ``ShardDeclaration.custom_strategy()`` (``fastmd.adapters.protocol``),
imported when called. The runtime reaches a model's P>1 code only through that
object and imports no model package; the distributed evaluator rejects a model
whose adapter provides none, at any rank count (:func:`strategy_of`).

MatRIS's boundary-bond partial exchange is the one strategy today
(``BondPartialStrategy`` in MatRIS's multi-GPU package). The members below are
the ones the runtime calls. (The capability negotiation's P>1 decision still
reads ``LayerExchange.exchange_strategy()``, the exchange's name in the plan
record; MatRIS declares both.) The topology epoch's demand, layout install and
per-step derivation (``fastmd.distributed.epoch``) still follow that
strategy's padded layout; they become members when a second strategy needs
another layout. ``docs/distributed_inference.md`` describes the protocol.
"""
from __future__ import annotations

from typing import Any, ContextManager, Protocol, runtime_checkable


@runtime_checkable
class ExchangeStrategy(Protocol):
    """A model's P>1 evaluation: admission, preparation, scopes and forward (all ranks call alike).

    * ``name``: the exchange scheme (``ParallelConfig.exchange_scheme``, part of the rank contract).
    * ``admit_model(model, single_rank=...)``: raise ``ValueError``/``TypeError`` for a model the strategy
      cannot run; ``single_rank``: one CUDA rank delegates to the single-GPU engine, which skips the P>1
      topology's rules. ``admit_single_rank(model)``: the single-GPU engine's further requirements.
    * ``model_fingerprint(model)``: a digest of the weights and mathematical flags (rank contract).
    * ``contract(model)``: the strategy's own entries of the rank contract (JSON-like, compared across ranks).
    * ``prepare(model, shard, context, **options)``: the collective preparation of one shard (a
      ``fastmd.distributed.execution.PreparedExecution`` whose ``strategy`` is this object);
      ``pad(prepared, capacity, **options)``: its private padded view in the agreed capacity's layout.
    * ``evaluation_scope(recompute, admission)``: wraps every evaluation (``evaluate``, ``md_step``): the
      memory admission's decisions bound for the forwards. ``forward_scope(local_backend)``: the kernel
      configuration every forward, refresh and capture runs in (also ``config_summary``'s kernel policy).
      ``backward_scope(prepared)``: wraps the VJP of a forward over ``prepared``.
    * ``forward(model, context, prepared, stress=False)``: the E/F forward on the autograd backend, after the
      caller's collective boundary validation; its ``ExecutionResult.backward_scope`` carries
      ``backward_scope``. ``stress=True`` (passed only when on) adds the rank's zero ``[3, 3]`` strain leaf
      (``ExecutionResult.strain``) that deforms its local rows and cell, each row once.
    * ``memory_admission(model, **options)``: the evaluator's device-memory admission (generic admission and
      the model's ``MemoryPredictor``; the evaluator calls it only when the admission is on).
    * ``close()``: release what the strategy holds (graphs before hooks). A strategy that holds captured graphs
      or module hooks registers itself in the shutdown registry (``fastmd.distributed.shutdown_registry``), whose
      release calls it before the process groups go; MatRIS's holds nothing (one shared, stateless instance).
    """

    name: str

    def admit_model(self, model, *, single_rank: bool) -> None:
        ...

    def admit_single_rank(self, model) -> None:
        ...

    def model_fingerprint(self, model) -> str:
        ...

    def contract(self, model) -> dict:
        ...

    def prepare(self, model, shard, context, **options) -> Any:
        ...

    def pad(self, prepared, capacity, **options) -> Any:
        ...

    def evaluation_scope(self, recompute, admission) -> ContextManager:
        ...

    def forward_scope(self, local_backend: str) -> ContextManager:
        ...

    def backward_scope(self, prepared) -> ContextManager:
        ...

    def forward(self, model, context, prepared, *, stress: bool = False) -> Any:
        ...

    def memory_admission(self, model, **options) -> Any:
        ...

    def close(self) -> None:
        ...


def strategy_of(adapter, world_size: int) -> ExchangeStrategy:
    """The exchange strategy ``adapter`` provides (``ShardDeclaration.custom_strategy()``).

    A model without one is rejected with ``NegotiationError`` (a ``ValueError``): no other model runs on
    the distributed runtime yet. ``world_size`` only words the message.
    """
    from fastmd.adapters.protocol import NegotiationError, ShardDeclaration

    name = getattr(adapter, "name", type(adapter).__name__)
    strategy = adapter.custom_strategy() if isinstance(adapter, ShardDeclaration) else None
    if strategy is None:
        hint = ("; it declares a ShardSpec: run it with fastmd.distributed.ShardedEvaluator"
                if callable(getattr(adapter, "shard_spec", None)) else "")
        raise NegotiationError(f"model {name!r} provides no multi-GPU exchange strategy: the distributed runtime "
                               f"(P={world_size}) runs MatRIS only{hint}")
    if not isinstance(strategy, ExchangeStrategy):
        raise TypeError(f"model {name!r}: custom_strategy() returned {type(strategy).__name__}, not an "
                        "ExchangeStrategy")
    return strategy
