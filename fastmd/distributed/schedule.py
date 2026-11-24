"""The prevalidated exchange schedule of one P>1 evaluation (design INV-3).

``PrevalidatedSchedule.prepare`` binds every exchange stage of the model
forward, in forward order, to its ``_PreparedExchange`` (routing indices,
payload shape and lowering) and admits the whole schedule collectively: stage
names, order, shapes, dtypes and every rank pair's split sizes. The autograd
backend (``fastmd.engine.grad``) walks it forward and the backward walks it
back in exact reverse; the graph runner snapshots and pads its bindings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .exchange import ExchangeContext, _PreparedExchange, _key


@dataclass(frozen=True)
class PrevalidatedSchedule:
    """Internal immutable admission contract for async eager/capture execution.

    ``prepare`` and ``validate_boundary`` are collective and MUST run outside
    CUDA capture. All hot-path functions must be the already validated model
    schedule, not arbitrary user callbacks. Unexpected runtime failures are
    fatal: launchers must enforce finite process-group timeouts and terminate
    the entire worker group. There is no rank-local eager fallback.

    Entries are ``(stage_name, plan, trailing_shape, dtype, device)`` in forward
    order. The reverse schedule is mechanically identical in reverse order.
    ``signature`` identifies the complete static model/local-stage contract;
    per-rank row counts belong to the plans, not that common signature.
    """

    context: ExchangeContext
    exchanges: tuple[tuple[str, _PreparedExchange], ...]
    signature: object
    stats: dict = field(default_factory=dict)

    @classmethod
    def prepare(cls, context, exchanges, *, signature, self_rows="local"):
        """``self_rows`` selects the payload lowering of every binding
        ("local": self rows bypass the data backend; "nccl": the original).
        It is part of the collectively admitted common contract."""
        started = perf_counter()
        error = None
        prepared = []
        descriptor_rows = []
        unique = {}
        route_ids = {}
        binding_stats = {}
        common = None
        try:
            common = _key((signature, self_rows), {})
            entries = tuple(exchanges)
            for name, plan, shape, dtype, device in entries:
                if not isinstance(name, str) or plan.context is not context:
                    raise ValueError("prepared stage must have a name and the same context")
            bindings = _PreparedExchange.build_many(
                [(plan, shape, dtype, device) for _, plan, shape, dtype, device in entries],
                stats=binding_stats, self_rows=self_rows)
            for (name, plan, _, _, _), binding in zip(entries, bindings):
                key = (id(plan), str(binding.device))
                route_ids.setdefault(key, len(route_ids))
                unique.setdefault(key, binding)
                prepared.append((name, binding))
                descriptor_rows.append((name, binding.descriptor()[:5], route_ids[key]))
            # Width/dtype/stage order are checked for every exchange, but a
            # route reused by many model blocks has only one peer-split table.
            # First-use route IDs also admit the sharing pattern collectively.
            descriptors = tuple(descriptor_rows)
            routes = tuple((binding.plan.name, binding.plan.sequence, binding.device.type,
                            tuple(binding.plan.send_counts), tuple(binding.plan.receive_counts))
                           for binding in unique.values())
        except Exception as exc:
            error = f"prepared schedule: {type(exc).__name__}: {exc}"
            descriptors = ()
            routes = ()
        built = perf_counter()
        gathered = context._collective((error, common, descriptors, routes), "execution.prepare")
        admitted = perf_counter()
        context._raise_errors([entry[0] for entry in gathered])
        if any(entry[1] != gathered[0][1] for entry in gathered):
            raise RuntimeError("prepared model/local-stage contract differs across ranks")
        protocols = [entry[2] for entry in gathered]
        if any(protocol != protocols[0] for protocol in protocols):
            raise RuntimeError("prepared exchange stage/order/shape/dtype differs across ranks")
        route_protocols = [tuple(route[:3] for route in entry[3]) for entry in gathered]
        if any(protocol != route_protocols[0] for protocol in route_protocols):
            raise RuntimeError("prepared unique route/order/device differs across ranks")
        for route in range(len(unique)):
            for sender in range(context.world_size):
                for receiver in range(context.world_size):
                    sent = gathered[sender][3][route][3][receiver]
                    received = gathered[receiver][3][route][4][sender]
                    if sent != received:
                        raise RuntimeError("prepared peer split sizes disagree")
        completed = perf_counter()
        stats = {"prepare_ms": (completed - started) * 1000,
                 **binding_stats,
                 "binding_ms": (built - started) * 1000,
                 "admission_collective_ms": (admitted - built) * 1000,
                 "protocol_validation_ms": (completed - admitted) * 1000,
                 "peer_split_checks": len(unique) * context.world_size ** 2,
                 "unique_plans": len(unique), "exchange_stages": len(prepared),
                 "self_rows": self_rows,
                 "routing_index_bytes": sum(t.numel() * t.element_size()
                                            for binding in unique.values() for t in binding.indices)}
        return cls(context, tuple(prepared), common, stats)

    def boundary_error(self):
        """Local integrity guard, for inclusion in the caller's collective gate.

        Returning an error is not permission to launch: callers must propagate
        it collectively before any payload. Independent capture experiments use
        ``validate_boundary`` below, which owns that collective itself.
        """
        started = perf_counter()
        error = None
        try:
            seen = set()
            for _, binding in self.exchanges:
                if id(binding.indices) not in seen:
                    binding.validate_integrity()
                    seen.add(id(binding.indices))
        except Exception as exc:
            error = f"prepared boundary: {type(exc).__name__}: {exc}"
        self.stats["boundary_integrity_ms"] = (perf_counter() - started) * 1000
        return error

    def validate_boundary(self):
        """One collective integrity guard before any run/warmup/capture launch."""
        error = self.boundary_error()
        started = perf_counter()
        self.context.collective_check(error)
        self.stats["boundary_collective_ms"] = (perf_counter() - started) * 1000
