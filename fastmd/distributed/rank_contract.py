"""The rank contract: every rank of a distributed evaluator agrees one plan before any payload.

At construction each rank resolves its plan locally (the adapter and its
exchange strategy, the model fingerprint, the geometry, the profile and
backend, execution, recompute, topology, precision, capture policy, ...) and
the ranks compare the resolved entries in one gathered round. Equal entries
mean every rank derives the same schedule, capacities and kernels from then
on; any difference fails the construction on every rank, naming the entries.
"""
from __future__ import annotations

# The rejection's historical wording (callers match it); the differing entries follow in parentheses.
MISMATCH = "model, cell, precision or configuration differs between ranks"
_MISSING = object()


def differing_entries(contracts: list[dict]) -> list[str]:
    """Names of the entries whose values are not equal on every rank (first rank's order, then the others')."""
    first = contracts[0]
    names = dict.fromkeys([*first, *(name for contract in contracts[1:] for name in contract)])
    return [name for name in names
            if any(contract.get(name, _MISSING) != first.get(name, _MISSING) for contract in contracts[1:])]


def agree_rank_contract(context, contract: dict) -> None:
    """Gather every rank's ``contract`` and reject any difference on every rank (collective)."""
    from .evaluator import _objects, collective_check

    contracts = _objects(context, contract)
    error = None
    if not all(other == contracts[0] for other in contracts):
        differing = differing_entries(contracts)
        error = MISMATCH + (f" ({', '.join(differing)})" if differing else "")
    collective_check(context, error)
