"""Which modules of a shard model may run on the owned slots only: the candidates of its ghost-row trimming.

At P>1 on the node-halo exchange a rank's rows are its owned slots, then its
ghost rows. A ghost row receives no complete message sum, and what a layer
computes on it after the sum is never read: the next exchange point replaces
it with its owner's row, and the energy sums the owned rows. The modules that
run there may run on the owned slots only (``ShardSpec.owned_slot_inputs``,
``zero_ghost_outputs``; :class:`~fastmd.distributed.node_halo.OwnedSlotHooks`).

:func:`ghost_row_report` finds them on a running :class:`ShardedEvaluator`
(collective, every rank in the same order, after a first ``evaluate``): a
clean forward without a backward records which modules run and which parts of
their inputs and outputs have the rank's rows; then one forward per probe,
compared with the clean forward's total energy:

* output: NaN in the ghost rows of the module's output (every call; a tensor
  and each tensor of a tuple or list, then one entry of a mapping at a time).
  A module whose NaN leaves the energy finite and unchanged (within the bound:
  the tolerance, at least four times the spread of five clean forwards, which
  on a GPU differ by their atomic sums) has dead ghost rows: nothing reads
  them. A read almost always gives a non-finite energy (NaN propagates); a
  finite change beyond the bound is reported apart ("changed");
* inputs: NaN in the ghost rows of the module's tensor arguments with the
  rank's rows (positional or keyword: what ``owned_slot_inputs`` cuts), then
  of one entry of a mapping argument at a time (in a shallow copy, the entry
  restored in the module's returned mapping, so that only this module reads
  the NaN), passed to this module only. A module whose tensor arguments' NaN
  leaves the energy unchanged neither reads its ghost-row inputs for its owned
  rows nor has a ghost-row output that is read: it may run on the owned slots
  only ("trimmable"). A module that reads the senders' rows (before the
  message sum) changes the energy here, even where its own ghost-row outputs
  are dead (a whole interaction block). Mapping entries are reported, not
  counted in "trimmable": a bare module name in ``owned_slot_inputs`` cuts
  tensor arguments only; a dead entry is declared as ``(module, (argument,
  key))``.

The report is the starting point of a declaration, which a person still
reviews: a module that is trimmable on one system may not be on another (a
branch some inputs take), and whether its output needs zero ghost rows
appended (``zero_ghost_outputs``: a consumer that reads every row, e.g. a
gather over all rows) shows only when the trimmed model runs; the generic
checks (``tests/ghost_row_checks.py``) then verify the declaration. Run it on
the untrimmed model (:func:`untrimmed`, :func:`declared_shard`). The Python
entry point is :func:`ghost_row_report` on an initialized evaluator.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from .node_halo import _ACTIVE, ExchangeHooks


def untrimmed(spec):
    """``spec`` without its ghost-row trimming (no module on the owned slots only)."""
    return dataclasses.replace(spec, owned_slot_inputs=(), zero_ghost_outputs=())


@contextlib.contextmanager
def declared_shard(adapter, spec):
    """``adapter.shard_spec()`` returns ``spec`` inside (an instance attribute over the method, removed after): an
    evaluator built inside runs ``spec``."""
    adapter.shard_spec = lambda: spec
    try:
        yield adapter
    finally:
        del adapter.shard_spec


def forward_energy(evaluator) -> float:
    """The total energy (FP64, the same on every rank) of one forward without a backward at the evaluator's current
    layout and positions, with the forward's exchanges and its declared trimming (collective; a P>1 node-halo
    evaluator, CUDA or the CPU reference, after a first ``evaluate``)."""
    engine = evaluator._engine
    if evaluator.single or getattr(evaluator, "_line", False):
        raise ValueError("forward_energy runs the node-halo forward of a P>1 or CPU-rank evaluator")
    if evaluator.device.type == "cuda":
        builder = engine.topology.builder
        batch, owned, layout, inputs = builder.batch, builder.owned_mask, engine.layout, engine.inputs
        derivative, n_total, first = engine.derivative, engine.n_total, engine.first
    else:
        batch, owned, layout = engine.batch, engine.owned_mask, engine.layout
        inputs, derivative = evaluator.adapter.prepare(batch), evaluator.derivative
        n_total = torch.tensor(float(evaluator.natoms), dtype=torch.float64)
        first = torch.tensor(evaluator.context.rank == 0)
    exchanges = derivative.context(layout, owned, n_total, first)
    with torch.no_grad(), ExchangeHooks.exchanging(exchanges):
        local = derivative.local_energy(inputs, batch.positions.detach(), batch.cell, owned, exchanges)
    total = local.detach().to(torch.float64).reshape(1).clone()
    if evaluator.context.world_size > 1:
        dist.all_reduce(total, group=evaluator.context.group)
    return float(total)


def _ghost_rows(trailing: int = 0):
    """(owned slots, every row: the ghost rows and ``trailing`` rows after them included) of the running P>1 forward,
    or None outside it (or without ghost rows)."""
    exchanges = _ACTIVE.get()
    if exchanges is None or not exchanges.ghost_rows:
        return None
    return exchanges.owned_rows, exchanges.owned_rows + exchanges.ghost_rows + trailing


def _rows(value, every: int) -> bool:
    return isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == every and value.is_floating_point()


def _integer_keys(values, every: int) -> set:
    """The keys of the mapping entries of ``values`` that are integer tensors with the rank's rows (atom types: no
    NaN to write, not probed)."""
    return {key for value in values if isinstance(value, Mapping) for key, entry in value.items()
            if isinstance(key, str) and isinstance(entry, Tensor) and entry.ndim > 0 and entry.shape[0] == every
            and not entry.is_floating_point()}


def _parts(values, every: int, *, nested: bool) -> set:
    """The probe parts of some values: None for the tensors with the rank's rows (with ``nested``, those in tuples
    and lists too: a padded output's), each key of a mapping whose entry has them."""
    parts = set()
    for value in values:
        if isinstance(value, Mapping):
            parts.update(key for key, entry in value.items() if isinstance(key, str) and _rows(entry, every))
        elif _rows(value, every) or (nested and isinstance(value, (tuple, list))
                                     and any(_rows(item, every) for item in value)):
            parts.add(None)
    return parts


def _poison(tensor, owned: int, every: int):
    if not _rows(tensor, every):
        return tensor
    tensor = tensor.clone()
    tensor[owned:] = math.nan
    return tensor


class _Recorder:
    """Forward pre-hook and hook: the calls of a module in a P>1 forward and the probe parts of its inputs and
    outputs."""

    def __init__(self, trailing: int = 0):
        self.calls, self.inputs, self.outputs, self.trailing = 0, set(), set(), trailing
        self.integers = set()

    def before(self, module, args, kwargs=None):
        rows = _ghost_rows(self.trailing)
        if rows is not None:
            self.calls += 1
            values = (*args, *(kwargs or {}).values())
            self.inputs |= _parts(values, rows[1], nested=False)
            self.integers |= _integer_keys(values, rows[1])

    def after(self, module, args, output):
        rows = _ghost_rows(self.trailing)
        if rows is not None:
            self.outputs |= _parts((output,), rows[1], nested=True)


class _PoisonInputs:
    """Forward pre-hook and hook: NaN in the ghost rows of the module's tensor arguments with the rank's rows
    (``part`` None), or of one entry of its mapping arguments (in a shallow copy; the entry the module passes on
    unchanged in its returned mapping gets its clean value back, so that only this module reads the NaN)."""

    def __init__(self, part, trailing: int = 0):
        self.part, self.swapped, self.trailing = part, [], trailing

    def before(self, module, args, kwargs=None):
        rows = _ghost_rows(self.trailing)
        if rows is None:
            return None
        owned, every = rows

        def poison(value):
            if self.part is None:
                return _poison(value, owned, every)
            if not isinstance(value, Mapping) or self.part not in value or not _rows(value[self.part], every):
                return value
            copied = value.copy() if hasattr(value, "copy") else dict(value)
            copied[self.part] = _poison(value[self.part], owned, every)
            self.swapped.append((copied[self.part], value[self.part]))
            return copied

        args = tuple(poison(arg) for arg in args)
        return args if kwargs is None else (args, {name: poison(value) for name, value in kwargs.items()})

    def after(self, module, args, output):
        swapped, self.swapped = self.swapped, []
        if self.part is None or not isinstance(output, Mapping) or self.part not in output:
            return None
        for poisoned, clean in swapped:
            if output[self.part] is poisoned:  # passed on unchanged: the module's successors read the clean entry
                restored = output.copy() if hasattr(output, "copy") else dict(output)
                restored[self.part] = clean
                return restored
        return None


class _PoisonOutput:
    """Forward hook: NaN in the ghost rows of the module's output (``part`` None: a tensor, each tensor of a tuple or
    list), or of one entry of a mapping output (in a shallow copy)."""

    def __init__(self, part, trailing: int = 0):
        self.part, self.trailing = part, trailing

    def __call__(self, module, args, output):
        rows = _ghost_rows(self.trailing)
        if rows is None:
            return None
        owned, every = rows
        if self.part is not None:
            if not isinstance(output, Mapping) or self.part not in output:
                return None
            copied = output.copy() if hasattr(output, "copy") else dict(output)
            copied[self.part] = _poison(output[self.part], owned, every)
            return copied
        if isinstance(output, Tensor):
            return _poison(output, owned, every)
        if isinstance(output, (tuple, list)):
            values = [_poison(item, owned, every) for item in output]
            return type(output)(*values) if hasattr(output, "_fields") else type(output)(values)
        return None


def _order(parts) -> list:
    return sorted(parts, key=lambda part: (part is not None, part or ""))


def ghost_row_report(evaluator, modules=None, *, tolerance: float = 0.0, repeats: int = 5) -> dict:
    """Each module's ghost rows in the evaluator's P>1 forward (collective; the same verdicts on every rank).

    ``modules``: the dotted paths to probe (default: every submodule but the
    root). Returns ``{"energy": the clean forward's total, "noise": the spread
    (largest minus smallest) of ``repeats`` clean forwards, "tolerance": the
    bound (``tolerance``, at least four times the noise), "modules": {path:
    {"calls": this rank's calls in a forward, "output": {part: probe},
    "inputs": {part: probe}, "trimmable": bool}}, "scripted": [paths]}``; a
    part is "" (the tensors) or a mapping's key, a probe ``{"verdict",
    "delta"}``: "dead" (NaN there leaves the energy finite and within the
    bound), "read" (a non-finite energy) or "changed" (finite, beyond the
    bound), with the energy's change from the first clean forward (None where
    it is not finite), or "integer" for a mapping argument's integer entry with
    the rank's rows (atom types), which takes no NaN and is not probed; "trimmable": the probe of its tensor arguments (part "")
    is dead. A module whose inputs and outputs never have the rank's rows has
    empty probes (it does not run, or its inputs hold the owned slots only
    already). TorchScript modules (and their submodules), which take no Python
    hooks, are not probed: ``"scripted"`` lists them.
    """
    module = evaluator.adapter.module
    names = [name for name, _ in module.named_modules() if name] if modules is None else list(modules)
    # A TorchScript module takes no Python hooks (nor do its submodules): left out, listed apart.
    scripted = [name for name in names if isinstance(module.get_submodule(name), torch.jit.ScriptModule)]
    names = [name for name in names if not any(name == other or name.startswith(other + ".") for other in scripted)]
    targets = {name: module.get_submodule(name) for name in names}
    trailing = int(getattr(evaluator.spec, "trailing_rows", 0))  # (a sink row after the ghost rows: TACE's)
    recorders = {name: _Recorder(trailing) for name in names}
    handles = []
    try:
        for name in names:
            handles.append(targets[name].register_forward_pre_hook(recorders[name].before, with_kwargs=True))
            handles.append(targets[name].register_forward_hook(recorders[name].after))
        clean = [forward_energy(evaluator) for _ in range(max(1, repeats))]
    finally:
        for handle in handles:
            handle.remove()
    energy = clean[0]
    noise = max(clean) - min(clean)
    bound = max(float(tolerance), 4.0 * noise)
    # Every rank probes the same list: the union of the ranks' parts.
    local = {name: (_order(recorders[name].inputs), _order(recorders[name].outputs), sorted(recorders[name].integers))
             for name in names}
    gathered = evaluator.context.all_gather_object(local) if evaluator.context.world_size > 1 else [local]

    def probe(target, hooks) -> dict:
        handles = [register(hook) for register, hook in hooks]
        try:
            value = forward_energy(evaluator)
        finally:
            for handle in handles:
                handle.remove()
        delta = value - energy
        verdict = "read" if not math.isfinite(value) else "dead" if abs(delta) <= bound else "changed"
        return {"verdict": verdict, "delta": delta if math.isfinite(delta) else None}

    report = {}
    for name in names:
        inputs = _order({part for ranks in gathered for part in ranks[name][0]})
        outputs = _order({part for ranks in gathered for part in ranks[name][1]})
        target = targets[name]
        entry = {"calls": recorders[name].calls // len(clean),
                 "output": {part or "": probe(target, [(target.register_forward_hook, _PoisonOutput(part, trailing))])
                            for part in outputs},
                 "inputs": {}}
        for part in inputs:
            poison = _PoisonInputs(part, trailing)
            entry["inputs"][part or ""] = probe(target, [
                (lambda hook: target.register_forward_pre_hook(hook, with_kwargs=True), poison.before),
                (target.register_forward_hook, poison.after)])
        for key in sorted({key for ranks in gathered for key in ranks[name][2]}):
            entry["inputs"][key] = {"verdict": "integer", "delta": None}  # (no NaN to write: not probed)
        entry["trimmable"] = entry["inputs"].get("", {}).get("verdict") == "dead"
        report[name] = entry
    return {"energy": energy, "noise": noise, "tolerance": bound, "modules": report, "scripted": scripted}


def suggested_declaration(report: dict) -> list[str]:
    """The trimmable modules of a :func:`ghost_row_report` that are not inside another trimmable module (no trimmable
    ancestor: a trimmable module's submodules run on the rows it gets): the starting point of
    ``owned_slot_inputs``."""
    modules = report["modules"]
    trimmable = [name for name, entry in modules.items() if entry["trimmable"]]
    return [name for name in trimmable
            if not any(name.startswith(other + ".") for other in trimmable if other != name)]
