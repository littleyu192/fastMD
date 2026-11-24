"""The P>1 preparation contract and run boundary; forces by one autograd backward over the exchange schedule.

A model's exchange strategy (``fastmd.distributed.strategy``; MatRIS's
``prepare_bond_partial``) prepares and runs the forward; this module holds
what every preparation shares: the model and shard stamps and guards,
``PreparedExecution`` (which names its strategy), ``validate_prepared`` and
``run_prepared``. The rows exchange's own preparation and forward were
removed in step 4.2.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import gc
import itertools
import operator
import time
from typing import Any

import torch
from torch import Tensor

from fastmd.engine.recorder import recorder_hook_ids
from fastmd.engine.cache_provenance import (
    PROJECTION_CACHE_FIELDS, cache_value_stamp, validate_projection_caches,
)
from fastmd.precision import gemm_precision_key

from fastmd.engine.grad import AutogradBackend, GradValue
from fastmd.engine.runtime import MODEL_FORWARD_METHODS

from .exchange import ExchangePlan


@dataclass
class ExecutionResult:
    # The forward's AutogradBackend (fastmd.engine.grad), which owns its
    # backward: ``backend.backward(energy)`` sets ``positions.grad``.
    # ``energy``/``positions`` are its opaque stage values.
    backend: AutogradBackend
    energy: GradValue
    positions: GradValue
    stats: dict[str, Any]
    # The strategy's backward scope of this forward (``ExchangeStrategy.backward_scope`` bound to its
    # preparation; None: none): the VJP runs inside it. MatRIS: the wide-GEMM thresholds the forward ran with,
    # which its backward's recompute recipes decide with too.
    backward_scope: Any = None
    # The zero ``[3, 3]`` strain leaf of a forward with the stress (``forward(..., stress=True)``), else
    # None: ``forces`` then also sets its gradient, which ``global_stress`` sums over the ranks.
    strain: GradValue | None = None

    def readout(self, context) -> tuple[Tensor, Tensor]:
        """The energy summed over ranks and the forces of the local rows.

        The P>1 part of the one readout: the owned rows' energy
        (``MatRIS.atomic_energy``, the P=1 per-atom rows summed over this rank)
        gets one VJP on the backend's token chain (``forces``) and one SUM over
        the data group (``global_energy``), in that order. Used eagerly and
        inside the whole-step capture alike; timed callers call the two parts.
        """
        forces = self.forces()
        return self.global_energy(context), forces

    def forces(self) -> Tensor:
        """Forces of the local rows: one VJP of the anchored owned energy (sets ``positions.grad``, and
        ``strain.grad`` with the stress)."""
        inputs = () if self.strain is None else ((self.positions, self.strain),)
        if self.backward_scope is None:
            self.backend.backward(self.energy, *inputs)
        else:
            with self.backward_scope():
                self.backend.backward(self.energy, *inputs)
        return -self.positions.grad

    def global_energy(self, context) -> Tensor:
        """The owned energy SUMmed over the data group (after ``forces``: the reverse exchanges come first)."""
        energy = self.energy.tensor.detach().clone()
        if context.world_size > 1:
            torch.distributed.all_reduce(energy, group=context.group)
        return energy.reshape(())

    def global_stress(self, context, cell: Tensor) -> Tensor:
        """The stress in eV/Å³ (after ``forces``), replicated: this rank's nine strain gradients SUMmed over
        the data group in FP64, over the volume of the unstrained ``cell`` (FP64), in the positions' dtype.

        Every rank strained each of its local rows and its cell once, so the SUM is the derivative of the
        global energy with respect to one strain of the whole system (``docs/distributed_inference.md``).
        A SUM of its own after the energy's, which is untouched: under deterministic kernels the energy and
        the forces keep the bits of a forward without the stress (the zero strain adds exact zeros). Each
        rank's gradient is symmetric; a SUM over three or more ranks may add the two halves in different
        orders, so the result is symmetrized (a no-op on a symmetric sum). The volume is |det(cell)|.
        """
        from fastmd.engine.energy_derivative import cell_volume
        gradient = self.strain.grad
        virial = gradient.to(torch.float64, copy=True)
        if context.world_size > 1:
            torch.distributed.all_reduce(virial, group=context.group)
            virial = 0.5 * (virial + virial.transpose(0, 1))
        return (virial / cell_volume(cell.to(torch.float64))).to(gradient.dtype)


def _tensor_stamp(tensor):
    if not isinstance(tensor, Tensor):
        buffers = getattr(tensor, "tensors", ())
        return (id(tensor), len(tensor) if tensor is not None else None,
                tuple(_tensor_stamp(buffer) for buffer in buffers))
    return (id(tensor), tensor._version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


def _shard_stamp(shard):
    """Cheap mutation guard for an immutable, finalized topology epoch.

    Canonical key tuples are immutable by contract; retaining their identity
    avoids re-hashing O(T) nested keys. Device tensors additionally carry their
    mutation version, so an in-place coordinate/index change is not mistaken
    for the same prepared input.
    """
    names = ("positions", "atomic_numbers", "cell", "owned_atom_ids", "atom_ids",
             "owned_bond_keys", "owned_triplet_keys", "directed_keys",
             "center_directed_keys", "line_triplet_keys", "line_source_index",
             "line_target_index", "line_owner_ranks", "line_owner_rows",
             "atom_owner_ranks", "atom_owner_rows", "center_bond_owner_ranks",
             "center_bond_owner_rows", "triplet_edge_index")
    stamps = tuple((name, _tensor_stamp(getattr(shard, name, None))) for name in names)
    topology = getattr(shard, "device_topology", None)
    # The mapping is small and fixed-size; never iterate a diagnostic entity
    # view or read its tensor values. Guard even buffers not consumed directly
    # by execution, since they jointly describe this admitted topology epoch.
    topology_stamp = (None if topology is None else
                      (id(topology), tuple((name, _tensor_stamp(value))
                                           for name, value in sorted(topology.items()))))
    return stamps + (("device_topology", topology_stamp),)


def _compact_row_map(rows, count):
    """Map admitted sorted unique rows to compact indices in bounded space.

    This is evaluation-local storage, never a cache across topology epochs.
    Callers validate the row set and all consumers before using -1 sentinels.
    """
    mapping = torch.full((count,), -1, dtype=torch.int64, device=rows.device)
    mapping[rows] = torch.arange(rows.numel(), device=rows.device)
    return mapping


def _hook_stamp(owner, names):
    """Executable callbacks and flags are model state, without calling them.

    The op-selection recorder's own site hooks only observe and are left out.
    """
    exempt = recorder_hook_ids()
    return tuple(tuple((key, id(value)) for key, value in getattr(owner, name, {}).items()
                       if key not in exempt)
                 for name in names)


_LOCAL_HOOKS = (
    "_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks",
    "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks_with_kwargs",
)
_LOCAL_HOOK_NAMES = frozenset(_LOCAL_HOOKS)
_MODULE_REGISTRIES = frozenset(("_parameters", "_buffers", "_modules"))
_PRIVATE_MODEL_FLAGS = frozenset(("_fuse_ln", "_core_act", "_gate_act", "_merged_attn_proj_splits"))
_GLOBAL_HOOKS = (
    "_global_forward_pre_hooks", "_global_forward_hooks", "_global_backward_pre_hooks",
    "_global_backward_hooks", "_global_forward_hooks_always_called", "_global_forward_hooks_with_kwargs",
)


@dataclass(frozen=True, eq=False)
class ModelStateStamp:
    """Full boundary state plus its cache-independent immutable program."""

    full: tuple
    program: tuple
    cache_errors: tuple

    def __eq__(self, other):
        if not isinstance(other, ModelStateStamp):
            return NotImplemented
        # Program entries are a strict subset of full state. Avoid comparing
        # every parameter and module binding twice at the run boundary.
        return self.full == other.full and self.cache_errors == other.cache_errors


def model_program_stamp(stamp):
    """Admit only an unchanged program with officially built frozen caches."""
    if stamp.cache_errors:
        raise ValueError("; ".join(stamp.cache_errors))
    return stamp.program


def _model_stamp(model):
    """Detect model mutation without copying weights or synchronizing CUDA.

    One fresh traversal per boundary inspects each module namespace once and
    stamps each aliased tensor once *within this call*. No inspection plan,
    tensor version, attribute, hook, or binding survives into the next call.
    Registry names/order and alias bindings are included, rather than relying
    on the deduplicated iteration order of three separate model traversals.
    """
    plain = (bool, int, float, str, type(None))
    modules, program_modules, cache_errors = [], [], []
    seen, tensor_stamps, canonical_metadata = set(), {}, {}
    pending = [model]

    def tensor_stamp(value):
        identity = id(value)
        stamp = tensor_stamps.get(identity)
        if stamp is None:
            # requires_grad changes native frozen-projection eligibility but
            # does not increment Tensor._version. ``param.data = other``
            # keeps id, version and layout: only the data pointer moves (a
            # captured graph would replay the released storage).
            try:
                pointer = value.data_ptr()
            except RuntimeError:  # no storage (e.g. sparse): the other fields decide
                pointer = None
            stamp = (_tensor_stamp(value), value.requires_grad, pointer)
            tensor_stamps[identity] = stamp
        return stamp

    def projection_metadata(value):
        # Ordinary executor stamps are (six_fields, requires_grad, ptr),
        # whereas provenance seals use eight flat fields ending in
        # requires_grad/ptr. Reuse this boundary's already-read fields, never
        # confuse formats or omit the pointer check. IDs distinguish aliased
        # Tensor *objects*; different views sharing storage still get their
        # own full metadata.
        identity = id(value)
        metadata = canonical_metadata.get(identity)
        if metadata is None:
            base, requires_grad, pointer = tensor_stamp(value)
            metadata = (*base, requires_grad, pointer)
            canonical_metadata[identity] = metadata
        return metadata

    while pending:
        module = pending.pop()
        identity = id(module)
        if identity in seen:
            continue
        seen.add(identity)
        namespace = vars(module)
        # Most stock modules have scalar attributes but no plain Tensor extras,
        # instance overrides, populated hooks, or projection fields. Allocate
        # those four append-only lists only if this fresh namespace needs them.
        # None is internal allocation state, never part of a returned stamp.
        attributes = []
        extra = methods = hooks = caches = None
        for name, value in namespace.items():
            if name in _LOCAL_HOOK_NAMES:
                # Empty hooks are overwhelmingly common; avoid allocating
                # seven empty tuples and doing seven getattr calls per layer.
                if value:
                    if hooks is None:
                        hooks = []
                    hooks.append((name, tuple((key, id(callback))
                                              for key, callback in value.items())))
                elif not isinstance(value, dict) and not isinstance(value, Mapping):
                    raise TypeError(f"invalid module hook registry: {name}")
                continue
            if name in _MODULE_REGISTRIES:
                continue
            if name in PROJECTION_CACHE_FIELDS:
                # Keep complete cache state in the fresh prepare/run guard,
                # but separate it from the immutable mathematical program.
                # Exclusion is legal only with builder provenance below.
                # Use canonical provenance format for the opaque full-cache
                # stamp too, so its tuple tree can be reused without another
                # recursive inspection. These stamps are not serialized APIs.
                if caches is None:
                    caches = []
                caches.append((name, cache_value_stamp(value, projection_metadata)))
                continue
            if name in MODEL_FORWARD_METHODS:
                if methods is None:
                    methods = []
                methods.append((name, id(value)))
            if (name[:1] != "_" or name in _PRIVATE_MODEL_FLAGS) and (
                isinstance(value, plain)
                or isinstance(value, tuple) and all(isinstance(part, plain) for part in value)
            ):
                attributes.append((name, value))
            elif isinstance(value, Tensor):
                if extra is None:
                    extra = []
                extra.append((name, tensor_stamp(value)))
        parameters = tuple((name, None if value is None else tensor_stamp(value))
                           for name, value in namespace.get("_parameters").items())
        buffers = tuple((name, None if value is None else tensor_stamp(value))
                        for name, value in namespace.get("_buffers").items())
        children = namespace.get("_modules")
        bindings = tuple((name, id(child)) for name, child in children.items())
        if children:
            pending.extend(child for child in reversed(tuple(children.values())) if child is not None)
        program = (identity, type(module), tuple(attributes), tuple(extra) if extra else (),
                   tuple(methods) if methods else (), tuple(hooks) if hooks else (),
                   namespace.get("_is_full_backward_hook"), parameters, buffers, bindings,
                   tuple(name for name, _ in caches) if caches else ())
        program_modules.append(program)
        modules.append((program, tuple(caches)) if caches else program)
        if caches:
            cache_errors.extend((identity, message)
                                for message in validate_projection_caches(
                                    module, namespace, field_stamps=dict(caches),
                                    tensor_metadata=projection_metadata))
    global_hooks = (_hook_stamp(torch.nn.modules.module, _GLOBAL_HOOKS),
                    getattr(torch.nn.modules.module, "_global_is_full_backward_hook", None))
    config = repr(getattr(model, "config", None))
    return ModelStateStamp((tuple(modules), config, global_hooks),
                           (tuple(program_modules), config, global_hooks),
                           tuple(f"module {identity}: {message}" for identity, message in cache_errors))


MODEL_MUTATED = "model program was mutated after evaluator construction"
_chain = itertools.chain.from_iterable
_is = operator.is_


def _global_hook_sizes():
    module = torch.nn.modules.module
    return tuple(len(getattr(module, name, ())) for name in _GLOBAL_HOOKS)


def _guard_tensor_metadata(tensor):
    """A tensor's value-free fields as the full stamp reads them (layout, version, requires_grad, pointer)."""
    try:
        pointer = tensor.data_ptr()
    except RuntimeError:  # no storage (e.g. sparse): the other fields decide
        pointer = None
    return _tensor_stamp(tensor), tensor.requires_grad, pointer


def _code_bindings(model):
    """Callable module attributes and registry value types, which the program omits.

    The program stamps MODEL_FORWARD_METHODS overrides but not, e.g., a
    private helper override (``model._normalize_real_counts = f``), and
    Tensor stamps omit the type (``parameter.__class__ = Subclass``). Reads
    no attribute through user code (``type``/``callable`` only).
    """
    namespaces = tuple(map(vars, model.modules()))
    code = {(index, name): value for index, namespace in enumerate(namespaces)
            for name, value in namespace.items() if callable(value)}
    types = tuple(type(value) for namespace in namespaces for name in sorted(_MODULE_REGISTRIES)
                  if isinstance(registry := namespace.get(name), dict) for value in registry.values())
    return code, types


class OverlappedModelGuard:
    """Per-evaluation immutable-program check for one rank (P=1), off the critical path.

    ``precheck()`` runs before any model code to stop a rebinding that can
    install code, so that it never runs (nor undoes itself before a later
    check). Against the last verified baseline it compares, by identity,
    every module's class and ``__dict__`` value (hook dicts, registries,
    instance methods, attributes, tensors, caches, the training flag), each
    child, parameter and buffer and its type, plus namespace sizes and last
    keys (a delete plus an add), local hook emptiness, registry sizes and
    global hook sizes. The baseline holds these objects, so an identity is
    never recycled. Only a difference runs, pre-launch, a check of callable
    attributes and registry value types against the constructor's (before
    anything reads a swapped tensor's metadata) and then the full program
    scan; a match (e.g. official caches materialized meanwhile) re-baselines,
    as does ``verify()``.

    ``verify()`` is the one full scan per steady-state evaluation, run after
    the device work is submitted (it overlaps it) and before any result is
    read. It alone sees in-place edits of an unchanged object (weight data,
    version counters, requires_grad, a ``.data`` rebinding via the data
    pointer, a dict such as ``model.config``); such an edit cannot run code,
    so rejecting it before readout loses nothing (a replay after a ``.data``
    rebinding has read the released address once; its result is discarded
    and the graphs dropped).
    Residual, as in the historical per-step stamp: code reached through class
    attributes (a patched class method) or held inside a non-callable
    attribute (a mutable container, a non-module object) is not part of the
    program; a Tensor subclass bound to a non-registry attribute is not
    type-checked, and its ``__torch_function__`` runs when a full scan reads
    its metadata.
    """

    def __init__(self, model, program):
        self.model, self.program = model, program
        self.full_scans = 0
        self.scan_seconds = 0.0
        self._code, self._code_types = _code_bindings(model)
        self._collect()

    def _collect(self):
        modules = tuple(self.model.modules())
        namespaces = tuple(map(vars, modules))
        registries = tuple(value for namespace in namespaces for name in _MODULE_REGISTRIES
                           if isinstance(value := namespace.get(name), dict))
        hooks = tuple(value for namespace in namespaces for name in _LOCAL_HOOKS
                      if isinstance(value := namespace.get(name), dict))
        self._modules, self._types = modules, tuple(map(type, modules))
        self._sizes = tuple(map(len, namespaces))
        self._last_keys = tuple(next(reversed(namespace), None) for namespace in namespaces)
        self._values = tuple(_chain(map(dict.values, namespaces)))
        # Held local hook dicts (admitted empty by validate_model) and
        # registries: any() is the size check of the empty ones (RDX: 5,536
        # of 5,960), cheaper than one length tuple over all of them.
        held = hooks + registries
        self._empty = tuple(value for value in held if not value)
        self._nonempty = tuple(value for value in held if value)
        self._nonempty_sizes = tuple(map(len, self._nonempty))
        self._filled = tuple(registry for registry in registries if registry)
        self._registry_values = tuple(_chain(map(dict.values, self._filled)))
        self._registry_types = tuple(map(type, self._registry_values))
        self._global_sizes = _global_hook_sizes()

    def _matches(self) -> bool:
        # RDX (596 modules, 11k namespace values): ~0.4 ms isolated, ~0.5 ms
        # inside an eager evaluate, above the 0.3 ms target; pure identity
        # (tuple/dict == could call a swapped object's __eq__). Order matters:
        # equal sizes make every map() below pair baseline and current
        # entries exactly; equal namespace values prove the held registry
        # dicts are the bound ones before their values are read (len() of a
        # held dict runs no model code).
        modules = self._modules
        namespaces = tuple(map(vars, modules))
        return (all(map(_is, map(type, modules), self._types))
                and tuple(map(len, namespaces)) == self._sizes
                and not any(self._empty)
                and tuple(map(len, self._nonempty)) == self._nonempty_sizes
                and all(map(_is, map(next, map(reversed, namespaces), itertools.repeat(None)), self._last_keys))
                and all(map(_is, _chain(map(dict.values, namespaces)), self._values))
                and all(map(_is, _chain(map(dict.values, self._filled)), self._registry_values))
                and all(map(_is, map(type, self._registry_values), self._registry_types))
                and _global_hook_sizes() == self._global_sizes)

    def _unchanged(self) -> bool:
        try:
            return self._matches()
        except Exception:  # e.g. a deleted attribute: the full scan decides
            return False

    def _rebaseline(self) -> None:
        try:
            self._collect()
        except Exception as exc:
            raise ValueError(MODEL_MUTATED) from exc

    def mutation(self) -> ValueError | None:
        """One full scan against the constructor program; the rejection, or None."""
        started = time.perf_counter()
        # The stamp's many acyclic temporaries otherwise trigger ~3 young-
        # generation collections per evaluation; refcounting frees them.
        collecting = gc.isenabled()
        gc.disable()
        try:
            error = (None if model_program_stamp(_model_stamp(self.model)) == self.program
                     else ValueError(MODEL_MUTATED))
        except Exception as exc:
            error = ValueError(f"{MODEL_MUTATED}: {exc}")
            error.__cause__ = exc
        finally:
            if collecting:
                gc.enable()
        self.full_scans += 1
        self.scan_seconds += time.perf_counter() - started
        return error

    def _code_mutation(self) -> ValueError | None:
        """Callable attributes and registry value types against the constructor's, by identity."""
        try:
            code, types = _code_bindings(self.model)
            admitted = self._code
            if (code.keys() == admitted.keys() and all(value is admitted[key] for key, value in code.items())
                    and len(types) == len(self._code_types) and all(map(_is, types, self._code_types))):
                return None
        except Exception as exc:
            error = ValueError(f"{MODEL_MUTATED}: {exc}")
            error.__cause__ = exc
            return error
        return ValueError(MODEL_MUTATED)

    def precheck(self) -> None:
        """Before model code: full scan only when a binding moved."""
        if self._unchanged():
            return
        error = self._code_mutation()
        if error is None:
            error = self.mutation()
        if error is not None:
            raise error
        self._rebaseline()

    def verify(self) -> None:
        """After submission, before readout: the full scan; re-baseline on success."""
        unchanged = self._unchanged()
        error = None if unchanged else self._code_mutation()
        if error is None:
            error = self.mutation()
        if error is not None:
            raise error
        if not unchanged:
            self._rebaseline()


class DistributedModelGuard(OverlappedModelGuard):
    """OverlappedModelGuard for the multi-rank protocol path, one per evaluator.

    ``prepare_bond_partial(model_guard=...)`` runs ``precheck()`` where the
    public path runs its first full scan (before any model attribute or code
    is used); the one full scan per evaluation is ``verify()``: at the eager
    reference-topology run boundary (``validate_prepared``, before the first
    payload), or after the step's device work is submitted: the epoch step,
    eager (C9) or replayed. Beyond the P=1 rules:

    * The precheck also compares every parameter, buffer and official
      projection-cache tensor by value-free metadata (storage pointer,
      version counter, layout, requires_grad; ``_collect``/``_matches``), so
      an in-place, ``.data`` or requires_grad change of one reaches the full
      scan before any payload. Deferring the eager scan past submission is
      therefore safe: model code runs only after the precheck, so no code
      installed since the last scan runs and no rank meets a changed tensor
      layout mid-step. What remains for the deferred scan is state that only
      the full scan reads (dict contents such as ``model.config``, a tensor
      attribute outside the registries and caches, mutable containers); its
      rejection is agreed in the finite round, every rank rejects alike and
      the step returns no result. Such a change cannot run code; if one
      rank's model code nevertheless fails on it, the step fails (the
      communicator's timeout), again without a result.

    * Outcomes are messages (``precheck_error``/``verify_error``): a rank must
      not raise before its peers reach the same collective, so callers put the
      message into their next agreement round and every rank rejects alike.
    * ``stamp`` retains the last admitted full state (program plus official
      caches) for the graph runner's recapture decision, instead of a fresh
      scan per boundary. It equals the full state of every accepted
      evaluation: a passing identity precheck proves each binding the full
      stamp records by identity unchanged since ``stamp`` was scanned; what is
      left (tensor versions, data pointers, requires_grad, dict contents such
      as ``model.config``, a hook replaced under the same key, a cache tensor
      edited in place) is program state or checked by the cache seals, which
      the evaluation's own full scan rejects. A rejected evaluation returns
      nothing, and a graph captured during it is not bound to ``stamp``.
    * No gc.disable() around the scan (unlike P=1): long-lived objects are
      frozen at warmup/capture points (``freeze_long_lived_objects``), so the
      scan's young collections (~2 per scan, ~0.25 ms at 596 modules) never
      traverse them, and ``stamp`` may outlive the scan.
    * Admission differs from the unguarded path (one rank; P>1 before 3.1a)
      in three accepted ways. Stricter: the P=1 precheck also rejects callable attributes
      rebound after construction and registry value-type changes
      (``_code_mutation``), which the program stamp admits. Later: a
      change only the full scan sees (a dict value such as ``model.config``,
      or, between replays, a tensor edit) is rejected at the run boundary, or
      after the step's submission, not in prepare's first round. A rejected
      model_graph evaluation may thus already have grown the padded capacity
      (TieredCapacity.plan) and replayed once, so later padding can
      differ from the unguarded path's; and a combined failure is reported
      by the first round that sees one (another rank's prepare error alone).
      Earlier: a program or cache change
      made by the model's own code during a (re)capture's warmup or capture
      fails the post-capture scan and rejects that evaluation (P=1 verifies
      after model code likewise); the unguarded path binds the post-capture
      stamp without comparing it, returns that evaluation's result and
      rejects from the next one. Every rank still rejects identically and a
      rejected evaluation returns no result.
    """

    def __init__(self, model, program, stamp: ModelStateStamp):
        if not isinstance(stamp, ModelStateStamp) or model_program_stamp(stamp) != program:
            raise ValueError("model guard stamp must be the scan that produced the admitted program")
        super().__init__(model, program)
        self.stamp = stamp

    def _collect(self):
        super()._collect()
        # Beyond identity: every parameter, buffer and official projection-cache
        # tensor by value-free metadata (storage pointer, version counter, shape,
        # stride, dtype, device, requires_grad). The eager epoch step prechecks
        # before its payloads and scans after submission (C9): an in-place or
        # ``.data`` change of these tensors must still stop before any payload,
        # since eager model code on one rank could otherwise raise mid-step.
        self._tensors = tuple(
            value for module in self._modules for value in itertools.chain(
                module._parameters.values(), module._buffers.values(),
                _chain(value if isinstance(value, tuple) else (value,)
                       for name, value in vars(module).items() if name in PROJECTION_CACHE_FIELDS))
            if isinstance(value, Tensor))
        self._tensor_metadata = tuple(map(_guard_tensor_metadata, self._tensors))

    def _matches(self) -> bool:
        # Identity first: equal registries and namespace values prove that
        # ``_tensors`` are still the bound objects (tuples are immutable).
        return super()._matches() and tuple(map(_guard_tensor_metadata, self._tensors)) == self._tensor_metadata

    def _unchanged(self) -> bool:
        # No retained state (a failed re-baseline): the full scan decides.
        return self.stamp is not None and super()._unchanged()

    def _rebaseline(self) -> None:
        try:
            super()._rebaseline()
        except BaseException:
            self.stamp = None
            raise

    def mutation(self) -> ValueError | None:
        """One full scan against the constructor program; retain the state it admits."""
        started = time.perf_counter()
        try:
            stamp = _model_stamp(self.model)
            # Steady state: equal to the retained state, whose program and
            # caches were admitted (full equality implies both). Keep that
            # object: the runner's capture-time comparison stays O(1).
            if stamp == self.stamp:
                error = None
            elif stamp.cache_errors:
                # The seals' own text (model_program_stamp's), as the
                # unguarded prepare_bond_partial reports a mutated cache.
                error = ValueError("; ".join(stamp.cache_errors))
            elif stamp.program != self.program:
                error = ValueError(MODEL_MUTATED)
            else:
                error = None
                self.stamp = stamp
        except Exception as exc:
            error = ValueError(f"{MODEL_MUTATED}: {exc}")
            error.__cause__ = exc
        self.full_scans += 1
        self.scan_seconds += time.perf_counter() - started
        return error

    def precheck_error(self) -> str | None:
        """``precheck()`` for an agreement round: this rank's rejection, or None."""
        try:
            self.precheck()
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return None

    def verify_error(self) -> str | None:
        """``verify()`` for an agreement round: this rank's rejection, or None."""
        try:
            self.verify()
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return None


# True from freeze_long_lived_objects() until thaw_long_lived_objects().
_frozen_long_lived = False


def freeze_long_lived_objects(*, collect: bool = True) -> None:
    """Collect, then move every object alive now to the permanent GC generation.

    Call at warmup/capture points only (P>1: after the evaluator's first
    evaluation and after each model_graph (re)capture), never per step. Later
    young and full collections then skip the model, evaluator, captured graph
    and allocator state: without this a full collection traverses them every
    ~75 MD steps (1,344 atoms on 4 GPUs: ~135 ms each, a 223 ms step). Frozen
    objects are still freed by reference counting; only cyclic garbage that
    was alive at the freeze waits for ``thaw_long_lived_objects()``, which
    DistributedGraphRunner runs (with a collection) before releasing a graph,
    so the permanent generation stays bounded by the live set at the last
    freeze. ``collect=False`` only for a caller that has just run a full
    collection. Process-global, like gc.freeze(). Never poll
    gc.get_freeze_count() per step: it walks the permanent generation
    (~10 ms for ~0.5M objects).
    """
    global _frozen_long_lived
    if collect:
        gc.collect()
    gc.freeze()
    _frozen_long_lived = True


def thaw_long_lived_objects() -> bool:
    """``gc.unfreeze()`` if freeze_long_lived_objects() ran since the last thaw; True if so.

    Without one of our freezes nothing is thawed: a process that never froze
    (one rank) keeps the pre-3.1a GC state, and an
    application's own gc.freeze() stays in effect. After ours it cannot be
    told apart (gc.freeze() merges both into one generation) and is thawed too.
    """
    global _frozen_long_lived
    if not _frozen_long_lived:
        return False
    gc.unfreeze()
    _frozen_long_lived = False
    return True


@dataclass(frozen=True)
class PreparedExecution:
    """Evaluation-local device inputs and routing, separate from autograd.

    This R0 eager contract is single-use and tied to one exact model, finalized
    shard, context, and caller epoch. Preparing a newer generation invalidates
    older unused preparations in that context. No topology cache is implied.
    A future graph runner may consume the validated tensors/plans into its own
    static buffers; graph replay is distinct from reusing this one-shot object.
    """

    model: Any = field(repr=False)
    shard: Any = field(repr=False)
    context: Any = field(repr=False)
    epoch: int | None
    generation: int
    has_line: bool
    skip_dead_threebody_tail: bool
    local_backend: str
    tensors: dict[str, Tensor] = field(repr=False)
    plans: dict[str, ExchangePlan] = field(repr=False)
    stats: dict[str, Any]
    _shard_stamp: tuple = field(repr=False)
    _model_stamp: ModelStateStamp = field(repr=False)
    _execution_stamp: tuple = field(repr=False)
    _tensor_stamps: tuple = field(repr=False)
    _plan_stamps: tuple = field(repr=False)
    schedule: Any = field(default=None, repr=False)
    _consumed: bool = field(default=False, repr=False)
    # Evaluator-private preparations: ``_model_stamp`` is the guard's retained
    # admitted state, and the run boundary verifies through the guard.
    _model_guard: DistributedModelGuard | None = field(default=None, repr=False)
    # The exchange strategy that prepared it (``fastmd.distributed.strategy``), which runs its forward.
    strategy: Any = field(default=None, repr=False)


def grad_backend(prepared) -> AutogradBackend:
    """The ``GradBackend`` of one P>1 forward over ``prepared``'s exchange schedule (step 4.2: the only one)."""
    return AutogradBackend(prepared.schedule, device=prepared.tensors["owned_positions"].device)


def _plan_stamps(plans):
    return tuple((name, id(plan), plan.sequence, plan.name, plan.send_counts, plan.receive_counts,
                  _tensor_stamp(plan.send_indices), _tensor_stamp(plan.receive_indices),
                  tuple((device, _tensor_stamp(send), _tensor_stamp(receive))
                        for device, (send, receive) in plan._device_indices.items()))
                 for name, plan in plans.items())


def run_prepared(model, shard, context, prepared: PreparedExecution, *,
                 epoch: int | None = None, stress: bool = False) -> ExecutionResult:
    """Run an already-prepared E/F forward, leaving the VJP to the returned backend.

    Binding and stale-state failures are exchanged collectively before the
    first payload, so one rank cannot leave its peers entering NCCL/Gloo.
    This single-use eager entrypoint does not enable CUDA Graphs.
    ``stress``: the forward also strains (``ExecutionResult.strain``).
    """
    validate_prepared(model, shard, context, prepared, epoch=epoch)
    return _run_prepared_tensors(model, shard, context, prepared, **({"stress": True} if stress else {}))


def validate_prepared(model, shard, context, prepared: PreparedExecution, *,
                      epoch: int | None = None, defer_model_scan: bool = False) -> None:
    """Collectively validate and consume one preparation before any payload.

    Eager and graph execution share this exact admission boundary. A successful
    call consumes the preparation and records guard timings; failed admission
    does neither. It does not run model kernels or communicate tensor features.

    A preparation without a model guard always compares a fresh full stamp
    with the prepared one. With a guard, nothing but the model can have moved
    since its precheck in prepare (the evaluator calls both back to back), so
    the guard's ``verify()`` is the evaluation's one full scan here, before
    the first payload; ``defer_model_scan`` (a guarded topology epoch refresh)
    skips it: the epoch step runs the full scan after submitting its device
    work (eager, C9, after the guard's precheck agreed in the admission round;
    model_graph, after a replay), and the caller agrees that result before
    any readout.
    """
    guard_started = guard_marked = time.perf_counter()
    guard_details = {}

    def mark_guard(name):
        nonlocal guard_marked
        now = time.perf_counter()
        guard_details[name] = (now - guard_marked) * 1000
        guard_marked = now

    error = None
    try:
        if not isinstance(prepared, PreparedExecution):
            raise TypeError("expected PreparedExecution")
        if prepared.model is not model or prepared.shard is not shard or prepared.context is not context:
            raise ValueError("prepared model, shard or context identity differs")
        admitted = getattr(context, "_prepared_execution_ref", None)
        if admitted is None or admitted() is not prepared:
            raise ValueError("prepared execution object is not admitted in this context")
        if prepared.epoch != epoch:
            raise ValueError("prepared execution epoch differs")
        if prepared._consumed:
            raise ValueError("prepared execution is single-use and already consumed")
        if prepared.generation != getattr(context, "_prepared_execution_generation", None):
            raise ValueError("prepared execution generation is stale")
        if prepared._execution_stamp != (prepared.has_line, prepared.skip_dead_threebody_tail,
                                          prepared.local_backend, gemm_precision_key()):
            raise ValueError("prepared local backend, precision or execution branch was mutated")
        mark_guard("identity_ms")
        if prepared._shard_stamp != _shard_stamp(shard):
            raise ValueError("prepared shard or topology was mutated")
        mark_guard("shard_stamp_ms")
        guard = prepared._model_guard
        if guard is None:
            if prepared._model_stamp != _model_stamp(model):
                raise ValueError("prepared model was mutated")
        elif not defer_model_scan:
            guard.verify()
        mark_guard("model_stamp_ms")
        if prepared._tensor_stamps != tuple((name, _tensor_stamp(tensor))
                                            for name, tensor in prepared.tensors.items()):
            raise ValueError("prepared device input or index was mutated")
        mark_guard("input_stamps_ms")
        if prepared._plan_stamps != _plan_stamps(prepared.plans):
            raise ValueError("prepared routing plan was mutated")
        mark_guard("plan_stamps_ms")
        if prepared.schedule is not None:
            error = prepared.schedule.boundary_error()
        mark_guard("schedule_integrity_ms")
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        error = str(exc)
    context.collective_check(error)
    mark_guard("collective_ms")
    object.__setattr__(prepared, "_consumed", True)
    prepared.stats["run_guard_details_ms"] = guard_details
    prepared.stats["run_guard_ms"] = (time.perf_counter() - guard_started) * 1000


def _run_prepared_tensors(model, shard, context, prepared: PreparedExecution, *,
                          stress: bool = False) -> ExecutionResult:
    """Internal tensor body, after the caller's collective boundary validation.

    Keeping this separate from the one-shot eager lifetime permits controlled
    side-stream warmup/capture experiments with the same validated static
    buffers. Only a prevalidated schedule is capture-compatible; this helper
    itself does not admit new inputs, cache topology, or enable graph replay.

    The forward is the preparation's exchange strategy's
    (``ExchangeStrategy.forward``: for MatRIS the resolved configuration of the
    local backend's profile, nested inside the evaluator's identical scope,
    with the P>1 OpDispatcher of the prepared layout). ``stress``: with the
    strain leaf (``ExecutionResult.strain``).
    """
    if stress:
        return prepared.strategy.forward(model, context, prepared, stress=True)
    return prepared.strategy.forward(model, context, prepared)
