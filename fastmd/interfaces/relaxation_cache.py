"""The GPU relaxations :class:`~fastmd.interfaces.relax.StructOptimizer` keeps across its ``relax`` calls.

One structure on StructOptimizer's GPU path is one :class:`~fastmd.interfaces.multi_system_relax.MultiSystemRelaxation`
of one slot of its atoms. Building it (the model's negotiation, the capacities at the structure's demand, the graph
builder, the whole-step CUDA graph captured after one warmup window) costs MatRIS 10M OAM about 0.8 s (up to 1.3 s
under deterministic algorithms), more than relaxing a small structure takes. :class:`RelaxationCache` keeps the
relaxations it built, at most ``limit`` (the least recently used one is released first), and relaxes a later
structure in a kept one (:meth:`~fastmd.interfaces.multi_system_relax.MultiSystemRelaxation.restart`: written into
the captured slot, the run's state as a new relaxation's) wherever a new relaxation of it would build the same, so
that the results and trajectory records are bit for bit those of a new one:

* the key (:func:`relaxation_key`): the model (the adapter object, and each parameter's and buffer's storage and
  version: weights changed in place make another key), the calculator's task, the atom count, every construction
  argument (``options``: fmax, steps, the cell filter, FIRE's parameters, the configuration, the device, the slots,
  windows, trajectory interval and stress... each compared by value, so that an argument added later is part of the
  key without a change here; an argument that cannot be compared by value is not cached), the configuration scope
  the construction runs in (an unconfigured one takes its configuration from it, a configured one still the
  scope's kernel options), the ``MATRIS_*`` environment (read where a configuration is resolved) and torch's
  settings that choose kernels at a capture (the float32 matmul precision as ``fastmd.precision`` reads it under
  either of torch's APIs, the effective cuBLAS math, the Triton dot precision, deterministic algorithms);
* the structure's own build (``fresh_signature``: its capacities, its pair sort keys and the model's automatic
  recompute state) equal to the kept relaxation's as captured (``build_signature``); kept relaxations of one key
  with different builds (capacity tiers, sort key bounds) coexist.

A relaxation is kept only after a run that finished with every structure converged or at its step limit; one whose
construction, restart or run raised, or whose structure failed (a non-finite energy or force), is released. A kept
relaxation holds its capture's CUDA graph pool, which nothing else can use: their pools together stay within
``KEPT_MEMORY_FRACTION`` of the device's memory (one whose pool alone exceeds it is not kept), and a construction
that runs out of CUDA memory while relaxations are kept (or the structure's build signature does) releases them and
is built again; where the CUDA allocator cannot measure a pool (``backend:cudaMallocAsync``), nothing is kept. A
list's batch of several structures is built per call and not kept (its slots, union size and drains follow measured
times); a structure of a list that relaxes alone on the GPU path (the list's only one there, or a part of a batch
that could not start and was halved) relaxes in, and is kept as, a one-structure relaxation like a single call.
:meth:`RelaxationCache.close` releases every kept relaxation.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import dataclasses
import enum
import gc
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # (imported where a relaxation is built, as StructOptimizer imported it before)
    from .multi_system_relax import MultiSystemRelaxation

# The names of the key's parts in the reasons (an option is named by its argument).
_PARTS = {"model": "model", "parameters": "model parameters", "task": "task", "atoms": "atom count",
          "scope": "configuration scope", "environment": "MATRIS_* environment", "torch": "torch kernel settings"}
# The kept relaxations' CUDA graph pools together hold at most this fraction of the device's memory (the least
# recently used released first; one whose pool alone exceeds it is not kept). A pool holds its capture's working set,
# which the caching allocator cannot give to anything else while the relaxation is kept (MatRIS 10M OAM, one
# structure: about 1.1 GiB at 8 atoms, 4.1 GiB at 64, 38 GiB at 256 atoms of fcc Cu): an eighth of a GB200 (about 23
# GiB) keeps four relaxations of up to about a hundred atoms, while larger structures, whose runs outlast their
# construction, are built per call as before.
KEPT_MEMORY_FRACTION = 1.0 / 8.0
# The keys whose relaxation was not kept for its memory, remembered for the reason of a later miss.
_DECLINED_KEYS = 16


def _out_of_memory(error: BaseException) -> bool:
    """Whether a construction failed for CUDA memory (torch's error, or the engine's NegotiationError naming it)."""
    return isinstance(error, torch.OutOfMemoryError) or "out of CUDA memory" in str(error)


class _NotComparable(TypeError):
    """A construction argument that :func:`comparable` cannot compare by value."""


def comparable(value):
    """``value`` as a hashable value equal to another's exactly where the values are equal, their types' kinds
    included (``True`` is not ``1``; a float by its bits' hexadecimal form, so ``-0.0`` is not ``0.0``): None,
    bool, int, str, float, torch devices and dtypes, enums, dataclasses (by their class and fields), mappings,
    sequences and sets of such values, numpy arrays and scalars and torch tensors (by dtype, shape and bytes) and
    classes (by identity: a class is not changed in place). Anything else raises :class:`_NotComparable`."""
    if value is None or isinstance(value, (bool, int, str)):
        return type(value).__name__, value
    if isinstance(value, float):
        return "float", float(value).hex()
    if isinstance(value, (torch.device, torch.dtype)):
        return type(value).__name__, str(value)
    if isinstance(value, enum.Enum):
        return type(value).__qualname__, value.name
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__module__, type(value).__qualname__,
                tuple((field.name, comparable(getattr(value, field.name))) for field in dataclasses.fields(value)))
    if isinstance(value, Mapping):
        return "mapping", tuple(sorted(((comparable(key), comparable(item)) for key, item in value.items()), key=repr))
    if isinstance(value, (list, tuple)):
        return type(value).__name__, tuple(comparable(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return "set", tuple(sorted((comparable(item) for item in value), key=repr))
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.ascontiguousarray(value)
        return "numpy", array.dtype.str, array.shape, array.tobytes()
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        return "tensor", str(data.dtype), tuple(data.shape), data.reshape(-1).view(torch.uint8).numpy().tobytes()
    if isinstance(value, type):
        return "class", value.__module__, value.__qualname__, id(value)
    raise _NotComparable(type(value).__name__)


def model_tensors(module: torch.nn.Module) -> tuple:
    """``(storage address, version)`` of every parameter and buffer of ``module`` and its submodules, in a fixed
    order: equal while no tensor is replaced or changed in place. (One walk over the submodules' own tables: a
    MatRIS of some thousand submodules takes about 1 ms, the ``parameters()`` and ``buffers()`` generators several.)"""
    out, pending, seen = [], [module], set()
    while pending:
        part = pending.pop()
        if id(part) in seen:
            continue
        seen.add(id(part))
        for table in (part._parameters, part._buffers):
            out.extend((tensor.data_ptr(), tensor._version) for tensor in table.values() if tensor is not None)
        pending.extend(child for child in reversed(part._modules.values()) if child is not None)
    return tuple(out)


def relaxation_key(model, structures, options: Mapping, *, task: str) -> dict:
    """The parts of the key of a relaxation of ``structures`` (one structure) by ``model`` (an adapter) with the
    construction arguments ``options``, by name (see the module docstring). Raises :class:`_NotComparable` naming
    an argument that is not comparable by value. Never reads torch's legacy precision getters directly (they raise
    in states set through torch's ``fp32_precision`` API)."""
    from ..config import current_config
    from ..precision import _snapshot, fp32_gemm_mode, triton_dot_precision

    parts = {"model": id(model), "parameters": model_tensors(model.module), "task": task,
             "atoms": len(structures[0])}
    for name, value in sorted(options.items()):
        try:
            parts[f"option {name}"] = comparable(value)
        except _NotComparable as error:
            raise _NotComparable(f"argument {name} ({error})") from None
    # Every construction runs in the active scope: an unconfigured one takes its configuration from it, a configured
    # one still its kernel options' defaults (the saved-tensor recompute, read through the scope's environment).
    parts["scope"] = comparable(current_config())
    parts["environment"] = tuple(sorted((name, value) for name, value in os.environ.items()
                                        if name.startswith("MATRIS_")))
    # The legacy precision string (None where it raises) and every fp32_precision entry, the effective cuBLAS math
    # (NVIDIA_TF32_OVERRIDE included), the Triton dots' requested precision and the deterministic algorithms.
    deterministic = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    parts["torch"] = (_snapshot(), fp32_gemm_mode(), triton_dot_precision(), *deterministic)
    return parts


class RelaxationCache:
    """At most ``limit`` built one-structure relaxations, by key and build (see the module docstring), whose CUDA
    graph pools hold at most ``memory_fraction`` of the device's memory together; ``limit`` 0 keeps none (every call
    builds its relaxation, as before the cache).

    :meth:`take` returns the relaxation of a ``relax`` call (a kept one restarted with the structure, or a new one;
    a new one that runs out of CUDA memory while relaxations are kept is built again after releasing them) and
    records in ``last`` whether it was reused and why not; :meth:`give_back` keeps it after its run (or releases
    it); :meth:`close` releases every kept one."""

    def __init__(self, limit: int, memory_fraction: float | None = None) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(f"the number of kept GPU relaxations must be an integer >= 0, got {limit!r}")
        self.limit = limit
        self.memory_fraction = float(KEPT_MEMORY_FRACTION if memory_fraction is None else memory_fraction)
        # (key, build signature) -> relaxation, the least recently used first
        self._kept: OrderedDict[tuple, MultiSystemRelaxation] = OrderedDict()
        self._held: dict[tuple, int] = {}  # (key, build signature) -> the bytes of its graph pool
        self._parts: dict[tuple, dict] = {}  # key -> its parts by name (the reasons)
        self._out: dict[int, tuple] = {}  # id(relaxation) -> its key, from take to give_back
        self._declined: OrderedDict[tuple, str] = OrderedDict()  # key -> why its relaxation was not kept
        # The last take: {"reused": bool, "reason": str}.
        self.last: dict | None = None

    def __len__(self) -> int:
        return len(self._kept)

    def summary(self) -> dict:
        """The kept relaxations: their count, the limit and the bytes of their CUDA graph pools."""
        return {"kept": len(self._kept), "limit": self.limit, "graph_pool_bytes": int(sum(self._held.values()))}

    def take(self, model, structures: list, options: dict, *, task: str) -> MultiSystemRelaxation:
        """The relaxation of ``structures`` by ``model`` (``MultiSystemRelaxation(model, structures, **options)``):
        a kept one of the same key and build, restarted with the structure (:meth:`MultiSystemRelaxation.restart`),
        else a new one (its construction's exceptions propagate). ``last`` says which and why."""
        reason, key = None, None
        if len(structures) != 1:
            reason = f"a list's batch of {len(structures)} structures is built for them (not kept)"
        elif self.limit == 0:
            reason = "cached_engines=0"
        else:
            try:
                parts = relaxation_key(model, structures, options, task=task)
            except _NotComparable as error:
                reason = f"{error} is not comparable by value"
            else:
                key = tuple(parts.items())
                self._prune_parts()
                starved = False
                try:
                    relaxation, reason = self._restarted(key, parts, structures[0])
                except Exception as error:  # the structure's build signature ran out of memory: as a construction
                    if not (self._kept and _out_of_memory(error)):
                        raise
                    relaxation, starved = None, True
                    why = str(error).splitlines()[0][:160] if str(error) else type(error).__name__
                    reason = f"computing the structure's build ran out of CUDA memory ({why})"
                if starved:  # outside the except block: its traceback would pin the scratch builders
                    self.close()
                    reason += "; the kept engines were released"
                if relaxation is not None:
                    self._parts[key] = parts
                    self._out[id(relaxation)] = key
                    self.last = {"reused": True, "reason": reason}
                    return relaxation
        self.last = {"reused": False, "reason": reason}
        from .multi_system_relax import MultiSystemRelaxation

        failed = False
        try:
            relaxation = MultiSystemRelaxation(model, structures, **options)
        except Exception as error:
            if not (self._kept and _out_of_memory(error)):
                raise
            failed = True
        if failed:  # outside the except block (its traceback would pin the failed capture's memory): without them
            self.close()
            self.last["reason"] += "; the kept engines were released (the construction ran out of CUDA memory)"
            relaxation = MultiSystemRelaxation(model, structures, **options)
        if key is not None:
            self._parts[key] = parts
            self._out[id(relaxation)] = key
        return relaxation

    def _restarted(self, key: tuple, parts: dict, structure) -> tuple[MultiSystemRelaxation | None, str]:
        """A kept relaxation of ``key`` restarted with ``structure`` and why, or None and why not."""
        same = [entry for entry in self._kept if entry[0] == key]
        if not same:
            if key in self._declined:
                return None, self._declined[key]
            if not self._kept:
                return None, "no engine kept yet"
            # what differs from the kept key closest to this one (the most recently used of the closest)
            differ = min(([name for name in parts if parts[name] != self._parts[entry[0]].get(name)]
                          for entry in reversed(self._kept)), key=len)
            names = (_PARTS.get(name, name.removeprefix("option ")) for name in differ)
            return None, f"no kept engine of this {', '.join(names)}"
        try:  # every relaxation of the key computes the same: its arguments are the key's
            signature = self._kept[same[-1]].fresh_signature(structure)
        except ValueError as error:  # the construction checks it the same way, and fails
            return None, f"the structure fails a check of the construction: {str(error)[:200]}"
        entry = (key, signature)
        relaxation = self._kept.get(entry)
        if relaxation is None:
            kept = ", ".join(str(other[1][1:5]) for other in same)
            return None, (f"no kept engine of this structure's capacities and sort keys {signature[1:5]} (pairs, "
                          f"triplets, key dtype, image bound; kept: {kept})" if all(
                              other[1][5] == signature[5] for other in same) else
                          "the model's automatic recompute state changed since the kept engines' captures")
        del self._kept[entry]
        held = self._held.pop(entry)
        try:
            relaxation.restart(structure, signature)
        except ValueError as error:  # its checks, before any change: kept for later structures
            self._kept[entry], self._held[entry] = relaxation, held
            return None, f"the structure fails a check of the construction: {str(error)[:200]}"
        except Exception as error:  # after its changes (a new runner's checks): released
            relaxation.release()
            return None, f"the kept engine could not take the structure ({str(error)[:200]}); released"
        except BaseException:
            relaxation.release()
            raise
        return relaxation, ("the engine of an earlier call (the same model, task, configuration, options and atom "
                            "count; the same capacities, sort keys and recompute state)")

    def give_back(self, relaxation: MultiSystemRelaxation, *, keep: bool) -> None:
        """After ``relaxation``'s run (or its failure): kept where ``keep`` (a run that finished, every structure
        converged or at its step limit), it came from :meth:`take` with a key and its graph pool fits the memory
        budget (``memory_fraction`` of the device's memory), the least recently used ones beyond ``limit`` or the
        budget released; else released."""
        key = self._out.pop(id(relaxation), None)
        signature = held = None
        if keep and key is not None and self.limit > 0:
            try:
                signature = relaxation.build_signature()
            except ValueError:  # e.g. its union split
                signature = None
        if signature is not None:
            held = relaxation.graph_pool_bytes()
            device = torch.cuda.get_device_properties(torch.cuda.current_device())
            budget = self.memory_fraction * device.total_memory
            if held is None:  # the allocator cannot measure the pool (backend:cudaMallocAsync): not kept
                self._declined[key] = "not kept after an earlier call: the CUDA allocator cannot measure its graph pool"
                signature = None
            elif held > budget:
                self._declined[key] = (f"not kept after an earlier call: its CUDA graph pool of {held / 2**30:.1f} "
                                       f"GiB exceeds the kept engines' {budget / 2**30:.1f} GiB "
                                       f"({self.memory_fraction:.3g} of the device's memory)")
                signature = None
            while len(self._declined) > _DECLINED_KEYS:
                self._declined.popitem(last=False)
        if signature is None:
            relaxation.release()
            self._prune_parts()
            return
        self._declined.pop(key, None)
        entry = (key, signature)
        replaced = self._kept.pop(entry, None)
        self._held.pop(entry, None)
        if replaced is not None and replaced is not relaxation:
            replaced.release()
        self._kept[entry], self._held[entry] = relaxation, held
        while len(self._kept) > self.limit or sum(self._held.values()) > budget:
            evicted_entry, evicted = self._kept.popitem(last=False)
            del self._held[evicted_entry]
            evicted.release()
        self._prune_parts()

    def _prune_parts(self) -> None:
        """Only the parts of the keys in use: the kept relaxations' and those taken and not given back yet."""
        live = {entry[0] for entry in self._kept} | set(self._out.values())
        if len(self._parts) > len(live):
            self._parts = {name: parts for name, parts in self._parts.items() if name in live}

    def close(self) -> None:
        """Release every kept relaxation and give their CUDA memory back to the device."""
        kept, self._kept, self._held = list(self._kept.values()), OrderedDict(), {}
        try:
            for relaxation in kept:
                relaxation.release()
        finally:
            self._prune_parts()
            if kept:
                del kept
                gc.collect()
                torch.cuda.empty_cache()
