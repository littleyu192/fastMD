"""Conservative, process-local reuse of measured CUDA memory bounds.

Entries hold numbers and metadata only; weak model keys never retain GPU weights.
A fit is tied to the same adapter/model, parameter and buffer mutation versions,
configuration, task, precision, and exact host geometry. In particular, a fit's
edge-capacity range does not establish a bound for a different number of atoms
or a different candidate list. No geometry-dependent costs are extrapolated.

As with the runtime's projection caches, unsupported writes through ``.data``
without a tensor version bump are outside the immutable inference contract.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading
from weakref import WeakKeyDictionary

import numpy as np
import torch


_CACHE = WeakKeyDictionary()
_LOCK = threading.RLock()
_MAX_ENTRIES = 32


def _tensor_stamp(value):
    return (id(value), value._version, tuple(value.shape), tuple(value.stride()),
            str(value.dtype), str(value.device), value.requires_grad, value.data_ptr())


def _model_stamp(model):
    def program(module):
        forward = module.forward
        forward = getattr(forward, "__func__", forward)
        hooks = tuple((name, tuple((key, id(hook)) for key, hook in getattr(module, name, {}).items()))
                      for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks"))
        return id(module), type(module), module.training, id(forward), hooks

    return (
        tuple((name, _tensor_stamp(value)) for name, value in model.named_parameters()),
        tuple((name, _tensor_stamp(value)) for name, value in model.named_buffers()),
        tuple((name, program(module)) for name, module in model.named_modules()),
    )


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", _tensor_stamp(value))
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(part)) for key, part in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(part) for part in value)
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    raise TypeError("Memory-probe cache requires explicit immutable system inputs")


@dataclass(frozen=True)
class ProbeCacheKey:
    model_state: tuple
    scope: tuple
    geometry: str


def probe_cache_key(adapter, atoms, config, *, task, device):
    """A cache key without device reads, or None when its provenance is unsupported."""
    try:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        digest = hashlib.sha256()
        for value, dtype in ((atoms.numbers, "<i8"), (atoms.positions, "<f8"),
                             (atoms.cell.array, "<f8"), (atoms.pbc, "?")):
            array = np.asarray(value, dtype=dtype)
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes(order="C"))
        system_inputs = getattr(adapter, "system_inputs", lambda: {})()
        from fastmd.adapters.patches import patch_report
        patches = patch_report(adapter.upstream_patches()) if hasattr(adapter, "upstream_patches") else []
        scope = (id(adapter), type(adapter), str(device), task, config.fingerprint,
                 str(torch.get_default_dtype()), torch.is_autocast_enabled(device.type),
                 str(torch.get_autocast_dtype(device.type)),
                 torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32, repr(adapter.topology_spec()),
                 repr(adapter.padding_rule()), repr(adapter.capture_contract()), _freeze(system_inputs),
                 _freeze(patches))
        return ProbeCacheKey(_model_stamp(adapter.module), scope, digest.hexdigest())
    except (AttributeError, RuntimeError, TypeError):
        # For example, inference tensors without version counters or custom
        # system-input objects: an uncached measurement is still safe.
        return None


def cached_memory(model, key, target_edges, target_triplets=None):
    """Return only a valid fit that still bounds the requested capacity."""
    if key is None:
        return None
    try:
        if _model_stamp(model) != key.model_state:
            return None
    except (RuntimeError, TypeError):
        return None
    with _LOCK:
        entries = _CACHE.get(model)
        if not entries:
            return None
        for stored_key in reversed(entries):
            if stored_key[0] != key:
                continue
            memory = entries[stored_key]
            if (memory.valid and memory.triplets == (target_triplets is not None)
                    and memory.bound("capture", target_edges, target_triplets or 0) is not None):
                entries.move_to_end(stored_key)
                return memory
    return None


def remember_memory(model, key, memory, target_edges, target_triplets=None):
    """Retain successful fits only, after verifying the model stayed unchanged."""
    if key is None or not memory.valid:
        return
    try:
        if _model_stamp(model) != key.model_state:
            return
    except (RuntimeError, TypeError):
        return
    with _LOCK:
        entries = _CACHE.setdefault(model, OrderedDict())
        stored_key = (key, int(target_edges), target_triplets)
        entries[stored_key] = memory
        entries.move_to_end(stored_key)
        while len(entries) > _MAX_ENTRIES:
            entries.popitem(last=False)
