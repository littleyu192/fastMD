"""Helpers the model adapters share: constant declarations, checkpoint hashes, load options, row-wise shifts.

:func:`declares` turns a model's constant declarations into the protocol methods the engine reads. It
installs only the methods it is given, so ``isinstance`` against the runtime-checkable protocols (which the
negotiation uses to tell "not declared" from a declaration) sees the same members as hand-written methods.
"""
from __future__ import annotations

import hashlib
import os

import torch
from torch import Tensor

from fastmd.adapters.patches import PatchRecord

_METHODS = {"padding": "padding_rule", "isolated": "isolated_atom_policy", "capacity": "capacity_step",
            "edge_shifts": "edge_shift_input", "batching": "system_batching", "references": "reference_energies"}


def _constant(name: str, value):
    def method(self):
        return value
    method.__name__ = method.__qualname__ = name
    return method


def declares(*, kernels=None, patches: tuple[str, ...] = (), **constants):
    """A class decorator: a method per declared constant (``padding`` -> ``padding_rule``, ``isolated`` ->
    ``isolated_atom_policy``, ``capacity`` -> ``capacity_step``, ``edge_shifts`` -> ``edge_shift_input``,
    ``batching`` -> ``system_batching``, ``references`` -> ``reference_energies``); ``kernels`` (a
    :class:`KernelChoice`) -> ``kernel_backends``; ``patches`` (attribute names of outcome lists) ->
    ``upstream_patches``, their concatenation. Declaring a method the class also defines raises
    ``TypeError`` rather than overwriting either definition."""
    def decorate(cls):
        methods = {_METHODS[key]: _constant(_METHODS[key], value) for key, value in constants.items()}
        if kernels is not None:
            methods["kernel_backends"] = _constant("kernel_backends", (kernels,))
        if patches:
            def upstream_patches(self) -> list:
                return [outcome for name in patches for outcome in getattr(self, name)]
            methods["upstream_patches"] = upstream_patches
        for name, method in methods.items():
            if name in vars(cls):
                raise TypeError(f"{cls.__name__}.{name} is both declared and defined")
            method.__qualname__ = f"{cls.__qualname__}.{name}"
            setattr(cls, name, method)
        return cls
    return decorate


def device_of(device: str | None) -> torch.device:
    """``device``, or CUDA where available (else the CPU) for None."""
    return torch.device("cuda" if device is None and torch.cuda.is_available() else (device or "cpu"))


def frozen(module):
    """``module`` in eval mode with its parameters frozen (the module itself)."""
    module = module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def dtype_name(module) -> str:
    """The dtype of ``module``'s parameters without the ``torch.`` prefix ("float32", "float64")."""
    return str(next(module.parameters()).dtype).removeprefix("torch.")


def times_cell(image: Tensor, cell: Tensor) -> Tensor:
    """``image @ cell`` for a ``[3, 3]`` cell, row by row: no TF32 GEMM, and an edge's arithmetic does not
    depend on the row count."""
    return image[:, 0:1] * cell[0] + image[:, 1:2] * cell[1] + image[:, 2:3] * cell[2]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(family: str, name: str, path, expected: str) -> dict:
    """The checkpoint record (name, path, sha256) of a pinned file; raises if its sha256 is not ``expected``."""
    found = file_sha256(path)
    if found != expected:
        raise ValueError(f"{family} checkpoint {name!r} at {path}: sha256 {found} is not the pinned {expected}")
    return {"name": name, "path": str(path), "sha256": found}


def local_file(family: str, path: str, hint: str = "", *, refuse: bool = False) -> dict:
    """The checkpoint record of an unpinned local file; raises if it does not exist (or ``refuse``)."""
    if refuse or not os.path.isfile(path):
        raise FileNotFoundError(f"{family} checkpoint not found: {path!r}{hint}")
    return {"name": os.path.basename(path), "path": str(path), "sha256": file_sha256(path)}


def check_options(optimization_profile, legacy: str, *, model_dtype=None, float64: str | None = None,
                  model_path=None, path_given: str | None = None, path_missing: str | None = None) -> None:
    """The load-option checks in ``load_adapter``'s order: the "legacy" profile (MatRIS's: ``legacy`` is the
    message), then float64, then the model path."""
    from fastmd.adapters.protocol import NegotiationError

    if optimization_profile == "legacy":
        raise NegotiationError(legacy)
    if float64 is not None and model_dtype == "float64":
        raise ValueError(float64)
    if path_given is not None and model_path is not None:
        raise ValueError(path_given)
    if path_missing is not None and model_path is None:
        raise ValueError(path_missing)


def recorder(distribution: str, versions: tuple[str, ...], *, name_prefix: str = "", target_prefix: str = ""):
    """A :class:`PatchRecord` factory of one upstream distribution: ``(name, target, sha256, reason, **fields)``."""
    def record(name: str, target: str, sha256: str, reason: str, **fields) -> PatchRecord:
        return PatchRecord(name=name_prefix + name, target=target_prefix + target, distribution=distribution,
                           versions=versions, source_sha256=sha256, reason=reason, **fields)
    return record
