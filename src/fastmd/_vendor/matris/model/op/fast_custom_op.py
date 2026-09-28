# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small low-overhead wrapper for Python-backed torch custom ops."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Union

import torch
from torch.library import Library, infer_schema, register_fake


_LIBS: dict[tuple[str, str], Library] = {}


def _get_library(namespace: str, kind: str = "FRAGMENT") -> Library:
    key = (namespace, kind)
    lib = _LIBS.get(key)
    if lib is None:
        lib = Library(namespace, kind)
        _LIBS[key] = lib
    return lib


def fast_custom_op(
    qualname: str,
    *,
    mutates_args: Union[Iterable[str], str] = (),
    device_types: str | tuple[str, ...] = "CUDA",
) -> Callable[[Callable], "FastCustomOp"]:
    """Register ``fn`` via ``Library.define + impl`` and return a callable op."""
    if "::" not in qualname:
        raise ValueError(f"qualname must be '<namespace>::<op>', got {qualname!r}")
    namespace, op_name = qualname.split("::", 1)
    mutates = mutates_args if isinstance(mutates_args, str) else tuple(mutates_args)
    dev_types = (device_types,) if isinstance(device_types, str) else tuple(device_types)

    def decorator(fn: Callable) -> "FastCustomOp":
        schema = infer_schema(fn, op_name=op_name, mutates_args=mutates)
        lib = _get_library(namespace)
        lib.define(schema)
        for device_type in dev_types:
            lib.impl(op_name, fn, device_type)
        return FastCustomOp(qualname, namespace, op_name, fn)

    return decorator


class FastCustomOp:
    __slots__ = ("qualname", "namespace", "op_name", "_python_fn", "_op")

    def __init__(self, qualname: str, namespace: str, op_name: str, python_fn: Callable):
        self.qualname = qualname
        self.namespace = namespace
        self.op_name = op_name
        self._python_fn = python_fn
        self._op = getattr(getattr(torch.ops, namespace), op_name)

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    def register_fake(self, fake_fn: Callable) -> Callable:
        register_fake(self.qualname, fake_fn)
        return fake_fn

    @property
    def python_impl(self) -> Callable:
        return self._python_fn
