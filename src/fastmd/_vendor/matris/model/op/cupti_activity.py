"""Minimal CUPTI activity timing helpers for local autotune.

The implementation intentionally uses CUPTI Activity records directly through
``ctypes`` instead of ``torch.profiler`` so tactic selection can measure kernel
GPU timestamps without CUDA event perturbation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import site
import threading
from typing import Callable

import torch


CUPTI_SUCCESS = 0
CUPTI_ERROR_MAX_LIMIT_REACHED = 12
CUPTI_ERROR_INVALID_KIND = 21
CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1

_KERNEL_KIND_OFFSET = 0
_KERNEL_START_OFFSET = 16
_KERNEL_END_OFFSET = 24
_BUFFER_SIZE = 4 * 1024 * 1024


class CuptiActivityError(RuntimeError):
    pass


def _candidate_libraries() -> list[str]:
    candidates = []
    found = ctypes.util.find_library("cupti")
    if found:
        candidates.append(found)
    for root in site.getsitepackages():
        candidates.extend(
            glob.glob(os.path.join(root, "nvidia", "**", "libcupti.so*"), recursive=True)
        )
    candidates.extend(
        [
            "/usr/local/cuda/targets/sbsa-linux/lib/libcupti.so",
            "/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcupti.so",
            "/usr/local/cuda-13/targets/sbsa-linux/lib/libcupti.so",
        ]
    )
    deduped = []
    for path in candidates:
        if path and path not in deduped and (path.startswith("lib") or os.path.exists(path)):
            deduped.append(path)
    return deduped


class _CuptiActivity:
    def __init__(self) -> None:
        self._lib = self._load_library()
        self._records_ns: list[int] = []
        self._buffers: list[ctypes.Array] = []
        self._lock = threading.Lock()
        self._collecting = False
        self._callback_error: str | None = None
        self._setup_api()
        self._register_callbacks()

    def _load_library(self):
        errors = []
        for path in _candidate_libraries():
            try:
                return ctypes.CDLL(path)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        raise CuptiActivityError("failed to load libcupti.so: " + "; ".join(errors))

    def _setup_api(self) -> None:
        self._lib.cuptiGetResultString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._lib.cuptiGetResultString.restype = ctypes.c_int
        self._lib.cuptiActivityRegisterCallbacks.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        self._lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
        self._lib.cuptiActivityEnable.restype = ctypes.c_int
        self._lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
        self._lib.cuptiActivityDisable.restype = ctypes.c_int
        self._lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        self._lib.cuptiActivityFlushAll.restype = ctypes.c_int
        self._lib.cuptiActivityGetNextRecord.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.cuptiActivityGetNextRecord.restype = ctypes.c_int

    def _result_string(self, result: int) -> str:
        value = ctypes.c_char_p()
        status = self._lib.cuptiGetResultString(result, ctypes.byref(value))
        if status == CUPTI_SUCCESS and value.value:
            return value.value.decode("utf-8", "replace")
        return f"CUPTI result {result}"

    def _check(self, result: int, what: str) -> None:
        if result != CUPTI_SUCCESS:
            raise CuptiActivityError(f"{what} failed: {self._result_string(result)}")

    def _register_callbacks(self) -> None:
        request_cb_t = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        complete_cb_t = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
        )

        def request(
            buffer: ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            size: ctypes.POINTER(ctypes.c_size_t),
            max_num_records: ctypes.POINTER(ctypes.c_size_t),
        ) -> None:
            storage = ctypes.create_string_buffer(_BUFFER_SIZE)
            self._buffers.append(storage)
            buffer[0] = ctypes.cast(storage, ctypes.POINTER(ctypes.c_uint8))
            size[0] = _BUFFER_SIZE
            max_num_records[0] = 0

        def complete(
            _context: ctypes.c_void_p,
            _stream_id: int,
            buffer: ctypes.POINTER(ctypes.c_uint8),
            _size: int,
            valid_size: int,
        ) -> None:
            if not self._collecting or valid_size == 0:
                return
            try:
                record = ctypes.c_void_p()
                while True:
                    status = self._lib.cuptiActivityGetNextRecord(
                        buffer, valid_size, ctypes.byref(record)
                    )
                    if status == CUPTI_SUCCESS:
                        address = record.value
                        if not address:
                            continue
                        kind = ctypes.c_uint32.from_address(address + _KERNEL_KIND_OFFSET).value
                        if kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                            start = ctypes.c_uint64.from_address(address + _KERNEL_START_OFFSET).value
                            end = ctypes.c_uint64.from_address(address + _KERNEL_END_OFFSET).value
                            if end > start:
                                with self._lock:
                                    self._records_ns.append(end - start)
                        continue
                    if status in (CUPTI_ERROR_MAX_LIMIT_REACHED, CUPTI_ERROR_INVALID_KIND):
                        break
                    self._callback_error = self._result_string(status)
                    break
            except Exception as exc:  # callbacks cannot raise into CUPTI safely
                self._callback_error = repr(exc)

        self._request_cb = request_cb_t(request)
        self._complete_cb = complete_cb_t(complete)
        self._check(
            self._lib.cuptiActivityRegisterCallbacks(self._request_cb, self._complete_cb),
            "cuptiActivityRegisterCallbacks",
        )

    def profile_ms(self, fn: Callable[[], None], repeat: int) -> float:
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        torch.cuda.synchronize()
        self._records_ns.clear()
        self._callback_error = None
        self._collecting = True
        self._check(
            self._lib.cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
            "cuptiActivityEnable(CONCURRENT_KERNEL)",
        )
        try:
            for _ in range(repeat):
                fn()
            torch.cuda.synchronize()
            self._check(
                self._lib.cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
                "cuptiActivityFlushAll",
            )
        finally:
            self._collecting = False
            self._lib.cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
        if self._callback_error is not None:
            raise CuptiActivityError(self._callback_error)
        if not self._records_ns:
            raise CuptiActivityError("no CUPTI kernel activity records collected")
        return sum(self._records_ns) / repeat / 1.0e6


_CUPTI_ACTIVITY: _CuptiActivity | None = None


def profile_kernel_ms(fn: Callable[[], None], repeat: int) -> float:
    global _CUPTI_ACTIVITY
    if _CUPTI_ACTIVITY is None:
        _CUPTI_ACTIVITY = _CuptiActivity()
    return _CUPTI_ACTIVITY.profile_ms(fn, repeat)
