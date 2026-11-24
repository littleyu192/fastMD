"""Finite-timeout process groups for one process per GPU (or CPU tests)."""
from __future__ import annotations

from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from typing import NoReturn

import torch
import torch.distributed as dist

from .exchange import ExchangeContext


def initialize(*, device: str = "cuda", timeout_seconds: int = 120) -> ExchangeContext:
    """Initialize torchrun workers. Fatal worker failures are handled by torchrun.

    Callers that already own process groups should construct ExchangeContext
    directly and retain responsibility for group teardown and their watchdog.
    """
    if dist.is_initialized():
        raise RuntimeError("a process group already exists; pass an ExchangeContext instead")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if device == "cuda":
        # The GPU is the node-local rank's (RANK spans every node).
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not 0 <= local_rank < torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK {local_rank} has no GPU: {torch.cuda.device_count()} visible "
                             f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); "
                             "launch one process per GPU and node")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    elif device == "cpu":
        backend = "gloo"
    else:
        raise ValueError("device must be 'cuda' or 'cpu'")
    if "RANK" not in os.environ:
        return ExchangeContext()
    timeout = timedelta(seconds=timeout_seconds)
    dist.init_process_group(backend, timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout) if backend == "nccl" else None
    return ExchangeContext(control_group=control)


def shutdown() -> None:
    """Destroy groups; do not add a barrier that can hang after another rank fails.

    Success path only. After any failure use fail_fast(): destroy_process_group
    was measured to block indefinitely behind a rank that had raised. Everything
    in the shutdown registry is released first (``fastmd.distributed.shutdown_registry``:
    open evaluators and topologies, whose captured graphs hold NCCL work, then
    hooks).
    """
    if dist.is_initialized():
        from .shutdown_registry import release_all
        release_all()
        dist.destroy_process_group()


def _abort_all() -> None:
    c10d = dist.distributed_c10d
    if hasattr(c10d, "_abort_process_group"):
        # Aborts all groups (NCCL inside one group start/end); Gloo is accepted.
        c10d._abort_process_group()
        return
    for group in list(getattr(getattr(c10d, "_world", None), "pg_map", {})):
        if hasattr(group, "abort"):
            group.abort()


def abort(timeout: float | None = None) -> bool:
    """Abort every process group without synchronizing with peers or streams.

    Returns False if the abort is still running after ``timeout`` seconds. An
    NCCL abort was measured to block for more than 20 s once collectives are
    captured in a live CUDA graph, so callers about to exit must not wait on it.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True
    failures = []

    def run():
        try:
            _abort_all()
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=run, name="matris-process-group-abort", daemon=True)
    worker.start()
    worker.join(timeout)
    if failures:
        raise failures[0]
    return not worker.is_alive()


_failing = threading.Lock()


class PeerFailedError(RuntimeError):
    """Another rank's fail_fast() announced a failure (a secondary error on this rank)."""


# fail_fast() records the first failing rank under this key of the default
# process-group store before its abort. A peer blocked in a Gloo collective
# fails on the dead rank's closed connection, but a peer blocked in an NCCL
# wait (every "nccl" control-plane round, payload exchanges, synchronize) is
# never released by it: its progress watchdog polls the key (PeerFailureProbe)
# and fails fast within one poll, whatever the launcher.
FAILURE_KEY = "fastmd.distributed.fail_fast"
_ANNOUNCE_SECONDS = 2.0
# Store calls of the watchdog's probe; a store that stopped answering then
# disables the probe instead of stalling the deadline checks for the PG timeout.
_PROBE_STORE_SECONDS = 5.0
# The rank serving the store's TCP server takes it down when it exits, before
# peers polling once per second (the CLI watchdog) may have read its
# announcement (measured: rank 0 exited 0.7 s after it under a custom
# launcher); so does a multi-node torchrun's node-0 agent once its failed
# worker exited. After announcing, such a rank stays up this long.
_HOST_LINGER_SECONDS = 2.0


def _serves_store() -> bool:
    """Whether this process's exit may take down the default store before peers read it.

    Rank 0 of an env:// or tcp:// rendezvous serves it. torchrun's workers are
    clients of the store of the agent on node GROUP_RANK 0
    (TORCHELASTIC_USE_AGENT_STORE); that agent exits once a local worker
    failed and it stopped the others, possibly before other nodes' ranks
    polled the announcement, so a multi-node job's node-0 workers linger too.
    A FileStore has no server.
    """
    try:
        if os.environ.get("TORCHELASTIC_USE_AGENT_STORE", "").lower() == "true":
            return (os.environ.get("GROUP_RANK") == "0"
                    and int(os.environ.get("GROUP_WORLD_SIZE", "1")) > 1)
        store = dist.distributed_c10d._get_default_store()
        while isinstance(store, dist.PrefixStore):
            store = store.underlying_store
        return isinstance(store, dist.TCPStore) and dist.get_rank() == 0
    except Exception:
        return False


def _announce_failure(rank, seconds: float = _ANNOUNCE_SECONDS) -> None:
    """Best effort within ``seconds``: record ``rank`` under FAILURE_KEY unless a rank already is."""
    if not (dist.is_available() and dist.is_initialized()):
        return

    def run():
        try:
            # Its own connection: never queued behind another thread's store
            # call (e.g. a main thread blocked in a rendezvous with a dead peer).
            store = dist.distributed_c10d._get_default_store().clone()
            store.compare_set(FAILURE_KEY, "", str(rank))  # the first failing rank wins
        except BaseException:
            pass  # e.g. a store without clone(), or its host already exited

    try:
        worker = threading.Thread(target=run, name="matris-fail-fast-announce", daemon=True)
        worker.start()
        worker.join(seconds)
    except BaseException:
        pass  # the error path must go on to the abort and exit


class PeerFailureProbe:
    """Another rank's fail_fast() announcement, polled by the progress watchdog thread.

    ``check()`` is one store lookup on this probe's own clone of the default
    process-group store (made at the first poll after the group exists). Any
    store error disables the probe: no store, a store without clone(), or a
    store host that exited (a custom launcher's rank 0 hosts it and may exit
    first on success) is no evidence of a failure. The progress deadline
    still applies, and without an announcement nothing changes.
    """

    def __init__(self, rank: int | None):
        self.rank = rank
        self._source = self._store = None
        self._disabled = False

    def check(self) -> str | None:
        """The announcing rank's message, or None."""
        if self._disabled or not (dist.is_available() and dist.is_initialized()):
            return None
        try:
            source = dist.distributed_c10d._get_default_store()
            if source is not self._source:
                self._source, self._store = source, source.clone()
                self._store.set_timeout(timedelta(seconds=_PROBE_STORE_SECONDS))
            if not self._store.check([FAILURE_KEY]):
                return None
            failed = self._store.get(FAILURE_KEY).decode()
        except Exception:
            self._disabled = True
            self._source = self._store = None
            return None
        if failed == str(self.rank):
            return None  # this rank's own fail_fast is already exiting
        return f"rank {failed} failed (fail_fast); stopping instead of waiting on it"


def _exit_later(seconds: float) -> bool:
    """Arm faulthandler's C-level timer: dump every stack, then exit(1), after ``seconds``.

    It fires even while a hang holds the GIL. sys.stderr may have no file
    descriptor (redirect_stderr, notebooks, logging wrappers); the dump then
    goes to fd 2. Returns False if no timer could be armed.
    """
    for file in (sys.stderr, 2):
        try:
            faulthandler.dump_traceback_later(seconds, exit=True, file=file)
            return True
        except (AttributeError, OSError, RuntimeError, ValueError):
            pass
    return False


def _report(text: str) -> None:
    """Best effort: a broken sys.stderr must not stop the error path."""
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except BaseException:
        pass


def _write_failure(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)


def fail_fast(error: BaseException, *, rank: int | None = None, phase: str | None = None,
              output: str | os.PathLike | None = None, stacks: str | None = None,
              abort_seconds: float = 5.0, grace_seconds: float = 30.0) -> NoReturn:
    """Worker error path: report, abort process groups, then ``os._exit(1)``.

    Prints the traceback prefixed with the rank, flushes, and optionally writes
    ``<output>.rank<r>.error.json`` (plus ``stacks``, e.g. every thread's stack
    from the progress watchdog). It deliberately performs no CUDA
    synchronize, graph reset, gc or destroy_process_group, any of which can wait
    forever on a peer that has already failed; torchrun then stops the other
    workers of its node. Before the abort it announces the failure in the
    process-group store (FAILURE_KEY, at most ~2 s): peers' progress watchdogs
    then fail fast with PeerFailedError even under a launcher that does not stop
    them (or on another node), and even while they wait in NCCL; a rank serving
    the store (see _serves_store) stays up ~2 s after announcing, so they can
    read it. The abort gets ``abort_seconds``; a C-level
    faulthandler timer exits after ``grace_seconds`` if anything else stalls
    (e.g. holding the GIL).
    Callable from any thread; never returns, even if reporting itself fails.
    """
    if not _failing.acquire(blocking=False):
        # Another thread is already reporting; its os._exit ends this process.
        time.sleep(grace_seconds)
        os._exit(1)
    try:
        failed_at = time.time()
        if not _exit_later(grace_seconds):
            # Last resort; unlike the faulthandler timer it needs the GIL.
            timer = threading.Timer(grace_seconds, os._exit, (1,))
            timer.daemon = True
            timer.start()
        try:
            initialized = dist.is_available() and dist.is_initialized()
            if rank is None:
                rank = dist.get_rank() if initialized else int(os.environ.get("RANK", "0"))
            world_size = dist.get_world_size() if initialized else int(os.environ.get("WORLD_SIZE", "1"))
        except BaseException:
            world_size = None
        prefix = f"[rank {rank}] "
        text = message = ""
        try:
            text = "".join(traceback.format_exception(error))
            message = str(error)
            sys.stdout.flush()
        except BaseException:
            pass
        _report("".join(prefix + line for line in (
            f"fatal error in phase {phase!r}; aborting process groups and exiting\n",
            *text.splitlines(True))))
        if output is not None:
            path = f"{output}.rank{rank}.error.json"
            try:
                record = {"rank": rank, "world_size": world_size, "phase": phase,
                          "type": f"{type(error).__module__}.{type(error).__qualname__}",
                          "message": message, "traceback": text, "time_unix": failed_at,
                          "host": socket.gethostname(), "pid": os.getpid()}
                if stacks is not None:
                    record["stacks"] = stacks
                _write_failure(Path(path), record)
            except BaseException as exc:
                _report(f"{prefix}could not write {path}: {exc!r}\n")
        # After the record: a peer's secondary record is always the later one.
        serving = _serves_store()
        _announce_failure(rank)
        announced = time.monotonic()
        try:
            if not abort(abort_seconds):
                _report(f"{prefix}process-group abort still running after "
                        f"{abort_seconds:g} s; exiting without it\n")
        except BaseException as exc:
            _report(f"{prefix}process-group abort failed: {exc!r}\n")
        try:
            sys.stdout.flush()
        except BaseException:
            pass
        if serving:
            # Peers read the announcement from this process's store server.
            time.sleep(max(0.0, _HOST_LINGER_SECONDS - (time.monotonic() - announced)))
    finally:
        os._exit(1)
