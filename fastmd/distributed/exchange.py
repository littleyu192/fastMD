"""Explicit row COPY plans and their scatter-SUM adjoints.

Only requested tensor rows cross the data process group. Key-based plans
(``ExchangePlan.build``, a test oracle since step 4.2) build their owner
directory from CPU metadata replicated through a Gloo control group. Numeric
owner-row plans, the production path, instead exchange only requested integer
row indices; they need no replicated key directory.
Process-group creation, finite timeouts, and whole-job termination belong to the
launcher, not to this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from numbers import Integral
import time
from typing import Any, Hashable, Sequence

import numpy as np
import torch

from fastmd.engine import saved_recompute as _saved_recompute
from fastmd.precision import deterministic_mode
import torch.distributed as dist
from torch import Tensor

from fastmd._nvtx import nvtx_range
from fastmd.engine import recorder as _recorder
from fastmd.engine.comm import LocalCommunicator

from .config import check_control_plane


class ExchangeContext:
    """Data/control groups with collective, rank-consistent validation.

No initialized process group means a normal local, one-rank execution.  NCCL
payloads require a separate Gloo control group with the same rank ordering.
Every rank must enter methods in the same order, including ranks with no rows.
At P>1 every control round travels through the DomainCommunicator's
fixed-shape rounds on the data group (fastmd.distributed.communicator), from
construction on; on NCCL it binds ``device`` or else the current CUDA device,
so set the device before constructing. ``control_plane`` accepts only
"nccl" ("gloo", the pre-3.1a rollback, is rejected for one version).
"""

    def __init__(self, group=None, control_group=None, device=None, *, control_plane: str = "nccl"):
        check_control_plane(control_plane)
        self.device = None if device is None else torch.device(device)
        self._sequence = 0
        self._plan_sequence = 0
        self._domain = None
        # Opt-in diagnostics (tests, the P > 4 oracle): a list collects
        # (plan name, plan sequence, reverse) of every payload exchange.
        self.exchange_trace = None
        initialized = dist.is_available() and dist.is_initialized()
        if not initialized:
            if group is not None or control_group is not None:
                raise ValueError("process groups were supplied before initialization")
            self.group = self.control_group = None
            self.rank, self.world_size, self.backend = 0, 1, "local"
        else:
            self.group = dist.group.WORLD if group is None else group
            self.rank = dist.get_rank(self.group)
            self.world_size = dist.get_world_size(self.group)
            self.backend = str(dist.get_backend(self.group)).lower()
            if self.backend not in {"gloo", "nccl"}:
                raise ValueError(f"unsupported data backend: {self.backend}")
            if control_group is None:
                if self.backend == "nccl" and self.world_size > 1:
                    raise ValueError("NCCL requires an explicit Gloo control_group")
                control_group = self.group if self.backend == "gloo" else None
            self.control_group = control_group
            if control_group is not None:
                if str(dist.get_backend(control_group)).lower() != "gloo":
                    raise ValueError("control_group must use Gloo")
                if (dist.get_world_size(control_group), dist.get_rank(control_group)) != (
                    self.world_size, self.rank
                ):
                    raise ValueError("data/control groups must have identical rank ordering")
                if dist.get_process_group_ranks(control_group) != dist.get_process_group_ranks(self.group):
                    raise ValueError("data/control groups must have identical members")
        # P=1: the protocol's local implementation; P>1: the DomainCommunicator
        # (fixed-shape rounds on the data group, Gloo only for slow paths).
        if self.world_size > 1:
            from .communicator import DomainCommunicator
            self._domain = DomainCommunicator(self.group, self.control_group,
                                              device=self.device if self.backend == "nccl" else None)
        self.comm = LocalCommunicator() if self.world_size == 1 else self._domain
        self.reset_stats()
        self.set_payload_transport("auto")

    def set_payload_transport(self, mode: str) -> None:
        """Select how per-peer-split data-group exchanges travel; local, no communication.

        "all_to_all": one ``all_to_all_single`` per exchange (every rank pair,
        zero splits included); "neighbor_p2p": one grouped P2P over the
        peers with rows (``communicator.grouped_p2p_exchange``), so a spatial
        exchange touches only neighbor domains; "auto": all_to_all up to 4
        ranks, neighbor P2P above (the evaluator resolves "auto" from its
        geometry first: see communicator.resolve_payload_transport). Rows,
        their order and every sum are identical: only the transport
        differs. Every rank must switch at the same point of its call
        sequence (the evaluator agrees the mode in its contract first).
        """
        from .communicator import resolve_payload_transport
        self.payload_transport = resolve_payload_transport(mode, self.world_size)
        if not hasattr(self, "_p2p_ready"):
            self._p2p_ready = self.backend != "nccl"

    def all_to_all_single(self, output: Tensor, input: Tensor, output_split_sizes,
                          input_split_sizes, *, control: bool = False) -> None:
        """``dist.all_to_all_single`` on the data group, or its neighbor-P2P equivalent.

        ``control=True``: the same on the Gloo control group (CPU routing
        indices of the reference path); its calls count as control calls.

        Split sizes are dim-0 row counts per rank (both transports), already
        admitted collectively, so both ends of every pair agree. Every rank
        enters this call, so two NCCL cases keep the collective under
        "neighbor_p2p" with identical rows: the context's first NCCL payload
        (NCCL creates the communicator lazily at the group's first op, and a
        batched P2P as that op needs every rank), and CUDA graph capture
        (ranks capture the same exchanges together): P2P tensors are
        recorded on the NCCL stream, so the capture could not reuse any
        exchanged buffer (measured +460 MiB/rank at 1,344 atoms on 4 GPUs);
        torch's NCCL all_to_all moves no bytes for zero splits.
        """
        _saved_recompute.check_no_collective("all_to_all")
        group = self.control_group if control else self.group
        if control:
            started = time.perf_counter()
        if self.payload_transport == "all_to_all" or not control and (
                not self._p2p_ready
                or (self.backend == "nccl" and torch.cuda.is_current_stream_capturing())):
            self._p2p_ready = self._p2p_ready or not control
            dist.all_to_all_single(output, input, output_split_sizes=list(output_split_sizes),
                                   input_split_sizes=list(input_split_sizes), group=group)
            posted = None
        else:
            from .communicator import grouped_p2p_exchange
            posted = grouped_p2p_exchange(output, input, output_split_sizes, input_split_sizes,
                                          group=group, rank=self.rank)
        if control:
            self.stats["control_calls"] += 1
            self.stats["control_seconds"] += time.perf_counter() - started
        if posted is None:
            return
        self.stats["p2p_calls"] += 1
        self.stats["p2p_ops"] += posted
        self.p2p_peers.update(peer for peer in range(self.world_size) if peer != self.rank
                              and (output_split_sizes[peer] or input_split_sizes[peer]))
        self.stats["p2p_peers"] = len(self.p2p_peers)

    def set_control_plane(self, mode: str) -> None:
        """Kept one version: "nccl" is the only control plane (a no-op); "gloo" raises its removal."""
        check_control_plane(mode)

    def reset_stats(self) -> None:
        self.stats = {
            "payload_calls": 0,
            "payload_seconds": 0.0,
            "payload_sent_bytes": 0,
            "payload_received_bytes": 0,
            "peer_sent_bytes": [0] * self.world_size,
            "peer_received_bytes": [0] * self.world_size,
            "control_calls": 0,
            "control_seconds": 0.0,
            # P>1 only: data-group control rounds (an overflowing gather
            # counts two), its second rounds, pickled bytes this rank
            # contributed, and Gloo slow-path gathers.
            "control_rounds": 0,
            "control_overflow_rounds": 0,
            "control_bytes": 0,
            "control_slow_path_calls": 0,
            "routing_payload_calls": 0,
            "routing_payload_seconds": 0.0,
            "routing_payload_sent_bytes": 0,
            "routing_payload_received_bytes": 0,
            # "neighbor_p2p" payload transport only: grouped P2P calls, the
            # isend/irecv ops they posted, and how many distinct peers
            # carried rows (the peers themselves: ``p2p_peers``).
            "p2p_calls": 0,
            "p2p_ops": 0,
            "p2p_peers": 0,
        }
        self.p2p_peers = set()
        if self._domain is not None:
            self._domain.stats = self.stats

    def _collective(self, value: Any, tag: str) -> list[Any]:
        """Exchange CPU metadata and reject mismatched protocol phases."""
        _saved_recompute.check_no_collective("control")
        envelope = (self._sequence, tag, value)
        self._sequence += 1
        start = time.perf_counter()
        gathered = [envelope] if self.world_size == 1 else self.comm.gather_object(envelope)
        self.stats["control_calls"] += 1
        self.stats["control_seconds"] += time.perf_counter() - start
        # Against this rank's own (sequence, tag), not only each other: a
        # round released by a peer's abort can return stale rows that agree
        # with each other.
        phases = [(item[0], item[1]) for item in gathered]
        expected = envelope[:2]
        if any(phase != expected for phase in phases):
            raise RuntimeError(f"distributed protocol phase mismatch: {phases}")
        return [item[2] for item in gathered]

    def agree_flags(self, flags: Tensor | None = None, *, tag: str, host=()) -> Tensor:
        """ONE MAX round of rank-local flags: CPU int64 ``[*host, *flags]`` (Communicator.agree_flags_host).

        Replaces a per-check gather when only "did any rank fail" (or a
        max/min, via negated entries) must be agreed. One rank downloads its
        local vector.
        """
        start = time.perf_counter()
        with nvtx_range("agree_flags " + tag):
            agreed = self.comm.agree_flags_host(flags, phase=tag, host=host)
        self.stats["control_calls"] += 1
        self.stats["control_seconds"] += time.perf_counter() - start
        return agreed

    def all_gather_object(self, value: Any) -> list[Any]:
        """Gather routing/configuration metadata, never tensor features."""
        return self._collective(value, "metadata")

    def collective_check(self, error: str | None) -> None:
        errors = self._collective(error, "check")
        self._raise_errors(errors)

    @staticmethod
    def _raise_errors(errors: Sequence[str | None]) -> None:
        failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
        if failures:
            raise RuntimeError("distributed validation failed: " + "; ".join(failures))

    def sum_int(self, value: int) -> int:
        return sum(self.all_gather_object(int(value)))


def _key(value: Any, cache: dict | None = None) -> Hashable:
    # Tuple-ize nested list IDs, but reject tensors: IDs must stay on the CPU
    # metadata path and never accidentally serialize coordinate/feature tensors.
    if isinstance(value, Tensor):
        raise TypeError("row keys must be CPU Python values, not tensors")
    if isinstance(value, (list, tuple)):
        cache = {} if cache is None else cache
        identity = id(value)
        if identity in cache:
            return cache[identity][1]
        items = []
        changed = not isinstance(value, tuple)
        for item in value:
            if isinstance(item, (list, tuple)):
                normalized = _key(item, cache)
            else:
                if isinstance(item, Tensor):
                    raise TypeError("row keys must be CPU Python values, not tensors")
                hash(item)
                normalized = item
            items.append(normalized)
            changed = changed or normalized is not item
        normalized = tuple(items) if changed else value
        # The owned and requested keys of one plan share this memo.
        # Only memoize already-immutable tuple trees: lists (including tuples
        # containing lists) must be normalized again if a caller mutates them.
        # The strong reference prevents a later object's ID from being reused.
        if normalized is value:
            cache[identity] = (value, normalized)
        return normalized
    hash(value)
    return value


def _keys(values: Sequence[Hashable], cache: dict) -> tuple[Hashable, ...]:
    if isinstance(values, (tuple, list)):
        return _key(values, cache)
    return tuple(_key(item, cache) for item in values)


def _owner_directory(owned_per_rank) -> dict[Hashable, tuple[int, int]]:
    directory = {}
    for rank, owned in enumerate(owned_per_rank):
        for index, key in enumerate(owned):
            if key in directory:
                raise RuntimeError(f"row key {key!r} has multiple owners")
            directory[key] = (rank, index)
    return directory


@dataclass(frozen=True)
class ExchangePlan:
    """A frozen COPY map from unique owned rows to requested rows.

Requests may repeat a key and have arbitrary order.  Reverse communication
therefore adds all repeated, local, and remote contributions at the owner.
The same plan serves row redistribution: give each produced row a unique key
and request the complete set of rows belonging to an owned segment.
"""

    context: ExchangeContext
    owned_keys: Sequence[Hashable]
    requested_keys: Sequence[Hashable]
    name: str
    sequence: int
    send_indices: tuple[int, ...] | Tensor
    receive_indices: tuple[int, ...] | Tensor
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    _device_indices: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def build(
        cls,
        owned_keys: Sequence[Hashable],
        requested_keys: Sequence[Hashable],
        *,
        context: ExchangeContext | None = None,
        name: str = "",
    ) -> "ExchangePlan":
        context = ExchangeContext() if context is None else context
        error = None
        try:
            key_cache = {}
            owned = _keys(owned_keys, key_cache)
            requested = _keys(requested_keys, key_cache)
        except Exception as exc:
            error = f"invalid row keys: {type(exc).__name__}: {exc}"
            owned = requested = ()
        sequence = context._plan_sequence
        context._plan_sequence += 1
        # Replicate only the ownership directory, not every consumer's keys.
        # Once a request has resolved to an owner, only its integer owner-row
        # index needs to travel to that owner.
        metadata = context._collective((error, name, sequence, owned), "plan.build")
        context._raise_errors([entry[0] for entry in metadata])
        if any(entry[1:3] != metadata[0][1:3] for entry in metadata):
            raise RuntimeError("exchange plan name/order differs across ranks")
        owners = _owner_directory(entry[3] for entry in metadata)
        requested_rows = [[] for _ in range(context.world_size)]
        receive_rows = [[] for _ in range(context.world_size)]
        error = None
        for index, key in enumerate(requested):
            owner = owners.get(key)
            if owner is None:
                error = f"requests row {key!r} without an owner"
                break
            peer, owner_row = owner
            requested_rows[peer].append(owner_row)
            receive_rows[peer].append(index)
        receive_counts = tuple(map(len, receive_rows))
        request_metadata = context._collective((error, receive_counts), "plan.requests")
        context._raise_errors([entry[0] for entry in request_metadata])
        send_counts = tuple(entry[1][context.rank] for entry in request_metadata)
        receives = tuple(index for rows in receive_rows for index in rows)
        packed_rows = [index for rows in requested_rows for index in rows]
        if context.world_size == 1:
            sends = tuple(packed_rows)
        else:
            start = time.perf_counter()
            outgoing = torch.tensor(packed_rows, dtype=torch.int64)
            incoming = torch.empty(sum(send_counts), dtype=torch.int64)
            # These are routing indices, never feature values. They stay on
            # the CPU control group even when payloads use NCCL. Every rank
            # enters, including ranks whose split sizes are all zero.
            context.all_to_all_single(incoming, outgoing, send_counts, receive_counts, control=True)
            sends = tuple(incoming.tolist())
        return cls(context, owned, requested, name, sequence, tuple(sends),
                   tuple(receives), tuple(send_counts), tuple(receive_counts))

    @classmethod
    def from_owner_rows(
        cls,
        owned_count: int,
        requested_owners: Tensor,
        requested_rows: Tensor,
        *,
        context: ExchangeContext | None = None,
        name: str = "",
    ) -> "ExchangePlan":
        """Build COPY/adjoint routing from producer rank and local row IDs.

        Both request tensors must be dense, one-dimensional int64 tensors on
        the same CPU or CUDA device, of equal length. They describe the consumer's output order, and may
        contain repeats. Producer-local row IDs refer to the same frozen
        evaluation's tensor with ``owned_count`` rows on that producer rank.

        Each plan must use the same CPU/CUDA mode on all ranks; separate plans
        may choose different modes. The CPU NumPy/Gloo path is independent of
        the CUDA stable-sort/NCCL path. CUDA routes transfer only O(world_size)
        counts/status to the CPU, never request/route arrays.

        Only tiny counts/protocol metadata are replicated. Requested row IDs
        are grouped numerically and sent directly to their producers; no keys
        or global ownership directory are gathered. The plan keeps independent
        routing tensors, so callers may reuse or mutate their input metadata.
        Numeric plans use range placeholders for the legacy key-length fields;
        those placeholders are not globally meaningful row identifiers.
        """
        context = ExchangeContext() if context is None else context
        error = None
        count = 0
        mode = "cpu"
        try:
            if isinstance(owned_count, bool) or not isinstance(owned_count, Integral):
                raise TypeError("owned_count must be a nonnegative integer")
            count = int(owned_count)
            if count < 0 or count > torch.iinfo(torch.int64).max:
                raise ValueError("owned_count must fit a nonnegative int64")
            if not isinstance(name, str):
                raise TypeError("plan name must be a string")
            for label, tensor in (("requested_owners", requested_owners),
                                  ("requested_rows", requested_rows)):
                if (not isinstance(tensor, Tensor) or tensor.device.type not in {"cpu", "cuda"}
                        or tensor.layout != torch.strided or tensor.dtype != torch.int64
                        or tensor.ndim != 1):
                    raise TypeError(f"{label} must be a dense one-dimensional CPU int64 or CUDA int64 tensor")
            if requested_owners.shape != requested_rows.shape:
                raise ValueError("requested owner/row tensors must have equal length")
            if requested_owners.device != requested_rows.device:
                raise ValueError("requested owner/row tensors must be on the same device")
            mode = requested_owners.device.type
            if mode == "cuda":
                if context.backend not in {"nccl", "local"}:
                    raise ValueError("CUDA numeric routing requires an NCCL or local data backend")
                expected = torch.device("cuda", torch.cuda.current_device())
                if requested_owners.device != expected:
                    raise ValueError("CUDA numeric routing must use the current worker device")
                context_device = context.device
                if context_device is not None and context_device.type == "cuda" and context_device.index is None:
                    context_device = expected
                if context_device is not None and context_device != expected:
                    raise ValueError("CUDA numeric routing device differs from its context")
        except Exception as exc:
            error = f"invalid numeric routing: {type(exc).__name__}: {exc}"
        sequence = context._plan_sequence
        context._plan_sequence += 1
        # Preserve the CPU metadata envelope. The optional CUDA mode marker
        # makes a one-rank mode mismatch fail collectively before either data
        # backend's integer all-to-all, without adding a control round trip.
        admission = (error, name, sequence, count)
        metadata = context._collective(admission if mode == "cpu" else (*admission, mode),
                                       "plan.owner_rows")
        context._raise_errors([entry[0] for entry in metadata])
        if any(entry[1:3] != metadata[0][1:3] for entry in metadata):
            raise RuntimeError("exchange plan name/order differs across ranks")
        modes = [entry[4] if len(entry) > 4 else "cpu" for entry in metadata]
        if any(item != modes[0] for item in modes):
            raise RuntimeError("numeric routing CPU/CUDA mode differs across ranks")
        if mode == "cuda":
            return cls._from_cuda_owner_rows(count, requested_owners, requested_rows,
                                              context=context, name=name, sequence=sequence,
                                              owned_counts=[entry[3] for entry in metadata])

        # NumPy performs grouping and packing without constructing one Python
        # tuple/int object per request. Stable grouping preserves original
        # consumer order within each peer's packet (including repeated rows).
        owners = requested_owners.resolve_neg().numpy()
        rows = requested_rows.resolve_neg().numpy()
        error = None
        receive_counts = (0,) * context.world_size
        try:
            if owners.size and (owners.min() < 0 or owners.max() >= context.world_size):
                raise ValueError("requested owner rank is out of range")
            owned_counts = np.asarray([entry[3] for entry in metadata], dtype=np.int64)
            if rows.size and (rows.min() < 0 or np.any(rows >= owned_counts[owners])):
                raise ValueError("requested producer-local row is out of bounds")
            receive_counts = tuple(np.bincount(owners, minlength=context.world_size).tolist())
        except Exception as exc:
            error = f"invalid numeric routing: {type(exc).__name__}: {exc}"
        requests = context._collective((error, receive_counts), "plan.owner_requests")
        context._raise_errors([entry[0] for entry in requests])
        send_counts = tuple(entry[1][context.rank] for entry in requests)
        order = np.argsort(owners, kind="stable")
        receives = torch.from_numpy(order)
        outgoing = torch.from_numpy(rows[order])
        if context.world_size == 1:
            sends = outgoing
        else:
            sends = torch.empty(sum(send_counts), dtype=torch.int64)
            context.all_to_all_single(sends, outgoing, send_counts, receive_counts, control=True)
        return cls(context, range(count), range(len(rows)), name, sequence,
                   sends, receives, send_counts, receive_counts)

    @classmethod
    def _from_cuda_owner_rows(cls, count, owners, rows, *, context, name, sequence, owned_counts):
        """CUDA branch after the common collective shape/device admission."""
        error = None
        receive_counts = (0,) * context.world_size
        try:
            # Never index producer counts with an unchecked owner. Clamping is
            # only for the validation kernel; an invalid request is rejected
            # collectively and never reaches a data exchange.
            safe_owners = owners.clamp(0, context.world_size - 1)
            producer_counts = torch.tensor(owned_counts, device=owners.device, dtype=torch.long)
            bad_owner = ((owners < 0) | (owners >= context.world_size)).any()
            bad_row = ((rows < 0) | (rows >= producer_counts[safe_owners])).any()
            counts = torch.bincount(safe_owners, minlength=context.world_size)
            small = torch.cat((torch.stack((bad_owner, bad_row)).long(), counts)).cpu().tolist()
            if small[0]:
                raise ValueError("requested owner rank is out of range")
            if small[1]:
                raise ValueError("requested producer-local row is out of bounds")
            receive_counts = tuple(small[2:])
        except Exception as exc:
            error = f"invalid CUDA numeric routing: {type(exc).__name__}: {exc}"
        requests = context._collective((error, receive_counts), "plan.owner_requests")
        context._raise_errors([entry[0] for entry in requests])
        send_counts = tuple(entry[1][context.rank] for entry in requests)
        started = time.perf_counter()
        receives = torch.argsort(owners, stable=True)
        outgoing = rows.index_select(0, receives).contiguous()
        if context.world_size == 1:
            sends = outgoing
        else:
            sends = torch.empty(sum(send_counts), dtype=torch.long, device=owners.device)
            context.all_to_all_single(sends, outgoing, send_counts, receive_counts)
        elapsed = time.perf_counter() - started
        # Integer producer-row traffic is a subset of NCCL payload accounting,
        # not CPU Gloo control traffic and not additional feature bytes.
        context.stats["payload_calls"] += 1
        context.stats["payload_seconds"] += elapsed
        context.stats["routing_payload_calls"] += 1
        context.stats["routing_payload_seconds"] += elapsed
        for peer in range(context.world_size):
            if peer != context.rank:
                sent, received = receive_counts[peer] * 8, send_counts[peer] * 8
                context.stats["peer_sent_bytes"][peer] += sent
                context.stats["peer_received_bytes"][peer] += received
                context.stats["payload_sent_bytes"] += sent
                context.stats["payload_received_bytes"] += received
                context.stats["routing_payload_sent_bytes"] += sent
                context.stats["routing_payload_received_bytes"] += received
        return cls(context, range(count), range(rows.numel()), name, sequence,
                   sends, receives, send_counts, receive_counts)

    @classmethod
    def from_owner_rows_batch(cls, specs, *, context=None, stats=None):
        """Build an ordered batch of numeric plans with two admission rounds.

        Each specification is ``(owned_count, owners, rows, name)`` and has the
        same contract as :meth:`from_owner_rows`. A plan's CPU/CUDA mode must
        agree across ranks; different plans may use different modes. CPU routes
        retain the independent NumPy/Gloo implementation. All CUDA plans use
        the current worker device and share one bounded flags/counts download.

        The first collective admits every schema, name, sequence and producer
        count. The second admits *all* bounds and request counts before any
        row-request payload. A malformed later plan therefore cannot strand
        peers in an earlier plan's payload. For now the admitted integer row
        exchanges remain one per plan, including plans with no requests.

        ``stats`` optionally receives nested host-wall timings and counter
        deltas; they include existing waits and add no CUDA synchronization.
        The legacy single-plan API and its protocol remain unchanged.
        """
        context = ExchangeContext() if context is None else context
        started = time.perf_counter()
        control_before = context.stats["control_seconds"]
        calls_before = context.stats["control_calls"]
        sequence = context._plan_sequence
        entries, descriptors = [], []
        error = None
        try:
            if stats is not None and not isinstance(stats, dict):
                raise TypeError("batch stats must be a dictionary")
            # Materialize the small specification list, never an entity array.
            for index, spec in enumerate(tuple(specs)):
                if not isinstance(spec, (tuple, list)) or len(spec) != 4:
                    raise TypeError("each numeric plan specification must have four fields")
                count, owners, rows, name = spec
                if isinstance(count, bool) or not isinstance(count, Integral):
                    raise TypeError("owned_count must be a nonnegative integer")
                count = int(count)
                if count < 0 or count > torch.iinfo(torch.int64).max:
                    raise ValueError("owned_count must fit a nonnegative int64")
                if not isinstance(name, str):
                    raise TypeError("plan name must be a string")
                for label, tensor in (("requested_owners", owners), ("requested_rows", rows)):
                    if (not isinstance(tensor, Tensor) or tensor.device.type not in {"cpu", "cuda"}
                            or tensor.layout != torch.strided or tensor.dtype != torch.int64
                            or tensor.ndim != 1):
                        raise TypeError(f"{label} must be a dense one-dimensional CPU int64 or CUDA int64 tensor")
                if owners.shape != rows.shape:
                    raise ValueError("requested owner/row tensors must have equal length")
                if owners.device != rows.device:
                    raise ValueError("requested owner/row tensors must be on the same device")
                mode = owners.device.type
                if mode == "cuda":
                    if context.backend not in {"nccl", "local"}:
                        raise ValueError("CUDA numeric routing requires an NCCL or local data backend")
                    expected = torch.device("cuda", torch.cuda.current_device())
                    if owners.device != expected:
                        raise ValueError("CUDA numeric routing must use the current worker device")
                    context_device = context.device
                    if context_device is not None and context_device.type == "cuda" and context_device.index is None:
                        context_device = expected
                    if context_device is not None and context_device != expected:
                        raise ValueError("CUDA numeric routing device differs from its context")
                entries.append((count, owners, rows, name, mode))
                descriptors.append((name, sequence + index, count, mode))
        except Exception as exc:
            error = f"invalid numeric routing batch: {type(exc).__name__}: {exc}"
        metadata = context._collective((error, tuple(descriptors)), "plan.owner_rows_batch")
        context._raise_errors([entry[0] for entry in metadata])
        protocols = [tuple((name, order, mode) for name, order, _, mode in entry[1])
                     for entry in metadata]
        if any(protocol != protocols[0] for protocol in protocols):
            raise RuntimeError("numeric routing batch plan count/name/order/CPU/CUDA mode differs across ranks")
        # Reserve sequences only after the entire batch schema is admitted.
        # Schema failure (including different batch lengths) cannot leave the
        # contexts with different plan-sequence increments.
        context._plan_sequence += len(entries)
        schema_done = time.perf_counter()

        receive_counts = [(0,) * context.world_size for _ in entries]
        cpu_arrays, device_flags, device_plan_indices = {}, [], []
        errors = []
        for index, (count, owners, rows, name, mode) in enumerate(entries):
            try:
                owned_counts = [entry[1][index][2] for entry in metadata]
                if mode == "cpu":
                    cpu_owners, cpu_rows = owners.resolve_neg().numpy(), rows.resolve_neg().numpy()
                    if cpu_rows.size and (cpu_owners.min() < 0 or cpu_owners.max() >= context.world_size):
                        raise ValueError("requested owner rank is out of range")
                    producers = np.asarray(owned_counts, dtype=np.int64)
                    if cpu_rows.size and (cpu_rows.min() < 0 or np.any(cpu_rows >= producers[cpu_owners])):
                        raise ValueError("requested producer-local row is out of bounds")
                    receive_counts[index] = tuple(np.bincount(cpu_owners, minlength=context.world_size).tolist())
                    cpu_arrays[index] = (cpu_owners, cpu_rows)
                else:
                    # Only sanitized owners index the producer-count table.
                    # Invalid rows are compared, never gathered from a payload.
                    safe_owners = owners.clamp(0, context.world_size - 1)
                    producers = torch.tensor(owned_counts, device=owners.device, dtype=torch.long)
                    bad_owner = ((owners < 0) | (owners >= context.world_size)).any()
                    bad_row = ((rows < 0) | (rows >= producers[safe_owners])).any()
                    counts = torch.bincount(safe_owners, minlength=context.world_size)
                    device_flags.append(torch.cat((torch.stack((bad_owner, bad_row)).long(), counts)))
                    device_plan_indices.append(index)
            except Exception as exc:
                errors.append(f"plan {index} ({name!r}): {type(exc).__name__}: {exc}")
        if device_flags:
            try:
                # Exactly O(number_of_plans * world_size), not request arrays.
                downloaded = torch.stack(device_flags).cpu().tolist()
                for index, small in zip(device_plan_indices, downloaded):
                    name = entries[index][3]
                    if small[0]:
                        errors.append(f"plan {index} ({name!r}): requested owner rank is out of range")
                    if small[1]:
                        errors.append(f"plan {index} ({name!r}): requested producer-local row is out of bounds")
                    receive_counts[index] = tuple(small[2:])
            except Exception as exc:
                errors.append(f"CUDA batch flags: {type(exc).__name__}: {exc}")
        validated_at = time.perf_counter()
        requests = context._collective(("; ".join(errors) or None, tuple(receive_counts)),
                                       "plan.owner_requests_batch")
        context._raise_errors([entry[0] for entry in requests])
        requests_done = time.perf_counter()

        plans, route_details = [], {}
        for index, (count, owners, rows, name, mode) in enumerate(entries):
            route_started = time.perf_counter()
            receives_count = receive_counts[index]
            sends_count = tuple(entry[1][index][context.rank] for entry in requests)
            if mode == "cpu":
                cpu_owners, cpu_rows = cpu_arrays[index]
                order = np.argsort(cpu_owners, kind="stable")
                receives = torch.from_numpy(order)
                outgoing = torch.from_numpy(cpu_rows[order])
            else:
                receives = torch.argsort(owners, stable=True)
                outgoing = rows.index_select(0, receives).contiguous()
            if context.world_size == 1:
                sends = outgoing
            else:
                sends = outgoing.new_empty(sum(sends_count))
                context.all_to_all_single(sends, outgoing, sends_count, receives_count,
                                          control=mode == "cpu")
            elapsed = time.perf_counter() - route_started
            if mode == "cuda":
                context.stats["payload_calls"] += 1
                context.stats["payload_seconds"] += elapsed
                context.stats["routing_payload_calls"] += 1
                context.stats["routing_payload_seconds"] += elapsed
                for peer in range(context.world_size):
                    if peer != context.rank:
                        sent, received = receives_count[peer] * 8, sends_count[peer] * 8
                        context.stats["peer_sent_bytes"][peer] += sent
                        context.stats["peer_received_bytes"][peer] += received
                        context.stats["payload_sent_bytes"] += sent
                        context.stats["payload_received_bytes"] += received
                        context.stats["routing_payload_sent_bytes"] += sent
                        context.stats["routing_payload_received_bytes"] += received
            plans.append(cls(context, range(count), range(rows.numel()), name, sequence + index,
                             sends, receives, sends_count, receives_count))
            # Include the ordinal: public batches may deliberately repeat names.
            route_details[f"{index}:{name}"] = elapsed * 1000
        finished = time.perf_counter()
        if stats is not None:
            stats.update(plan_count=len(entries), admission_calls=2,
                         control_calls=context.stats["control_calls"] - calls_before,
                         control_ms=(context.stats["control_seconds"] - control_before) * 1000,
                         schema_ms=(schema_done - started) * 1000,
                         validate_counts_ms=(validated_at - schema_done) * 1000,
                         requests_ms=(requests_done - validated_at) * 1000,
                         route_ms=(finished - requests_done) * 1000,
                         total_ms=(finished - started) * 1000,
                         route_details_ms=route_details,
                         cuda_count_downloads=int(bool(device_flags)),
                         cuda_control_shape=(len(device_flags), context.world_size + 2),
                         cuda_control_elements=len(device_flags) * (context.world_size + 2),
                         cuda_control_scope="per CUDA plan: two bounds flags plus per-peer counts; no entity arrays")
        return tuple(plans)

    def _validate(self, tensor: Tensor, reverse: bool) -> None:
        expected_rows = len(self.requested_keys if reverse else self.owned_keys)
        error = None
        signature = None
        try:
            if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
                raise TypeError("payload must be a dense strided tensor")
            if tensor.ndim == 0 or tensor.shape[0] != expected_rows:
                raise ValueError(f"expected {expected_rows} rows, got {tuple(tensor.shape)}")
            if self.context.backend == "nccl" and tensor.device.type != "cuda":
                raise ValueError("NCCL payload must be on CUDA")
            if self.context.backend == "gloo" and tensor.device.type != "cpu":
                raise ValueError("Gloo reference payload must be on CPU")
            signature = (self.sequence, self.name, reverse, tuple(tensor.shape[1:]),
                         str(tensor.dtype), tensor.device.type)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        metadata = self.context._collective((error, signature), "plan.payload")
        self.context._raise_errors([entry[0] for entry in metadata])
        signatures = [entry[1] for entry in metadata]
        if any(item != signatures[0] for item in signatures):
            raise RuntimeError(f"exchange payload phase/shape/dtype mismatch: {signatures}")

    def _indices(self, device: torch.device) -> tuple[Tensor, Tensor]:
        # A plan belongs to one frozen evaluation epoch. The immutable routing
        # indices can be reused by every block and by its reverse schedule;
        # no cache survives migration/reconstruction into the next plan.
        if device not in self._device_indices:
            self._device_indices[device] = (
                torch.as_tensor(self.send_indices, device=device, dtype=torch.long),
                torch.as_tensor(self.receive_indices, device=device, dtype=torch.long),
            )
        return self._device_indices[device]

    def _communicate(self, tensor: Tensor, reverse: bool, *, _prepared=None) -> Tensor:
        _saved_recompute.check_no_collective("exchange")
        if self.context.exchange_trace is not None:
            self.context.exchange_trace.append((self.name, self.sequence, reverse))
        if _prepared is None:
            self._validate(tensor, reverse)
        else:
            # Internal prepared execution has already collectively checked the
            # complete protocol and both directions' index bounds. Metadata
            # assertions are local here; unexpected failures are fatal to the
            # finite-timeout worker group, not a recoverable partial fallback.
            _prepared.validate_payload(self, tensor, reverse)
            if _prepared.self_rows == "local":
                return self._communicate_local_self(tensor, reverse, _prepared)
        context = self.context
        start = time.perf_counter()
        send_counts = self.receive_counts if reverse else self.send_counts
        receive_counts = self.send_counts if reverse else self.receive_counts
        output_rows = len(self.owned_keys if reverse else self.requested_keys)
        trailing_shape = tuple(tensor.shape[1:])
        width = math.prod(trailing_shape)
        with torch.no_grad():
            send_indices, receive_indices = (self._indices(tensor.device) if _prepared is None
                                             else _prepared.indices)
            indices = receive_indices if reverse else send_indices
            packed = tensor.detach().index_select(0, indices).contiguous().reshape(-1)
            received = tensor.new_empty(sum(receive_counts) * width)
            if context.world_size == 1:
                received.copy_(packed)
            else:
                # Every rank calls this, even if all split sizes are zero.
                # A distinct receive allocation survives all downstream users.
                context.all_to_all_single(received, packed,
                                          [count * width for count in receive_counts],
                                          [count * width for count in send_counts])
            output = tensor.new_zeros((output_rows, *trailing_shape))
            destination = send_indices if reverse else receive_indices
            rows = received.reshape((sum(receive_counts), *trailing_shape))
            if reverse:
                if (_prepared is not None and _prepared._private_reverse_sum_enabled()
                        and tensor.device.type == "cuda" and tensor.dtype == torch.float32
                        and len(trailing_shape) == 1):
                    # Only the internal padded-native adapter can bind this
                    # policy. No public plan flag or tensor value enables it.
                    from fastmd.engine.kernels.triton_zero_skip_index import zero_skip_index_add_
                    zero_skip_index_add_(output, destination, rows)
                else:
                    output.index_add_(0, destination, rows)
            else:
                output.index_copy_(0, destination, rows)
            if tensor.is_cuda and _prepared is None:
                # MVP has no overlap.  Make NCCL completion, unpacking, and
                # subsequent consumer/storage lifetime unambiguous.
                torch.cuda.synchronize(tensor.device)
        item_bytes = width * tensor.element_size()
        for peer in range(context.world_size):
            if peer != context.rank:
                sent = send_counts[peer] * item_bytes
                received_bytes = receive_counts[peer] * item_bytes
                context.stats["peer_sent_bytes"][peer] += sent
                context.stats["peer_received_bytes"][peer] += received_bytes
                context.stats["payload_sent_bytes"] += sent
                context.stats["payload_received_bytes"] += received_bytes
        context.stats["payload_calls"] += 1
        context.stats["payload_seconds"] += time.perf_counter() - start
        return output

    def _communicate_local_self(self, tensor: Tensor, reverse: bool, prepared) -> Tensor:
        """Prepared COPY/reverse-SUM whose self rows never enter the data backend.

        This rank's own block of the admitted routing (the rows it both sends
        and receives) is one fused gather-scatter from the input into the fresh
        output. Only remote rows are packed and exchanged; the all_to_all is
        still entered by every rank at P>1, with a zero self split, so its
        sizes stay the admitted, capture-fixed ones. COPY writes every output
        row (receive indices are an admitted permutation), so its output is
        uninitialized storage; reverse-SUM zero-fills and adds remote rows of
        lower ranks, then self rows, then higher ranks: the order in which the
        NCCL path adds its one received buffer. Deterministic mode keeps that
        path's single ordered SUM over [lower | self | higher] (index_add_, or
        the private zero-skipping SUM, which then lowers to an ordered
        index_add_ too): torch's deterministic kernel sums each call's
        duplicates before adding them, so splitting the call would regroup the
        sum and lose bitwise parity with the "nccl" lowering. The output is
        always a fresh buffer, never the input or storage saved by autograd
        (INV-8).
        """
        context = self.context
        start = time.perf_counter()
        rank = context.rank
        # Rows this rank gathers and sends / receives and scatters, per peer.
        send_counts = self.receive_counts if reverse else self.send_counts
        receive_counts = self.send_counts if reverse else self.receive_counts
        output_rows = len(self.owned_keys if reverse else self.requested_keys)
        trailing_shape = tuple(tensor.shape[1:])
        width = math.prod(trailing_shape)
        send_indices, receive_indices = prepared.indices
        gather, scatter = (receive_indices, send_indices) if reverse else (send_indices, receive_indices)
        own = send_counts[rank]
        # Collectively admitted (every sender/receiver pair, including self).
        assert receive_counts[rank] == own, "prepared self send/receive split sizes differ"
        gather_start, scatter_start = sum(send_counts[:rank]), sum(receive_counts[:rank])
        source = tensor.detach()
        zero_skip = (reverse and prepared._private_reverse_sum_enabled()
                     and source.device.type == "cuda" and source.dtype == torch.float32
                     and len(trailing_shape) == 1)
        with torch.no_grad():
            output = (source.new_zeros if reverse else source.new_empty)((output_rows, *trailing_shape))
            received = None
            if context.world_size > 1:
                packed = source.new_empty((sum(send_counts) - own, *trailing_shape))
                for rows, part in ((gather[:gather_start], packed[:gather_start]),
                                   (gather[gather_start + own:], packed[gather_start:])):
                    if rows.numel():
                        torch.index_select(source, 0, rows, out=part)
                received = source.new_empty((sum(receive_counts) - own, *trailing_shape))
                # Every rank calls this, even if all remote split sizes are zero.
                context.all_to_all_single(
                    received.reshape(-1), packed.reshape(-1),
                    [0 if peer == rank else count * width for peer, count in enumerate(receive_counts)],
                    [0 if peer == rank else count * width for peer, count in enumerate(send_counts)])

            def unpack(rows, destination):
                if not destination.numel():
                    return
                if not reverse:
                    output.index_copy_(0, destination, rows)
                elif zero_skip:
                    from fastmd.engine.kernels.triton_zero_skip_index import zero_skip_index_add_
                    zero_skip_index_add_(output, destination, rows)
                else:
                    output.index_add_(0, destination, rows)

            ordered = reverse and deterministic_mode()
            if ordered and received is not None:
                rows = torch.cat((received[:scatter_start],
                                  source.index_select(0, gather[gather_start:gather_start + own]),
                                  received[scatter_start:]))
                if zero_skip:
                    from fastmd.engine.kernels.triton_zero_skip_index import zero_skip_index_add_
                    zero_skip_index_add_(output, scatter, rows)
                else:
                    output.index_add_(0, scatter, rows)
            else:
                if received is not None:
                    unpack(received[:scatter_start], scatter[:scatter_start])
                _self_rows(output, source, gather[gather_start:gather_start + own],
                           scatter[scatter_start:scatter_start + own], accumulate=reverse,
                           zero_skip=zero_skip)
                if received is not None:
                    unpack(received[scatter_start:], scatter[scatter_start + own:])
        item_bytes = width * tensor.element_size()
        for peer in range(context.world_size):
            if peer != rank:
                sent = send_counts[peer] * item_bytes
                received_bytes = receive_counts[peer] * item_bytes
                context.stats["peer_sent_bytes"][peer] += sent
                context.stats["peer_received_bytes"][peer] += received_bytes
                context.stats["payload_sent_bytes"] += sent
                context.stats["payload_received_bytes"] += received_bytes
        context.stats["payload_calls"] += 1
        context.stats["payload_seconds"] += time.perf_counter() - start
        return output

    def forward(self, tensor: Tensor) -> Tensor:
        return self._communicate(tensor, reverse=False)

    def reverse(self, cotangent: Tensor) -> Tensor:
        return self._communicate(cotangent, reverse=True)


def _self_rows(output, source, source_rows, output_rows, *, accumulate, zero_skip):
    """``output[output_rows] (+)= source[source_rows]``, fused when qualified.

    Output rows are unique for COPY (an admitted permutation). SUM keeps an
    ordered ATen lowering in deterministic mode (the private zero-skipping SUM
    too: ``zero_skip_index_add_`` lowers to an ordered index_add_ there),
    exactly like the NCCL path's unpack.
    """
    if not source_rows.numel():
        return
    fused = output.ndim == 2 and source.ndim == 2 and output.device.type == "cuda"
    if fused and accumulate and deterministic_mode():
        fused = False
    if fused:
        from fastmd.engine.kernels.triton_row_gather_scatter import (
            row_gather_scatter_, row_gather_scatter_reason)
        fused = row_gather_scatter_reason(output, source, source_rows, output_rows) is None
    if fused:
        row_gather_scatter_(output, source, source_rows, output_rows,
                            accumulate=accumulate, skip_zero=zero_skip)
        return
    rec = _recorder._ACTIVE
    if rec is not None:
        rec.record("self_rows", "aten_index_copy" if not accumulate else (
            "zero_skip_index_add" if zero_skip else "aten_index_add"), {
            "rows": source_rows.numel(), "width": source.shape[-1],
        }, phase="bwd" if torch._C._current_autograd_node() is not None else "fwd")
    rows = source.index_select(0, source_rows)
    if not accumulate:
        output.index_copy_(0, output_rows, rows)
    elif zero_skip:
        from fastmd.engine.kernels.triton_zero_skip_index import zero_skip_index_add_
        zero_skip_index_add_(output, output_rows, rows)
    else:
        output.index_add_(0, output_rows, rows)


_SELF_ROW_MODES = ("nccl", "local")


def _index_identity(value):
    if isinstance(value, Tensor):
        return (id(value), value._version, tuple(value.shape), value.dtype, value.device)
    return tuple(value)


_PADDED_NATIVE_SUM_AUTHORITY = object()


@dataclass(frozen=True)
class _PreparedExchange:
    """Private checked payload binding of one stage of a ``PrevalidatedSchedule``.

    Used by the autograd backend's exchange Functions, the graph runner's
    static snapshot and the padding helpers. Construction and collective
    admission happen outside the hot path. Routing tensors are independent
    snapshots and remain alive through all consumers and VJPs.

    INV-8: each COPY/reverse-SUM returns a newly allocated buffer that the
    binding never retains, reuses or writes again. Autograd takes a reverse
    SUM's output as a gradient and may accumulate into it in place, and a
    COPY's output may be saved for backward: neither may share storage with a
    saved tensor or with a later payload.
    """

    plan: ExchangePlan
    trailing_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    indices: tuple[Tensor, Tensor]
    _identity: tuple
    _index_versions: tuple[int, int]
    # Binding-local capability, not an ExchangePlan/public constructor option.
    # It is valid only for these exact plan/route objects and tensor metadata.
    _private_reverse_sum: tuple | None = field(default=None, repr=False, compare=False)
    # "local": this rank's own rows bypass the data backend (fused gather-
    # scatter); "nccl": the original lowering, every row through all_to_all.
    self_rows: str = "local"

    def _private_reverse_sum_identity(self):
        return (_PADDED_NATIVE_SUM_AUTHORITY, id(self.plan), id(self.indices),
                self.trailing_shape, self.dtype, self.device)

    def _private_reverse_sum_enabled(self):
        if self._private_reverse_sum is None:
            return False
        if self._private_reverse_sum != self._private_reverse_sum_identity():
            raise ValueError("private padded reverse-SUM binding changed or was transplanted")
        return True

    @staticmethod
    def _plan_identity(plan):
        return (id(plan.context), len(plan.owned_keys), len(plan.requested_keys),
                plan.name, plan.sequence, tuple(plan.send_counts), tuple(plan.receive_counts),
                _index_identity(plan.send_indices), _index_identity(plan.receive_indices))

    @classmethod
    def build(cls, plan: ExchangePlan, trailing_shape, dtype, device, *, _shared=None,
              _validation_flags=None, self_rows="local"):
        if not isinstance(plan, ExchangePlan):
            raise TypeError("prepared payload requires an ExchangePlan")
        if self_rows not in _SELF_ROW_MODES:
            raise ValueError(f"prepared self_rows must be one of {_SELF_ROW_MODES}")
        shape = tuple(trailing_shape)
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in shape):
            raise ValueError("prepared trailing shape must contain nonnegative integers")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("prepared payload dtype must be a torch dtype")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if plan.context.backend == "nccl" and device.type != "cuda":
            raise ValueError("NCCL prepared payload must be on CUDA")
        if plan.context.backend == "gloo" and device.type != "cpu":
            raise ValueError("Gloo prepared payload must be on CPU")
        if _shared is not None:
            if _shared.plan is not plan or _shared.device != device:
                raise ValueError("shared prepared routing requires the same plan and device")
            return cls(plan, shape, dtype, device, _shared.indices, _shared._identity,
                       _shared._index_versions, self_rows=self_rows)
        size = plan.context.world_size
        for counts in (plan.send_counts, plan.receive_counts):
            if len(counts) != size or any(isinstance(n, bool) or not isinstance(n, Integral)
                                          or n < 0 for n in counts):
                raise ValueError("prepared split sizes must be nonnegative integers per peer")
        copied = []
        for label, values, counts, limit in (
            ("send", plan.send_indices, plan.send_counts, len(plan.owned_keys)),
            ("receive", plan.receive_indices, plan.receive_counts, len(plan.requested_keys)),
        ):
            if isinstance(values, Tensor):
                if values.layout != torch.strided or values.ndim != 1 or values.dtype != torch.long:
                    raise TypeError(f"prepared {label} indices must be one-dimensional int64")
                if values.device.type == "cuda":
                    if values.device != device:
                        raise ValueError(f"prepared {label} CUDA index device differs from payload device")
                    if values.numel() != sum(counts):
                        raise ValueError(f"prepared {label} index count differs from split sizes")
                    if label == "receive" and values.numel() != limit:
                        raise ValueError("prepared receive indices must assign each requested row exactly once")
                    bad_bounds = ((values < 0) | (values >= limit)).any()
                    if label == "receive":
                        # Invalid indices must not trigger a device assertion
                        # before the other ranks can receive a collective
                        # admission error. Only bounded, sanitized indices
                        # enter the histogram; the originals remain unchanged.
                        safe = values.clamp(0, max(limit - 1, 0))
                        bad_permutation = (torch.bincount(safe, minlength=limit) != 1).any()
                        flags = (bad_bounds, bad_permutation)
                    else:
                        flags = (bad_bounds,)
                    messages = (f"prepared {label} index is out of bounds",
                                "prepared receive indices must assign each requested row exactly once")
                    if _validation_flags is None:
                        for bad, message in zip(torch.stack(flags).cpu().tolist(), messages):
                            if bad:
                                raise ValueError(message)
                    else:
                        # No payload consumes these snapshots until the full
                        # schedule's collective admission succeeds. Validate
                        # every unique route in one small download per device.
                        _validation_flags.extend(zip(flags, messages))
                    copied.append(values.detach().clone())
                    continue
                if values.device.type != "cpu":
                    raise ValueError(f"prepared {label} indices must be on CPU or CUDA")
                cpu = values.detach()
            else:
                if any(isinstance(n, bool) or not isinstance(n, Integral) for n in values):
                    raise TypeError(f"prepared {label} indices must be integers")
                cpu = torch.tensor(values, dtype=torch.long)
            if cpu.numel() != sum(counts):
                raise ValueError(f"prepared {label} index count differs from split sizes")
            if cpu.numel() and (int(cpu.min()) < 0 or int(cpu.max()) >= limit):
                raise ValueError(f"prepared {label} index is out of bounds")
            # Receive rows are a permutation. A bounded linear histogram is
            # sufficient; sorting every O(T) route for validation is wasteful.
            if label == "receive" and (cpu.numel() != limit or
                                       bool((torch.bincount(cpu, minlength=limit) != 1).any())):
                raise ValueError("prepared receive indices must assign each requested row exactly once")
            copied.append(cpu.to(device=device, copy=True))
        return cls(plan, shape, dtype, device, tuple(copied), cls._plan_identity(plan),
                   tuple(tensor._version for tensor in copied), self_rows=self_rows)

    @classmethod
    def build_many(cls, specs, *, stats=None, self_rows="local"):
        """Snapshot and locally validate a complete schedule's route bindings.

        This performs no communication. The caller MUST collectively admit all
        errors before using any returned binding. Invalid GPU indices enter
        only sanitized histograms and copies, never payload gathers/scatters.
        The standalone ``build`` API keeps its immediate validation behavior.
        """
        unique, prepared, flags = {}, [], []
        for plan, shape, dtype, device in specs:
            device = torch.device(device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            key = (id(plan), device)
            binding = cls.build(plan, shape, dtype, device, _shared=unique.get(key),
                                _validation_flags=flags, self_rows=self_rows)
            unique.setdefault(key, binding)
            prepared.append(binding)
        grouped = {}
        for flag, message in flags:
            grouped.setdefault(flag.device, []).append((flag, message))
        for items in grouped.values():
            bad = torch.stack([flag for flag, _ in items]).cpu().tolist()
            for invalid, (_, message) in zip(bad, items):
                if invalid:
                    raise ValueError(message)
        if stats is not None:
            stats.update(routing_validation_downloads=len(grouped),
                         routing_validation_elements=len(flags))
        return tuple(prepared)

    def descriptor(self):
        # Device ordinals differ by rank; dtype/width and stage protocol must
        # agree, while per-rank owned/requested counts may legitimately differ.
        return (self.plan.name, self.plan.sequence, self.trailing_shape, str(self.dtype),
                self.device.type, tuple(self.plan.send_counts), tuple(self.plan.receive_counts))

    def validate_integrity(self):
        if self._private_reverse_sum is not None:
            # pad_bond_partial is constructed only AFTER the fresh public prepare/
            # run boundary. Its private numerical view cannot itself pass that
            # boundary or turn a normal eager schedule into an optimized one.
            raise RuntimeError("private padded reverse-SUM binding cannot enter a public boundary")
        if self._plan_identity(self.plan) != self._identity:
            raise RuntimeError("prepared exchange plan changed after admission")
        if tuple(tensor._version for tensor in self.indices) != self._index_versions:
            raise RuntimeError("prepared exchange routing buffer changed after admission")

    def validate_payload(self, plan, tensor, reverse):
        self._private_reverse_sum_enabled()
        if plan is not self.plan:
            raise ValueError("payload uses a different prepared exchange plan")
        expected = (len(plan.requested_keys if reverse else plan.owned_keys), *self.trailing_shape)
        if (not isinstance(tensor, Tensor) or tensor.layout != torch.strided
                or tuple(tensor.shape) != expected or tensor.dtype != self.dtype
                or tensor.device != self.device):
            raise ValueError(f"prepared payload differs from admitted shape/dtype/device: {expected}")


def _bind_private_padded_reverse_sum(binding):
    """Internal adapter hook; capacity calls only after fresh native admission.

    Static snapshots must explicitly rebind their newly owned plan/routes.
    Merely copying a binding or attaching a flag to a public plan cannot enable
    the lowering. CPU/other dtype qualification still uses the original SUM.
    """
    return replace(binding, _private_reverse_sum=binding._private_reverse_sum_identity())
