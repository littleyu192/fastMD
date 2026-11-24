"""P>1 ``Communicator``: host control agreement over the data group (unified step 3.1a, H2).

Before 3.1a every control agreement was one Gloo ``all_gather_object`` on the
CPU control group (~0.7 ms each; 35 per steady-state evaluation at 1,344
atoms on 4 GPUs). The same agreements are now fixed-shape tensor collectives
on the DATA group (NCCL on CUDA; in CPU tests the data group is Gloo):

* ``gather_object`` (internal transport of ``ExchangeContext._collective``):
  each rank writes ``[length, pickled bytes]`` into an int64 slot of
  ``capacity`` bytes; one ``all_gather_into_tensor`` and one download return
  every rank's bytes, unpickled into exactly the list ``all_gather_object``
  returns. A longer payload makes every rank (all saw the same lengths) run
  a second, exact-size round and grow the slot to a power of two.
* ``agree_flags_host``: one int64 MAX ``all_reduce`` of a fixed-width vector
  and one download. A header (phase code, length and round number, each with
  its negation: max(-x) = -min(x)) makes a round whose phase, length or order
  differs across ranks raise on every rank instead of reducing unrelated
  vectors; one participation slot per rank (its round number, 0 elsewhere)
  makes a reduction that some rank did not contribute to raise as well.
* The Gloo control group carries only slow paths: error strings after a flag
  reported a failure, and differing configuration fingerprints.

Host waits are CUDA event synchronizations, which release the GIL: the
progress watchdog thread can still run ``launch.fail_fast()`` while a peer
hangs (K2), exactly as during any other NCCL wait. Unlike a Gloo collective,
an NCCL wait is not ended by a dead peer's closed connection: a peer's
``fail_fast`` announces itself in the process-group store, and this rank's
progress watchdog fails fast on seeing it (``launch.PeerFailureProbe``).
A round released by an abort instead of a transfer may return stale or
partial bytes: a gather whose own row is not what this rank sent, or a flag
reduction missing a rank's participation, raises instead of returning them.
Such errors after another rank's failure are secondary, like a Gloo
"Connection closed by peer". A control round inside a CUDA graph capture
would be recorded instead of executed, so it raises.
"""
from __future__ import annotations

import hashlib
import math
import pickle
import weakref

import torch

from fastmd.engine import saved_recompute as _saved_recompute
import torch.distributed as dist
from torch import Tensor

from fastmd.engine.comm import NOT_YET, flag_parts, gather_fields

# Bytes per rank slot. The largest in-evaluate envelope is the legacy
# capacity policy's graph.admit (9.0-9.1 KB at 1,344..43,008 atoms on 4 GPUs;
# every other one, including 3.2's capture.agree, is smaller: <= 2.9 KB);
# the CLI's per-step stats gather is 7 KB with model_graph, 11.3 KB eager.
# A larger payload costs one extra round, once, and grows every later slot
# to the next power of two.
DEFAULT_CAPACITY = 16 * 1024
# int64 entries a flag round carries after its header: the vector is
# _HEADER + world_size + FLAG_ENTRIES wide, one fixed NCCL shape per group.
# Participation slots widen it rather than eat this budget, so a round that
# fits at 2 ranks fits at any world size (evaluate.admission uses 8).
FLAG_ENTRIES = 56
# phase, -phase, length, -length, round, -round; then world_size
# participation slots (rank r writes the round number into slot r).
_HEADER = 6


def _phase_code(phase: str) -> int:
    """Stable 56-bit code of a round name; its negation also fits int64."""
    return int.from_bytes(hashlib.blake2b(phase.encode(), digest_size=7).digest(), "little")


def _round8(size: int) -> int:
    return -(-size // 8) * 8


# Storage addresses of live pinned host staging buffers. Their downloads are
# fixed-size control metadata (slots or flag vectors bounded by the capacity,
# bytes that host code produced), never atom/edge/triplet arrays; guards that
# reject bulk device-to-host copies can recognize them with is_control_staging.
_CONTROL_STAGING: set[int] = set()


def is_control_staging(tensor: Tensor) -> bool:
    """True for (a view of) a DomainCommunicator's pinned host staging buffer."""
    return (isinstance(tensor, Tensor) and tensor.device.type == "cpu"
            and tensor.untyped_storage().data_ptr() in _CONTROL_STAGING)


# --- per-peer slice exchange (A1: P > 4, neighbor-only P2P) ---
PAYLOAD_TRANSPORTS = ("auto", "all_to_all", "neighbor_p2p")
# "auto" keeps all_to_all_single up to this world size (P <= 4 unchanged)
# and runs the per-peer-split payload exchanges as grouped P2P above it.
P2P_MIN_WORLD = 5


def resolve_payload_transport(mode: str, world_size: int, *, every_peer_neighbor: bool = False) -> str:
    """The transport ``mode`` selects at ``world_size``: "all_to_all" or "neighbor_p2p".

    ``every_peer_neighbor``: every rank's neighbor domains are all other
    ranks (e.g. a 2x2x2 grid of domains wider than the halo). Grouped P2P
    then removes no peer but costs more host time per call than
    all_to_all_single (8 GB300 ranks over MNNVL: 342 vs 78 us to 7 peers;
    10,752-atom eager step 126.6 vs 115.7 ms), so "auto" keeps all_to_all.
    Every rank must pass the same value (a function of the agreed geometry).
    """
    if mode not in PAYLOAD_TRANSPORTS:
        raise ValueError(f"payload_transport must be one of {PAYLOAD_TRANSPORTS}")
    if mode == "auto":
        return ("neighbor_p2p" if world_size >= P2P_MIN_WORLD and not every_peer_neighbor
                else "all_to_all")
    return mode


def _offsets(counts) -> list[int]:
    offsets, total = [], 0
    for count in counts:
        offsets.append(total)
        total += count
    return offsets


def grouped_p2p_exchange(output: Tensor, input: Tensor, output_split_sizes, input_split_sizes,
                         *, group, rank: int) -> int:
    """``all_to_all_single``'s result through ONE grouped P2P over the peers with rows.

    Splits are dim-0 row counts per rank, as in ``all_to_all_single``, and
    must be the collectively admitted ones: rank p sends q exactly the rows q
    expects from p, so both ends of every pair agree which pairs carry rows
    and no rank waits for a message nobody sends. Pairs without rows post
    nothing, so a spatial exchange touches only its neighbor domains (the
    per-peer slices of a halo exchange). This rank's own slice is a local copy.
    All receives and sends of the call are posted in one ``batch_isend_irecv``
    (an NCCL group: no ordering deadlock between pairs), then waited on (NCCL:
    the current stream waits, the host does not). Returns the number of
    remote ops posted.
    """
    _saved_recompute.check_no_collective("grouped_p2p_exchange")
    world = len(input_split_sizes)
    if len(output_split_sizes) != world:
        raise ValueError("one input and one output split size per rank are required")
    if not 0 <= rank < world:
        raise ValueError("rank outside the split sizes")
    input_offsets, output_offsets = _offsets(input_split_sizes), _offsets(output_split_sizes)
    if (sum(input_split_sizes) != input.shape[0] or sum(output_split_sizes) != output.shape[0]):
        raise ValueError("split sizes differ from the exchanged tensors' rows")
    own = input_split_sizes[rank]
    if own != output_split_sizes[rank]:
        raise ValueError("this rank's own send and receive split sizes differ")
    if own:
        output.narrow(0, output_offsets[rank], own).copy_(input.narrow(0, input_offsets[rank], own))
    ops = []
    # Receives before sends, peers in ascending order: identical on every
    # rank (the grouped call itself does not depend on the order).
    for peer in range(world):
        if peer != rank and output_split_sizes[peer]:
            ops.append(dist.P2POp(dist.irecv, output.narrow(0, output_offsets[peer], output_split_sizes[peer]),
                                  group=group, group_peer=peer))
    for peer in range(world):
        if peer != rank and input_split_sizes[peer]:
            ops.append(dist.P2POp(dist.isend, input.narrow(0, input_offsets[peer], input_split_sizes[peer]),
                                  group=group, group_peer=peer))
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()
    return len(ops)


class SliceExchange:
    """Epoch-frozen per-peer contiguous slices: COPY forward, reverse-SUM backward.

    ``send_counts[q]`` rows of this rank's send buffer (gathered from owned
    rows by ``send_rows``, grouped by peer in rank order) land in peer q's
    receive buffer at slice ``p``; ``receive_counts[q]`` rows arrive from q.
    Counts are fixed for the epoch and must match across every pair (p's
    send count to q is q's receive count from p); they are not re-agreed per
    call. ``copy`` gathers and exchanges (the ghost COPY); ``reverse_sum``
    sends the receive-shaped cotangent back and index-adds it into owned rows
    (its adjoint). Both are one grouped P2P over the peers with rows
    (``grouped_p2p_exchange``), or ``all_to_all_single`` when
    ``transport="all_to_all"`` (the reference the P2P path must equal).
    """

    def __init__(self, send_rows: Tensor, send_counts, receive_counts, *, group=None,
                 transport: str = "neighbor_p2p"):
        if transport not in ("all_to_all", "neighbor_p2p"):
            raise ValueError("slice exchange transport must be 'all_to_all' or 'neighbor_p2p'")
        initialized = dist.is_available() and dist.is_initialized()
        self.group = (dist.group.WORLD if group is None else group) if initialized else None
        self.world_size = dist.get_world_size(self.group) if initialized else 1
        self.rank = dist.get_rank(self.group) if initialized else 0
        send_counts, receive_counts = tuple(send_counts), tuple(receive_counts)
        for label, counts in (("send", send_counts), ("receive", receive_counts)):
            if len(counts) != self.world_size or any(
                    isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
                raise ValueError(f"{label} counts must be one nonnegative integer per rank")
        if (not isinstance(send_rows, Tensor) or send_rows.ndim != 1 or send_rows.dtype != torch.long
                or send_rows.numel() != sum(send_counts)):
            raise ValueError("send_rows must be one int64 owned-row index per sent row")
        if send_counts[self.rank] != receive_counts[self.rank]:
            raise ValueError("this rank's own send and receive counts differ")
        self.send_rows, self.transport = send_rows, transport
        self.send_counts, self.receive_counts = send_counts, receive_counts
        self.peers = tuple(peer for peer in range(self.world_size) if peer != self.rank
                           and (send_counts[peer] or receive_counts[peer]))

    def _exchange(self, output: Tensor, input: Tensor, output_counts, input_counts) -> None:
        _saved_recompute.check_no_collective("slice_exchange")
        if self.world_size == 1:
            output.copy_(input)
        elif self.transport == "all_to_all":
            width = math.prod(input.shape[1:])
            dist.all_to_all_single(output.reshape(-1), input.reshape(-1),
                                   output_split_sizes=[n * width for n in output_counts],
                                   input_split_sizes=[n * width for n in input_counts], group=self.group)
        else:
            grouped_p2p_exchange(output, input, output_counts, input_counts, group=self.group, rank=self.rank)

    def copy(self, owned: Tensor) -> Tensor:
        """Owned rows into a fresh receive buffer (``sum(receive_counts)`` rows)."""
        send = owned.index_select(0, self.send_rows.to(owned.device)).contiguous()
        receive = owned.new_empty((sum(self.receive_counts), *owned.shape[1:]))
        self._exchange(receive, send, self.receive_counts, self.send_counts)
        return receive

    def reverse_sum(self, cotangent: Tensor, owned_rows: int) -> Tensor:
        """Adjoint of ``copy``: receive-shaped cotangent SUMmed into ``owned_rows`` fresh rows."""
        if cotangent.shape[0] != sum(self.receive_counts):
            raise ValueError("the cotangent must have one row per received row")
        returned = cotangent.new_empty((sum(self.send_counts), *cotangent.shape[1:]))
        self._exchange(returned, cotangent.contiguous(), self.send_counts, self.receive_counts)
        output = cotangent.new_zeros((owned_rows, *cotangent.shape[1:]))
        return output.index_add_(0, self.send_rows.to(cotangent.device), returned)


class DomainCommunicator:
    """Control agreement for one spatial domain's rank; all ranks call in the same order.

    ``group`` is the data group (NCCL, or Gloo in CPU tests), ``control_group``
    the Gloo group used only by slow paths. Buffers (pinned host staging and
    device slots on CUDA) are allocated on first use and kept: a round
    allocates nothing, so its cost is the collective plus one download.
    ``stats`` counts rounds, overflow rounds, contributed pickled bytes and
    Gloo slow-path calls; an ExchangeContext points it at its own stats.
    """

    def __init__(self, group, control_group, *, device=None, capacity: int = DEFAULT_CAPACITY):
        if not (dist.is_available() and dist.is_initialized()):
            raise ValueError("DomainCommunicator requires initialized process groups")
        backend = str(dist.get_backend(group)).lower()
        world_size = dist.get_world_size(group)
        if world_size < 2:
            raise ValueError("DomainCommunicator needs world_size > 1; one rank uses LocalCommunicator")
        if backend == "nccl":
            device = torch.device("cuda") if device is None else torch.device(device)
            if device.type != "cuda":
                raise ValueError("an NCCL data group keeps control buffers on its CUDA device")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        elif backend == "gloo":
            device = torch.device("cpu") if device is None else torch.device(device)
            if device.type != "cpu":
                raise ValueError("a Gloo data group keeps control vectors on the CPU")
        else:
            raise ValueError(f"unsupported control-plane data backend: {backend}")
        if control_group is None or str(dist.get_backend(control_group)).lower() != "gloo":
            raise ValueError("the control-plane slow path requires a Gloo control group")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 8:
            raise ValueError("control slot capacity must be an integer of at least 8 bytes")
        self.group, self.control_group = group, control_group
        self.rank, self.world_size = dist.get_rank(group), world_size
        self.backend, self.device = backend, device
        self.capacity = _round8(capacity)
        self.flag_width = _HEADER + world_size + FLAG_ENTRIES
        self._round = 0
        self._objects = self._flags = self._event = None
        self.stats = {"control_rounds": 0, "control_overflow_rounds": 0,
                      "control_bytes": 0, "control_slow_path_calls": 0}

    # --- buffers and waits ---
    def _buffer(self, count: int, *, host: bool) -> Tensor:
        if self.device.type != "cuda":
            return torch.zeros(count, dtype=torch.long)
        if host:
            staging = torch.zeros(count, dtype=torch.long, pin_memory=True)
            address = staging.untyped_storage().data_ptr()
            _CONTROL_STAGING.add(address)
            # A freed address may be reused by an unrelated pinned tensor.
            weakref.finalize(staging, _CONTROL_STAGING.discard, address)
            return staging
        return torch.zeros(count, dtype=torch.long, device=self.device)

    def _object_buffers(self):
        if self._objects is None or self._objects[0] != self.capacity:
            words = 1 + self.capacity // 8
            host_send = self._buffer(words, host=True)
            host_receive = self._buffer(self.world_size * words, host=True)
            if self.device.type == "cuda":
                send = self._buffer(words, host=False)
                receive = self._buffer(self.world_size * words, host=False)
            else:
                send, receive = host_send, host_receive
            send_view, receive_view = host_send.numpy(), host_receive.numpy()
            # Byte memoryviews of the staging: payloads are copied in, compared
            # and unpickled through slices of these (no per-round numpy views
            # or fancy indexing: ~5 us less per round at 4 ranks).
            self._objects = (self.capacity, send_view, send, receive, host_receive, receive_view,
                             host_send, memoryview(send_view).cast("B"),
                             memoryview(receive_view).cast("B"))
        return self._objects[1:]

    def _flag_buffers(self):
        if self._flags is None:
            staging = self._buffer(self.flag_width, host=True)
            if self.device.type == "cuda":
                vector = self._buffer(self.flag_width, host=False)
                result = self._buffer(self.flag_width, host=True)
            else:
                vector = result = staging
            self._flags = (staging, staging.numpy(), vector, result, result.numpy())
        return self._flags

    def _download(self, host: Tensor, device: Tensor) -> None:
        """One D2H into pinned memory; wait for this stream only (the wait releases the GIL)."""
        if self.device.type != "cuda":
            return
        host.copy_(device, non_blocking=True)
        if self._event is None:
            self._event = torch.cuda.Event()
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()

    def _begin_round(self) -> int:
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("control collective inside a CUDA graph capture: agreement must run "
                               "outside capture (a captured round is recorded, not executed)")
        self._round += 1
        self.stats["control_rounds"] += 1
        return self._round

    # --- transport of ExchangeContext._collective (not part of the protocol) ---
    def _gather_slots(self, payload: bytes, slot: int):
        """One data-group all_gather of ``[length, bytes]`` slots; (lengths, row stride in bytes)."""
        self._begin_round()
        words, size = 1 + slot // 8, len(payload)
        (send_view, send, receive, host_receive, receive_view, host_send,
         send_bytes, _) = self._object_buffers()
        if slot != self.capacity:
            # Overflow round: exact-size prefixes of the grown buffers. The
            # steady state uses the whole buffers (per-round host cost is
            # dominated by launches; no view creation on that path).
            send, host_send = send[:words], host_send[:words]
            receive = receive[:self.world_size * words]
            host_receive = host_receive[:self.world_size * words]
        send_view[0] = size
        if size <= slot:
            send_bytes[8:8 + size] = payload
        if self.device.type == "cuda":
            send.copy_(host_send, non_blocking=True)
        dist.all_gather_into_tensor(receive, send, group=self.group)
        self._download(host_receive, receive)
        # Row r starts at int64 word r * words: its length word, then bytes.
        lengths = receive_view[0:self.world_size * words:words].tolist()
        if min(lengths) < 0:
            raise RuntimeError(f"corrupt control payload lengths: {lengths}")
        return lengths, 8 * words

    def gather_object(self, value) -> list:
        """``dist.all_gather_object`` over the data group: every rank's value, in rank order."""
        _saved_recompute.check_no_collective("gather_object")
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        size = len(payload)
        self.stats["control_bytes"] += size
        slot = self.capacity
        lengths, stride = self._gather_slots(payload, slot)
        largest = max(lengths)
        if largest > slot:
            # Every rank received the same lengths, so every rank is here:
            # one exact-size round, then a power-of-two slot for later rounds.
            self.stats["control_overflow_rounds"] += 1
            self.capacity = 1 << (largest - 1).bit_length()
            slot = _round8(largest)
            lengths, stride = self._gather_slots(payload, slot)
            if max(lengths) > slot:
                raise RuntimeError("control payload grew between the overflow rounds")
        received = self._objects[-1]
        # A collective always returns this rank's own bytes in its row. A
        # round that an abort released without a transfer (a peer failed)
        # leaves the previous round's rows in the staging instead: never
        # return those as this round's agreement. tobytes() makes it one
        # memcpy + memcmp (~0.4 us at 9 KB): comparing the memoryview with
        # bytes directly unpacks element by element (~11 us).
        own = self.rank * stride + 8
        if lengths[self.rank] != size or received[own:own + size].tobytes() != payload:
            raise RuntimeError(
                "control round returned stale data: this rank's own row is not what it sent "
                "(a peer failed or aborted the process group; a secondary error)")
        # Unpickling copies out of the staging rows (no out-of-band buffers).
        return [pickle.loads(received[start:start + length])
                for start, length in zip(range(8, self.world_size * stride, stride), lengths)]

    # --- Communicator protocol ---
    def agree_flags_host(self, flags: Tensor | None = None, *, phase: str, host=()) -> Tensor:
        """MAX over ranks of ``[*host, *flags]`` in ONE all_reduce; a CPU int64 vector.

        ``host`` entries travel in the same pinned upload as the header;
        ``flags`` (on the control device, e.g. device predicates, or the CPU)
        are copied on the current stream, never downloaded on their own.
        """
        _saved_recompute.check_no_collective("agree_flags_host")
        if not isinstance(phase, str):
            raise TypeError("flag round phase must be a string")
        flags, values = flag_parts(flags, host)
        count = len(values) + (0 if flags is None else flags.numel())
        header = _HEADER + self.world_size
        if count > FLAG_ENTRIES:
            raise ValueError(f"a flag round holds at most {FLAG_ENTRIES} entries, got {count}")
        if flags is not None and flags.device not in (self.device, torch.device("cpu")):
            raise ValueError("flags must be on the control device or the CPU")
        number = self._begin_round()
        code = _phase_code(phase)
        staging, staging_view, vector, result, result_view = self._flag_buffers()
        staging_view[:_HEADER] = (code, -code, count, -count, number, -number)
        staging_view[_HEADER:] = 0
        staging_view[_HEADER + self.rank] = number
        staging_view[header:header + len(values)] = values
        if self.device.type == "cuda":
            vector.copy_(staging, non_blocking=True)
        if flags is not None and flags.numel():
            vector[header + len(values):header + count].copy_(flags)
        dist.all_reduce(vector, op=dist.ReduceOp.MAX, group=self.group)
        self._download(result, vector)
        agreed = result_view.copy()
        # A slot below this round's number: that rank's contribution is
        # missing (a reduction released by an abort returns this rank's own
        # staging, whose header alone is self-consistent) or from another round.
        slots = agreed[_HEADER:header]
        if (agreed[0] != -agreed[1] or agreed[2] != -agreed[3] or agreed[4] != -agreed[5]
                or (slots != number).any()):
            raise RuntimeError(
                f"distributed protocol phase mismatch in flag round {phase!r}: phase codes "
                f"{-agreed[1]}..{agreed[0]}, lengths {-agreed[3]}..{agreed[2]}, "
                f"rounds {-agreed[5]}..{agreed[4]}, per-rank rounds {slots.tolist()} (this rank: {number})")
        return torch.from_numpy(agreed[header:header + count])

    def errors_slow_path(self, error: str | None, flag_any: bool, *, phase: str = "errors") -> list:
        """Every rank's error string over Gloo, only after an agreed flag reported a failure.

        ``flag_any`` must be the agreed (replicated) flag, so every rank enters
        or skips this Gloo round together. The strings let callers raise the
        historical per-rank messages, identically on every rank.
        """
        if not flag_any:
            return [None] * self.world_size
        self.stats["control_slow_path_calls"] += 1
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, (phase, error), group=self.control_group)
        phases = [item[0] for item in gathered]
        if any(item != phases[0] for item in phases):
            raise RuntimeError(f"distributed protocol phase mismatch: {phases}")
        return [item[1] for item in gathered]

    def check_config(self, fingerprint: str) -> None:
        """Compare SHA-256 digests over the data group; full strings over Gloo only on a mismatch."""
        if not isinstance(fingerprint, str):
            raise TypeError("configuration fingerprint must be a string")
        digests = self.gather_object(("config", hashlib.sha256(fingerprint.encode()).digest()))
        if all(item == digests[0] for item in digests):
            return
        names = self.errors_slow_path(fingerprint, True, phase="config")
        raise ValueError("configuration fingerprint differs between ranks: " + "; ".join(
            f"rank {rank}: {name}" for rank, name in enumerate(names)))

    def agree_capacity(self, demand, policy, current):
        """``(key, recapture_any)`` from every rank's demand via ``gather_object``.

        A 3.1a stand-in with no caller: step 3.2's ``CaptureCoordinator``
        (fastmd/engine/capture.py) agrees the capacity demand, entry state and
        recapture votes in ONE ``ExchangeContext._collective`` round instead,
        which travels by ``gather_object``.
        """
        gathered = self.gather_object(("capacity", current, demand))
        if any(item[1] != current for item in gathered):
            raise RuntimeError("capacity keys differ across ranks")
        key = policy([item[2] for item in gathered], current)
        return key, key != current

    def fatal(self, exc: BaseException):
        from .launch import fail_fast
        fail_fast(exc, rank=self.rank)

    def _check_vector(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor) or tensor.device != self.device:
            raise ValueError("step reductions need a tensor on the control device")

    def allreduce_sum_(self, vec: Tensor) -> None:
        _saved_recompute.check_no_collective("allreduce_sum_")
        self._check_vector(vec)
        dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=self.group)

    def allreduce_max_(self, flags: Tensor) -> None:
        _saved_recompute.check_no_collective("allreduce_max_")
        self._check_vector(flags)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self.group)

    def slice_exchange(self, send_rows: Tensor, send_counts, receive_counts, *,
                       transport: str = "neighbor_p2p") -> SliceExchange:
        """Epoch-frozen per-peer slice COPY / reverse-SUM on the data group (``SliceExchange``)."""
        _saved_recompute.check_no_collective("slice_exchange")
        return SliceExchange(send_rows, send_counts, receive_counts, group=self.group, transport=transport)

    def gather_by_id(self, root: int, ids: Tensor, **fields) -> dict | None:
        """Every rank's rows on ``root`` in stable-ID order (MD log/restart steps; unified step 3.3).

        One flag round agrees the largest row count, input errors, the field
        schema (code and negation: max(-x) = -min(x)) and the integer bound
        (IDs in [0, 2**53), other integers below 2**53 in magnitude: exact in
        float64). Then each rank packs ``[id, *fields]`` as float64 columns
        into that many rows (padding ID -1) and ONE data-group
        ``all_gather_into_tensor`` delivers them; root drops the padding,
        sorts by ID and restores the dtypes. Gloo carries only the error
        strings after a reported failure. Synchronizes: not a step hook.
        """
        error = gather_fields(ids, fields)
        if error is None and (isinstance(root, bool) or not isinstance(root, int)
                              or not 0 <= root < self.world_size):
            error = f"gather_by_id: root must be a rank of the group, got {root!r}"
        schema, bound = "", torch.zeros(1, dtype=torch.bool, device=self.device)
        if error is None:
            ids = ids.to(self.device)
            fields = {name: value.to(self.device) for name, value in fields.items()}
            schema = repr((root, str(ids.dtype), [(name, str(value.dtype), tuple(value.shape[1:]))
                                                  for name, value in fields.items()]))
            # Only int64 holds magnitudes of 2**53 (a narrower dtype would wrap the limit).
            limit = 1 << 53
            bound |= (ids < 0).any()
            for value in (ids, *fields.values()):
                if value.dtype == torch.int64:
                    bound |= (value.abs() >= limit).any()
        code = _phase_code(schema)
        n = 0 if error is not None else ids.shape[0]
        agreed = self.agree_flags_host(bound, phase="gather_by_id",
                                       host=(error is not None, n, code, -code))
        if agreed[0] or agreed[4]:
            if error is None and bool(bound.item()):
                error = "gather_by_id: IDs must be in [0, 2**53) and integer fields below 2**53"
            errors = self.errors_slow_path(error, True, phase="gather_by_id")
            raise ValueError("gather_by_id failed: " + "; ".join(
                f"rank {rank}: {message}" for rank, message in enumerate(errors) if message))
        if agreed[2] != -agreed[3]:
            raise ValueError("gather_by_id: field names, dtypes, shapes or root differ across ranks")
        rows = int(agreed[1])
        widths = [math.prod(value.shape[1:]) for value in fields.values()]
        send = torch.full((rows, 1 + sum(widths)), -1.0, dtype=torch.float64, device=self.device)
        if n:
            send[:n] = torch.cat([ids.reshape(n, 1).to(torch.float64)] + [
                value.reshape(n, width).to(torch.float64) for value, width in zip(fields.values(), widths)], 1)
        receive = send.new_empty((self.world_size * rows, send.shape[1]))
        if rows:
            dist.all_gather_into_tensor(receive, send, group=self.group)
        if self.rank != root:
            return None
        receive = receive[receive[:, 0] >= 0]
        receive = receive[torch.argsort(receive[:, 0], stable=True)]
        result, offset = {"ids": receive[:, 0].to(ids.dtype)}, 1
        for (name, value), width in zip(fields.items(), widths):
            result[name] = receive[:, offset:offset + width].to(value.dtype).reshape(-1, *value.shape[1:])
            offset += width
        return result

    def begin_epoch(self, owned, per_atom_state, *, halo_cutoff):
        raise NotImplementedError(NOT_YET)

    def atom_copy_binding(self, name):
        raise NotImplementedError(NOT_YET)

    def row_route_binding(self, name):
        raise NotImplementedError(NOT_YET)

    def ghost_copy_(self, rows):
        raise NotImplementedError(NOT_YET)

    def ghost_reverse_sum_(self, rows):
        raise NotImplementedError(NOT_YET)
