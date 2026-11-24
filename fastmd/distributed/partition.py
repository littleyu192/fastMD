"""Spatial ownership and coordinate-only exchanges for one periodic system.

No operation here gathers the global coordinates.  Only counts and field schemas
are collective CPU metadata; atom records are sent to their destination ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import time
from typing import Any

import torch
from torch import Tensor
import torch.distributed as dist

from fastmd.graph.geometry import small_matmul


# Native NCCL scalar types plus complex types that PyTorch transports through
# view_as_real. Unsupported integer/float8 records must fail collectively before
# the first packet, not while one rank is already inside all_to_all_single.
_CUDA_RECORD_DTYPES = frozenset((torch.bool, torch.int8, torch.uint8, torch.int32,
                                torch.int64, torch.float16, torch.bfloat16,
                                torch.float32, torch.float64, torch.complex64,
                                torch.complex128))


@dataclass
class AtomState:
    ids: Tensor
    positions: Tensor
    numbers: Tensor
    payload: dict[str, Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if isinstance(self.positions, Tensor) and self.positions.is_cuda:
            self._validate_cuda()
            return
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        n = self.positions.shape[0]
        if not self.positions.is_floating_point() or not torch.isfinite(self.positions).all():
            raise ValueError("positions must be finite floating-point values")
        for name, value in (("ids", self.ids), ("numbers", self.numbers)):
            if value.shape != (n,) or value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer tensor of shape (N,)")
        if len(set(self.ids.detach().cpu().tolist())) != n:
            raise ValueError("local atom IDs must be unique")
        if n and (self.ids.min().item() < 0 or self.numbers.min().item() < 1):
            raise ValueError("atom IDs must be nonnegative and atomic numbers positive")
        for name, value in self.payload.items():
            if not isinstance(name, str) or name in {"ids", "numbers", "positions"}:
                raise ValueError("payload names must be strings other than reserved atom fields")
            if not isinstance(value, Tensor) or value.ndim == 0 or value.shape[0] != n:
                raise ValueError(f"payload {name!r} must have leading dimension N")

    def _validate_cuda(self) -> None:
        """Validate device data with only four scalar flags returning to CPU."""
        positions, ids, numbers = self._validate_cuda_host()
        self._raise_cuda_flags(torch.stack(self._cuda_flags(positions, ids, numbers)).cpu().tolist())

    @staticmethod
    def _cuda_flags(positions: Tensor, ids: Tensor, numbers: Tensor,
                    sorted_ids: Tensor | None = None) -> list[Tensor]:
        """Device predicates of CUDA validation, for one packed download."""
        if sorted_ids is None:
            sorted_ids = ids.sort().values
        # max |x| propagates NaN and is inf for +-inf: two launches instead of
        # isfinite().all()'s five (RDX 1,344: packed preflight -9%). An
        # empty norm raises; nothing is non-finite then.
        finite = (torch.linalg.vector_norm(positions, math.inf) < math.inf if positions.numel()
                  else torch.isfinite(positions).all())
        return [finite, (sorted_ids[1:] == sorted_ids[:-1]).any(), (ids < 0).any(), (numbers < 1).any()]

    @staticmethod
    def _raise_cuda_flags(flags) -> None:
        if not flags[0]:
            raise ValueError("positions must be finite floating-point values")
        if flags[1]:
            raise ValueError("local atom IDs must be unique")
        if flags[2] or flags[3]:
            raise ValueError("atom IDs must be nonnegative and atomic numbers positive")

    def _validate_cuda_host(self) -> tuple[Tensor, Tensor, Tensor]:
        """Host-side CUDA checks; returns positions, ids and numbers on its device."""
        positions = self.positions
        if positions.layout != torch.strided or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        n = positions.shape[0]
        if not positions.is_floating_point():
            raise ValueError("positions must be finite floating-point values")
        if positions.dtype not in (torch.float32, torch.float64):
            raise ValueError("CUDA positions must use float32 or float64")
        for name, value in (("ids", self.ids), ("numbers", self.numbers)):
            if (not isinstance(value, Tensor) or value.layout != torch.strided
                    or value.shape != (n,) or value.dtype not in (torch.int32, torch.int64)):
                raise ValueError(f"{name} must be an integer tensor of shape (N,)")
        for name, value in self.payload.items():
            if not isinstance(name, str) or name in {"ids", "numbers", "positions"}:
                raise ValueError("payload names must be strings other than reserved atom fields")
            if not isinstance(value, Tensor) or value.ndim == 0 or value.shape[0] != n:
                raise ValueError(f"payload {name!r} must have leading dimension N")
            if value.layout != torch.strided or value.dtype not in _CUDA_RECORD_DTYPES:
                raise ValueError(f"CUDA payload {name!r} must use dense NCCL-supported dtype")
        return positions, self.ids.to(positions.device), self.numbers.to(positions.device)

    def select(self, indices: Tensor) -> AtomState:
        def take(value: Tensor) -> Tensor:
            return value.index_select(0, indices.to(value.device))
        return AtomState(take(self.ids), take(self.positions), take(self.numbers),
                         {key: take(value) for key, value in self.payload.items()})

    def sorted(self) -> AtomState:
        return self.select(torch.argsort(self.ids))

    @property
    def atomic_numbers(self) -> Tensor:
        return self.numbers


class DomainPartition:
    """Fixed fractional-coordinate cuts of a nonsingular, fully periodic cell.

    For a skew cell, distances to its fractional-coordinate planes give a
    conservative halo. The model's graph builder applies the exact Cartesian
    cutoff to these candidates, retaining the original lattice and images.
    """

    def __init__(self, cell: Tensor, world_size: int, domain_grid: str | tuple[int, int, int] = "auto",
                 pbc: tuple[bool, bool, bool] = (True, True, True)):
        cell = torch.as_tensor(cell)
        if cell.shape != (3, 3) or not cell.is_floating_point() or not torch.isfinite(cell).all():
            raise ValueError("cell must be a finite floating-point 3 x 3 tensor")
        if tuple(bool(x) for x in pbc) != (True, True, True):
            raise ValueError("distributed MVP requires full periodicity on all three axes")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        check = cell.detach().cpu().double()
        lengths = check.norm(dim=1)
        if (lengths <= 0).any():
            raise ValueError("cell lengths must be positive")
        unit = check / lengths[:, None]
        if not bool(torch.isfinite(torch.linalg.det(check))) or torch.linalg.det(check) == 0:
            raise ValueError("cell must be nonsingular")
        self.orthogonal = torch.allclose(unit @ unit.T, torch.eye(3, dtype=torch.float64), atol=1e-7, rtol=0)
        # Rows are lattice vectors; column i of the inverse is the gradient
        # of fractional coordinate i. Its reciprocal norm is the separation
        # of the f_i=0 and f_i=1 planes (not the length of lattice vector i).
        self.face_distances = torch.linalg.inv(check).norm(dim=0).reciprocal()
        self.cell = cell.detach().clone()
        self.lengths = lengths
        self.world_size = world_size
        if domain_grid == "auto":
            # Least domain surface (sum of grid/length over the axes), then
            # more cuts along x, y first: 1/2/4 ranks as before; for a cubic
            # box 8 -> 2x2x2, 16 -> 4x2x2, 32 -> 4x4x2.
            choices = [(x, y, world_size // (x * y))
                       for x in range(1, world_size + 1)
                       for y in range(1, world_size + 1) if world_size % (x * y) == 0]
            lens = self.face_distances.tolist()
            self.domain_grid = min(choices, key=lambda g: (
                sum(g[i] / lens[i] for i in range(3)), tuple(-x for x in g)))
        else:
            if not isinstance(domain_grid, (tuple, list)) or len(domain_grid) != 3:
                raise ValueError("domain_grid must be 'auto' or three positive integers")
            if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in domain_grid):
                raise ValueError("domain_grid entries must be positive integers")
            if math.prod(domain_grid) != world_size:
                raise ValueError("domain_grid product must equal world_size")
            self.domain_grid = tuple(domain_grid)

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.domain_grid

    def fractional(self, positions: Tensor) -> Tensor:
        # small_matmul, as in the single-GPU builders: a cuBLAS product runs in
        # TF32 whenever allow_tf32 is set (NGC TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1)
        # and moves ownership, halo and wrapped coordinates by ~1e-2 A.
        cell = self.cell.to(device=positions.device, dtype=positions.dtype)
        return torch.remainder(small_matmul(positions, torch.linalg.inv(cell)), 1.0)

    def wrap_positions(self, positions: Tensor) -> Tensor:
        return small_matmul(self.fractional(positions),
                            self.cell.to(device=positions.device, dtype=positions.dtype))

    def owner_of(self, positions: Tensor) -> Tensor:
        grid = torch.tensor(self.domain_grid, device=positions.device, dtype=positions.dtype)
        ijk = torch.floor(self.fractional(positions) * grid).long()
        # Guard rounded fractional*grid at the upper endpoint.
        ijk = torch.minimum(ijk, grid.long() - 1)
        return (ijk[:, 0] * self.domain_grid[1] + ijk[:, 1]) * self.domain_grid[2] + ijk[:, 2]

    def rank_indices(self, rank: int) -> tuple[int, int, int]:
        if not 0 <= rank < self.world_size:
            raise ValueError("rank outside the partition")
        ny, nz = self.domain_grid[1:]
        return rank // (ny * nz), (rank // nz) % ny, rank % nz

    def check_halo(self, halo: float) -> None:
        """Fail fast unless ``halo`` (rc + skin) fits one periodic image per axis.

        A policy of the >4-rank engine (design §6.3: a ghost slice holds one
        periodic image of an atom per axis), not a limit of today's halo:
        its candidates (images -1, 0, +1 of each axis) are complete for any
        cutoff, and the local builder enumerates every image of a record.
        """
        if not math.isfinite(halo) or halo <= 0:
            raise ValueError("halo width must be finite and positive")
        for axis, length in enumerate(self.face_distances.tolist()):
            if halo > length:
                raise ValueError(
                    f"halo width {halo:g} A (cutoff + skin) exceeds the box face spacing {length:g} A along "
                    f"axis {'xyz'[axis]}: the distributed engine allows one periodic image per axis; "
                    f"use a supercell with every face spacing >= {halo:g} A")

    def neighbor_ranks(self, rank: int, halo: float) -> tuple[int, ...]:
        """Peer domains within ``halo`` of ``rank``'s domain (periodic), ascending; never ``rank``.

        The set of peers a halo exchange of width ``halo`` can involve, in
        both directions (the box distance is symmetric). Pure geometry of the
        grid, identical on every rank: up to 26 peers when every domain is at
        least ``halo`` wide; thinner domains add farther peers. The margin is
        wider than halo_destinations', so the set contains every peer that
        exchanges halo rows.
        """
        self.check_halo(halo)
        margin = max(1e-5, 64 * torch.finfo(torch.float32).eps * max(float(self.lengths.max()), halo))
        mine = self.rank_indices(rank)
        result = []
        for peer in range(self.world_size):
            if peer == rank:
                continue
            theirs = self.rank_indices(peer)
            squared = 0.0
            for axis in range(3):
                spacing = self.lengths if self.orthogonal else self.face_distances
                n, length = self.domain_grid[axis], float(spacing[axis])
                lo, hi = mine[axis] / n, (mine[axis] + 1) / n
                peer_lo, peer_hi = theirs[axis] / n, (theirs[axis] + 1) / n
                gap = min(max(0.0, peer_lo + shift - hi, lo - (peer_hi + shift)) for shift in (-1, 0, 1))
                distance = (gap * length) ** 2
                squared = squared + distance if self.orthogonal else max(squared, distance)
            if squared <= (halo + margin) ** 2:
                result.append(peer)
        return tuple(result)

    def halo_destinations(self, positions: Tensor, cutoff: float) -> list[Tensor]:
        """Conservative distance-to-periodic-domain candidates, one list per peer."""
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        fractions = self.fractional(positions).detach()
        # This margin only broadens halo candidates, never the final graph cutoff.
        eps = torch.finfo(positions.dtype).eps
        margin = max(1e-6, 32 * eps * max(float(self.lengths.max()), cutoff))
        if positions.is_cuda:
            squared = self._squared_domain_distances(fractions)
            return [torch.where(squared[peer] <= (cutoff + margin) ** 2)[0]
                    for peer in range(self.world_size)]
        fractions = fractions.cpu().double()
        result = []
        for rank in range(self.world_size):
            ijk = self.rank_indices(rank)
            squared = torch.zeros(len(fractions), dtype=torch.float64)
            for axis in range(3):
                lo = ijk[axis] / self.domain_grid[axis]
                hi = (ijk[axis] + 1) / self.domain_grid[axis]
                x = fractions[:, axis]
                distances = [torch.maximum(torch.maximum(lo - (x + shift), (x + shift) - hi),
                                           torch.zeros_like(x)) for shift in (-1, 0, 1)]
                spacing = self.lengths if self.orthogonal else self.face_distances
                distance = torch.stack(distances).amin(0).square() * spacing[axis].square()
                squared = squared + distance if self.orthogonal else torch.maximum(squared, distance)
            result.append(torch.where(squared <= (cutoff + margin) ** 2)[0])
        return result

    def halo_mask(self, positions: Tensor, cutoff: float, rank: int) -> Tensor:
        """``halo_destinations`` of CUDA ``positions`` as a ``[world_size, N]`` bool mask, on the device.

        The same decisions (arithmetic and margin) without building one index
        list per peer; the row of ``rank`` itself is False.
        """
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not positions.is_cuda:
            raise ValueError("halo_mask needs CUDA positions")
        eps = torch.finfo(positions.dtype).eps
        margin = max(1e-6, 32 * eps * max(float(self.lengths.max()), cutoff))
        mask = self._squared_domain_distances(self.fractional(positions).detach()) <= (cutoff + margin) ** 2
        mask[rank] = False
        return mask

    def _squared_domain_distances(self, fractions: Tensor) -> Tensor:
        """Squared distance lower bounds to every domain (exact for orthogonal cells).

        Each face distance is at most the Cartesian distance to the domain.
        Their maximum therefore cannot omit a neighbor in a triclinic cell;
        summing their squares would incorrectly exclude some skew corners.
        """
        # Preserve the original arithmetic: fractional wrapping in the
        # input dtype, followed by FP64 distance-to-domain calculations.
        # Vectorize peers, but accumulate x/y/z in the old order. Neither
        # positions nor fractional coordinates leave the device.
        device = fractions.device
        fractions = fractions.double()
        lengths = (self.lengths if self.orthogonal else self.face_distances).to(device=device)
        lower = torch.tensor([[self.rank_indices(peer)[axis] / self.domain_grid[axis]
                               for axis in range(3)] for peer in range(self.world_size)],
                             dtype=torch.float64, device=device)
        upper = torch.tensor([[(self.rank_indices(peer)[axis] + 1) / self.domain_grid[axis]
                               for axis in range(3)] for peer in range(self.world_size)],
                             dtype=torch.float64, device=device)
        squared = torch.zeros((self.world_size, len(fractions)), dtype=torch.float64, device=device)
        for axis in range(3):
            x = fractions[:, axis][None, :]
            distances = [torch.maximum(torch.maximum(lower[:, axis, None] - (x + shift),
                                                       (x + shift) - upper[:, axis, None]),
                                       torch.zeros_like(x)) for shift in (-1, 0, 1)]
            distance = torch.stack(distances).amin(0).square() * lengths[axis].square()
            squared = squared + distance if self.orthogonal else torch.maximum(squared, distance)
        return squared


def _collective_check(context: Any, error: str | None) -> None:
    if hasattr(context, "collective_check"):
        context.collective_check(error)
    elif error is not None:
        raise ValueError(error)


def _record_payload(context: Any, tensor: Tensor, send_counts: list[int],
                    recv_counts: list[int], started: float, *, synchronize: bool = True) -> None:
    if tensor.is_cuda and synchronize:
        # There is deliberately no communication/computation overlap in MVP.
        torch.cuda.synchronize(tensor.device)
    if not hasattr(context, "stats"):
        return
    row_bytes = math.prod(tensor.shape[1:]) * tensor.element_size()
    for peer in range(context.world_size):
        if peer != context.rank:
            sent, received = send_counts[peer] * row_bytes, recv_counts[peer] * row_bytes
            context.stats["peer_sent_bytes"][peer] += sent
            context.stats["peer_received_bytes"][peer] += received
            context.stats["payload_sent_bytes"] += sent
            context.stats["payload_received_bytes"] += received
    context.stats["payload_calls"] += 1
    context.stats["payload_seconds"] += time.perf_counter() - started


def _data_all_to_all(context: Any, output: Tensor, input: Tensor, output_counts, input_counts,
                     *, control: bool = False) -> None:
    """Per-peer row splits on the data (or control) group through the context's payload transport."""
    if hasattr(context, "all_to_all_single"):
        context.all_to_all_single(output, input, output_counts, input_counts, control=control)
    else:
        dist.all_to_all_single(output, input, output_split_sizes=list(output_counts),
                               input_split_sizes=list(input_counts),
                               group=context.control_group if control else context.group)


def _validate_backend(state: AtomState, context: Any) -> None:
    backend = getattr(context, "backend", None)
    if backend == "nccl" and not state.positions.is_cuda:
        raise ValueError("NCCL atom records must use CUDA positions")
    if backend == "gloo" and state.positions.is_cuda:
        raise ValueError("Gloo atom records must use CPU positions")
    if state.positions.is_cuda and state.positions.device.index != torch.cuda.current_device():
        raise ValueError("atom records must use the current worker CUDA device")


def _record_admission(context, error, schema, counts):
    """One collective validates errors, typed field schema and packet counts."""
    value = (error, schema, counts)
    if hasattr(context, "_collective"):
        metadata = context._collective(value, "partition.records")
        context._raise_errors([item[0] for item in metadata])
    else:
        metadata = context.all_gather_object(value) if context.world_size > 1 else [value]
        _collective_check(context, next((item[0] for item in metadata if item[0]), None))
    if any(item[1] != metadata[0][1] for item in metadata):
        raise RuntimeError("atom record schemas differ across ranks")
    return [item[2][context.rank] for item in metadata]


def _exchange_records_cuda(state: AtomState, destinations: list[Tensor], context: Any,
                           *, state_prevalidated: bool, complete: bool) -> AtomState:
    """Typed device packets with one schema check and no per-field GPU wait.

    Fields of different dtypes never share a numeric cast or packet. Grouping
    homogeneous dtypes preserves int64 IDs/metadata and floating-point payload
    bit patterns. Output sorting materializes independent per-field storage.
    PyTorch's synchronous NCCL API orders consumers on the current stream;
    completion is observed by the caller's subsequent validation/use, not by
    synchronizing the entire device once per field.
    """
    device = state.positions.device
    fields, schema, send_counts, indices = {}, (), (), None
    error = None
    try:
        if not state_prevalidated:
            state.validate()
        _validate_backend(state, context)
        if len(destinations) != context.world_size:
            raise ValueError("one destination index array per rank is required")
        for value in destinations:
            if (not isinstance(value, Tensor) or value.layout != torch.strided or value.ndim != 1
                    or value.dtype not in (torch.int32, torch.int64)):
                raise ValueError("destination indices must be dense one-dimensional integer tensors")
        indices = torch.cat([value.to(device=device, dtype=torch.long) for value in destinations])
        send_counts = tuple(value.numel() for value in destinations)
        if complete and sum(send_counts) != len(state.ids):
            raise ValueError("spatial ownership did not assign every atom exactly once")
        if bool(((indices < 0) | (indices >= len(state.ids))).any()):
            raise ValueError("destination atom index is out of bounds")
        fields = {"ids": state.ids.to(device=device, dtype=torch.int64),
                  "positions": state.positions,
                  "numbers": state.numbers.to(device=device, dtype=torch.int64),
                  **{key: value.to(device) for key, value in state.payload.items()}}
        for key, value in fields.items():
            if value.layout != torch.strided:
                raise ValueError(f"atom field {key!r} must use dense strided storage")
            if value.dtype not in _CUDA_RECORD_DTYPES:
                raise ValueError(f"atom field {key!r} must use NCCL-supported dtype")
        schema = tuple((key, str(fields[key].dtype), tuple(fields[key].shape[1:]))
                       for key in sorted(fields))
    except (ValueError, TypeError, RuntimeError, IndexError, AttributeError) as exc:
        error = str(exc)
    recv_counts = _record_admission(context, error, schema, send_counts)
    groups = {}
    for key in sorted(fields):
        groups.setdefault(fields[key].dtype, []).append(key)
    received = {}
    for keys in groups.values():
        started = time.perf_counter()
        widths = [math.prod(fields[key].shape[1:]) for key in keys]
        packed = torch.cat([fields[key].reshape(len(state.ids), width)
                            for key, width in zip(keys, widths)], dim=1)
        outgoing = packed.index_select(0, indices).contiguous()
        incoming = outgoing.new_empty((sum(recv_counts), sum(widths)))
        if context.world_size == 1:
            incoming.copy_(outgoing)
        else:
            _data_all_to_all(context, incoming, outgoing, recv_counts, send_counts)
        _record_payload(context, outgoing, list(send_counts), recv_counts, started, synchronize=False)
        offset = 0
        for key, width in zip(keys, widths):
            received[key] = incoming[:, offset:offset + width].reshape(
                sum(recv_counts), *fields[key].shape[1:])
            offset += width
    return AtomState(received.pop("ids"), received.pop("positions"),
                     received.pop("numbers"), received).sorted()


def _exchange_records(state: AtomState, destinations: list[Tensor], context: Any, *,
                      _state_prevalidated: bool = False, _complete: bool = False) -> AtomState:
    """Variable-sized record all-to-all; every receive owns fresh storage."""
    if state.positions.is_cuda:
        return _exchange_records_cuda(state, destinations, context,
                                      state_prevalidated=_state_prevalidated, complete=_complete)
    rank, world = context.rank, context.world_size
    error = None
    try:
        state.validate()
        if len(destinations) != world:
            raise ValueError("one destination index array per rank is required")
    except (ValueError, TypeError, IndexError) as exc:
        error = str(exc)
    _collective_check(context, error)
    device = state.positions.device
    fields = {"ids": state.ids.to(device=device, dtype=torch.int64),
              "positions": state.positions, "numbers": state.numbers.to(device=device, dtype=torch.int64),
              **{key: value.to(device) for key, value in state.payload.items()}}
    schema = [(key, str(fields[key].dtype), tuple(fields[key].shape[1:])) for key in sorted(fields)]
    send_counts = [int(idx.numel()) for idx in destinations]
    metadata = {"schema": schema, "counts": send_counts}
    all_meta = context.all_gather_object(metadata) if world > 1 else [metadata]
    _collective_check(context, None if all(x["schema"] == schema for x in all_meta)
                      else "atom record schemas differ across ranks")
    recv_counts = [item["counts"][rank] for item in all_meta]
    indices = torch.cat([idx.to(device=device, dtype=torch.long) for idx in destinations])
    received = {}
    for key in sorted(fields):
        started = time.perf_counter()
        outgoing = fields[key].index_select(0, indices).contiguous()
        incoming = outgoing.new_empty((sum(recv_counts), *outgoing.shape[1:]))
        if world == 1:
            incoming.copy_(outgoing)
        else:
            _data_all_to_all(context, incoming, outgoing, recv_counts, send_counts)
        _record_payload(context, outgoing, send_counts, recv_counts, started)
        received[key] = incoming
    return AtomState(received.pop("ids"), received.pop("positions"),
                     received.pop("numbers"), received).sorted()


def _validate_global_ids_cuda(ids: Tensor, context: Any) -> None:
    """Distributed uniqueness check, NOT physical atom ownership.

    Every stable ID is sent once to its deterministic ``ID % world`` validator.
    Only counts and one duplicate flag are host metadata. Unrelated IDs may
    share a validator; sorting detects equality without lossy integer packing
    or gathering all IDs on every rank. Authoritative state is never changed.
    """
    world = context.world_size
    if world == 1:
        checked = ids
    else:
        validators = ids.remainder(world)
        counts = torch.bincount(validators, minlength=world).cpu().tolist()
        matrix = context.all_gather_object(counts)
        receive_counts = [item[context.rank] for item in matrix]
        order = validators.argsort(stable=True)
        outgoing = ids.index_select(0, order).contiguous()
        checked = ids.new_empty(sum(receive_counts))
        started = time.perf_counter()
        dist.all_to_all_single(checked, outgoing, output_split_sizes=receive_counts,
                               input_split_sizes=counts, group=context.group)
        _record_payload(context, outgoing, counts, receive_counts, started, synchronize=False)
    ordered = checked.sort().values
    duplicate = bool((ordered[1:] == ordered[:-1]).any())
    _collective_check(context, "global atom IDs must have exactly one authoritative owner"
                      if duplicate else None)


def migrate_atoms(state: AtomState, partition: DomainPartition, context: Any) -> AtomState:
    """Move authoritative records; preserve unwrapped positions and MD payload."""
    error = None
    try:
        state.validate()
        _validate_backend(state, context)
        if context.world_size != partition.world_size:
            raise ValueError("partition and process-group world sizes differ")
        owners = partition.owner_of(state.positions)
        destinations = [torch.where(owners == peer)[0] for peer in range(context.world_size)]
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        error = str(exc)
    _collective_check(context, error)
    moved = _exchange_records(state, destinations, context,
                               _state_prevalidated=True, _complete=True)
    if moved.positions.is_cuda:
        _validate_global_ids_cuda(moved.ids, context)
        return moved
    # IDs alone are allowed control metadata and detect duplicate authoritative owners.
    local_ids = moved.ids.detach().cpu().tolist()
    all_ids = context.all_gather_object(local_ids) if context.world_size > 1 else [local_ids]
    flattened = list(itertools.chain.from_iterable(all_ids))
    _collective_check(context, None if len(set(flattened)) == len(flattened)
                      else "global atom IDs must have exactly one authoritative owner")
    return moved


def exchange_halo(state: AtomState, partition: DomainPartition, context: Any, cutoff: float) -> AtomState:
    """Return unique owned+halo coordinates; no MD payload is replicated."""
    error = None
    try:
        state.validate()
        _validate_backend(state, context)
        if partition.world_size != context.world_size:
            raise ValueError("partition and process-group world sizes differ")
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive")
        if not (partition.owner_of(state.positions) == context.rank).all():
            raise ValueError("migrate_atoms must establish spatial ownership before halo exchange")
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        error = str(exc)
    _collective_check(context, error)
    # Producer rows refer to the current authoritative, ID-sorted atom table.
    # They must be regenerated after every migration, even when the row count
    # is unchanged. Never propagate a previous halo's provenance from payload.
    # Sorting a new state also leaves the caller's unwrapped MD state untouched.
    producer = AtomState(state.ids, state.positions, state.numbers).sorted()
    coordinates = AtomState(producer.ids, partition.wrap_positions(producer.positions), producer.numbers,
                            {"_owner": torch.full((len(producer.ids),), context.rank,
                                                  dtype=torch.int64, device=producer.positions.device),
                             "_owner_row": torch.arange(len(producer.ids), dtype=torch.int64,
                                                        device=producer.positions.device)})
    result = _exchange_records(coordinates, partition.halo_destinations(producer.positions, cutoff), context)
    error = None
    try:
        result.validate()
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        error = str(exc)
    _collective_check(context, error)
    return result
