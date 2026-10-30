"""Structure relaxation of many systems in the slots of whole-step CUDA graphs: ASE's FIRE per system, with a
fixed cell or ASE's FrechetCellFilter, the structures streaming through the slots.

:class:`MultiSystemRelaxation` relaxes each of a list of structures as ASE would one at a time
(``FIRE(atoms).run(fmax, steps)``, or ``FIRE(FrechetCellFilter(atoms))`` for a variable cell; ASE's default
parameters), for MatRIS and for a model the engine differentiates that declares its systems may be evaluated as one
disjoint union (or pools each system over its own atoms) and takes the engine's per-edge shifts (MACE's
``ScaleShiftMACE``, SevenNet, CHGNet and ORB v3, as :class:`~fastmd.interfaces.multi_system_md.MultiSystemMolecularDynamics`). Up to ``slots``
structures relax at a time in the slots of unions (:mod:`fastmd.execution.system_relax`): one graph build, one model
call and one backward pass per FIRE step for all the structures of a union, ``steps_per_launch`` steps captured as one
CUDA graph, the optimizer's state and the convergence of every system on the device. Each window ends with one read of
every system's phase; a structure that converged, reached the step limit or failed leaves its slot, and the next
structure that fits is written into the slot in place, without a recapture (in-flight batching). The results come back
per input structure.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import gc
import math
import time

import numpy as np
import torch
from ase import Atoms

from ..execution.replica_batch import DEFAULT_UNION_ROWS
from ..execution.system_relax import (
    ACTIVE,
    FINISHED,
    IDLE,
    PENDING,
    PHASES,
    FIREParameters,
    SlotRelaxationRunner,
    SystemFIRE,
    relaxation_builder,
)
from ..adapters.protocol import NegotiationError
from .multi_system_md import UNION_THROUGHPUT_SLACK, MultiSystemMolecularDynamics, _SlotUnion, step_seconds_at
from .replica_md import _COUNTERS

# The ratio of consecutive slot size classes: a structure sits in the smallest class that holds it, so at most a fifth
# of a slot's rows are vacant while its own class has structures left (with fewer slots than classes on this grid, the
# ratio doubles until each class has a slot, and so does the vacancy bound).
SLOT_CLASS_RATIO = 1.25
# The default slots (``slots`` and ``union_rows`` None): the smallest union whose measured rows per second are within
# UNION_THROUGHPUT_SLACK of the best union's (multi_system_md's comment), holding at least this many slots where there
# are as many structures, where a window of that union takes at least IN_FLIGHT_MIN_WINDOW_SECONDS; else every
# structure in a slot (#79's default). A smaller union keeps its rows on structures that still relax (it takes the
# next one as one converges) and leaves fewer to drain at the end, so it wins where the rows per second stay level and
# a step is long against the host's work between windows and the captures of the drains (ORB v3 in IEEE FP32 and
# MatRIS 10M on #79's WBM set: level at 20k and 10k rows/s from a few hundred rows, a window of 64 slots 0.7 s and
# more; 64 slots the best slot count of ORB's, 32 to 256 measured). Where the rows per second keep growing up to all
# the structures (MACE-MPA-0, SevenNet-0) or a step is short, every structure is in a slot.
THROUGHPUT_MIN_SLOTS = 64
# The shortest window of the in-flight union (seconds): the host's work between windows (the finished structures'
# results, the writes into freed slots, one candidate refresh: about 10-20 ms per window for 64 slots on GB200
# hosts) and the captures of the drains cost a fast model's small union more than its finished rows do
# (CHGNet 0.3.0 and ORB v3 with TF32, a window of about 0.1 s: 64 slots 4-7 % slower than the earlier default, every
# structure in a slot).
IN_FLIGHT_MIN_WINDOW_SECONDS = 0.2
# Draining: a union is captured again with only its relaxing systems (finished systems are evaluated with the others
# at every step until then; with no structure left to write each goes into a slot of its own atoms, before that the
# relaxing systems keep their slots and only the slots that no queued structure fits any more are left out) as soon as
# that pays DRAIN_PAYBACK times the estimated time of the capture: the steps the relaxing systems are expected to take,
# each saving the measured step time less that of the smaller union (the measured step scaled by the construction's
# probes, ``step_seconds_at``), or the time the finished systems' rows have cost since the union took its systems
# (right after the queue empties, the relaxing systems are mostly fresh ones, and their median underestimates what
# they will take). Their remaining steps are unknown: as many as the median of them has taken (the slowest structures
# of a set take several times the median's steps: WBM, MACE-MPA-0, a variable cell: median 37, largest 914). The
# capture again of a fast model costs what many of its steps save (SevenNet-0 on #79's WBM set: 0.64-0.72 s against
# 3-15 ms per step), so its late drains can cost more than they save; the payback covers that. A union too small to
# probe is taken to wait on its launches and does not drain. A model whose step time grows with the rows saves much at
# every finished structure and drains early, and again while that pays (ORB v3 in IEEE FP32 on #79's WBM set: four
# drains); a fast model's step saves milliseconds, and it drains late in the tail if at all. The decision reads
# measured times, so whether and when a union drains can differ from run to run (and with it the rounding of the
# systems that move: ``drain=False`` keeps the unions).
DRAIN_PAYBACK = 2.0
# ... and only to at most this fraction of its rows: each drain at least halves the union, so a tail drains a few times
# (down to unions of one structure). A drain that leaves more rows saves little before the next one, which its
# relaxing systems' steps do not foresee (ORB v3 in IEEE FP32, a variable cell, 256 WBM structures: five of six such
# drains cost more time than they saved before the next one).
DRAIN_FRACTION = 0.5
# The relaxing systems' expected steps count only while at least this many relax: the median of fewer says little of
# what remains, and the last structures of a tail run far fewer steps than their count so far (SevenNet-0, a variable
# cell, #79's WBM set: its last structure, at step 464-494 of 542-545, drained a union of 1,409-1,499 rows for 0.65 s
# and saved 0.1-0.25 s). Fewer drain on the time the finished rows have cost alone.
DRAIN_MIN_SYSTEMS = 4
# A variable cell may shrink while it relaxes: the union's sort keys hold the queued cells compressed by this factor.
CELL_FILTER_COMPRESSION = 0.8


@dataclass
class RelaxationResult:
    """One structure's relaxation: ``atoms``, the final structure (positions, and its cell with a cell filter) with
    a ``SinglePointCalculator`` of its energy, forces (and stress); ``fmax`` the optimizer's largest force row (with
    a cell filter the filter's, its cell rows included: ASE's convergence criterion); ``steps`` its FIRE steps
    (ASE's ``nsteps``); ``status`` "converged", "max_steps" or "failed" (a non-finite energy or force: the positions
    of the step that gave it, the non-finite values, ``fmax`` inf); ``trajectory`` (with ``trajectory_interval``)
    the states an ASE observer attached with that interval sees (``SystemRecords.read``: ``steps``, ``energies``,
    ``positions``, ``forces``, ``stresses`` and ``cells``)."""

    index: int
    atoms: Atoms
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    fmax: float
    steps: int
    status: str
    trajectory: dict | None = None

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def plan_slots(sizes: Sequence[int], slots: int) -> tuple[list[int], list[int], list[list[int]]]:
    """``(ceilings, first, queues)`` for ``slots`` concurrent structures of atom counts ``sizes``: every structure
    when there are no more than ``slots`` (each slot its own structure's size); otherwise size classes on a
    geometric grid of ratio ``SLOT_CLASS_RATIO`` (each structure in the smallest class that holds it, each class's
    ceiling its largest structure; the ratio doubled while there are more classes than slots), the slots shared
    among the classes in proportion to their atoms (one each first, then each next slot to the class with the most
    atoms per slot, never more slots than structures). ``first``: the structure first in each slot; ``queues``: the
    rest of each class's structures in input order, one queue per class, smallest class first."""
    sizes = [int(n) for n in sizes]
    if len(sizes) <= slots:
        return list(sizes), list(range(len(sizes))), []
    ratio = SLOT_CLASS_RATIO
    while True:
        grid, value = [], min(sizes)
        while value < max(sizes) and ratio * min(sizes) < max(sizes):
            grid.append(value)
            value = max(value + 1, math.ceil(value * ratio))
        grid.append(max(value, max(sizes)))  # one class holds them all once the ratio spans the sizes
        members: dict[int, list[int]] = {}
        for index, n in enumerate(sizes):
            members.setdefault(next(c for c in grid if c >= n), []).append(index)
        if len(members) <= slots:
            break
        ratio *= 2.0
    classes = sorted(members)
    work = [sum(sizes[i] for i in members[c]) for c in classes]
    share = [1] * len(classes)
    for _ in range(slots - len(classes)):  # each next slot to the class with the most atoms per slot
        open_classes = [k for k, c in enumerate(classes) if share[k] < len(members[c])]
        if not open_classes:
            break
        share[max(open_classes, key=lambda k: (work[k] / share[k], -k))] += 1
    ceilings, first, queues = [], [], []
    for k, c in enumerate(classes):
        queue = list(members[c])
        count = share[k]
        ceiling = max(sizes[i] for i in queue)
        ceilings += [ceiling] * count
        first += queue[:count]
        queues.append(queue[count:])
    return ceilings, first, queues


class _SlotRelaxationEngine(MultiSystemMolecularDynamics):
    """The slot unions of :class:`~fastmd.interfaces.multi_system_md.MultiSystemMolecularDynamics` (negotiation,
    self-check, measured memory, unions sharing a graph pool, captures and capacity growth) with
    :class:`~fastmd.execution.system_relax.SystemFIRE` as each union's integrator and
    :class:`~fastmd.execution.system_relax.SlotRelaxationRunner` as its runner. Private: the relaxation's driver."""

    _entry = "MultiSystemRelaxation"

    def __init__(self, model, systems, *, relaxation: dict, admit_cells, structures=None, planner=None, first=None,
                 drain: bool = True, in_flight: bool = False, **options) -> None:
        self._relaxation = relaxation
        self._drain = drain
        self._admit_cells = admit_cells
        # With the slots left to the runtime: every structure, the planner of the slots of a union of given rows
        # (``(ceilings, first, queues)`` of ``plan_slots``) and the structures ``systems`` are (``first``).
        self._structures, self._planner, self._first = structures, planner, first
        self.in_flight = in_flight  # structures queue for the slots (written into them as others finish)
        self.slot_plan = None  # the planner's slots, where it ran
        # Called with (old union, [(new union, the old union's slots it holds), ...]) when unions are captured anew.
        self.moved = lambda old, parts: None
        # The MD constructor's thermostat arguments are unused: the integrator is FIRE (its float64 state, as the
        # Nose-Hoover chain's, takes float64 models too).
        super().__init__(model, systems, ensemble="nvt_nhc", **options)

    def _union_rows(self, union_rows):
        """The rows per union (``MultiSystemMolecularDynamics._union_rows``) of a relaxation: with the slots left
        to the runtime (a planner), the smallest measured union within ``UNION_THROUGHPUT_SLACK`` of the best rows
        per second, at least ``THROUGHPUT_MIN_SLOTS`` slots, where a window of it takes at least
        ``IN_FLIGHT_MIN_WINDOW_SECONDS`` (its slots planned and their first structures in them), else every
        structure in a slot (as planned); with the slots or rows given, as given. The probes also price the drains
        (``MultiSystemRelaxation._drain``): they run with the slots or rows given too, up to the union's rows, unless
        nothing drains."""
        rows = DEFAULT_UNION_ROWS if union_rows is None else union_rows
        if self._planner is None and not self._drain:  # nothing to choose or price
            self._step_seconds = []
            return rows, None
        # (each step's stress: a cell filter's, or the trajectory's records' with a fixed cell)
        stress = bool(self._relaxation["cell_filter"] or self._relaxation.get("record_stress"))
        points = self._step_seconds = self._probe_step_seconds(min(sum(self.ceilings), rows), stress=stress,
                                                               stop_when_level=True)
        if self._planner is None or len(points) < 2:
            return rows, None
        best = max(size / seconds for size, seconds, *_ in points)
        chosen = min(size for size, seconds, *_ in points if size / seconds >= (1.0 - UNION_THROUGHPUT_SLACK) * best)
        window = self.steps_per_launch * step_seconds_at(points, sum(self._planner(chosen, True)[0]))
        if window < IN_FLIGHT_MIN_WINDOW_SECONDS:  # every structure in a slot (as planned: all that fit)
            return rows, {**self._throughput_record(points, sum(self.ceilings)), "in_flight_window_s": round(window, 4)}
        return self._plan_slots(chosen), {**self._throughput_record(points, chosen),
                                          "in_flight_window_s": round(window, 4)}

    def _plan_slots(self, rows: int, *, at_least: bool = True) -> int:
        """The planner's slots for a union of ``rows`` rows (``at_least``: no fewer than ``THROUGHPUT_MIN_SLOTS``)
        with their first structures in them (the others queue); the rows per union that hold them in one."""
        ceilings, first, queues = self.slot_plan = self._planner(rows, at_least)
        self.in_flight = bool(queues)
        if (list(first), list(ceilings)) != (self._first, self.ceilings):
            self.ceilings, self.systems, self._first = list(ceilings), len(first), list(first)
            self._system_tables([self._structures[index] for index in first])
        return max(sum(ceilings), 1) if queues else DEFAULT_UNION_ROWS

    def _size_unions(self, plan, union_rows: int):
        """``MultiSystemMolecularDynamics._size_unions``; where the slots are the planner's and the measured memory
        admits fewer rows than they hold, fewer slots in one union of the admitted rows (rather than several unions
        of the slots: a smaller union leaves fewer structures to drain, and each union's growth is a capture)."""
        chosen, plan = super()._size_unions(plan, union_rows)
        if self._planner is None or self.slot_plan is None or chosen >= sum(self.ceilings):
            return chosen, plan
        rows = self._plan_slots(chosen, at_least=False)
        self._capacity_tiers()  # MatRIS's headroom follows the systems in the slots
        plan.decisions[-1].evidence["slots"] = len(self.ceilings)
        return rows, plan

    def _start_union(self, union, positions: list, momenta: list) -> None:
        """The FIRE integrator of ``union`` with its systems pending at ``positions``, and its first runner."""
        union.atoms = [len(self._numbers[index]) for index in range(union.start, union.stop)]
        union.carried = False
        union.cells = torch.as_tensor(np.stack(self._cells[union.start:union.stop]), dtype=torch.float64,
                                      device=self.device)
        union.numbers, coordinates = self._union_tables(union, positions)
        integrator = SystemFIRE(union.layout, device=self.device, **self._relaxation)
        for slot, index in enumerate(range(union.start, union.stop)):
            integrator.write_system(slot, positions=positions[slot], cell=self._cells[index])
        integrator.sync()
        union.integrator = integrator
        # In flight, with the pairs' growth headroom from the start: the structures written into freed slots change the
        # union's counts, and a relaxation's first steps move the atoms (and cells) the most (ORB v3 in IEEE FP32, 64
        # slots: its first window grew the union, 1.7 s). Its triplets keep the model's tiers (every padded triplet runs
        # a line graph's angle networks), and with every structure in a slot the pairs too (SevenNet-0, a variable cell:
        # its union did not grow without them).
        self._set_demand_capacities(union, coordinates, first=True)
        self._capture(union, coordinates, integrator.momenta.to(torch.float32), forces=None)

    def _capture(self, union, positions, momenta, *, forces) -> None:
        """A runner of ``union`` at its capacity (a recapture when ``forces``, the cached forces of the state, are
        given), its builder in the integrator's current cells."""
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self._keep_graph_pool_alive()
        integrator = union.integrator
        builder = relaxation_builder(
            self._adapter, cell_filter=integrator.cell_filter, layout=union.layout,
            cells=integrator.cells if integrator.cell_filter else union.cells, atomic_numbers=union.numbers,
            atoms=union.atoms, u_capacity=union.u_capacity, t_capacity=union.t_capacity,
            candidate_skin=self.candidate_skin, device=self.device, admit_cells=self._admit_cells)
        union.runner = SlotRelaxationRunner(
            model=self._adapter, builder=builder, integrator=integrator, positions=positions, momenta=momenta,
            forces=forces, steps_per_launch=self.steps_per_launch, warmup=self._config.capacity.warmup,
            graph_pool=self._graph_pool, config=self._config, recapture=forces is not None,
            validate_isolated_atoms=not self._config.handle_isolated_atoms)
        torch.cuda.synchronize(self.device)
        union.capture_seconds.append(time.perf_counter() - started)

    @torch.no_grad()
    def write_structure(self, union, slot: int, atoms: Atoms) -> None:
        """``atoms`` (checked at construction) into ``slot`` of ``union`` in place: its species, cell and atom count
        in the builder, a pending system with a fresh optimizer in the integrator. The caller then calls
        :meth:`commit_writes`."""
        builder = union.runner.builder
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        cell64 = builder.check_cell(cell)
        builder.write_system(slot, atomic_numbers=torch.as_tensor(numbers), cell=cell64, fill_number=int(numbers[0]))
        union.integrator.write_system(slot, positions=atoms.get_positions(), cell=cell)
        index = union.start + slot
        self._numbers[index], self._cells[index] = numbers, cell
        self.masses[index] = np.asarray(atoms.get_masses(), dtype=np.float64)
        self.templates[index] = atoms.copy()
        self._writes[index] += 1
        union.atoms[slot] = len(numbers)
        union.cells[slot].copy_(cell64)
        union.numbers.copy_(builder.atomic_numbers_real)

    @torch.no_grad()
    def compact(self, position: int, keep: list[int], *, own_size: bool = True) -> None:
        """The union at ``position`` in ``_unions`` replaced by one that holds only the systems of its slots
        ``keep`` (still relaxing), each in a slot of its own atom count (``own_size``) or of its slot's ceiling,
        their optimizer state carried over (:meth:`SystemFIRE.take_systems`), captured again; its other slots'
        systems are finished and collected. A failed capture raises and changes nothing: the old union, its runner
        and the per-system tables stay."""
        old = self._unions[position]
        union = self._carry(old, keep, own_size=own_size)
        for name in _COUNTERS:
            union.accumulated[name] = old.accumulated[name] + old.runner.stats()[name]
        union.recaptures = old.recaptures + 1
        union.capture_seconds = old.capture_seconds + union.capture_seconds
        self._unions[position] = union
        self.moved(old, [(union, keep)])
        old.runner = old.integrator = None

    def _split(self, union, positions, momenta, forces, why: str):
        """``union``'s growth ran out of CUDA memory (``why``): its systems still relaxing (its slots that are not
        idle) run on in two unions with about half its rows each (four, eight... where a part runs out of memory
        again), each slot its ceiling and its system's optimizer state (``_carry``), captured at the parts' demand
        with the growth headroom, in ``union``'s place; a system that does not fit alone raises
        :class:`~fastmd.adapters.protocol.NegotiationError`. Returns the parts."""
        slots = [slot for slot, phase in enumerate(union.integrator.phase.tolist()) if int(phase) != IDLE]
        position, count, length = self._unions.index(union), 2, len(self._numbers)

        def release(parts) -> None:  # the parts captured so far and their per-system tables
            for part in parts:
                part.runner = part.integrator = None
            for table in (self._numbers, self._cells, self.templates, self.masses, self._writes, self.ceilings):
                del table[length:]
            self.systems = length

        while True:
            groups = [[slots[k] for k in group] for group in
                      self._split_groups([union.layout.ceilings[slot] for slot in slots], count)]
            if max(sum(union.layout.ceilings[slot] for slot in group) for group in groups) >= union.layout.rows:
                raise NegotiationError(f"MultiSystemRelaxation: {why}, and it holds one structure: relax it with "
                                       "less memory in use")
            parts, failure = [], None
            try:
                for group in groups:
                    parts.append(self._carry(union, group, own_size=False))
            except torch.OutOfMemoryError as error:
                failure = str(error).splitlines()[0][:160]
            except BaseException:
                release(parts)
                raise
            if failure is None:
                break
            release(parts)  # outside the except block: its traceback would pin the failed part's buffers
            parts = []
            gc.collect()
            torch.cuda.empty_cache()
            if len(groups) == len(slots):
                raise NegotiationError(f"MultiSystemRelaxation: {why}, and a union of one structure ran out of CUDA "
                                       f"memory too ({failure}): relax them with less memory in use")
            count *= 2
        parts[0].accumulated = dict(union.accumulated)
        parts[0].recaptures = union.recaptures + 1
        parts[0].capture_seconds = union.capture_seconds + parts[0].capture_seconds
        parts[0].max_edge_count, parts[0].max_triplet_count = union.max_edge_count, union.max_triplet_count
        self._unions[position:position + 1] = parts
        self.splits.append((union.layout.rows, [part.layout.rows for part in parts], why))
        self.moved(union, list(zip(parts, groups)))
        union.runner = union.integrator = None
        return parts

    def _carry(self, old, slots: list[int], *, own_size: bool) -> _SlotUnion:
        """A union of ``old``'s systems in its ``slots``, each in a slot of its own atom count (``own_size``) or of
        its old slot's ceiling, their optimizer state carried over (:meth:`SystemFIRE.take_systems`), captured at
        their demand with the growth headroom, and its per-system tables appended. A failed capture raises and
        changes nothing."""
        integrator = old.integrator
        atoms = [integrator.host_atoms[slot] for slot in slots]
        ceilings = atoms if own_size else [old.layout.ceilings[slot] for slot in slots]
        start = len(self._numbers)
        union = _SlotUnion(start, start + len(slots), ceilings)
        union.carried = True
        cells = integrator.cells if integrator.cell_filter else old.cells
        union.atoms = list(atoms)
        union.cells = cells.index_select(0, torch.as_tensor(slots, device=self.device)).clone()
        union.numbers = torch.cat([old.numbers[old.layout.starts[slot]:old.layout.starts[slot] + n]
                                   for slot, n in zip(slots, ceilings)])
        union.integrator = SystemFIRE(union.layout, device=self.device, **self._relaxation)
        union.integrator.take_systems(integrator, slots)
        union.integrator.sync()
        positions = union.integrator.positions.to(torch.float32)
        self._set_demand_capacities(union, positions)
        self._capture(union, positions, union.integrator.momenta.to(torch.float32), forces=union.integrator.forces)
        for slot in slots:  # the per-system tables of the new union's systems (rows start..stop)
            index = old.start + slot
            for table in (self._numbers, self._cells, self.templates, self.masses, self._writes):
                table.append(table[index])
        self.ceilings += ceilings
        self.systems += len(slots)
        return union

    def _set_demand_capacities(self, union, positions, *, first: bool = False) -> None:
        """``union``'s capacities with the growth headroom (``_set_capacities``; its ``first`` capture: the pairs'
        only) at its demand at ``positions``, and the time the demand took (``demand_seconds``: one eager build of
        the union, part of a capture again's cost)."""
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self._set_capacities(union, positions, **({"pair_headroom": self.in_flight} if first else {"grown": True}))
        torch.cuda.synchronize(self.device)
        union.demand_seconds = time.perf_counter() - started

    @torch.no_grad()
    def commit_writes(self, union) -> None:
        """After writes into ``union``: the integrator's derived state, the runner's mirrors and the union's
        candidates (one refresh for all the writes)."""
        integrator, runner = union.integrator, union.runner
        integrator.sync()
        runner.reset(positions=integrator.positions.to(torch.float32), momenta=integrator.momenta.to(torch.float32),
                     forces=integrator.forces.to(runner.forces.dtype), potential_energy=integrator.potential_energy,
                     reset_counters=False)


class MultiSystemRelaxation:
    """Relax ``structures`` (see the module docstring).

    Parameters
    ----------
    model:
        An adapter, or a name or checkpoint path that :func:`~fastmd.adapters.load_adapter` loads (``model_path``,
        ``model_dtype`` as there); the models of ``MultiSystemMolecularDynamics`` (a model that does not declare
        ``"union"`` or ``"segments"`` and the engine's per-edge shifts raises
        :class:`~fastmd.adapters.protocol.NegotiationError`).
    structures:
        ASE atoms, periodic in all three directions, each with its own cell, without constraints; checked at
        construction (with the model's elements, on one structure holding the elements of those that do not start in
        a slot).
    fmax, max_steps:
        ASE's ``run(fmax, steps)``: a structure converges when its largest force row is below ``fmax`` (eV/Å; with
        a cell filter the filter's rows, its cell's included), and stops unconverged after ``max_steps`` steps.
    cell_filter:
        None (fixed cell) or ``"frechet"``: ASE's ``FrechetCellFilter`` with its default arguments (the atom count as
        the cell factor, no pressure, every strain component).
    fire:
        :class:`~fastmd.execution.system_relax.FIREParameters`; default ASE's ``FIRE`` defaults.
    slots:
        The structures relaxed at a time; default (``union_rows`` None too) by the model's measured throughput
        (``THROUGHPUT_MIN_SLOTS``' comment): at least 64 slots in one union where the rows per second stay level and
        a window of them is long (fewer where the measured memory admits fewer rows), else every structure; with
        ``union_rows`` as many as the planned slots' ceilings fit in it. With more structures than slots, the slots
        come in size classes (``plan_slots``) and a freed slot takes the next structure of its class, else of the
        largest smaller class.
    steps_per_launch:
        FIRE steps per captured window: a finished structure leaves its slot at the end of its window. Default 5:
        early in a relaxation the atoms (and cells) move fast, and a window that outruns the candidates' skin is
        replayed (10-step windows did so in the first windows of variable-cell relaxations), and a capture again
        (draining) runs the window in its warmup; a window costs the host one read.
    drain:
        Capture a union again with only its relaxing structures where that pays back (``DRAIN_PAYBACK``'s comment;
        before the queue empties only leaving out the slots no queued structure fits); it reads measured times, so
        whether and when it happens can differ between runs.
    union_rows, capacity_headroom, config, device, self_check:
        As in :class:`~fastmd.interfaces.multi_system_md.MultiSystemMolecularDynamics` (``union_rows`` caps the
        slots' rows per union; the self-check runs on the structures that start in a slot, with the default slots
        on all of them, before the slots are planned; a union's first capture holds the
        growth headroom, and one whose growth runs out of CUDA memory splits, its relaxing structures' optimizer
        state carried over).
    trajectory_interval, trajectory_stress:
        Record each structure's states as an ASE observer attached to its optimizer with ``interval =
        trajectory_interval`` sees them (every step that is a multiple of it, the start included; ``<= 0``: only step
        ``-trajectory_interval``): positions, the model's forces, energy, the cell (a cell filter) and, with a cell
        filter or ``trajectory_stress``, the stress, on the device in every step that evaluates them, read once as the
        structure leaves its slot (``RelaxationResult.trajectory``). A fixed cell with ``trajectory_stress``
        evaluates the stress every step. Memory, preallocated: 48 bytes per slot row and 160 per slot for each of
        ``max_steps // trajectory_interval + 1`` records and one scratch record (and 72 per slot for the last
        stress).

    The structures that do not start in a slot get the slots' construction checks on the host (the model's elements,
    the candidate buffer, isolated atoms where the configuration rejects them). What a relaxation brings about later
    (a variable cell that compresses beyond them, or thinner than the union's sort keys hold) raises from :meth:`run`;
    the results so far stay in ``results``.
    """

    def __init__(self, model, structures: Sequence[Atoms], *, fmax: float = 0.05, max_steps: int = 1000,
                 cell_filter: str | None = None, fire: FIREParameters | None = None, slots: int | None = None,
                 steps_per_launch: int = 5, union_rows: int | None = None, capacity_headroom: float | None = None,
                 drain: bool = True, config=None, model_path: str | None = None, model_dtype: str | None = None,
                 device=None, self_check: bool = True, trajectory_interval: int | None = None,
                 trajectory_stress: bool = False) -> None:
        where = "MultiSystemRelaxation"
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)) or not math.isfinite(fmax) or fmax <= 0:
            raise ValueError(f"{where}: fmax must be finite and positive")
        for name, value, low in (("max_steps", max_steps, 0), ("steps_per_launch", steps_per_launch, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{where}: {name} must be an integer >= {low}")
        if cell_filter not in (None, "frechet"):
            raise ValueError(f"{where}: cell_filter must be None or 'frechet', got {cell_filter!r}")
        if fire is not None and not isinstance(fire, FIREParameters):
            raise ValueError(f"{where}: fire must be FIREParameters")
        if slots is not None and (isinstance(slots, bool) or not isinstance(slots, int) or slots < 1):
            raise ValueError(f"{where}: slots must be a positive integer")
        if trajectory_interval is not None and (isinstance(trajectory_interval, bool)
                                                or not isinstance(trajectory_interval, int)):
            raise ValueError(f"{where}: trajectory_interval must be None or an integer")
        structures = list(structures)
        if not structures or not all(isinstance(atoms, Atoms) and len(atoms) > 0 for atoms in structures):
            raise ValueError(f"{where}: structures must be one or more ASE Atoms with atoms")
        for index, atoms in enumerate(structures):
            MultiSystemMolecularDynamics._check_system(atoms, f"{where}: structure {index}")
            if atoms.constraints:  # ASE's optimizers apply them to every step; the batch does not
                raise ValueError(f"{where}: structure {index} has constraints "
                                 f"({', '.join(type(c).__name__ for c in atoms.constraints)}), which it does not apply")
        self._structures = [atoms.copy() for atoms in structures]
        for atoms in self._structures:
            atoms.calc = None
        sizes = [len(atoms) for atoms in structures]
        planner = None
        if slots is None and union_rows is None:  # by throughput: the engine measures the model and plans the slots
            planner = lambda rows, at_least: plan_slots(sizes, max(self._fitting_slots(sizes, rows), min(
                THROUGHPUT_MIN_SLOTS if at_least else 1, len(sizes))))
        if slots is None:  # every structure, else as many slots as the planned ceilings fit in one union's rows
            slots = self._fitting_slots(sizes, DEFAULT_UNION_ROWS if union_rows is None else int(union_rows))
        ceilings, first, queues = plan_slots(sizes, slots)
        self.cell_filter = cell_filter
        self.fmax, self.max_steps = float(fmax), int(max_steps)
        cells = np.stack([np.asarray(atoms.cell.array, dtype=np.float64) for atoms in structures])
        admit = cells * CELL_FILTER_COMPRESSION if cell_filter else cells
        self._engine = _SlotRelaxationEngine(
            model, [self._structures[index] for index in first], slot_atoms=ceilings, union_rows=union_rows,
            steps_per_launch=steps_per_launch, capacity_headroom=capacity_headroom, config=config,
            model_path=model_path, model_dtype=model_dtype, device=device, self_check=self_check,
            relaxation=dict(fmax=float(fmax), max_steps=int(max_steps), parameters=fire,
                            cell_filter=cell_filter is not None, record_interval=trajectory_interval,
                            record_stress=bool(trajectory_stress)), admit_cells=torch.as_tensor(admit),
            structures=self._structures if planner is not None else None, planner=planner, first=list(first),
            drain=bool(drain), in_flight=bool(queues))
        if self._engine.slot_plan is not None:
            ceilings, first, queues = self._engine.slot_plan
        try:
            self._occupant = {}  # (union, slot) -> structure index
            for system, index in enumerate(first):
                union = next(union for union in self._engine._unions if union.start <= system < union.stop)
                self._occupant[(union, system - union.start)] = index
            self._engine.moved = self._moved
            # The queues follow the classes, smallest first; a class's ceiling (its largest structure) is its slots'.
            self._queues = [deque(queue) for queue in queues]
            self._class_ceilings = sorted(set(ceilings)) if queues else []
            self._check_elements(set(range(len(structures))) - set(first))
            self._check_neighbours(set(range(len(structures))) - set(first))
        except BaseException:
            self.close()
            raise
        self.results: list[RelaxationResult | None] = [None] * len(structures)
        self.windows = 0
        self.writes = 0
        self.compactions = 0
        self.row_steps = 0  # the rows the windows evaluated, times their steps
        self.drain = bool(drain)
        self.drain_failures: list[str] = []  # captures again that failed (the union kept running as it was)
        self.drains: list[dict] = []  # each capture again with fewer rows: its numbers
        self._kept: set = set()  # unions whose capture again failed: not tried again
        self._window_seconds: dict = {}  # union -> the wall seconds of its last windows
        self._waste: dict = {}  # union -> the seconds its finished systems' rows cost since it took its systems
        self.turnover_seconds = 0.0

    @staticmethod
    def _fitting_slots(sizes: list[int], rows: int) -> int:
        """Every structure, else as many slots as the planned ceilings (``plan_slots``) fit in ``rows``."""
        slots = len(sizes)
        while slots > 1 and sum(plan_slots(sizes, slots)[0]) > rows:
            slots = max(1, min(slots - 1, int(slots * rows / sum(plan_slots(sizes, slots)[0]))))
        return slots

    def _check_elements(self, queued: set[int]) -> None:
        """The model's own input check (its ``prepare``) on one structure holding every element of the queued
        structures that no structure in a slot has: ValueError naming the structures with an element it rejects."""
        engine = self._engine
        known = set(int(z) for numbers in engine._numbers for z in numbers)
        missing = sorted({int(z) for index in queued for z in self._structures[index].get_atomic_numbers()} - known)
        if not missing:
            return
        side = math.ceil(len(missing) ** (1.0 / 3.0))
        spacing = 2.5
        grid = np.array([(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:len(missing)])
        cell = np.eye(3) * spacing * side
        try:
            with engine._config.scope():
                engine._evaluate_alone(np.asarray(missing, dtype=np.int32), grid * spacing, cell, False,
                                       "MultiSystemRelaxation")
        except ValueError as error:
            rejected = set()
            for z in missing:  # which elements: each alone, two atoms 2.5 A apart
                try:
                    with engine._config.scope():
                        engine._evaluate_alone(np.asarray([z, z], dtype=np.int32), np.array([[0.0] * 3, [2.5, 0, 0]]),
                                               np.eye(3) * 5.0, False, "MultiSystemRelaxation")
                except ValueError:
                    rejected.add(z)
            bad = sorted(index for index in queued
                         if set(map(int, self._structures[index].get_atomic_numbers())) & (rejected or set(missing)))
            raise ValueError(f"MultiSystemRelaxation: the model rejects elements {sorted(rejected) or missing} of "
                             f"structures {bad[:16]}{' ...' if len(bad) > 16 else ''}: {error}") from error

    def _check_neighbours(self, queued: set[int]) -> None:
        """The checks a slot's construction runs, on the host, for the structures that do not start in a slot (written
        into a captured union later): every atom's neighbours within the candidate cutoff fit the union's candidate
        buffer and, where the union rejects isolated atoms, every atom has a neighbour within the cutoff (ASE's
        neighbour list). ValueError naming the structures that fail."""
        if not queued:
            return
        from ase.neighborlist import neighbor_list

        runner = self._engine._unions[0].runner
        builder = runner.builder
        dense, lone = [], []
        for index in sorted(queued):
            atoms = self._structures[index]
            if np.bincount(neighbor_list("i", atoms, builder.candidate_cutoff), minlength=len(atoms)).max() > \
                    builder.buffer_max:
                dense.append(index)
            if runner.validate_isolated_atoms and not np.bincount(
                    neighbor_list("i", atoms, builder.atom_graph_cutoff), minlength=len(atoms)).all():
                lone.append(index)
        for bad, why in ((dense, f"more than {builder.buffer_max} neighbours within {builder.candidate_cutoff} A "
                                 "(the candidate buffer)"),
                         (lone, f"an atom without a neighbour within {builder.atom_graph_cutoff} A")):
            if bad:
                raise ValueError(f"MultiSystemRelaxation: structures {bad[:16]}{' ...' if len(bad) > 16 else ''} have "
                                 f"{why}")

    # ------------------------------------------------------------------ the run
    def run(self) -> list[RelaxationResult]:
        """Relax every structure; returns their results in input order (also kept as ``results``)."""
        engine = self._engine
        with engine._config.scope():
            while True:
                live = [union for union in engine._unions if self._live(union)]
                if not live:
                    break
                for union in live:  # each live union runs one window; an overflow grows and captures it again
                    started = time.perf_counter()
                    self.row_steps += union.layout.rows * engine.steps_per_launch
                    engine._window(union)
                    # the last few windows' wall times (their minimum: a window that grew or refreshed costs more)
                    self._window_seconds.setdefault(union, []).append(time.perf_counter() - started)
                    del self._window_seconds[union][:-4]
                self.windows += 1
                started = time.perf_counter()
                for union in list(engine._unions):  # those that ran (a union split by a growth: its parts)
                    self._turnover(union)
                if self.drain:
                    for u in range(len(engine._unions)):
                        self._drain(u)
                self.turnover_seconds += time.perf_counter() - started
        missing = [index for index, result in enumerate(self.results) if result is None]
        if missing:
            raise RuntimeError(f"MultiSystemRelaxation: structures {missing[:16]} were never relaxed")
        return list(self.results)

    def _live(self, union) -> bool:
        phases = union.runner.phases or tuple(union.integrator.phase.tolist())
        return any(int(phase) not in FINISHED + (IDLE,) for phase in phases)

    def _next(self, ceiling: int) -> int | None:
        """The next queued structure for a freed slot of ``ceiling`` rows: of the largest class whose structures it
        holds and whose queue is not empty (its own class first)."""
        for size, queue in zip(reversed(self._class_ceilings), reversed(self._queues)):
            if size <= ceiling and queue:
                return queue.popleft()
        return None

    def _turnover(self, union) -> None:
        """The finished structures of ``union`` leave their slots (their results), the next ones come in."""
        finished = [slot for slot, phase in enumerate(union.runner.phases) if int(phase) in FINISHED]
        if not finished:
            return
        self._collect(union, finished)
        engine, wrote = self._engine, False
        for slot in finished:
            index = self._next(union.layout.ceilings[slot])
            if index is None:
                union.integrator.release(slot)
                continue
            engine.write_structure(union, slot, self._structures[index])
            self._occupant[(union, slot)] = index
            self.writes += 1
            wrote = True
        if wrote:
            engine.commit_writes(union)
        union.runner.phases = tuple(union.integrator.phase.tolist())  # the written and released slots' phases

    def _drain(self, u: int) -> None:
        """The union ``u`` captured again with only its relaxing systems where that pays back (``DRAIN_PAYBACK``'s
        comment; ``_SlotRelaxationEngine.compact``): with no structure left to write, each in a slot of its own atom
        count; else in their slots (which take the next structures of their classes), leaving out the slots no queued
        structure fits any more."""
        union = self._engine._unions[u]
        live = [slot for slot, phase in enumerate(union.runner.phases) if int(phase) in (PENDING, ACTIVE)]
        queued = any(self._queues)
        rows = sum((union.layout.ceilings if queued else union.integrator.host_atoms)[slot] for slot in live)
        seconds = self._window_seconds.get(union)
        if not live or rows >= union.layout.rows or not seconds or not union.capture_seconds:
            return
        step = min(seconds) / self._engine.steps_per_launch
        saving = step - self._step_seconds(union, rows, step)
        # the time the finished systems' rows cost since the union took its systems
        self._waste[union] = waste = self._waste.get(union, 0.0) + saving * self._engine.steps_per_launch
        if rows > DRAIN_FRACTION * union.layout.rows or union in self._kept:
            return
        steps = float(np.median(union.integrator.state.steps[live].cpu().numpy()))
        cost = self._recapture_seconds(union, rows)
        expected = steps if len(live) >= DRAIN_MIN_SYSTEMS else 0.0
        if max(expected * saving, waste) < DRAIN_PAYBACK * cost:
            return
        failure = None
        try:
            self._engine.compact(u, live, own_size=not queued)
            self.drains.append({"rows": union.layout.rows, "kept": rows, "queued": queued, "steps": steps,
                                "saving_s": saving, "waste_s": waste, "estimated_s": cost,
                                "capture_s": self._engine._unions[u].capture_seconds[-1],
                                "demand_s": self._engine._unions[u].demand_seconds})
        except Exception as error:  # noqa: BLE001 - an optimisation: the union runs on as it was
            failure = f"{type(error).__name__}: {str(error).splitlines()[0][:200] if str(error) else ''}"
        if failure is not None:  # outside the except block: its traceback would pin the failed capture's memory
            gc.collect()
            torch.cuda.empty_cache()
            self.drain_failures.append(failure)
            self._kept.add(union)
            return
        self._window_seconds.pop(union, None)
        self._waste.pop(union, None)
        union = self._engine._unions[u]
        union.runner.phases = tuple(union.integrator.phase.tolist())
        self.compactions += 1

    def _moved(self, old, parts) -> None:
        """The structures of ``old``'s slots into the new unions that hold them (``parts``: ``(union, the old
        slots it holds in its slot order)``)."""
        for union, slots in parts:
            for slot, old_slot in enumerate(slots):
                self._occupant[(union, slot)] = self._occupant.pop((old, old_slot))

    def _step_seconds(self, union, rows: int, step: float) -> float:
        """The step time of a union of ``rows`` rows, estimated from ``union``'s measured ``step``: scaled by the
        construction's probes (``step_seconds_at``); without them (a union too small to probe, or none measured)
        ``step``: a union that small is taken to wait on its launches, and no drain pays."""
        points = self._engine._step_seconds
        if not points:
            return step
        return step * step_seconds_at(points, rows) / step_seconds_at(points, union.layout.rows)

    def _recapture_seconds(self, union, rows: int | None = None) -> float:
        """The time a capture again of ``union`` with only ``rows`` rows takes, estimated: what it runs (its first
        evaluation and warmup windows eagerly, and the capture of a window: the probes' eager step and capture
        times at ``rows``), the rest of ``union``'s last capture (builders, buffers, the graph's instantiation:
        its measured time less what it ran by the probes) and the demand count before it (its measured time).
        Without probes: its last capture's time, its first capture's scaled from that capture's warmup windows to
        a capture again's."""
        from ..engine.capacity_policy import CapturePolicy

        engine, warmups = self._engine, CapturePolicy.from_config().recapture_warmup
        points, steps = engine._step_seconds, engine.steps_per_launch
        # the last capture's warmup windows (a carried union's first capture is a capture again)
        last = warmups if union.recaptures or union.carried else engine._config.capacity.warmup
        if not points:
            return union.capture_seconds[-1] * (1.0 if union.recaptures else (warmups + 1) / (last + 1))
        ran = lambda size, windows: ((1 + windows * steps) * step_seconds_at(points, size, 2)
                                     + steps * step_seconds_at(points, size, 3))
        rest = max(union.capture_seconds[-1] - ran(union.layout.rows, last), 0.0)
        return union.demand_seconds + rest + ran(union.layout.rows if rows is None else rows, warmups)

    def _collect(self, union, slots: list[int]) -> None:
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.stress import full_3x3_to_voigt_6_stress

        integrator = union.integrator
        state = integrator.state
        positions = integrator.positions.cpu().numpy()
        forces = integrator.forces.cpu().numpy()
        energy, fmax, steps, phase = torch.stack((state.energy, state.fmax, state.steps, state.phase)).cpu().numpy()
        cells = integrator.cells.cpu().numpy() if integrator.cell_filter else None
        stress = state.stress.cpu().numpy() if integrator.cell_filter else None
        for slot in slots:
            index = self._occupant.pop((union, slot))
            start, n = union.layout.starts[slot], integrator.host_atoms[slot]
            atoms = self._structures[index].copy()
            if cells is not None:
                atoms.set_cell(cells[slot], scale_atoms=False)
            atoms.set_positions(positions[start:start + n])
            properties = dict(energy=float(energy[slot]), forces=forces[start:start + n].copy())
            if stress is not None:
                properties["stress"] = full_3x3_to_voigt_6_stress(stress[slot])
            atoms.calc = SinglePointCalculator(atoms, **properties)
            records = (None if integrator.records is None else
                       integrator.records.read(slot, start, n, int(steps[slot])))
            self.results[index] = RelaxationResult(
                index=index, atoms=atoms, energy=float(energy[slot]), forces=properties["forces"],
                stress=None if stress is None else stress[slot].copy(), fmax=float(fmax[slot]), steps=int(steps[slot]),
                status=PHASES[int(phase[slot])], trajectory=records)

    # ------------------------------------------------------------------ records
    def stats(self) -> dict:
        """The unions' counters (``MultiSystemMolecularDynamics.stats``: windows launched, candidate refreshes,
        fallback windows, recaptures, capacities and their fill, capture times), the relaxation's windows, writes
        (structures written into freed slots) and host time between windows (s), and the results so far by
        status."""
        done = [result for result in self.results if result is not None]
        statuses = {name: sum(result.status == name for result in done) for name in ("converged", "max_steps",
                                                                                     "failed")}
        unions = {key: value for key, value in self._engine.stats().items() if key not in ("systems", "writes")}
        return {**unions, "relaxation_windows": self.windows, "structure_writes": self.writes,
                "compactions": self.compactions, "drain_failures": list(self.drain_failures),
                "evaluated_row_steps": self.row_steps, "drains": [dict(drain) for drain in self.drains],
                "turnover_seconds": self.turnover_seconds, "finished": statuses, "structures": len(self.results)}

    def plan_record(self):
        """The negotiated plan (:class:`~fastmd.adapters.negotiate.EnginePlanRecord`)."""
        return self._engine.plan_record()

    @property
    def adapter(self):
        return self._engine.adapter

    def close(self) -> None:
        """Release the model adapter and owner callbacks; context exit closes too.

        Keep results and statistics readable, but break the engine's references back to this driver so dropping
        it immediately releases its runners, CUDA graphs and trajectory buffers without waiting for cyclic GC.
        """
        try:
            self._engine.close()
        finally:
            self._engine.moved = None
            self._engine._planner = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
