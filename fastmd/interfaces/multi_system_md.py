"""Molecular dynamics of a set of different systems, advanced together in whole-step CUDA graphs.

:class:`MultiSystemMolecularDynamics` runs one trajectory for each of S
systems (each with its own atoms, atom count, cell, positions, momenta and
thermostat) for MatRIS (its own derivative, by its own rules) and for a
model the engine differentiates that declares its systems
may be evaluated as one disjoint union (``SystemBatching`` ``"union"``, or
``"segments"`` where the model pools each system over its own atoms) and takes
the engine's per-edge shifts (``EdgeShiftInput``: MACE's ``ScaleShiftMACE``,
SevenNet, CHGNet with its line graph, ORB v3). The systems sit in the slots of
unions (:mod:`fastmd.execution.system_batch`): one graph build (with each
system's line graph for a model with triplets), one model call and one
backward pass per step for all systems of a union, ``steps_per_launch`` steps
of integrator, neighbour list, model and derivative captured as one CUDA graph.
A slot holds at most its atom ceiling; :meth:`write_system` puts another system
into a slot in place, without a recapture.
:class:`~fastmd.interfaces.replica_md.ReplicaMolecularDynamics` stays the entry
point for replicas of one system, ``GPUResidentMolecularDynamics`` for one
system.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import gc
import math
import numbers
import time
import warnings

import numpy as np
import torch
from ase import Atoms, units

from ..adapters import load_adapter
from ..adapters.memory import MeasuredMemoryModel
from ..adapters.negotiate import batch_systems, check_system_cells, dry_run, negotiate, negotiated, size_unions
from ..adapters.protocol import CapacityStep, NegotiationError
from ..config import ConfigurationWarning, InferenceConfig, resolve_config
from ..engine import saved_recompute
from ..engine._graph_capture import capture_graph
from ..engine.entry_config import apply_checkpoint, release_adapter, use_adapter
from ..engine.policy import fusion_scope
from ..execution.gpu_integrator import default_fix_com, kinetic_energy_fp64
from ..execution.gpu_nhc import maxwell_boltzmann_momenta
from ..engine.energy_derivative import engine_derivative
from ..execution.replica_batch import (
    DEFAULT_UNION_ROWS,
    UNION_PROBE_FRACTIONS,
    graph_builder_options,
    union_capacity,
    union_demand,
    union_headrooms,
    union_triplet_capacity,
)
from ..execution.system_batch import (
    SlotLayout,
    SlotWholeStepRunner,
    SystemNoseHooverChainNVT,
    SystemVelocityVerlet,
    slot_union_builder,
    slot_union_demand,
    slot_union_memory_runs,
    slot_union_persistent_bytes,
    slot_union_probe_capacities,
    system_energy_derivative,
    systems_self_check,
    validate_cell,
)
from ..graph.fixed_capacity_builder import FixedCapacityGraphBuilder
from ..precision import warn_implicit_tf32
from .calculator import require_finite_geometry
from .replica_md import _COUNTERS, _MEASURED_UNION, ENSEMBLES, _Union, _UnionMolecularDynamics

# Union size by throughput (``union_rows`` None; MatRIS in MD, every model in a relaxation): the step time of unions of
# the first systems is measured at rows doubling from the largest system (at least UNION_PROBE_MIN_ROWS rows) up to all
# the systems' rows (at most DEFAULT_UNION_ROWS): one evaluation of the union, eager and then captured and replayed
# (MatRIS at the recompute level its own rule picks for that union, which may climb with the rows and costs time). A
# union is as fast as the best one where its rows per second fall short of the best measured ones by at most
# UNION_THROUGHPUT_SLACK (the probes' timing noise and the mix of their first systems are a few percent). Where the
# rows per second keep growing with the rows (a model whose small unions wait on their launches: MACE, SevenNet), the
# largest union is the fastest; where they stay level (a model whose arithmetic grows with its rows: ORB v3 in IEEE
# FP32, MatRIS 10M), a smaller union does as much in the same time, and a union that needs recompute does less.
UNION_PROBE_MIN_ROWS = 128
UNION_THROUGHPUT_SLACK = 0.05
# Replays timed per probe (their fastest counts).
_PROBE_REPLAYS = 2


def step_seconds_at(points, rows: float, column: int = 1) -> float | None:
    """The time of a union of ``rows`` rows from measured ``points`` (``(rows, seconds of a step captured, of a step
    eager, of the capture of a step)`` per probe, ascending; ``column`` 1, 2 or 3): piecewise linear between them, the
    smallest one's below it (the launches of a small union bound its step from below), and above the largest one at
    its rows per second (the per-row arithmetic). None without points."""
    if not points:
        return None
    sizes = [point[0] for point in points]
    values = [point[column] for point in points]
    if rows <= sizes[0]:
        return float(values[0])
    if rows >= sizes[-1]:
        return float(values[-1]) * rows / sizes[-1]
    return float(np.interp(rows, sizes, values))


class _SlotUnion(_Union):
    """One union of systems ``[start, stop)`` in slots: its layout, the current systems' cells, species and
    atom counts, and its integrator, runner and pair capacity."""

    def __init__(self, start: int, stop: int, ceilings) -> None:
        super().__init__(start, stop)
        self.layout = SlotLayout(tuple(int(c) for c in ceilings))
        self.cells = None  # [systems, 3, 3] float64 on the device
        self.numbers = None  # [rows] int32 on the device (a vacant row: its system's first element)
        self.atoms: list[int] = []


class MultiSystemMolecularDynamics(_UnionMolecularDynamics):
    """S different systems advanced together (see the module docstring).

    Parameters
    ----------
    model:
        An adapter, or a name or checkpoint path that :func:`~fastmd.adapters.load_adapter` loads
        (``model_path``, ``model_dtype`` as there). MatRIS runs by its own rules (as in
        ``ReplicaMolecularDynamics``; its evaluate takes each system's cell from the batch's systems). A model
        that does not declare ``"union"`` or ``"segments"`` or does not take the engine's per-edge shifts
        (``check_system_cells``), or fails its check on two of the systems, raises
        :class:`~fastmd.adapters.protocol.NegotiationError`.
    systems:
        The systems (ASE atoms, periodic in all three directions, each with its own cell).
    momenta:
        One ``(n_s, 3)`` float64 array per system in ASE units; default: Maxwell-Boltzmann at ``temperature`` for
        system ``s`` with ``seed + s`` (ASE's ``MaxwellBoltzmannDistribution``).
    ensemble, temperature, timestep, taut, tdamp, tchain, tloop, fix_com:
        As in :class:`~fastmd.interfaces.replica_md.ReplicaMolecularDynamics`, per system (its own chain, degrees
        of freedom ``3 n_s`` and centre of mass).
    steps_per_launch:
        MD steps per captured window; ``run``, ``loginterval`` and the logged steps are whole windows.
    slot_atoms:
        The slots' atom ceilings: None (each system's own atom count), one int (at least that many for every
        slot) or one per system (at least its atom count). A slot's rows past its system are vacant: room for a
        larger system written in later (:meth:`write_system`).
    union_rows:
        The most rows (slot ceilings) per union; default ``DEFAULT_UNION_ROWS``, and for MatRIS the rows of the
        union its measured throughput favours (``UNION_THROUGHPUT_SLACK``'s comment: the largest of the measured
        unions near the best rows per second at the lowest recompute level). More systems run as several unions of
        consecutive systems with balanced rows, sharing one CUDA graph memory pool. Unions that run out of CUDA
        memory as they start are released and started again with at most half the largest union's rows (a union
        of one system that does: :class:`~fastmd.adapters.protocol.NegotiationError`); a union whose capacity
        growth runs out of CUDA memory during :meth:`run` is split in two (``stats()["splits"]``; a system that
        does not fit alone: ``NegotiationError``).
    capacity_headroom:
        The relative pair headroom a capacity growth adds over the overflowing demand (default: the model's, else
        ``UNION_GROWTH_HEADROOM``); a union's first capture holds the model's own capacity tiers (MatRIS's pairs:
        with the pair headroom too). A line graph's
        triplets grow by the model's triplet headroom (``CapacityStep.t_union_headroom``, else
        ``UNION_TRIPLET_GROWTH_HEADROOM``).
    config, model_path, model_dtype, device, self_check:
        As in ``ReplicaMolecularDynamics`` (the self-check evaluates the smallest system and the smallest one in
        another cell, else the second smallest, as a union with vacant rows in its slots and alone, and the union
        with twice the cell as the model's ``cell`` argument; a set of one system checks two differently rattled
        copies of it the same way).
    trajectory, logfile, loginterval:
        ``trajectory``: a file name with ``{system}`` (e.g. ``"md_{system:04d}.traj"``): one ASE trajectory per
        system, a frame every ``loginterval`` steps and at the start; ``logfile``: one line per system and
        logged step (time, energies, temperature).
    """

    _members = "systems"
    _entry = "MultiSystemMolecularDynamics"  # the public entry point, in warnings

    def __init__(self, model, systems: Sequence[Atoms], *, momenta=None, ensemble: str = "nvt_nhc",
                 temperature: float = 300.0, timestep: float = 1.0, taut: float = 100.0, tdamp: float | None = None,
                 tchain: int = 3, tloop: int = 1, fix_com: bool | None = None, seed: int = 0,
                 steps_per_launch: int = 10, slot_atoms=None, union_rows: int | None = None,
                 capacity_headroom: float | None = None, config: InferenceConfig | None = None,
                 model_path: str | None = None, model_dtype: str | None = None, device=None,
                 trajectory: str | None = None, logfile: str | None = None, loginterval: int = 0,
                 self_check: bool = True) -> None:
        where = "MultiSystemMolecularDynamics"
        ensemble = str(ensemble).lower()
        if ensemble not in ENSEMBLES:
            raise ValueError(f"{where}: ensemble must be one of {ENSEMBLES}, got {ensemble!r}")
        if ensemble == "nvt_nhc" and fix_com:
            raise ValueError(f"{where}: ensemble='nvt_nhc' has no centre-of-mass correction (ASE "
                             "NoseHooverChainNVT); fix_com must be None or False")
        for name, value, low in (("steps_per_launch", steps_per_launch, 1), ("loginterval", loginterval, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{where}: {name} must be an integer >= {low}")
        if loginterval % steps_per_launch:
            raise ValueError(f"{where}: loginterval={loginterval} must be a multiple of "
                             f"steps_per_launch={steps_per_launch} (a window's inner steps are not visible)")
        if trajectory is not None and "{system" not in str(trajectory):
            raise ValueError(f"{where}: trajectory must name each system's file with '{{system}}'")
        if (trajectory is not None or logfile is not None) and not loginterval:
            raise ValueError(f"{where}: trajectory and logfile write every loginterval steps: give loginterval > 0")
        finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        if not finite(timestep) or timestep <= 0:
            raise ValueError(f"{where}: timestep must be finite and positive")
        if ensemble != "nve" and (not finite(temperature) or temperature <= 0):
            raise ValueError(f"{where}: ensemble={ensemble!r} needs a finite positive temperature")
        for name, value in (("taut", taut), ("tdamp", tdamp)):
            if value is not None and (not finite(value) or value <= 0):
                raise ValueError(f"{where}: {name} must be finite and positive")
        if capacity_headroom is not None and (not finite(capacity_headroom) or not 0.0 <= capacity_headroom <= 1.0):
            raise ValueError(f"{where}: capacity_headroom must be None or in [0, 1]")
        if union_rows is not None and (isinstance(union_rows, bool) or not isinstance(union_rows, int)
                                       or union_rows < 1):
            raise ValueError(f"{where}: union_rows must be a positive integer")
        systems = list(systems)
        if not systems or not all(isinstance(atoms, Atoms) and len(atoms) > 0 for atoms in systems):
            raise ValueError(f"{where}: systems must be one or more ASE Atoms with atoms")
        for index, atoms in enumerate(systems):
            self._check_system(atoms, f"{where}: system {index}")
        counts = [len(atoms) for atoms in systems]
        if momenta is None:
            momenta = [None] * len(systems)
        elif len(momenta) != len(systems):
            raise ValueError(f"{where}: momenta must hold one (n_s, 3) array per system")
        momenta = [None if value is None else np.asarray(value, dtype=np.float64) for value in momenta]
        for index, (value, n) in enumerate(zip(momenta, counts)):
            if value is not None and (value.shape != (n, 3) or not np.isfinite(value).all()):
                raise ValueError(f"{where}: momenta of system {index} must be finite with shape ({n}, 3)")
        self.ceilings = self._ceilings(slot_atoms, counts, where)
        self.systems = len(systems)
        self.ensemble = ensemble
        self.timestep = float(timestep)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.steps_per_launch = steps_per_launch
        self.loginterval = loginterval
        self._headroom = capacity_headroom
        self.device = torch.device(device if device is not None else "cuda")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError(f"{where} runs on a CUDA device")
        if self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        torch.cuda.set_device(self.device)
        resolved = resolve_config(config, default_execution="whole_step")
        if resolved.execution not in ("auto", "whole_step"):
            raise ValueError(f"{where} runs whole-step CUDA graphs: execution must be 'auto' or 'whole_step', got "
                             f"{resolved.execution!r}")
        if resolved.checkpoint_enabled is not False:  # as GPU-resident MD's entry points
            warnings.warn("Activation checkpointing inside CUDA Graph capture is not validated. Use checkpoint='off' "
                          "for the validated graph path; 'auto' is a row-count threshold, not a capture-safety policy. "
                          "The requested setting is unchanged.", ConfigurationWarning, stacklevel=2)
        self._adapter = load_adapter(model, model_path=model_path, device=str(self.device), model_dtype=model_dtype,
                                     enable_checkpoint=resolved.checkpoint_enabled,
                                     optimization_profile=resolved.optimization_profile)
        use_adapter(self, model)  # an adapter loaded by name is closed with its last user (also on a failure below)
        self._trajectories = False
        self._log = None
        try:
            if self._adapter is model or self._adapter.module is model:  # supplied, not loaded
                self._adapter.module.to(self.device)
                apply_checkpoint(self._adapter.module, resolved.checkpoint_enabled)
            self._setup(systems, momenta, resolved, taut=taut, tdamp=tdamp, tchain=tchain, tloop=tloop,
                        fix_com=fix_com, union_rows=union_rows, self_check=self_check)
            with self._config.scope():  # at the precision the model calls run with (another model's: IEEE FP32)
                warn_implicit_tf32(self._config.matmul_precision, self._entry)
            self._trajectory_name = trajectory
            if logfile is not None:
                self._log = open(logfile, "w")
                self._log.write(f"{'Time[ps]':>12} {'System':>8} {'Etot[eV]':>16} {'Epot[eV]':>16} "
                                f"{'Ekin[eV]':>14} {'T[K]':>10}\n")
        except BaseException:
            self._pool_keeper = None
            release_adapter(self)  # a failed constructor gives its adapter back now, not at its collection
            raise
        self.nsteps = 0
        self._logged_initial = False

    @staticmethod
    def _check_system(atoms: Atoms, where: str) -> np.ndarray:
        """Host checks of a system: periodic in all three directions, finite coordinates, a non-singular cell;
        its cell as float64."""
        if not bool(np.all(atoms.get_pbc())):
            raise ValueError(f"{where}: the cell must be periodic in all three directions")
        require_finite_geometry(atoms.positions, atoms.cell.array, where)
        return validate_cell(atoms.cell.array, f"{where}: the cell")

    @staticmethod
    def _ceilings(slot_atoms, counts: list[int], where: str) -> list[int]:
        if slot_atoms is None:
            return list(counts)
        if isinstance(slot_atoms, numbers.Integral) and not isinstance(slot_atoms, bool):
            return [max(n, int(slot_atoms)) for n in counts]
        ceilings = [int(c) for c in slot_atoms]
        if len(ceilings) != len(counts) or any(c < n for c, n in zip(ceilings, counts)):
            raise ValueError(f"{where}: slot_atoms must hold one ceiling per system, at least its atom count")
        return ceilings

    # ------------------------------------------------------------------ construction
    def _setup(self, systems, momenta, resolved, *, taut, tdamp, tchain, tloop, fix_com, union_rows,
               self_check) -> None:
        adapter = self._adapter
        check_system_cells(adapter)  # before anything is negotiated or measured
        whole_step = replace(resolved, execution="whole_step")
        if negotiated(adapter):
            plan = negotiate(adapter, whole_step, entry="gpu_md", device=self.device)
        else:  # MatRIS: the runtime's own rules, recorded as GPU-resident MD records them (its largest system)
            plan = dry_run(adapter, whole_step, entry="gpu_md", device=self.device,
                           atoms=max(systems, key=len))
        execution = plan.decision("execution")
        if execution.chosen != "whole_step":
            raise NegotiationError(f"MultiSystemMolecularDynamics needs a whole-step CUDA graph for model "
                                   f"{adapter.name!r}: {execution.reason}")
        # The plan's precision applies to every model call from here on (IEEE FP32 unless TF32 is asked for;
        # MatRIS's configuration as it is).
        self._config = (replace(whole_step, matmul_precision=plan.matmul_precision) if negotiated(adapter)
                        else whole_step)
        # As GPU-resident MD: eval mode and frozen weights (forces are derivatives in the coordinates; MatRIS's
        # inference kernels need weights without gradients), and the baseline of MatRIS's memory decisions.
        module = adapter.module
        module.eval()
        saved_recompute.check_model(module, saved_recompute.explicit_mode(resolved.saved_tensor_recompute))
        saved_recompute.note_entry_point(module)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        # float64 state: a float64 model the engine differentiates (MatRIS evaluates float32 graphs)
        self._float64 = engine_derivative(adapter) is not None and adapter.topology_spec().model_dtype == "float64"
        if self._float64 and self.ensemble != "nvt_nhc":
            raise ValueError("MultiSystemMolecularDynamics: a float64 model needs ensemble='nvt_nhc' (the integrator "
                             "with a float64 state); nve and nvt keep a float32 state")
        self.candidate_skin = float(self._config.capacity.candidate_skin)
        self._line_graph = adapter.topology_spec().triplet_cutoff is not None  # each system's triplets too
        self._system_tables(systems)
        self.splits = []  # unions split by a growth out of memory: (rows, the parts' rows, why)
        self._kinds = dict(taut=float(taut), tdamp=float(100.0 * self.timestep if tdamp is None else tdamp),
                           tchain=int(tchain), tloop=int(tloop),
                           fix_com=default_fix_com(self.ensemble) if fix_com is None else bool(fix_com))
        # The checks and the probes of time and memory in the runner's fusion scope (MatRIS's fused kernels; generic:
        # none).
        with self._config.scope(), fusion_scope(self._config.optimization_profile != "generic"):
            if self_check:
                floor, energy_floor = (1e-10, 1e-12) if self._float64 else (1e-5, 1e-6)
                check = self._self_check(floor, energy_floor)
                plan = batch_systems(plan, adapter, systems=self.systems, self_check=check, own_cells=True)
            else:
                plan = batch_systems(plan, adapter, systems=self.systems, own_cells=True)
            # The rows per union by throughput where they are not given (the systems in the slots may change here:
            # a relaxation's slots and queue follow the union it chooses).
            union_rows, throughput = self._union_rows(union_rows)
            self._capacity_tiers()
            union_rows, plan = self._size_unions(plan, union_rows)
            if throughput is not None:
                plan.decisions[-1].evidence["throughput"] = throughput
            positions = [np.asarray(atoms.get_positions(), dtype=np.float64) for atoms in self.templates]
            if len(momenta) != self.systems:  # other systems in the slots (a relaxation's: FIRE starts at rest)
                momenta = [None] * self.systems
            momenta = [maxwell_boltzmann_momenta(masses, self.temperature, seed=self.seed + index) if value is None
                       else value for index, (masses, value) in enumerate(zip(self.masses, momenta))]
            while True:  # unions that run out of CUDA memory as they start: fewer rows, down to one system each
                bounds = self._partition(union_rows)
                self._unions = [_SlotUnion(start, stop, self.ceilings[start:stop])
                                for start, stop in zip(bounds[:-1], bounds[1:])]
                # One CUDA graph memory pool for every union's captures: they replay one after another.
                self._graph_pool = torch.cuda.graph_pool_handle()
                try:
                    for union in self._unions:
                        self._start_union(union, positions[union.start:union.stop], momenta[union.start:union.stop])
                    break
                except torch.OutOfMemoryError as error:
                    why = str(error).splitlines()[0][:160]
                # As ReplicaMolecularDynamics: release them outside the except block (its traceback would pin the
                # failed union's buffers), and start them again with at most half the largest union's rows (less
                # where the partition would not change).
                count, failed = len(self._unions), max(union.layout.rows for union in self._unions)
                self._unions, self._graph_pool, self._pool_keeper = [], None, None
                gc.collect()
                torch.cuda.empty_cache()
                if count == self.systems:
                    raise NegotiationError(f"{self.systems} systems do not fit: a union of one system ran out of CUDA "
                                           f"memory as it started ({why})")
                union_rows = max(1, failed // 2)
                while union_rows > 1 and len(self._partition(union_rows)) - 1 <= count:
                    union_rows = max(1, union_rows // 2)
                plan.decisions = [replace(decision, chosen=str(union_rows), reason=(
                    f"{decision.reason}; unions of {failed} rows ran out of CUDA memory as they started ({why}): "
                    f"{union_rows} rows per union"), evidence={**decision.evidence, "out_of_memory_at_start": [
                        *decision.evidence.get("out_of_memory_at_start", []), failed]})
                                  if decision.id == "union_size" else decision for decision in plan.decisions]
            # The decision records the partition the systems run in (rows per union).
            plan.decisions = [replace(decision, evidence={**decision.evidence, "unions": [
                union.layout.rows for union in self._unions]}) if decision.id == "union_size" else decision
                for decision in plan.decisions]
            self._plan = plan

    def _system_tables(self, systems) -> None:
        """The per-system tables of ``systems`` in slot order: templates, masses, species, cells, writes."""
        self.templates = [atoms.copy() for atoms in systems]
        self._writes = [0] * self.systems  # the systems written into each slot since construction
        self.masses = [np.asarray(atoms.get_masses(), dtype=np.float64) for atoms in systems]
        self._numbers = [np.asarray(atoms.get_atomic_numbers(), dtype=np.int32) for atoms in systems]
        self._cells = [np.asarray(atoms.cell.array, dtype=np.float64) for atoms in systems]

    def _capacity_tiers(self) -> None:
        """The model's capacity step and the unions' growth headrooms; MatRIS's from the configuration and the
        systems in the slots (another model declares its own)."""
        adapter = self._adapter
        self._capacity_step = (adapter.capacity_step() if callable(getattr(adapter, "capacity_step", None))
                               else None if negotiated(adapter) else self._matris_capacity_step(self.templates))
        self.capacity_headroom, self.triplet_headroom = union_headrooms(self._capacity_step, self._headroom)

    def _union_rows(self, union_rows: int | None) -> tuple[int, dict | None]:
        """The rows per union before the memory sizing, and the throughput measurement's record (None where none
        ran): ``union_rows`` as given; else for a model the runtime plans by its own rules (MatRIS, whose
        recompute level climbs with a union's rows) by throughput (``UNION_THROUGHPUT_SLACK``'s comment): of the
        measured unions within the slack of the best rows per second, the largest of those at the lowest recompute
        level, ``DEFAULT_UNION_ROWS`` where that is the union of every system; else ``DEFAULT_UNION_ROWS`` (a union
        boundary costs launches and gains nothing where the rows per second hold: #76's 128 WBM structures, MACE,
        SevenNet, CHGNet and ORB v3, level or growing up to all of them; the probes would cost 7-20 s)."""
        self._step_seconds = []
        if union_rows is not None:
            return union_rows, None
        if negotiated(self._adapter):
            return DEFAULT_UNION_ROWS, None
        points = self._step_seconds = self._probe_step_seconds(min(sum(self.ceilings), DEFAULT_UNION_ROWS))
        if len(points) < 2:
            return DEFAULT_UNION_ROWS, None
        best = max(rows / seconds for rows, seconds, *_ in points)
        level = [point for point in points if point[0] / point[1] >= (1.0 - UNION_THROUGHPUT_SLACK) * best]
        lowest = min(point[4] for point in level)
        chosen = max(point[0] for point in level if point[4] == lowest)
        record = self._throughput_record(points, chosen)
        # a probe holds its first systems' atoms; the unions hold slots of their ceilings
        k = int(np.searchsorted(np.cumsum([len(numbers) for numbers in self._numbers]), chosen)) + 1
        return (DEFAULT_UNION_ROWS if k >= self.systems else sum(self.ceilings[:k])), record

    def _throughput_record(self, points, chosen: int) -> dict:
        return {"chosen_rows": int(chosen), "slack": UNION_THROUGHPUT_SLACK,
                "probes": [{"rows": int(rows), "step_s": round(seconds, 6), "eager_step_s": round(eager, 6),
                            "capture_s": round(capture, 6), "rows_per_s": round(rows / seconds, 1),
                            "recompute": saved_recompute.LADDER[level]}
                           for rows, seconds, eager, capture, level in points],
                "probe_limit": getattr(self, "_probe_limit", None),
                "probe_seconds": round(getattr(self, "_probe_seconds", 0.0), 3)}

    def _probe_step_seconds(self, total: int, *, stress: bool = False,
                            stop_when_level: bool = False) -> list[tuple[int, float, float, float, int]]:
        """The step time of slot unions of the first systems (each slot its own atoms, in slot order) at rows
        doubling from the largest system (at least ``UNION_PROBE_MIN_ROWS``) up to ``total``: one ``(rows, seconds
        captured, seconds eager, seconds of the capture, recompute level)`` per union (one evaluation, with each
        system's stress for ``stress``; the eager one after a first, untimed; the level MatRIS's automatic recompute
        chose, as an index of ``saved_recompute.LADDER``, 0 for another model), none where ``total`` is less than
        twice the first. Stops at a union that runs
        out of CUDA memory or fails (``_probe_limit`` says why), and with ``stop_when_level`` after two doublings
        whose rows per second gained no more than ``UNION_THROUGHPUT_SLACK`` on the best before them. MatRIS's
        automatic recompute decides each union by itself, in a state of the probes' own
        (``saved_recompute.detached_auto``: its rule never steps down from a larger union's level); the entry
        points' state stays as it was."""
        self._probe_limit, started = None, time.perf_counter()
        sizes = [len(numbers) for numbers in self._numbers]
        first = max(UNION_PROBE_MIN_ROWS, max(sizes))
        if total < 2 * first:
            return []
        targets, size = [], first
        while 2 * size <= total:
            targets.append(size)
            size *= 2
        targets.append(int(total))
        counts = np.cumsum(sizes)
        points, why = [], None
        stream = torch.cuda.Stream(self.device)
        with saved_recompute.detached_auto(self._adapter.module, self._config) as forget:
            for target in targets:
                k = min(int(np.searchsorted(counts, target)) + 1, self.systems)
                if points and int(counts[k - 1]) <= points[-1][0]:
                    continue
                forget()
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    points.append(self._probe_union(k, stream, stress))
                except (torch.OutOfMemoryError, RuntimeError) as error:  # the probes inform a choice: they stop
                    why = (f"a union of {int(counts[k - 1])} rows failed ({type(error).__name__}: "
                           f"{str(error).splitlines()[0][:120] if str(error) else ''})")
                if why is not None:  # outside the except block: its traceback would pin the failed union's buffers
                    self._probe_limit = why
                    break
                rates = [rows / seconds for rows, seconds, *_ in points]
                if stop_when_level and len(rates) >= 3 and all(
                        rates[last] <= (1.0 + UNION_THROUGHPUT_SLACK) * max(rates[:last]) for last in (-1, -2)):
                    self._probe_limit = f"the rows per second stayed level up to {points[-1][0]} rows"
                    break
        gc.collect()
        torch.cuda.empty_cache()
        self._probe_seconds = time.perf_counter() - started
        return points

    def _probe_union(self, k: int, stream, stress: bool) -> tuple[int, float, float, float, int]:
        """``(rows, seconds captured, seconds eager, seconds of the capture, recompute level)`` of one evaluation of
        the slot union of the first ``k`` systems at its own demand (the model's capacity tiers)."""
        adapter, device = self._adapter, self.device
        atoms = [len(numbers) for numbers in self._numbers[:k]]
        layout = SlotLayout(tuple(atoms))
        numbers = torch.as_tensor(np.concatenate(self._numbers[:k]), device=device)
        positions = torch.as_tensor(np.concatenate([system.get_positions() for system in self.templates[:k]]),
                                    dtype=torch.float32, device=device)
        cells = torch.as_tensor(np.stack(self._cells[:k]), dtype=torch.float64, device=device)
        pairs, triplets = slot_union_demand(adapter, positions, layout=layout, cells=cells, atomic_numbers=numbers,
                                            atoms=atoms, candidate_skin=self.candidate_skin, device=device)
        step = self._probe_capacity_step()
        builder = slot_union_builder(
            adapter, layout=layout, cells=cells, atomic_numbers=numbers, atoms=atoms,
            u_capacity=union_capacity(pairs, step, line_graph=self._line_graph),
            t_capacity=union_triplet_capacity(triplets, step, replicas=k) if self._line_graph else 0,
            candidate_skin=self.candidate_skin, device=device)
        builder.refresh_candidates(positions)

        def evaluate() -> None:
            graph, _ = builder.build(positions, rebuild_candidates=False)
            system_energy_derivative(adapter, builder.batch, graph, stress=stress)

        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            evaluate()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            evaluate()
            torch.cuda.synchronize(device)
            eager = time.perf_counter() - started
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with capture_graph(graph, stream=stream):
            evaluate()
        torch.cuda.synchronize(device)
        capture = time.perf_counter() - started
        replays = []
        for _ in range(_PROBE_REPLAYS):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize(device)
            replays.append(time.perf_counter() - started)
        del graph, builder
        level = saved_recompute.LADDER.index(saved_recompute.current_mode(adapter.module, self._config))
        return layout.rows, min(replays), eager, capture, level

    def _probe_capacity_step(self):
        """The capacity tiers of a probe: the model's, MatRIS's the configuration's (no headroom: a probe holds its
        own demand)."""
        if callable(getattr(self._adapter, "capacity_step", None)):
            return self._adapter.capacity_step()
        if negotiated(self._adapter):
            return None
        capacity = self._config.capacity
        return CapacityStep(u_step=capacity.u_step, min_pad_u=capacity.min_pad_u, t_step=capacity.t_step,
                            min_pad_t=capacity.min_pad_t)

    def _matris_capacity_step(self, systems) -> CapacityStep:
        """MatRIS's capacity tiers in slots: the configuration's pair and triplet steps and padding, and a headroom
        (its first capture's pairs' and each growth's) of CapturePolicy's growth fraction (GPU-resident MD's
        whole-step growth) plus the minimum padding of all systems but one relative to their counts (as
        ``ReplicaMolecularDynamics`` pads MatRIS's replicas): a union pads as its systems would alone."""
        from ..engine.capacity_policy import CapturePolicy
        capacity = self._config.capacity
        growth = CapturePolicy.from_config(capacity).growth_headroom
        pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        pairs = triplets = 0
        for atoms in systems:
            demand = union_demand(self._adapter, torch.as_tensor(atoms.get_positions(), dtype=torch.float32,
                                                                 device=self.device), replicas=1,
                                  cell=torch.as_tensor(atoms.cell.array, dtype=torch.float64, device=self.device),
                                  atomic_numbers=torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.int32,
                                                                 device=self.device),
                                  pbc=pbc, candidate_skin=self.candidate_skin, device=self.device)
            pairs, triplets = pairs + demand[0], triplets + demand[1]
        others = len(systems) - 1
        return CapacityStep(u_step=capacity.u_step, min_pad_u=capacity.min_pad_u, t_step=capacity.t_step,
                            min_pad_t=capacity.min_pad_t,
                            u_union_headroom=min(1.0, growth + others * capacity.min_pad_u / max(pairs, 1)),
                            t_union_headroom=min(1.0, growth + others * capacity.min_pad_t / max(triplets, 1)))

    def _self_check(self, floor: float, energy_floor: float):
        """The negotiation's check of the declaration (``systems_self_check``: in slots with vacant rows, and with
        twice the cell as the model's argument): the smallest system and the smallest one in another cell (else
        the second smallest) as one union and alone; two differently rattled copies of the system for a set of
        one."""
        geometry = dict(candidate_skin=self.candidate_skin, device=self.device)
        order = sorted(range(self.systems), key=lambda index: (len(self.templates[index]), index))
        other = next((index for index in order[1:] if not np.array_equal(self._cells[index], self._cells[order[0]])),
                     order[1] if self.systems > 1 else order[0])
        smallest = [order[0], other]
        pair = [(self.templates[index].get_positions(), self._cells[index], self._numbers[index])
                for index in smallest]
        copies = {"evaluated": "two copies of the system"} if self.systems == 1 else {}
        return lambda: {**systems_self_check(self._adapter, pair, absolute_floor=floor, energy_floor=energy_floor,
                                             **geometry), "system_indices": smallest, **copies}

    def _size_unions(self, plan, union_rows: int):
        """The rows per union the measured memory admits, and the plan with that decision ("union_size"):
        ``union_rows``, or less where the largest union of that partition would not fit (``size_unions``, in rows).
        The memory of a union is measured on sub-unions of the largest one (its first systems, at
        ``UNION_PROBE_FRACTIONS`` of them, eager and captured, peak reserved bytes), fitted over their edge
        capacities (with a line graph also their triplet capacities: ``slot_union_probe_capacities``) and predicted
        at the union's capacity after one growth, plus the other unions' buffers outside the shared pool; a union of
        fewer than ``_MEASURED_UNION`` systems is not measured."""
        bounds = self._partition(union_rows)
        rows = [sum(self.ceilings[start:stop]) for start, stop in zip(bounds[:-1], bounds[1:])]
        largest = int(np.argmax(rows))
        start, stop = int(bounds[largest]), int(bounds[largest + 1])
        requested, total = int(rows[largest]), int(sum(self.ceilings))
        systems = [(self.templates[index].get_positions(), self._cells[index], self._numbers[index])
                   for index in range(start, stop)]
        runs, demand = slot_union_memory_runs(self._adapter, systems, candidate_skin=self.candidate_skin,
                                              device=self.device, step=self._capacity_step,
                                              headroom=self.capacity_headroom, ceilings=self.ceilings[start:stop],
                                              triplet_headroom=self.triplet_headroom)
        occupied = sum(len(numbers) for _, _, numbers in systems)
        line_graph = self._line_graph

        def union_edges(size: int) -> int:  # a union of `size` rows, at the largest union's pairs per row
            return 2 * union_capacity(math.ceil(demand(stop - start)[0] * size / requested), self._capacity_step,
                                      headroom=self.capacity_headroom, line_graph=line_graph)

        def union_triplets(size: int) -> int:  # and at its triplets per row, its systems in proportion
            return union_triplet_capacity(math.ceil(demand(stop - start)[1] * size / requested), self._capacity_step,
                                          headroom=self.triplet_headroom,
                                          replicas=max(1, math.ceil((stop - start) * size / requested)))

        def probe_edges(k: int) -> int:  # the sub-union of the first k systems, grown once
            return 2 * union_capacity(demand(k)[0], self._capacity_step, headroom=self.capacity_headroom,
                                      line_graph=line_graph)

        def probe_triplets(k: int) -> int:
            return union_triplet_capacity(demand(k)[1], self._capacity_step, headroom=self.triplet_headroom,
                                          replicas=k)

        memory, persistent = None, 0
        if stop - start >= _MEASURED_UNION:
            # A line graph: (systems, triplet capacity) probes whose triplet ratios vary (the fit's per-triplet cost)
            # over those of the unions the sizing may choose.
            probes = (slot_union_probe_capacities(stop - start, requested, probe_edges, probe_triplets, union_edges,
                                                  union_triplets, slots=self.ceilings) if line_graph else
                      sorted({max(1, int(round(fraction * (stop - start)))) for fraction in UNION_PROBE_FRACTIONS}))
            try:
                memory = MeasuredMemoryModel.measure(runs, probes, self.device, triplets=line_graph,
                                                     metric="reserved")
            except torch.OutOfMemoryError as error:
                why = f"ran out of CUDA memory ({str(error).splitlines()[0][:160]})"
            except RuntimeError as error:
                if "capture" not in str(error).lower():
                    raise
                why = f"failed in a capture ({str(error).splitlines()[0][:160]})"
            # Outside the except blocks: a traceback would pin the failed run's graph and buffers.
            gc.collect()
            torch.cuda.empty_cache()
            if memory is None:
                memory = MeasuredMemoryModel.failed(f"the union memory measurement {why}", triplets=line_graph)
            else:  # the other unions' buffers, should the sizing leave more than one
                first = min(probe[0] if line_graph else probe for probe in probes)
                layout = SlotLayout(tuple(self.ceilings[start:start + first]))
                numbers = np.concatenate([np.full(ceiling, z[0], dtype=np.int32) for ceiling, (_, _, z) in
                                          zip(layout.ceilings, systems)])
                pairs, triplets = demand(first)
                builder = slot_union_builder(
                    self._adapter, layout=layout,
                    cells=torch.as_tensor(np.stack([cell for _, cell, _ in systems[:first]])),
                    atomic_numbers=torch.as_tensor(numbers), atoms=[len(z) for _, _, z in systems[:first]],
                    u_capacity=union_capacity(pairs, line_graph=line_graph),
                    t_capacity=union_triplet_capacity(triplets, replicas=first) if line_graph else 0,
                    candidate_skin=self.candidate_skin, device=self.device)
                persistent = slot_union_persistent_bytes(builder)
                del builder
                torch.cuda.empty_cache()
        triplet_sizes = dict(union_triplets=union_triplets) if line_graph else {}
        chosen, plan = size_unions(plan, requested=requested, memory=memory, union_edges=union_edges,
                                   device=self.device, replicas=total, persistent_per_replica=persistent, unit="rows",
                                   **triplet_sizes)
        plan.decisions[-1].evidence.update(occupied_rows=occupied, largest_union_systems=stop - start,
                                           largest_slot_rows=max(self.ceilings))
        if chosen >= requested:
            return union_rows, plan
        # A slot is the smallest union there is (its system alone): a slot beyond the admitted rows runs alone where
        # the measured memory admits a union of its own size.
        index = int(np.argmax(self.ceilings))
        if self.ceilings[index] > chosen:
            alone, _ = size_unions(plan, requested=self.ceilings[index], memory=memory, union_edges=union_edges,
                                   device=self.device, replicas=total, persistent_per_replica=persistent, unit="rows",
                                   **triplet_sizes)
            if alone < self.ceilings[index]:
                raise NegotiationError(f"system {index}'s slot of {self.ceilings[index]} rows does not fit the "
                                       f"measured memory (unions of {chosen} rows do): run it with fewer systems or a "
                                       "smaller slot_atoms")
        return chosen, plan

    def _partition(self, union_rows: int) -> np.ndarray:
        """The union bounds: consecutive systems in as few unions of at most ``union_rows`` rows as the systems in
        order allow (a slot larger than that alone in its union), with balanced rows (the smallest limit that needs
        no more unions)."""
        def pack(limit: int) -> list[int]:
            bounds, rows = [0], 0
            for index, ceiling in enumerate(self.ceilings):
                if rows and rows + ceiling > limit:
                    bounds.append(index)
                    rows = 0
                rows += ceiling
            return bounds + [self.systems]

        count = len(pack(int(union_rows))) - 1
        low, high = 1, int(union_rows)
        while low < high:  # the number of unions only falls as the limit grows
            middle = (low + high) // 2
            low, high = (low, middle) if len(pack(middle)) - 1 <= count else (middle + 1, high)
        return np.asarray(pack(low))

    def _union_tables(self, union: _SlotUnion, positions: list) -> tuple[torch.Tensor, torch.Tensor]:
        """The union's row tables at ``positions`` (one ``(n, 3)`` per system): its species (a vacant row: its
        system's first element) and float32 coordinates (a vacant row: the origin)."""
        layout = union.layout
        numbers = np.zeros(layout.rows, dtype=np.int32)
        coordinates = np.zeros((layout.rows, 3), dtype=np.float64)
        for slot, index in enumerate(range(union.start, union.stop)):
            start, n = layout.starts[slot], len(self._numbers[index])
            numbers[start:start + layout.ceilings[slot]] = self._numbers[index][0]
            numbers[start:start + n] = self._numbers[index]
            coordinates[start:start + n] = positions[slot]
        return (torch.as_tensor(numbers, device=self.device),
                torch.as_tensor(coordinates, dtype=torch.float32, device=self.device))

    def _start_union(self, union: _SlotUnion, positions: list, momenta: list) -> None:
        """The integrator of ``union`` at ``positions`` and ``momenta`` (one ``(n, 3)`` per system), and its first
        runner."""
        kinds = self._kinds
        union.atoms = [len(self._numbers[index]) for index in range(union.start, union.stop)]
        union.cells = torch.as_tensor(np.stack(self._cells[union.start:union.stop]), dtype=torch.float64,
                                      device=self.device)
        union.numbers, coordinates = self._union_tables(union, positions)
        layout = union.layout
        if self.ensemble == "nvt_nhc":
            union.integrator = SystemNoseHooverChainNVT(
                layout, self.masses[union.start:union.stop], positions, momenta,
                temperature_K=self.temperature, timestep=self.timestep * units.fs, tdamp=kinds["tdamp"] * units.fs,
                tchain=kinds["tchain"], tloop=kinds["tloop"], device=self.device,
                force_dtype=torch.float64 if self._float64 else torch.float32)
            state = union.integrator.positions.to(torch.float32), union.integrator.momenta.to(torch.float32)
        else:
            masses = np.ones(layout.rows, dtype=np.float64)
            moment = np.zeros((layout.rows, 3), dtype=np.float64)
            for slot, index in enumerate(range(union.start, union.stop)):
                start, n = layout.starts[slot], union.atoms[slot]
                masses[start:start + n] = self.masses[index]
                moment[start:start + n] = momenta[slot]
            occupied = torch.zeros(layout.rows, dtype=torch.bool, device=self.device)
            occupied.index_fill_(0, torch.as_tensor(layout.occupied_rows(union.atoms), device=self.device), True)
            union.integrator = SystemVelocityVerlet(
                torch.as_tensor(masses, dtype=torch.float32, device=self.device), self.timestep,
                ceilings=torch.tensor(layout.ceilings, dtype=torch.int64, device=self.device),
                atoms=torch.tensor(union.atoms, dtype=torch.int64, device=self.device),
                system_of_row=layout.system_of_row(self.device), occupied=occupied, ensemble=self.ensemble,
                temperature=self.temperature, taut_fs=kinds["taut"], fix_com=kinds["fix_com"])
            state = coordinates, torch.as_tensor(moment, dtype=torch.float32, device=self.device)
        self._set_capacities(union, state[0])
        self._capture(union, *state, forces=None)

    def _set_capacities(self, union: _SlotUnion, positions, *, grown: bool = False,
                        pair_headroom: bool = False) -> None:
        """``union``'s pair (and, with a line graph, triplet) capacity for its demand at ``positions``: the model's
        tiers, or with the union's growth headrooms (``grown``: a union captured again; ``pair_headroom``: the pairs'
        only)."""
        pairs, triplets = slot_union_demand(self._adapter, positions, layout=union.layout, cells=union.cells,
                                            atomic_numbers=union.numbers, atoms=union.atoms,
                                            candidate_skin=self.candidate_skin, device=self.device)
        # MatRIS's union also holds its systems' pair padding from the start (ReplicaMolecularDynamics._start_union).
        first = 0.0 if negotiated(self._adapter) and not pair_headroom else self.capacity_headroom
        union.u_capacity = union_capacity(pairs, self._capacity_step, headroom=self.capacity_headroom if grown else first,
                                          line_graph=self._line_graph)
        union.t_capacity = (union_triplet_capacity(triplets, self._capacity_step, replicas=union.replicas,
                                                   headroom=self.triplet_headroom if grown else 0.0)
                            if self._line_graph else 0)

    def _capture(self, union: _SlotUnion, positions, momenta, *, forces) -> None:
        """A runner of ``union`` at its capacity (a recapture when ``forces``, the cached F of the state, are
        given)."""
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self._keep_graph_pool_alive()
        builder = slot_union_builder(self._adapter, layout=union.layout, cells=union.cells,
                                     atomic_numbers=union.numbers, atoms=union.atoms, u_capacity=union.u_capacity,
                                     t_capacity=union.t_capacity, candidate_skin=self.candidate_skin,
                                     device=self.device)
        union.runner = SlotWholeStepRunner(
            model=self._adapter, builder=builder, integrator=union.integrator, positions=positions, momenta=momenta,
            forces=forces, steps_per_launch=self.steps_per_launch, warmup=self._config.capacity.warmup,
            graph_pool=self._graph_pool, config=self._config, recapture=forces is not None,
            validate_isolated_atoms=not self._config.handle_isolated_atoms)
        torch.cuda.synchronize(self.device)
        union.capture_seconds.append(time.perf_counter() - started)

    # ------------------------------------------------------------------ a growth out of memory
    @staticmethod
    def _split_groups(rows: Sequence[int], count: int) -> list[list[int]]:
        """``count`` groups (at most one per slot) of consecutive slots of ``rows`` rows each, each boundary where
        the rows before it come closest to their share."""
        n = len(rows)
        count, cumulative, edges = min(count, n), np.cumsum(rows), [0]
        for k in range(1, count):
            share = cumulative[-1] * k / count
            edges.append(min(range(edges[-1] + 1, n - count + k + 1), key=lambda b: abs(cumulative[b - 1] - share)))
        edges.append(n)
        return [list(range(a, b)) for a, b in zip(edges[:-1], edges[1:])]

    def _split(self, union: _SlotUnion, positions, momenta, forces, why: str):
        """``union``'s growth ran out of CUDA memory (``why``): its systems run on in two unions of consecutive
        slots with about half its rows each (four, eight... where a part runs out of memory again), each
        integrator's state carried over (positions, momenta, forces and energies, each system's thermostat chain),
        captured at the parts' demand with the growth headroom, in ``union``'s place; a system that does not fit
        alone raises :class:`~fastmd.adapters.protocol.NegotiationError`. Returns the parts."""
        position, count = self._unions.index(union), 2
        while True:
            groups = self._split_groups(union.layout.ceilings, count)
            if len(groups) == 1:  # one system: no smaller union holds it
                raise NegotiationError(f"{self._members} [{union.start}, {union.stop}): {why}, and it holds one "
                                       "system: run it with less memory in use")
            parts, failure = [], None
            try:
                for slots in groups:
                    parts.append(self._split_part(union, slots, positions, momenta, forces))
            except torch.OutOfMemoryError as error:
                failure = str(error).splitlines()[0][:160]
            except BaseException:
                for part in parts:
                    part.runner = part.integrator = None
                raise
            if failure is None:
                break
            for part in parts:  # outside the except block: its traceback would pin the failed part's buffers
                part.runner = part.integrator = None
            parts = []
            gc.collect()
            torch.cuda.empty_cache()
            if len(groups) == union.layout.count:
                raise NegotiationError(f"{self._members} [{union.start}, {union.stop}): {why}, and a union of one "
                                       f"system ran out of CUDA memory too ({failure}): run them with less memory "
                                       "in use or fewer systems")
            count *= 2
        parts[0].accumulated = dict(union.accumulated)
        parts[0].recaptures = union.recaptures + 1
        parts[0].capture_seconds = union.capture_seconds + parts[0].capture_seconds
        parts[0].max_edge_count, parts[0].max_triplet_count = union.max_edge_count, union.max_triplet_count
        self._unions[position:position + 1] = parts
        self.splits.append((union.layout.rows, [part.layout.rows for part in parts], why))
        union.runner = union.integrator = None
        return parts

    def _split_part(self, union: _SlotUnion, slots: list[int], positions, momenta, forces) -> _SlotUnion:
        """The union of ``union``'s consecutive ``slots`` with their systems' state (``positions``, ``momenta``,
        ``forces``: ``union``'s runner rows at its window's start; the Nose-Hoover chain's from its integrator),
        captured."""
        part = _SlotUnion(union.start + slots[0], union.start + slots[-1] + 1,
                          [union.layout.ceilings[slot] for slot in slots])
        starts, ceilings = union.layout.starts, union.layout.ceilings
        rows = torch.cat([torch.arange(starts[slot], starts[slot] + ceilings[slot]) for slot in slots]).to(self.device)
        part.atoms = [union.atoms[slot] for slot in slots]
        part.cells = union.cells.index_select(0, torch.as_tensor(slots, device=self.device)).clone()
        part.numbers = union.numbers.index_select(0, rows).clone()
        old, layout = union.integrator, part.layout
        if self.ensemble == "nvt_nhc":
            part.integrator = SystemNoseHooverChainNVT(
                layout, self.masses[part.start:part.stop], [old.system_view(old.state.q, slot) for slot in slots],
                [old.system_view(old.state.p, slot) for slot in slots], temperature_K=self.temperature,
                timestep=self.timestep * units.fs, tdamp=self._kinds["tdamp"] * units.fs,
                tchain=self._kinds["tchain"], tloop=self._kinds["tloop"], device=self.device,
                force_dtype=old.force_dtype)
            new = part.integrator
            with torch.no_grad():
                for k, slot in enumerate(slots):
                    new.system_view(new.state.f, k).copy_(old.system_view(old.state.f, slot))
                    for name in ("eta", "p_eta", "potential_energy"):
                        getattr(new.state, name)[k].copy_(getattr(old.state, name)[slot])
                new.backup()
            state = new.positions.to(torch.float32), new.momenta.to(torch.float32), new.forces.to(forces.dtype)
        else:
            masses = old.masses.reshape(-1).index_select(0, rows)
            occupied = old.occupied.index_select(0, rows)
            part.integrator = SystemVelocityVerlet(
                masses, self.timestep, ceilings=torch.tensor(layout.ceilings, dtype=torch.int64, device=self.device),
                atoms=torch.tensor(part.atoms, dtype=torch.int64, device=self.device),
                system_of_row=layout.system_of_row(self.device), occupied=occupied, ensemble=self.ensemble,
                temperature=self.temperature, taut_fs=self._kinds["taut"], fix_com=self._kinds["fix_com"])
            state = (positions.index_select(0, rows), momenta.index_select(0, rows), forces.index_select(0, rows))
        self._set_capacities(part, state[0], grown=True)
        self._capture(part, state[0], state[1], forces=state[2])
        return part

    # ------------------------------------------------------------------ a system into a slot
    def _locate(self, index: int) -> tuple[_SlotUnion, int]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < self.systems:
            raise IndexError(f"system index {index!r} out of range [0, {self.systems})")
        union = next(union for union in self._unions if union.start <= index < union.stop)
        return union, int(index) - union.start

    @torch.no_grad()
    def write_system(self, index: int, atoms: Atoms, *, momenta=None) -> None:
        """Put ``atoms`` into system ``index``'s slot in place: its positions, species, cell and atom count (at
        most the slot's ceiling), momenta (default: Maxwell-Boltzmann at the temperature with ``seed + index``), a
        fresh thermostat, and its forces and energy evaluated as a fresh single-system run starts (the system
        alone). Checks first, before anything changes: periodic in all three directions, finite coordinates, a
        non-singular cell the union can hold, at most the ceiling's atoms, and the model's own input check (its
        elements, in its ``prepare``) and the candidate capacity of the system alone. No recapture unless the
        union's edges then outgrow its capacity (an overflow at the next window grows it, as in ``run``). Its next
        windows run as a fresh single-system run would from this state; the other systems keep theirs. Between
        ``run`` calls only."""
        union, slot = self._locate(index)
        where = f"MultiSystemMolecularDynamics.write_system({index})"
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise ValueError(f"{where}: atoms must be ASE Atoms with atoms")
        cell = self._check_system(atoms, where)
        n, ceiling = len(atoms), union.layout.ceilings[slot]
        if n > ceiling:
            raise ValueError(f"{where}: {n} atoms exceed the slot's ceiling of {ceiling} (slot_atoms)")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
        masses = np.asarray(atoms.get_masses(), dtype=np.float64)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if momenta is None:
            momenta = maxwell_boltzmann_momenta(masses, self.temperature, seed=self.seed + int(index))
        momenta = np.asarray(momenta, dtype=np.float64)
        if momenta.shape != (n, 3) or not np.isfinite(momenta).all():
            raise ValueError(f"{where}: momenta must be finite with shape ({n}, 3)")
        runner, builder = union.runner, union.runner.builder
        try:
            cell64 = builder.check_cell(cell)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from error
        with self._config.scope():
            forces, energy = self._evaluate_alone(numbers, positions, cell, runner.validate_isolated_atoms, where)
        # Commit: nothing below rejects the system.
        builder.write_system(slot, atomic_numbers=torch.as_tensor(numbers), cell=cell64, fill_number=int(numbers[0]))
        self._numbers[index], self._cells[index], self.masses[index] = numbers, cell, masses
        self.templates[index] = atoms.copy()
        self._writes[index] += 1
        union.atoms[slot] = n
        union.cells[slot].copy_(cell64)
        union.numbers.copy_(builder.atomic_numbers_real)
        start = union.layout.starts[slot]
        rows = slice(start, start + ceiling)
        integrator = union.integrator
        if self.ensemble == "nvt_nhc":
            integrator.write_system(slot, masses=masses, positions=positions, momenta=momenta)
            runner.positions.copy_(integrator.positions)
            runner.momenta.copy_(integrator.momenta)
        else:
            integrator.masses[rows].fill_(1.0)
            integrator.masses[start:start + n].copy_(torch.as_tensor(masses[:, None]))
            integrator.atoms[slot] = n
            integrator.occupied.copy_(builder.occupied)
            for buffer, values in ((runner.positions, positions), (runner.momenta, momenta)):
                buffer[rows].zero_()
                buffer[start:start + n].copy_(torch.as_tensor(values))
        runner.forces[rows].zero_()
        runner.forces[start:start + n].copy_(forces)
        runner.potential_energy[slot].copy_(energy)
        if self.ensemble == "nvt_nhc":
            integrator.set_forces(runner.forces, energy=runner.potential_energy)
            integrator.backup()
        runner.reset(positions=runner.positions.clone(), momenta=runner.momenta.clone(), forces=runner.forces.clone(),
                     potential_energy=runner.potential_energy.clone(), reset_counters=False)

    def _evaluate_alone(self, numbers: np.ndarray, positions: np.ndarray, cell: np.ndarray, isolated: bool,
                        where: str) -> tuple[torch.Tensor, torch.Tensor]:
        """F(q0) ``(n, 3)`` (the model's dtype) and E(q0) (FP64) of one system alone, as a fresh single-system run
        starts: a fixed-capacity build at its own demand (the adapter's ``prepare`` checks its input) and the
        engine-owned derivative; ValueError where the model rejects the system, its candidates overflow, or it has
        isolated atoms that ``isolated`` (the union's check) rejects."""
        adapter, device = self._adapter, self.device
        cutoff = float(adapter.topology_spec().cutoff)
        geometry = dict(cell=torch.as_tensor(cell, dtype=torch.float64, device=device),
                        atomic_numbers=torch.as_tensor(numbers, device=device),
                        pbc=torch.ones(3, dtype=torch.bool, device=device), device=device)
        coordinates = torch.as_tensor(positions, dtype=torch.float64 if self._float64 else torch.float32,
                                      device=device)
        try:
            pairs, triplets = union_demand(adapter, coordinates.float(), replicas=1, candidate_skin=self.candidate_skin,
                                           **geometry)
            builder = FixedCapacityGraphBuilder(
                u_capacity=union_capacity(pairs, line_graph=self._line_graph), atom_graph_cutoff=cutoff,
                candidate_cutoff=cutoff + self.candidate_skin, padding=adapter.padding_rule(),
                prepare=adapter.prepare,
                **graph_builder_options(adapter, union_triplet_capacity(triplets) if self._line_graph else 0),
                **geometry)
        except ValueError as error:
            raise ValueError(f"{where}: the model rejects the system: {error}") from error
        builder.refresh_candidates(coordinates)
        try:
            builder.assert_candidate_capacity()
        except Exception as error:  # nvalchemiops' neighbour overflow, as a rejected system
            raise ValueError(f"{where}: {error}") from error
        graph, status = builder.build(coordinates, rebuild_candidates=False)
        if isolated and int(status.isolated_atom_count.item()):
            raise ValueError(f"{where}: {int(status.isolated_atom_count.item())} atoms without a neighbour within "
                             f"the cutoff {cutoff} Å")
        n = len(numbers)
        if engine_derivative(adapter) is None:  # a model that differentiates its own energy (MatRIS)
            with torch.enable_grad():  # its forces are autograd's; write_system runs under no_grad
                out = adapter.evaluate([graph], task="ef", is_training=False, n_real=n)
            energy = adapter.extensive_energy(out["e"].reshape(1), n).to(torch.float64)
            return out["f"][0][:n].detach(), energy.detach().reshape(-1)[0]
        out = engine_derivative(adapter)(builder.batch, graph, task="ef", n_real=n)
        return out["f"][0][:n].detach(), out["e"].reshape(-1)[0]

    # ------------------------------------------------------------------ state and observables
    def _system_rows(self, read) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array per system of the per-row tensor ``read(union)`` (``(rows, 3)``)."""
        out = []
        for union in self._unions:
            values = read(union).to(torch.float64).cpu().numpy()
            out.extend(values[start:start + n] for start, n in zip(union.layout.starts, union.atoms))
        return out

    def get_positions(self) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array of unwrapped positions per system (the MD state: the Nose-Hoover chain's
        float64 positions, or the runner's float32 ones for nve and nvt)."""
        return self._system_rows(lambda union: union.integrator.positions if self.ensemble == "nvt_nhc"
                                 else union.runner.positions)

    def get_momenta(self) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array of momenta per system."""
        return self._system_rows(lambda union: union.integrator.momenta if self.ensemble == "nvt_nhc"
                                 else union.runner.momenta)

    def get_forces(self) -> list[np.ndarray]:
        """One ``(n_s, 3)`` float64 array of forces per system at the current positions."""
        return self._system_rows(lambda union: union.integrator.forces if self.ensemble == "nvt_nhc"
                                 else union.runner.forces)

    def kinetic_energies(self) -> torch.Tensor:
        """``(S,)`` float64 kinetic energies (eV)."""
        momenta, masses = self.get_momenta(), self.masses
        return torch.tensor([float(kinetic_energy_fp64(torch.as_tensor(p), torch.as_tensor(m)))
                             for p, m in zip(momenta, masses)], dtype=torch.float64)

    def temperatures(self) -> torch.Tensor:
        """``(S,)`` temperatures (K): each system's ``Atoms.get_temperature()``."""
        atoms = torch.tensor([len(m) for m in self.masses], dtype=torch.float64)
        return 2.0 * self.kinetic_energies() / (3 * atoms * units.kB)

    def potential_energies(self) -> torch.Tensor:
        """``(S,)`` float64 potential energies (eV)."""
        return self._per_union(lambda union: (union.integrator.potential_energy if self.ensemble == "nvt_nhc"
                                              else union.runner.potential_energy).to(torch.float64)).cpu()

    def conserved_energies(self) -> torch.Tensor:
        """``(S,)``: the Nose-Hoover chain's conserved energy (``nvt_nhc``), else potential plus kinetic."""
        if self.ensemble == "nvt_nhc":
            return self._per_union(lambda union: union.integrator.conserved_energy()).cpu()
        return self.potential_energies() + self.kinetic_energies()

    def health(self) -> np.ndarray:
        """``(S,)`` bool: the systems whose state is finite (a window that leaves one non-finite raises
        :class:`~fastmd.execution.replica_batch.ReplicaHealthError` naming them)."""
        return torch.cat([union.runner.health() for union in self._unions]).numpy()

    def get_atoms(self, index: int) -> Atoms:
        """System ``index`` as ASE atoms with its positions and momenta."""
        atoms = self.templates[index].copy()
        atoms.set_positions(self.get_positions()[index])
        atoms.set_momenta(self.get_momenta()[index])
        return atoms

    def snapshot(self) -> dict:
        """The systems' MD state (device copies), for :meth:`restore` (the systems in the slots stay)."""
        unions = []
        for union in self._unions:
            runner = union.runner
            unions.append({"integrator": union.integrator.snapshot() if self.ensemble == "nvt_nhc" else None,
                           "atoms": list(union.atoms),
                           "state": [tensor.clone() for tensor in (runner.positions, runner.momenta, runner.forces,
                                                                   runner.potential_energy)]})
        return {"nsteps": self.nsteps, "layout": [(union.start, union.stop) for union in self._unions],
                "ensemble": self.ensemble, "writes": list(self._writes), "unions": unions}

    @torch.no_grad()
    def restore(self, snapshot: dict, *, momenta=None) -> None:
        """Back to ``snapshot`` (:meth:`snapshot` of this object, taken with the same systems in the slots: none
        written since), with new momenta if given (one ``(n_s, 3)`` per system; positions, forces and energies stay
        the snapshot's): the captured graphs and the grown capacities are kept; the window counters of
        :meth:`stats` start again."""
        if (snapshot.get("layout") != [(union.start, union.stop) for union in self._unions]
                or snapshot.get("ensemble") != self.ensemble or snapshot.get("writes") != self._writes):
            raise ValueError("MultiSystemMolecularDynamics.restore: the snapshot is not one of this object's states "
                             "with the systems it holds now")
        if momenta is not None:
            if len(momenta) != self.systems:
                raise ValueError("MultiSystemMolecularDynamics.restore: momenta must hold one (n_s, 3) array per "
                                 "system")
            momenta = [np.asarray(value, dtype=np.float64) for value in momenta]
            for index, (value, masses) in enumerate(zip(momenta, self.masses)):
                if value.shape != (len(masses), 3) or not np.isfinite(value).all():
                    raise ValueError(f"MultiSystemMolecularDynamics.restore: momenta of system {index} must be "
                                     f"finite with shape ({len(masses)}, 3)")
        with self._config.scope():
            for union, saved in zip(self._unions, snapshot["unions"]):
                positions, moment, forces, energies = saved["state"]
                if self.ensemble == "nvt_nhc":
                    integrator = union.integrator
                    integrator.load_snapshot(saved["integrator"])
                    if momenta is not None:
                        for slot, index in enumerate(range(union.start, union.stop)):
                            integrator.system_view(integrator.state.p, slot).copy_(
                                torch.as_tensor(np.asarray(momenta[index], dtype=np.float64)))
                    integrator.backup()
                    positions, moment = integrator.positions.to(torch.float32), integrator.momenta.to(torch.float32)
                    forces, energies = integrator.forces.to(union.runner.forces.dtype), integrator.potential_energy
                elif momenta is not None:
                    moment = moment.clone()
                    for slot, index in enumerate(range(union.start, union.stop)):
                        start, n = union.layout.starts[slot], union.atoms[slot]
                        moment[start:start + n].copy_(torch.as_tensor(np.asarray(momenta[index])))
                union.runner.reset(positions=positions, momenta=moment, forces=forces, potential_energy=energies)
                union.accumulated = dict.fromkeys(_COUNTERS, 0)
                union.max_edge_count = union.max_triplet_count = 0
        self.nsteps = snapshot["nsteps"]

    def stats(self) -> dict:
        """Counters over every union and every capture of it since construction or the last :meth:`restore`:
        windows launched, candidate refreshes, fallback windows, recovery replays and the largest edge count (with
        the capacities and their fill); since construction: recaptures (capacity growth) and capture times (s),
        the unions split by a growth out of memory ((rows, the parts' rows, why)); the slots' ceilings and atoms."""
        return {**self._union_stats(systems=self.systems),
                "slot_atoms": [list(union.layout.ceilings) for union in self._unions],
                "atoms": [list(union.atoms) for union in self._unions], "writes": list(self._writes),
                "splits": [list(split) for split in self.splits]}

    # ------------------------------------------------------------------ output
    def _write_log(self) -> None:
        if self._log is None and self._trajectory_name is None:
            return
        positions, momenta, forces = self.get_positions(), self.get_momenta(), self.get_forces()
        potential = self.potential_energies().numpy()
        kinetic = self.kinetic_energies().numpy()
        if self._trajectory_name is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            from ase.io import Trajectory

            # One file open at a time; the first logged step creates every system's file, later ones append.
            mode = "a" if self._trajectories else "w"
            self._trajectories = True
            for s in range(self.systems):
                atoms = self.templates[s].copy()
                atoms.set_positions(positions[s])
                atoms.set_momenta(momenta[s])
                atoms.info["system_writes"] = self._writes[s]  # a write into the slot starts another system
                atoms.calc = SinglePointCalculator(atoms, energy=float(potential[s]), forces=forces[s])
                with Trajectory(self._trajectory_name.format(system=s), mode) as writer:
                    writer.write(atoms)
        if self._log is not None:
            time_ps = self.nsteps * self.timestep / 1000.0
            temperatures = self.temperatures().numpy()
            for s in range(self.systems):
                self._log.write(f"{time_ps:12.4f} {s:8d} {potential[s] + kinetic[s]:16.6f} {potential[s]:16.6f} "
                                f"{kinetic[s]:14.6f} {temperatures[s]:10.2f}\n")
            self._log.flush()
