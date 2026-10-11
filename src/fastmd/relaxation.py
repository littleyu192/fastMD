"""ASE geometry relaxation with the existing accelerated MLIP calculators.

The optimizer and cell filter run in ASE. CUDA Graphs accelerate supported model
inference; this module does not claim to capture a complete optimization step.
"""
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from time import perf_counter

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE

from .calculator import FastMDCalculator
from .config import CUDAGraphConfig


@dataclass(frozen=True)
class RelaxationConfig:
    """ASE defaults for fixed-cell or simultaneous position/cell relaxation.

    ``fmax`` applies to the forces returned by the optimizer's target, including
    the scaled cell forces for ``relax_cell=True``. It is not a stress threshold.
    ``steps=0`` evaluates the initial structure without taking an optimizer step.
    ``optimizer_kwargs`` and ``filter_kwargs`` are forwarded to ASE. A restart
    file is deliberately unsupported so independent structures never share an
    optimizer's velocities or Hessian. Graph capture is included in elapsed time.
    """

    relax_cell: bool = True
    optimizer: str = "FIRE"
    fmax: float = 0.05
    steps: int = 500
    optimizer_kwargs: dict = field(default_factory=dict)
    filter_kwargs: dict = field(default_factory=dict)
    logfile: object = None
    trajectory: object = None

    def __post_init__(self):
        if type(self.relax_cell) is not bool:
            raise ValueError("relax_cell must be a boolean")
        if self.optimizer not in {"FIRE", "BFGS"}:
            raise ValueError("optimizer must be 'FIRE' or 'BFGS'")
        if isinstance(self.fmax, bool) or not math.isfinite(self.fmax) or self.fmax <= 0:
            raise ValueError("fmax must be finite and positive")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for name in ("optimizer_kwargs", "filter_kwargs"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a dictionary")
            object.__setattr__(self, name, deepcopy(value))
        reserved = {"atoms", "restart", "logfile", "trajectory", "append_trajectory"}
        if reserved & self.optimizer_kwargs.keys():
            raise ValueError(f"Reserved optimizer_kwargs: {sorted(reserved & self.optimizer_kwargs.keys())}")
        if "atoms" in self.filter_kwargs:
            raise ValueError("filter_kwargs must not override atoms")
        if not self.relax_cell and self.filter_kwargs:
            raise ValueError("filter_kwargs require relax_cell=True")


@dataclass
class RelaxationResult:
    """Owned terminal geometry and arrays, independent of subsequent predictions.

    ``forces`` and ``max_force`` include ASE positional constraints. ``stress``
    is the potential stress in ASE Voigt order/eV per cubic Angstrom, or None for
    fixed-cell relaxation. ``max_filter_force`` determines convergence and also
    includes cell forces when applicable. ``stats`` snapshots calculator counters:
    backend call counts are cumulative; graph-runner cache counters may reset
    when the caller reuses a calculator for a different structure.
    """

    atoms: Atoms
    initial_energy: float
    final_energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    converged: bool
    steps: int
    elapsed_seconds: float
    max_force: float
    max_filter_force: float
    stats: dict


def make_relaxation_calculator(model, checkpoint=None, device="auto", cuda_graph="auto",
                               relax_cell=True, model_kwargs=None):
    """Construct a calculator with a stable property task for relaxation.

    MatRIS and MACE retain fused CUDA Graph energy/force/stress inference during
    cell relaxation. CHGNet stress currently needs eager inference: auto selects
    it explicitly, while a strict Graph request raises. ALIGNN supports fixed
    cells only. Fixed-cell workflows retain each model's usual graph behavior.
    """
    if isinstance(model, str):
        model = model.lower()
    options = dict(model_kwargs or {})
    graph = cuda_graph if isinstance(cuda_graph, CUDAGraphConfig) else CUDAGraphConfig(
        enabled="auto" if cuda_graph is None else cuda_graph)
    if relax_cell:
        if model == "alignn":
            raise PropertyNotImplementedError(
                "ALIGNN has no stress support; use relax_cell=False for fixed-cell relaxation")
        if model in {"matris", "mace", "chgnet"}:
            if options.get("compute_stress", True) is not True:
                raise ValueError("Cell relaxation requires compute_stress=True")
            options["compute_stress"] = True
        if model == "chgnet":
            if graph.enabled is True:
                raise ValueError("CHGNet cell relaxation requires eager stress; use cuda_graph='auto' or False")
            graph = replace(graph, enabled=False)
    calculator = FastMDCalculator(model, checkpoint=checkpoint, device=device,
                                  cuda_graph=graph, model_kwargs=options)
    if graph.enabled is True:
        properties = frozenset({"energy", "forces", "stress"} if relax_cell else {"energy", "forces"})
        reason = calculator.backend.graph_unavailable_reason(properties)
        if reason:
            raise RuntimeError(f"CUDA Graph was required but is unavailable: {reason}")
    return calculator


def _finite(name, value):
    if not np.isfinite(value).all():
        raise ValueError(f"Relaxation produced non-finite {name}")


def _maximum_force(forces):
    return float(np.linalg.norm(forces, axis=1).max())


def relax(atoms, calculator, config=None):
    """Relax a copy of ``atoms`` with a fresh ASE optimizer.

    The input geometry, constraints, momenta and calculator attachment are not
    changed. The supplied calculator is reusable but must not be used concurrently.
    The returned atoms carry a SinglePointCalculator so later relaxations cannot
    invalidate their final results. Model loading is outside ``elapsed_seconds``;
    initial evaluation, capture, every optimizer step and final evaluation are in it.
    """
    config = RelaxationConfig() if config is None else config
    if not isinstance(config, RelaxationConfig):
        raise TypeError("config must be a RelaxationConfig")
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise ValueError("relax requires a nonempty ASE Atoms object")
    _finite("positions", atoms.positions)
    _finite("cell", atoms.cell.array)
    if config.relax_cell:
        if "stress" not in calculator.implemented_properties:
            raise PropertyNotImplementedError(
                "Cell relaxation requires a calculator providing stress; use relax_cell=False")
        if not atoms.pbc.all() or atoms.cell.volume <= 0:
            raise ValueError("Cell relaxation requires a nonzero fully periodic cell")
    working = atoms.copy()
    working.calc = calculator
    target = FrechetCellFilter(working, **deepcopy(config.filter_kwargs)) if config.relax_cell else working

    def check_state():
        _finite("positions", working.positions)
        _finite("cell", working.cell.array)
        if config.relax_cell:
            if working.cell.volume <= 0:
                raise ValueError("Relaxation produced a zero-volume cell")
            _finite("stress", working.get_stress())
        _finite("forces", working.get_forces())
        _finite("energy", working.get_potential_energy())

    started = perf_counter()
    # Stress first ensures that even generic ASE calculators can return E/F/S in
    # one evaluation; the accelerated MatRIS/MACE factories stabilize that task.
    check_state()
    initial_energy = float(working.get_potential_energy())
    optimizer_class = {"FIRE": FIRE, "BFGS": BFGS}[config.optimizer]
    with optimizer_class(target, logfile=config.logfile, trajectory=config.trajectory,
                         **deepcopy(config.optimizer_kwargs)) as optimizer:
        optimizer.attach(check_state, interval=1)
        optimizer.run(fmax=config.fmax, steps=config.steps)
        steps = optimizer.nsteps
    check_state()
    filter_forces = np.array(target.get_forces(), dtype=float, copy=True)
    _finite("filter forces", filter_forces)
    forces = np.array(working.get_forces(), dtype=float, copy=True)
    raw_forces = np.array(working.get_forces(apply_constraint=False), dtype=float, copy=True)
    energy = float(working.get_potential_energy())
    stress = (np.array(working.get_stress(apply_constraint=False), dtype=float, copy=True)
              if config.relax_cell else None)
    _finite("unconstrained forces", raw_forces)
    if stress is not None:
        _finite("unconstrained stress", stress)
    max_filter_force = _maximum_force(filter_forces)
    stats = deepcopy(calculator.stats()) if hasattr(calculator, "stats") else {}
    elapsed = perf_counter() - started
    output = working.copy()
    results = dict(energy=energy, free_energy=energy, forces=raw_forces)
    if stress is not None:
        results["stress"] = stress.copy()
    output.calc = SinglePointCalculator(output, **results)
    return RelaxationResult(
        atoms=output, initial_energy=initial_energy, final_energy=energy,
        forces=forces, stress=stress, converged=max_filter_force < config.fmax,
        steps=int(steps), elapsed_seconds=elapsed, max_force=_maximum_force(forces),
        max_filter_force=max_filter_force, stats=stats,
    )
