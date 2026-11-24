"""Structure relaxation of different systems in the slots of a whole-step CUDA graph: ASE's FIRE or
GoodOldQuasiNewton per system, with a fixed cell or a variable one.

The relaxation counterpart of the slot unions of :mod:`fastmd.execution.system_batch`, whose systems advance
by MD: here each slot's system takes one optimizer step per whole-step step, its optimizer state and its
convergence on the device, so a captured window of k steps relaxes every system by up to k steps without the
host:

* :class:`SystemFIRE`: ASE's ``FIRE`` (``ase.optimize.FIRE``, its defaults: :class:`FIREParameters`) per
  system: its own velocity, time step, mixing parameter and count of downhill steps, each dot product and the
  step-length limit over its own degrees of freedom; it converges when its largest force row is below ``fmax``
  (strictly, as ASE's optimizers), stops at a step limit, or fails on a non-finite state. A system stops in the
  step it finishes (its step count is ASE's ``nsteps``) and keeps the forces, energy and stress of its last
  state. With ``cell_filter`` (:class:`CellFilter`), ASE's ``FrechetCellFilter`` or ``ExpCellFilter`` per
  system: its degrees of freedom are the positions in its original cell and its cell's deformation (Frechet: the
  logarithm of the deformation scaled by the atom count; Exp: the deformation gradient itself); its forces are
  the filter's, from its stress.
* :class:`SystemGOQN`: ASE's ``GoodOldQuasiNewton`` (``ase.optimize.oldqn``, its defaults:
  :class:`GOQNParameters`) per system, with the same filters, convergence and records: each system's BFGS
  Hessian, trust radius and step rejection. Its step needs each Hessian's eigendecomposition, which a CUDA graph
  cannot hold: :meth:`SystemGOQN.propose` computes every relaxing system's next step eagerly on the device
  between windows of one step, and the captured step moves the systems there.
* :class:`CellRelaxationBuilder`: the slot builder whose cells change in the graph
  (:meth:`~CellRelaxationBuilder.set_cells`); its skin test bounds the change of every pair distance since the
  candidate refresh by the atoms' moves (in fractional coordinates, times the refresh's cell) and each cell's
  deformation since then.
* :class:`SlotRelaxationRunner`: the slot runner over them; a window's record holds one phase per system, read
  with the window's flags in one copy.

``fastmd.interfaces.multi_system_relax`` drives these parts, with a queue of structures through the slots.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..engine.entry_config import scoped_call
from ..graph.geometry import small_matmul
from .gpu_integrator import StatefulIntegrator
from .gpu_nhc import _HAS_TRITON
from .system_batch import (
    SlotLayout,
    SlotUnionGraphBuilder,
    SlotWholeStepRunner,
    segment_sums,
    system_energy_derivative,
    system_volumes,
)
from .whole_step_cuda_graph import WholeStepGraphRunner

if _HAS_TRITON:
    from ..graph.fixed_capacity_builder import allocate_cell_list, estimate_batch_cell_list_sizes

# The phase of each slot's system: one float64 per system on the device, in the window record.
IDLE, PENDING, ACTIVE, CONVERGED, STEP_LIMIT, FAILED = 0, 1, 2, 3, 4, 5
FINISHED = (CONVERGED, STEP_LIMIT, FAILED)
PHASES = {IDLE: "idle", PENDING: "pending", ACTIVE: "active", CONVERGED: "converged", STEP_LIMIT: "max_steps",
          FAILED: "failed"}

# The matrix exponential of the cell filter inside the graph: a degree-12 Taylor polynomial of A / 2**8 (Horner),
# squared 8 times, in float64; fixed shapes and no host decision (torch.linalg.matrix_exp chooses its scaling on
# the host). The polynomial and the squarings carry exp(A) - 1, so a small A keeps its relative accuracy instead of
# rounding at the identity's scale. Up to a norm of 10 the truncation stays below 1e-25 relative, under the
# squarings' float64 rounding; a relaxation's log-deformations and the normalised gradient blocks are far smaller.
_EXPM_SQUARINGS = 8
_EXPM_DEGREE = 12


def expm_small(matrices: Tensor) -> Tensor:
    """``exp`` of each ``[..., d, d]`` float64 matrix (see ``_EXPM_SQUARINGS``): device ops, capture-safe."""
    scaled = matrices / float(1 << _EXPM_SQUARINGS)
    eye = torch.eye(matrices.shape[-1], dtype=matrices.dtype, device=matrices.device).expand(matrices.shape)
    # Horner for exp(X) - 1 = X (1 + X/2 (1 + X/3 (...))), the bracket from the inside out, then times X
    bracket = eye + scaled / _EXPM_DEGREE
    for k in range(_EXPM_DEGREE - 1, 1, -1):
        bracket = eye + torch.matmul(scaled, bracket) / k
    deviation = torch.matmul(scaled, bracket)
    for _ in range(_EXPM_SQUARINGS):  # (1 + E)^2 - 1 = 2 E + E E
        deviation = 2.0 * deviation + torch.matmul(deviation, deviation)
    return eye + deviation


def expm_frechet_adjoint(log_deformation: Tensor, gradient: Tensor) -> Tensor:
    """``[S, 3, 3]``: ``L(Y^T, G)``, the Fréchet derivative of ``expm`` at ``Y^T`` in the direction ``G``, which is
    the adjoint of ``L(Y, .)`` in the Frobenius product: ASE's ``FrechetCellFilter`` cell force (before its
    division by ``exp_cell_factor``), whose loop over the nine directions ``E`` takes ``<L(Y, E), G>``. The top
    right block of ``expm([[Y^T, G], [0, Y^T]])``, with ``G`` scaled by its largest entry (``L`` is linear in
    ``G``)."""
    scale = gradient.abs().amax(dim=(1, 2)).clamp_min(1e-300)[:, None, None]
    top = log_deformation.transpose(1, 2)
    block = torch.cat((torch.cat((top, gradient / scale), dim=2),
                       torch.cat((torch.zeros_like(top), top), dim=2)), dim=1)
    return expm_small(block)[:, :3, 3:] * scale


def inverse33(matrices: Tensor) -> Tensor:
    """The inverse of each ``[S, 3, 3]`` matrix from its rows' cross products and determinant (elementwise: no
    solver, no host synchronization)."""
    m0, m1, m2 = matrices[:, 0], matrices[:, 1], matrices[:, 2]
    columns = torch.stack((torch.linalg.cross(m1, m2, dim=1), torch.linalg.cross(m2, m0, dim=1),
                           torch.linalg.cross(m0, m1, dim=1)), dim=2)
    return columns / (m0 * columns[:, :, 0]).sum(dim=1)[:, None, None]


# The matrix logarithm of the cell filters inside the graph: log M = 2 artanh(Z), Z = (M - 1)(M + 1)^-1, a polynomial
# of _LOGM_TERMS terms in Z^2 (Horner), float64; fixed shapes and no host decision (scipy's logm, which ASE's filters
# call, takes a Schur decomposition on the host). The truncation is below |Z|^(2 _LOGM_TERMS + 1): under float64
# rounding for |Z| < 0.3 (a cell's principal stretches between 0.54 and 1.86) and 1e-14 relative at |Z| = 0.5
# (stretches from 1/3 to 3), far beyond the deformations of a relaxation.
_LOGM_TERMS = 20


def logm_near_identity(matrices: Tensor) -> Tensor:
    """``log`` of each ``[S, 3, 3]`` float64 matrix whose eigenvalues have positive real parts (see
    ``_LOGM_TERMS``): device ops, capture-safe."""
    eye = torch.eye(3, dtype=matrices.dtype, device=matrices.device).expand(matrices.shape)
    z = torch.matmul(matrices - eye, inverse33(matrices + eye))
    z2 = torch.matmul(z, z)
    # artanh(Z) = Z (1 + Z^2/3 (1 + ...)) as Z sum_k Z^2k / (2k + 1), the bracket from the highest term down
    bracket = eye / (2 * _LOGM_TERMS - 1)
    for k in range(_LOGM_TERMS - 2, -1, -1):
        bracket = eye / (2 * k + 1) + torch.matmul(z2, bracket)
    return 2.0 * torch.matmul(z, bracket)


def exp_cell_force(deformation: Tensor, virial: Tensor) -> Tensor:
    """``[S, 3, 3]``: ASE's ``ExpCellFilter`` cell force (ASE 3.23 and later) at the deformation gradient ``F``
    (``deformation``) of the (pressure-shifted, hydrostatic, masked) ``virial`` ``V``: the exact derivative of the
    energy with respect to ``L = log F``, ``-L(L, -V exp(-L))`` (the top right block of ``expm([[L, B], [0, L]])``,
    ``exp(-L) = F^-1``) with its off-diagonal pairs averaged, unless it is ``np.isclose`` to the virial everywhere
    or within the cosine 0.8 of its direction: then the virial itself (ASE's "naive" force)."""
    log = logm_near_identity(deformation)
    exact = -expm_frechet_adjoint(log.transpose(1, 2), -torch.matmul(virial, inverse33(deformation)))
    exact = 0.5 * (exact + exact.transpose(1, 2))
    close = ((exact - virial).abs() <= 1e-8 + 1e-5 * virial.abs()).all(dim=2).all(dim=1)
    cosine = (exact * virial).sum(dim=(1, 2)) / torch.sqrt((exact * exact).sum(dim=(1, 2))
                                                           * (virial * virial).sum(dim=(1, 2)))
    return torch.where((close | (cosine > 0.8))[:, None, None], virial, exact)


CELL_FILTERS = ("frechet", "exp")


@dataclass(frozen=True)
class CellFilter:
    """An ASE cell filter of each system, with ASE's arguments:

    * ``kind``: ``"frechet"``, ASE's ``FrechetCellFilter`` (the cell's degrees of freedom the logarithm of its
      deformation gradient times ``exp_cell_factor``, default the system's atom count; its forces the exact
      derivative of the energy with respect to them, divided by the factor), or ``"exp"``, ASE's
      ``ExpCellFilter`` as ASE 3.23 and later define it (the deformation gradient itself, unscaled, as the cell's
      degrees of freedom; its force the virial, or the exact derivative with respect to its logarithm where the
      two point apart: :func:`exp_cell_force`).
    * ``scalar_pressure`` (eV/Å³): the enthalpy's ``p V`` (its virial ``-V (sigma + p)``, and ``p V`` in the
      filter energy).
    * ``mask``: six Voigt components (``xx, yy, zz, yz, xz, xy``) or a 3x3 array of booleans (or 0 and 1), true
      where the strain relaxes; None: every component. The masked virial components are zero, and each new
      deformation is masked before it deforms the cell (``set_positions`` of ASE 3.25 and later; 3.23 and 3.24 mask
      the virial only), so a fixed component stays at the identity's.
    * ``hydrostatic_strain``: the virial replaced by its trace's third on the diagonal.
    * ``constant_volume``: the trace's third taken off the cell force's diagonal.
    """

    kind: str = "frechet"
    scalar_pressure: float = 0.0
    mask: tuple | None = None
    hydrostatic_strain: bool = False
    constant_volume: bool = False
    exp_cell_factor: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in CELL_FILTERS:
            raise ValueError(f"CellFilter: kind must be one of {CELL_FILTERS}, got {self.kind!r}")
        pressure = self.scalar_pressure
        if isinstance(pressure, bool) or not isinstance(pressure, (int, float)) or not math.isfinite(pressure):
            raise ValueError("CellFilter: scalar_pressure must be a finite number")
        for name in ("hydrostatic_strain", "constant_volume"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"CellFilter: {name} must be a bool")
        factor = self.exp_cell_factor
        if factor is not None:
            if self.kind != "frechet":
                raise ValueError("CellFilter: exp_cell_factor is FrechetCellFilter's (ExpCellFilter's cell "
                                 "degrees of freedom are unscaled)")
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0 < factor < math.inf:
                raise ValueError("CellFilter: exp_cell_factor must be finite and positive")
        if self.mask is not None:
            import numpy as np

            mask = np.asarray(self.mask, dtype=np.float64)
            if mask.shape == (6,):  # Voigt (xx, yy, zz, yz, xz, xy): ASE's voigt_6_to_full_3x3_stress
                mask = np.array([[mask[0], mask[5], mask[4]], [mask[5], mask[1], mask[3]],
                                 [mask[4], mask[3], mask[2]]])
            elif mask.shape != (3, 3):
                raise ValueError("CellFilter: the mask's shape must be (6,) or (3, 3)")
            if not np.isin(mask, (0.0, 1.0)).all():  # ASE's booleans (its filters multiply by other values, twice)
                raise ValueError("CellFilter: the mask's components must be 0 or 1 (False or True), as ASE takes them")
            object.__setattr__(self, "mask", tuple(tuple(float(value) for value in row) for row in mask))

    @property
    def masked(self) -> bool:
        """Whether a mask leaves a component out (ASE's ``(mask != 1).any()``)."""
        return self.mask is not None and any(value != 1.0 for row in self.mask for value in row)

    def factor(self, atoms: int) -> float:
        """The cell degrees of freedom's scale of a system of ``atoms`` atoms (Frechet: ``exp_cell_factor``,
        default the atom count; Exp: 1)."""
        if self.kind == "exp":
            return 1.0
        return float(atoms) if self.exp_cell_factor is None else float(self.exp_cell_factor)

    @classmethod
    def of(cls, value) -> "CellFilter | None":
        """A filter from ``value``: None or False (a fixed cell), True or ``"frechet"`` (FrechetCellFilter's
        defaults), ``"exp"`` (ExpCellFilter's), or a :class:`CellFilter`."""
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, str):
            return cls(kind=value)
        if isinstance(value, cls):
            return value
        raise ValueError(f"cell_filter must be None, 'frechet', 'exp' or a CellFilter, got {value!r}")


def rows_times(rows: Tensor, matrices: Tensor) -> Tensor:
    """``rows[i] @ matrices[i]`` for ``(n, 3)`` rows and ``(n, 3, 3)`` matrices, row by row (no GEMM)."""
    return rows[:, 0:1] * matrices[:, 0] + rows[:, 1:2] * matrices[:, 1] + rows[:, 2:3] * matrices[:, 2]


def segment_max(values: Tensor, ceilings: Tensor, atoms: Tensor) -> Tensor:
    """``(S,)``: the largest of ``values`` (``(rows,)``) over each system's rows, the first ``atoms[s]`` of its
    slot of ``ceilings[s]`` rows (:func:`~fastmd.execution.system_batch.segment_sums`' segments)."""
    lengths = torch.stack((atoms, ceilings - atoms), dim=1).reshape(-1)
    return torch.segment_reduce(values, "max", lengths=lengths, unsafe=True)[::2]


@dataclass(frozen=True)
class FIREParameters:
    """ASE's ``FIRE`` parameters, its defaults: the initial time step ``dt``, the longest step of a system
    ``maxstep`` (the norm over all its degrees of freedom, with a cell filter its cell's too), ``dtmax``, the
    downhill steps ``Nmin`` before the time step grows, ``finc``, ``fdec``, the mixing parameter ``astart`` after
    an uphill step, ``fa`` and the initial mixing parameter ``a``."""

    dt: float = 0.1
    maxstep: float = 0.2
    dtmax: float = 1.0
    Nmin: int = 5
    finc: float = 1.1
    fdec: float = 0.5
    astart: float = 0.1
    fa: float = 0.99
    a: float = 0.1

    def __post_init__(self) -> None:
        for name in ("dt", "maxstep", "dtmax", "finc", "fdec", "astart", "fa", "a"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"FIREParameters: {name} must be finite and positive")
        if isinstance(self.Nmin, bool) or not isinstance(self.Nmin, int) or self.Nmin < 0:
            raise ValueError("FIREParameters: Nmin must be a nonnegative integer")


class SystemRecords:
    """Each system's states at the steps an ASE observer attached with ``interval`` sees (``Dynamics.call_observers``:
    every step that is a multiple of ``interval``, the start included; ``interval <= 0``: only step ``-interval``),
    up to the step limit, on the device: per record and row the positions and the model's forces, per record and
    system the step, the energy, the stress (eV/Å³, with ``stress``; else zero) and the cell (with a cell filter; else
    zero), float64. A system's ``k``-th record sits at index ``k`` (its step divided by ``interval``), written by the
    step that evaluates it (capture-safe: the index is the state's; a window replayed after a rollback writes the same
    records again); the systems that record nothing in a step write into a last, unread record. 48 bytes per row and
    160 per system per record. With ``stress``, also each system's stress at its last evaluation (``final_stress``:
    the state it stops in, recorded or not)."""

    def __init__(self, layout: SlotLayout, *, interval: int, max_steps: int, stress: bool, cell_filter: bool,
                 device) -> None:
        self.interval = int(interval)
        self.stress = bool(stress)
        self.cell_filter = bool(cell_filter)
        self.count = max_steps // self.interval + 1 if self.interval > 0 else 1
        self.rows = torch.zeros((self.count + 1, layout.rows, 6), dtype=torch.float64, device=device)
        self.systems = torch.zeros((self.count + 1, layout.count, 20), dtype=torch.float64, device=device)
        self.final_stress = torch.zeros((layout.count, 3, 3), dtype=torch.float64, device=device)
        self._row_index = torch.arange(layout.rows, device=device)
        self._system_index = torch.arange(layout.count, device=device)

    def recorded_steps(self, steps: int) -> list[int]:
        """The steps recorded of a system that took ``steps`` steps."""
        if self.interval > 0:
            return list(range(0, steps + 1, self.interval))
        return [-self.interval] if steps >= -self.interval else []

    def record(self, fire: "SystemFIRE", evaluated: Tensor, stress: Tensor | None) -> None:
        """The evaluated systems' current states whose step the observer sees (device ops only)."""
        st = fire.state
        if self.interval > 0:
            seen = evaluated & (torch.remainder(st.steps, self.interval) == 0)
            index = torch.div(st.steps, self.interval, rounding_mode="floor").long().clamp(0, self.count - 1)
        else:
            seen = evaluated & (st.steps == -self.interval)
            index = torch.zeros_like(st.steps, dtype=torch.int64)
        index = torch.where(seen, index, torch.full_like(index, self.count))
        self.rows.index_put_((index.index_select(0, fire.system_of_row), self._row_index),
                             torch.cat((fire.positions, st.f), dim=1))
        systems = st.steps.shape[0]
        zeros = st.f.new_zeros((systems, 9))
        stresses = stress.reshape(systems, 9).to(torch.float64) if self.stress and stress is not None else zeros
        cells = fire.cells.reshape(systems, 9) if self.cell_filter else zeros
        self.systems.index_put_((index, self._system_index),
                                torch.cat((st.steps[:, None], st.energy[:, None], stresses, cells), dim=1))
        if self.stress and stress is not None:
            self.final_stress.copy_(torch.where(evaluated[:, None, None], stress.to(torch.float64), self.final_stress))

    def read(self, slot: int, start: int, atoms: int, steps: int) -> dict:
        """The records of the system in ``slot`` (rows ``start``, ``atoms`` of them) that took ``steps`` steps: the
        recorded ``steps``, ``energies`` ``(k,)``, ``positions`` and ``forces`` ``(k, atoms, 3)``, ``stresses`` and
        ``cells`` ``(k, 3, 3)`` (None where not recorded), and ``final_stress`` ``(3, 3)`` (None without stress),
        float64 on the host."""
        recorded = self.recorded_steps(steps)
        k = len(recorded)
        rows = self.rows[:k, start:start + atoms].cpu().numpy()
        systems = self.systems[:k, slot].cpu().numpy()
        if not (systems[:, 0] == recorded).all():
            raise RuntimeError(f"slot {slot}: records of steps {systems[:, 0].tolist()}, expected {recorded}")
        return {"steps": recorded, "energies": systems[:, 1].copy(), "positions": rows[:, :, :3].copy(),
                "forces": rows[:, :, 3:].copy(),
                "stresses": systems[:, 2:11].reshape(k, 3, 3).copy() if self.stress else None,
                "cells": systems[:, 11:20].reshape(k, 3, 3).copy() if self.cell_filter else None,
                "final_stress": self.final_stress[slot].cpu().numpy().copy() if self.stress else None}

    def take(self, source: "SystemRecords", destination_rows: Tensor, source_rows: Tensor, systems: Tensor) -> None:
        """The records of ``source``'s rows and systems into these (in their order): systems that move on."""
        self.rows.zero_().index_copy_(1, destination_rows, source.rows.index_select(1, source_rows))
        self.systems.copy_(source.systems.index_select(1, systems))
        self.final_stress.copy_(source.final_stress.index_select(0, systems))


class SystemFIREState:
    """Views into one flat float64 buffer: per row ``x`` (the optimizer's positions: Cartesian, or with a cell
    filter in the system's original cell), ``v`` (velocity), ``g`` (the optimizer's forces: ASE's negative
    gradient) and ``f`` (the model's forces), ``(rows, 3)``; per system ``dt``, ``a``, ``npos`` (downhill steps),
    ``fresh`` (no step yet), ``phase``, ``steps``, ``fmax``, ``energy`` and ``factor`` (the scale of the cell's
    degrees of freedom: :meth:`CellFilter.factor`), ``(S,)``; with a cell filter per system ``cell_dof`` (the
    cell's degrees of freedom over ``factor``: Frechet the logarithm of the deformation gradient, Exp the
    deformation gradient), ``cell_velocity``, ``cell_gradient``, ``stress`` (eV/Å³) and ``original_cells``,
    ``(S, 3, 3)``; and the ``extra`` fields of an optimizer that keeps more (``(rows, 3)``, ``(S,)`` and
    ``(S, 3, 3)`` names)."""

    ROWS = ("x", "v", "g", "f")
    SYSTEMS = ("dt", "a", "npos", "fresh", "phase", "steps", "fmax", "energy", "factor")
    CELLS = ("cell_dof", "cell_velocity", "cell_gradient", "stress", "original_cells")

    def __init__(self, rows: int, systems: int, cell_filter: bool, device, extra: tuple = ((), (), ())) -> None:
        self.rows_fields = self.ROWS + tuple(extra[0])
        self.system_fields = self.SYSTEMS + tuple(extra[1])
        self.cell_fields = (self.CELLS + tuple(extra[2])) if cell_filter else ()
        shapes = ([(name, (rows, 3)) for name in self.rows_fields]
                  + [(name, (systems,)) for name in self.system_fields]
                  + [(name, (systems, 3, 3)) for name in self.cell_fields])
        self.flat = torch.zeros(sum(math.prod(shape) for _, shape in shapes), dtype=torch.float64, device=device)
        offset = 0
        for name, shape in shapes:
            size = math.prod(shape)
            setattr(self, name, self.flat[offset: offset + size].view(shape))
            offset += size


class SystemFIRE(StatefulIntegrator):
    """ASE's ``FIRE`` for the systems in the slots of :class:`~fastmd.execution.system_batch.SlotLayout` (see
    the module docstring), one optimizer step per step half pair:

        positions = fire.step_pre_force()        # each active system's FIRE step from its cached forces
        forces, energies[, stresses] = the union's evaluation at positions (in fire.cells)
        fire.step_post_force(forces, energy=energies[, stress=stresses])   # forces, convergence, phase

    A written system (:meth:`write_system`) is pending: the next step evaluates it without moving it (ASE's
    forces at the start), checks its convergence there (0 steps) and makes it active. ``positions`` are the
    systems' Cartesian positions ``(rows, 3)`` float64 (a vacant row: the origin, never moved); ``forces`` the
    model's at them, ``potential_energy`` ``(S,)``; ``cells`` ``(S, 3, 3)`` float64 (a cell filter). The step
    halves, :meth:`backup` and :meth:`restore` are capture-safe; :meth:`sync` recomputes the positions and cells
    from the state (after :meth:`restore` or a write). No restart files. ``cell_filter``: :meth:`CellFilter.of`'s
    values (``cell_filter`` is then whether there is one, ``filter`` the :class:`CellFilter`).
    """

    ensemble = "fire"
    comm = None
    # The state's fields besides SystemFIREState's: (rows, systems, cells) names (an optimizer that keeps more).
    EXTRA_STATE: tuple = ((), (), ())

    def __init__(self, layout: SlotLayout, *, fmax: float, max_steps: int, parameters: FIREParameters | None = None,
                 cell_filter: "bool | str | CellFilter | None" = False, device="cuda",
                 record_interval: int | None = None, record_stress: bool = False) -> None:
        if not math.isfinite(fmax) or fmax <= 0:
            raise ValueError("fmax must be finite and positive")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        self.device = torch.device(device)
        self.layout = layout
        self.systems = layout.count
        self.parameters = self.default_parameters() if parameters is None else parameters
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.filter = CellFilter.of(cell_filter)
        self.cell_filter = self.filter is not None
        self.generation = 0
        self.state = self.new_state()
        self._backup = self.new_state()
        self.system_of_row = layout.system_of_row(self.device).long()
        self.ceilings = torch.tensor(layout.ceilings, dtype=torch.int64, device=self.device)
        self.atoms = torch.ones(self.systems, dtype=torch.int64, device=self.device)
        self.occupied = torch.zeros((layout.rows, 1), dtype=torch.bool, device=self.device)
        self.host_atoms = [0] * self.systems
        # Derived from the state (sync() and every pre-force half): the positions the model sees and, with a cell
        # filter, each system's deformation, its inverse and its cell.
        self._positions = torch.zeros((layout.rows, 3), dtype=torch.float64, device=self.device)
        if self.cell_filter:
            eye = torch.eye(3, dtype=torch.float64, device=self.device)
            self._eye = eye
            self._deformation = eye.expand(self.systems, 3, 3).clone()
            self._inverse_deformation = eye.expand(self.systems, 3, 3).clone()
            self._cells = eye.expand(self.systems, 3, 3).clone()
            # ASE's set_positions masks the deformation's transpose: the cell degrees of freedom keep mask^T
            self._mask_t = (torch.tensor(self.filter.mask, dtype=torch.float64, device=self.device).t().contiguous()
                            if self.filter.masked else None)
            self._mask = None if self._mask_t is None else self._mask_t.t().contiguous()
        # ``record_interval``: each system's states as an ASE observer of that interval sees them (SystemRecords;
        # ``record_stress``: with their stress, which a fixed cell then evaluates every step).
        self.records = (None if record_interval is None else
                        SystemRecords(layout, interval=record_interval, max_steps=self.max_steps,
                                      stress=record_stress or self.cell_filter, cell_filter=self.cell_filter,
                                      device=self.device))

    @staticmethod
    def default_parameters() -> FIREParameters:
        return FIREParameters()

    # ------------------------------------------------------------------ systems
    @torch.no_grad()
    def write_system(self, slot: int, *, positions, cell=None) -> None:
        """A system of ``n`` atoms at ``positions`` (``(n, 3)`` Å; with a cell filter in ``cell``, ``(3, 3)``,
        its original cell) in ``slot``, pending, with a fresh optimizer (ASE's initial time step and mixing
        parameter, no velocity, no step). Eager, between windows; :meth:`sync` before the next window."""
        positions = torch.as_tensor(positions, dtype=torch.float64).reshape(-1, 3)
        n, ceiling = int(positions.shape[0]), self.layout.ceilings[slot]
        if not 1 <= n <= ceiling:
            raise ValueError(f"slot {slot} holds 1 to {ceiling} atoms, got {n}")
        if self.cell_filter and cell is None:
            raise ValueError("a cell filter needs each system's cell")
        st, start = self.state, self.layout.starts[slot]
        for name in st.rows_fields:
            getattr(st, name)[start:start + ceiling].zero_()
        st.x[start:start + n].copy_(positions)
        values = {name: 0.0 for name in st.system_fields}
        values.update(self._fresh_values(n), phase=float(PENDING), fmax=math.inf,
                      factor=self.filter.factor(n) if self.cell_filter else float(n))
        for name, value in values.items():
            getattr(st, name)[slot] = value
        if self.cell_filter:
            for name in st.cell_fields:
                getattr(st, name)[slot].zero_()
            if self.filter.kind == "exp":  # the deformation gradient of the original cell
                st.cell_dof[slot].copy_(self._eye)
            st.original_cells[slot].copy_(torch.as_tensor(cell, dtype=torch.float64).reshape(3, 3))
        self.atoms[slot] = n
        self.occupied[start:start + ceiling] = False
        self.occupied[start:start + n] = True
        self.host_atoms[slot] = n

    def _fresh_values(self, atoms: int) -> dict:
        """A written system's optimizer values (per-system state fields): ASE's FIRE at its start."""
        p = self.parameters
        return {"dt": p.dt, "a": p.a, "npos": 0.0, "fresh": 1.0, "steps": 0.0, "energy": 0.0}

    def release(self, slot: int) -> None:
        """Slot ``slot``'s finished system stays where it is, frozen, and is no longer reported (idle)."""
        self.state.phase[slot] = float(IDLE)

    @torch.no_grad()
    def take_systems(self, source: "SystemFIRE", slots: list[int]) -> None:
        """Slot ``k`` takes the system of ``source``'s slot ``slots[k]`` with its optimizer state as it is (its
        positions, velocity, forces, time step, mixing parameter, steps, phase and, with a cell filter, its cell
        variables): it continues where it was. Eager; :meth:`sync` follows."""
        if source.filter != self.filter or type(source) is not type(self) or len(slots) != self.systems:
            raise ValueError("take_systems needs one source slot per slot, the same optimizer and cell filter")
        destination_rows, source_rows = [], []
        for k, s in enumerate(slots):
            n = source.host_atoms[s]
            if n > self.layout.ceilings[k]:
                raise ValueError(f"slot {k} holds at most {self.layout.ceilings[k]} atoms, got {n}")
            destination_rows.append(torch.arange(self.layout.starts[k], self.layout.starts[k] + n))
            source_rows.append(torch.arange(source.layout.starts[s], source.layout.starts[s] + n))
            self.atoms[k] = n
            self.host_atoms[k] = n
            self.occupied[self.layout.starts[k]:self.layout.starts[k] + self.layout.ceilings[k]] = False
            self.occupied[self.layout.starts[k]:self.layout.starts[k] + n] = True
        destination = torch.cat(destination_rows).to(self.device)
        rows = torch.cat(source_rows).to(self.device)
        systems = torch.as_tensor(slots, dtype=torch.int64, device=self.device)
        for name in self.state.rows_fields:
            rows_of_source = getattr(source.state, name).index_select(0, rows)
            getattr(self.state, name).zero_().index_copy_(0, destination, rows_of_source)
        for name in self.state.system_fields + self.state.cell_fields:
            getattr(self.state, name).copy_(getattr(source.state, name).index_select(0, systems))
        if self.records is not None:
            self.records.take(source.records, destination, rows, systems)

    @torch.no_grad()
    def sync(self, state: SystemFIREState | None = None) -> None:
        """The positions (and cells) derived from the state: after a write, a restore or a snapshot load."""
        self._derive(self.state if state is None else state)

    def _deformation_of(self, cell_dof: Tensor) -> Tensor:
        """``[S, 3, 3]``: the deformation gradient of the cell degrees of freedom (Frechet: ``expm``; Exp: as is),
        masked as ASE's ``set_positions`` masks it (``1 + (F - 1) * mask^T``)."""
        deformation = expm_small(cell_dof) if self.filter.kind == "frechet" else cell_dof
        if self._mask_t is None:
            return deformation
        return self._eye + (deformation - self._eye) * self._mask_t

    def _set_cell_dof(self, st: SystemFIREState, moving: Tensor, cell_dof: Tensor) -> None:
        """The moving systems' new cell degrees of freedom, masked as ASE's next ``get_positions`` reads them back
        from the masked cell (Frechet: the logarithm of the masked deformation)."""
        if self._mask_t is not None:
            masked = self._deformation_of(cell_dof)
            cell_dof = logm_near_identity(masked) if self.filter.kind == "frechet" else masked
        st.cell_dof.copy_(torch.where(moving[:, None, None], cell_dof, st.cell_dof))

    def _derive(self, st: SystemFIREState) -> None:
        if not self.cell_filter:
            self._positions.copy_(st.x)
            return
        deformation = self._deformation_of(st.cell_dof)
        self._deformation.copy_(deformation)
        if self.filter.kind == "frechet" and self._mask_t is None:
            self._inverse_deformation.copy_(expm_small(-st.cell_dof))
        else:
            self._inverse_deformation.copy_(inverse33(deformation))
        transposed = deformation.transpose(1, 2)
        self._cells.copy_(torch.matmul(st.original_cells, transposed))
        # r = u F^T, row by row; vacant rows stay at the origin
        self._positions.copy_(rows_times(st.x, transposed.index_select(0, self.system_of_row)))

    # ------------------------------------------------------------------ the StatefulIntegrator state
    @property
    def n_owned(self) -> int:
        return self.layout.rows

    @property
    def positions(self) -> Tensor:
        return self._positions

    @property
    def momenta(self) -> Tensor:
        """The optimizer's velocities (unit masses), ``(rows, 3)``."""
        return self.state.v

    @property
    def forces(self) -> Tensor:
        return self.state.f

    @property
    def potential_energy(self) -> Tensor:
        return self.state.energy

    @property
    def cells(self) -> Tensor | None:
        """``(S, 3, 3)`` float64: each system's current cell (a cell filter; None otherwise)."""
        return self._cells if self.cell_filter else None

    @property
    def phase(self) -> Tensor:
        return self.state.phase

    @torch.no_grad()
    def set_forces(self, forces, energy=None, state: SystemFIREState | None = None) -> None:
        """The model's forces ``(rows, 3)`` (and energies ``(S,)``) at the current state, stored as they are; the
        optimizer's forces and the phases change only in :meth:`step_post_force` (capture-safe)."""
        st = self.state if state is None else state
        st.f.copy_(torch.as_tensor(forces, device=self.device).reshape(-1, 3))
        if energy is not None:
            st.energy.copy_(torch.as_tensor(energy, device=self.device).reshape(self.systems))

    # -------------------------------------------------------------- step halves
    def _sums(self, rows: list[Tensor], cells: list[Tensor]) -> Tensor:
        """``(S, k)`` float64: each system's sums of the ``k`` per-row values ``rows`` (``(rows,)``), plus the cell
        filter's per-system ``cells`` (``(S,)``)."""
        sums = segment_sums(torch.stack(rows, dim=1), self.ceilings, self.atoms)
        return sums + torch.stack(cells, dim=1) if self.cell_filter else sums

    def step_pre_force(self, state: SystemFIREState | None = None) -> Tensor:
        """Every active system's ASE ``FIRE.step`` from its cached forces (a pending, finished or idle system does
        not move); returns the float64 ``(rows, 3)`` positions to evaluate (and sets ``cells``)."""
        st = self.state if state is None else state
        p = self.parameters
        rows = self.system_of_row
        moving = st.phase == ACTIVE
        first = moving & (st.fresh != 0)
        later = moving & (st.fresh == 0)
        dot_rows = [(st.g * st.v).sum(dim=1), (st.g * st.g).sum(dim=1), (st.v * st.v).sum(dim=1)]
        dot_cells = []
        if self.cell_filter:
            gc, vc = st.cell_gradient, st.cell_velocity
            dot_cells = [(gc * vc).sum(dim=(1, 2)), (gc * gc).sum(dim=(1, 2)), (vc * vc).sum(dim=(1, 2))]
        vf, g2, v2 = self._sums(dot_rows, dot_cells).unbind(dim=1)
        downhill = later & (vf > 0.0)
        uphill = later & ~(vf > 0.0)
        reset = first | uphill
        grow = downhill & (st.npos > p.Nmin)
        # The time step first (ASE's step uses the new one), then the velocity: mixed with the forces downhill (ASE's
        # arithmetic order, the mixing parameter before its update), zero uphill and on the first step, unchanged
        # where the system does not move; then the move, its norm limited per system.
        dt = torch.where(grow, torch.clamp(st.dt * p.finc, max=p.dtmax), torch.where(uphill, st.dt * p.fdec, st.dt))
        a, root_g2, root_v2 = st.a.clone(), g2.sqrt(), v2.sqrt()
        per_row = torch.stack((a, root_g2, root_v2, dt), dim=1).index_select(0, rows)
        a_row, g2_row, v2_row, dt_row = per_row[:, 0:1], per_row[:, 1:2], per_row[:, 2:3], per_row[:, 3:4]
        down_row, reset_row = downhill[rows][:, None], reset[rows][:, None]
        move_row = moving[rows][:, None] & self.occupied
        v = torch.where(down_row, (1.0 - a_row) * st.v + a_row * st.g / g2_row * v2_row,
                        torch.where(reset_row, torch.zeros_like(st.v), st.v))
        v = torch.where(move_row, v + dt_row * st.g, v)
        dr = dt_row * v
        if self.cell_filter:
            a_cell, g2_cell, v2_cell, dt_cell = (value[:, None, None] for value in (a, root_g2, root_v2, dt))
            move_cell = moving[:, None, None]
            vc = torch.where(downhill[:, None, None], (1.0 - a_cell) * vc + a_cell * gc / g2_cell * v2_cell,
                             torch.where(reset[:, None, None], torch.zeros_like(vc), vc))
            vc = torch.where(move_cell, vc + dt_cell * gc, vc)
            dr_cell = dt_cell * vc
            norm = self._sums([(dr * dr).sum(dim=1)], [(dr_cell * dr_cell).sum(dim=(1, 2))])[:, 0].sqrt()
        else:
            norm = self._sums([(dr * dr).sum(dim=1)], [])[:, 0].sqrt()
        limited = norm > p.maxstep
        norm_row = norm[:, None].index_select(0, rows)
        dr = torch.where(limited[rows][:, None], p.maxstep * dr / norm_row, dr)
        st.v.copy_(v)
        st.x.copy_(torch.where(move_row, st.x + dr, st.x))
        if self.cell_filter:
            dr_cell = torch.where(limited[:, None, None], p.maxstep * dr_cell / norm[:, None, None], dr_cell)
            st.cell_velocity.copy_(vc)
            self._set_cell_dof(st, moving, st.cell_dof + dr_cell / st.factor[:, None, None])
        st.dt.copy_(dt)
        st.a.copy_(torch.where(grow, a * p.fa, torch.where(uphill, torch.full_like(a, p.astart), a)))
        st.npos.copy_(torch.where(downhill, st.npos + 1.0, torch.where(uphill, torch.zeros_like(st.npos), st.npos)))
        st.fresh.copy_(torch.where(moving, torch.zeros_like(st.fresh), st.fresh))
        st.steps.copy_(st.steps + moving.to(torch.float64))
        self._derive(st)
        return self._positions

    def _virial(self, sigma: Tensor) -> Tensor:
        """``[S, 3, 3]``: the cell filter's virial ``-V (sigma + p)`` of the current cells, hydrostatic and masked
        as ASE's filters make it (the order of ASE's ``get_forces``)."""
        flt = self.filter
        if flt.scalar_pressure:
            sigma = sigma + flt.scalar_pressure * self._eye
        virial = -system_volumes(self._cells)[:, None, None] * sigma
        if flt.hydrostatic_strain:
            trace = virial.diagonal(dim1=1, dim2=2).sum(dim=1)
            virial = (trace / 3.0)[:, None, None] * self._eye
        if self._mask is not None:
            virial = virial * self._mask
        return virial

    def _cell_force(self, st: SystemFIREState, sigma: Tensor) -> Tensor:
        """``[S, 3, 3]``: the cell filter's cell force before its division by ``factor``: Frechet ``L(Y^T,
        virial F^-T)`` (``Y`` the logarithm of the deformation ``F``), Exp :func:`exp_cell_force`; with
        ``constant_volume`` its trace's third off its diagonal."""
        virial = self._virial(sigma)
        if self.filter.kind == "frechet":
            force = expm_frechet_adjoint(st.cell_dof, torch.matmul(virial, self._inverse_deformation.transpose(1, 2)))
        else:
            force = exp_cell_force(self._deformation, virial)
        if self.filter.constant_volume:
            trace = force.diagonal(dim1=1, dim2=2).sum(dim=1)
            force = force - (trace / 3.0)[:, None, None] * self._eye
        return force

    def filter_energy(self, energy: Tensor) -> Tensor:
        """``(S,)``: the energy an ASE optimizer reads of each system (a cell filter's ``get_potential_energy``:
        ``E + p V`` of the current cell), float64."""
        energy = energy.to(torch.float64)
        if not self.cell_filter or not self.filter.scalar_pressure:
            return energy
        return energy + self.filter.scalar_pressure * system_volumes(self._cells)

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None, state: SystemFIREState | None = None,
                        stress: Tensor | None = None) -> None:
        """The model's forces ``(rows, 3)``, energies ``(S,)`` (and, with a cell filter, stresses ``(S, 3, 3)`` in
        eV/Å³) at the positions of :meth:`step_pre_force`: each evaluated (active or pending) system's optimizer
        forces, largest force row and phase (converged below ``fmax``, else the step limit, else active; failed
        where its state is not finite); finished and idle systems keep theirs."""
        st = self.state if state is None else state
        rows = self.system_of_row
        evaluated = (st.phase == ACTIVE) | (st.phase == PENDING)
        evaluated_row = evaluated[rows][:, None] & self.occupied
        model_forces = forces.reshape(-1, 3).to(torch.float64)
        st.f.copy_(torch.where(evaluated_row, model_forces, st.f))
        if energy is not None:
            st.energy.copy_(torch.where(evaluated, energy.reshape(-1).to(torch.float64), st.energy))
        if self.cell_filter:
            # The filter's get_forces: the atoms' f F, and the cell's force / factor from the virial
            gradient = rows_times(model_forces, self._deformation.index_select(0, rows))
            sigma = stress.to(torch.float64)
            evaluated_cell = evaluated[:, None, None]
            st.stress.copy_(torch.where(evaluated_cell, sigma, st.stress))
            st.cell_gradient.copy_(torch.where(evaluated_cell, self._cell_force(st, sigma) / st.factor[:, None, None],
                                               st.cell_gradient))
        else:
            gradient = model_forces
        st.g.copy_(torch.where(evaluated_row, gradient, st.g))
        norms = torch.linalg.vector_norm(st.g, dim=1)
        largest = segment_max(torch.where(torch.isfinite(norms), norms, torch.full_like(norms, math.inf)),
                              self.ceilings, self.atoms)
        if self.cell_filter:
            cell_norms = torch.linalg.vector_norm(st.cell_gradient, dim=2).amax(dim=1)
            largest = torch.maximum(largest, torch.where(torch.isfinite(cell_norms), cell_norms,
                                                         torch.full_like(cell_norms, math.inf)))
        finite = torch.isfinite(largest) & torch.isfinite(st.energy)
        phase = torch.where(~finite, FAILED, torch.where(largest < self.fmax, CONVERGED,
                                                         torch.where(st.steps >= self.max_steps, STEP_LIMIT, ACTIVE)))
        st.fmax.copy_(torch.where(evaluated, largest, st.fmax))
        st.phase.copy_(torch.where(evaluated, phase.to(torch.float64), st.phase))
        if self.records is not None:
            self.records.record(self, evaluated, stress)

    # ------------------------------------------------------------ transactions
    def backup(self, state: SystemFIREState | None = None) -> None:
        self._backup.flat.copy_((self.state if state is None else state).flat)

    def restore(self, state: SystemFIREState | None = None) -> None:
        (self.state if state is None else state).flat.copy_(self._backup.flat)

    def snapshot(self) -> tuple:
        return self.state.flat.clone(), self._backup.flat.clone()

    def load_snapshot(self, snapshot: tuple) -> None:
        state, backup = snapshot
        self.state.flat.copy_(state)
        self._backup.flat.copy_(backup)
        self.sync()

    def new_state(self) -> SystemFIREState:
        return SystemFIREState(self.layout.rows, self.layout.count, self.cell_filter, self.device, self.EXTRA_STATE)

    def propose(self) -> None:
        """The optimizer's eager part before a window (none for FIRE: its step is in the graph)."""

    def forget_history(self) -> None:
        """What the optimizer keeps across its systems beyond their state, as a new one's (none for FIRE)."""

    def step_bound(self) -> float:
        """The longest step of a system's degrees of freedom in the next window's steps (FIRE: ``maxstep``)."""
        return float(self.parameters.maxstep)

    def state_dict(self) -> dict:
        raise NotImplementedError("restart files of the batched relaxation are not implemented")

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError("restart files of the batched relaxation are not implemented")


@dataclass(frozen=True)
class GOQNParameters:
    """ASE's ``GoodOldQuasiNewton`` parameters, its defaults: the initial Hessian's ``diagonal`` (eV/Å², each
    degree of freedom's), the initial trust ``radius`` and its largest ``maxradius`` (None: ``0.05 sqrt(n) / 10``
    and ``0.5 sqrt(n)`` for a system of ``n`` degrees of freedom, three per atom and nine for a cell filter's
    cell), and ``forcemin`` (True: a step that raises the energy by more than 1e-4 eV is rejected, and the radius
    follows how well the quadratic model predicted the energy's change; False: 1e-2 eV, and the radius grows by
    1.5 after each accepted step). The Hessian update is ASE's default, BFGS; no transition-state search."""

    diagonal: float = 20.0
    radius: float | None = None
    maxradius: float | None = None
    forcemin: bool = True

    def __post_init__(self) -> None:
        for name in ("diagonal", "radius", "maxradius"):
            value = getattr(self, name)
            if value is None and name != "diagonal":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
                raise ValueError(f"GOQNParameters: {name} must be finite and positive")
        if not isinstance(self.forcemin, bool):
            raise ValueError("GOQNParameters: forcemin must be a bool")


def _scale_radius_energy(factor: Tensor) -> Tensor:
    """ASE's ``scale_radius_energy`` of each energy prediction ``factor`` (``|actual / predicted - 1|``), as its
    products: 1.4 per threshold the prediction beats, 1 / 1.4 per threshold it misses."""
    scale = torch.ones_like(factor)
    for threshold in (0.01, 0.05, 0.10, 0.40):
        scale = torch.where(factor < threshold, scale * 1.4, scale)
    for threshold in (0.5, 0.7, 1.0):
        scale = torch.where(factor > threshold, scale * (1.0 / 1.4), scale)
    return scale


# A step's radius grows at most by scale_radius_energy's 1.4**4 between steps.
_RADIUS_GROWTH = 1.4 ** 4
# find_lamda's search below the trust-region bound: the k-th step's length 0.1 * 1.25**((k - 1) // 100), as ASE's;
# the search finds its count by bisection over at most 2**_LOWER_SEARCH_BITS steps (beyond 1e33 below the bound).
_LOWER_SEARCH_BITS = 15
# The Hessians are kept per size class of the slots: the classes' bounds on a geometric grid of this ratio from
# _HESSIAN_CLASS_FLOOR (or the smallest slot's degrees of freedom, if more), each class's Hessians n x n for n its
# largest slot's (uncoupled diagonal rows beyond a system's own), decomposed in one batched eigh per class: above the
# floor, a slot's decomposition is at most this ratio larger than the slot (a structure takes its slot's size; ASE
# decomposes each system's own), whatever else the union holds. Few wide classes rather than many narrow ones: a batched float64 eigh takes about as
# long for four systems as for sixty-four at a few hundred degrees of freedom (a sequence of launches per column; on
# GB200 about 7 ms at n = 150 and 13 ms at n = 300), and each class's step adds its own launches.
_HESSIAN_CLASS_RATIO = 2.0
# The smallest class's bound: the systems up to this many degrees of freedom (32 atoms, fixed cell) share one class
# (a batch of 234 decomposes in about 2 ms at n = 96 on GB200).
_HESSIAN_CLASS_FLOOR = 96
# propose's temporaries: about eight n x n float64 matrices per system; a class's relaxing systems are decomposed in
# batches whose temporaries stay within this many bytes, and the eigenvectors held until their trust-region search
# within as many again (one search per group of batches within it).
_PROPOSE_BATCH_BYTES = 1 << 30


class SystemGOQN(SystemFIRE):
    """ASE's ``GoodOldQuasiNewton`` (``ase.optimize.oldqn``, ``hessianupdate="BFGS"``, :class:`GOQNParameters`)
    for the systems in the slots of :class:`~fastmd.execution.system_batch.SlotLayout`, with
    :class:`SystemFIRE`'s positions, cell filters, convergence, phases and records: each system's degrees of
    freedom ``x`` (its atoms' positions, then its cell's nine with a filter: ASE's ``optimizable.get_x()``), its
    gradient ``G`` (the negative optimizer forces) and energy (a filter's ``E + p V``) per step, its Hessian ``H``
    (``diagonal`` times the identity at the start, BFGS updates), trust ``radius`` and last accepted state.

    Each step of a system, as ASE's ``step``: a step that raised the energy by more than the threshold is rejected
    (back to the last accepted state, half the radius); else the Hessian is updated and the radius scaled by the
    energy prediction; then ``H = V diag(b) V^T`` (``torch.linalg.eigh``), the Newton step ``-H^-1 G`` where it is
    shorter than the radius, else ``-(H - lambda)^-1 G`` with ASE's ``find_lamda`` (the search below
    ``min(0, b_0) - 1e-12`` in steps of 0.1 growing by 1.25 every 100, then bisection to 1e-6), and the energy the
    quadratic model predicts. A system whose Hessian has no nonnegative eigenvalue fails (ASE's inertia count
    raises there).

    The decomposition cannot run in a CUDA graph: :meth:`propose` computes every relaxing system's next ``x``
    eagerly on the device (the decompositions batched over the systems of a size class, their Hessians padded to the
    class's largest slot with uncoupled ``diagonal`` rows: ``_HESSIAN_CLASS_RATIO``, ``_HESSIAN_CLASS_FLOOR``; the
    trust-region searches of every class in one batch), and the captured :meth:`step_pre_force` moves the systems
    there, so windows hold one step. A system moves only to a proposal made for it (a window without
    :meth:`propose`, a construction's warmup or probe, moves nothing). The Hessians and last accepted states are
    outside the window's backup: a rolled back window keeps its proposal and replays it.
    """

    ensemble = "goqn"
    EXTRA_STATE = (("px",), ("radius", "maxradius", "oldenergy", "estimate", "started", "proposed"), ("pcell",))
    _EPS = 1e-12

    def __init__(self, layout: SlotLayout, *, fmax: float, max_steps: int, parameters: GOQNParameters | None = None,
                 cell_filter: "bool | str | CellFilter | None" = False, device="cuda",
                 record_interval: int | None = None, record_stress: bool = False) -> None:
        if parameters is not None and not isinstance(parameters, GOQNParameters):
            raise ValueError("SystemGOQN: parameters must be GOQNParameters")
        super().__init__(layout, fmax=fmax, max_steps=max_steps, parameters=parameters, cell_filter=cell_filter,
                         device=device, record_interval=record_interval, record_stress=record_stress)
        sizes = [self._dof_count(ceiling) for ceiling in layout.ceilings]
        self.dofs = max(sizes)
        bounds, value = [], max(min(sizes), _HESSIAN_CLASS_FLOOR)
        while not bounds or bounds[-1] < self.dofs:  # the size classes' upper bounds
            bounds.append(value)
            value = max(value + 1, math.ceil(value * _HESSIAN_CLASS_RATIO))
        members: dict[int, list[int]] = {}
        for slot, size in enumerate(sizes):
            members.setdefault(next(bound for bound in bounds if bound >= size), []).append(slot)
        # each class's Hessians (its largest slot's size: one per class) and each slot's (size, row) among them
        self._identity = {}
        self.hessians: dict[int, Tensor] = {}
        self._slot_hessian: list[tuple[int, int]] = [(0, 0)] * self.systems
        for slots in members.values():
            n = max(sizes[slot] for slot in slots)
            self._identity[n] = self.parameters.diagonal * torch.eye(n, dtype=torch.float64, device=self.device)
            self.hessians[n] = self._identity[n].expand(len(slots), n, n).clone()
            for row, slot in enumerate(slots):
                self._slot_hessian[slot] = (n, row)
        shape = (self.systems, self.dofs)
        self.old_x = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.old_gradient = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._dof_index = None
        self._step_length = None  # the last proposal's longest step (step_bound)
        self.rejected_steps = 0  # ASE's "reject step"s, all systems
        self.trust_region_steps = 0  # steps on the radius (find_lamda), all systems

    @staticmethod
    def default_parameters() -> GOQNParameters:
        return GOQNParameters()

    def _dof_count(self, atoms: int) -> int:
        """ASE's ``optimizable.ndofs()``: three per atom, and a filter's three cell rows."""
        return 3 * atoms + (9 if self.cell_filter else 0)

    def _fresh_values(self, atoms: int) -> dict:
        n, p = self._dof_count(atoms), self.parameters
        radius = 0.05 * math.sqrt(n) / 10.0 if p.radius is None else float(p.radius)
        maxradius = 0.5 * math.sqrt(n) if p.maxradius is None else float(p.maxradius)
        return {"steps": 0.0, "energy": 0.0, "radius": max(min(radius, maxradius), 0.0001), "maxradius": maxradius}

    def hessian(self, slot: int) -> Tensor:
        """The Hessian of the system in ``slot``, ``n x n`` for ``n`` its size class's (its own degrees of freedom
        first, then uncoupled ``diagonal`` rows), a view."""
        n, row = self._slot_hessian[slot]
        return self.hessians[n][row]

    @torch.no_grad()
    def write_system(self, slot: int, *, positions, cell=None) -> None:
        super().write_system(slot, positions=positions, cell=cell)
        self.hessian(slot).copy_(self._identity[self._slot_hessian[slot][0]])
        self.old_x[slot].zero_()
        self.old_gradient[slot].zero_()
        self._dof_index = None

    @torch.no_grad()
    def take_systems(self, source: "SystemGOQN", slots: list[int]) -> None:
        """:meth:`SystemFIRE.take_systems`, with each system's Hessian and last accepted state."""
        super().take_systems(source, slots)
        for k, s in enumerate(slots):
            n = self._dof_count(source.host_atoms[s])
            self.hessian(k).copy_(self._identity[self._slot_hessian[k][0]])
            self.hessian(k)[:n, :n].copy_(source.hessian(s)[:n, :n])
            for name in ("old_x", "old_gradient"):
                getattr(self, name)[k].zero_()
                getattr(self, name)[k, :n].copy_(getattr(source, name)[s, :n])
        self._dof_index = None

    def _dof_layout(self) -> tuple[Tensor, Tensor]:
        """``(index, real)``, ``(S, dofs)``: each system's degrees of freedom in ASE's order (its atoms' rows, then
        its cell's) as indices into the flat ``[rows x 3 | S x 9 | 0]`` of a row field and a cell field, the
        padding at the last (zero) element; ``real`` where an index is a degree of freedom."""
        if self._dof_index is None:
            rows3, cells9 = 3 * self.layout.rows, 9 * self.systems if self.cell_filter else 0
            index = torch.full((self.systems, self.dofs), rows3 + cells9, dtype=torch.int64)
            for s, (start, atoms) in enumerate(zip(self.layout.starts, self.host_atoms)):
                index[s, :3 * atoms] = torch.arange(3 * start, 3 * (start + atoms))
                if self.cell_filter and atoms:
                    index[s, 3 * atoms:3 * atoms + 9] = torch.arange(rows3 + 9 * s, rows3 + 9 * s + 9)
            self._dof_index = index.to(self.device)
            self._dof_real = self._dof_index != rows3 + cells9
        return self._dof_index, self._dof_real

    def _flat(self, rows: Tensor, cells: Tensor | None) -> Tensor:
        parts = [rows.reshape(-1)] + ([cells.reshape(-1)] if cells is not None else []) + [rows.new_zeros(1)]
        return torch.cat(parts)

    def _f(self, lam: Tensor, gbar: Tensor, b: Tensor, radius: Tensor) -> Tensor:
        """ASE's ``f``: ``radius^2 - |Gbar / (b - lambda)|^2`` per system (``|b - lambda| < 1e-40`` as 1e-40)."""
        shifted = b - lam[:, None]
        shifted = torch.where(shifted.abs() < 1e-40, torch.full_like(shifted, 1e-40), shifted)
        quotient = gbar / shifted
        return radius * radius - (quotient * quotient).sum(dim=1)

    def _find_lamda(self, upper: Tensor, gbar: Tensor, b: Tensor, radius: Tensor) -> Tensor:
        """ASE's ``find_lamda`` per system: the first point of its search below ``upper`` where ``f >= 0`` (``f``
        grows as lambda falls below the lowest eigenvalue: the step count by bisection over the counts), then its
        bisection between there and ``upper`` until they are within 1e-6, the last midpoint."""
        f_upper = self._f(upper, gbar, b, radius)

        def below(count: Tensor) -> Tensor:  # the sum of the search's first ``count`` steps
            blocks, rest = torch.div(count, 100, rounding_mode="floor"), torch.remainder(count, 100)
            growth = torch.pow(1.25, blocks.to(torch.float64))
            return 0.1 * (100.0 * (growth - 1.0) / 0.25 + rest.to(torch.float64) * growth)

        low = torch.zeros_like(upper, dtype=torch.int64)  # f < 0 there (or the count is 0)
        high = torch.full_like(low, 1 << _LOWER_SEARCH_BITS)
        for _ in range(_LOWER_SEARCH_BITS):
            middle = torch.div(low + high, 2, rounding_mode="floor")
            met = self._f(upper - below(middle), gbar, b, radius) >= 0
            high, low = torch.where(met, middle, high), torch.where(met, low, middle)
        lower = torch.where(f_upper >= 0, upper, upper - below(high))
        width = float((upper - lower).max().item())
        iterations = 1 + max(0, math.ceil(math.log2(max(width, 1e-300) / 1e-6))) + 1
        lam, done = upper.clone(), torch.zeros_like(upper, dtype=torch.bool)
        for _ in range(iterations):
            middle = (upper + lower) / 2.0
            lam = torch.where(done, lam, middle)
            f_middle = self._f(middle, gbar, b, radius)
            down = ~done & (f_upper * f_middle < 0)
            up = ~done & ~(f_upper * f_middle < 0)
            lower = torch.where(down, middle, lower)
            upper = torch.where(up, middle, upper)
            f_upper = torch.where(up, f_middle, f_upper)
            done = done | ((upper - lower).abs() < 1e-6)
        return lam

    @torch.no_grad()
    def propose(self) -> None:
        """Every active system's next step (see the class docstring), eagerly: its rejection or Hessian update,
        radius and decomposition per size class in batches (``_PROPOSE_BATCH_BYTES``); ASE's trust-region search of
        every batch's systems at once (:meth:`_propose_steps`); the step's ``x`` into the proposal the next window's
        step moves it to. Waits for the device at each batch's decomposition (its check) and a few times per call."""
        st = self.state
        active = torch.nonzero(st.phase == ACTIVE).flatten().tolist()
        if not active:
            return
        classes: dict[int, list[int]] = {}
        for slot in active:
            classes.setdefault(self._slot_hessian[slot][0], []).append(slot)
        flat_x = self._flat(st.x, st.cell_dof * st.factor[:, None, None] if self.cell_filter else None)
        flat_g = self._flat(st.g, st.cell_gradient if self.cell_filter else None)
        energy = self.filter_energy(st.energy)
        cell_p = st.pcell * st.factor[:, None, None] if self.cell_filter else None
        proposals = self._flat(st.px, cell_p)
        lengths, rejected, group, held = [], [], [], 0
        for n, slots in classes.items():
            batch = max(1, _PROPOSE_BATCH_BYTES // (64 * n * n))
            for first in range(0, len(slots), batch):
                part = self._propose_decompose(slots[first:first + batch], n, flat_x, flat_g, energy)
                rejected.append(part.pop("rejected"))
                size = 8 * n * n * len(part["slots"])  # its eigenvectors, held until its step
                if group and held + size > _PROPOSE_BATCH_BYTES:
                    lengths += self._propose_steps(group, proposals)
                    group, held = [], 0
                group.append(part)
                held += size
        lengths += self._propose_steps(group, proposals)
        rows3 = 3 * self.layout.rows
        st.px.copy_(proposals[:rows3].view(-1, 3))
        if self.cell_filter:
            st.pcell.copy_(proposals[rows3:-1].view(-1, 3, 3) / st.factor[:, None, None])
        length, count = torch.stack([torch.stack(lengths).max(),
                                     torch.stack(rejected).sum().to(torch.float64)]).tolist()
        self._step_length = length
        self.rejected_steps += int(count)

    def _propose_decompose(self, slots: list[int], n: int, flat_x: Tensor, flat_g: Tensor, energies: Tensor) -> dict:
        """:meth:`propose`'s first part for the active systems in ``slots``, of the size class ``n`` (degrees of
        freedom and gradients from ``flat_x`` and ``flat_g``, energies from ``energies``): rejection, BFGS update,
        radius, decomposition and Newton step; their state written but for the step's. Returns what
        :meth:`_propose_steps` needs (``rejected``: their count of rejected steps, a device scalar)."""
        st = self.state
        host = torch.tensor([slots, [self._slot_hessian[slot][1] for slot in slots]], dtype=torch.int64)
        active, rows = host.to(self.device, non_blocking=True).unbind(0)  # staged at once, no wait for the device
        index, real = self._dof_layout()
        index, real = index.index_select(0, active)[:, :n], real.index_select(0, active)[:, :n]
        pick = lambda tensor: tensor.index_select(0, active)  # noqa: E731
        x, gradient, energy = flat_x[index], -flat_g[index], pick(energies)
        started, oldenergy, estimate = pick(st.started) != 0, pick(st.oldenergy), pick(st.estimate)
        radius, maxradius = pick(st.radius), pick(st.maxradius)
        hessian = self.hessians[n].index_select(0, rows)
        old_x, old_gradient = pick(self.old_x)[:, :n], pick(self.old_gradient)[:, :n]
        p = self.parameters
        # A step that raised the energy too much: back to the last accepted state, its gradient and energy
        reject = started & (energy - oldenergy > (1e-4 if p.forcemin else 1e-2))
        accept = started & ~reject
        x = torch.where(reject[:, None], old_x, x)
        gradient = torch.where(reject[:, None], old_gradient, gradient)
        energy = torch.where(reject, oldenergy, energy)
        radius = torch.where(reject, radius * 0.5, radius)
        # An accepted one: BFGS from the step and the gradient's change since the last accepted state
        dgrad, dpos = gradient - old_gradient, x - old_x
        dotg = (dgrad * dpos).sum(dim=1)
        tvec = torch.matmul(dpos[:, None, :], hessian)[:, 0, :]
        dott = (dpos * tvec).sum(dim=1)
        update = accept & (dott.abs() > self._EPS) & (dotg.abs() > self._EPS)
        rank_two = (dgrad[:, :, None] * dgrad[:, None, :] / dotg[:, None, None]
                    - tvec[:, :, None] * tvec[:, None, :] / dott[:, None, None])
        hessian = torch.where(update[:, None, None], hessian + rank_two, hessian)
        del rank_two
        if p.forcemin:  # the radius from the energy the quadratic model predicted
            factor = ((energy - oldenergy) / estimate - 1.0).abs()
            scaled = accept & (estimate.abs() > self._EPS)
            radius = torch.where(scaled, radius * _scale_radius_energy(factor), radius)
        else:
            radius = torch.where(accept, radius * 1.5, radius)
        radius = torch.where(started, torch.clamp(torch.minimum(radius, maxradius), min=0.0001), radius)
        settled = ~started | accept
        old_x = torch.where(settled[:, None], x, old_x)
        old_gradient = torch.where(settled[:, None], gradient, old_gradient)
        # The decomposition and the Newton step (inside the radius, else the trust region's: _propose_steps)
        b, vectors = torch.linalg.eigh(hessian)
        gbar = torch.matmul(gradient[:, None, :], vectors)[:, 0, :]
        newton = -gbar / b
        outside = ~(torch.sqrt((newton * newton).sum(dim=1)) < radius)
        # the state: these systems' rows of each field (the predicted energy after the step)
        st.radius.index_copy_(0, active, radius)
        st.oldenergy.index_copy_(0, active, energy)
        st.started.index_fill_(0, active, 1.0)
        st.proposed.index_fill_(0, active, 1.0)
        self.hessians[n].index_copy_(0, rows, hessian)
        self.old_x[active, :n] = old_x
        self.old_gradient[active, :n] = old_gradient
        return {"slots": slots, "n": n, "active": active, "index": index, "real": real, "x": x, "b": b,
                "vectors": vectors, "gbar": gbar, "radius": radius, "outside": outside, "rejected": reject.sum()}

    def _propose_steps(self, parts: list[dict], proposals: Tensor) -> list[Tensor]:
        """:meth:`propose`'s second part for decomposed batches (:meth:`_propose_decompose`): ASE's trust-region
        search (``find_lamda``) for every system of ``parts`` whose Newton step is outside its radius, in one batch
        (the parts' eigenvalues padded with ones and their ``gbar`` with zeros to the widest part's: terms of zero),
        then each part's step, predicted energy, inertia check and proposal (into ``proposals``). Returns each
        part's longest step (device scalars)."""
        if not parts:
            return []
        st = self.state
        outside = torch.cat([part["outside"] for part in parts])
        lam = torch.zeros(outside.shape, dtype=torch.float64, device=self.device)
        trust = torch.nonzero(outside).flatten()
        if trust.numel():  # the systems whose Newton step is not inside their radius
            width = max(part["n"] for part in parts)

            def widened(name: str, value: float) -> Tensor:
                return torch.cat([torch.nn.functional.pad(part[name], (0, width - part["n"]), value=value)
                                  for part in parts]).index_select(0, trust)

            b = widened("b", 1.0)
            radius = torch.cat([part["radius"] for part in parts]).index_select(0, trust)
            upper = torch.clamp(b[:, 0], max=0.0) - 1e-12
            lam.index_copy_(0, trust, self._find_lamda(upper, widened("gbar", 0.0), b, radius))
            self.trust_region_steps += int(trust.numel())
        lengths = []
        for part, part_lam in zip(parts, torch.split(lam, [len(part["slots"]) for part in parts])):
            b, gbar, real, active = part["b"], part["gbar"], part["real"], part["active"]
            steps = -gbar / (b - part_lam[:, None])
            step = torch.matmul(part["vectors"], steps[:, :, None])[:, :, 0]
            st.estimate.index_copy_(0, active, (steps * gbar + 0.5 * steps * b * steps).sum(dim=1))
            lengths.append(torch.sqrt((step * step * real).sum(dim=1)).max())
            # every real eigenvalue negative (the padding's are positive): ASE's inertia count runs past the end
            failed = (b < 0).sum(dim=1) >= real.sum(dim=1)
            st.phase.index_copy_(0, active, torch.where(failed, torch.full_like(part["radius"], float(FAILED)),
                                                        st.phase.index_select(0, active)))
            proposals.scatter_(0, part["index"].reshape(-1), (part["x"] + step).reshape(-1))
        return lengths

    def forget_history(self) -> None:
        """As a new optimizer's: no last proposal's step length (``step_bound``), no counted rejections or
        trust-region steps."""
        self._step_length = None
        self.rejected_steps = self.trust_region_steps = 0

    def step_pre_force(self, state: SystemFIREState | None = None) -> Tensor:
        """Every active system with a proposal moves to it (one step); the others stay. Returns the float64
        ``(rows, 3)`` positions to evaluate (and sets ``cells``)."""
        st = self.state if state is None else state
        moving = (st.phase == ACTIVE) & (st.proposed != 0)
        move_row = moving[self.system_of_row][:, None] & self.occupied
        st.x.copy_(torch.where(move_row, st.px, st.x))
        if self.cell_filter:
            self._set_cell_dof(st, moving, st.pcell)
        st.steps.copy_(st.steps + moving.to(torch.float64))
        st.proposed.copy_(torch.where(moving, torch.zeros_like(st.proposed), st.proposed))
        self._derive(st)
        return self._positions

    def step_bound(self) -> float:
        """The longest step of the next proposal, for the candidate refresh rule: a step is at most its radius,
        which grows at most 1.4**4 (and to ``maxradius``) per step, the largest over the relaxing systems; and
        after a proposal at most twice its longest step (a quasi-Newton step shrinks as it converges while its
        radius may grow: the radius alone would refresh at every window; a longer step is caught by the skin
        test, whose fallback window rebuilds the candidates)."""
        st = self.state
        relaxing = (st.phase == ACTIVE) | (st.phase == PENDING)
        bound = torch.where(relaxing, torch.minimum(st.maxradius, _RADIUS_GROWTH * st.radius),
                            torch.zeros_like(st.radius))
        bound = float(bound.max().item())
        return bound if self._step_length is None else min(bound, 2.0 * self._step_length)

class CellRelaxationBuilder(SlotUnionGraphBuilder):
    """:class:`~fastmd.execution.system_batch.SlotUnionGraphBuilder` whose systems' cells change inside the
    captured graph (a variable-cell relaxation).

    :meth:`set_cells` writes every cell table on the device. The skin test bounds how far any pair distance has
    moved since the candidate refresh: a pair within the cutoff now was within ``cutoff / (1 - e) + 2 delta`` of
    it at the refresh, ``delta`` the largest move of an atom in fractional coordinates times the refresh's cell
    and ``e = sqrt(|A|_1 |A|_inf)`` (``A = C_ref^-1 C - 1``) a bound on the spectral norm of its cell's
    deformation since then (``1 - e`` bounds the deformation's smallest singular value). The candidates hold
    while ``2 delta + cutoff e / (1 - e) <= skin`` and ``e < 1/2`` (the far self-loop padding, at least twice the
    cutoff long at the refresh, then stays beyond the cutoff), so the reported displacement is
    ``delta + cutoff e / (2 (1 - e))``, infinite from ``e = 1/2``. Fractional coordinates do not change with the
    cell, so the candidates' images and the wraps stay exact. A candidate refresh takes the current cells as the
    reference (the cell list's bins, the far self-loop images and the sort keys' image bound with them; eager)."""

    def __init__(self, *, cells, **kwargs) -> None:
        device = torch.device(kwargs.get("device", "cuda"))
        self.current_cells = torch.as_tensor(cells, dtype=torch.float64).reshape(-1, 3, 3).to(device).clone()
        super().__init__(cells=self.current_cells.clone(), **kwargs)
        self.reference_cells = self.current_cells.clone()
        self.reference_inverse = inverse33(self.reference_cells)
        self.row_reference_cells = self.row_cells.clone()
        self._eye = torch.eye(3, dtype=torch.float64, device=self.device)

    def _set_cell(self, slot: int, cell64: Tensor) -> None:
        super()._set_cell(slot, cell64)
        self.current_cells[slot].copy_(cell64)

    def set_cells(self, cells: Tensor) -> None:
        """Every system's cell ``cells`` (``[S, 3, 3]`` float64 on the device) into the tables, in place (device
        ops only, capture-safe): the float32 search cells and their inverses, each row's copies, the model's cells
        (``systems.cells``) and the batch's cell (the first system's)."""
        self.current_cells.copy_(cells)
        self.cell.copy_(cells)
        # Match a single-system builder's inverse of each rounded float32 cell.
        # Inverting cells in float64 first changes the model's fractional coordinates.
        # inv_ex avoids the host error check in inv, so this remains capture-safe.
        for slot in range(self.layout.count):
            self.inv_cells[slot].copy_(torch.linalg.inv_ex(self.cell[slot], check_errors=False).inverse)
        self.systems.cells.copy_(cells)
        rows = self.systems.system_of_row
        self.row_cells.copy_(self.cell.index_select(0, rows))
        self.row_inv_cells.copy_(self.inv_cells.index_select(0, rows))
        if self.row_model_cells is not None:
            self.row_model_cells.copy_(self.systems.cells.index_select(0, rows))
        self.batch.cell.copy_(self.systems.cells[0])

    def _displacement(self, positions: Tensor) -> Tensor:
        fractional = small_matmul(positions.unsqueeze(1), self.row_inv_cells).squeeze(1)
        moved = small_matmul(fractional.unsqueeze(1), self.row_reference_cells).squeeze(1)
        delta = torch.linalg.vector_norm(moved - self.reference_positions, dim=1).max()
        change = (torch.matmul(self.reference_inverse, self.current_cells) - self._eye).abs()
        deformation = torch.sqrt(change.sum(dim=1).amax(dim=1) * change.sum(dim=2).amax(dim=1))
        strain = torch.where(deformation < 0.5, self.atom_graph_cutoff * deformation / (1.0 - deformation),
                             torch.full_like(deformation, math.inf)).max()
        return delta + (0.5 * strain).to(delta.dtype)

    @torch.no_grad()
    def _take_reference_cells(self) -> None:
        """The current cells become the skin test's reference, with the cell list's bins, the far self-loop
        images and the sort keys' image bound (host work: a candidate refresh, eager)."""
        if self.sorted_edges and self._image_bound(self.cell) > self.sort_key_max_image:
            raise ValueError(f"a relaxed cell became thinner than the union's sort keys allow (image bound "
                             f"{self._image_bound(self.cell)} > {self.sort_key_max_image})")
        cells, radius = estimate_batch_cell_list_sizes(self.cells, self.pbc, self.candidate_cutoff)
        if cells > self._cell_list_cells:  # the cell list is eager only: a new one is no recapture
            self._cell_list_cells = int(cells)
            self.cell_list_cache = allocate_cell_list(self.n_real, int(cells), radius, self.device)
        else:
            self.cell_list_cache[1].copy_(radius)
        if self._loop_images is not None:  # self_loop_padding's far image (its longest lattice vector) of every cell
            lengths = self.cell.double().norm(dim=2)
            longest, axis = lengths.max(dim=1)
            multiple = torch.ceil(max(self.padding.min_length, 2.0 * self.atom_graph_cutoff) / longest).clamp(min=1)
            self._loop_images.zero_().scatter_(1, axis[:, None], multiple[:, None].to(torch.int32))
            if self._loop_lengths is not None:  # a line graph's padded pairs: the loops' length, its overflow loops
                self._loop_lengths.copy_(multiple * longest)
                self._set_overflow_images()
        self.reference_cells.copy_(self.current_cells)
        self.reference_inverse.copy_(inverse33(self.current_cells))
        self.row_reference_cells.copy_(self.row_cells)

    @torch.no_grad()
    def refresh_candidates(self, positions: Tensor) -> None:
        self._take_reference_cells()
        super().refresh_candidates(positions)

    @torch.no_grad()
    def build(self, positions: Tensor, *, rebuild_candidates: bool = True, predicate=None):
        if rebuild_candidates:
            self._take_reference_cells()
        return super().build(positions, rebuild_candidates=rebuild_candidates, predicate=predicate)


class SlotRelaxationRunner(SlotWholeStepRunner):
    """:class:`~fastmd.execution.system_batch.SlotWholeStepRunner` relaxing its systems with :class:`SystemFIRE`
    (a :class:`CellRelaxationBuilder` with a cell filter): each step the integrator's FIRE step, its cells into
    the builder (a cell filter), the union's evaluation (with each system's stress for a cell filter) and the
    integrator's convergence. The window record holds each system's phase (read with the window's flags in one
    copy: :attr:`phases` after :meth:`settle`); a system whose state turns non-finite is a failed phase, not an
    error. The mirrors are the integrator's state (positions, velocities, the stored forces and energies)."""

    # A relaxation captures its unions again as they drain: each capture again warms up on two steps rather than
    # a whole window (ORB v3 in IEEE FP32, #79's WBM set: the warmup window was two thirds of a drain's 0.45 s).
    recapture_warmup_steps = 2

    def __init__(self, **kwargs) -> None:
        self._stress = None
        self.phases: tuple = ()
        super().__init__(**kwargs)
        # FIRE moves a system's atoms at most maxstep per step (the norm of its whole step), GOQN at most its trust
        # radius: an accelerating relaxation outgrows the growth its earlier windows measured, so the refresh rule
        # takes this as a floor of the growth (with a cell filter the skin test's strain term comes on top: a
        # floor, not a bound); GOQN's follows its radii (each settle)
        self.step_displacement_bound = self.integrator.step_bound()

    def _list_built_at_window_start(self) -> bool:
        """:meth:`WholeStepGraphRunner._list_built_at_window_start`, and with a cell filter the reference cells
        of the list are the window's start cells."""
        if not super()._list_built_at_window_start():
            return False
        builder = self.builder
        return not isinstance(builder, CellRelaxationBuilder) or bool(
            torch.equal(builder.reference_cells, builder.current_cells))

    @staticmethod
    def _union_shape(builder, integrator) -> tuple[int, int]:
        if not isinstance(builder, SlotUnionGraphBuilder) or not isinstance(integrator, SystemFIRE):
            raise TypeError("SlotRelaxationRunner needs a SlotUnionGraphBuilder and a SystemFIRE")
        if integrator.layout != builder.layout:
            raise ValueError("builder and integrator must hold the same slots")
        if integrator.cell_filter != isinstance(builder, CellRelaxationBuilder):
            raise ValueError("a cell filter relaxes with a CellRelaxationBuilder, a fixed cell without one")
        return builder.layout.count, 0

    def _health(self) -> Tensor:
        """``(S,)`` float64: each system's phase (``SystemFIRE.phase``)."""
        return self.integrator.phase

    @scoped_call
    def _evaluate(self, positions: Tensor, *, rebuild_candidates: bool, stress: bool = False):
        if stress:
            raise ValueError("SlotRelaxationRunner records no stress window")
        graph, status = self.builder.build(positions, rebuild_candidates=rebuild_candidates)
        records = self.integrator.records
        if not self.integrator.cell_filter and not (records is not None and records.stress):
            energies, forces = system_energy_derivative(self.adapter, self.builder.batch, graph)
            return forces, energies, status
        energies, forces, stresses = system_energy_derivative(self.adapter, self.builder.batch, graph, stress=True)
        if self._stress is None:  # the first evaluation is eager (construction)
            self._stress = torch.zeros_like(stresses)
        self._stress.copy_(stresses.detach())
        return forces, energies, status

    def _stateful_step_body(self, *, rebuild_candidates: bool, stress: bool = False, mirror: bool = True) -> object:
        integrator = self.stateful_integrator
        positions = integrator.step_pre_force()
        if integrator.cell_filter:
            self.builder.set_cells(integrator.cells)
        forces, energy, status = self._evaluate(positions, rebuild_candidates=rebuild_candidates,
                                                **({"stress": True} if stress else {}))
        with torch.no_grad():
            integrator.step_post_force(forces.detach(), energy=energy.detach(), stress=self._stress)
            if not mirror:
                return status
            self.positions.copy_(positions)
            self.momenta.copy_(integrator.momenta)
            self.forces.copy_(integrator.forces)
            self.potential_energy.copy_(integrator.potential_energy)
        return status

    def _sync_cells(self) -> None:
        self.integrator.sync()
        if self.integrator.cell_filter:
            self.builder.set_cells(self.integrator.cells)

    def _restore_backup(self) -> None:
        super()._restore_backup()
        self._sync_cells()

    def _capture(self) -> None:
        try:
            super()._capture()
        finally:  # the warmup windows moved the builder's cells; the integrator's state is the saved one
            self._sync_cells()

    def rewind(self) -> None:
        super().rewind()
        self._sync_cells()

    def forget_history(self) -> None:
        """:meth:`WholeStepGraphRunner.forget_history`, the optimizer's (``SystemFIRE.forget_history``) and the
        refresh rule's floor from it, as a new runner's."""
        super().forget_history()
        self.integrator.forget_history()
        self.step_displacement_bound = self.integrator.step_bound()


    def settle(self, *, record: bool = False, flags: tuple | None = None) -> None:
        """:meth:`WholeStepGraphRunner.settle` (an overflow's fallback, the counts, the proactive candidate
        refresh), and the systems' phases (:attr:`phases`) from the same copy, or from the integrator after a
        fallback window (whose eager steps write no record)."""
        packed = tuple(self._window_record.tolist()) if flags is None else tuple(flags)
        fallbacks = self.fallback_windows
        if isinstance(self.integrator, SystemGOQN):
            self.step_displacement_bound = self.integrator.step_bound()
        WholeStepGraphRunner.settle(self, record=record, flags=packed[:4])
        self.phases = (packed[4:] if self.fallback_windows == fallbacks
                       else tuple(self.integrator.phase.tolist()))

    @torch.no_grad()
    def reset(self, *, positions: Tensor, momenta: Tensor, forces: Tensor, potential_energy: Tensor,
              reset_counters: bool = True) -> None:
        self._sync_cells()
        super().reset(positions=positions, momenta=momenta, forces=forces, potential_energy=potential_energy,
                      reset_counters=reset_counters)


def relaxation_builder(adapter, *, cell_filter: bool, layout: SlotLayout, cells, atomic_numbers: Tensor, atoms,
                       u_capacity: int, candidate_skin: float, device, admit_cells=None,
                       t_capacity: int = 0, config=None) -> SlotUnionGraphBuilder:
    """The slot union builder of ``adapter`` (:func:`~fastmd.execution.system_batch.slot_union_builder`'s
    options, ``t_capacity`` triplets for a model with a line graph; ``config``: its geometry dtype), a
    :class:`CellRelaxationBuilder` for a cell filter; ``admit_cells``: the cells of the systems written later (the
    sort keys hold them)."""
    from .replica_batch import graph_builder_options

    cutoff = float(adapter.topology_spec().cutoff)
    kind = CellRelaxationBuilder if cell_filter else SlotUnionGraphBuilder
    return kind(layout=layout, cells=cells, atomic_numbers=atomic_numbers, atoms=atoms, u_capacity=int(u_capacity),
                atom_graph_cutoff=cutoff, candidate_cutoff=cutoff + candidate_skin, device=device,
                padding=adapter.padding_rule(), prepare=adapter.prepare, admit_cells=admit_cells,
                **graph_builder_options(adapter, t_capacity, config))
