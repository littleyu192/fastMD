from __future__ import annotations

from collections.abc import Callable

import torch
from ase import units
from torch import Tensor

from .._nvtx import nvtx_range


class StatefulIntegrator:
    """Integrator that owns the authoritative MD state in its own buffers.

    :class:`GPUIntegrator` is functional: the MD driver keeps the float32
    state and passes it in. A stateful integrator (e.g. the float64
    :class:`~matris.applications.gpu_nhc.GPUNoseHooverChainNVT`) keeps
    positions, momenta, the cached forces F(positions), the potential energy
    and any thermostat variables itself; the drivers only keep float32
    mirrors of positions/momenta/forces/energy for graph building, logging
    and statistics. One MD step is split around the single force evaluation
    and must be capture-safe (no host syncs or allocations of state)::

        q = integrator.step_pre_force()                 # evaluation positions
        forces, energy = model(q)
        integrator.step_post_force(forces, energy=energy)

    Transactions: :meth:`backup` / :meth:`restore` snapshot the full state
    with device copies (recorded inside whole-step CUDA graphs);
    :meth:`snapshot` / :meth:`load_snapshot` return/restore host-held clones
    of state and backup (capture warmup, admission trials, precapture).
    """

    positions: Tensor
    momenta: Tensor
    forces: Tensor
    potential_energy: Tensor

    def step_pre_force(self) -> Tensor:
        raise NotImplementedError

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None) -> None:
        raise NotImplementedError

    def set_forces(self, forces: Tensor, energy: Tensor | None = None) -> None:
        raise NotImplementedError

    def backup(self) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> object:
        raise NotImplementedError

    def load_snapshot(self, snapshot: object) -> None:
        raise NotImplementedError

    def kinetic_energy(self) -> Tensor:
        raise NotImplementedError

    def temperature(self) -> Tensor:
        raise NotImplementedError


class GPUIntegrator:
    """Small GPU-resident Velocity-Verlet / Berendsen NVT integrator.

    The equations intentionally mirror ASE's implementations for unconstrained
    systems.  Positions, momenta, forces, and masses stay as tensors, so an MD
    loop can consume model forces without copying them back to NumPy each step.
    """

    def __init__(
        self,
        masses: Tensor,
        dt_fs: float,
        ensemble: str = "nvt",
        temperature: float = 300.0,
        taut_fs: float = 100.0,
        fix_com: bool | None = None,
        degrees_of_freedom: int | None = None,
    ) -> None:
        ensemble = ensemble.lower()
        if ensemble not in {"nve", "nvt"}:
            raise ValueError(
                f"unsupported GPUIntegrator ensemble: {ensemble} "
                "(Nose-Hoover chain: GPUNoseHooverChainNVT / ensemble='nvt_nhc')"
            )
        self.masses = masses.reshape(-1, 1)
        self.dt = float(dt_fs) * units.fs
        self.ensemble = ensemble
        self.temperature_target = float(temperature)
        self.taut = float(taut_fs) * units.fs
        self.fix_com = (ensemble == "nvt") if fix_com is None else bool(fix_com)
        self.degrees_of_freedom = (
            int(degrees_of_freedom)
            if degrees_of_freedom is not None
            else int(self.masses.numel() * 3)
        )

    def kinetic_energy(self, momenta: Tensor) -> Tensor:
        kinetic = 0.5 * momenta * momenta / self.masses
        if momenta.ndim == 2:
            return kinetic.sum()
        if momenta.ndim == 3:
            return kinetic.sum(dim=(-2, -1))
        raise ValueError(
            f"momenta must have shape [N, 3] or [B, N, 3], got {momenta.shape}"
        )

    def temperature(self, momenta: Tensor) -> Tensor:
        return 2.0 * self.kinetic_energy(momenta) / (
            self.degrees_of_freedom * units.kB
        )

    def scale_velocities(self, momenta: Tensor) -> Tensor:
        if self.ensemble != "nvt":
            return momenta
        old_temperature = self.temperature(momenta).clamp_min(1e-12)
        tautscl = self.dt / self.taut
        scale = torch.sqrt(
            1.0 + (self.temperature_target / old_temperature - 1.0) * tautscl
        )
        scale = torch.clamp(scale, 0.9, 1.1)
        if momenta.ndim == 3:
            scale = scale[:, None, None]
        return momenta * scale

    def step(
        self,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None,
        force_fn: Callable[[Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        with nvtx_range("gpu_integrator"):
            momenta = self.scale_velocities(momenta)
        if forces is None:
            forces = force_fn(positions)

        with nvtx_range("gpu_integrator"):
            momenta = momenta + 0.5 * self.dt * forces
            if self.fix_com:
                atom_dim = -2
                momenta = momenta - momenta.sum(
                    dim=atom_dim, keepdim=True
                ) / float(momenta.shape[atom_dim])

            positions = positions + self.dt * momenta / self.masses
        forces = force_fn(positions)
        with nvtx_range("gpu_integrator"):
            momenta = momenta + 0.5 * self.dt * forces
        return positions, momenta, forces
