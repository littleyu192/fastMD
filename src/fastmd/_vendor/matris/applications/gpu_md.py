from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import gc
import time

import numpy as np
import torch
from ase import Atoms, units
from ase.io import Trajectory
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch import Tensor

from ..graph import FixedCapacityGraphBuilder, TensorGraphBuilder
from ..model.model import MatRIS
from ..config import InferenceConfig, env_value
from ._config import (
    apply_checkpoint, configure_entrypoint, config_summary, scoped_call,
)
from .capacity_policy import CapacityKey, JointCapacityPolicy
from .cuda_graph import BucketedGraphRunner
from .gpu_integrator import GPUIntegrator, StatefulIntegrator
from .gpu_logging import AsyncGPUMDLogger
from .gpu_nhc import GPUNoseHooverChainNVT
from .whole_step_cuda_graph import FixedCapacityOverflow, WholeStepGraphRunner


def _env_flag(name: str, default: str = "0") -> bool:
    return env_value(name, default) == "1"


# S8: whole-step tier cache — keep replaced-capacity graphs keyed by
# (u_capacity, t_capacity) with a shared graph pool, switch between them at
# transaction boundaries, and shrink behind hysteresis instead of paying a
# permanent padding tax after transient densification spikes.
_WHOLE_STEP_TIER_CACHE = _env_flag("MATRIS_WHOLE_STEP_TIER_CACHE")
_WHOLE_STEP_MAX_TIERS = int(env_value("MATRIS_WHOLE_STEP_MAX_TIERS", "3"))
_TIER_HYSTERESIS_WINDOW = int(env_value("MATRIS_TIER_HYSTERESIS_WINDOW", "2000"))
_TIER_HYSTERESIS_SIGMA = float(env_value("MATRIS_TIER_HYSTERESIS_SIGMA", "3.0"))
_TIER_CHECK_INTERVAL = int(env_value("MATRIS_TIER_CHECK_INTERVAL", "256"))
# After an overflow-driven up-switch, block shrinks for this many replays so
# a demand spike recurring on a period longer than the window cannot drive
# unbounded down/up flip-flop (default: two hysteresis windows).
_TIER_DOWN_COOLDOWN = int(
    env_value("MATRIS_TIER_DOWN_COOLDOWN", str(2 * _TIER_HYSTERESIS_WINDOW))
)
# Device-memory guard for tier captures. Per-tier byte attribution on a
# shared pool is path-dependent and ill-defined, so the budget asks the
# DEVICE instead: before capturing a tier, require mem_get_info free bytes
# to cover a conservative standalone estimate plus this floor. If not, the
# pool-reset valve runs first; if still short, the capture proceeds and an
# OOM unwinds through the exception-safe switch path.
_TIER_MIN_FREE_GIB = float(env_value("MATRIS_TIER_MIN_FREE_GIB", "6.0"))
_TIER_COST_MARGIN = 1.25
# S9: lattice cold starts under-estimate thermal triplet demand (measured
# +20% on Si-1000 within 0.3 ps of thermalization); an inflation factor at
# calibration time avoids the early grow-recapture chain.
_CAPACITY_THERMAL_FACTOR = float(env_value("MATRIS_CAPACITY_THERMAL_FACTOR", "1.0"))


@dataclass
class GPUMDState:
    positions: Tensor
    momenta: Tensor
    forces: Tensor | None
    potential_energy: Tensor | None


class GPUResidentMolecularDynamics:
    """GPU-resident MD fast path for fixed-cell single-system MD.

    This class is intentionally narrower than the ASE-compatible
    :class:`MolecularDynamics`: it targets the hot benchmark path where atom
    positions, momenta, forces, and graph inputs stay on CUDA tensors.  ASE or
    pymatgen objects are used only for initialization and optional snapshots.

    Isolated atoms retain their reference energies by default. Set
    ``config=InferenceConfig(isolated_atoms='error')`` for strict rejection; models without a
    reference-energy table always use strict rejection.

    Ensembles (``timestep``, ``taut`` and ``tdamp`` in fs):

    * ``"nve"``: velocity Verlet; ``"nvt"``: Berendsen (``taut``). Both remove
      the centre-of-mass momentum every step and keep a float32 state.
    * ``"nvt_nhc"``: Nose-Hoover chain, an ASE-exact port of
      ``ase.md.nose_hoover_chain.NoseHooverChainNVT`` (``tdamp``, default
      ``100 * timestep``; ``tchain``; ``tloop``) with a float64 state
      (:class:`~matris.applications.gpu_nhc.GPUNoseHooverChainNVT`). Initial
      momenta come from ``atoms.get_momenta()`` (e.g. set by ASE's
      ``MaxwellBoltzmannDistribution`` or
      :func:`~matris.applications.gpu_nhc.maxwell_boltzmann_momenta`); no
      centre-of-mass correction, as in ASE. It runs in the eager, model-graph
      and whole-step CUDA-graph paths; ``self.state`` holds float32 mirrors
      and ``self.integrator`` the authoritative state. Trajectory frames and
      :meth:`snapshot_atoms` carry the float64 positions/momenta; trajectory
      frames also carry ``atoms.info["nhc_eta"]``, ``["nhc_p_eta"]`` and
      ``["nhc_conserved_energy"]``.
    """

    @configure_entrypoint("gpu_md")
    def __init__(
        self,
        atoms: Atoms | Structure | Molecule,
        model_path: str | None = None,
        model: MatRIS | str = "matris_10m_oam",
        task: str = "efsm",
        device: torch.device | str = "cuda",
        ensemble: str = "nvt",
        temperature: float = 300.0,
        timestep: float = 1.0,
        taut: float = 100.0,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        logger_ring: int = 8,
        precapture_steps: int = 0,
        config: InferenceConfig | None = None,
        tdamp: float | None = None,
        tchain: int = 3,
        tloop: int = 1,
    ) -> None:
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        self.atoms_template = atoms.copy()
        self.task = task
        self.force_task = "ef"
        self.timestep = float(timestep)
        self.nsteps = 0
        self._logged_initial = False
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        if isinstance(model, MatRIS):
            self.model = model.to(self.device)
            apply_checkpoint(self.model, self._inference_config.checkpoint_enabled)
        else:
            self.model = MatRIS.load(
                model_path=model_path,
                model_name=model,
                device=str(self.device),
                enable_checkpoint=self._inference_config.checkpoint_enabled,
            )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device
        )
        self.cell = torch.as_tensor(
            atoms.get_cell().array, dtype=torch.float32, device=self.device
        )
        self.pbc = torch.as_tensor(atoms.get_pbc(), dtype=torch.bool, device=self.device)
        self.masses = torch.as_tensor(
            atoms.get_masses(), dtype=torch.float32, device=self.device
        )
        self.n_atoms = int(self.atomic_numbers.shape[0])
        self.composition = atoms.get_chemical_formula()

        positions = torch.as_tensor(
            atoms.get_positions(), dtype=torch.float32, device=self.device
        )
        if atoms.has("momenta"):
            momenta_np = atoms.get_momenta()
        else:
            momenta_np = np.zeros((self.n_atoms, 3), dtype=np.float32)
        momenta = torch.as_tensor(momenta_np, dtype=torch.float32, device=self.device)
        self.state = GPUMDState(
            positions=positions,
            momenta=momenta,
            forces=None,
            potential_energy=None,
        )

        if ensemble.lower() == "nvt_nhc":
            # float64 state from the float64 ASE arrays (ASE-exact dynamics)
            self.integrator = GPUNoseHooverChainNVT(
                atoms.get_masses(),
                atoms.get_positions(),
                atoms.get_momenta(),
                temperature_K=temperature,
                timestep=self.timestep * units.fs,
                tdamp=(100.0 * self.timestep if tdamp is None else float(tdamp))
                * units.fs,
                tchain=tchain,
                tloop=tloop,
                device=self.device,
                force_dtype=torch.float32,
            )
            self.integrator.warmup_compile()
        else:
            self.integrator = GPUIntegrator(
                self.masses,
                dt_fs=timestep,
                ensemble=ensemble,
                temperature=temperature,
                taut_fs=taut,
                fix_com=True,
                degrees_of_freedom=self.n_atoms * 3,
            )
        self.stateful_integrator = (
            self.integrator
            if isinstance(self.integrator, StatefulIntegrator)
            else None
        )
        self.whole_step_cuda_graph = self._inference_config.execution == "whole_step"
        self.enable_model_fusions = self._inference_config.optimization_profile != "generic"
        self.validate_isolated_atoms = (
            not self._inference_config.handle_isolated_atoms
            or self.model.reference_energy is None
        )
        self.candidate_skin = self._inference_config.capacity.candidate_skin
        self.runner = None
        self.whole_step_runner = None
        self.whole_step_recaptures = 0
        self._whole_step_accumulated = {
            "launches": 0,
            "candidate_refreshes": 0,
            "fallback_windows": 0,
            "recovery_replays": 0,
            "max_edge_count": 0,
            "max_triplet_count": 0,
            "max_displacement": 0.0,
        }
        self.whole_step_capacity_history: list[tuple[int, int]] = []
        self.capacity_policy: JointCapacityPolicy | None = None
        self._tier_runners: dict[CapacityKey, WholeStepGraphRunner] = {}
        self._tier_lru: list[CapacityKey] = []
        self._tier_pool = None
        self._tier_demand_u: deque[int] = deque(maxlen=_TIER_HYSTERESIS_WINDOW)
        self._tier_demand_t: deque[int] = deque(maxlen=_TIER_HYSTERESIS_WINDOW)
        self._tier_switch_up = 0
        self._tier_switch_down = 0
        self._tier_captures = 0
        self._tier_evictions = 0
        self._tier_check_counter = 0
        self._tier_replay_counter = 0
        self._tier_last_up_at = -(10**18)
        self._tier_cost_per_triplet = 0.0
        self._tier_pool_resets = 0
        self.atom_graph_cutoff = float(self.model.graph_converter.atom_graph_cutoff)
        self.line_graph_cutoff = float(self.model.graph_converter.line_graph_cutoff)
        self.graph_builder = TensorGraphBuilder(
            cell=self.cell,
            atomic_numbers=self.atomic_numbers,
            pbc=self.pbc,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
            device=self.device,
            composition=self.composition,
            check_isolated_atoms=self.validate_isolated_atoms,
        )
        if self.whole_step_cuda_graph:
            self.enable_whole_step_cuda_graph(precapture_steps=precapture_steps)
        elif self._inference_config.execution == "model_graph":
            self.enable_model_cuda_graph(precapture_steps=precapture_steps)
        self.logger = AsyncGPUMDLogger(
            atoms_template=self.atoms_template,
            device=self.device,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
            ring=logger_ring,
            # nvt_nhc: log the float64 state and the thermostat variables
            **(
                {
                    "state_dtype": torch.float64,
                    "extras": {
                        "nhc_eta": self.integrator.tchain,
                        "nhc_p_eta": self.integrator.tchain,
                        "nhc_conserved_energy": 1,
                    },
                }
                if isinstance(self.integrator, GPUNoseHooverChainNVT)
                else {}
            ),
        )

    def config_summary(self) -> dict:
        """Requested/resolved configuration and observed execution state."""
        return config_summary(self)

    @scoped_call
    def enable_model_cuda_graph(self, precapture_steps: int = 0) -> None:
        """Enable model-only replay from the current GPU-resident MD state."""
        if self.device.type != "cuda":
            raise ValueError("model CUDA graph capture requires a CUDA device")
        if self.whole_step_runner is not None:
            raise ValueError("cannot enable model replay with a whole-step runner")
        if self.runner is not None:
            return
        capacity = self._inference_config.capacity
        self.runner = BucketedGraphRunner(
            self.model,
            task=self.force_task,
            enable_model_fusions=self.enable_model_fusions,
            u_step=capacity.u_step, t_step=capacity.t_step,
            n_dummy=capacity.n_dummy, min_pad_u=capacity.min_pad_u,
            warmup=capacity.warmup,
            config=self._inference_config,
        )
        self._inference_config = replace(self._inference_config, execution="model_graph")
        self.runner._inference_config = self._inference_config
        self.precapture(steps=precapture_steps, proactive=True)

    @scoped_call
    def enable_whole_step_cuda_graph(self, precapture_steps: int = 0) -> None:
        """Capture the whole-step path from the current GPU-resident MD state."""
        if self.device.type != "cuda":
            raise ValueError("whole-step CUDA graph capture requires a CUDA device")
        if self.whole_step_runner is not None:
            return
        if self.runner is not None:
            raise ValueError("cannot replace an active bucketed CUDA graph runner")

        self.whole_step_cuda_graph = True
        sample_graph = self.graph_builder.build(self.state.positions)
        capacity = self._inference_config.capacity
        self.capacity_policy = JointCapacityPolicy(
            u_step=capacity.u_step, t_step=capacity.t_step,
            min_pad_u=capacity.min_pad_u, min_pad_t=capacity.min_pad_t,
        )
        u_demand = sample_graph.undirected2directed.shape[0]
        t_demand = sample_graph.line_graph.shape[0]
        if _CAPACITY_THERMAL_FACTOR != 1.0:
            t_demand = int(round(t_demand * _CAPACITY_THERMAL_FACTOR))
        initial_capacity = self.capacity_policy.start(u_demand, t_demand)
        self._whole_u_capacity = initial_capacity.u_capacity
        self._whole_t_capacity = initial_capacity.t_capacity
        self.whole_step_capacity_history.append(
            (self._whole_u_capacity, self._whole_t_capacity)
        )
        if _WHOLE_STEP_TIER_CACHE and self._tier_pool is None:
            self._tier_pool = torch.cuda.graph_pool_handle()
        while True:
            try:
                self.whole_step_runner = self._make_whole_step_runner(
                    positions=self.state.positions,
                    momenta=self.state.momenta,
                    forces=self.state.forces,
                )
                break
            except FixedCapacityOverflow as overflow:
                # The sizing graph comes from the dynamic builder; if the
                # fixed-capacity build of the same state does not fit (e.g. a
                # pair exactly at the cutoff), grow before the first capture
                # instead of running on the substitute topology.
                current = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
                grown = self.capacity_policy.grow(
                    current, (overflow.edge_count + 1) // 2, overflow.triplet_count
                )
                if grown == current:
                    raise
                self._whole_u_capacity = grown.u_capacity
                self._whole_t_capacity = grown.t_capacity
                self.whole_step_capacity_history.append(
                    (self._whole_u_capacity, self._whole_t_capacity)
                )
        initial_capacity = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        self._inference_config = replace(self._inference_config, execution="whole_step")
        self.whole_step_runner._inference_config = self._inference_config
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_runners[initial_capacity] = self.whole_step_runner
            if initial_capacity in self._tier_lru:
                self._tier_lru.remove(initial_capacity)
            self._tier_lru.append(initial_capacity)
        self._sync_whole_state()
        if precapture_steps <= 0:
            return

        initial = (
            self.state.positions.clone(),
            self.state.momenta.clone(),
            self.state.forces.clone(),
            self.state.potential_energy.clone(),
        )
        integrator_initial = (
            self.stateful_integrator.snapshot()
            if self.stateful_integrator is not None
            else None
        )
        for _ in range(int(precapture_steps)):
            self._replay_whole_step()
        if integrator_initial is not None:
            self.stateful_integrator.load_snapshot(integrator_initial)
        self.whole_step_runner.reset(
            positions=initial[0],
            momenta=initial[1],
            forces=initial[2],
            potential_energy=initial[3],
        )
        self._sync_whole_state()
        self.capacity_policy.reset_observations()
        # The precapture trajectory was rolled back; its demand samples must
        # not seed the production hysteresis window either.
        self._tier_demand_u.clear()
        self._tier_demand_t.clear()
        self._tier_check_counter = 0

    @scoped_call
    def _make_whole_step_runner(
        self,
        *,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None = None,
    ) -> WholeStepGraphRunner:
        fixed_builder = FixedCapacityGraphBuilder(
            cell=self.cell,
            atomic_numbers=self.atomic_numbers,
            pbc=self.pbc,
            u_capacity=self._whole_u_capacity,
            t_capacity=self._whole_t_capacity,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
            candidate_cutoff=self.atom_graph_cutoff + self.candidate_skin,
            device=self.device,
            composition=self.composition,
            n_dummy=self._inference_config.capacity.n_dummy,
        )
        return WholeStepGraphRunner(
            model=self.model,
            builder=fixed_builder,
            integrator=self.integrator,
            positions=positions,
            momenta=momenta,
            forces=forces,
            task=self.force_task,
            steps_per_launch=1,
            enable_model_fusions=self.enable_model_fusions,
            graph_pool=self._tier_pool,
            validate_isolated_atoms=self.validate_isolated_atoms,
            warmup=self._inference_config.capacity.warmup,
            config=self._inference_config,
        )

    def _sync_whole_state(self) -> None:
        assert self.whole_step_runner is not None
        whole_state = self.whole_step_runner.state
        self.state = GPUMDState(
            positions=whole_state.positions,
            momenta=whole_state.momenta,
            forces=whole_state.forces,
            potential_energy=whole_state.potential_energy[0],
        )

    def set_activation_checkpoint(self, enabled: bool | None) -> None:
        """Update inference checkpointing before capture without reloading weights."""
        if enabled is not None and not isinstance(enabled, bool):
            raise TypeError("enabled must be bool or None")
        captured = self.whole_step_runner is not None or (
            self.runner is not None and self.runner.captures > 0
        )
        if captured:
            raise ValueError("Changing checkpoint policy does not change already captured CUDA graphs; create a new instance with the desired config")
        apply_checkpoint(self.model, enabled)
        self._inference_config = replace(self._inference_config, checkpoint_enabled=enabled)
        if self.runner is not None:
            self.runner._inference_config = self._inference_config

    def _clone_state(self) -> tuple[GPUMDState, int, object]:
        return (
            GPUMDState(
                positions=self.state.positions.clone(),
                momenta=self.state.momenta.clone(),
                forces=None if self.state.forces is None else self.state.forces.clone(),
                potential_energy=(
                    None
                    if self.state.potential_energy is None
                    else self.state.potential_energy.clone()
                ),
            ),
            self.nsteps,
            (
                self.stateful_integrator.snapshot()
                if self.stateful_integrator is not None
                else None
            ),
        )

    def _restore_dynamic_state(self, checkpoint: tuple[GPUMDState, int, object]) -> None:
        state, nsteps, integrator_state = checkpoint
        if integrator_state is not None:
            self.stateful_integrator.load_snapshot(integrator_state)
        self.state = GPUMDState(
            positions=state.positions.clone(),
            momenta=state.momenta.clone(),
            forces=None if state.forces is None else state.forces.clone(),
            potential_energy=(
                None
                if state.potential_energy is None
                else state.potential_energy.clone()
            ),
        )
        self.nsteps = int(nsteps)

    def _timed_trial(self, steps: int) -> list[float]:
        samples_ms = []
        torch.cuda.synchronize(self.device)
        for _ in range(steps):
            started = time.perf_counter()
            self.step()
            torch.cuda.synchronize(self.device)
            samples_ms.append((time.perf_counter() - started) * 1.0e3)
        return samples_ms

    @scoped_call
    def select_whole_step_cuda_graph(
        self,
        *,
        warmup_steps: int = 5,
        min_speedup: float = 1.01,
    ) -> dict[str, object]:
        """Select dynamic execution or whole-step replay from matched states."""
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if min_speedup < 1.0:
            raise ValueError("min_speedup must be at least 1.0")
        if self.whole_step_runner is not None or self.runner is not None:
            raise ValueError("execution-path selection requires uncaptured GPU MD")
        if self.logger.enabled:
            raise ValueError("execution-path selection requires logging to be disabled")

        self._ensure_forces()
        initial = self._clone_state()
        dynamic_samples = self._timed_trial(warmup_steps)
        dynamic_final = self._clone_state()

        self._restore_dynamic_state(initial)
        self.enable_whole_step_cuda_graph()
        graph_samples = self._timed_trial(warmup_steps)
        graph_trial_stats = self.whole_step_stats()
        dynamic_median = float(np.median(dynamic_samples))
        graph_median = float(np.median(graph_samples))
        graph_speedup = dynamic_median / graph_median
        selected_path = "whole_step_cuda_graph"
        if graph_speedup < min_speedup:
            old_runner = self.whole_step_runner
            self.whole_step_runner = None
            self.whole_step_cuda_graph = False
            self._inference_config = replace(self._inference_config, execution="eager")
            self.capacity_policy = None
            # Tear down tier bookkeeping too: the trial runner is registered
            # in the tier cache, which would otherwise pin its graph and
            # workspaces past the del below and leave duplicate LRU entries
            # for a later re-enable.
            self._tier_runners.clear()
            self._tier_lru.clear()
            self._tier_pool = None
            self._tier_demand_u.clear()
            self._tier_demand_t.clear()
            self._tier_check_counter = 0
            self._restore_dynamic_state(dynamic_final)
            del old_runner
            gc.collect()
            torch.cuda.empty_cache()
            selected_path = "dynamic_gpu"

        return {
            "warmup_steps": warmup_steps,
            "min_speedup": min_speedup,
            "dynamic_samples_ms": dynamic_samples,
            "graph_samples_ms": graph_samples,
            "dynamic_median_ms": dynamic_median,
            "graph_median_ms": graph_median,
            "graph_speedup": graph_speedup,
            "selected_path": selected_path,
            "graph_trial_stats": graph_trial_stats,
        }

    def _accumulate_whole_step_stats(self, runner: WholeStepGraphRunner) -> None:
        old_stats = runner.stats()
        for name in (
            "launches",
            "candidate_refreshes",
            "fallback_windows",
            "recovery_replays",
        ):
            self._whole_step_accumulated[name] += old_stats[name]
        for name in ("max_edge_count", "max_triplet_count", "max_displacement"):
            self._whole_step_accumulated[name] = max(
                self._whole_step_accumulated[name], old_stats[name]
            )

    def _tier_capture_estimate_bytes(self, key: CapacityKey) -> int:
        """Conservative standalone-size estimate for capturing a tier."""
        if self._tier_cost_per_triplet <= 0.0:
            return 0
        return int(
            self._tier_cost_per_triplet * key.t_capacity * _TIER_COST_MARGIN
        )

    @scoped_call
    def reset_whole_step_graph_pool(self) -> None:
        """Whole-pool reset: the only true memory-release valve.

        Shared-pool eviction is reuse-only (deleting one graph returns its
        blocks to the pool, not to the device), so genuinely reclaiming
        graph memory requires dropping EVERY runner sharing the pool,
        emptying the cache, and recapturing the active tier into a fresh
        pool. Intended triggers: budget pressure before a tier capture,
        protocol phase boundaries (e.g. between annealing segments), and
        before switching to checkpoint-off replay. If the recapture fails,
        gpu_md.state is restored from a snapshot and the exception
        propagates with no whole-step runner active; callers may fall back
        to the dynamic path.
        """
        if self.whole_step_runner is None:
            return
        state = self.whole_step_runner.state
        snapshot = (
            state.positions.clone(),
            state.momenta.clone(),
            state.forces.clone(),
            state.potential_energy.clone(),
        )
        self._accumulate_whole_step_stats(self.whole_step_runner)
        self.whole_step_runner = None
        self._tier_runners.clear()
        self._tier_lru.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self._tier_pool = (
            torch.cuda.graph_pool_handle() if _WHOLE_STEP_TIER_CACHE else None
        )
        self._tier_pool_resets += 1
        key = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        try:
            runner = self._make_whole_step_runner(
                positions=snapshot[0],
                momenta=snapshot[1],
                forces=snapshot[2],
            )
        except BaseException:
            self.state = GPUMDState(
                positions=snapshot[0],
                momenta=snapshot[1],
                forces=snapshot[2],
                potential_energy=snapshot[3][0],
            )
            raise
        self.whole_step_runner = runner
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_runners[key] = runner
            self._tier_lru.append(key)
        self._sync_whole_state()

    def _switch_whole_step_tier(self, key: CapacityKey) -> None:
        """Switch the active whole-step tier, capturing it on first visit.

        Tier switches happen only between transactions: the caller holds a
        committed (or restored) state, so moving it into another tier's
        persistent buffers is the same reset pattern the post-precapture
        path already relies on. Cached tiers replay mutually exclusively
        and share one graph memory pool.
        """
        assert self.whole_step_runner is not None
        current = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        if key == current:
            return
        old_runner = self.whole_step_runner
        old_state = old_runner.state
        # Run every fallible operation (capture can OOM, the candidate
        # assert can raise) BEFORE committing any bookkeeping, so an
        # exception leaves the old tier fully active and consistent.
        cached = self._tier_runners.get(key)
        created = cached is None
        if created:
            # Evict BEFORE capturing: pool eviction is reuse-only (deleting a
            # graph returns its blocks to the shared pool, not to the device),
            # so the new capture can only consume evicted blocks if the
            # eviction happens first — the reverse order grows the pool by a
            # full tier and strands the evicted blocks. Losing a cached tier
            # if the capture below then fails is acceptable: the active
            # runner and all bookkeeping remain consistent.
            while len(self._tier_runners) >= _WHOLE_STEP_MAX_TIERS:
                evict_key = next(
                    (k for k in self._tier_lru if k != current), None
                )
                if evict_key is None:
                    break
                self._tier_lru.remove(evict_key)
                evicted = self._tier_runners.pop(evict_key)
                del evicted
                self._tier_evictions += 1
                gc.collect()
                torch.cuda.empty_cache()
            # Budget guard: ask the device, not a ledger. If free memory
            # cannot cover a conservative estimate of the new tier, fire the
            # pool-reset valve (which also re-registers the ACTIVE tier and
            # refreshes old_state's backing runner) before capturing.
            estimate = self._tier_capture_estimate_bytes(key)
            floor_bytes = int(_TIER_MIN_FREE_GIB * (1 << 30))
            if estimate:
                free, _ = torch.cuda.mem_get_info()
                if free < estimate + floor_bytes:
                    self.reset_whole_step_graph_pool()
                    old_runner = self.whole_step_runner
                    old_state = old_runner.state
            self._whole_u_capacity = key.u_capacity
            self._whole_t_capacity = key.t_capacity
            free_before, _ = torch.cuda.mem_get_info()
            try:
                cached = self._make_whole_step_runner(
                    positions=old_state.positions,
                    momenta=old_state.momenta,
                    forces=old_state.forces,
                )
            except BaseException:
                self._whole_u_capacity = current.u_capacity
                self._whole_t_capacity = current.t_capacity
                raise
            free_after, _ = torch.cuda.mem_get_info()
            consumed = max(0, free_before - free_after)
            if key.t_capacity > 0 and consumed > 0:
                self._tier_cost_per_triplet = max(
                    self._tier_cost_per_triplet,
                    consumed / key.t_capacity,
                )
            self._tier_runners[key] = cached
        else:
            cached.reset(
                positions=old_state.positions,
                momenta=old_state.momenta,
                forces=old_state.forces,
                potential_energy=old_state.potential_energy,
                reset_counters=True,
            )
            cached.builder.assert_candidate_capacity()
            self._whole_u_capacity = key.u_capacity
            self._whole_t_capacity = key.t_capacity
        self._accumulate_whole_step_stats(old_runner)
        self.whole_step_capacity_history.append(
            (key.u_capacity, key.t_capacity)
        )
        if created:
            self.whole_step_recaptures += 1
            self._tier_captures += 1
        self.whole_step_runner = cached
        if key in self._tier_lru:
            self._tier_lru.remove(key)
        self._tier_lru.append(key)
        while len(self._tier_runners) > _WHOLE_STEP_MAX_TIERS:
            evict_key = next((k for k in self._tier_lru if k != key), None)
            if evict_key is None:
                break
            self._tier_lru.remove(evict_key)
            evicted = self._tier_runners.pop(evict_key)
            del evicted
            self._tier_evictions += 1
            gc.collect()
        self._tier_demand_u.clear()
        self._tier_demand_t.clear()
        self._sync_whole_state()

    def _maybe_switch_down_tier(self) -> None:
        """Shrink to a smaller tier once demand has stably receded.

        Hysteresis: the window must be full, the bucket-rounded p95 plus a
        sigma-scaled margin must select a strictly smaller tier, and the
        window maximum must still fit that tier with minimum padding. The
        window is cleared on every switch, so consecutive switches are at
        least one full window apart.
        """
        maxlen = self._tier_demand_t.maxlen
        if not maxlen or len(self._tier_demand_t) < maxlen:
            return
        if self._tier_replay_counter - self._tier_last_up_at < _TIER_DOWN_COOLDOWN:
            return
        assert self.capacity_policy is not None
        u_values = sorted(self._tier_demand_u)
        t_values = sorted(self._tier_demand_t)
        index = int(0.95 * (len(u_values) - 1))
        mean_u = sum(u_values) / len(u_values)
        mean_t = sum(t_values) / len(t_values)
        std_u = (sum((v - mean_u) ** 2 for v in u_values) / len(u_values)) ** 0.5
        std_t = (sum((v - mean_t) ** 2 for v in t_values) / len(t_values)) ** 0.5
        # Sigma margins only; key_for already enforces min_pad headroom on
        # top, so folding min_pad in here would double-count padding and can
        # round the candidate ABOVE the current tier, blocking every shrink.
        margin_u = int(_TIER_HYSTERESIS_SIGMA * std_u)
        margin_t = int(_TIER_HYSTERESIS_SIGMA * std_t)
        candidate = self.capacity_policy.key_for(
            u_values[index] + margin_u,
            t_values[index] + margin_t,
        )
        current = CapacityKey(self._whole_u_capacity, self._whole_t_capacity)
        if candidate == current:
            return
        if (
            candidate.u_capacity > current.u_capacity
            or candidate.t_capacity > current.t_capacity
        ):
            return
        # The fixed-capacity builder needs at least one padding undirected
        # pair (u == u_capacity is an overflow), even with min_pad_u == 0.
        if (
            candidate.u_capacity
            < u_values[-1] + max(self.capacity_policy.min_pad_u, 1)
            or candidate.t_capacity < t_values[-1] + self.capacity_policy.min_pad_t
        ):
            return
        self._tier_switch_down += 1
        self._switch_whole_step_tier(candidate)

    def _grow_whole_step_runner(self, overflow: FixedCapacityOverflow) -> None:
        assert self.whole_step_runner is not None
        assert self.capacity_policy is not None
        old_runner = self.whole_step_runner
        old_state = old_runner.state
        if not _WHOLE_STEP_TIER_CACHE:
            # Pre-tier-cache ordering: stats accumulate before the
            # un-growable re-raise below (behavior-identical default path).
            self._accumulate_whole_step_stats(old_runner)
        required_u = (overflow.edge_count + 1) // 2
        self.capacity_policy.record_overflow(required_u, overflow.triplet_count)
        grown = self.capacity_policy.grow(
            CapacityKey(self._whole_u_capacity, self._whole_t_capacity),
            required_u,
            overflow.triplet_count,
        )
        if (
            grown.u_capacity <= self._whole_u_capacity
            and grown.t_capacity <= self._whole_t_capacity
        ):
            raise overflow
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_switch_up += 1
            self._tier_last_up_at = self._tier_replay_counter
            self._switch_whole_step_tier(grown)
            return
        self._whole_u_capacity = grown.u_capacity
        self._whole_t_capacity = grown.t_capacity
        self.whole_step_recaptures += 1
        self.whole_step_capacity_history.append(
            (self._whole_u_capacity, self._whole_t_capacity)
        )
        self.whole_step_runner = self._make_whole_step_runner(
            positions=old_state.positions,
            momenta=old_state.momenta,
            forces=old_state.forces,
        )
        self._sync_whole_state()
        del old_runner
        gc.collect()

    def _replay_whole_step(self) -> None:
        assert self.whole_step_runner is not None
        while True:
            try:
                self.whole_step_runner.replay(strict=True)
                break
            except FixedCapacityOverflow as overflow:
                self._grow_whole_step_runner(overflow)
        assert self.capacity_policy is not None
        self.capacity_policy.observe(
            self.whole_step_runner.last_edge_count // 2,
            self.whole_step_runner.last_triplet_count,
            CapacityKey(self._whole_u_capacity, self._whole_t_capacity),
        )
        if _WHOLE_STEP_TIER_CACHE:
            self._tier_replay_counter += 1
            self._tier_demand_u.append(self.whole_step_runner.last_edge_count // 2)
            self._tier_demand_t.append(self.whole_step_runner.last_triplet_count)
            self._tier_check_counter += 1
            if self._tier_check_counter >= _TIER_CHECK_INTERVAL:
                self._tier_check_counter = 0
                self._maybe_switch_down_tier()
        self._sync_whole_state()

    @scoped_call
    def build_graph(self, positions: Tensor):
        return self.graph_builder.build(positions)

    @scoped_call
    def evaluate(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        graph = self.build_graph(positions)
        if self.runner is not None:
            out, n_real = self.runner.run(graph)
            forces = out["f"][0][:n_real]
        else:
            out = self.model(
                [graph], task=self.force_task, is_training=False,
                handle_isolated_atoms=not self.validate_isolated_atoms,
            )
            forces = out["f"][0] if isinstance(out["f"], (list, tuple)) else out["f"]

        energy = out["e"].reshape(-1)[0]
        if self.model.is_intensive:
            energy = energy * self.n_atoms
        return forces.detach(), energy.detach()

    def force_fn(self, positions: Tensor) -> Tensor:
        forces, energy = self.evaluate(positions)
        self.state.potential_energy = energy
        return forces

    def _ensure_forces(self) -> None:
        if self.state.forces is None:
            if self.stateful_integrator is not None:
                # F(q_0) at the float64 positions (the builder casts to fp32)
                forces = self.force_fn(self.stateful_integrator.positions)
                self.stateful_integrator.set_forces(
                    forces, energy=self.state.potential_energy.reshape(1)
                )
                self.state.forces = forces
            else:
                self.state.forces = self.force_fn(self.state.positions)

    def _kinetic_energy(self) -> Tensor:
        if self.stateful_integrator is not None:
            return self.stateful_integrator.kinetic_energy()
        return self.integrator.kinetic_energy(self.state.momenta)

    def _log_current(self) -> None:
        if not self.logger.enabled:
            return
        self._ensure_forces()
        assert self.state.forces is not None
        assert self.state.potential_energy is not None
        # a stateful integrator's float64 state, else the float32 state
        source = (
            self.stateful_integrator
            if self.stateful_integrator is not None
            else self.state
        )
        extras = None
        if isinstance(self.integrator, GPUNoseHooverChainNVT):
            extras = {
                "nhc_eta": self.integrator.eta,
                "nhc_p_eta": self.integrator.p_eta,
                "nhc_conserved_energy": self.integrator.conserved_energy(),
            }
        self.logger.enqueue(
            step=self.nsteps,
            time_ps=self.nsteps * self.timestep / 1000.0,
            positions=source.positions,
            momenta=source.momenta,
            forces=self.state.forces,
            cell=self.cell,
            potential_energy=self.state.potential_energy,
            kinetic_energy=self._kinetic_energy(),
            temperature=self.temperature(),
            extras=extras,
        )

    def _log_initial_if_needed(self) -> None:
        if self.logger.enabled and not self._logged_initial:
            self._log_current()
            self._logged_initial = True

    @scoped_call
    def precapture(self, steps: int = 0, proactive: bool = True) -> None:
        if self.whole_step_runner is not None:
            return
        if self.runner is None:
            return
        if steps > 0 and self.stateful_integrator is not None:
            integrator = self.stateful_integrator
            saved = integrator.snapshot()
            try:
                with self.runner.exact_capacity_capture():
                    forces, energy = self.evaluate(integrator.positions)
                    integrator.set_forces(forces, energy=energy.reshape(1))
                    for _ in range(steps):
                        forces, energy = self.evaluate(integrator.step_pre_force())
                        integrator.step_post_force(forces, energy=energy.reshape(1))
                sample_positions = integrator.positions.to(torch.float32)
            finally:
                integrator.load_snapshot(saved)
        elif steps > 0:
            positions = self.state.positions.clone()
            momenta = self.state.momenta.clone()
            with self.runner.exact_capacity_capture():
                forces, _ = self.evaluate(positions)
                force_fn = lambda pos: self.evaluate(pos)[0]
                for _ in range(steps):
                    positions, momenta, forces = self.integrator.step(
                        positions, momenta, forces, force_fn
                    )
            sample_positions = positions
        else:
            sample_positions = self.state.positions
        self.runner.precapture(self.build_graph(sample_positions), proactive=proactive)

    @scoped_call
    def step(self) -> GPUMDState:
        self._log_initial_if_needed()
        if self.whole_step_runner is not None:
            self._replay_whole_step()
            self.nsteps += 1
            if self.logger.should_log(self.nsteps):
                self._log_current()
            return self.state
        self._ensure_forces()
        if self.stateful_integrator is not None:
            self._stateful_step()
        else:
            positions, momenta, forces = self.integrator.step(
                self.state.positions,
                self.state.momenta,
                self.state.forces,
                self.force_fn,
            )
            self.state = GPUMDState(
                positions=positions,
                momenta=momenta,
                forces=forces,
                potential_energy=self.state.potential_energy,
            )
        self.nsteps += 1
        if self.logger.should_log(self.nsteps):
            self._log_current()
        return self.state

    def _stateful_step(self) -> None:
        """Dynamic (eager or model-graph) step of a stateful integrator:
        pre-force half -> one force evaluation -> post-force half."""
        integrator = self.stateful_integrator
        positions = integrator.step_pre_force()
        forces = self.force_fn(positions)
        energy = self.state.potential_energy
        integrator.step_post_force(forces, energy=energy.reshape(1))
        self.state = GPUMDState(
            positions=positions.to(torch.float32),
            momenta=integrator.momenta.to(torch.float32),
            forces=forces,
            potential_energy=energy,
        )

    @scoped_call
    def run(self, steps: int) -> GPUMDState:
        for _ in range(int(steps)):
            self.step()
        return self.state

    def temperature(self) -> Tensor:
        if self.stateful_integrator is not None:
            return self.stateful_integrator.temperature()
        return self.integrator.temperature(self.state.momenta)

    def conserved_energy(self) -> Tensor:
        """NHC conserved quantity E_pot + E_kin + E_thermostat (``nvt_nhc``)."""
        if not isinstance(self.integrator, GPUNoseHooverChainNVT):
            raise ValueError("conserved_energy is defined for ensemble='nvt_nhc'")
        self._ensure_forces()
        return self.integrator.conserved_energy()

    @scoped_call
    def whole_step_stats(self) -> dict[str, object] | None:
        if self.whole_step_runner is None:
            return None
        stats: dict[str, object] = self.whole_step_runner.stats()
        for name in (
            "launches",
            "candidate_refreshes",
            "fallback_windows",
            "recovery_replays",
        ):
            stats[name] += self._whole_step_accumulated[name]
        for name in ("max_edge_count", "max_triplet_count", "max_displacement"):
            stats[name] = max(stats[name], self._whole_step_accumulated[name])
        stats["capacity_recaptures"] = self.whole_step_recaptures
        stats["pool_resets"] = self._tier_pool_resets
        stats["capacity_history"] = list(self.whole_step_capacity_history)
        if self.capacity_policy is not None:
            stats["capacity_policy"] = self.capacity_policy.stats()
        if _WHOLE_STEP_TIER_CACHE:
            stats["tier_cache"] = {
                "tiers_cached": len(self._tier_runners),
                "tier_keys": [
                    (key.u_capacity, key.t_capacity) for key in self._tier_lru
                ],
                "switch_up": self._tier_switch_up,
                "switch_down": self._tier_switch_down,
                "captures": self._tier_captures,
                "evictions": self._tier_evictions,
            }
        return stats

    def model_graph_stats(self) -> dict[str, object] | None:
        if self.runner is None:
            return None
        return self.runner.stats()

    def snapshot_atoms(self) -> Atoms:
        atoms = self.atoms_template.copy()
        # the float64 state of a stateful integrator, else the float32 state
        source = (
            self.stateful_integrator
            if self.stateful_integrator is not None
            else self.state
        )
        atoms.set_positions(source.positions.detach().cpu().numpy())
        atoms.set_momenta(source.momenta.detach().cpu().numpy())
        return atoms

    def flush_log(self) -> None:
        self.logger.flush()

    def close(self) -> None:
        self.logger.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
