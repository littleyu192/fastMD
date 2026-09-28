"""Whole-step CUDA Graph execution for fixed-cell MatRIS molecular dynamics."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from ..graph.fixed_capacity_builder import (
    BatchedCandidateRefresher,
    FixedCapacityGraphBuilder,
    fixed_capacity_exceeded,
)
from ..graph.validation import raise_if_isolated_atoms
from ..model.feature_embed import (
    restore_fused_feature_graph_optimizations,
    set_fused_feature_graph_optimizations,
)
from ..model.interaction_block import (
    restore_fused_graph_optimizations,
    set_fused_graph_optimizations,
)
from .gpu_integrator import GPUIntegrator, StatefulIntegrator
from ._config import scoped_call, scoped_init
from ._graph_capture import capture_graph


class FixedCapacityOverflow(RuntimeError):
    def __init__(self, edge_count: int, triplet_count: int) -> None:
        super().__init__(
            "fixed graph capacity exceeded: "
            f"edges={edge_count}, triplets={triplet_count}"
        )
        self.edge_count = edge_count
        self.triplet_count = triplet_count


@dataclass
class WholeStepState:
    positions: Tensor
    momenta: Tensor
    forces: Tensor
    potential_energy: Tensor


@contextmanager
def _fused_capture_mode(enabled: bool = True):
    previous = set_fused_graph_optimizations(enabled)
    previous_feature = set_fused_feature_graph_optimizations(enabled)
    try:
        yield
    finally:
        restore_fused_feature_graph_optimizations(previous_feature)
        restore_fused_graph_optimizations(previous)


class WholeStepGraphRunner:
    """Capture integrator + exact graph re-filter + model force evaluation.

    Candidate cell-list refresh remains an eager rare path. A captured window
    checkpoints its input state and ORs all device-side overflow flags. In
    strict mode an invalid window is rolled back and replayed eagerly with a
    fresh candidate list before its state becomes visible to the caller.
    Isolated atoms are permitted when the model has reference energies unless
    ``validate_isolated_atoms=True``; capacity and candidate guards remain active.

    ``integrator`` is either a :class:`GPUIntegrator` (NVE / Berendsen NVT;
    the runner's float32 buffers are the MD state) or a
    :class:`StatefulIntegrator` such as the float64 Nose-Hoover chain, which
    owns the MD state: each step runs ``step_pre_force`` -> model ->
    ``step_post_force``, the runner's buffers become float32 mirrors, and the
    integrator's backup/restore joins the window transaction and the capture
    warmup snapshot. Runners of different capacity tiers may share one
    stateful integrator; its state is the committed MD state.
    """

    @scoped_init
    def __init__(
        self,
        *,
        model,
        builder: FixedCapacityGraphBuilder,
        integrator: GPUIntegrator | StatefulIntegrator,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None = None,
        task: str = "ef",
        steps_per_launch: int = 1,
        warmup: int = 3,
        proactive_refresh_fraction: float = 0.25,
        enable_model_fusions: bool = True,
        graph_pool=None,
        validate_isolated_atoms: bool = False,
        config=None,
    ) -> None:
        if steps_per_launch <= 0:
            raise ValueError("steps_per_launch must be positive")
        self.model = model
        self.builder = builder
        self.integrator = integrator
        self.stateful_integrator = (
            integrator if isinstance(integrator, StatefulIntegrator) else None
        )
        self.task = task
        self.steps_per_launch = int(steps_per_launch)
        self.warmup = int(warmup)
        self.proactive_refresh_fraction = float(proactive_refresh_fraction)
        self.enable_model_fusions = bool(enable_model_fusions)
        if config is not None and config.optimization_profile != "legacy":
            self.enable_model_fusions = config.optimization_profile != "generic"
        self.validate_isolated_atoms = (
            bool(validate_isolated_atoms)
            or self.model.reference_energy is None
        )
        # Optional shared CUDA graph memory pool. Tier-cached runners replay
        # mutually exclusively, so their capture-time intermediates may share
        # one pool (same discipline as the bucketed model-graph cache).
        self.graph_pool = graph_pool
        if not 0.0 < self.proactive_refresh_fraction < 0.5:
            raise ValueError("proactive_refresh_fraction must be in (0, 0.5)")
        self.n_real = builder.n_real
        self.device = builder.device

        positions = positions.to(device=self.device, dtype=torch.float32)
        momenta = momenta.to(device=self.device, dtype=torch.float32)
        self.positions = positions.clone()
        self.momenta = momenta.clone()
        self.forces = torch.empty_like(self.positions)
        self.potential_energy = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.backup_positions = torch.empty_like(self.positions)
        self.backup_momenta = torch.empty_like(self.momenta)
        self.backup_forces = torch.empty_like(self.forces)
        self.backup_energy = torch.empty_like(self.potential_energy)
        self.window_overflow = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.window_capacity_overflow = torch.zeros(
            1, dtype=torch.bool, device=self.device
        )
        self.window_max_edges = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.window_max_triplets = torch.zeros(
            1, dtype=torch.int64, device=self.device
        )
        self.window_max_isolated = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        self.fallback_windows = 0
        self.candidate_refreshes = 1
        self.recovery_replays = 0
        self.launches = 0
        self.max_edge_count = 0
        self.max_triplet_count = 0
        self.max_displacement_seen = 0.0
        self.last_edge_count = 0
        self.last_triplet_count = 0

        self.builder.refresh_candidates(self.positions)
        self.builder.assert_candidate_capacity()
        with _fused_capture_mode(self.enable_model_fusions):
            initial_forces, initial_energy, initial_status = self._evaluate(
                self.positions,
                rebuild_candidates=False,
            )
            # The builder substitutes the all-sink topology when the initial
            # graph does not fit, so its forces/energy would be silently
            # wrong: reject the capacity like a replay window does.
            if bool(fixed_capacity_exceeded(initial_status, self.builder).item()):
                raise FixedCapacityOverflow(
                    int(initial_status.edge_count.item()),
                    int(initial_status.triplet_count.item()),
                )
            if self.validate_isolated_atoms:
                raise_if_isolated_atoms(initial_status.isolated_atom_count)
            if forces is None:
                self.forces.copy_(initial_forces)
                if self.stateful_integrator is not None:
                    self.stateful_integrator.set_forces(
                        initial_forces.detach(), energy=initial_energy.detach()
                    )
            else:
                self.forces.copy_(forces.to(device=self.device, dtype=torch.float32))
            self.potential_energy.copy_(initial_energy)

        self._capture()

    @scoped_call
    def _evaluate(
        self,
        positions: Tensor,
        *,
        rebuild_candidates: bool,
    ) -> tuple[Tensor, Tensor, object]:
        graph, status = self.builder.build(
            positions,
            rebuild_candidates=rebuild_candidates,
        )
        out = self.model(
            [graph],
            task=self.task,
            is_training=False,
            n_real=self.n_real,
        )
        forces = out["f"][0] if isinstance(out["f"], (list, tuple)) else out["f"]
        forces = forces[: self.n_real]
        energy = out["e"].reshape(-1)[:1]
        if self.model.is_intensive:
            energy = energy * self.n_real
        return forces, energy, status

    def _step_body(self, *, rebuild_candidates: bool) -> object:
        if self.stateful_integrator is not None:
            return self._stateful_step_body(rebuild_candidates=rebuild_candidates)
        momenta = self.integrator.scale_velocities(self.momenta)
        momenta = momenta + 0.5 * self.integrator.dt * self.forces
        if self.integrator.fix_com:
            momenta = momenta - momenta.sum(dim=0, keepdim=True) / float(
                momenta.shape[0]
            )
        positions_new = (
            self.positions
            + self.integrator.dt * momenta / self.integrator.masses
        )
        forces, energy, status = self._evaluate(
            positions_new,
            rebuild_candidates=rebuild_candidates,
        )
        momenta = momenta + 0.5 * self.integrator.dt * forces
        self.positions.copy_(positions_new)
        self.momenta.copy_(momenta)
        self.forces.copy_(forces.detach())
        self.potential_energy.copy_(energy.detach())
        return status

    def _stateful_step_body(self, *, rebuild_candidates: bool) -> object:
        integrator = self.stateful_integrator
        positions = integrator.step_pre_force()
        forces, energy, status = self._evaluate(
            positions,
            rebuild_candidates=rebuild_candidates,
        )
        with torch.no_grad():
            forces = forces.detach()
            energy = energy.detach()
            integrator.step_post_force(forces, energy=energy)
            # float32 mirrors read by replay / refresh / fallback / stats
            self.positions.copy_(positions)
            self.momenta.copy_(integrator.momenta)
            self.forces.copy_(forces)
            self.potential_energy.copy_(energy)
        return status

    def _window_body(self) -> None:
        if self.stateful_integrator is not None:
            self.stateful_integrator.backup()
        self.backup_positions.copy_(self.positions)
        self.backup_momenta.copy_(self.momenta)
        self.backup_forces.copy_(self.forces)
        self.backup_energy.copy_(self.potential_energy)
        self.window_overflow.zero_()
        self.window_capacity_overflow.zero_()
        self.window_max_edges.zero_()
        self.window_max_triplets.zero_()
        self.window_max_isolated.zero_()
        for _ in range(self.steps_per_launch):
            status = self._step_body(rebuild_candidates=False)
            capacity_overflow = fixed_capacity_exceeded(status, self.builder)
            overflow = self.window_overflow | status.overflow
            if self.validate_isolated_atoms:
                overflow = overflow | (status.isolated_atom_count > 0)
            self.window_overflow.copy_(overflow)
            self.window_capacity_overflow.copy_(
                self.window_capacity_overflow | capacity_overflow.reshape(1)
            )
            self.window_max_edges.copy_(
                torch.maximum(self.window_max_edges, status.edge_count)
            )
            self.window_max_triplets.copy_(
                torch.maximum(self.window_max_triplets, status.triplet_count)
            )
            self.window_max_isolated.copy_(
                torch.maximum(
                    self.window_max_isolated,
                    status.isolated_atom_count,
                )
            )

    def _restore_backup(self) -> None:
        self.positions.copy_(self.backup_positions)
        self.momenta.copy_(self.backup_momenta)
        self.forces.copy_(self.backup_forces)
        self.potential_energy.copy_(self.backup_energy)
        if self.stateful_integrator is not None:
            self.stateful_integrator.restore()

    @scoped_call
    def _capture(self) -> None:
        saved_positions = self.positions.clone()
        saved_momenta = self.momenta.clone()
        saved_forces = self.forces.clone()
        saved_energy = self.potential_energy.clone()
        # warmup windows advance a stateful integrator's own state as well
        integrator = self.stateful_integrator
        saved_integrator = integrator.snapshot() if integrator is not None else None
        with _fused_capture_mode(self.enable_model_fusions):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self._window_body()
                    self.positions.copy_(saved_positions)
                    self.momenta.copy_(saved_momenta)
                    self.forces.copy_(saved_forces)
                    self.potential_energy.copy_(saved_energy)
                    if integrator is not None:
                        integrator.load_snapshot(saved_integrator)
            torch.cuda.current_stream().wait_stream(side)
            self.positions.copy_(saved_positions)
            self.momenta.copy_(saved_momenta)
            self.forces.copy_(saved_forces)
            self.potential_energy.copy_(saved_energy)
            if integrator is not None:
                integrator.load_snapshot(saved_integrator)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with capture_graph(self.graph, pool=self.graph_pool):
                self._window_body()
        self.positions.copy_(saved_positions)
        self.momenta.copy_(saved_momenta)
        self.forces.copy_(saved_forces)
        self.potential_energy.copy_(saved_energy)
        if integrator is not None:
            integrator.load_snapshot(saved_integrator)
        torch.cuda.synchronize()

    @scoped_call
    def _fallback(self) -> None:
        self._restore_backup()
        if bool(self.window_capacity_overflow.item()):
            edge_count = int(self.window_max_edges.item())
            triplet_count = int(self.window_max_triplets.item())
            raise FixedCapacityOverflow(edge_count, triplet_count)
        self.builder.refresh_candidates(self.positions)
        self.builder.assert_candidate_capacity()
        self.candidate_refreshes += 1
        self.graph.replay()
        self.recovery_replays += 1
        if bool(self.window_capacity_overflow.item()):
            self._restore_backup()
            raise FixedCapacityOverflow(
                int(self.window_max_edges.item()),
                int(self.window_max_triplets.item()),
            )
        if self.validate_isolated_atoms:
            isolated_count = int(self.window_max_isolated.item())
            if isolated_count:
                self._restore_backup()
                raise_if_isolated_atoms(isolated_count)
        if bool(self.window_overflow.item()):
            self._restore_backup()
            with _fused_capture_mode(self.enable_model_fusions):
                for _ in range(self.steps_per_launch):
                    status = self._step_body(rebuild_candidates=True)
                    capacity_overflow = bool(
                        fixed_capacity_exceeded(status, self.builder).item()
                    )
                    if capacity_overflow:
                        self._restore_backup()
                        raise FixedCapacityOverflow(
                            int(status.edge_count.item()),
                            int(status.triplet_count.item()),
                        )
                    if self.validate_isolated_atoms:
                        isolated_count = int(status.isolated_atom_count.item())
                        if isolated_count:
                            self._restore_backup()
                            raise_if_isolated_atoms(isolated_count)
        self.fallback_windows += 1

    @scoped_call
    def replay(self, *, strict: bool = True) -> WholeStepState:
        self.graph.replay()
        self.launches += 1
        if strict and bool(self.window_overflow.item()):
            self._fallback()
        if strict:
            self.last_edge_count = int(self.builder.edge_offsets[-1].item())
            self.last_triplet_count = int(self.builder.pair_offsets[-1].item())
            self.max_edge_count = max(self.max_edge_count, self.last_edge_count)
            self.max_triplet_count = max(
                self.max_triplet_count, self.last_triplet_count
            )
            displacement = float(self.builder._max_displacement.item())
            self.max_displacement_seen = max(
                self.max_displacement_seen,
                displacement,
            )
            refresh_threshold = self.builder.skin * self.proactive_refresh_fraction
            if displacement >= refresh_threshold:
                self.builder.refresh_candidates(self.positions)
                self.candidate_refreshes += 1
        return self.state

    @property
    def state(self) -> WholeStepState:
        return WholeStepState(
            positions=self.positions,
            momenta=self.momenta,
            forces=self.forces,
            potential_energy=self.potential_energy,
        )

    @torch.no_grad()
    def reset(
        self,
        *,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor,
        potential_energy: Tensor,
        reset_counters: bool = True,
    ) -> None:
        """Reset persistent state without changing captured tensor addresses.

        With a stateful integrator the arguments are the float32 mirrors of
        its (already current) state; the integrator state is not changed.
        """
        self.positions.copy_(positions)
        self.momenta.copy_(momenta)
        self.forces.copy_(forces)
        self.potential_energy.copy_(potential_energy.reshape(1))
        self.builder.refresh_candidates(self.positions)
        if reset_counters:
            self.fallback_windows = 0
            self.candidate_refreshes = 1
            self.recovery_replays = 0
            self.launches = 0
            self.max_edge_count = 0
            self.max_triplet_count = 0
            self.max_displacement_seen = 0.0
            self.last_edge_count = 0
            self.last_triplet_count = 0

    def stats(self) -> dict[str, int | float]:
        return {
            "steps_per_launch": self.steps_per_launch,
            "launches": self.launches,
            "candidate_refreshes": self.candidate_refreshes,
            "fallback_windows": self.fallback_windows,
            "recovery_replays": self.recovery_replays,
            "max_edge_count": self.max_edge_count,
            "max_triplet_count": self.max_triplet_count,
            "max_displacement": self.max_displacement_seen,
            "u_capacity": self.builder.u_capacity,
            "t_capacity": self.builder.t_capacity,
        }


class BatchedWholeStepGraphRunner:
    """Capture multiple homogeneous MD replicas in one whole-step graph.

    Each replica retains an independent candidate cache, capacity status,
    thermostat, and state. The model sees one graph batch so large sparse
    motifs and GEMMs amortize launch latency across replicas.
    Isolated-atom validation follows :class:`WholeStepGraphRunner`.
    """

    @scoped_init
    def __init__(
        self,
        *,
        model,
        builders: Sequence[FixedCapacityGraphBuilder],
        integrator: GPUIntegrator,
        positions: Tensor,
        momenta: Tensor,
        forces: Tensor | None = None,
        task: str = "ef",
        steps_per_launch: int = 1,
        warmup: int = 3,
        proactive_refresh_fraction: float = 0.25,
        parallel_graph_build: bool = True,
        validate_isolated_atoms: bool = False,
        enable_model_fusions: bool = True,
        config=None,
    ) -> None:
        if not builders:
            raise ValueError("builders must not be empty")
        if steps_per_launch <= 0:
            raise ValueError("steps_per_launch must be positive")
        if isinstance(integrator, StatefulIntegrator):
            raise ValueError(
                "BatchedWholeStepGraphRunner supports GPUIntegrator (nve/nvt) only"
            )
        self.model = model
        self.builders = list(builders)
        self.integrator = integrator
        self.task = task
        self.batch_size = len(self.builders)
        self.steps_per_launch = int(steps_per_launch)
        self.warmup = int(warmup)
        self.parallel_graph_build = bool(parallel_graph_build)
        self.enable_model_fusions = bool(enable_model_fusions)
        if config is not None and config.optimization_profile != "legacy":
            self.enable_model_fusions = config.optimization_profile != "generic"
        self.validate_isolated_atoms = (
            bool(validate_isolated_atoms)
            or self.model.reference_energy is None
        )
        self.proactive_refresh_fraction = float(proactive_refresh_fraction)
        if not 0.0 < self.proactive_refresh_fraction < 0.5:
            raise ValueError("proactive_refresh_fraction must be in (0, 0.5)")

        self.device = self.builders[0].device
        self.n_real = self.builders[0].n_real
        if any(builder.device != self.device for builder in self.builders):
            raise ValueError("all builders must use the same device")
        if any(builder.n_real != self.n_real for builder in self.builders):
            raise ValueError("batched runner currently requires equal atom counts")
        expected_shape = (self.batch_size, self.n_real, 3)
        if tuple(positions.shape) != expected_shape:
            raise ValueError(
                f"positions must have shape {expected_shape}, got {tuple(positions.shape)}"
            )
        if tuple(momenta.shape) != expected_shape:
            raise ValueError(
                f"momenta must have shape {expected_shape}, got {tuple(momenta.shape)}"
            )

        self.real_counts = torch.full(
            (self.batch_size,),
            self.n_real,
            dtype=torch.int32,
            device=self.device,
        )
        self.positions = positions.to(self.device, torch.float32).clone()
        self.momenta = momenta.to(self.device, torch.float32).clone()
        self.forces = torch.empty_like(self.positions)
        self.potential_energy = torch.zeros(
            self.batch_size, dtype=torch.float32, device=self.device
        )
        self.backup_positions = torch.empty_like(self.positions)
        self.backup_momenta = torch.empty_like(self.momenta)
        self.backup_forces = torch.empty_like(self.forces)
        self.backup_energy = torch.empty_like(self.potential_energy)
        self.window_overflow = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.window_capacity_overflow = torch.zeros_like(self.window_overflow)
        self.window_max_edges = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.window_max_triplets = torch.zeros_like(self.window_max_edges)
        self.window_max_isolated = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self.fallback_windows = 0
        self.candidate_refreshes = 1
        self.recovery_replays = 0
        self.launches = 0
        self.max_edge_counts = [0] * self.batch_size
        self.max_triplet_counts = [0] * self.batch_size
        self.max_displacements = [0.0] * self.batch_size
        self.last_edge_counts = [0] * self.batch_size
        self.last_triplet_counts = [0] * self.batch_size
        self.candidate_refresher = BatchedCandidateRefresher(self.builders)
        self.build_streams = (
            [torch.cuda.Stream(device=self.device) for _ in self.builders]
            if self.parallel_graph_build and self.batch_size > 1
            else []
        )

        self.candidate_refresher.refresh(self.positions)
        for builder in self.builders:
            builder.assert_candidate_capacity()
        with _fused_capture_mode(self.enable_model_fusions):
            initial_forces, initial_energy, initial_statuses = self._evaluate(
                self.positions,
                rebuild_candidates=False,
            )
            if any(
                bool(fixed_capacity_exceeded(status, builder).item())
                for status, builder in zip(initial_statuses, self.builders)
            ):
                raise FixedCapacityOverflow(
                    max(int(status.edge_count.item()) for status in initial_statuses),
                    max(int(status.triplet_count.item()) for status in initial_statuses),
                )
            if self.validate_isolated_atoms:
                raise_if_isolated_atoms(
                    torch.cat(
                        [status.isolated_atom_count for status in initial_statuses]
                    )
                )
            if forces is None:
                self.forces.copy_(initial_forces)
            else:
                self.forces.copy_(forces.to(self.device, torch.float32))
            self.potential_energy.copy_(initial_energy)
        self._capture()

    @scoped_call
    def _evaluate(
        self,
        positions: Tensor,
        *,
        rebuild_candidates: bool,
    ) -> tuple[Tensor, Tensor, list[object]]:
        graphs = []
        statuses = []
        use_parallel_build = self.build_streams and not rebuild_candidates
        if use_parallel_build:
            origin_stream = torch.cuda.current_stream(self.device)
            for replica, (builder, stream) in enumerate(
                zip(self.builders, self.build_streams)
            ):
                stream.wait_stream(origin_stream)
                with torch.cuda.stream(stream):
                    graph, status = builder.build(
                        positions[replica],
                        rebuild_candidates=False,
                    )
                graphs.append(graph)
                statuses.append(status)
            for stream in self.build_streams:
                origin_stream.wait_stream(stream)
        else:
            for replica, builder in enumerate(self.builders):
                graph, status = builder.build(
                    positions[replica],
                    rebuild_candidates=rebuild_candidates,
                )
                graphs.append(graph)
                statuses.append(status)
        output = self.model(
            graphs,
            task=self.task,
            is_training=False,
            n_real=self.real_counts,
        )
        force_parts = output["f"]
        if not isinstance(force_parts, (list, tuple)):
            raise RuntimeError("batched model did not return per-graph forces")
        forces = torch.stack(
            [force[: self.n_real] for force in force_parts],
            dim=0,
        )
        energy = output["e"].reshape(self.batch_size)
        if self.model.is_intensive:
            energy = energy * self.real_counts
        return forces, energy, statuses

    def _step_body(self, *, rebuild_candidates: bool) -> list[object]:
        momenta = self.integrator.scale_velocities(self.momenta)
        momenta = momenta + 0.5 * self.integrator.dt * self.forces
        if self.integrator.fix_com:
            momenta = momenta - momenta.sum(dim=1, keepdim=True) / float(
                self.n_real
            )
        positions_new = (
            self.positions
            + self.integrator.dt * momenta / self.integrator.masses
        )
        forces, energy, statuses = self._evaluate(
            positions_new,
            rebuild_candidates=rebuild_candidates,
        )
        momenta = momenta + 0.5 * self.integrator.dt * forces
        self.positions.copy_(positions_new)
        self.momenta.copy_(momenta)
        self.forces.copy_(forces.detach())
        self.potential_energy.copy_(energy.detach())
        return statuses

    def _window_body(self) -> None:
        self.backup_positions.copy_(self.positions)
        self.backup_momenta.copy_(self.momenta)
        self.backup_forces.copy_(self.forces)
        self.backup_energy.copy_(self.potential_energy)
        self.window_overflow.zero_()
        self.window_capacity_overflow.zero_()
        self.window_max_edges.zero_()
        self.window_max_triplets.zero_()
        self.window_max_isolated.zero_()
        for _ in range(self.steps_per_launch):
            statuses = self._step_body(rebuild_candidates=False)
            for replica, (status, builder) in enumerate(
                zip(statuses, self.builders)
            ):
                capacity_overflow = fixed_capacity_exceeded(status, builder)
                overflow = (
                    self.window_overflow[replica : replica + 1]
                    | status.overflow
                )
                if self.validate_isolated_atoms:
                    overflow = overflow | (status.isolated_atom_count > 0)
                self.window_overflow[replica : replica + 1].copy_(overflow)
                self.window_capacity_overflow[replica : replica + 1].copy_(
                    self.window_capacity_overflow[replica : replica + 1]
                    | capacity_overflow.reshape(1)
                )
                self.window_max_edges[replica : replica + 1].copy_(
                    torch.maximum(
                        self.window_max_edges[replica : replica + 1],
                        status.edge_count,
                    )
                )
                self.window_max_triplets[replica : replica + 1].copy_(
                    torch.maximum(
                        self.window_max_triplets[replica : replica + 1],
                        status.triplet_count,
                    )
                )
                self.window_max_isolated[replica : replica + 1].copy_(
                    torch.maximum(
                        self.window_max_isolated[replica : replica + 1],
                        status.isolated_atom_count,
                    )
                )

    def _restore_backup(self) -> None:
        self.positions.copy_(self.backup_positions)
        self.momenta.copy_(self.backup_momenta)
        self.forces.copy_(self.backup_forces)
        self.potential_energy.copy_(self.backup_energy)

    @scoped_call
    def _capture(self) -> None:
        saved_positions = self.positions.clone()
        saved_momenta = self.momenta.clone()
        saved_forces = self.forces.clone()
        saved_energy = self.potential_energy.clone()
        with _fused_capture_mode(self.enable_model_fusions):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self._window_body()
                    self.positions.copy_(saved_positions)
                    self.momenta.copy_(saved_momenta)
                    self.forces.copy_(saved_forces)
                    self.potential_energy.copy_(saved_energy)
            torch.cuda.current_stream().wait_stream(side)
            self.positions.copy_(saved_positions)
            self.momenta.copy_(saved_momenta)
            self.forces.copy_(saved_forces)
            self.potential_energy.copy_(saved_energy)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with capture_graph(self.graph):
                self._window_body()
        self.positions.copy_(saved_positions)
        self.momenta.copy_(saved_momenta)
        self.forces.copy_(saved_forces)
        self.potential_energy.copy_(saved_energy)
        torch.cuda.synchronize()

    @scoped_call
    def _fallback(self) -> None:
        self._restore_backup()
        if bool(self.window_capacity_overflow.any().item()):
            raise FixedCapacityOverflow(
                int(self.window_max_edges.max().item()),
                int(self.window_max_triplets.max().item()),
            )
        self.candidate_refresher.refresh(self.positions)
        for builder in self.builders:
            builder.assert_candidate_capacity()
        self.candidate_refreshes += 1
        self.graph.replay()
        self.recovery_replays += 1
        if bool(self.window_capacity_overflow.any().item()):
            self._restore_backup()
            raise FixedCapacityOverflow(
                int(self.window_max_edges.max().item()),
                int(self.window_max_triplets.max().item()),
            )
        if self.validate_isolated_atoms:
            isolated_count = int(self.window_max_isolated.sum().item())
            if isolated_count:
                self._restore_backup()
                raise_if_isolated_atoms(isolated_count)
        if bool(self.window_overflow.any().item()):
            self._restore_backup()
            with _fused_capture_mode(self.enable_model_fusions):
                for _ in range(self.steps_per_launch):
                    statuses = self._step_body(rebuild_candidates=True)
                    capacity_overflow = any(
                        bool(fixed_capacity_exceeded(status, builder).item())
                        for status, builder in zip(statuses, self.builders)
                    )
                    if capacity_overflow:
                        self._restore_backup()
                        raise FixedCapacityOverflow(
                            max(int(status.edge_count.item()) for status in statuses),
                            max(
                                int(status.triplet_count.item())
                                for status in statuses
                            ),
                        )
                    if self.validate_isolated_atoms:
                        isolated_count = sum(
                            int(status.isolated_atom_count.item())
                            for status in statuses
                        )
                        if isolated_count:
                            self._restore_backup()
                            raise_if_isolated_atoms(isolated_count)
        self.fallback_windows += 1

    @scoped_call
    def replay(self, *, strict: bool = True) -> WholeStepState:
        self.graph.replay()
        self.launches += 1
        if strict and bool(self.window_overflow.any().item()):
            self._fallback()
        if strict:
            refresh_thresholds = [
                builder.skin * self.proactive_refresh_fraction
                for builder in self.builders
            ]
            refresh_needed = False
            for replica, builder in enumerate(self.builders):
                edge_count = int(builder.edge_offsets[-1].item())
                triplet_count = int(builder.pair_offsets[-1].item())
                displacement = float(builder._max_displacement.item())
                self.last_edge_counts[replica] = edge_count
                self.last_triplet_counts[replica] = triplet_count
                self.max_edge_counts[replica] = max(
                    self.max_edge_counts[replica], edge_count
                )
                self.max_triplet_counts[replica] = max(
                    self.max_triplet_counts[replica], triplet_count
                )
                self.max_displacements[replica] = max(
                    self.max_displacements[replica], displacement
                )
                if displacement >= refresh_thresholds[replica]:
                    refresh_needed = True
            if refresh_needed:
                self.candidate_refresher.refresh(self.positions)
                self.candidate_refreshes += 1
        return self.state

    @property
    def state(self) -> WholeStepState:
        return WholeStepState(
            positions=self.positions,
            momenta=self.momenta,
            forces=self.forces,
            potential_energy=self.potential_energy,
        )

    def stats(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "steps_per_launch": self.steps_per_launch,
            "parallel_graph_build": self.parallel_graph_build,
            "launches": self.launches,
            "candidate_refreshes": self.candidate_refreshes,
            "fallback_windows": self.fallback_windows,
            "recovery_replays": self.recovery_replays,
            "max_edge_counts": list(self.max_edge_counts),
            "max_triplet_counts": list(self.max_triplet_counts),
            "max_displacements": list(self.max_displacements),
            "u_capacities": [builder.u_capacity for builder in self.builders],
            "t_capacities": [builder.t_capacity for builder in self.builders],
        }
