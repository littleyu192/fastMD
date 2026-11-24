"""P=1 short circuit: one CUDA rank runs the single-GPU engine behind the distributed API.

``DistributedMatRISEvaluator`` delegates here when world_size == 1 and the
model is on CUDA (docs/unified_engine_design.md §7). Graph construction, operator selection,
checkpoint semantics and CUDA-graph buckets (incl. proactive precapture) are
those of ``GPUResidentMolecularDynamics``; enable_compile models are rejected
(see ``admit``). This adapter only keeps the distributed
contract: AtomState in and out, preflight conditions and messages, the
per-evaluation immutable-model guard, DistributedPrediction fields and
additive wall-time stats. Its own host syncs are the packed preflight flags
and the final finiteness flag; the builder's are the single-GPU ones.
"""
from __future__ import annotations

from dataclasses import replace
import time

import torch
from torch import Tensor

from fastmd.engine.entry_config import kernel_policy_summary
from fastmd.graph.gpu_graph_builder import TensorGraphBuilder, op_available
from fastmd.graph.neighbor_list_backend import UNAVAILABLE_REASON

from .auto_execution import AutoExecutionState
from .evaluator import DistributedPrediction, collective_check
from .execution import OverlappedModelGuard
from ..engine.memory_admission import no_bound_reason
from .partition import AtomState
from .watchdog import enter_long_phase, exit_long_phase

# wall_breakdown() buckets that have no single-GPU counterpart.
_GRAPH_TIMING = ("halo_seconds", "build_local_seconds", "route_triplet_seconds")
_TAPE_TIMING = ("forward_input_ms", "forward_local_ms", "forward_exchange_ms", "backward_setup_ms",
                "backward_local_ms", "backward_exchange_ms", "backward_finish_ms")


class SingleRankEngine:
    """Single-GPU E/F for one CUDA rank; the evaluator owns admission and epochs."""

    @staticmethod
    def admit(model, strategy) -> None:
        """Requirements beyond the distributed admission; never a silent fallback.

        The model's own ones are its exchange strategy's (``admit_single_rank``: MatRIS rejects
        enable_compile models and converter cutoffs other than model.config's)."""
        if not op_available:
            raise ValueError("the single-rank engine needs nvalchemiops for TensorGraphBuilder "
                             f"(CUDA runs need it at any rank count): {UNAVAILABLE_REASON}")
        strategy.admit_single_rank(model)

    def __init__(self, model, partition, config, device, *, requested: dict, profile_source: str,
                 isolated_atoms: str, program, context, auto_execution=None, admission=None, strategy):
        self.admit(model, strategy)
        if config.execution not in ("eager", "model_graph"):
            raise ValueError("the single-rank engine runs execution='eager' or 'model_graph'")
        if config.execution == "model_graph" and config.checkpoint_enabled is not False:
            raise ValueError("the single-rank engine rejects checkpoint with execution='model_graph'")
        self.model, self.partition, self.config = model, partition, config
        self.device = torch.device(device)
        self.requested, self.profile_source = dict(requested), profile_source
        self.cutoffs = (float(model.config["pairwise_cutoff"]), float(model.config["three_body_cutoff"]))
        # GPUResidentMolecularDynamics' isolated-atom mapping.
        self.check_isolated_atoms = isolated_atoms == "error" or model.reference_energy is None
        # DomainPartition admits only full periodicity.
        self._pbc = torch.ones(3, dtype=torch.bool, device=self.device)
        # Nothing here refers back to the evaluator (its preflight is passed
        # per call): a dropped evaluator frees graphs and builder by refcount.
        self._context = context
        self.guard = OverlappedModelGuard(model, program)
        self.builder = self._reference = None
        self.last_graph = None
        # ``execution="auto"`` (InferenceConfig.execution=None, resolved to the
        # eager default): AutoExecution's single-rank threshold on this rank's
        # triplet rows, decided at the first evaluation, with its hysteresis.
        self.auto = None
        self.execution = config.execution
        # MemoryAdmission (None: MATRIS_MEMORY_ADMISSION=off): the first
        # evaluation and every evaluation that would capture are checked.
        self.admission = admission
        if auto_execution is not None:
            self.auto = AutoExecutionState(
                auto_execution, device_type=self.device.type, single_rank=True,
                fixed=("checkpoint on/auto runs eager (not validated inside CUDA graph capture)"
                       if config.checkpoint_enabled is not False else
                       no_bound_reason(admission.memory, "architecture outside the K6 calibration")
                       + ": eager (no memory bound for a capture)"
                       if admission is not None and not admission.calibrated else None),
                # Legacy and topology single-engine model_graph is faster than eager at every measured size
                # (topology, 2026-09-30: 1,344-21,504 RDX atoms, -17..-67%); only while K6 bounds the model.
                prefer_model_graph=(admission is not None and admission.calibrated
                                    and config.optimization_profile in ("legacy", "topology")))
            self.execution = self.auto.mode  # None until the first evaluation decides
        # The stress: once an evaluation asked for it, the model_graph runner's task is "efs" (its graphs
        # serve the evaluations without it too).
        self.stress = False
        self.runner = self._new_runner() if self.execution == "model_graph" else None
        self.captures = self.capture_calls = self.replays = self.runner_resets = 0
        self._initial_captures = None
        self._closed = False

    def _new_runner(self):
        from fastmd.execution.cuda_graph import BucketedGraphRunner
        capacity = self.config.capacity
        # Exactly as GPUResidentMolecularDynamics.enable_model_cuda_graph.
        return BucketedGraphRunner(
            self.model, task="efs" if self.stress else "ef",
            enable_model_fusions=self.config.optimization_profile != "generic",
            u_step=capacity.u_step, t_step=capacity.t_step, n_dummy=capacity.n_dummy,
            min_pad_u=capacity.min_pad_u, warmup=capacity.warmup, config=self.config)

    def _drop_graphs(self) -> None:
        """Fresh runner: after a rejected step (its graphs must never replay), an atom-count change or the
        first evaluation with the stress (task "efs")."""
        if self.runner is not None:
            self.runner = None
            self.runner_resets += 1

    def _graphs_bound_to(self, n: int) -> bool:
        """Captured graphs replay a fixed atom count (_PaddedGraphWorkspace.n_real);
        GPU MD never changes it, the distributed API may."""
        runner = self.runner
        return runner is None or not runner.cache or next(iter(runner.cache.values()))["n_real"] == n

    def _runs_model_code(self, graph) -> bool:
        """False only for a pure replay of an already captured bucket.

        A replay executes captured kernels, never Python model code: mutated
        code cannot run, so ``verify()`` alone decides (overlapped).
        """
        runner = self.runner
        if runner is None or not runner.cache:
            return True
        return runner.captures_next(graph)

    def _admit(self, graph, n: int, epoch: int):
        """Memory admission before the first evaluation and before any capture.

        Returns the fitting modes when checked (None otherwise); raises
        MemoryAdmissionError when the requested mode (auto: every mode) does not fit.
        """
        rows = int(graph.line_graph.shape[0])
        first = not self.admission.admitted
        mode = self.execution if self.auto is None else self.auto.propose(rows)
        capture = mode == "model_graph" and (self.runner is None or self.execution != "model_graph"
                                             or not self._graphs_bound_to(n) or self._runs_model_code(graph))
        if not (first or capture):
            return None
        return self.admission.check(("model_graph", "eager"), rows, requested=self.config.execution
                                    if self.auto is None else None,
                                    event=f"epoch {epoch}: " + ("first evaluation" if first else "capture"))

    def _probe(self, order: Tensor, sorted_ids: Tensor, numbers: Tensor):
        """Preflight extra riding the packed download: have the numbers changed?"""
        if sorted_ids.device != self.device:
            return [], None  # preflight reports the device mismatch itself
        numbers = numbers.index_select(0, order)
        builder = self.builder
        if builder is None or builder.n_atoms != len(numbers):
            return [], (order, sorted_ids, numbers)
        return [(numbers != builder.atomic_numbers).any()], (order, sorted_ids, numbers)

    def evaluate(self, state: AtomState, *, epoch: int, entered: float, preflight,
                 stress: bool = False) -> DistributedPrediction:
        """``preflight`` is the evaluator's shared ``_preflight(state, extra)``. ``stress``: also the stress in
        eV/Å³ (the model's GPa through the calculator's conversion); model_graph then captures task "efs"
        from here on."""
        error, changed, carry = preflight(state, self._probe)
        collective_check(self._context, error)
        # migrate_atoms' backend admission (same type and message): builder
        # kernels and CUDA graphs use the current device's stream.
        if self.device.index != torch.cuda.current_device():
            self._context.collective_check("atom records must use the current worker CUDA device")
        n = len(state.ids)
        collective_check(self._context, "empty global system" if n == 0 else None)
        scans, scan_seconds = self.guard.full_scans, self.guard.scan_seconds
        start = time.perf_counter()
        # migrate_atoms at P=1: ID order, int64 IDs/numbers, fresh storage,
        # positions not wrapped, payload rows follow.
        order, sorted_ids, numbers = carry
        state = AtomState(sorted_ids.to(torch.int64), state.positions.index_select(0, order),
                          numbers.to(torch.int64),
                          {key: value.to(self.device).index_select(0, order)
                           for key, value in state.payload.items()})
        migrated = time.perf_counter()
        rebuilt = not changed or bool(changed[0])
        with self.config.scope():
            if rebuilt:
                previous = self.builder
                builder = TensorGraphBuilder(
                    cell=self.partition.cell, atomic_numbers=state.numbers, pbc=self._pbc,
                    atom_graph_cutoff=self.cutoffs[0], line_graph_cutoff=self.cutoffs[1],
                    device=self.device, check_isolated_atoms=self.check_isolated_atoms,
                    max_neighbors=None if previous is None else previous.max_neighbors)
                self.builder, self._reference = builder, None
            graph = self.builder.build(state.positions)
            if stress and not self.stress:
                # The graphs of task "ef" have no stress: a runner of task "efs" from here on, its captures
                # admitted below like any other (the old runner's pool released first).
                self.stress = True
                if self.runner is not None:
                    self._drop_graphs()
                    torch.cuda.empty_cache()
            switch_ms = None
            admitted = time.perf_counter()
            fits = None if self.admission is None else self._admit(graph, n, epoch)
            admission_ms = (time.perf_counter() - admitted) * 1000
            if self.auto is not None:
                switching = time.perf_counter()
                rows = int(graph.line_graph.shape[0])
                if self.auto.observe(epoch, [(n, rows, self.auto.state)], fits=fits):
                    self.execution = self.auto.mode
                    if self.execution == "eager":
                        # Leaving model_graph releases its graphs and pool.
                        self._drop_graphs()
                switch_ms = (time.perf_counter() - switching) * 1000
            graph_run = self.execution == "model_graph"
            if self._closed and graph_run:
                # As DistributedGraphRunner.run_whole's check (same type; it
                # raises locally, without the collective prefix).
                # After the auto decision, so a closed engine never captures.
                self._context.collective_check("model_graph runner is closed")
            graphed = time.perf_counter()
            # The reference energy is fixed while the numbers are: a copy of the
            # per-builder value (callers own each prediction), launched while
            # the device idles after the builder's last sync, not behind the replay.
            reference = None if self._reference is None else self._reference.clone()
            if graph_run and not self._graphs_bound_to(n):
                self._drop_graphs()
            checking = time.perf_counter()
            # Before any model code; after the builder, where the historical
            # program check also ran. A pure replay runs none: verify() decides.
            runs_code = self._runs_model_code(graph)
            if runs_code:
                try:
                    self.guard.precheck()
                except Exception:
                    self._drop_graphs()
                    raise
            prechecked = time.perf_counter()
            try:
                with torch.enable_grad():
                    if not graph_run:
                        out = self.model([graph], task="efs" if stress else "ef", is_training=False,
                                         handle_isolated_atoms=self.config.handle_isolated_atoms)
                        forces = out["f"][0] if isinstance(out["f"], (list, tuple)) else out["f"]
                        forces = forces.detach()
                    else:
                        if self.runner is None:
                            self.runner = self._new_runner()
                        captures = self.runner.captures
                        precaptured = not self.runner.cache
                        # As DistributedGraphRunner: a capture (and its first-use
                        # kernel compiles) may take many steady-state steps and must
                        # not stretch the next step's deadline; a replay stays under it.
                        long_phase = enter_long_phase("model_graph_capture") if runs_code else None
                        try:
                            if precaptured:
                                # As GPUResidentMolecularDynamics.enable_model_cuda_graph:
                                # the next larger U/T buckets too, so MD rarely captures mid-run.
                                self.runner.precapture(graph, proactive=True)
                            out, n_real = self.runner.run(graph)
                        finally:
                            exit_long_phase(long_phase)
                        replayed = time.perf_counter()
                        # Graph outputs are static buffers rewritten by the next replay.
                        forces = out["f"][0][:n_real].detach().clone()
                    if stress:
                        from ase import units
                        # eV/Å³: MatRISCalculator's conversion of the model's GPa.
                        virial = out["s"][0].detach() * units.GPa
                    # The model's energy per atom (the readout's single rounding;
                    # energy / n would round twice), copied: never a static
                    # graph output. The product owns new storage too.
                    per_atom = out["e"].reshape(-1)[0].detach().clone()
                    energy = per_atom * n
                    # Eager: frees the forward's autograd graph now, overlapping
                    # the device (stream-ordered), not after the readout wait.
                    del out
            except Exception as exc:
                self._drop_graphs()
                mutation = self.guard.mutation()
                if mutation is not None:
                    raise mutation from exc
                raise
        forwarded = time.perf_counter()
        try:
            # The full scan overlaps the submitted device work, before readout.
            self.guard.verify()
        except Exception:
            self._drop_graphs()
            raise
        if reference is None:
            # Once per builder, after the guard: it reads a model tensor, which
            # (like model code) must not run before the program is checked.
            table = self.model.reference_energy
            self._reference = (energy.new_zeros(()) if table is None
                               else table.fc.weight[0, state.numbers - 1].sum().reshape(()))
            reference = self._reference.clone()
        # Everything up to the finiteness read is host work queued behind the
        # device; results and stats are discarded if the check fails.
        finite = torch.isfinite(forces).all() & torch.isfinite(energy)
        stats = {
            "engine": "single", "rank": 0, "epoch": epoch,
            "owned_atoms": n, "ghost_atoms": 0,
            "owned_bonds": int(graph.atom_graph.shape[0]),  # directed edges
            "owned_triplets": int(graph.line_graph.shape[0]),
            "preflight_ms": (start - entered) * 1000,
            "precheck_ms": (prechecked - checking) * 1000,
            "migration_ms": (migrated - start) * 1000,
            "graph_ms": (graphed - migrated) * 1000,
            "plan_build_ms": 0.0,
            "forward_and_plan_ms": (forwarded - graphed) * 1000,
            "vjp_ms": 0.0,
            "builder_rebuilt": rebuilt, "builder_max_neighbors": int(self.builder.max_neighbors),
            "timing_semantics": ("host wall intervals; model E/F is an asynchronous submission, GPU "
                                 "completion wait included in readout; precheck_ms (0 for a pure "
                                 "model_graph replay) is inside forward_and_plan_ms, model_stamp_ms "
                                 "inside forward_and_plan_ms/readout_ms"),
            "graph_timing": dict.fromkeys(_GRAPH_TIMING, 0.0),
            "tape_timing": {**dict.fromkeys(_TAPE_TIMING, 0.0), "timing_mode": "single_gpu_no_stage_tape"},
        }
        if fits is not None:
            stats["memory_admission"] = {**self.admission.decisions[-1], "admission_ms": admission_ms,
                                         "timing": "included in graph_ms"}
        if self.auto is not None:
            stats["auto_execution"] = {"mode": self.auto.mode, "decisions": len(self.auto.decisions),
                                       "max_rank_triplet_rows": self.auto.last_max_rows,
                                       "switch_ms": switch_ms, "switch_timing": "included in graph_ms"}
        if graph_run:
            captured = self.runner.captures - captures
            if captured:
                self.capture_calls += 1
                if self._initial_captures is None:
                    self._initial_captures = captured
            self.captures += captured
            self.replays += 1
            stats["model_graph_ms"] = (replayed - prechecked) * 1000
            stats["model_graph"] = {
                "mode": "bucketed", "generation": self.capture_calls, "capture_count": self.captures,
                # Graphs captured after the first proactive set: mid-run bucket
                # misses and re-precaptures after a rejected step, an
                # atom-count change or the switch to task "efs" (all counted
                # in runner_resets).
                "recapture_count": self.captures - (self._initial_captures or 0),
                "replay_count": self.replays, "precaptured_this_call": precaptured,
                "captured_this_call": bool(captured), "capacity_changed": bool(captured),
                "capture_ms": stats["model_graph_ms"] if captured else 0.0, "warmup_ms": None,
                "runner_resets": self.runner_resets, "resident_entries": len(self.runner.cache),
                "runner": self.runner.stats(),
                "timing": ("host wall; a capture call's capture_ms also holds its side-stream warmup "
                           "and first replay; replay completion is included in readout"),
            }
        memory = torch.cuda.memory_stats(self.device)  # one query, not one per key
        stats.update(peak_allocated_bytes=memory.get("allocated_bytes.all.peak", 0),
                     peak_reserved_bytes=memory.get("reserved_bytes.all.peak", 0))
        prediction = DistributedPrediction(state, forces, energy, per_atom, reference, n, epoch, stats,
                                           stress=virial if stress else None)
        collective_check(self._context, None if bool(finite) else "non-finite energy or forces")
        finished = time.perf_counter()
        stats.update(readout_ms=(finished - forwarded) * 1000, total_ms=(finished - start) * 1000,
                     model_stamp_ms=(self.guard.scan_seconds - scan_seconds) * 1000,
                     model_stamp_scans=self.guard.full_scans - scans)
        self.last_graph = graph
        completed = time.perf_counter()
        stats["cleanup_ms"] = (completed - finished) * 1000
        stats["evaluate_wall_ms"] = (completed - entered) * 1000
        return prediction

    def config_summary(self) -> dict:
        pending = self.execution is None
        # Under auto the resolved config keeps the eager default; report the mode run.
        config = self.config if pending else replace(self.config, execution=self.execution)
        graph = self.execution == "model_graph"
        return {
            "engine": "single", "single_rank_engine": "single", "execution": self.execution or "auto",
            "auto_execution": None if self.auto is None else self.auto.summary(),
            "memory_admission": ({"enabled": False} if self.admission is None else self.admission.summary()),
            "requested_profile": self.requested["optimization_profile"],
            "resolved_profile": config.optimization_profile, "profile_source": self.profile_source,
            "requested_checkpoint": self.requested["checkpoint"],
            "effective_checkpoint": config.as_dict()["checkpoint"],
            "isolated_atoms": "error" if self.check_isolated_atoms else "reference",
            "configuration": config.as_dict(), "config_fingerprint": config.fingerprint,
            "kernel_policy": kernel_policy_summary(self.config),
            "forward_implementation": "single_gpu_model",
            "operators": "single-GPU dispatch of the resolved profile (GPUResidentMolecularDynamics kernels)",
            "enable_model_fusions": config.optimization_profile != "generic" if graph else None,
            "topology": ("single-GPU TensorGraphBuilder (nvalchemiops cell list, overflow-checked ragged "
                         "neighbors); rebuilt every evaluation, reused while atomic numbers are unchanged"),
            "execution_schedule": ("pending: execution='auto' chooses eager or model_graph from this rank's "
                                   "triplet rows at the first evaluation" if pending else
                                   "BucketedGraphRunner: one E/F CUDA graph per (U, T) capacity bucket; the "
                                   "first use precaptures the next larger U/T buckets too (as GPU MD), "
                                   "later misses capture on demand; replayed otherwise; an atom-count change "
                                   "drops the graphs and precaptures again" if graph else
                                   "eager model([graph]) under ResolvedConfig.scope()"),
            "model_guard": ("identity/structure precheck before any model code (skipped for a pure CUDA-graph "
                            "replay, which runs none); on a change, callable attributes and parameter/buffer/"
                            "child types vs construction, then the program scan; one full program scan per "
                            "evaluation, overlapped with device work before readout"),
        }

    def close(self) -> None:
        """Release captured graphs and the cached builder."""
        self.runner = self.builder = self._reference = self.last_graph = None
        self._closed = True
