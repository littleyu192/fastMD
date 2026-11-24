"""Coordinated, single-entry CUDA graph of the topology epoch's whole step (P>1 model_graph).

The topology refresh, its capacity agreement and preparation stay outside
capture (``fastmd.distributed.epoch``); the captured step reads persistent
per-layout-key buffers the refresh rewrites in place. The pre-4.2 runner of
the rebuild topology (a fresh padded preparation per evaluation, and its
exact-shape oracle) was removed in step 4.2.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import gc
from time import perf_counter

import torch
import torch.distributed as dist

from fastmd._nvtx import nvtx_range, nvtx_ranged
from fastmd.precision import gemm_precision_key

from .execution import _run_prepared_tensors, freeze_long_lived_objects, thaw_long_lived_objects
from .watchdog import enter_long_phase, exit_long_phase, fault_point


def _capture_graph(graph, *, stream):
    """``torch.cuda.graph`` with the single-GPU capture policy (thread_local error mode, design §5.5).

    In the default global mode any other host thread's CUDA call during the
    capture (an MD logger's event wait) invalidates it; the NCCL watchdog
    already queries under a thread-local guard (4.1 spike, Q2). No replay-time
    weight check: the model guard rejects a model edit on every rank alike."""
    from fastmd.engine._graph_capture import capture_graph
    return capture_graph(graph, stream=stream, check_replay=False)


def structure_signature(prepared):
    """The part of a padded preparation's layout that capacity does not set.

    ``pad_bond_partial`` builds every padded tensor and route from the
    installed capacity (row and per-peer split counts, contiguous) and from
    this structure: schedule, line branch, backend, tensor names and per-row
    shapes/dtypes, routes and exchange bindings, and the FP32 GEMM mode that
    selects the private padded reverse SUM (with the Triton dot precision, which
    the captured kernels bake in). With the capacity generation it keys the
    topology epoch's captured whole step (``EpochTopology.graph_key``).
    """
    return (prepared.schedule.signature, prepared.has_line, prepared.local_backend, gemm_precision_key(),
            tuple((name, tuple(value.shape[1:]), str(value.dtype), str(value.device), str(value.layout))
                  for name, value in prepared.tensors.items()),
            tuple(prepared.plans),
            tuple((name, binding.plan.name, binding.trailing_shape, str(binding.dtype),
                   str(binding.device), binding._private_reverse_sum_enabled(), binding.self_rows)
                  for name, binding in prepared.schedule.exchanges))


@dataclass
class GraphResult:
    energy: torch.Tensor
    forces: torch.Tensor
    stats: dict
    # A guarded evaluator's model scan overlapping the replay (set by the
    # evaluator's epoch step): this rank's rejection message or None. The
    # caller must agree it collectively (with its finiteness check) before
    # reading energy/forces; None without a guard (the full program stamp was
    # agreed before the replay).
    model_error: str | None = None
    # The stress (eV/Å³, replicated) where the run asked for it (its graph computes it), else None.
    stress: torch.Tensor | None = None


class DistributedGraphRunner:
    """The topology epoch's model_graph: no rank-local fallback, one resident graph.

    ``run_whole`` replays the captured whole step; it captures only at an
    agreed epoch refresh, once per layout key (the capacities are the
    topology's ``TieredCapacity``, agreed at the refresh). ``capacity`` (the
    evaluator's CapacityConfig) sets the warmup.

    All run_whole() calls are collective. close() is local but must be called on all
    ranks before destroying their NCCL process group. A failed capture is a
    fatal worker-group error, not permission for one rank to execute eagerly.
    ``freeze_gc`` (the evaluator's policy) freezes the long-lived objects
    after each (re)capture; releasing a graph first thaws such a freeze
    (thaw_long_lived_objects), so the release collection sees frozen cycles.
    """

    freeze_gc = False
    coordinator = None
    # ParallelConfig.debug_replay_check (set by the evaluator): agree every
    # replay's identity across ranks after it (off: no cost). A captured
    # collective is invisible to the NCCL watchdog and carries no operation
    # identity: without the check a rank that skipped or added a replay hangs
    # the step until the progress watchdog's deadline, or pairs its collectives
    # with another step's (wrong forces, 4.1 spike); with it every rank raises.
    replay_check = False
    # The stress: once a run asked for it (``run_whole(stress=True)``), every later capture of this runner
    # computes it (``captured_stress``), and its graph serves the runs without it too.
    stress = False
    captured_stress = False

    def __init__(self, model, context, *, capacity, freeze_gc=False):
        from fastmd.config import CapacityConfig
        from fastmd.engine.capture import CaptureCoordinator

        self.model, self.context = model, context
        self.freeze_gc = bool(freeze_gc)
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("distributed model_graph requires a CUDA model")
        if not isinstance(capacity, CapacityConfig):
            raise TypeError("capacity must be CapacityConfig")
        self.coordinator = CaptureCoordinator(context._collective, rank=context.rank,
                                              world_size=context.world_size)
        # CUDA graph preparation always needs at least one side-stream
        # execution, even if optional user warmup repeats are set to zero. A
        # recapture runs the topology's CapturePolicy count (run_whole).
        self.warmup = max(1, capacity.warmup)
        self.generation = self.capture_count = self.replay_count = 0
        self._graph = self._static = self._outputs = None
        self._model_stamp = None
        self._closed = False

    def _release_entry(self, *, refreeze=False):
        had_graph = self._graph is not None
        if had_graph:
            torch.cuda.synchronize(self.device)
            self._graph.reset()
        self._outputs = self._graph = self._static = None
        self._captured_key = None
        self._whole = self._md = None
        self._model_stamp = None
        self.captured_stress = False
        if had_graph:
            # Captured autograd/NCCL Work objects can survive in Python cycles.
            # Release them before communicator teardown or constructing a new
            # entry; this is never part of compatible replay's hot path. A
            # capture point may have frozen them (freeze_long_lived_objects,
            # here or in the evaluator): gc.collect() skips frozen objects.
            thawed = thaw_long_lived_objects()
            gc.collect()
            if refreeze and thawed:
                # A recapture or an auto switch to eager, not teardown: the
                # live set was just collected, so freeze it again right away.
                # The post-capture freeze then walks only warmup/capture
                # objects: one full-heap pass per recapture, not two (~130 ms
                # each at 1,344 atoms on 4 GPUs), and eager keeps the freeze.
                freeze_long_lived_objects(collect=False)
            torch.cuda.synchronize(self.device)

    def close(self, *, refreeze=False):
        """Release captured NCCL work before process-group destruction.

        ``refreeze`` (the evaluator's auto switch to eager, which keeps
        running): restore a thawed long-lived-object freeze after the release
        collection. Teardown leaves it thawed.
        """
        self._release_entry(refreeze=refreeze)
        self._closed = True

    def unbind_model(self):
        """After a collectively rejected model verification: never replay this graph again.

        A replay's deferred scan (GraphResult.model_error) rejects only after
        the replay was submitted. Unbinding makes the next step refresh the
        epoch and recapture (``_evaluate_epoch_whole``), and the scan agreed
        before that warmup rejects a model that is still mutated: the graph
        replays a rejected program at most once (P=1 drops its graphs
        likewise). Local and cheap; the evaluator calls it on every rank from
        the agreed verdict, and the recapture vote is gathered in any case.
        """
        self._model_stamp = None

    def stress_stale(self, stress: bool = False) -> bool:
        """Whether a run with ``stress`` must recapture for the stress (asked now or by an earlier run) that
        the resident graph does not compute; an epoch refresh must precede it."""
        return (stress or self.stress) and not self.captured_stress

    _captured_key = None
    _whole = None
    # The in-graph MD step's integrator hook of the captured graph (unified step 3.5 PR C), else None.
    _md = None

    def _check_replay(self, energy):
        """Debug (``replay_check``): every rank ran the same replay of the same graph.

        First the host-only identity (replay count, graph generation) over the
        Gloo control group, before anything waits on the device: a rank that
        skipped or added a replay must not block on the peers' replay (which
        waits for its collectives) or on the data communicator the replay
        uses. Then the replayed global energy's bits (the in-graph SUM,
        identical on all ranks unless their collectives paired with another
        step's), which waits for the replay.
        """
        context = self.context
        identity = (self.replay_count, self.generation)
        if context.world_size > 1 and context.control_group is not None:
            gathered = [None] * context.world_size
            dist.all_gather_object(gathered, identity, group=context.control_group)
            if any(item != gathered[0] for item in gathered):
                raise RuntimeError(f"model_graph replay mismatch across ranks (replay count, generation): "
                                   f"{gathered}")
        bits = int(energy.detach().double().reshape(-1)[0].view(torch.int64).item())
        gathered = context._collective((identity, bits), "graph.replay_check")
        if any(item != gathered[0] for item in gathered):
            raise RuntimeError(f"model_graph replay mismatch across ranks ((replay count, generation), "
                               f"energy bits): {gathered}")

    def _whole_body(self):
        """One whole topology-epoch step: exact filter, exchanges, model, VJP, flags (captured).

        With an MD hook (``run_whole(md=...)``) the integrator's halves run
        inside it, on the owned capacity rows: backup and pre-force half into
        ``_pos_in``, then the step, then the post-force half with the forces
        and energy, and the integrator's non-finite flag after the step flags.
        A graph captured with the stress (``captured_stress``) also returns it.
        """
        topology, md = self._whole, self._md
        rows = topology._pos_in.shape[0]
        with torch.no_grad():
            if md is not None:
                topology._pos_in.copy_(md.pre(rows))
            topology._fill()
            topology._derive()
        if self.captured_stress:
            result = _run_prepared_tensors(self.model, None, self.context, self._static, stress=True)
        else:
            result = _run_prepared_tensors(self.model, None, self.context, self._static)
        energy, forces = result.readout(self.context)
        stress = (result.global_stress(self.context, self._static.tensors["cell"]),) if self.captured_stress else ()
        with torch.no_grad():
            flags = topology.whole_flags(energy, forces, result)
            if md is not None:
                md.post(forces[:rows], energy, rows)
                flags = torch.cat((flags, md.flags(rows)))
        return (energy, forces, flags, *stress)

    @nvtx_ranged("DistributedGraphRunner.run_whole")
    def run_whole(self, topology, *, key, may_capture, md=None, stress=False):
        """The WholeStep-style topology-epoch step (unified step 3.4, M-d): one replay.

        The captured graph reads the caller's coordinates from
        ``topology._pos_in`` and runs the ghost COPY, the builder's exact
        filter and the topology derivation, the model forward and VJP with
        their exchanges, the global energy SUM and the step's flag vector
        (``EpochTopology.whole_flags``). Every buffer it reads is persistent
        per layout key and rewritten in place at an epoch refresh, so it is
        captured once per ``key`` (agreed capacities, structure, layout) and
        only at an agreed refresh step (``may_capture``). No control round and
        no host synchronization here; the caller agrees the flags.

        ``md`` (the in-graph MD step, unified step 3.5 PR C): the integrator
        hook (``pre(rows)``, ``post(forces, energy, rows)``, ``flags(rows)``,
        ``restore()``) whose halves the graph also runs (``_whole_body``);
        ``key`` must then identify its buffers too. Each capture warmup
        restores the integrator, so a capture leaves the MD state unchanged.

        ``stress`` (the same on every rank): the result carries the stress
        (``GraphResult.stress``); from then on every capture of this runner
        computes it (``stress``), and a graph captured without it recaptures,
        which needs ``may_capture``.
        """
        entered = perf_counter()
        if self._closed:
            raise RuntimeError("model_graph runner is closed")
        # _model_stamp None: unbound after a rejected model scan (unbind_model).
        recapture = (self._graph is None or self._captured_key != key or self._model_stamp is None
                     or self._md is not md or self.stress_stale(stress))
        if stress:
            self.stress = True
        # The full warmup count for this runner's first capture and after an
        # unbound model (CapturePolicy.warmups); a capacity recapture runs one.
        first = self.capture_count == 0 or self._model_stamp is None
        if may_capture and self.context.world_size > 1:
            # A refresh step: every rank captures if any rank's entry is stale
            # (the capture's warmup and barrier are collective), with one
            # warmup count (its iterations run the step's collectives).
            votes = self.context._collective((recapture, first), "whole_step.capture")
            recapture, first = any(vote[0] for vote in votes), any(vote[1] for vote in votes)
        if recapture and not may_capture:
            raise RuntimeError("topology epoch: a model_graph recapture outside an epoch refresh")
        warmup_ms = capture_ms = release_ms = 0.0
        if recapture:
            long_phase = enter_long_phase("model_graph_capture")
            started = perf_counter()
            self._release_entry(refreeze=True)
            torch.cuda.empty_cache()
            release_ms = (perf_counter() - started) * 1000
            self._static, self._whole, self._md = topology.prepared, topology, md
            self.captured_stress = self.stress
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            saved_stats = deepcopy(self.context.stats)
            started = perf_counter()
            warmups = topology.capture_policy.warmups(first, distributed=True)
            from fastmd.engine.kernels.triton_wide_gemm import capture_warmup
            # GEMMs between the capture and eager row thresholds use cuBLAS
            # ordinarily. Exercise their captured Triton path here so module
            # loading cannot first occur inside the graph capture.
            with torch.cuda.stream(stream), capture_warmup():
                for _ in range(warmups):
                    outputs = self._whole_body()
                    del outputs
                    if md is not None:
                        md.restore()  # the body's own backup: the state before the warmup
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            warmup_ms = (perf_counter() - started) * 1000
            self.context.stats.clear()
            self.context.stats.update(saved_stats)
            self.coordinator.barrier(self.generation)
            saved_stats = deepcopy(self.context.stats)
            graph = torch.cuda.CUDAGraph()
            self._graph = graph
            started = perf_counter()
            with _capture_graph(graph, stream=stream):
                self._outputs = self._whole_body()
            torch.cuda.synchronize(self.device)
            capture_ms = (perf_counter() - started) * 1000
            self.context.stats.clear()
            self.context.stats.update(saved_stats)
            self._captured_key = key
            # Bound: the evaluator agreed a model scan before this capture.
            self._model_stamp = "whole_step"
            self.capture_count += 1
            self.generation += 1
            if self.freeze_gc:
                freeze_long_lived_objects()
            exit_long_phase(long_phase)
        if not fault_point("replay", self.replay_count, self.context.rank):
            with nvtx_range("cudaGraphLaunch"):
                self._graph.replay()
            self.replay_count += 1
        energy = self._outputs[0].clone()
        forces = self._outputs[1].clone()
        flags = self._outputs[2].clone()
        virial = self._outputs[3].clone() if stress else None
        if self.replay_check:
            self._check_replay(energy)
        stats = {"model_graph": {
            "mode": "topology_epoch_whole_step", "generation": self.generation,
            "capture_count": self.capture_count, "recapture_count": self.capture_count - 1,
            "replay_count": self.replay_count, "captured_this_call": recapture,
            "capacity_changed": recapture, "warmup_ms": warmup_ms, "capture_ms": capture_ms,
            "release_ms": release_ms, "run_host_ms": (perf_counter() - entered) * 1000,
            "scope": ("integrator pre-force half + " if self._md is not None else "")
                     + "ghost COPY + exact filter + topology derivation + model forward + explicit VJP "
                     "+ NCCL COPY/reverse-SUM + global energy SUM"
                     + (" + strain gradient SUM" if self.captured_stress else "") + " + step flags"
                     + (" + integrator post-force half" if self._md is not None else ""),
            "resident_entries": 1,
        }, "forward_implementation": "shared_model"}
        return GraphResult(energy, forces, stats, stress=virial), flags
