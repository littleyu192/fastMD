"""Experimental sharded single-system conservative evaluator (MatRIS, through its exchange strategy)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import wraps
import json
import time
import weakref

import torch
import torch.distributed as dist
from torch import Tensor

from fastmd._nvtx import nvtx_range, nvtx_ranged
from fastmd.adapters import as_adapter
from fastmd.config import InferenceConfig, resolve_config
from fastmd.engine.policy import policy_summary
from fastmd.precision import (
    deterministic_mode, matmul_precision_scope, precision_state, triton_dot_precision_scope, warn_implicit_tf32,
)
from . import shutdown_registry
from .auto_execution import AutoExecution, AutoExecutionState
from .config import ParallelConfig, distributed_resolved, inference_config
from .communicator import resolve_payload_transport
from .exchange import ExchangeContext
from .execution import (
    DistributedModelGuard, _model_stamp, freeze_long_lived_objects, model_program_stamp, validate_prepared,
    run_prepared,
)
from .graph import build_distributed_graph
from ..engine import memory_admission
from .partition import AtomState, DomainPartition, migrate_atoms
from .rank_contract import agree_rank_contract
from .strategy import strategy_of
from .watchdog import fault_point


def _objects(context, value):
    return context.all_gather_object(value)


def _raise_preflight(errors) -> None:
    """The rejection of ``collective_check`` for per-rank messages (rank order)."""
    if any(errors):
        raise ValueError("distributed preflight failed: " + "; ".join(
            f"rank {rank}: {message}" for rank, message in enumerate(errors) if message))


def collective_check(context, error: str | None) -> None:
    """Everyone completes the same check before any following payload operation."""
    _raise_preflight(_objects(context, error))


# Device predicates of the packed CUDA preflight (AtomState._cuda_flags plus
# the embedding-table Z range): fixed entries of the admission flag round.
_PREFLIGHT_PREDICATES = 5
# Host entry 0 of the post-submission flag round: this rank's model program
# changed since construction (the P>1 OverlappedModelGuard.verify() result,
# unified step 3.1a H3); see DistributedMatRISEvaluator._model_error_after_submission.
FINITE_ROUND_MODEL_MUTATED = 0


def _precision_scoped(method):
    """Run under ``config.matmul_precision`` and ``config.triton_dot_precision`` (if set), restoring the
    caller's."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        config = getattr(self, "config", None)
        with matmul_precision_scope(getattr(config, "matmul_precision", None)), \
                triton_dot_precision_scope(getattr(config, "triton_dot_precision", None)):
            return method(self, *args, **kwargs)
    return wrapped


def reduce_sum(context, value: Tensor) -> Tensor:
    result = value.detach().clone()
    if context.world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=context.group)
    return result


def validate_root(context, root: int) -> int:
    roots = _objects(context, root)
    if (any(type(value) is not int or not 0 <= value < context.world_size for value in roots)
            or any(value != roots[0] for value in roots)):
        raise ValueError("all ranks must specify the same valid group-relative root")
    return dist.get_global_rank(context.control_group, root) if context.world_size > 1 else root


@dataclass
class DistributedPrediction:
    """Total energy replicated as a scalar; forces exist only for owned atoms; the stress (``[3, 3]`` in eV/Å³,
    replicated) where the evaluation computed it, else None."""

    state: AtomState
    forces: Tensor
    energy: Tensor
    energy_per_atom: Tensor
    reference_energy: Tensor
    natoms: int
    epoch: int
    stats: dict
    stress: Tensor | None = None


def use_single_rank_engine(world_size: int, device: torch.device) -> bool:
    """One CUDA rank runs the single-GPU engine (the multi-rank protocol: P>1, or CPU)."""
    return world_size == 1 and device.type == "cuda"


def _live_evaluators() -> list:
    """Evaluators constructed and not closed (the shutdown registry's first phase). apply_checkpoint's flags are
    part of every admitted program: re-flagging a shared model would make each of these reject all later
    evaluations."""
    return shutdown_registry.live("evaluators")


def _checkpoint_flags(model):
    """What ``apply_checkpoint`` sets (the recompute mode separately)."""
    return (getattr(model, "enable_checkpoint", None), model.config.get("enable_checkpoint"),
            tuple(getattr(block, "enable_checkpoint", None) for block in model.interaction_block))


def _requested_device(model):
    # Engine selection precedes model admission (which reports bad models).
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


class _LazyEpochGraph:
    """``last_graph`` of a topology-epoch step: its key view is built on first attribute access."""

    def __init__(self, topology):
        object.__setattr__(self, "_topology", topology)
        object.__setattr__(self, "_view", None)

    def __getattr__(self, name):
        view = object.__getattribute__(self, "_view")
        if view is None:
            from .epoch import EpochGraphView
            view = EpochGraphView(object.__getattribute__(self, "_topology"))
            object.__setattr__(self, "_view", view)
        return getattr(view, name)


class DistributedMatRISEvaluator:
    """Spatial/graph decomposition, not batch parallel inference.

    All methods involving communication must be called by all ranks in the same
    order. ``evaluate`` migrates atoms and returns the new owned state alongside
    its matching forces. ``model`` is a ``ModelAdapter`` or a model object
    (MatRIS's module, wrapped as it is: ``fastmd.adapters.as_adapter``); the
    evaluator runs it through the exchange strategy the adapter provides
    (``fastmd.distributed.strategy``: MatRIS's boundary-bond partial
    exchange; a model without one is rejected). The model module becomes
    frozen and enters eval mode. ``config.matmul_precision`` (if set) applies
    to every evaluation.
    One CUDA rank delegates to the single-GPU engine (fastmd.distributed.
    single_rank) behind the same API; the multi-rank protocol runs at P>1, and
    on CPU at any P.
    """

    def __init__(self, model, cell, *, pbc=(True, True, True),
                 config: InferenceConfig | None = None,
                 parallel: ParallelConfig | None = None,
                 context: ExchangeContext | None = None):
        self.context = ExchangeContext() if context is None else context
        if not isinstance(self.context, ExchangeContext):
            raise TypeError("context must be an ExchangeContext")
        error = None
        self._engine = self._resolved = None
        try:
            # The adapter (a model object is wrapped as it is), the strategy it
            # provides, and the module every rank evaluates.
            self.adapter = as_adapter(model)
            self.strategy = strategy_of(self.adapter, self.context.world_size)
            model = self.adapter.module
            request = ParallelConfig() if parallel is None else parallel
            if not isinstance(request, ParallelConfig):
                raise TypeError("parallel must be ParallelConfig")
            single = use_single_rank_engine(self.context.world_size, _requested_device(model))
            # One knob: the profile selects the local operators (P>1 backend
            # derived; the deprecated local_backend alias warns and maps).
            # The deprecation warning once per job (rank 0); every rank resolves alike.
            self.config, backend, self._profile_source = inference_config(
                config, single_rank=single, local_backend=request.local_backend, warn=self.context.rank == 0)
            self._requested_profile = None if config is None else config.optimization_profile
            # The request stays as given; the resolved backend and auto policy are private.
            self.parallel = request
            self._local_backend = backend
            self._auto_policy = request.auto_execution
            if (self._auto_policy is not None and config is not None
                    and config.execution not in (None, "auto")):
                raise ValueError("ParallelConfig.auto_execution requires InferenceConfig.execution=None or 'auto'; "
                                 "an explicit 'eager' or 'model_graph' selects the mode directly")
            if config is not None and config.execution == "auto" and self._auto_policy is None:
                self._auto_policy = AutoExecution()
            self.strategy.admit_model(model, single_rank=single)
            parameter = next(model.parameters())
            self.device, self.dtype = parameter.device, parameter.dtype
            admission = memory_admission.enabled() if self.device.type == "cuda" else None
            if single:
                from .single_rank import SingleRankEngine
                # TensorGraphBuilder computes in FP32 (validate_model agrees).
                if self.dtype != torch.float32:
                    raise ValueError("the single-rank engine requires an FP32 model")
                SingleRankEngine.admit(model, self.strategy)
                self._requested = {
                    "optimization_profile": self._requested_profile,
                    "checkpoint": "off" if config is None else config.checkpoint}
                self._resolved = resolve_config(self.config)
                enabled = self._resolved.checkpoint_enabled
                wanted = (enabled, enabled, (enabled,) * len(model.interaction_block))
                if (_checkpoint_flags(model) != wanted
                        and any(other.model is model for other in _live_evaluators())):
                    raise ValueError(
                        f"checkpoint={self._requested['checkpoint']!r} would re-flag a model shared with "
                        "a live evaluator admitted with other checkpoint flags (it would then reject "
                        "every evaluation): evaluators sharing a model need the same effective "
                        "checkpoint (model_graph: 'off'); close() the other one or load a separate model")
            if self.config.execution == "model_graph" and self.device.type != "cuda":
                raise ValueError("distributed model_graph requires a CUDA model")
            from fastmd.config import env_value, recompute_policy_default
            from fastmd.engine.saved_recompute import explicit_mode
            requested = (self._resolved.saved_tensor_recompute if single else
                         self.config.saved_tensor_recompute or env_value("MATRIS_SAVED_TENSOR_RECOMPUTE"))
            self._recompute = requested
            self._recompute_policy = (self._resolved.recompute_policy if single else
                                      self.config.recompute_policy or recompute_policy_default())
            if (not single and getattr(model, "saved_recompute", "off") != explicit_mode(requested)
                    and any(other.model is model for other in _live_evaluators())):
                raise ValueError(
                    f"saved_tensor_recompute={requested!r} would re-flag a model shared with a live P>1 "
                    "evaluator admitted with another mode (it would then reject every evaluation); "
                    "close() the other one or load a separate model")
            # The resolved topology: None for the single-GPU engine (its own
            # builder), the skin epoch on CUDA (P>1), and on CPU the internal
            # reference topology, rebuilt every evaluation (_evaluate_reference;
            # the epoch's builder is CUDA only).
            if single:
                self.topology_engine = None
            elif self.device.type == "cuda":
                self.topology_engine = "epoch"
            elif self.parallel.topology_engine == "epoch":
                raise ValueError("topology_engine='epoch' requires CUDA (CPU runs use the internal "
                                 "reference topology)")
            else:
                self.topology_engine = "reference"
            self._topology_skin = float(self.parallel.topology_skin)
            if self.context.backend == "nccl" and self.device.type != "cuda":
                raise ValueError("NCCL requires a CUDA model and atom state")
            if self.context.backend == "gloo" and self.device.type != "cpu":
                raise ValueError("the Gloo reference data backend requires a CPU model")
            if any(p.device != self.device for p in model.parameters()):
                raise ValueError("all model parameters must be on the local worker device")
            self.partition = DomainPartition(
                torch.as_tensor(cell, dtype=self.dtype, device="cpu"),
                self.context.world_size, self.parallel.domain_grid, pbc)
            every_peer_neighbor = False
            if self.context.world_size > 4:
                # More than 4 ranks keep one periodic image per axis (design
                # §6.3 ghost slices). The halo is the atom-graph cutoff plus
                # the epoch's skin, which "auto" clamps to what the box holds
                # (the CPU reference is rebuilt every evaluation: no skin).
                # Up to 4 ranks accept smaller cells, as before.
                halo = float(model.config["pairwise_cutoff"])
                if self.topology_engine == "epoch":
                    from .epoch import effective_skin
                    self._topology_skin = effective_skin(
                        self.partition.face_distances.tolist(), halo, self._topology_skin,
                        explicit=self.parallel.topology_engine == "epoch")
                    halo += self._topology_skin
                self.partition.check_halo(halo)
                every_peer_neighbor = all(
                    len(self.partition.neighbor_ranks(rank, halo)) == self.context.world_size - 1
                    for rank in range(self.context.world_size))
            # "auto" from the agreed geometry (cell, grid, cutoff): identical on every rank.
            self._payload_transport = resolve_payload_transport(
                self.parallel.payload_transport, self.context.world_size,
                every_peer_neighbor=every_peer_neighbor)
            self.fingerprint = self.strategy.model_fingerprint(model)
            # Tiers, bands and warmups of the topology epoch: the tier knobs are
            # the P>1 forward profile's registered options (the environment is
            # ignored, as for GPU MD under that profile), the recapture warmup
            # this rank's environment: a bad value fails here on every rank, a
            # differing one in the contract below.
            from fastmd.engine.capacity_policy import CapturePolicy
            self._capture_policy = CapturePolicy.from_config(
                self.config.capacity, distributed=True,
                options=distributed_resolved(self._local_backend).kernel_options)
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            error = str(exc)
        collective_check(self.context, error)
        # The rank contract: every rank derives the same plan from these entries (the adapter, its exchange
        # strategy, the model, the geometry and every resolved option), or the construction fails on all ranks.
        contract = {
            "adapter": self.adapter.name, "strategy": self.strategy.contract(model),
            "model": self.fingerprint, "cell": self.partition.cell.tolist(),
            "grid": self.parallel.domain_grid, "dtype": str(self.dtype),
            "isolated_atoms": self.config.isolated_atoms,
            "execution": self.config.execution or "eager", "profile": self.config.optimization_profile,
            "backend": self._local_backend,
            "single_rank_engine": self.parallel.single_rank_engine,  # always "single" since the rollback cleanup
            "auto_execution": None if self._auto_policy is None else asdict(self._auto_policy),
            "exchange_self_rows": self.parallel.exchange_self_rows,
            "control_plane": self.parallel.control_plane,  # always "nccl" since the rollback cleanup; key kept
            "payload_transport": self.parallel.payload_transport,
            "legacy_capacity_policy": self.parallel.legacy_capacity_policy,
            "exchange_scheme": self.parallel.exchange_scheme,
            "grad_mechanism": self.parallel.grad_mechanism,
            "debug_replay_check": self.parallel.debug_replay_check,
            "saved_tensor_recompute": self._recompute,
            "recompute_policy": self._recompute_policy,
            "topology_engine": (self.parallel.topology_engine, self.topology_engine, self._topology_skin),
            "memory_admission": admission,
            "matmul_precision": {key: value for key, value in self._matmul_precision().items()
                                 if key != "matmul_precision"},
            "capture_policy": self._capture_policy.fingerprint,
            # Selects the kernels (and those captured): fixed for the evaluator's lifetime.
            "deterministic": deterministic_mode(),
        }
        agree_rank_contract(self.context, contract)
        self._deterministic = contract["deterministic"]
        # Every rank agreed the payload transport above and switches here, at
        # the same point of the (collective) construction; evaluate()
        # re-selects it in case another evaluator sharing this context chose
        # differently.
        self.context.set_payload_transport(self._payload_transport)
        self.model = model.eval().requires_grad_(False)
        if self._resolved is not None:
            # As in the single-GPU entry points; the stamp below admits the result.
            from fastmd.engine.entry_config import apply_checkpoint
            apply_checkpoint(self.model, self._resolved.checkpoint_enabled)
        else:
            # P>1 forwards run under the distributed kernel scope, not this
            # config's: the model carries the mode (part of its stamp).
            # Only an explicit "on"/"deep" is model state; "auto" binds its admission's
            # decision around each evaluation (saved_recompute.bound_admission).
            self.model.saved_recompute = explicit_mode(self._recompute)
        error = None
        try:
            stamp = _model_stamp(self.model)
            self._model_program = model_program_stamp(stamp)
            self._model_graph_contract = self._graph_input_contract()
            # Every P>1 evaluator (the flag-round path; unified step 3.1a
            # H3/H4): the guard's identity precheck replaces prepare's
            # full scan, and the evaluation's one full scan is at the CPU
            # reference topology's run boundary (before the first payload) or, for the
            # epoch step (eager, C9) and a model_graph replay, after its
            # submission, agreed in the finite flag round;
            # long-lived objects are frozen at warmup/capture points. One
            # rank (the CPU reference, no flag round) keeps both full stamps.
            self._model_guard = (DistributedModelGuard(self.model, self._model_program, stamp)
                                 if self._resolved is None and self.context.world_size > 1 else None)
        except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as exc:
            error = f"model admission failed: {exc}"
        collective_check(self.context, error)
        self.epoch = 0
        self.last_graph = None
        self._graph_runner = None
        # Device-memory admission (the strategy's: the generic admission with the model's memory predictor;
        # one hook per engine).
        self._admission = None
        if contract["memory_admission"]:
            self._admission = self.strategy.memory_admission(
                self.model, resolved=self._resolved, local_backend=self._local_backend, device=self.device,
                recompute=self._recompute, policy=self._recompute_policy, epoch=self.topology_engine == "epoch")
        if self._resolved is not None:
            from .single_rank import SingleRankEngine
            error = None
            try:
                self._engine = SingleRankEngine(
                    self.model, self.partition, self._resolved, self.device,
                    requested=self._requested, profile_source=self._profile_source,
                    isolated_atoms=self.config.isolated_atoms, program=self._model_program,
                    context=self.context, auto_execution=self._auto_policy,
                    admission=self._admission, strategy=self.strategy)
            except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as exc:
                error = str(exc)
            collective_check(self.context, error)
        elif self.config.execution == "model_graph":
            # (CUDA only, so on the topology epoch.)
            from .graph_runner import DistributedGraphRunner
            self._graph_runner = DistributedGraphRunner(
                self.model, self.context, capacity=self.config.capacity,
                freeze_gc=self._model_guard is not None)
            self._graph_runner.replay_check = self.parallel.debug_replay_check
        self._auto = None
        if self._auto_policy is not None and self._engine is None:
            # The first evaluate decides, from the gathered per-rank counts.
            # (The single-GPU engine owns its own decision.)
            # Under memory admission P>1 with the native backend takes
            # model_graph whenever it fits: measured faster at every size
            # (bond_partial at 1.9M-15.2M triplet rows per rank; the removed
            # rows exchange 10.5-11.5% at 1.9M-7.6M). torch_reference keeps the row
            # threshold: its padded model_graph replays are 2.2-4.4x slower
            # than eager at 1.9M rows per rank. The threshold is also the rule
            # without admission.
            self._auto = AutoExecutionState(self._auto_policy,
                                            device_type=self.device.type,
                                            fixed=(memory_admission.no_bound_reason(
                                                       self._admission.memory, "architecture outside the K6 "
                                                       "calibration") + ": eager (no memory bound for a capture)"
                                                   if self._admission is not None
                                                   and not self._admission.calibrated else None),
                                            prefer_model_graph=(self._admission is not None
                                                                and self._admission.calibrated
                                                                and self.context.world_size > 1
                                                                and self._local_backend == "native"))
        self._topology = None
        if self.topology_engine == "epoch":
            from .epoch import EpochTopology
            # execution="auto" decides at the first epoch refresh (its K6
            # check); start from the likely mode (model_graph where preferred).
            eager = self._graph_runner is None and not (self._auto is not None and self._auto.prefer_model_graph)
            self._topology = EpochTopology(
                self.model, self.partition, self.context, skin=self._topology_skin,
                handle_isolated_atoms=self.config.isolated_atoms == "reference",
                capacity_config=self.config.capacity, eager=eager, capture_policy=self._capture_policy,
                strategy=self.strategy)
            self._topology.isolated_is_error = (self.config.isolated_atoms == "error"
                                                or self.model.reference_energy is None)
            # The topology can outlive this evaluator (``last_graph`` holds it):
            # an evaluator collected without close() releases its captured
            # topology graph (NCCL work inside) here, not at process-group
            # destruction, where a live one hangs NCCL's finalize.
            finalizer = weakref.finalize(self, self._topology.release_graph)
            finalizer.atexit = False
        # Closed at shutdown before the process groups go, unless closed or collected first.
        shutdown_registry.register(self, "evaluators")
        if self.device.type == "cuda":
            with matmul_precision_scope(self.config.matmul_precision):
                warn_implicit_tf32(self.config.matmul_precision, "MatRIS distributed evaluator")
        if self.config.report:
            print(json.dumps(self.config_summary(), sort_keys=True), flush=True)

    def _matmul_precision(self) -> dict:
        """Requested precision and what torch reports and runs inside evaluate."""
        with matmul_precision_scope(self.config.matmul_precision), \
                triton_dot_precision_scope(self.config.triton_dot_precision):
            state = precision_state()
        return {"matmul_precision": state["float32_matmul_precision"],
                "requested_matmul_precision": self.config.matmul_precision,
                "fp32_gemm": state["fp32_gemm"], "triton_dot": state["triton_dot"]}

    @property
    def engine(self) -> str:
        """``"single"`` (one CUDA rank) or ``"distributed"`` (the multi-rank protocol: P>1, or CPU at any P)."""
        return "single" if self._engine is not None else "distributed"

    def config_summary(self) -> dict:
        if self._engine is not None:
            return {
                "rank": self.context.rank, "world_size": self.context.world_size,
                "requested_local_backend": self.parallel.local_backend,
                "effective_local_backend": None,
                "local_backend_note": ("deprecated alias of optimization_profile (see profile_source); "
                                       "the single-GPU engine runs the profile's kernels"),
                "exchange_self_rows": None,
                "requested_control_plane": self.parallel.control_plane, "control_plane": "local",
                "model_sha256": self.fingerprint, "dtype": str(self.dtype),
                **self._matmul_precision(), "domain_grid": list(self.partition.domain_grid),
                **self._engine.config_summary(),
            }
        # Until auto's first evaluation neither mode nor schedule is known.
        pending = self._auto is not None and self._auto.mode is None
        # INV-4: strict (the multi-rank protocol). The policy of
        # the local backend's profile scope, which the forward runs in.
        with self._forward_scope():
            kernel_policy = policy_summary(strict_errors=True)
        eager_because = ("execution='eager'" if self._auto is None else
                         "execution='auto': " + self._auto.summary()["reason"])
        from fastmd.engine.saved_recompute import explicit_mode
        # The level the last admission check chose ("off" before the first evaluation).
        recompute = self._admission.recompute if self._admission is not None else explicit_mode(self._recompute)
        return {
            "rank": self.context.rank, "world_size": self.context.world_size,
            "engine": self.engine, "single_rank_engine": self.parallel.single_rank_engine,
            "execution": (self.config.execution or "eager") if self._auto is None else (self._auto.mode or "auto"),
            "auto_execution": None if self._auto is None else self._auto.summary(),
            "memory_admission": ({"enabled": False} if self._admission is None else self._admission.summary()),
            "requested_profile": self._requested_profile,
            "resolved_profile": distributed_resolved(self._local_backend).optimization_profile,
            "profile_source": self._profile_source,
            # The only operator engine since step 2.6 (the switch was removed); kept for report stability.
            "dispatch_engine": "unified",
            "requested_local_backend": self.parallel.local_backend,
            "effective_local_backend": self._local_backend, "model_sha256": self.fingerprint,
            "kernel_policy": kernel_policy,
            "exchange_self_rows": self.parallel.exchange_self_rows,
            "requested_control_plane": self.parallel.control_plane,
            "control_plane": "local" if self.context.world_size == 1 else self.parallel.control_plane,
            "exchange_scheme": self.parallel.exchange_scheme,
            "grad_mechanism": self.parallel.grad_mechanism,
            "payload_transport": self._payload_transport,
            # The resolved topology and recompute (part of the rank contract); whole_step / topology_graph:
            # the model_graph whole-step graph or the eager epoch's captured topology step (None until
            # execution='auto' decides).
            "topology_engine": {"requested": self.parallel.topology_engine, "resolved": self.topology_engine,
                                "requested_skin": float(self.parallel.topology_skin),
                                "skin": self._topology_skin if self._topology is not None else None,
                                "whole_step": None if pending else self._graph_runner is not None,
                                "topology_graph": None if pending else (self._topology is not None
                                                                        and self._graph_runner is None)},
            "saved_tensor_recompute": {"requested": self.config.saved_tensor_recompute, "resolved": self._recompute,
                                       "current": {"narrow": "on"}.get(recompute, recompute)},
            "recompute_policy": self._recompute_policy,
            "requested_checkpoint": self.config.checkpoint, "effective_checkpoint": "off",
            "forward_implementation": "shared_model",
            "dtype": str(self.dtype), **self._matmul_precision(),
            "domain_grid": list(self.partition.domain_grid),
            "operators": ("the single-GPU OpDispatcher selection of the resolved profile on the prepared "
                          "layout (interleaved pairs, owner CSR, private sink, shared line rows); forces by one "
                          "autograd VJP (grad_mechanism='autograd')"),
            "operator_capabilities": {
                "indexed_projection": self._local_backend == "native",
                "merged_projection": self._local_backend == "native",
                "fused_owner_csr_attention": self._local_backend == "native",
                # A layout capability of the private padded sink (model_graph), in either profile.
                "zero_skip_index": None if pending else self._graph_runner is not None,
                "zero_skip_index_scope": "private-padded gather VJP, local SUM and reverse exchange SUM; qualified FP32 CUDA only",
                "fused_basis": self._local_backend == "native",
                "semantics": "allowed capabilities, not actual hits; qualification includes device/dtype/shape/precision/layout",
                "actual_selection": ("pending: execution='auto' decides at the first evaluation" if pending else
                                     "not reported; the op-selection recorder (fastmd.engine.recorder) records it"),
            },
            "execution_schedule": ("pending: execution='auto' chooses eager or padded model_graph at the first evaluation"
                                   if pending else
                                   "topology epoch; one whole-step CUDA graph (ghost COPY, exact filter, model, VJP) per layout key"
                                   if self._graph_runner is not None else
                                   "topology epoch; padded layout, boundary-validated asynchronous eager"
                                   if self._topology is not None else
                                   "fresh prepared plans every evaluation; boundary-validated eager"),
            "disabled": {**({"custom_fused_kernels": "generic profile (torch_reference): the profile-switched "
                                                     "fused kernels; the private-sink zero-skip kernels still run"}
                            if self._local_backend == "torch_reference" else {}),
                         **({} if self._graph_runner is not None or pending else {"cuda_graph": eager_because}),
                         **({"capacity": "unused by the CPU reference topology (exact rows)"}
                            if self._topology is None else {})},
            "topology": ("topology epoch: superset shard per skin epoch (cutoffs + skin), exact device filter "
                         "every step with the single-GPU fixed-capacity builder; owned+halo only"
                         if self._topology is not None else
                         "CPU reference: independent candidate/key reference with exact device cutoff, rebuilt "
                         "every evaluation"),
            "control_metadata": ("device atom/bond/triplet keys at epoch refreshes; per-step admission and "
                                 "step flags in the flag rounds; padded rows and per-peer splits per epoch"
                                 if self._topology is not None else
                                 "CPU reference candidates; bond_partial device-topology shard"),
        }

    def _forward_scope(self):
        """The P>1 forward's configuration scope (the strategy's ``forward_scope``)."""
        return self.strategy.forward_scope(self._local_backend)

    def close(self):
        """Release any captured NCCL work before destroying process groups."""
        shutdown_registry.unregister(self, "evaluators")
        if self._engine is not None:
            self._engine.close()
        if self._graph_runner is not None:
            self._graph_runner.close()
        if self._topology is not None:
            self._topology.close()

    def _observe_execution(self, gathered, fits=None) -> None:
        """Apply the auto choice; every rank sees the same gathered counts.

        A switch happens on the topology epoch only (at a refresh,
        ``_admit_epoch``); the CPU reference topology's state is pinned to
        eager and only records the gathered maximum.
        """
        if not self._auto.observe(self.epoch, gathered, fits=fits):
            return
        if self._auto.mode == "model_graph":
            from .graph_runner import DistributedGraphRunner
            self._graph_runner = DistributedGraphRunner(
                self.model, self.context, capacity=self.config.capacity,
                freeze_gc=self._model_guard is not None)
            self._graph_runner.replay_check = self.parallel.debug_replay_check
            # Once an evaluation asked for the stress, every captured step computes it.
            self._graph_runner.stress = self._stress_asked
        elif self._graph_runner is not None:
            # Collective in effect: all ranks switch at the same evaluation.
            # Eager has no capture point to freeze again: keep the freeze
            # through the release collection (a guarded evaluator's H4).
            self._graph_runner.close(refreeze=True)
            self._graph_runner = None

    def _admit_epoch(self, topology) -> None:
        """K6 and execution="auto" at the first epoch and every capacity change (agreed).

        The bond_partial calibration's rows are the installed capacity's
        triplet rows (the tier of the global MAX demand, shared by every rank
        in eager and model_graph alike), so a recapture for grown capacities
        is admitted here too. "auto" chooses
        from every rank's triplet rows of this refresh (already gathered for
        the capacities) and the admitted modes; a switch creates or closes the
        graph runner here, before this refresh's step (which then captures, or
        runs eager).
        """
        fits = None
        if self._admission is not None:
            # The installed capacity's triplet rows: the tier of the MAX demand over ranks plus its growth
            # band, the same on every rank; eager and model_graph allocate them, not this rank's demand.
            rows = topology.capacity.rows[0]["t"] if topology.capacity.rows else topology.demand_rows.get("t", 0)
            event = f"epoch {self.epoch}: topology epoch capacity {topology.capacity.generation}"
            if self._auto is None:
                mode = self.config.execution or "eager"
                self._admission.check((mode,), rows, rows, requested=mode, context=self.context, event=event,
                                      shared=("model_graph",) if mode == "model_graph" else ())
            else:
                fits = self._admission.check(("model_graph", "eager"), rows, rows, requested=None,
                                             context=self.context, event=event, shared=("model_graph",))
        if self._auto is not None:
            state, graphed = self._auto.state, self._graph_runner is not None
            self._observe_execution([(n, t, state) for n, t in topology.rank_rows], fits)
            topology.set_eager(self._graph_runner is None)
            if graphed and self._graph_runner is None and self._model_guard is not None:
                # Switched to eager inside a step that began as model_graph (no scan
                # yet): the eager rule, the full scan before the step's payloads. Agreed.
                self.context.collective_check(self._model_guard.verify_error())

    def _graph_input_contract(self):
        """Small preflight subset read before the full prepare model guard.

        The complete immutable-program check reuses prepare's one fresh scan
        (single engine: its OverlappedModelGuard).
        These bindings must be checked earlier because topology construction
        and atom-number admission consume them before prepare is called.
        """
        embedding = self.model.atom_embedding.embedding
        return (self.model.config["pairwise_cutoff"], self.model.config["three_body_cutoff"],
                id(self.model.atom_embedding), id(embedding), embedding.num_embeddings,
                id(self.model.reference_energy))

    def distribute(self, atoms=None, *, root: int = 0, payload_dtype: torch.dtype | None = None) -> AtomState:
        """Initial CPU scatter from ASE Atoms on root; other ranks pass None.

        Initial masses and momenta are carried as payload for optional NVE, in
        the model dtype unless ``payload_dtype`` is given; ``torch.float64``
        also carries the float64 input positions as "positions64" (the
        float64 MD state of ``nvt_nhc``, see
        ``GPUResidentMolecularDynamics.from_evaluator``).
        No global coordinates or forces are gathered by the inference loop.
        """
        global_root = validate_root(self.context, root)
        chunks = None
        error = None
        if self.context.rank == root:
            try:
                if atoms is None or len(atoms) == 0:
                    raise ValueError("root must supply nonempty ASE Atoms")
                if not all(atoms.pbc):
                    raise ValueError("only full PBC is supported")
                cell = torch.as_tensor(atoms.cell.array, dtype=self.dtype)
                if not torch.equal(cell, self.partition.cell.to(self.dtype)):
                    raise ValueError("input atoms cell differs from evaluator cell")
                positions = torch.as_tensor(atoms.positions, dtype=self.dtype)
                if not bool(torch.isfinite(positions).all()):  # before owner_of bins them
                    raise ValueError("positions must be finite")
                numbers = torch.as_tensor(atoms.numbers, dtype=torch.long)
                ids = torch.arange(len(atoms), dtype=torch.long)
                owners = self.partition.owner_of(positions)
                dtype = self.dtype if payload_dtype is None else payload_dtype
                if dtype not in (torch.float32, torch.float64):
                    raise ValueError("payload_dtype must be torch.float32 or torch.float64")
                payload = {"masses": torch.as_tensor(atoms.get_masses(), dtype=dtype),
                           "momenta": torch.as_tensor(atoms.get_momenta(), dtype=dtype)}
                if payload_dtype == torch.float64:
                    payload["positions64"] = torch.as_tensor(atoms.positions, dtype=torch.float64)
                chunks = []
                for rank in range(self.context.world_size):
                    mask = owners == rank
                    chunks.append((ids[mask], positions[mask], numbers[mask],
                                   {k: v[mask] for k, v in payload.items()}))
            except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
                error = str(exc)
        collective_check(self.context, error)
        if self.context.world_size == 1:
            chunk = chunks[0]
        else:
            output = [None]
            dist.scatter_object_list(output, chunks, src=global_root, group=self.context.control_group)
            chunk = output[0]
        return AtomState(chunk[0].to(self.device), chunk[1].to(self.device),
                         chunk[2].to(self.device), {k: v.to(self.device) for k, v in chunk[3].items()})

    def distribute_restart(self, path, *, root: int = 0, atoms=None) -> dict:
        """Scatter an MD restart file over the current ranks (all ranks call), as :meth:`distribute`.

        Root reads the file (:func:`~fastmd.execution.md_protocol.read_restart`)
        and checks it against this evaluator's cell (and, if given, root's
        ``atoms`` numbers); every atom goes to the owner of its float32
        position under the current partition, computed on root's device as
        migration computes it, and each rank's rows are in ID order: exactly
        the rows and order evaluate produces. Initialization-time Gloo
        scatter. Returns ``{"prediction": the cached DistributedPrediction
        (not re-evaluated; epoch -1), "step", "parameters", "temperature_K"
        (None for nve), "nhc_state" (nvt_nhc, else None), "offsets", "engine"
        (the writer: "domain" or "native")}``, the ``restart`` argument of
        ``GPUResidentMolecularDynamics.from_evaluator``; the payload dtype
        follows the saved ensemble (float64 plus "positions64" for nvt_nhc).
        """
        import numpy as np
        from fastmd.execution.md_protocol import read_restart

        context = self.context
        global_root = validate_root(context, root)
        chunks, error = None, None
        if context.rank == root:
            try:
                data = read_restart(path)
                cell = torch.as_tensor(data.cell).to(self.dtype)
                if not torch.equal(cell, self.partition.cell.to(self.dtype)) or not data.pbc.all():
                    raise ValueError("restart cell/pbc differs from the evaluator's")
                ids, numbers = torch.tensor(data.ids), torch.tensor(data.numbers)
                if atoms is not None and not np.array_equal(data.numbers, atoms.get_atomic_numbers()):
                    raise ValueError("restart atomic numbers do not match the input structure")
                nhc = data.parameters["ensemble"] == "nvt_nhc"
                dtype = torch.float64 if nhc else self.dtype
                positions64 = torch.tensor(data.positions)
                positions = positions64.to(self.dtype)
                payload = {"masses": torch.tensor(data.masses).to(dtype), "momenta": torch.tensor(data.momenta).to(dtype)}
                if nhc:
                    payload["positions64"] = positions64
                forces = torch.tensor(data.forces)
                shared = {"step": data.step, "parameters": data.parameters, "energy": torch.tensor(data.energy),
                          "reference_energy": torch.tensor(data.reference_energy), "natoms": len(ids),
                          "temperature_K": data.temperature_K, "nhc_state": data.nhc_state, "offsets": data.offsets,
                          "engine": data.engine}
                owners = self.partition.owner_of(positions.to(self.device)).cpu()
                chunks = []
                for rank in range(context.world_size):
                    mask = owners == rank
                    chunks.append(((ids[mask], positions[mask], numbers[mask],
                                    {key: value[mask] for key, value in payload.items()}, forces[mask]), shared))
            except Exception as exc:  # e.g. EOFError, zipfile.BadZipFile: every rank raises it below
                error = str(exc)
        collective_check(context, error)
        if context.world_size == 1:
            chunk, shared = chunks[0]
        else:
            output = [None]
            dist.scatter_object_list(output, chunks, src=global_root, group=context.control_group)
            chunk, shared = output[0]
        device = self.device
        ids, positions, numbers, payload, forces = chunk
        state = AtomState(ids.to(device), positions.to(device), numbers.to(device),
                          {key: value.to(device) for key, value in payload.items()})
        energy = shared["energy"].to(device)
        prediction = DistributedPrediction(state, forces.to(device), energy, energy / shared["natoms"],
                                           shared["reference_energy"].to(device), shared["natoms"], -1, {})
        return {"prediction": prediction, "step": shared["step"], "parameters": shared["parameters"],
                "temperature_K": shared["temperature_K"], "nhc_state": shared["nhc_state"],
                "offsets": shared["offsets"], "engine": shared["engine"]}

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _preflight(self, state: AtomState, extra=None, *, deferred: list | None = None
                   ) -> tuple[str | None, list, object]:
        """Admission of one evaluate() input, shared by every engine.

        Returns (first error message or None, extra flag values, extra carry).
        Conditions, messages and first-error order are the historical ones:
        graph-input contract, ambient autocast, AtomState.validate, shapes,
        device/dtype, finite positions, Z range, payload rows. On CUDA every
        device predicate, plus the caller's ``extra(order, sorted_ids, numbers)
        -> (flags, carry)`` (``order`` sorts the IDs), crosses in ONE packed
        download instead of one per check.

        ``deferred`` (P>1, the flag rounds): a list that receives the
        packed predicates as int64 failure indicators instead of that
        download. They then read as passed here, so the result is the first
        host-side error only; the admission flag round agrees the rest, and
        after a reported failure the caller reruns this for the exact message.
        """
        values, carry = [], None
        try:
            try:
                if self._graph_input_contract() != self._model_graph_contract:
                    raise ValueError("model graph-input contract was mutated after construction")
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError(f"model graph-input contract was mutated: {exc}") from exc
            if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
                raise ValueError("distributed inference does not support ambient autocast")
            if deterministic_mode() != self._deterministic:
                raise ValueError("torch.use_deterministic_algorithms changed after this evaluator was constructed "
                                 "(its kernel selection and captured graphs follow the setting at construction): "
                                 "construct a new evaluator")
            packed = isinstance(state.positions, Tensor) and state.positions.is_cuda
            if packed:
                positions, ids, numbers = state._validate_cuda_host()
                sorted_ids, order = ids.sort()
                # The contract above pins the embedding module and its size.
                max_z = self.model.atom_embedding.embedding.num_embeddings
                flags = [*AtomState._cuda_flags(positions, ids, numbers, sorted_ids), (numbers > max_z).any()]
                if extra is not None:
                    more, carry = extra(order, sorted_ids, numbers)
                    flags.extend(more)
                if deferred is None:
                    values = torch.stack(flags).cpu().tolist()
                else:
                    # The first predicate means "finite"; MAX needs failures.
                    deferred.append(torch.stack([~flags[0], *flags[1:]]).long())
                    values = [True] + [False] * (len(flags) - 1)
                AtomState._raise_cuda_flags(values[:4])
            else:
                state.validate()
            n = state.ids.numel()
            if state.ids.ndim != 1 or state.positions.shape != (n, 3) or state.numbers.shape != (n,):
                raise ValueError("invalid atom-state shapes")
            if state.positions.device != self.device or state.positions.dtype != self.dtype:
                raise ValueError("positions must match model device and dtype")
            if not packed:
                if not bool(torch.isfinite(state.positions).all()):
                    raise ValueError("positions must be finite")
                max_z = self.model.atom_embedding.embedding.num_embeddings
            if (values[4] if packed else bool(((state.numbers < 1) | (state.numbers > max_z)).any())):
                raise ValueError("atomic numbers outside the model embedding table")
            if any(v.shape[0] != n for v in state.payload.values()):
                raise ValueError("payload first dimension must match owned atom count")
        except (ValueError, RuntimeError, AttributeError, TypeError, IndexError) as exc:
            return str(exc), values[5:], carry
        return None, values[5:], carry

    def _select_payload_transport(self) -> bool:
        """(Re)select this evaluator's payload transport; True for the P>1 flag rounds.

        Evaluators sharing one context may have been admitted with different
        payload transports; evaluate() is collective and the mode is part of
        the contract, so every rank switches here, at the same call.
        """
        self.context.set_payload_transport(self._payload_transport)
        return self.context.world_size > 1

    def _admit_by_flags(self, state: AtomState, extra=None, host_extra=()):
        """Preflight and epoch agreement in ONE flag round (P>1).

        Historically: the packed predicate download, then three Gloo rounds
        (preflight check, epoch gather, epoch check). Here one MAX round of
        ``[host error, epoch, -epoch, device predicate failures...]``: any
        nonzero failure entry means some rank failed, max(epoch) != min(epoch)
        a mismatch. Rejections keep the historical messages, types and order:
        after a reported failure every rank reruns the preflight for its exact
        message (one local download) and only those strings travel, over Gloo;
        the epoch message is a function of the agreed vector.
        """
        comm = self.context.comm
        deferred = []
        error, _, _ = self._preflight(state, deferred=deferred)
        predicates = deferred[0] if deferred else None
        if predicates is None:
            # Host-rejected or CPU input: the round's length must not depend on it.
            predicates = torch.zeros(_PREFLIGHT_PREDICATES, dtype=torch.long, device=comm.device)
        elif predicates.numel() != _PREFLIGHT_PREDICATES:
            # Code, not input: zeros here would admit every device check.
            raise RuntimeError(f"admission flag round expects {_PREFLIGHT_PREDICATES} device "
                               f"predicates, the preflight packed {predicates.numel()}")
        elif predicates.device != comm.device:
            predicates = predicates.to(comm.device)  # rejected by the host device check anyway
        if extra is not None:
            # Topology epoch: [atom set changed, max displacement bits] ride along.
            predicates = torch.cat((predicates, extra.to(predicates.device)))
        agreed = self.context.agree_flags(predicates, tag="evaluate.admission",
                                          host=(error is not None, self.epoch, -self.epoch, *host_extra))
        # Topology epoch: (the agreed host_extra entries, the agreed extra device entries).
        width = len(host_extra)
        tail = (agreed[3:3 + width].tolist(), agreed[3 + width + _PREFLIGHT_PREDICATES:].tolist())
        agreed = torch.cat((agreed[:3], agreed[3 + width:3 + width + _PREFLIGHT_PREDICATES]))
        if agreed[0] or bool(agreed[3:].any()):
            error, _, _ = self._preflight(state)
            _raise_preflight(comm.errors_slow_path(error, True, phase="evaluate.preflight"))
            raise ValueError("distributed preflight failed: the admission flag round reported a "
                             "failure that no rank reproduced")
        if agreed[1] != -agreed[2]:
            _raise_preflight(["evaluation epoch mismatch"] * self.context.world_size)
        return tail

    @staticmethod
    def _model_error_after_submission(result) -> str | None:
        """Hook: this rank's model-program rejection, decided after device work is submitted.

        Only a model_graph step of a guarded evaluator (``self._model_guard``,
        unified step 3.1a H3) has one: the topology epoch's step stores the
        ``DistributedModelGuard.verify()`` scan it ran after submitting the
        replay as ``result.model_error``. The verdict is
        host entry ``FINITE_ROUND_MODEL_MUTATED`` of the post-submission flag
        round, so every rank rejects identically, as a ``RuntimeError`` in
        ExchangeContext's "distributed validation failed" form (that of the
        prepare/run model-stamp rejections). The eager epoch step (C9) sets it
        from the scan it runs after submitting the step's device work. None
        otherwise: a guarded eager reference-topology preparation's one scan is
        at its run boundary (validate_prepared, before the first payload), and
        an unguarded one (one rank) keeps the full stamps in
        prepare_bond_partial and validate_prepared: one rank has no flag round
        to carry a deferred verdict.
        """
        return getattr(result, "model_error", None)

    def _agree_after_submission(self, energy: Tensor, forces: Tensor, result, extra=None):
        """The post-submission checks in ONE flag round (P>1).

        ``[model mutated, non-finite energy, non-finite forces]``: the
        predicates are computed on the device and reduced with the model
        verdict, without the two historical ``bool()`` synchronizations and
        the Gloo check round. Only after a reported failure do the per-rank
        messages travel (Gloo), giving the historical rejection.
        """
        comm = self.context.comm
        model_error = self._model_error_after_submission(result)
        nonfinite = torch.stack((~torch.isfinite(energy).all(), ~torch.isfinite(forces).all()))
        host = [0]
        host[FINITE_ROUND_MODEL_MUTATED] = int(model_error is not None)
        flags = nonfinite if extra is None else torch.cat((nonfinite.long(), extra.to(nonfinite.device)))
        agreed = self.context.agree_flags(flags, tag="evaluate.finite", host=host)
        tail = agreed[len(host) + 2:]
        agreed = agreed[:len(host) + 2]
        if agreed[FINITE_ROUND_MODEL_MUTATED]:
            # A mutated program's outputs are not meaningful: it is reported
            # first. Every rank saw the verdict: none replays this graph again.
            if self._graph_runner is not None:
                self._graph_runner.unbind_model()
            self.context._raise_errors(comm.errors_slow_path(model_error, True, phase="evaluate.model"))
            raise RuntimeError("distributed validation failed: the model flag round reported a "
                               "mutation that no rank reproduced")
        if bool(agreed[len(host):].any()):
            local = "non-finite energy or forces" if bool(nonfinite.any()) else None
            _raise_preflight(comm.errors_slow_path(local, True, phase="evaluate.finite"))
            raise ValueError("distributed preflight failed: the finite-check flag round reported a "
                             "failure that no rank reproduced")
        return tail

    @_precision_scoped
    @nvtx_ranged("DistributedMatRISEvaluator.evaluate")
    def evaluate(self, state: AtomState, *, stress: bool = False) -> DistributedPrediction:
        """Energy and forces of the owned atoms (collective); ``stress=True`` (the same on every rank) also
        returns the system's stress (``DistributedPrediction.stress``, ``[3, 3]`` in eV/Å³, replicated: the
        single-GPU calculator's units and sign for a right-handed cell). The energy and the forces are those
        of the same call without the stress (bitwise under deterministic kernels). At P>1 model_graph
        captures a step with the stress at the epoch refresh the first such call forces, and every whole
        step the evaluator captures later computes it too, serving the calls with and without it; one CUDA
        rank switches its runner to task "efs"."""
        if not isinstance(stress, bool):
            raise TypeError("stress must be a bool")
        # The forwards use the decisions this evaluator's collective memory admission chose.
        with self.strategy.evaluation_scope(self._recompute, self._admission):
            return self._evaluate(state, **({"stress": True} if stress else {}))

    # An evaluation (or in-graph MD step) asked for the stress: a model_graph runner that execution="auto"
    # creates later computes it from its first capture.
    _stress_asked = False

    def _evaluate(self, state: AtomState, stress: bool = False) -> DistributedPrediction:
        entered = time.perf_counter()
        if stress:
            self._stress_asked = True
        fault_point("evaluate", self.epoch, self.context.rank)
        if self._engine is not None:
            prediction = self._engine.evaluate(state, epoch=self.epoch, entered=entered,
                                               preflight=self._preflight, **({"stress": True} if stress else {}))
            self.last_graph = self._engine.last_graph
            self.epoch += 1
            return prediction
        if self._topology is not None:
            if self._graph_runner is not None:
                return self._evaluate_epoch_whole(state, entered, **({"stress": True} if stress else {}))
            return self._evaluate_epoch(state, entered, **({"stress": True} if stress else {}))
        return self._evaluate_reference(state, entered, **({"stress": True} if stress else {}))

    def _evaluate_reference(self, state: AtomState, entered: float, stress: bool = False) -> DistributedPrediction:
        """``evaluate`` on the internal CPU reference topology: rebuilt every evaluation, eager.

        CPU ranks only (the topology epoch's builder is CUDA only): migrate,
        build the exact bond_partial shard (``build_distributed_graph``),
        prepare and run it with the explicit VJP. It carries the CPU (Gloo)
        P>1 protocol and ``--device cpu``; before step 4.2 it was the
        ``topology_engine="rebuild"`` path, whose CUDA model_graph runner and
        exact-shape oracle were removed.
        """
        flag_rounds = self._select_payload_transport()
        admission_calls = self.context.stats["control_calls"]
        if flag_rounds:
            self._admit_by_flags(state)
        else:
            error, _, _ = self._preflight(state)
            collective_check(self.context, error)
            epochs = _objects(self.context, self.epoch)
            collective_check(self.context, None if len(set(epochs)) == 1 else "evaluation epoch mismatch")
        admission_calls = self.context.stats["control_calls"] - admission_calls
        self.context.reset_stats()
        # The admission's control rounds precede the reset (one flag round at
        # P>1, three local rounds at one rank); control_calls counts the rest.
        self.context.stats["admission_control_calls"] = admission_calls
        start = time.perf_counter()
        state = migrate_atoms(state, self.partition, self.context)
        migrated = time.perf_counter()
        graph = build_distributed_graph(
            state, self.partition, self.context,
            self.model.config["pairwise_cutoff"], self.model.config["three_body_cutoff"])
        topology = graph.device_topology
        counts = torch.bincount(topology["center_owned_target"], minlength=len(graph.owned_atom_ids))
        isolated = bool((counts == 0).any())
        isolated_error = ("isolated atoms require a reference table and isolated_atoms='reference'"
                          if isolated and (self.config.isolated_atoms == "error"
                                           or self.model.reference_energy is None) else None)
        local_counts = (len(graph.owned_atom_ids), len(graph.owned_triplet_keys),
                        None if self._auto is None else self._auto.state)
        if flag_rounds:
            # One gather instead of three: the isolated verdict rides with the
            # counts, and "empty global system" is a function of the gathered
            # counts, identical on every rank. Messages, types and order are
            # those of the three historical rounds below.
            gathered = _objects(self.context, (isolated_error, local_counts))
            _raise_preflight([item[0] for item in gathered])
            counts = [item[1] for item in gathered]
            natoms = sum(count[0] for count in counts)
            if not natoms:
                _raise_preflight(["empty global system"] * self.context.world_size)
        else:
            collective_check(self.context, isolated_error)
            counts = _objects(self.context, local_counts)
            natoms = sum(count[0] for count in counts)
            collective_check(self.context, "empty global system" if not natoms else None)
        if self._auto is not None:
            # CPU: pinned to eager; records the gathered maximum and checks the
            # ranks' states (AutoExecutionState.observe).
            self._observe_execution(counts)
        graphed = time.perf_counter()
        # The whole E/F (prepare, forward, VJP) in the local backend profile's
        # scope; prepare agrees its kernel policy across ranks.
        with self._forward_scope(), torch.enable_grad():
            prepared = self.strategy.prepare(
                self.model, graph, self.context, epoch=self.epoch,
                local_backend=self._local_backend,
                model_guard=self._model_guard,
                expected_model_program=None if self._model_guard is not None else self._model_program,
                exchange_self_rows=self.parallel.exchange_self_rows,
                skip_dead_threebody_tail=True,
                handle_isolated_atoms=self.config.isolated_atoms == "reference")
            prepared_at = time.perf_counter()
            result = run_prepared(self.model, graph, self.context, prepared, epoch=self.epoch,
                                  **({"stress": True} if stress else {}))
            forwarded = time.perf_counter()
            forces = result.forces()
            backwarded = time.perf_counter()
        # Execution uses sorted owned IDs; explicitly map back to returned state order.
        owned_ids = topology["atom_ids"][topology["owned_atom_indices"]]
        order = torch.searchsorted(owned_ids, state.ids)
        forces = forces[order].detach()
        energy = result.global_energy(self.context)
        virial = result.global_stress(self.context, prepared.tensors["cell"]) if stress else None
        if flag_rounds:
            self._agree_after_submission(energy, forces, result)
        else:
            collective_check(self.context, None if bool(torch.isfinite(energy).all())
                             and bool(torch.isfinite(forces).all()) else "non-finite energy or forces")
        if self.model.reference_energy is None:
            reference = energy.new_zeros(())
        else:
            reference = self.model.reference_energy.fc.weight[0, state.numbers.long() - 1].sum()
            reference = reduce_sum(self.context, reference).reshape(())
        finished = time.perf_counter()
        stats = dict(result.stats)
        stats.update({"rank": self.context.rank, "epoch": self.epoch,
                      "owned_atoms": len(graph.owned_atom_ids),
                      "ghost_atoms": len(graph.atom_ids) - len(graph.owned_atom_ids),
                      "owned_bonds": len(graph.owned_bond_keys),
                      "owned_triplets": len(graph.owned_triplet_keys),
                      "preflight_ms": (start - entered) * 1000,
                      "migration_ms": (migrated - start) * 1000,
                      "graph_ms": (graphed - migrated) * 1000,
                      "prepare_execution_ms": (prepared_at - graphed) * 1000,
                      "run_prepared_forward_ms": (forwarded - prepared_at) * 1000,
                      "forward_and_plan_ms": (forwarded - graphed) * 1000,
                      "vjp_and_readout_ms": (finished - forwarded) * 1000,
                      "vjp_ms": (backwarded - forwarded) * 1000,
                      "readout_ms": (finished - backwarded) * 1000,
                      "total_ms": (finished - start) * 1000,
                      "timing_semantics": "host wall intervals (CPU reference topology)",
                      "graph_timing": dict(getattr(graph, "graph_timings", {})),
                      "device_graph": dict(getattr(graph, "device_graph_stats", {})),
                      "prepared_schedule": dict(prepared.schedule.stats),
                      "tape_timing": result.backend.stats,
                      "communication": {**self.context.stats,
                          "payload_scope": "model COPY/adjoint, atom migration, halo, distributed ID validation and device row-plan requests; excludes Gloo control bytes and scalar reductions"}})
        if self._auto is not None:
            stats["auto_execution"] = {"mode": self._auto.mode, "decisions": len(self._auto.decisions),
                                       "max_rank_triplet_rows": self._auto.last_max_rows}
        self.last_graph = graph
        prediction = DistributedPrediction(state, forces, energy, energy / natoms,
                                           reference, natoms, self.epoch, stats, stress=virial)
        self.epoch += 1
        # Account for the real lifetime cost before returning. The prediction
        # owns detached energy/forces and plain stats, not the VJP tape. Without
        # this explicit release, CPU tensor/key teardown was only visible in the
        # external benchmark wall time, outside every recorded phase.
        del result, prepared
        completed = time.perf_counter()
        stats["cleanup_ms"] = (completed - finished) * 1000
        stats["evaluate_wall_ms"] = (completed - entered) * 1000
        if self.epoch == 1 and self._model_guard is not None:
            # Once per evaluator, after its first accepted evaluation: the
            # long-lived objects (model, plans, communicator staging) exist
            # now, and later collections skip them.
            freeze_long_lived_objects()
        return prediction

    def _evaluate_epoch_whole(self, state: AtomState, entered: float, stress: bool = False) -> DistributedPrediction:
        """model_graph with the topology epoch: the WholeStep-style step (unified step 3.4, M-d).

        Within an epoch a step is one replay of the captured whole step
        (``DistributedGraphRunner.run_whole``: ghost COPY, exact filter,
        topology derivation, model, VJP, energy SUM, step flags) and ONE flag
        round after it, which also carries this rank's host verdicts (model
        scan overlapping the replay, state identity, contract, autocast). There
        is no admission round: the step's own displacement is checked after
        the fact, and every condition that the admission round would have
        caught (skin exceeded, capacity overflow, another atom set or order,
        non-finite coordinates) makes every rank evaluate the same state again
        through ``_evaluate_epoch`` (admission round, refresh, replay), so the
        returned result is always exact. A look-ahead on the agreed
        displacement refreshes one step early, so a skin redo is rare.
        ``stress`` on a graph captured without it refreshes and recaptures.
        """
        topology, runner = self._topology, self._graph_runner
        stress_capture = runner.stress_stale(stress)
        if (topology.prepared is None or topology.refresh_ahead or runner._model_stamp is None
                or runner._md is not None or stress_capture):
            # (Unbound after a rejected model scan: recapture, after an agreed scan, at a refresh. The
            # graph of an in-graph MD step (md_step) also runs an integrator: capture the plain step.
            # The first step with the stress captures it.)
            return self._evaluate_epoch(state, entered, force_refresh=(
                topology.refresh_ahead or runner._model_stamp is None or runner._md is not None or stress_capture),
                **({"stress": True} if stress else {}))
        # The topology epoch runs on CUDA at P>1 only (one CUDA rank is the
        # single-GPU engine): always guarded, always the flag rounds.
        self._select_payload_transport()
        self.context.reset_stats()
        self.context.stats["admission_control_calls"] = 0
        start = time.perf_counter()
        try:
            host_ok = (state.ids is topology._ids and state.numbers is topology._numbers
                       and not (torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"))
                       and deterministic_mode() == self._deterministic
                       and self._graph_input_contract() == self._model_graph_contract
                       and topology.load_positions(state.positions))
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            host_ok = False
        with self._forward_scope(), torch.enable_grad():
            result, flags = runner.run_whole(topology, key=topology.graph_key, may_capture=False,
                                             **({"stress": True} if stress else {}))
        submitted = time.perf_counter()
        # The step's model scan, overlapping the replay.
        model_error = self._model_guard.verify_error()
        # [model rejected, host input unusable, epoch, -epoch]: the evaluation epochs
        # agree when max(epoch) == min(epoch) (the admission round's rule).
        host = [int(model_error is not None), int(not host_ok), self.epoch, -self.epoch]
        agreed = self.context.agree_flags(flags, tag="evaluate.whole_step", host=host).tolist()
        agreed_at = time.perf_counter()
        if agreed[0]:
            runner.unbind_model()
            self.context._raise_errors(self.context.comm.errors_slow_path(
                model_error, True, phase="evaluate.model"))
            raise RuntimeError("distributed validation failed: the model flag round reported a "
                               "mutation that no rank reproduced")
        if agreed[2] != -agreed[3]:
            _raise_preflight(["evaluation epoch mismatch"] * self.context.world_size)
        fields = topology.whole_fields(agreed[len(host):])
        redo = (agreed[1] or fields["bad_input"] or fields["capacity"] or fields["stale_candidates"]
                or 2.0 * fields["displacement"] > topology.skin)
        if not redo:
            if fields["isolated"]:
                local = bool(topology.builder._isolated_atom_count.item()) and topology.isolated_is_error
                message = "isolated atoms require a reference table and isolated_atoms='reference'"
                _raise_preflight(self.context.comm.errors_slow_path(
                    message if local else None, True, phase="evaluate.isolated"))
                _raise_preflight([message])
            if fields["nonfinite"]:
                local = not bool((torch.isfinite(result.energy).all() & torch.isfinite(result.forces).all()).item())
                _raise_preflight(self.context.comm.errors_slow_path(
                    "non-finite energy or forces" if local else None, True, phase="evaluate.finite"))
                _raise_preflight(["non-finite energy or forces"])
            fatal = [name for name in topology.STEP_FLAGS[2:] if fields[name]]
            asymmetric = topology.pair_asymmetry(fields["pairs"])
            if asymmetric:
                fatal.append(f"line slot hits differ between rank pairs {asymmetric}")
            if fatal:
                raise RuntimeError(f"topology epoch: {', '.join(fatal)} on some rank (agreed)")
        if redo:
            # Another state, a skin or capacity violation, or unusable input on
            # some rank: every rank evaluates this state again, exactly.
            if fields["capacity"]:
                topology.overflows += 1
            topology.redos = getattr(topology, "redos", 0) + 1
            return self._evaluate_epoch(state, entered, force_refresh=bool(
                fields["capacity"] or fields["stale_candidates"] or 2.0 * fields["displacement"] > topology.skin),
                **({"stress": True} if stress else {}))
        topology.steps += 1
        topology.observe_displacement(fields["displacement"])
        return self._whole_prediction(state, result, fields, entered=entered, start=start, submitted=submitted,
                                      agreed_at=agreed_at, **({"stress": True} if stress else {}))

    def _whole_prediction(self, state, result, fields, *, entered, start, submitted, agreed_at, refresh=None,
                          attempts=1, stress=False) -> DistributedPrediction:
        """The prediction and stats of an accepted whole step (``_evaluate_epoch_whole``, ``md_step``); the
        replay's stress where the call asked for it."""
        topology = self._topology
        forces = result.forces[:topology.n_owned]
        energy = result.energy.reshape(())
        counts = topology.step_counts()
        finished = time.perf_counter()
        natoms = topology.natoms
        tape_timing = {name: 0.0 for name in (
            "forward_input_ms", "forward_local_ms", "forward_exchange_ms", "backward_setup_ms",
            "backward_local_ms", "backward_exchange_ms", "backward_finish_ms")}
        tape_timing["timing_mode"] = "cuda_graph_no_host_stage_timing"
        stats = dict(result.stats)
        migration_ms = (refresh or {}).get("migration_ms", 0.0)
        stats.update({"rank": self.context.rank, "epoch": self.epoch, "owned_atoms": topology.n_owned, **counts,
                      "preflight_ms": (start - entered) * 1000, "migration_ms": migration_ms,
                      "graph_ms": (refresh or {}).get("refresh_ms", 0.0) - migration_ms,
                      "graph_timing": {"halo_seconds": 0.0, "build_local_seconds": 0.0,
                                       "route_owner_seconds": 0.0, "route_triplet_seconds": 0.0},
                      "prepare_execution_ms": 0.0, "plan_build_ms": 0.0,
                      "model_graph_ms": (submitted - start) * 1000,
                      "run_prepared_forward_ms": (submitted - start) * 1000,
                      "forward_and_plan_ms": (submitted - start) * 1000,
                      "vjp_and_readout_ms": (finished - submitted) * 1000, "vjp_ms": 0.0,
                      "readout_ms": (finished - submitted) * 1000,
                      "flag_round_ms": (agreed_at - submitted) * 1000,
                      "total_ms": (finished - start) * 1000,
                      "timing_semantics": "host wall intervals; model_graph_ms = whole-step replay submission",
                      "tape_timing": tape_timing,
                      "topology": {"engine": "epoch", "whole_step": True, "skin": topology.skin,
                                   "refreshed": refresh is not None,
                                   "attempts": attempts, "max_displacement": fields["displacement"],
                                   "epochs": topology.epochs, "steps": topology.steps,
                                   "overflows": topology.overflows, "redos": getattr(topology, "redos", 0),
                                   "refresh_ahead": topology.refresh_ahead, "refresh": refresh},
                      "communication": {**self.context.stats,
                          "payload_scope": "captured whole step (ghost COPY, model COPY/adjoint); excludes control"}})
        # One allocator query (the two max_memory_* calls flatten all stats twice, ~0.14 ms).
        memory = torch.cuda.memory_stats_as_nested_dict(self.device)
        stats.update(peak_allocated_bytes=memory["allocated_bytes"]["all"]["peak"],
                     peak_reserved_bytes=memory["reserved_bytes"]["all"]["peak"])
        prediction = DistributedPrediction(state, forces, energy, energy / natoms,
                                           topology.reference_energy.clone(), natoms, self.epoch, stats,
                                           stress=result.stress if stress else None)
        self.epoch += 1
        del result
        completed = time.perf_counter()
        if self._auto is not None:
            stats["auto_execution"] = {"mode": self._auto.mode, "decisions": len(self._auto.decisions),
                                       "max_rank_triplet_rows": self._auto.last_max_rows,
                                       "decided_at": "epoch refreshes with a capacity change"}
        stats["cleanup_ms"] = (completed - finished) * 1000
        stats["evaluate_wall_ms"] = (completed - entered) * 1000
        self.last_graph = _LazyEpochGraph(topology)
        return prediction

    @_precision_scoped
    @nvtx_ranged("DistributedMatRISEvaluator.md_step")
    def md_step(self, md, *, stress: bool = False) -> DistributedPrediction | None:
        """One MD step with the integrator inside the epoch whole-step graph (unified step 3.5 PR C). Collective.

        ``md`` is the evaluator-driven MD driver's hook
        (``GPUResidentMolecularDynamics``): ``refresh_state()`` (the committed
        state q_n, its payload also carrying the cached forces F(q_n)),
        ``install(state, rows)`` (the refreshed rows into integrator slots of
        at least ``rows`` rows), ``capture_key()``, the captured halves
        ``pre(rows)`` (backup, pre-force half on the owned capacity rows,
        their float32 positions), ``post(forces, energy, rows)`` and
        ``flags(rows)``, and ``restore()`` / ``state()``.

        A step is one replay of the captured graph (the pre-force half, the
        whole step of ``_evaluate_epoch_whole``, the post-force half) and ONE
        flag round. An epoch refresh (the look-ahead, the first step, a model
        unbound after a rejected scan, a redo) runs BEFORE the step, on the
        committed positions q_n: admission round, migration with the forces,
        capacities, the integrator's reorder into slots of the topology's
        owned capacity, an eager derive at the refresh positions for the slot
        admission, and a capture if the key changed. A step that must be
        discarded (capacity overflow, stale candidates, skin violation)
        restores the integrator to q_n (the backup the graph took) and is
        redone after a refresh there; an overflow grows the capacities to the
        overflowed step's untruncated counts, retried while that agreed
        capacity grows. Non-finite positions, momenta, energy or forces, a
        rejected model or input and every fatal flag restore the integrator,
        then raise on every rank. An exception inside a refresh or capture
        leaves the state as ``evaluate`` does (the driver stops).

        Returns None if this evaluator does not run the topology epoch's
        model_graph (CPU reference, single-rank engine, eager), also after
        execution="auto" switched to eager at this step's refresh, and when a
        step violates the skin (or leaves its candidates stale) right after a
        refresh at q_n (its own drift exceeds the epoch; the integrator is
        restored): the caller then runs the step with the integrator outside
        (``evaluate``, which refreshes at q_{n+1}).

        ``stress`` (the same on every rank): the prediction carries the stress
        of the step's forces (``evaluate``'s; a graph captured without it
        refreshes and recaptures, and keeps it for every later step).
        """
        if self._engine is not None or self._topology is None or self._graph_runner is None:
            return None
        if stress:
            self._stress_asked = True
        with self.strategy.evaluation_scope(self._recompute, self._admission):
            return self._md_step(md, **({"stress": True} if stress else {}))

    # Refreshes one in-graph MD step may take (the look-ahead or first one, then redos).
    _MD_STEP_REFRESHES = 4

    def _md_step(self, md, stress: bool = False) -> DistributedPrediction | None:
        entered = time.perf_counter()
        fault_point("evaluate", self.epoch, self.context.rank)
        topology = self._topology
        self._select_payload_transport()
        floor = overflowed = refresh_stats = None
        attempts = refreshes = 0
        while True:
            attempts += 1
            runner = self._graph_runner
            # Agreed on every rank: the look-ahead and the model verdict come from flag rounds.
            refresh = (attempts > 1 or topology.prepared is None or topology.refresh_ahead
                       or runner._model_stamp is None or runner._md is not md or runner.stress_stale(stress))
            if refresh:
                refreshes += 1
                if refreshes > self._MD_STEP_REFRESHES:
                    raise RuntimeError(f"topology epoch: the in-graph MD step was discarded after "
                                       f"{self._MD_STEP_REFRESHES} refreshes")
                self._md_refresh(md, floor)
                refresh_stats = dict(topology.stats)
                if overflowed is not None and not any(
                        topology.capacity.local().get(name, 0) > overflowed.get(name, 0) for name in floor):
                    # (The capacities are the agreed tier of the MAX demand: the same verdict on every rank.)
                    raise RuntimeError("topology epoch: the in-graph MD step still overflows after a capacity "
                                       "refresh")
                if self._graph_runner is None:  # execution="auto" chose eager at this refresh
                    return None
                runner = self._graph_runner
            else:
                self.context.reset_stats()
                self.context.stats["admission_control_calls"] = 0
            outcome = self._md_replay(md, refresh, entered=entered, attempts=attempts,
                                      refresh_stats=refresh_stats if refresh else None,
                                      **({"stress": True} if stress else {}))
            if not isinstance(outcome, str):
                return outcome
            floor = overflowed = None
            if outcome == "refresh" and refresh:
                # A skin violation or stale candidates right after a refresh at q_n: this step's own drift
                # exceeds the epoch; another refresh at q_n cannot help. The integrator is restored: the
                # caller runs this step outside the graph (evaluate refreshes at q_{n+1}).
                return None
            if outcome == "capacity":
                # This rank's demand of the discarded step, a floor for the refresh at q_n.
                floor, overflowed = topology.untruncated_rows(), dict(topology.capacity.local())

    def _refresh_topology(self, state, floor=None):
        """``EpochTopology.refresh`` with this evaluator's admission and preparation (collective)."""
        return self._topology.refresh(
            state, epoch=self.epoch, validate=lambda prepared: validate_prepared(
                self.model, prepared.shard, self.context, prepared, epoch=self.epoch,
                # Every epoch step runs the guard's full scan itself,
                # after its submission (eager and model_graph alike).
                defer_model_scan=True),
            prepare_kwargs=dict(
                local_backend=self._local_backend, model_guard=self._model_guard,
                expected_model_program=None,
                exchange_self_rows=self.parallel.exchange_self_rows,
                skip_dead_threebody_tail=True,
                handle_isolated_atoms=self.config.isolated_atoms == "reference"),
            floor=floor)

    def _set_reference_energy(self, numbers) -> None:
        """The epoch's reference energy (constant within it): one SUM per (agreed) refresh."""
        if self.model.reference_energy is None:
            reference = torch.zeros((), dtype=self.dtype, device=self.device)
        else:
            reference = self.model.reference_energy.fc.weight[0, numbers.long() - 1].sum()
            reference = reduce_sum(self.context, reference).reshape(())
        self._topology.reference_energy = reference.detach().clone()

    def _md_refresh(self, md, floor) -> None:
        """The in-graph MD step's epoch refresh at the committed state q_n (collective)."""
        topology = self._topology
        state = md.refresh_state()
        calls = self.context.stats["control_calls"]
        # The admission round of an epoch step (preflight, epoch agreement; no eager precheck here).
        self._admit_by_flags(state, extra=topology.admission_flags(state, self.device), host_extra=(0,))
        admission = self.context.stats["control_calls"] - calls
        self.context.reset_stats()
        self.context.stats["admission_control_calls"] = admission
        try:
            with self._forward_scope():
                state = self._refresh_topology(state, floor=floor)
                if topology.capacity_changed:
                    self._admit_epoch(topology)
            self._set_reference_energy(state.numbers)
            md.install(state, topology.capacity.local()["n"])
            with self._forward_scope():
                # The slot admission at the refresh positions (the step derives after its drift).
                topology.derive_at_epoch()
            topology._fresh = False
        except BaseException:
            topology.invalidate()
            raise
        topology.observe_displacement(0.0)

    def _md_replay(self, md, refreshed, *, entered, attempts, refresh_stats, stress=False):
        """One replay of the in-graph MD step and its flag round: the prediction, or "capacity"/"refresh" (redo)."""
        topology, runner, context = self._topology, self._graph_runner, self.context
        start = time.perf_counter()
        try:
            host_ok = (not (torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"))
                       and deterministic_mode() == self._deterministic
                       and self._graph_input_contract() == self._model_graph_contract)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            host_ok = False
        key = ("md", topology.graph_key, md.capture_key())
        if refreshed:
            capture = (runner._graph is None or runner._captured_key != key or runner._model_stamp is None
                       or runner._md is not md or runner.stress_stale(stress))
            if any(context._collective(capture, "evaluate.whole_capture")):
                # Warmup runs model code with payloads: agree the scan first.
                context.collective_check(self._model_guard.verify_error())
        try:
            with self._forward_scope(), torch.enable_grad():
                result, flags = runner.run_whole(topology, key=key, may_capture=refreshed, md=md,
                                                 **({"stress": True} if stress else {}))
        except BaseException:
            topology.invalidate()
            raise
        submitted = time.perf_counter()
        # The step's model scan, overlapping the replay.
        model_error = self._model_guard.verify_error()
        host = [int(model_error is not None), int(not host_ok), self.epoch, -self.epoch]
        agreed = context.agree_flags(flags, tag="evaluate.whole_step", host=host).tolist()
        agreed_at = time.perf_counter()
        fields = topology.whole_fields(agreed[len(host):])
        if agreed[0] or agreed[2] != -agreed[3] or agreed[1] or fields["bad_input"]:
            md.restore()
        if agreed[0]:
            runner.unbind_model()
            context._raise_errors(context.comm.errors_slow_path(model_error, True, phase="evaluate.model"))
            raise RuntimeError("distributed validation failed: the model flag round reported a "
                               "mutation that no rank reproduced")
        if agreed[2] != -agreed[3]:
            _raise_preflight(["evaluation epoch mismatch"] * context.world_size)
        if agreed[1]:
            # Autocast, the deterministic setting or the graph-input contract changed: the preflight's message.
            self._admit_by_flags(md.state(), host_extra=(0,))
            raise ValueError("distributed preflight failed: the whole-step flag round reported unusable "
                             "host input that no rank reproduced")
        if fields["bad_input"]:
            local = "non-finite MD positions" if topology.whole_fields(flags.tolist())["bad_input"] else None
            _raise_preflight(context.comm.errors_slow_path(local, True, phase="evaluate.finite"))
            _raise_preflight(["non-finite MD positions"])
        if fields["capacity"] or fields["stale_candidates"] or 2.0 * fields["displacement"] > topology.skin:
            # Discarded: the state before the step, and a refresh at it.
            md.restore()
            if fields["capacity"]:
                topology.overflows += 1
            topology.redos = getattr(topology, "redos", 0) + 1
            return "capacity" if fields["capacity"] else "refresh"
        fatal = [name for name in topology.STEP_FLAGS[2:] if fields[name]]
        asymmetric = topology.pair_asymmetry(fields["pairs"])
        if asymmetric:
            fatal.append(f"line slot hits differ between rank pairs {asymmetric}")
        if fields["isolated"] or fields["nonfinite"] or fields["md_nonfinite"] or fatal:
            md.restore()
        if fields["isolated"]:
            local = bool(topology.builder._isolated_atom_count.item()) and topology.isolated_is_error
            message = "isolated atoms require a reference table and isolated_atoms='reference'"
            _raise_preflight(context.comm.errors_slow_path(message if local else None, True,
                                                           phase="evaluate.isolated"))
            _raise_preflight([message])
        if fields["nonfinite"] or fields["md_nonfinite"]:
            mine = topology.whole_fields(flags.tolist())
            local = ("non-finite energy or forces" if mine["nonfinite"] else
                     "non-finite MD positions or momenta" if mine["md_nonfinite"] else None)
            _raise_preflight(context.comm.errors_slow_path(local, True, phase="evaluate.finite"))
            _raise_preflight(["non-finite energy, forces, MD positions or momenta"])
        if fatal:
            raise RuntimeError(f"topology epoch: {', '.join(fatal)} on some rank (agreed)")
        topology.steps += 1
        topology.observe_displacement(fields["displacement"])
        return self._whole_prediction(md.state(), result, fields, entered=entered, start=start,
                                      submitted=submitted, agreed_at=agreed_at, refresh=refresh_stats,
                                      attempts=attempts, **({"stress": True} if stress else {}))

    def _evaluate_epoch(self, state: AtomState, entered: float, force_refresh: bool = False,
                        stress: bool = False) -> DistributedPrediction:
        """``evaluate`` with the skin-epoch topology (``topology_engine="epoch"``, step 3.4).

        Two control rounds per step: the admission round also agrees the epoch
        decision (atom set changed, twice the global maximum owned
        displacement above the skin), and the finite round also agrees a
        capacity overflow of the step (then the capacities grow, the epoch is
        refreshed and the step is evaluated again) and isolated atoms. The
        returned state is the migrated one after a refresh, else ``state``;
        forces are in its row order.
        """
        from .epoch import EpochTopology, step_verdict  # noqa: F401  (the evaluator's _topology)
        from .execution import _run_prepared_tensors
        topology = self._topology
        # CUDA at P>1 only (one CUDA rank is the single-GPU engine): always
        # guarded, always the flag rounds.
        self._select_payload_transport()
        admission_calls = self.context.stats["control_calls"]
        # The model check. Eager (C9): the guard's precheck precedes every payload
        # of the step: a rebinding that can install code, and any change of a
        # parameter, buffer or official cache tensor (in place, ``.data``,
        # requires_grad), is agreed before model code runs, so no rank can raise
        # mid-step on it; the full scan follows the step's submission and its
        # verdict rides in the finite round, as after a model_graph replay. What
        # only the full scan sees (e.g. a ``model.config`` value) is rejected there,
        # before any result is read. model_graph: a replay runs no model code, so
        # the scan follows the replay's submission; a capture is preceded by an
        # agreed scan below.
        model_error = self._model_guard.precheck_error() if self._graph_runner is None else None
        extra = topology.admission_flags(state, self.device)
        agreed = self._admit_by_flags(state, extra=extra, host_extra=(int(model_error is not None),))
        agreed = [*agreed[1][:2], agreed[0][0]]
        if agreed[2]:
            self.context._raise_errors(self.context.comm.errors_slow_path(
                model_error, True, phase="evaluate.model"))
            raise RuntimeError("distributed validation failed: the model flag round reported a "
                               "mutation that no rank reproduced")
        refresh, displacement = topology.needs_refresh(agreed)
        if not refresh:
            # The same atoms (compared by value on every rank) in other tensors,
            # e.g. an MD driver's own state: adopt them, so that the whole-step
            # path's identity check holds again from the next step on.
            topology._ids, topology._numbers = state.ids, state.numbers
        refresh = refresh or force_refresh
        admission_calls = self.context.stats["control_calls"] - admission_calls
        self.context.reset_stats()
        self.context.stats["admission_control_calls"] = admission_calls
        start = time.perf_counter()
        refresh_stats, attempts = None, 0
        isolated_policy = self.config.isolated_atoms == "error" or self.model.reference_energy is None
        try:
            with self._forward_scope():
                while True:
                    attempts += 1
                    if refresh:
                        state = self._refresh_topology(state)
                        refresh_stats = dict(topology.stats)
                        if topology.capacity_changed:
                            self._admit_epoch(topology)
                    graphed = time.perf_counter()
                    whole_flags = None
                    if self._graph_runner is None:
                        prepared = topology.step(state)
                    else:
                        # model_graph: the whole step (coordinates, exact filter,
                        # derivation, model) is one captured graph.
                        topology._fresh = False
                        if not topology.load_positions(state.positions):
                            raise ValueError("topology epoch: the state's positions do not match the epoch")
                    derived = time.perf_counter()
                    if self._graph_runner is None:
                        with torch.enable_grad():
                            with nvtx_range("evaluate.forward"):
                                result = _run_prepared_tensors(self.model, None, self.context, prepared,
                                                               **({"stress": True} if stress else {}))
                            forwarded = time.perf_counter()
                            with nvtx_range("evaluate.backward"):
                                forces = result.forces()[:topology.n_owned]
                        backwarded = time.perf_counter()
                        energy = result.global_energy(self.context)
                        virial = result.global_stress(self.context, prepared.tensors["cell"]) if stress else None
                        # The step's full scan (C9), overlapping the submitted device
                        # work; agreed below, before any result is read.
                        deferred = self._model_guard.verify_error()
                    else:
                        # Captured per epoch capacity/structure key, only at an
                        # (agreed) refresh step; otherwise a replay.
                        runner = self._graph_runner
                        capture = (runner._graph is None or runner._captured_key != topology.graph_key
                                   or runner._model_stamp is None or runner.stress_stale(stress))
                        if refresh:
                            # Agreed (every rank refreshed): the scan below is collective,
                            # and run_whole's own vote decides the same way.
                            capture = any(self.context._collective(capture, "evaluate.whole_capture"))
                        if capture:
                            # Warmup runs model code with payloads: agree the scan first.
                            self.context.collective_check(self._model_guard.verify_error())
                        with torch.enable_grad():
                            result, whole_flags = runner.run_whole(topology, key=topology.graph_key,
                                                                   may_capture=refresh,
                                                                   **({"stress": True} if stress else {}))
                        topology.steps += 1
                        forwarded = backwarded = time.perf_counter()
                        forces = result.forces[:topology.n_owned]
                        energy = result.energy.reshape(())
                        virial = result.stress if stress else None
                        # The step's scan, overlapping the submitted replay; agreed below.
                        result.model_error = self._model_guard.verify_error()
                    verdict = result if self._graph_runner is not None else type(
                        "EpochStep", (), {"model_error": deferred})()
                    isolated = (topology.builder._isolated_atom_count.reshape(1).long().clamp(max=1)
                                if isolated_policy else torch.zeros(1, dtype=torch.long, device=self.device))
                    step_flags = (torch.cat((isolated, topology.step_flags(result))) if whole_flags is None
                                  else whole_flags)
                    tail = self._agree_after_submission(energy, forces, verdict, extra=step_flags).tolist()
                    # An overflow on some rank discards the step before any other
                    # flag is judged (epoch.step_verdict; as the whole-step path).
                    redo, fatal = step_verdict(tail, self.context.world_size)
                    if not redo:
                        if tail[0]:
                            local = int(isolated[0]) > 0
                            message = "isolated atoms require a reference table and isolated_atoms='reference'"
                            _raise_preflight(self.context.comm.errors_slow_path(
                                message if local else None, True, phase="evaluate.isolated"))
                            _raise_preflight([message])
                        if fatal:
                            raise RuntimeError(f"topology epoch: {', '.join(fatal)} on some rank (agreed)")
                        # model_graph: the refresh's slot-key admission reads this
                        # (agreed non-overflowed) replay's slot rows. Collective.
                        topology.check_slots()
                        break
                    # Some rank's graph did not fit: its exact counts grow the
                    # capacities at a refresh of the same positions; evaluate again.
                    topology.overflows += 1
                    if attempts >= 2:
                        raise RuntimeError("topology epoch: the step still overflows after a capacity refresh")
                    refresh = True
                    del result
        except BaseException:
            # A refresh or step interrupted mid-way can leave the persistent
            # buffers half rewritten: the next evaluation starts a new epoch.
            topology.invalidate()
            raise
        if refresh_stats is not None:
            # The reference part of the energy: constant within the epoch (its
            # atoms and numbers), so one gather and SUM per (agreed) refresh,
            # not per step. The steps return a copy (never the caller's tensor).
            self._set_reference_energy(state.numbers)
        topology.observe_displacement(displacement if refresh_stats is None else 0.0)
        natoms = topology.natoms
        self._sync()
        finished = time.perf_counter()
        stats = dict(result.stats)
        forces = forces.detach()
        if self._graph_runner is None:
            tape_timing = result.backend.stats
        else:
            tape_timing = {name: 0.0 for name in (
                "forward_input_ms", "forward_local_ms", "forward_exchange_ms", "backward_setup_ms",
                "backward_local_ms", "backward_exchange_ms", "backward_finish_ms")}
            tape_timing["timing_mode"] = "cuda_graph_no_host_stage_timing"
            stats["model_graph_ms"] = (forwarded - derived) * 1000
        migration_ms = (refresh_stats or {}).get("migration_ms", 0.0) if refresh_stats else 0.0
        counts = topology.step_counts()
        if "bond_partial" in stats:
            # The summary of the epoch's superset preparation: report this step's rows.
            stats["bond_partial"] = {**stats["bond_partial"], "local_bonds": counts["owned_bonds"],
                                     "local_triplets": counts["owned_triplets"],
                                     "line_slots_scope": "epoch superset (boundary bonds within r3 + skin)"}
        stats.update({"rank": self.context.rank, "epoch": self.epoch,
                      "owned_atoms": topology.n_owned, **counts,
                      "preflight_ms": (start - entered) * 1000,
                      "migration_ms": migration_ms,
                      # Everything before the final step's forward: refreshes,
                      # overflowed attempts and this step's device topology.
                      "graph_ms": (derived - start) * 1000 - migration_ms,
                      "graph_timing": {"halo_seconds": 0.0, "build_local_seconds": (derived - graphed),
                                       "route_owner_seconds": 0.0, "route_triplet_seconds": 0.0},
                      "prepare_execution_ms": 0.0, "plan_build_ms": 0.0,
                      "run_prepared_forward_ms": (forwarded - derived) * 1000,
                      "forward_and_plan_ms": (forwarded - derived) * 1000,
                      "vjp_and_readout_ms": (finished - forwarded) * 1000,
                      "vjp_ms": (backwarded - forwarded) * 1000,
                      "readout_ms": (finished - backwarded) * 1000,
                      "total_ms": (finished - start) * 1000,
                      "timing_semantics": "host wall intervals; graph_ms = epoch refresh (if any) + device topology step",
                      "tape_timing": tape_timing,
                      "topology": {"engine": "epoch", "skin": topology.skin, "refreshed": bool(refresh),
                                   "attempts": attempts, "max_displacement": displacement,
                                   "epochs": topology.epochs, "steps": topology.steps,
                                   "overflows": topology.overflows,
                                   "refresh": refresh_stats if refresh else None},
                      "communication": {**self.context.stats,
                          "payload_scope": "model COPY/adjoint and the ghost coordinate COPY; epoch refreshes add migration, halo, ID validation and plan requests"}})
        if self.device.type == "cuda":
            stats.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        prediction = DistributedPrediction(state, forces, energy, energy / natoms,
                                           topology.reference_energy.clone(), natoms, self.epoch, stats,
                                           stress=virial)
        self.epoch += 1
        del result
        # Diagnostic key view of this step (built on access only).
        self.last_graph = _LazyEpochGraph(topology)
        completed = time.perf_counter()
        if self._auto is not None:
            stats["auto_execution"] = {"mode": self._auto.mode, "decisions": len(self._auto.decisions),
                                       "max_rank_triplet_rows": self._auto.last_max_rows,
                                       "decided_at": "epoch refreshes with a capacity change"}
        stats["cleanup_ms"] = (completed - finished) * 1000
        stats["evaluate_wall_ms"] = (completed - entered) * 1000
        if self.epoch == 1:
            freeze_long_lived_objects()
        return prediction

    def gather(self, prediction: DistributedPrediction, *, root: int = 0) -> dict | None:
        """Diagnostic/output sampling only; never called by evaluate or the MD loop."""
        global_root = validate_root(self.context, root)
        state = prediction.state
        value = {"ids": state.ids.detach().cpu(), "positions": state.positions.detach().cpu(),
                 "numbers": state.numbers.detach().cpu(), "forces": prediction.forces.cpu(),
                 **{key: tensor.detach().cpu() for key, tensor in state.payload.items()}}
        if self.context.world_size == 1:
            values = [value]
        else:
            values = [None] * self.context.world_size if self.context.rank == root else None
            dist.gather_object(value, values, dst=global_root, group=self.context.control_group)
        if self.context.rank != root:
            return None
        result = {key: torch.cat([v[key] for v in values]) for key in value}
        order = result["ids"].argsort()
        return {key: tensor[order] for key, tensor in result.items()}
