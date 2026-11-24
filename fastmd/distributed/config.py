"""Small, deliberately restrictive configuration for distributed inference."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import warnings

from fastmd.config import CapacityConfig, ConfigurationWarning, InferenceConfig
from .auto_execution import AutoExecution


@dataclass(frozen=True)
class ParallelConfig:
    """Spatial decomposition; process count and device binding belong to torchrun.

    ``single_rank_engine`` is ``"single"`` only: one CUDA rank always runs the
    single-GPU engine; the multi-rank protocol runs at 2+ ranks, and on CPU at
    any rank count. ``"plugin_loopback"`` (the multi-rank protocol at one
    CUDA rank, the rollback of unified step 1) was removed in the rollback cleanup and is
    rejected for one version.
    ``auto_execution`` (with ``InferenceConfig.execution=None``) chooses eager
    or model_graph from the per-rank triplet rows; see ``AutoExecution``.
    ``control_plane`` is ``"nccl"`` only: P>1 ranks agree on host control
    metadata with fixed-shape rounds on the data group (NCCL on CUDA, the
    Gloo data group on CPU); Gloo carries error strings only after a
    failure. One rank never communicates. ``"gloo"`` (one Gloo
    ``all_gather_object`` per round, the pre-3.1a path kept one version as
    the rollback) was removed in the rollback cleanup and is rejected for one version. Every
    P>1 evaluator runs the P>1 model guard (``DistributedModelGuard``), which
    follows the single-GPU guard's rules and so also rejects a callable
    attribute rebound after construction (e.g. ``nn.Module.compile()``'s
    ``_compiled_call_impl``) or a registry value whose type changed, both of
    which the unguarded one-rank program stamp admits. It rejects an
    in-place edit that only a full scan sees later in the evaluation (run
    boundary or after the replay), and a program change the model's own
    code makes during a model_graph (re)capture in that evaluation rather
    than the next one (``DistributedModelGuard`` lists the differences).
    With peers that fail, NCCL waits end through the peers' fail_fast
    announcement or deadlines, not a closed Gloo connection
    (docs/distributed_inference.md).
    """

    domain_grid: str | tuple[int, int, int] = "auto"
    # Deprecated alias of InferenceConfig.optimization_profile (one version):
    # "native" -> "topology", "torch_reference" -> "generic"; None (default):
    # the profile decides. Kept as given; the evaluator reports the resolved
    # choice (config_summary: resolved_profile, effective_local_backend).
    local_backend: str | None = None
    # Only "single" since the rollback cleanup ("plugin_loopback" is rejected); the field goes in the next version.
    single_rank_engine: str = "single"
    auto_execution: AutoExecution | None = None
    # "local": a rank's own exchange rows are a fused on-device gather-scatter
    # and only remote rows go through NCCL; "nccl" keeps the original lowering.
    exchange_self_rows: str = "local"
    # Only "nccl" since the rollback cleanup ("gloo" is rejected); the field goes in the next version.
    control_plane: str = "nccl"
    # How per-peer-split payload exchanges travel (A1): "all_to_all", or
    # "neighbor_p2p" (one grouped P2P over the peers with rows, so only
    # neighbor domains communicate); "auto" = all_to_all up to 4 ranks, and
    # above them neighbor_p2p unless every rank's neighbors are all other
    # ranks (communicator.resolve_payload_transport). Rows and sums are
    # identical either way.
    payload_transport: str = "auto"
    # P>1 model_graph capacity: the single-GPU JointCapacityPolicy tiers plus
    # hysteresis agreed by CaptureCoordinator (unified step 3.2). True (the
    # pre-3.2 policy of 12.5%/25% headroom, which only worked with the rows
    # exchange) was removed in step 4.2 and is rejected for one version.
    legacy_capacity_policy: bool = False
    # P>1 model exchange (step A2, fastmd/models/fastmd/distributed/bond_partial.py):
    # "bond_partial" keeps every bond with an owned end on its rank (boundary
    # bonds on both ends) and exchanges one line partial per short boundary
    # bond with the next ghost rows (two per interaction block). It is the
    # only scheme since step 4.2; "rows" (triplet and bond rows routed to
    # their owners) was removed and is rejected for one version.
    exchange_scheme: str = "bond_partial"
    # P>1 topology (unified step 3.4, fastmd/distributed/epoch.py): "epoch"
    # builds the local graph once per skin epoch with every cutoff widened by
    # ``topology_skin`` (Angstrom) and filters it exactly on the device every
    # step with the single-GPU fixed-capacity builder (CUDA only). "auto"
    # (default): the epoch on CUDA, above 4 ranks with the skin clamped to
    # the cell's face spacings (one periodic image per axis:
    # min(h) - cutoff - 1e-3 A, h the lengths of an orthogonal cell or the
    # heights across a skew cell's lattice vectors;
    # epoch.effective_skin); CPU runs use an internal
    # reference topology rebuilt every evaluation. "rebuild" (the CUDA
    # rollback) was removed in step 4.2 and is rejected for one version.
    topology_engine: str = "auto"
    topology_skin: float = 1.0
    # P>1 forces: one autograd.grad over exchanges that are autograd Functions
    # threaded on a token chain (fastmd/engine/grad.py; steps 4.1/4.2). Since
    # 4.2 "autograd" is the only mechanism: the StageTape ("tape") was deleted
    # and the value is rejected with a clear error for one version. P=1
    # ignores it.
    grad_mechanism: str = "autograd"
    # Debug: after every P>1 model_graph replay, agree its identity (replay
    # count, graph generation, replayed global energy bits) across ranks
    # (DistributedGraphRunner.replay_check). Off: no cost. Captured collectives
    # are invisible to the NCCL watchdog, and a mispaired replay can otherwise
    # hang the step or return wrong forces without an error.
    debug_replay_check: bool = False

    def __post_init__(self):
        if self.local_backend not in {None, "torch_reference", "native"}:
            raise ValueError("local_backend must be None, 'torch_reference' or 'native'")
        if self.single_rank_engine == "plugin_loopback":
            raise ValueError("single_rank_engine='plugin_loopback' was removed (the multi-rank protocol at one "
                             "CUDA rank, kept one version as the rollback of unified step 1): one CUDA rank "
                             "always runs the single-GPU engine; the multi-rank protocol runs at 2+ ranks, and "
                             "on CPU at any rank count")
        if self.single_rank_engine != "single":
            raise ValueError("single_rank_engine must be 'single'")
        if self.auto_execution is not None and not isinstance(self.auto_execution, AutoExecution):
            raise TypeError("auto_execution must be None or AutoExecution")
        if self.exchange_self_rows not in {"nccl", "local"}:
            raise ValueError("exchange_self_rows must be 'nccl' or 'local'")
        if self.grad_mechanism == "tape":
            raise ValueError("grad_mechanism='tape' was removed in step 4.2 (the StageTape is deleted); "
                             "'autograd' is the only P>1 gradient mechanism")
        if self.grad_mechanism != "autograd":
            raise ValueError("grad_mechanism must be 'autograd'")
        if type(self.debug_replay_check) is not bool:
            raise ValueError("debug_replay_check must be a bool")
        if self.topology_engine == "rebuild":
            raise ValueError("topology_engine='rebuild' was removed in step 4.2 (superseded by the topology "
                             "epoch, the default since #10); CPU runs use the internal reference topology "
                             "automatically")
        if self.topology_engine not in {"auto", "epoch"}:
            raise ValueError("topology_engine must be 'auto' or 'epoch'")
        if not (isinstance(self.topology_skin, (int, float)) and math.isfinite(self.topology_skin)
                and self.topology_skin > 0):
            raise ValueError("topology_skin must be a finite positive number of Angstrom")
        check_control_plane(self.control_plane)
        if self.payload_transport not in {"auto", "all_to_all", "neighbor_p2p"}:
            raise ValueError("payload_transport must be 'auto', 'all_to_all' or 'neighbor_p2p'")
        if type(self.legacy_capacity_policy) is not bool:
            raise ValueError("legacy_capacity_policy must be a bool")
        if self.legacy_capacity_policy:
            raise ValueError("legacy_capacity_policy=True was removed in step 4.2 (it required "
                             "exchange_scheme='rows'); P>1 model_graph uses the tiered capacity policy")
        if self.exchange_scheme == "rows":
            raise ValueError("exchange_scheme='rows' was removed in step 4.2 (superseded by 'bond_partial', "
                             "the default since #6); 'bond_partial' is the only P>1 exchange")
        if self.exchange_scheme != "bond_partial":
            raise ValueError("exchange_scheme must be 'bond_partial'")
        if self.domain_grid != "auto":
            grid = self.domain_grid
            if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                    or any(type(x) is not int or x < 1 for x in grid)):
                raise ValueError("domain_grid must be 'auto' or three positive integers")
            object.__setattr__(self, "domain_grid", tuple(grid))


def check_control_plane(control_plane) -> None:
    """``ParallelConfig.control_plane`` and ``ExchangeContext(control_plane=...)``: "nccl" only."""
    if control_plane == "gloo":
        raise ValueError("control_plane='gloo' was removed (the pre-3.1a Gloo object rounds, kept one version "
                         "as the rollback of the NCCL control plane): 'nccl' is the only multi-rank control "
                         "plane; CPU runs use it on their Gloo data group")
    if control_plane != "nccl":
        raise ValueError("control_plane must be 'nccl'")


# One knob (2026-09-30): InferenceConfig.optimization_profile selects the
# local operators at P>1 as at P=1. ParallelConfig.local_backend is kept one
# version as a deprecated alias of it.
_LOCAL_BACKEND_PROFILES = {"torch_reference": "generic", "native": "topology"}
_PROFILE_LOCAL_BACKENDS = {"generic": "torch_reference", "topology": "native",
                           # legacy is single-GPU only; one CUDA rank runs the single-GPU engine
                           "legacy": "native"}


def resolve_profile(config: InferenceConfig | None, local_backend: str | None, *,
                    single_rank: bool, warn: bool = True) -> tuple[str, str, str]:
    """(effective profile, P>1 local backend, source) of a distributed request.

    ``optimization_profile`` decides (default "topology"). The deprecated
    ``local_backend`` alias warns and maps native -> topology, torch_reference
    -> generic. Until 2026-09-30 the distributed API required the request
    "generic" (or no config) and chose the operators with local_backend, so
    "generic" or no config next to an explicit alias is that old spelling and
    the alias decides; any other disagreement is an error at P>1 (one rank
    always ran an explicit profile as given). "legacy" (environment-driven kernel options,
    process local) is single-GPU only: one CUDA rank accepts it, P>1 rejects it.
    """
    from fastmd.config import DEFAULT_PROFILE, ConfigurationDeprecationWarning
    requested = None if config is None else config.optimization_profile
    profile, source = (requested, "optimization_profile") if requested else (DEFAULT_PROFILE, "default")
    if local_backend is not None:
        mapped = _LOCAL_BACKEND_PROFILES[local_backend]
        if warn:
            import warnings
            warnings.warn(f"ParallelConfig.local_backend={local_backend!r} (--local-backend) is deprecated and "
                          f"will be removed in the next version: use optimization_profile={mapped!r} (--profile)",
                          ConfigurationDeprecationWarning, stacklevel=3)
        if requested is None or requested == "generic":
            profile, source = mapped, f"deprecated local_backend {local_backend}->{mapped}"
        elif requested != mapped and single_rank:
            profile = requested
            # One rank always ran an explicit topology/legacy profile as given.
            source = f"optimization_profile (deprecated local_backend={local_backend!r} ignored at one rank)"
        elif requested != mapped:
            raise ValueError(f"optimization_profile={requested!r} conflicts with the deprecated "
                             f"ParallelConfig.local_backend={local_backend!r} (= {mapped!r}); drop local_backend")
    if profile == "legacy" and not single_rank:
        raise ValueError("optimization_profile='legacy' is single-GPU only (its MATRIS_* environment options "
                         "are process local and are not agreed across ranks): use 'topology' (default) or 'generic'")
    return profile, _PROFILE_LOCAL_BACKENDS[profile], source


def single_rank_profile(requested: str | None, local_backend: str | None) -> tuple[str, str]:
    """Effective single-engine profile and where it came from (``resolve_profile`` at one rank)."""
    config = InferenceConfig(optimization_profile=requested, report=False)
    profile, _, source = resolve_profile(config, local_backend, single_rank=True, warn=False)
    return profile, source


_DISTRIBUTED_RESOLVED: dict = {}
# Per backend: the wide GEMMs' configured row threshold, which the P>1 scope sets to 0 and the topology epoch's
# rank-uniform layout restores (``triton_wide_gemm.rank_uniform``).
_WIDE_GEMM_MIN_ROWS: dict = {}


# Row thresholds of the kernels that produce replicated bond rows under the
# bond_partial scheme, and the switch that enables each kernel at all.
_RANK_UNIFORM_THRESHOLDS = (
    ("MATRIS_GATED_LN_FUSE_MIN_ROWS", None),
    ("MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS", "MATRIS_FUSED_INDEXED_CAT_LINEAR"),
    ("MATRIS_TRITON_WIDE_GEMM_MIN_ROWS", "MATRIS_TRITON_WIDE_GEMM"),
)


def distributed_resolved(local_backend: str):
    """The resolved configuration whose ``scope()`` a P>1 forward runs in (§4.6).

    ``local_backend`` is the resolved P>1 backend of the request's profile
    (torch_reference = generic, native = topology). Kernel options only: no execution, checkpoint
    or precision intent (the evaluator applies the requested precision
    itself). One object per backend, so the evaluator's scope and the one
    around each forward nest without rebinding.

    The bond_partial exchange: both ends of a boundary bond compute its rows
    inside batches of different sizes, so a kernel chosen by the local row
    count could differ between the two replicas. Every row threshold of a
    kernel that produces bond rows (fused gated LayerNorm, indexed
    projection, wide GEMMs) is 0 wherever that kernel is enabled at all: the
    choice is then the same on every rank. On the topology epoch's agreed
    layout, whose row counts other than the atom rows are the same on every
    rank, the wide GEMMs' threshold applies again, moved off the ranks'
    atom-row range (``wide_gemm_min_rows``, ``triton_wide_gemm.rank_uniform``).
    With the native backend the refinement reductions use the CSR segment sum (D6).
    """
    resolved = _DISTRIBUTED_RESOLVED.get(local_backend)
    if resolved is None:
        from fastmd.config import resolve_config

        profile = _LOCAL_BACKEND_PROFILES[local_backend]
        options = resolve_config(InferenceConfig(execution="eager", optimization_profile=profile,
                                                 checkpoint="off", report=False)).kernel_options
        overrides = {name: 0 for name, switch in _RANK_UNIFORM_THRESHOLDS
                     if options[name] < 1_000_000_000 and (switch is None or options[switch])}
        _WIDE_GEMM_MIN_ROWS[local_backend] = int(options["MATRIS_TRITON_WIDE_GEMM_MIN_ROWS"])
        if local_backend == "native":
            # The refinement reductions over the local rows (and, padded,
            # the sink rows) take the deterministic CSR segment sum (D6)
            # with the sink as the skipped segment, not atomics that all
            # padding rows serialize on. Neither reduction is replicated.
            overrides["MATRIS_CSR_SEGMENT_SUM"] = True
        resolved = resolve_config(InferenceConfig(
            execution="eager", optimization_profile=profile, checkpoint="off", report=False,
            expert_overrides=overrides))
        _DISTRIBUTED_RESOLVED[local_backend] = resolved
    return resolved


def wide_gemm_min_rows(local_backend: str) -> int:
    """The wide GEMMs' configured row threshold of the backend's profile (before the P>1 scope's 0)."""
    distributed_resolved(local_backend)
    return _WIDE_GEMM_MIN_ROWS[local_backend]


# InferenceConfig fields the distributed evaluator does not read, warned once per process (rank 0).
_IGNORED_FIELDS_WARNED: set[str] = set()


def _warn_ignored_fields(config: InferenceConfig, *, warn: bool) -> None:
    ignored = {}
    if config.topology_reuse != "auto":
        ignored["topology_reuse"] = (
            f"InferenceConfig.topology_reuse={config.topology_reuse!r} is a MatRISCalculator option and has no "
            "effect in fastmd.distributed: the P=1 single-GPU engine rebuilds the graph every evaluation, P>1 on "
            "CUDA keeps a topology epoch (ParallelConfig.topology_skin, --topology-skin)")
    if config.capacity.candidate_skin != CapacityConfig().candidate_skin:
        ignored["candidate_skin"] = (
            f"InferenceConfig.capacity.candidate_skin={config.capacity.candidate_skin} (calculator skin reuse, "
            "GPU-resident MD whole_step) has no effect in fastmd.distributed; the P>1 topology skin is "
            "ParallelConfig.topology_skin (--topology-skin)")
    for field, message in ignored.items():
        if warn and field not in _IGNORED_FIELDS_WARNED:
            _IGNORED_FIELDS_WARNED.add(field)
            warnings.warn(message, ConfigurationWarning, stacklevel=4)


def inference_config(config: InferenceConfig | None, *, single_rank: bool = False,
                     local_backend: str | None = None, warn: bool = True) -> tuple[InferenceConfig, str, str]:
    """Reject unsupported requests, without silently selecting another backend.

    Returns (request with its effective profile, P>1 local backend, profile
    source); see ``resolve_profile``. ``single_rank`` admits the single-GPU
    engine's wider set (legacy, checkpoint with eager). ``execution="auto"``
    is returned as None (the evaluator then applies ParallelConfig.auto_execution,
    the default policy if unset).
    """
    if config is None:
        config = InferenceConfig(execution="eager", checkpoint="off", report=False)
    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig")
    if config.execution not in (None, "auto", "eager", "model_graph"):
        raise ValueError("distributed inference requires execution='auto', 'eager' or 'model_graph'")
    profile, backend, source = resolve_profile(config, local_backend, single_rank=single_rank, warn=warn)
    _warn_ignored_fields(config, warn=warn)
    if config.execution == "auto":
        config = replace(config, execution=None)
    if profile != config.optimization_profile:
        config = replace(config, optimization_profile=profile)
    if single_rank:
        if config.checkpoint != "off" and config.execution == "model_graph":
            raise ValueError("the single-rank engine rejects checkpoint with execution='model_graph': "
                             "checkpoint recomputation inside CUDA graph capture is not validated "
                             "(unified_engine_design §5.6); use checkpoint='off' or execution='eager'")
        # Unchanged restriction: per-operator overrides stay a single-GPU API.
        if config.expert_overrides:
            raise ValueError("distributed inference does not accept expert_overrides")
        return config, backend, source
    if config.checkpoint != "off":
        raise ValueError("distributed inference currently requires checkpoint='off'")
    if config.expert_overrides:
        raise ValueError("distributed inference does not accept expert_overrides")
    return config, backend, source
