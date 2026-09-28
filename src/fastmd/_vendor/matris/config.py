"""Typed, scoped inference configuration, without importing torch or CUDA.

``legacy`` preserves the historical defaults and environment overrides.
``topology`` and ``generic`` are reproducible profiles: registered environment
variables are ignored in favour of registry defaults plus the selected profile.
Use ``expert_overrides`` for intentional per-operator changes. Profile names
describe requested optimizations, not a promise of kernel eligibility.

Scopes serialize managed calls while temporarily binding existing module
constants. They do not mutate ``os.environ``. Unmanaged raw model calls from
other threads must not run concurrently with a configured inference scope.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import cached_property
import hashlib
import json
import math
import os
import re
import sys
import threading
from types import MappingProxyType
from typing import Any
import warnings


class ConfigurationWarning(UserWarning):
    """A legacy or ineffective configuration needs the caller's attention."""


def parse_bool(value: Any, *, name: str = "boolean") -> bool:
    """Accept common explicit boolean spellings; never use string truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "on", "yes"}:
            return True
        if normalized in {"0", "false", "off", "no"}:
            return False
    raise ValueError(f"{name}: expected true/false, 1/0, on/off or yes/no; got {value!r}")


@dataclass(frozen=True)
class OptionSpec:
    name: str
    kind: str
    default: Any
    category: str
    description: str
    bindings: tuple[tuple[str, str], ...] = ()
    choices: tuple[Any, ...] = ()
    minimum: float | None = None

    def parse(self, value: Any) -> Any:
        if value is None:
            if self.default is None:
                return None
            raise ValueError(f"{self.name}: None is not supported")
        if self.kind == "bool":
            result = parse_bool(value, name=self.name)
        elif self.kind == "int":
            if isinstance(value, bool) or not re.fullmatch(r"[+-]?\d+", str(value).strip()):
                raise ValueError(f"{self.name}: expected an integer, got {value!r}")
            result = int(value)
        elif self.kind == "float":
            if isinstance(value, bool):
                raise ValueError(f"{self.name}: expected a finite number, got {value!r}")
            try:
                result = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.name}: expected a finite number, got {value!r}") from exc
            if not math.isfinite(result):
                raise ValueError(f"{self.name}: expected a finite number, got {value!r}")
        elif self.kind == "pair":
            parts = value.split(",") if isinstance(value, str) else value
            if not isinstance(parts, (tuple, list)) or len(parts) != 2:
                raise ValueError(f"{self.name}: expected two positive integers, e.g. '128,256'")
            if any(isinstance(p, bool) or not re.fullmatch(r"\+?\d+", str(p).strip()) for p in parts):
                raise ValueError(f"{self.name}: expected two positive integers")
            result = tuple(int(p) for p in parts)
            if min(result) <= 0:
                raise ValueError(f"{self.name}: both dimensions must be positive")
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{self.name}: expected a nonempty string")
            result = value.strip().lower()
        if self.minimum is not None and result < self.minimum:
            raise ValueError(f"{self.name}: must be >= {self.minimum:g}, got {result!r}")
        if self.choices and result not in self.choices:
            raise ValueError(f"{self.name}: expected one of {self.choices!r}, got {result!r}")
        return result

    def format(self, value: Any) -> str | None:
        if value is None:
            return None
        if self.kind == "bool":
            return "1" if value else "0"
        if self.kind == "pair":
            return ",".join(str(part) for part in value)
        return str(value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _option(
    suffix: str, kind: str, default: Any, category: str, description: str,
    module: str = "", attribute: str = "", *, choices: tuple[Any, ...] = (),
    minimum: float | None = None,
) -> OptionSpec:
    binding = (("fastmd._vendor.matris." + module, attribute),) if module else ()
    return OptionSpec("MATRIS_" + suffix, kind, default, category, description, binding, choices, minimum)


# This registry is the source of truth for defaults, validation, documentation
# and loaded-module bindings. Optional tuning values use None to mean auto.
_OPTIONS = [
    _option("GATED_LN_FUSE_MIN_ROWS", "int", 10000, "model", "Minimum rows for fused gated LayerNorm.", "model.functions", "_LN_FUSE_THRESH", minimum=0),
    _option("FUSED_GATED_LN_RESIDUAL", "bool", True, "model", "Fuse gated LayerNorm with residual when eligible.", "model.functions", "_FUSED_GATED_LN_RESIDUAL"),
    _option("FUSED_SEGMENT_SOFTMAX", "bool", True, "model", "Use fused segmented softmax.", "model.functions", "_FUSED_SEGMENT_SOFTMAX"),
    _option("FUSED_INDEXED_CAT_LINEAR", "bool", False, "model", "Enable indexed first-linear projection; required by merged projections.", "model.functions", "_INDEXED_CAT_LINEAR_FUSE"),
    _option("FUSED_INDEXED_CAT_LINEAR_BACKEND", "str", "decomposed_indexed_silu", "model", "Indexed projection implementation.", "model.functions", "_INDEXED_CAT_LINEAR_BACKEND", choices=("decomposed_indexed_silu", "cutedsl_dense_silu")),
    _option("INDEXED_CAT_LINEAR_MIN_ROWS", "int", 20000, "model", "Minimum rows for indexed projections.", "model.functions", "_INDEXED_CAT_LINEAR_MIN_ROWS", minimum=0),
    _option("FUSED_WEIGHTED_SEGMENT_EAGER", "bool", False, "model", "Allow fused weighted reduction outside graph capture.", "model.interaction_block", "_FUSED_WEIGHTED_SEGMENT_EAGER"),
    _option("FUSED_GATHER_CAT_EAGER", "bool", False, "model", "Allow fused gather/concatenate outside graph capture.", "model.interaction_block", "_FUSED_GATHER_CAT_EAGER"),
    _option("FUSED_DIRECTED_PAIR_AGGREGATE", "bool", True, "model", "Fuse directed-pair aggregation.", "model.interaction_block", "_FUSED_DIRECTED_PAIR_AGGREGATE"),
    _option("FUSED_DIRECTED_PAIR_EXPAND", "bool", True, "model", "Fuse directed-pair expansion.", "model.interaction_block", "_FUSED_DIRECTED_PAIR_EXPAND"),
    _option("FUSED_SEGMENT_ATTENTION", "bool", True, "model", "Enable segmented attention fusion.", "model.interaction_block", "_FUSED_SEGMENT_ATTENTION_ENABLED"),
    _option("FUSED_SEGMENT_ATTENTION_EAGER", "bool", False, "model", "Allow segmented attention fusion outside graph capture.", "model.interaction_block", "_FUSED_SEGMENT_ATTENTION_EAGER"),
    _option("FUSED_SEGMENT_ATTENTION_MIN_ROWS", "int", 0, "model", "Minimum rows for fused segmented attention.", "model.interaction_block", "_FUSED_SEGMENT_ATTENTION_MIN_ROWS", minimum=0),
    _option("FUSED_PAIRED_SEGMENT_ATTENTION", "bool", True, "model", "Share paired attention work when eligible.", "model.interaction_block", "_FUSED_PAIRED_SEGMENT_ATTENTION"),
    _option("FUSED_SEGMENT_ATTENTION_REUSE_CSR", "bool", True, "model", "Reuse verified CSR for segmented attention.", "model.interaction_block", "_FUSED_SEGMENT_ATTENTION_REUSE_CSR"),
    _option("FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR", "bool", True, "model", "Use sorted-target CSR attention.", "model.interaction_block", "_FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR"),
    _option("FUSED_LINE_ENVELOPE", "bool", True, "model", "Enable fused line-graph envelope.", "model.interaction_block", "_FUSED_LINE_ENVELOPE_ENABLED"),
    _option("FUSED_LINE_ENVELOPE_EAGER", "bool", False, "model", "Allow fused line envelope outside graph capture.", "model.interaction_block", "_FUSED_LINE_ENVELOPE_EAGER"),
    _option("FUSED_RESIDUAL_ADD", "bool", True, "model", "Fuse frozen-weight residual addition.", "model.interaction_block", "_FUSED_RESIDUAL_ADD_ENABLED"),
    _option("FUSED_RESIDUAL_ADD_MIN_ROWS", "int", 0, "model", "Minimum rows for residual-add fusion.", "model.interaction_block", "_FUSED_RESIDUAL_ADD_MIN_ROWS", minimum=0),
    _option("MERGED_ATTENTION_PROJECTIONS", "bool", True, "model", "Merge compatible attention projections; requires indexed-linear fusion and equal input widths.", "model.interaction_block", "_MERGED_ATTN_PROJECTIONS"),
    _option("MERGED_REFINEMENT_PROJECTION", "bool", True, "model", "Merge compatible refinement projections; requires indexed-linear fusion.", "model.interaction_block", "_MERGED_REFINEMENT_PROJECTION"),
    _option("SKIP_DEAD_THREEBODY_TAIL", "bool", True, "model", "Skip a three-body tail proven unused by the output.", "model.interaction_block", "_SKIP_DEAD_THREEBODY_TAIL"),
    _option("FUSED_THREEBODY_LINEAR", "bool", True, "model", "Enable fused three-body basis projection.", "model.feature_embed", "_FUSED_THREEBODY_LINEAR_ENABLED"),
    _option("FUSED_THREEBODY_LINEAR_EAGER", "bool", False, "model", "Allow three-body projection fusion outside graph capture.", "model.feature_embed", "_FUSED_THREEBODY_LINEAR_EAGER"),
    _option("FUSED_THREEBODY_LINEAR_MIN_ROWS", "int", 60000, "model", "Minimum rows for three-body projection fusion.", "model.feature_embed", "_FUSED_THREEBODY_LINEAR_MIN_ROWS", minimum=0),
    _option("FUSED_THREEBODY_BASIS", "bool", True, "model", "Enable fused three-body Fourier basis.", "model.feature_embed", "_FUSED_THREEBODY_BASIS_ENABLED"),
    _option("FUSED_THREEBODY_BASIS_EAGER", "bool", False, "model", "Allow three-body basis fusion outside graph capture.", "model.feature_embed", "_FUSED_THREEBODY_BASIS_EAGER"),
    _option("FUSED_THREEBODY_BASIS_MIN_ROWS", "int", 60000, "model", "Minimum rows for three-body basis fusion.", "model.feature_embed", "_FUSED_THREEBODY_BASIS_MIN_ROWS", minimum=0),
    _option("UNDIRECTED_EDGE_INIT", "bool", True, "model", "Initialize equivalent edge pairs only once.", "model.feature_embed", "_UNDIRECTED_EDGE_INIT"),
    _option("COMPILED_LOWERINGS", "bool", False, "kernel", "Opt into torch.compile operator lowerings.", "model.op.compiled_lowerings", "COMPILED_LOWERINGS_ENABLED"),
    _option("COMPILED_LOWERINGS_MODE", "str", "max-autotune-no-cudagraphs", "kernel", "torch.compile mode for operator lowerings.", "model.op.compiled_lowerings", "_COMPILE_MODE", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")),
    _option("DECOMPOSED_INDEXED_SILU_BM", "int", 2, "kernel", "Indexed SiLU forward row tile.", "model.op.decomposed_indexed_cat_silu_linear", "_EPILOGUE_BM", minimum=1),
    _option("DECOMPOSED_INDEXED_SILU_BWD_BM", "int", 8, "kernel", "Indexed SiLU backward row tile.", "model.op.decomposed_indexed_cat_silu_linear", "_EPILOGUE_BWD_BM", minimum=1),
    _option("DECOMPOSED_INDEXED_SORTED_A_PAIR_BWD", "bool", True, "kernel", "Use sorted paired-index backward.", "model.op.decomposed_indexed_cat_silu_linear", "_SORTED_A_PAIR_BWD"),
    _option("DECOMPOSED_INDEXED_GEMM_BACKEND", "str", "torch", "kernel", "Dense GEMM backend for decomposed projections.", "model.op.decomposed_indexed_cat_silu_linear", "_GEMM_BACKEND", choices=("torch", "cutedsl")),
    _option("DECOMPOSED_INDEXED_DEBUG_CHECKS", "bool", False, "kernel", "Expensive index checks and debug messages."),
    _option("DECOMPOSED_INDEXED_CUTEDSL_FALLBACK", "bool", True, "kernel", "Fall back to torch GEMM if CuTeDSL is unavailable."),
    _option("FUSED_RESIDUAL_ADD_BM", "int", 4, "kernel", "Residual-add row tile.", "model.op.triton_residual_add", "_BM", minimum=1),
    _option("FUSED_RESIDUAL_ADD_BD", "int", 128, "kernel", "Residual-add feature tile.", "model.op.triton_residual_add", "_BD", minimum=1),
    _option("THREEBODY_FOURIER_BM", "int", 128, "kernel", "Three-body Fourier row tile.", "model.op.triton_threebody_fourier", "_BM", minimum=1),
    _option("LINE_ENVELOPE_BM", "int", 4, "kernel", "Line-envelope row tile.", "model.op.triton_line_envelope", "_BM", minimum=1),
    _option("LINE_ENVELOPE_BD", "int", 128, "kernel", "Line-envelope feature tile.", "model.op.triton_line_envelope", "_BD", minimum=1),
    _option("DIRECTED_PAIR_AGG_BM", "int", 8, "kernel", "Directed-pair aggregation row tile.", "model.op.triton_directed_pair_aggregate", "_PAIR_AGG_BM", minimum=1),
    _option("DIRECTED_PAIR_AGG_BD", "int", 128, "kernel", "Directed-pair aggregation feature tile.", "model.op.triton_directed_pair_aggregate", "_PAIR_AGG_BD", minimum=1),
    _option("FUSED_SEGMENT_ATTENTION_NUM_WARPS", "int", 4, "kernel", "Warps per segmented-attention program.", "model.op.triton_segment_attention", "_NUM_WARPS", choices=(1, 2, 4, 8, 16, 32)),
    _option("FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BM", "int", 8, "kernel", "Paired attention backward row tile.", "model.op.triton_segment_attention", "_PAIRED_BWD_BM", minimum=1),
    _option("FUSED_PAIRED_SEGMENT_ATTENTION_BWD_BD", "int", 128, "kernel", "Paired attention backward feature tile.", "model.op.triton_segment_attention", "_PAIRED_BWD_BD", minimum=1),
    _option("CUTEDSL_SILU_SPLIT_BM", "int", 2, "kernel", "CuTeDSL split-SiLU row tile.", "model.op.cutedsl_dense_silu_linear", "_SPLIT_BM", minimum=1),
    _option("CUTEDSL_SILU_TUNE_WARMUP", "int", 2, "kernel", "CuTeDSL SiLU autotune warmup iterations.", minimum=0),
    _option("CUTEDSL_SILU_TUNE_REPEAT", "int", 5, "kernel", "CuTeDSL SiLU autotune measured iterations.", minimum=1),
    _option("CUTEDSL_SILU_EPILOGUE", "bool", False, "kernel", "Use experimental fused CuTeDSL SiLU epilogue."),
    _option("CUTEDSL_USE_TVM_FFI", "bool", False, "kernel", "Use optional TVM FFI for CuTeDSL launch."),
    _option("CUTEDSL_MMA_TILER_MN", "pair", None, "kernel", "Manual GEMM M,N tile; None selects automatic tuning."),
    _option("CUTEDSL_CLUSTER_SHAPE_MN", "pair", None, "kernel", "Manual GEMM M,N cluster shape."),
    _option("CUTEDSL_USE_2CTA", "bool", None, "kernel", "Manual two-CTA mode; None selects automatically."),
    _option("CUTEDSL_SWIZZLE_SIZE", "int", None, "kernel", "Manual scheduler swizzle size.", choices=(1, 2, 4, 8)),
    _option("CUTEDSL_RASTER_ALONG", "str", None, "kernel", "Manual scheduler raster direction.", choices=("m", "n")),
    _option("CUTEDSL_AUTOTUNE_SCHEDULER", "bool", True, "kernel", "Tune GEMM scheduler choices."),
    _option("CUTEDSL_AUTOTUNE", "bool", True, "kernel", "Enable CuTeDSL tactic autotuning."),
    _option("CUTEDSL_AUTOTUNE_EXHAUSTIVE", "bool", False, "kernel", "Search the larger experimental GEMM tactic space."),
    _option("CUTEDSL_TUNE_WARMUP", "int", 10, "kernel", "CuTeDSL GEMM autotune warmup iterations.", minimum=0),
    _option("CUTEDSL_TUNE_REPEAT", "int", 50, "kernel", "CuTeDSL GEMM autotune measured iterations.", minimum=1),
    _option("CUTEDSL_TUNE_TIMER", "str", "cupti", "kernel", "Autotune timing implementation.", choices=("cupti", "cupti_subprocess", "cupti_inline", "cuda_event", "event", "events")),
    _option("CUTEDSL_TUNE_CUPTI_STRICT", "bool", False, "kernel", "Reject CUPTI errors instead of timer fallback."),
    _option("CUTEDSL_TUNE_CUPTI_TIMEOUT_S", "float", 300.0, "kernel", "Timeout in seconds for isolated CUPTI tuning.", minimum=0.001),
    _option("WHOLE_STEP_TIER_CACHE", "bool", False, "capacity", "Keep multiple whole-step capacity tiers.", "applications.gpu_md", "_WHOLE_STEP_TIER_CACHE"),
    _option("WHOLE_STEP_MAX_TIERS", "int", 3, "capacity", "Maximum cached whole-step capacity tiers.", "applications.gpu_md", "_WHOLE_STEP_MAX_TIERS", minimum=1),
    _option("TIER_HYSTERESIS_WINDOW", "int", 2000, "capacity", "Demand history length for tier hysteresis.", "applications.gpu_md", "_TIER_HYSTERESIS_WINDOW", minimum=1),
    _option("TIER_HYSTERESIS_SIGMA", "float", 3.0, "capacity", "Demand standard-deviation margin for tier sizing.", "applications.gpu_md", "_TIER_HYSTERESIS_SIGMA", minimum=0),
    _option("TIER_CHECK_INTERVAL", "int", 256, "capacity", "Replays between tier shrink checks.", "applications.gpu_md", "_TIER_CHECK_INTERVAL", minimum=1),
    _option("TIER_DOWN_COOLDOWN", "int", 4000, "capacity", "Shrink cooldown; legacy unset default is twice hysteresis window.", "applications.gpu_md", "_TIER_DOWN_COOLDOWN", minimum=0),
    _option("TIER_MIN_FREE_GIB", "float", 6.0, "capacity", "Free-memory floor for capacity-tier captures in GiB.", "applications.gpu_md", "_TIER_MIN_FREE_GIB", minimum=0),
    _option("CAPACITY_THERMAL_FACTOR", "float", 1.0, "capacity", "Initial capacity multiplier for thermal expansion.", "applications.gpu_md", "_CAPACITY_THERMAL_FACTOR", minimum=1),
    _option("LINE_GRAPH_TRITON", "bool", True, "graph", "Use Triton line-graph construction.", "graph.gpu_graph_builder", "_LINE_GRAPH_TRITON"),
    _option("LINE_GRAPH_TRITON_BLOCK", "int", 256, "graph", "Line-graph builder block size.", "graph.gpu_graph_builder", "_LINE_GRAPH_TRITON_BLOCK", minimum=1),
    _option("PRESORTED_BUILD", "bool", True, "graph", "Use presorted graph construction.", "graph.gpu_graph_builder", "_PRESORTED_BUILD"),
    _option("NARROW_SORT_KEYS", "bool", True, "graph", "Use narrow sort keys when bounds permit."),
    _option("TOPOLOGY_VERIFY", "str", "builder", "graph", "Topology verification strength.", choices=("builder", "sampled", "exhaustive")),
    _option("M3GNET_INDEXED_AFFINE", "bool", True, "integration", "Enable M3GNet indexed-affine optimization."),
    _option("M3GNET_INDEXED_EDGE", "bool", True, "integration", "Enable M3GNet edge indexed-affine path."),
    _option("M3GNET_INDEXED_NODE", "bool", True, "integration", "Enable M3GNet node indexed-affine path."),
]
OPTION_REGISTRY: Mapping[str, OptionSpec] = MappingProxyType({spec.name: spec for spec in _OPTIONS})
INTERNAL_OPTIONS: Mapping[str, OptionSpec] = MappingProxyType({
    "_MATRIS_CUTEDSL_TUNE_WORKER": OptionSpec(
        "_MATRIS_CUTEDSL_TUNE_WORKER", "bool", False, "internal", "Internal isolated-autotune worker marker."
    ),
})
OBSOLETE_ENVIRONMENT = MappingProxyType({
    "MATRIS_CUDA_GRAPH_U_STEP": "CapacityConfig(u_step=...)",
    "MATRIS_CUDA_GRAPH_T_STEP": "CapacityConfig(t_step=...)",
    "MATRIS_WHOLE_STEP_CUDA_GRAPH": "InferenceConfig(execution='whole_step')",
})
# External benchmark metadata/control, not model configuration.
_EXTERNAL_ENVIRONMENT = frozenset({
    "MATRIS_COMPILE_CPP_GUARDS", "MATRIS_COMPILE_GUARD_NN_MODULES",
    "MATRIS_COMPILE_SKIP_PROCESS_GRAPHS", "MATRIS_COMPILE_PLAIN_ACTIVATIONS",
    "MATRIS_COMPILE_SCOPE",
    "MATRIS_COMPILE_FULLGRAPH_PROBE", "MATRIS_BENCHMARK_PHYSICAL_GPU", "MATRIS_BENCHMARK_NUMA_NODE",
    "MATRIS_REPOSITORY", "MATRIS_MODEL_PATH", "MATRIS_BASELINE_REPOSITORY",
    # optional test assets (tests/matris_test_assets.py) and reference tree
    # (tests/test_gpu_nhc.py)
    "MATRIS_TEST_CHECKPOINT", "MATRIS_TEST_S01_FRAMES", "MATRIS_TEST_REFERENCE_TREE",
})

_PROFILE_SWITCHES = (
    "FUSED_WEIGHTED_SEGMENT_EAGER", "FUSED_GATHER_CAT_EAGER", "FUSED_SEGMENT_SOFTMAX",
    "FUSED_INDEXED_CAT_LINEAR", "FUSED_DIRECTED_PAIR_AGGREGATE", "FUSED_DIRECTED_PAIR_EXPAND",
    "FUSED_SEGMENT_ATTENTION", "FUSED_SEGMENT_ATTENTION_EAGER", "FUSED_PAIRED_SEGMENT_ATTENTION",
    "FUSED_LINE_ENVELOPE", "FUSED_LINE_ENVELOPE_EAGER", "FUSED_THREEBODY_LINEAR",
    "FUSED_THREEBODY_LINEAR_EAGER", "FUSED_THREEBODY_BASIS", "FUSED_THREEBODY_BASIS_EAGER",
    "UNDIRECTED_EDGE_INIT", "FUSED_RESIDUAL_ADD", "FUSED_GATED_LN_RESIDUAL",
)
_ACTIVE: ContextVar[ResolvedConfig | None] = ContextVar("matris_inference_config", default=None)
_SCOPE_LOCK = threading.RLock()


def current_config() -> ResolvedConfig | None:
    """Return the active configuration without importing an inference backend."""
    return _ACTIVE.get()


def current_options_environment() -> dict[str, str]:
    """Export active options for a child process, or legacy environment defaults.

    Callers should first remove all OPTION_REGISTRY keys from the inherited
    child environment: optional None-valued automatic choices are omitted.
    """
    active = _ACTIVE.get()
    options = active.kernel_options if active is not None else _kernel_options("legacy", {})
    return {
        name: formatted for name, value in options.items()
        if (formatted := OPTION_REGISTRY[name].format(value)) is not None
    }


def env_value(name: str, default: Any = None) -> str | None:
    """Read a validated normalized option, respecting the active inference scope.

    Registered options always use their registry default, not a duplicated
    caller default. Unknown names retain ``os.getenv`` fallback semantics.
    The cooldown's default tracks twice the selected hysteresis window.
    """
    spec = OPTION_REGISTRY.get(name) or INTERNAL_OPTIONS.get(name)
    active = _ACTIVE.get()
    if active is not None and name in active.kernel_options:
        value = active.kernel_options[name]
    elif spec is not None and name not in os.environ:
        if name == "MATRIS_TIER_DOWN_COOLDOWN":
            value = 2 * int(env_value("MATRIS_TIER_HYSTERESIS_WINDOW"))
        else:
            value = spec.default
    else:
        value = os.environ.get(name, default)
    if value is None:
        return None
    if spec is None:
        return str(value)
    return spec.format(spec.parse(value))


def _warn_environment(profile: str) -> None:
    for name in sorted(os.environ):
        if name in OBSOLETE_ENVIRONMENT:
            warnings.warn(f"{name} is obsolete and has no effect; use {OBSOLETE_ENVIRONMENT[name]}.", ConfigurationWarning, stacklevel=3)
        elif name.startswith("MATRIS_") and name not in OPTION_REGISTRY and name not in _EXTERNAL_ENVIRONMENT:
            warnings.warn(f"Unknown MatRIS environment option {name}; it has no registered effect.", ConfigurationWarning, stacklevel=3)
    if profile != "legacy":
        ignored = sorted(name for name in OPTION_REGISTRY if name in os.environ)
        if ignored:
            warnings.warn(
                f"optimization_profile={profile!r} ignores registered environment options: "
                + ", ".join(ignored) + "; use expert_overrides for intentional changes.",
                ConfigurationWarning, stacklevel=3,
            )


def _kernel_options(profile: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
    if profile not in {"legacy", "topology", "generic"}:
        raise ValueError("optimization_profile must be 'legacy', 'topology' or 'generic'")
    options = {name: spec.default for name, spec in OPTION_REGISTRY.items()}
    if profile == "legacy":
        for name, spec in OPTION_REGISTRY.items():
            if name in os.environ:
                options[name] = spec.parse(os.environ[name])
    else:
        options.update({"MATRIS_" + suffix: profile == "topology" for suffix in _PROFILE_SWITCHES})
        options["MATRIS_GATED_LN_FUSE_MIN_ROWS"] = 10000 if profile == "topology" else 1_000_000_000
    for name, value in overrides.items():
        if name not in OPTION_REGISTRY:
            replacement = OBSOLETE_ENVIRONMENT.get(name)
            hint = f"; use {replacement}" if replacement else ""
            raise ValueError(f"Unknown expert override {name!r}{hint}")
        options[name] = OPTION_REGISTRY[name].parse(value)
    cooldown = "MATRIS_TIER_DOWN_COOLDOWN"
    if cooldown not in overrides and not (profile == "legacy" and cooldown in os.environ):
        options[cooldown] = 2 * options["MATRIS_TIER_HYSTERESIS_WINDOW"]
    return options


def profile_env(profile: str) -> dict[str, str]:
    """Normalized profile values for subprocesses and legacy benchmark scripts.

    Optional auto-valued settings are omitted; callers configuring a fresh
    subprocess should remove inherited registered MATRIS variables first.
    ``legacy`` intentionally reflects the current process environment.
    """
    return {
        name: formatted for name, value in _kernel_options(profile, {}).items()
        if (formatted := OPTION_REGISTRY[name].format(value)) is not None
    }


@dataclass(frozen=True)
class CapacityConfig:
    """Graph capacity and capture setup (counts, except skin in angstrom)."""
    u_step: int = 512
    t_step: int = 8192
    n_dummy: int = 64
    min_pad_u: int = 256
    min_pad_t: int = 4096
    warmup: int = 3
    candidate_skin: float = 2.0

    def __post_init__(self) -> None:
        for name in ("u_step", "t_step", "n_dummy", "min_pad_u", "min_pad_t", "warmup"):
            minimum = 1 if name in {"u_step", "t_step", "n_dummy"} else 0
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"capacity.{name} must be an integer >= {minimum}")
        skin = self.candidate_skin
        if isinstance(skin, bool) or not isinstance(skin, (int, float)) or not math.isfinite(skin) or skin < 0:
            raise ValueError("capacity.candidate_skin must be finite and >= 0 angstrom")


@dataclass(frozen=True)
class InferenceConfig:
    """Public inference controls; use expert overrides only for kernel research.

    ``execution=None`` preserves each entry point's historical execution mode.
    ``checkpoint='auto'`` uses the model's existing row-count threshold policy;
    it is not a memory-budget-based automatic selection.
    No explicit configuration changes any process-wide environment variable.
    """
    execution: str | None = None
    optimization_profile: str = "legacy"
    checkpoint: str = "off"
    isolated_atoms: str = "reference"
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    expert_overrides: Mapping[str, Any] = field(default_factory=dict)
    report: bool = True

    def __post_init__(self) -> None:
        if self.execution not in {None, "eager", "model_graph", "whole_step"}:
            raise ValueError("execution must be None, 'eager', 'model_graph' or 'whole_step'")
        if self.optimization_profile not in {"legacy", "topology", "generic"}:
            raise ValueError("optimization_profile must be 'legacy', 'topology' or 'generic'")
        if self.checkpoint not in {"off", "on", "auto"}:
            raise ValueError("checkpoint must be 'off', 'on' or 'auto'")
        if self.isolated_atoms not in {"reference", "error"}:
            raise ValueError("isolated_atoms must be 'reference' or 'error'")
        if not isinstance(self.capacity, CapacityConfig):
            raise TypeError("capacity must be a CapacityConfig")
        if not isinstance(self.report, bool):
            raise TypeError("report must be a bool")
        parsed = {}
        for name, value in self.expert_overrides.items():
            if name not in OPTION_REGISTRY:
                replacement = OBSOLETE_ENVIRONMENT.get(name)
                hint = f"; use {replacement}" if replacement else ""
                raise ValueError(f"Unknown expert override {name!r}{hint}")
            parsed[name] = OPTION_REGISTRY[name].parse(value)
        object.__setattr__(self, "expert_overrides", MappingProxyType(parsed))


@dataclass(frozen=True)
class ResolvedConfig:
    execution: str
    optimization_profile: str
    checkpoint_enabled: bool | None
    handle_isolated_atoms: bool
    capacity: CapacityConfig
    kernel_options: Mapping[str, Any]
    report: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_options", MappingProxyType(dict(self.kernel_options)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution": self.execution,
            "optimization_profile": self.optimization_profile,
            "checkpoint": "auto" if self.checkpoint_enabled is None else ("on" if self.checkpoint_enabled else "off"),
            "isolated_atoms": "reference" if self.handle_isolated_atoms else "error",
            "capacity": asdict(self.capacity),
            "kernel_options": dict(self.kernel_options),
            "report": self.report,
        }

    @cached_property
    def fingerprint(self) -> str:
        payload = self.as_dict()
        payload.pop("report")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    @contextmanager
    def scope(self) -> Iterator[ResolvedConfig]:
        """Temporarily bind options, restoring both old and lazy-imported modules.

        The lock is deliberately held over the complete inference operation:
        module globals are shared even though dynamic readers use ContextVar.
        Nested reuse of the identical resolved configuration is inexpensive.
        """
        with _SCOPE_LOCK:
            outer = _ACTIVE.get()
            if outer is self:
                yield self
                return
            # Baseline values for modules imported lazily during this scope.
            # Ignored malformed ENV in an explicit reproducible profile must
            # not make cleanup fail or prevent already-loaded globals restoring.
            fallback = dict(outer.kernel_options) if outer is not None else {
                name: spec.default for name, spec in OPTION_REGISTRY.items()
            }
            if outer is None:
                for name, spec in OPTION_REGISTRY.items():
                    if name in os.environ:
                        try:
                            fallback[name] = spec.parse(os.environ[name])
                        except ValueError:
                            pass
                if "MATRIS_TIER_DOWN_COOLDOWN" not in os.environ:
                    fallback["MATRIS_TIER_DOWN_COOLDOWN"] = 2 * fallback["MATRIS_TIER_HYSTERESIS_WINDOW"]
            saved: dict[tuple[str, str], Any] = {}
            for name, spec in OPTION_REGISTRY.items():
                for module_name, attribute in spec.bindings:
                    module = sys.modules.get(module_name)
                    if module is not None and hasattr(module, attribute):
                        saved[module_name, attribute] = getattr(module, attribute)
                        setattr(module, attribute, self.kernel_options[name])
            token = _ACTIVE.set(self)
            try:
                yield self
            finally:
                _ACTIVE.reset(token)
                for name, spec in OPTION_REGISTRY.items():
                    for module_name, attribute in spec.bindings:
                        module = sys.modules.get(module_name)
                        if module is not None and hasattr(module, attribute):
                            setattr(module, attribute, saved.get((module_name, attribute), fallback[name]))


def resolve_config(
    config: InferenceConfig | None,
    *,
    default_execution: str = "eager",
    default_isolated_atoms: str = "reference",
) -> ResolvedConfig:
    """Resolve once at entry-point construction; values are then immutable.

    Entry points declare execution and isolated-atom defaults here. Omitted
    configuration disables checkpointing and uses those defaults. An outer
    scope supplies only its kernel profile/options to an unconfigured entry;
    explicit InferenceConfig is the sole interface for changing public controls.
    """
    if default_execution not in {"eager", "model_graph", "whole_step"}:
        raise ValueError("invalid default_execution")
    if default_isolated_atoms not in {"reference", "error"}:
        raise ValueError("default_isolated_atoms must be 'reference' or 'error'")
    if config is not None and not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig or None")
    inherited = _ACTIVE.get() if config is None else None
    profile = (inherited.optimization_profile if inherited else "legacy") if config is None else config.optimization_profile
    if inherited is None:
        _warn_environment(profile)
    return ResolvedConfig(
        execution=default_execution if config is None else (config.execution or default_execution),
        optimization_profile=profile,
        checkpoint_enabled=False if config is None else {"off": False, "on": True, "auto": None}[config.checkpoint],
        handle_isolated_atoms=(default_isolated_atoms if config is None else config.isolated_atoms) == "reference",
        capacity=CapacityConfig() if config is None else config.capacity,
        kernel_options=inherited.kernel_options if inherited is not None else _kernel_options(profile, {} if config is None else config.expert_overrides),
        report=False if config is None else config.report,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Print the typed expert-option registry as JSON")
    group.add_argument("--profile", choices=("legacy", "topology", "generic"), help="Print a normalized environment profile as JSON")
    args = parser.parse_args()
    data = [spec.as_dict() for spec in OPTION_REGISTRY.values()] if args.list else profile_env(args.profile)
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
