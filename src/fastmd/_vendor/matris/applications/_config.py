"""Scoped application configuration, validation, and runtime reporting."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
import inspect
import warnings

from ..config import ConfigurationWarning, current_config, resolve_config


REMOVED_INFERENCE_ARGUMENTS = frozenset({
    "handle_isolated_atoms", "validate_isolated_atoms", "enable_checkpoint",
    "use_cuda_graph", "whole_step_cuda_graph", "candidate_skin",
    "enable_model_fusions", "u_step", "t_step", "n_dummy", "min_pad_u",
    "min_pad_t", "warmup",
})


def reject_removed_arguments(kwargs):
    """Do not let ASE's permissive **kwargs silently accept removed flags."""
    removed = REMOVED_INFERENCE_ARGUMENTS.intersection(kwargs)
    if removed:
        raise TypeError(
            "Removed inference arguments: " + ", ".join(sorted(removed))
            + ". Use config=InferenceConfig(...) instead."
        )


def scoped_init(init):
    """Runner constructor scope; ``config`` is an already resolved config."""
    signature = inspect.signature(init)
    @wraps(init)
    def wrapped(self, *args, **kwargs):
        self._inference_config = kwargs.get("config") or current_config()
        resolved = self._inference_config
        if resolved is not None:
            bound = signature.bind(self, *args, **kwargs)
            bound.arguments["config"] = resolved
            for name in ("u_step", "t_step", "n_dummy", "min_pad_u", "warmup"):
                if name not in signature.parameters:
                    continue
                value = getattr(resolved.capacity, name)
                if name in bound.arguments and bound.arguments[name] != value:
                    raise ValueError(f"Runner argument {name} conflicts with config.capacity.{name}")
                bound.arguments[name] = value
            args, kwargs = bound.args[1:], bound.kwargs
        with resolved.scope() if resolved is not None else nullcontext():
            return init(self, *args, **kwargs)
    return wrapped


def scoped_call(method):
    """Run an application method under its instance's kernel configuration."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        resolved = getattr(self, "_inference_config", None)
        with resolved.scope() if resolved is not None else nullcontext():
            result = method(self, *args, **kwargs)
        if method.__name__ in {"_predict_structure", "evaluate", "forward"}:
            self._config_has_evaluated = True
        return result
    return wrapped


def configure_entrypoint(kind):
    """Resolve the single public configuration before loading or inference."""
    def decorate(init):
        signature = inspect.signature(init)

        @wraps(init)
        def wrapped(self, *args, **kwargs):
            import torch
            reject_removed_arguments(kwargs)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            config = bound.arguments.get("config")
            device_arg = bound.arguments.get("device")
            device = (
                torch.device(device_arg) if device_arg is not None
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            default_execution = "model_graph" if kind == "gpu_md" and device.type == "cuda" else "eager"
            default_isolated = "error" if kind == "torchsim" else "reference"
            resolved = resolve_config(
                config, default_execution=default_execution,
                default_isolated_atoms=default_isolated,
            )
            allowed = {"eager", "model_graph", "whole_step"} if kind == "gpu_md" else {"eager", "model_graph"}
            if kind == "torchsim":
                allowed = {"eager"}
            if resolved.execution not in allowed:
                raise ValueError(f"{kind} does not support execution={resolved.execution!r}; supported: {sorted(allowed)}")
            if resolved.execution != "eager" and (device.type != "cuda" or not torch.cuda.is_available()):
                raise ValueError(f"execution={resolved.execution!r} requires an available CUDA device; got {device}")
            if resolved.execution != "eager" and resolved.checkpoint_enabled is not False:
                warnings.warn(
                    "Activation checkpointing may access RNG state during CUDA Graph capture. "
                    "Use checkpoint='off' for the validated graph path; 'auto' is a row-count "
                    "threshold, not a capture-safety policy. The requested setting is unchanged.",
                    ConfigurationWarning, stacklevel=2,
                )
            self._inference_config = resolved
            self._config_has_evaluated = False
            self._config_kind = kind
            self._requested_inference_config = (
                {
                    "execution": config.execution,
                    "optimization_profile": config.optimization_profile,
                    "checkpoint": config.checkpoint,
                    "isolated_atoms": config.isolated_atoms,
                    "capacity": asdict(config.capacity),
                    "expert_overrides": dict(config.expert_overrides),
                } if config is not None else {
                    "source": "entrypoint defaults",
                    "execution": default_execution,
                    "isolated_atoms": default_isolated,
                    "checkpoint": "off",
                }
            )
            with resolved.scope():
                init(*bound.args, **bound.kwargs)
            if kind == "calculator" and resolved.execution == "model_graph" and self._graph_runner is None:
                raise ValueError("execution='model_graph' requires the GPU graph converter; the loaded model uses a CPU/legacy converter")
            if resolved.report:
                summary = config_summary(self)
                effective = summary["effective"]
                print(
                    f"MatRIS config: execution={effective['execution']}, "
                    f"profile={resolved.optimization_profile}, checkpoint={resolved.as_dict()['checkpoint']}, "
                    f"isolated_atoms={effective['isolated_atoms']}; "
                    f"merged projections={effective['merged_projections']['status']}"
                )
                for reason in effective["notes"]:
                    print(f"  {reason}")
        return wrapped
    return decorate

def apply_checkpoint(model, enabled):
    """Keep model metadata and every interaction block in agreement."""
    model.enable_checkpoint = enabled
    model.config["enable_checkpoint"] = enabled
    for block in model.interaction_block:
        block.enable_checkpoint = enabled


def config_summary(instance):
    """Return requested/resolved settings and observed state, not promises."""
    resolved = instance._inference_config
    kind = instance._config_kind
    if kind == "gpu_md":
        execution = "whole_step" if instance.whole_step_runner is not None else "model_graph" if instance.runner is not None else "eager"
    elif kind == "calculator":
        execution = "model_graph" if instance._graph_runner is not None else "eager"
    else:
        execution = "eager"
    notes = []
    if execution != resolved.execution:
        notes.append(f"Requested {resolved.execution}; effective {execution} because CUDA replay requirements were not met.")
    isolated = "reference" if resolved.handle_isolated_atoms else "error"
    if instance.model.reference_energy is None:
        isolated = "error"
        if resolved.handle_isolated_atoms:
            notes.append("Model has no reference-energy table; isolated atoms are rejected.")
    try:
        device = next(instance.model.parameters()).device.type
    except (StopIteration, TypeError):
        device = str(getattr(instance, "device", "unknown")).split(":")[0]
    if device != "cuda":
        notes.append("CUDA-only kernel optimizations are not used on this device.")
    observed = bool(instance._config_has_evaluated)
    # Whole-step constructors already evaluate and capture before returning;
    # unlike the bucket runner, they do not expose a captures counter.
    observed = observed or getattr(instance, "whole_step_runner", None) is not None
    runners = (getattr(instance, "runner", None), getattr(instance, "_graph_runner", None), getattr(instance, "whole_step_runner", None))
    observed = observed or any(getattr(r, "captures", 0) > 0 for r in runners if r is not None)
    materialized = sum(
        getattr(module, "_merged_attn_proj_weight", None) is not None
        for module in instance.model.modules()
    ) if observed else None
    notes.append("Merged projection caches indicate materialization, not per-call use; CUDA device, frozen weights, dtype, feature widths, and row thresholds still gate kernel selection.")
    return {
        "requested": instance._requested_inference_config,
        "configuration": resolved.as_dict(),
        "fingerprint": resolved.fingerprint,
        "effective": {
            "execution": execution,
            "checkpoint": getattr(instance.model, "enable_checkpoint", None),
            "isolated_atoms": isolated,
            "merged_projections": {
                "status": "observed" if observed else "pending first evaluation",
                "materialized_module_count": materialized,
            },
            "notes": notes,
        },
    }
