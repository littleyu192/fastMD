"""MatRIS's admission to distributed inference.

``validate_model`` admits a stock, unhooked FP32 MatRIS model (the single-rank
engine behind the distributed API skips the P>1 topology's rules),
``model_fingerprint`` hashes its weights, buffers and mathematical flags once
at construction. ``validate_model`` rejects instance overrides of
``MODEL_FORWARD_METHODS``, the forward entry points the run boundary's model
stamp covers (defined with the engine's runtime hooks, ``fastmd.engine.runtime``).
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers

import torch
from torch import nn

from fastmd.engine.recorder import recorder_hook_ids
from fastmd.engine.runtime import MODEL_FORWARD_METHODS


def reject_enable_compile(model: nn.Module) -> None:
    """Every distributed engine refuses enable_compile=True models: the multi-rank forward passes
    a runtime and never compiles, and the single-GPU engine's first forward would swap the
    interaction-block layers for torch.compile wrappers after admission."""
    compiled = [name or "model" for name, module in model.named_modules() if getattr(module, "enable_compile", False)]
    if compiled:
        raise ValueError(f"distributed inference does not run enable_compile=True models "
                         f"({', '.join(compiled[:3])}{', ...' if len(compiled) > 3 else ''}): "
                         "load with enable_compile=False")


def check_converter_cutoffs(model: nn.Module) -> None:
    """The distributed engines build with model.config's cutoffs, the single-GPU applications with
    the graph converter's: both must be one graph."""
    cutoffs = (float(model.config["pairwise_cutoff"]), float(model.config["three_body_cutoff"]))
    converter = model.graph_converter
    if cutoffs != (float(converter.atom_graph_cutoff), float(converter.line_graph_cutoff)):
        raise ValueError(f"model.config cutoffs {cutoffs} differ from graph_converter cutoffs "
                         f"{(converter.atom_graph_cutoff, converter.line_graph_cutoff)}: single-GPU applications "
                         "build with the converter, the distributed evaluator with model.config")


def validate_model(model: nn.Module, *, single_rank: bool = False) -> None:
    """Admit stock, unhooked FP32 models; frozen FP64 toys are internal only.

    Model stages are shared with single-device execution; primitive policies
    may still interpret leaf modules. Subclasses, hooks and replaced stage
    methods cannot silently change the admitted distributed program.
    ``single_rank`` (the single-GPU engine) skips the P>1 topology's rules.
    """
    from fastmd.models.matris.converter import GraphConverter
    from fastmd.models.matris.basis_function import (
        BesselExpansion, FourierExpansion, GaussianExpansion, PolynomialEnvelope,
        SphericalExpansion,
    )
    from fastmd.models.matris.feature_embed import AtomTypeEmbedding, EdgeBasisEmbedding, ThreebodyEmbedding
    from fastmd.models.matris.functions import FusedSiLU, FusedSigmoid, GatedMLP, GraphPooling, MLP, SwishLayer
    from fastmd.models.matris.interaction_block import Graph_Attention_Layer, Interaction_Block, Refinement
    from fastmd.models.matris.model import MatRIS
    from fastmd.models.matris.readout import EnergyHead, ForceStressHead, MagmomHead
    from fastmd.models.matris.reference_energy import AtomRef

    stock_types = {
        MatRIS, GraphConverter, AtomTypeEmbedding, EdgeBasisEmbedding, ThreebodyEmbedding,
        FusedSiLU, FusedSigmoid, GatedMLP, GraphPooling, MLP, SwishLayer,
        Graph_Attention_Layer, Interaction_Block, Refinement,
        BesselExpansion, FourierExpansion, GaussianExpansion, PolynomialEnvelope, SphericalExpansion,
        EnergyHead, ForceStressHead, MagmomHead, AtomRef,
        nn.ModuleList, nn.Sequential, nn.Identity, nn.Linear, nn.Embedding,
        nn.LayerNorm, nn.RMSNorm, nn.BatchNorm1d,
        nn.SiLU, nn.Sigmoid, nn.ReLU, nn.GELU, nn.Softplus, nn.Tanh, nn.Dropout,
    }
    # The op-selection recorder's own site hooks only observe (step 2.0).
    exempt = recorder_hook_ids()
    for name in ("_global_forward_hooks", "_global_forward_pre_hooks",
                 "_global_backward_hooks", "_global_backward_pre_hooks"):
        if set(getattr(torch.nn.modules.module, name, {})) - exempt:
            raise ValueError("custom global module hooks are unsupported by torch_reference")

    if type(model) is not MatRIS:
        raise TypeError("distributed inference requires an uncompiled MatRIS model")
    if getattr(model, "is_intensive", None) is not True:
        raise ValueError("only intensive MatRIS models are supported")
    if not model.force_stress_head.is_conservation:
        raise ValueError("distributed inference only supports conservative energy/forces")
    reference = model.reference_energy
    if reference is not None and getattr(reference, "is_intensive", None) is not True:
        raise ValueError("reference-energy and model intensive flags must both be True")
    for name, module in model.named_modules():
        if getattr(module, "_is_compiled", False) or hasattr(module, "_orig_mod"):
            raise ValueError(f"compiled module is unsupported: {name}")
        if type(module) not in stock_types:
            raise TypeError(f"torch_reference requires exact stock module types; unsupported {name}: "
                            f"{type(module).__qualname__}")
        if MODEL_FORWARD_METHODS.intersection(vars(module)):
            raise ValueError(f"instance forward replacement is unsupported: {name}")
        if any(getattr(module, hook, {}) for hook in
               ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")):
            raise ValueError(f"custom module hooks are unsupported: {name}")
        if type(module) is MLP and module.use_fp16:
            raise ValueError(f"MLP.use_fp16=True is an unvalidated variant: {name}")
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats or module.running_mean is None:
                raise ValueError(f"BatchNorm must have fixed running statistics: {name}")
    reject_enable_compile(model)
    # The P>1 forward skips the dead three-body tail on the final block only;
    # the constructor sets last_block there alone. (The single engine runs the
    # model as is.)
    blocks = list(model.interaction_block)
    for index, block in enumerate(blocks[:-1]):
        if not single_rank and getattr(block, "last_block", False):
            raise ValueError(f"last_block is set on interaction block {index} of {len(blocks)}; "
                             "only the final block may carry it")
    for key in ("pairwise_cutoff", "three_body_cutoff"):
        value = model.config[key]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {key}: {value}")
    if not single_rank and model.config["three_body_cutoff"] > model.config["pairwise_cutoff"]:
        # The topology epoch's builder (FixedCapacityGraphBuilder) rejects a line
        # cutoff above the atom cutoff; every P>1 topology agrees (step 4.2).
        raise ValueError(f"distributed inference requires three_body_cutoff "
                         f"({model.config['three_body_cutoff']}) <= pairwise_cutoff "
                         f"({model.config['pairwise_cutoff']})")
    check_converter_cutoffs(model)
    dtypes = {p.dtype for p in model.parameters()}
    dtypes.update(buffer.dtype for buffer in model.buffers() if buffer.is_floating_point())
    dtypes.update(module.coef.dtype for module in model.modules() if type(module) is SphericalExpansion)
    if dtypes != {torch.float32}:
        raise ValueError("public distributed inference requires homogeneous FP32 model tensors; "
                         "FP64 is reserved for internal CPU gradient tests")


def model_fingerprint(model: nn.Module) -> str:
    """Hash weights, buffers and loaded mathematical flags once at construction."""
    from fastmd.models.matris.basis_function import (
        BesselExpansion, FourierExpansion, GaussianExpansion, PolynomialEnvelope,
        SphericalExpansion,
    )
    from fastmd.models.matris.feature_embed import EdgeBasisEmbedding, ThreebodyEmbedding
    from fastmd.models.matris.functions import MLP
    from fastmd.models.matris.interaction_block import Refinement

    # Explicitly enumerate the non-state attributes read by the reference
    # implementation. Do not hash repr(module), __dict__, object addresses,
    # transient dispatch caches, or unrelated application annotations.
    scalar_fields = (
        (nn.Embedding, ("padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse")),
        (nn.LayerNorm, ("normalized_shape", "eps", "elementwise_affine")),
        (nn.RMSNorm, ("normalized_shape", "eps", "elementwise_affine")),
        (nn.modules.batchnorm._BatchNorm, ("eps", "momentum", "affine", "track_running_stats")),
        (nn.GELU, ("approximate",)),
        (nn.Softplus, ("beta", "threshold")),
        (nn.Dropout, ("p",)),
        (MLP, ("use_fp16",)),
        (PolynomialEnvelope, ("cutoff", "p", "a", "b", "c")),
        (BesselExpansion, ("num_radial", "inv_cutoff", "norm_const")),
        (GaussianExpansion, ("num_output", "coeff")),
        (FourierExpansion, ("max_f", "scale")),
        (SphericalExpansion, ("max_n", "max_l", "cutoff")),
        (Refinement, ("graph_type", "use_smoothed_for_delta_edge")),
        (EdgeBasisEmbedding, ("distance_expansion",)),
        (ThreebodyEmbedding, ("three_body_expansion",)),
    )

    def plain(value):
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, numbers.Integral):
            return int(value)
        if isinstance(value, numbers.Real):
            if not math.isfinite(value):
                raise ValueError("non-finite model mathematical attribute")
            return float(value)
        if isinstance(value, (tuple, list)):
            return [plain(part) for part in value]
        raise TypeError(f"unsupported mathematical attribute type: {type(value).__qualname__}")

    digest = hashlib.sha256()
    module_metadata = []
    extra_tensors = []
    for name, module in model.named_modules():
        values = {}
        for kind, fields in scalar_fields:
            if isinstance(module, kind):
                values.update({field: plain(getattr(module, field)) for field in fields})
        module_metadata.append({"name": name,
                                "type": f"{type(module).__module__}.{type(module).__qualname__}",
                                "math": values})
        if isinstance(module, SphericalExpansion):
            # Historical SH coefficients are a plain Tensor, not a registered
            # buffer, but the shared SH basis reads them directly.
            extra_tensors.append((f"{name}.unregistered_coef", module.coef))
    flags = {
        "config": model.config,
        "intensive": model.is_intensive,
        "reference_intensive": getattr(model.reference_energy, "is_intensive", None),
        "layers": [{
            "last_block": getattr(block, "last_block", False),
            "smoothed": block.refine_block_atom_graph.use_smoothed_for_delta_edge,
        } for block in model.interaction_block],
        "modules": module_metadata,
    }
    digest.update(json.dumps(flags, sort_keys=True, default=plain, allow_nan=False).encode())
    for name, value in [*model.state_dict().items(), *extra_tensors]:
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
