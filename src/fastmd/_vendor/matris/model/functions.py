from __future__ import annotations

from fastmd._vendor.matris.config import env_value

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import math
from .op import fused_silu, fused_sigmoid

# Row-count threshold above which the fused Triton LayerNorm+activation (Fusion C)
# is used instead of eager nn.LayerNorm+act.
_LN_FUSE_THRESH = int(env_value("MATRIS_GATED_LN_FUSE_MIN_ROWS", "10000"))
_FUSED_GATED_LN_RESIDUAL = env_value("MATRIS_FUSED_GATED_LN_RESIDUAL", "1") != "0"
_FUSED_SEGMENT_SOFTMAX = env_value("MATRIS_FUSED_SEGMENT_SOFTMAX", "1") != "0"
_INDEXED_CAT_LINEAR_FUSE = env_value("MATRIS_FUSED_INDEXED_CAT_LINEAR", "0") == "1"
_INDEXED_CAT_LINEAR_BACKEND = env_value(
    "MATRIS_FUSED_INDEXED_CAT_LINEAR_BACKEND", "decomposed_indexed_silu"
).lower()
_INDEXED_CAT_LINEAR_MIN_ROWS = int(
    env_value("MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS", "20000")
)
_INDEXED_CAT_LINEAR_BACKENDS = {
    "cutedsl_dense_silu",
    "decomposed_indexed_silu",
}
if _INDEXED_CAT_LINEAR_FUSE and _INDEXED_CAT_LINEAR_BACKEND not in _INDEXED_CAT_LINEAR_BACKENDS:
    raise ValueError(
        "unsupported MATRIS_FUSED_INDEXED_CAT_LINEAR_BACKEND="
        f"{_INDEXED_CAT_LINEAR_BACKEND!r}; expected one of "
        f"{sorted(_INDEXED_CAT_LINEAR_BACKENDS)}"
    )

class FusedSiLU(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_silu(x)
        else:
            return torch.nn.functional.silu(x) 

class FusedSigmoid(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_sigmoid(x)
        else:
            return torch.nn.functional.sigmoid(x)

def get_activation(name: str) -> nn.Module:
    """Return an activation function"""
    activation_map = {
        "relu": nn.ReLU,
        "silu": FusedSiLU,  # Using fused version for better performance
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "sigmoid": FusedSigmoid,  # Using fused version for better performance
        "tanh": nn.Tanh,
    }
    
    name_lower = name.lower()
    if name_lower not in activation_map:
        raise NotImplementedError(
            f"Activation '{name}' is not implemented. "
            f"Supported activations: {list(activation_map.keys())}"
        )
    return activation_map[name_lower]()

def get_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function"""
    if name is None:
        return None
        
    normalization_map = {
        "layer": nn.LayerNorm(dim),
        "rms": nn.RMSNorm(dim), # torch >= 2.6.0
        "batch": nn.BatchNorm1d(dim),
    }
    name_lower = name.lower()
    return normalization_map[name_lower]

class SwishLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to use bias in the linear layer. Default: True.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act = get_activation("silu")
    
    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas: shape (feas_num, in_dim)
            
        Returns:
            output: shape (feas_num, out_dim)
        """
        return self.act(self.linear(feas))

def segment_count(segment: Tensor, num_segment: int) -> Tensor:
    """Sync-free replacement for ``torch.bincount(segment, minlength=num_segment)``.

    ``torch.bincount`` reads ``max()`` on the host (a D2H sync / stall) on CUDA
    even when ``minlength`` is given. Counting via ``index_add_`` into a
    fixed-size buffer is sync-free and CUDA-graph capturable.
    """
    counts = torch.zeros(num_segment, dtype=torch.long, device=segment.device)
    counts.index_add_(0, segment.long(), torch.ones_like(segment, dtype=torch.long))
    return counts


def Dimwise_softmax(feas: Tensor, segment: Tensor, num_segment=None) -> Tensor:
    """Computes a sparsely evaluated softmax.
    
    Args:
        feas: The source tensor. shape: [num, dim]
        segment: specify the segment of each row [num, 1] 
    """
    num, dim = feas.shape
    if num_segment is None:
        num_segment = int(segment.max()) + 1
    if feas.is_cuda and _FUSED_SEGMENT_SOFTMAX:
        # Fusion A: single fused Triton segment-softmax (3 fwd + 2 bwd kernels)
        # replacing ~8 eager ops; validated exact (FP64 gradcheck) and 2-2.6x faster.
        from .op.triton_segment_softmax import fused_segment_softmax
        return fused_segment_softmax(feas, segment, num_segment)

    segment_expanded = segment.unsqueeze(1).expand(-1, dim) # [num, dim]
    
    feas_max = torch.empty( num_segment, dim, dtype=feas.dtype, device=feas.device )
    feas_max.fill_(float("-inf"))
    feas_max = feas_max.scatter_reduce(
        0, segment_expanded, feas, reduce='amax', include_self=False,
    ) #[num_segment, dim]
    # Gather: [num_segment, dim] -> [num, dim]
    feas_max = feas_max[segment]
    out = (feas - feas_max).exp()
    
    # =========== scatter sum ============
    out_sum = torch.zeros(num_segment, dim, device=feas.device, dtype=feas.dtype)
    out_sum = out_sum.scatter_reduce(
        0, segment_expanded, out, reduce='sum', include_self=False
    )
    # Gather: [num_segment, dim] -> [num, dim]
    out_sum = out_sum[segment]
    score = out / out_sum
    return score

def aggregate(data: torch.Tensor, 
              segment: torch.Tensor, 
              bin_count: torch.Tensor = None, 
              average=True, 
              num_segment=None) -> torch.Tensor:
    """Aggregate rows in data by specifying the segment.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        segment (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    if bin_count is None:
        # torch.bincount syncs (reads max() on host) even with minlength; use a
        # sync-free index_add_ count when the segment size is known.
        if num_segment is not None:
            bin_count = segment_count(segment, num_segment)
        else:
            bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

    if (num_segment is not None) and (bin_count.shape[0] != num_segment):
        difference = num_segment - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, segment, data)
    if average:
        output = (output.T / bin_count).T
    return output


class MLP(nn.Module):
        
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MLP layer.
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Number of hidden units. Can be an integer for a single
                hidden layer, a sequence of integers for multiple hidden layers,
                or None for no hidden layers. Default: (128, 128).
            output_dim: Dimension of output predictions. Default: 128.
            dropout: Dropout rate applied before each linear layer. Default: 0.0.
            activation: Activation function. Supported: "relu", "silu", "tanh", "gelu".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Dropout rate must be in [0.0, 1.0), got {dropout}")
        
        self.use_fp16 = use_fp16
        activation_func = get_activation(activation)

        layers = []
        if hidden_dim in (None, 0):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        elif isinstance(hidden_dim, int):
            # Single hidden layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim, bias=bias),
                activation_func,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim, bias=bias),
            ])
        elif isinstance(hidden_dim, Sequence):
            # Multiple hidden layers
            layers.extend([
                nn.Linear(input_dim, hidden_dim[0], bias=bias),
                activation_func,
            ])
            # Additional hidden layers
            for i in range(len(hidden_dim) - 1):
                layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=bias),
                    activation_func,
                ])
            # Output layer
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim[-1], output_dim, bias=bias)])
        else:
            raise TypeError(
                f"hidden_dim must be an integer, sequence of integers, or None, "
                f"got {type(hidden_dim).__name__}"
            )
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, feas: Tensor) -> Tensor:
        """
            Args:
                feas: Input tensor of shape (features, input_dim)
            Returns:
                Output tensor of shape (features, output_dim)
        """
        if self.use_fp16 and feas.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self.layers(feas)
            out = out.to(torch.float32)
        else:
            out = self.layers(feas) 
        return out


class GatedMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm_type: str = "layer",
        bias: bool = True,
        use_fp16: bool = False,
    ) -> None:
        """
        Args:
            input_dim: The input dimension.
            hidden_dim: A list of integers or a single integer representing the number 
                of hidden units in each layer of the MLP. Default: None.
            output_dim: The output dimension.
            dropout: The dropout rate. Default: 0.0.
            activation: The name of the activation function. Must be one of "relu", 
                "silu", "tanh", or "gelu". Default: "silu".
            norm_type: The name of the normalization layer to use. Must be one of 
                "layer", "rms", "batch", "group", or None. Default: "layer".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        self.use_fp16 = use_fp16
        self.activation_func = get_activation(activation)
        self.activation_gate = get_activation("sigmoid")
        self.gate_norm = get_normalization(name=norm_type, dim=output_dim)
        self.core_norm = get_normalization(name=norm_type, dim=output_dim)
        self.mlp_core = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )
        # Fusion C: fuse LayerNorm+activation when the norm is LayerNorm and the
        # activations are SiLU(core)/Sigmoid(gate). Only applied to large row counts
        # (line-graph tensors), where the fused Triton kernel is a net win (~2.6x);
        # torch's LayerNorm already wins at small row counts, so guard by a threshold.
        from .op.triton_layernorm_act import ACT_SILU, ACT_SIGMOID
        self._fuse_ln = (
            isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
        )
        self._core_act, self._gate_act = ACT_SILU, ACT_SIGMOID
        self._indexed_cat_linear_cache_key = None
        self._indexed_cat_linear_weight = None
        self._indexed_cat_linear_bias = None
        self._indexed_cat_linear_split_cache_key = None
        self._indexed_cat_linear_split_weights = None
        self._indexed_cat_linear_split_bias = None

    def _finish_gate(self, core_x: Tensor, gate_x: Tensor) -> Tensor:
        if self.gate_norm is None:
            core = self.activation_func(core_x)
            gate = self.activation_gate(gate_x)
        elif self._fuse_ln and core_x.is_cuda and core_x.shape[0] >= _LN_FUSE_THRESH:
            cn, gn = self.core_norm, self.gate_norm
            if not (
                cn.weight.requires_grad
                or cn.bias.requires_grad
                or gn.weight.requires_grad
                or gn.bias.requires_grad
            ):
                from .op.triton_gated_ln import fused_gated_ln
                return fused_gated_ln(
                    core_x,
                    gate_x,
                    cn.weight,
                    cn.bias,
                    gn.weight,
                    gn.bias,
                    cn.eps,
                    gn.eps,
                )
            from .op.triton_layernorm_act import fused_ln_act
            core = fused_ln_act(core_x, cn.weight, cn.bias, cn.eps, self._core_act)
            gate = fused_ln_act(gate_x, gn.weight, gn.bias, gn.eps, self._gate_act)
        else:
            core = self.activation_func(self.core_norm(core_x))
            gate = self.activation_gate(self.gate_norm(gate_x))
        return core * gate # gate mul

    def forward_with_residual(
        self,
        feas: Tensor,
        residual: Tensor,
        res_weight: Tensor,
    ) -> Tensor | None:
        if not _FUSED_GATED_LN_RESIDUAL:
            return None
        if not (
            self._fuse_ln
            and feas.is_cuda
            and residual.is_cuda
            and res_weight.is_cuda
            and residual.dim() == 2
            and residual.shape[0] >= _LN_FUSE_THRESH
            and residual.shape[1] == res_weight.numel()
            and not res_weight.requires_grad
        ):
            return None
        cn, gn = self.core_norm, self.gate_norm
        if (
            cn.weight.requires_grad
            or cn.bias.requires_grad
            or gn.weight.requires_grad
            or gn.bias.requires_grad
        ):
            return None
        core_x = self.mlp_core(feas)
        gate_x = self.mlp_gate(feas)
        if core_x.shape != residual.shape or gate_x.shape != residual.shape:
            return None
        from .op.triton_gated_ln import fused_gated_ln_residual

        return fused_gated_ln_residual(
            core_x,
            gate_x,
            cn.weight,
            cn.bias,
            gn.weight,
            gn.bias,
            residual,
            res_weight,
            cn.eps,
            gn.eps,
        )

    def _first_linear_modules(self) -> tuple[nn.Linear, nn.Linear] | None:
        if len(self.mlp_core.layers) == 0 or len(self.mlp_gate.layers) == 0:
            return None
        core_linear = self.mlp_core.layers[0]
        gate_linear = self.mlp_gate.layers[0]
        if not isinstance(core_linear, nn.Linear) or not isinstance(gate_linear, nn.Linear):
            return None
        return core_linear, gate_linear

    def _can_fuse_indexed_cat_first_linear(self, rows: int, dim: int, parts: int) -> bool:
        if not _INDEXED_CAT_LINEAR_FUSE:
            return False
        if self.use_fp16 or self.gate_norm is None or not self._fuse_ln:
            return False
        if rows < _INDEXED_CAT_LINEAR_MIN_ROWS:
            return False
        modules = self._first_linear_modules()
        if modules is None:
            return False
        core_linear, gate_linear = modules
        input_dim = parts * dim
        if (
            core_linear.in_features != input_dim
            or gate_linear.in_features != input_dim
            or core_linear.out_features != gate_linear.out_features
        ):
            return False
        if (core_linear.bias is None) != (gate_linear.bias is None):
            return False
        tensors = [core_linear.weight, gate_linear.weight]
        if core_linear.bias is not None:
            tensors += [core_linear.bias, gate_linear.bias]
        tensors += [self.core_norm.weight, self.core_norm.bias, self.gate_norm.weight, self.gate_norm.bias]
        return not any(t.requires_grad for t in tensors)

    def _merged_first_linear_params(
        self,
        rows: int,
        dim: int,
        parts: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None] | None:
        if not self._can_fuse_indexed_cat_first_linear(rows, dim, parts):
            return None
        modules = self._first_linear_modules()
        if modules is None:
            return None
        core_linear, gate_linear = modules
        if core_linear.weight.device != device or core_linear.weight.dtype != dtype:
            return None
        if gate_linear.weight.device != device or gate_linear.weight.dtype != dtype:
            return None
        bias_key = ()
        if core_linear.bias is not None:
            if core_linear.bias.device != device or core_linear.bias.dtype != dtype:
                return None
            if gate_linear.bias.device != device or gate_linear.bias.dtype != dtype:
                return None
            bias_key = (
                core_linear.bias.data_ptr(),
                gate_linear.bias.data_ptr(),
                core_linear.bias._version,
                gate_linear.bias._version,
            )
        key = (
            core_linear.weight.data_ptr(),
            gate_linear.weight.data_ptr(),
            core_linear.weight._version,
            gate_linear.weight._version,
            device,
            dtype,
            dim,
            parts,
            bias_key,
        )
        if self._indexed_cat_linear_cache_key != key:
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                return None
            with torch.no_grad():
                self._indexed_cat_linear_weight = torch.cat(
                    [core_linear.weight.detach(), gate_linear.weight.detach()],
                    dim=0,
                ).contiguous()
                if core_linear.bias is None:
                    self._indexed_cat_linear_bias = None
                else:
                    self._indexed_cat_linear_bias = torch.cat(
                        [core_linear.bias.detach(), gate_linear.bias.detach()],
                        dim=0,
                    ).contiguous()
                self._indexed_cat_linear_cache_key = key
        return self._indexed_cat_linear_weight, self._indexed_cat_linear_bias

    def _split_first_linear_params(
        self,
        rows: int,
        dim: int,
        parts: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[tuple[Tensor, ...], Tensor | None] | None:
        if not self._can_fuse_indexed_cat_first_linear(rows, dim, parts):
            return None
        modules = self._first_linear_modules()
        if modules is None:
            return None
        core_linear, gate_linear = modules
        if core_linear.weight.device != device or core_linear.weight.dtype != dtype:
            return None
        if gate_linear.weight.device != device or gate_linear.weight.dtype != dtype:
            return None
        bias_key = ()
        if core_linear.bias is not None:
            if core_linear.bias.device != device or core_linear.bias.dtype != dtype:
                return None
            if gate_linear.bias.device != device or gate_linear.bias.dtype != dtype:
                return None
            bias_key = (
                core_linear.bias.data_ptr(),
                gate_linear.bias.data_ptr(),
                core_linear.bias._version,
                gate_linear.bias._version,
            )
        key = (
            core_linear.weight.data_ptr(),
            gate_linear.weight.data_ptr(),
            core_linear.weight._version,
            gate_linear.weight._version,
            device,
            dtype,
            dim,
            parts,
            bias_key,
        )
        if self._indexed_cat_linear_split_cache_key != key:
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                return None
            with torch.no_grad():
                weights = []
                for part in range(parts):
                    start = part * dim
                    end = start + dim
                    weights.append(
                        torch.cat(
                            [
                                core_linear.weight.detach()[:, start:end],
                                gate_linear.weight.detach()[:, start:end],
                            ],
                            dim=0,
                        ).contiguous()
                    )
                self._indexed_cat_linear_split_weights = tuple(weights)
                if core_linear.bias is None:
                    self._indexed_cat_linear_split_bias = None
                else:
                    self._indexed_cat_linear_split_bias = torch.cat(
                        [core_linear.bias.detach(), gate_linear.bias.detach()],
                        dim=0,
                    ).contiguous()
                self._indexed_cat_linear_split_cache_key = key
        return self._indexed_cat_linear_split_weights, self._indexed_cat_linear_split_bias

    @staticmethod
    def _run_after_first(mlp: MLP, x: Tensor) -> Tensor:
        for layer in list(mlp.layers)[1:]:
            x = layer(x)
        return x

    @staticmethod
    def _run_after_first_activation(mlp: MLP, x: Tensor) -> Tensor:
        for layer in list(mlp.layers)[2:]:
            x = layer(x)
        return x

    def aligned_first_linear_block(
        self,
        rows: int,
        dim: int,
        parts: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Aligned-column block of the split first linear ([2H, dim]) or None.

        Lets a caller merge the p0 projection into a wider frozen GEMM and
        pass the result back via ``precomputed_p0``. Only valid for the
        decomposed backend, under the same fusion guards.
        """
        if _INDEXED_CAT_LINEAR_BACKEND != "decomposed_indexed_silu":
            return None
        params = self._split_first_linear_params(rows, dim, parts, device, dtype)
        if params is None:
            return None
        weights, _ = params
        return weights[0]

    def forward_aligned_gather_cat3(
        self,
        aligned: Tensor,
        gathered: Tensor,
        index_a: Tensor,
        index_b: Tensor,
        index_a_sorted: bool = False,
        precomputed_p0: Tensor | None = None,
    ) -> Tensor | None:
        if not (
            aligned.is_cuda
            and gathered.is_cuda
            and (aligned.is_contiguous() or precomputed_p0 is not None)
            and gathered.is_contiguous()
        ):
            return None
        rows, dim = aligned.shape
        if _INDEXED_CAT_LINEAR_BACKEND == "decomposed_indexed_silu":
            params = self._split_first_linear_params(
                rows, dim, 3, aligned.device, aligned.dtype
            )
            if params is None:
                return None
            weights, bias = params
            from .op.decomposed_indexed_cat_silu_linear import (
                decomposed_indexed_cat3_silu_linear,
            )

            fused_first = decomposed_indexed_cat3_silu_linear(
                aligned,
                gathered,
                index_a,
                index_b,
                weights,
                bias,
                index_a_sorted,
                precomputed_p0=precomputed_p0,
            )
            if fused_first is None:
                return None
            core_first, gate_first = fused_first
            core_x = self._run_after_first_activation(self.mlp_core, core_first)
            gate_x = self._run_after_first_activation(self.mlp_gate, gate_first)
            return self._finish_gate(core_x, gate_x)

        params = self._merged_first_linear_params(rows, dim, 3, aligned.device, aligned.dtype)
        if params is None:
            return None
        weight, bias = params
        if _INDEXED_CAT_LINEAR_BACKEND == "cutedsl_dense_silu":
            from .op.triton_gather_cat import aligned_gather_cat3
            from .op.cutedsl_dense_silu_linear import cutedsl_dense_silu_linear

            merged_input = aligned_gather_cat3(aligned, gathered, index_a, index_b)
            fused_first = cutedsl_dense_silu_linear(merged_input, weight, bias)
            if fused_first is None:
                return None
            core_first, gate_first = fused_first
            core_x = self._run_after_first_activation(self.mlp_core, core_first)
            gate_x = self._run_after_first_activation(self.mlp_gate, gate_first)
            return self._finish_gate(core_x, gate_x)
        return None

    def forward_aligned_gather_cat4(
        self,
        aligned: Tensor,
        gathered_a: Tensor,
        index_a: Tensor,
        gathered_b: Tensor,
        index_b: Tensor,
        index_c: Tensor,
        index_a_sorted: bool = False,
        precomputed_p0: Tensor | None = None,
    ) -> Tensor | None:
        if not (
            aligned.is_cuda
            and gathered_a.is_cuda
            and gathered_b.is_cuda
            and (aligned.is_contiguous() or precomputed_p0 is not None)
            and gathered_a.is_contiguous()
            and gathered_b.is_contiguous()
        ):
            return None
        rows, dim = aligned.shape
        if _INDEXED_CAT_LINEAR_BACKEND == "decomposed_indexed_silu":
            params = self._split_first_linear_params(
                rows, dim, 4, aligned.device, aligned.dtype
            )
            if params is None:
                return None
            weights, bias = params
            from .op.decomposed_indexed_cat_silu_linear import (
                decomposed_indexed_cat4_silu_linear,
            )

            fused_first = decomposed_indexed_cat4_silu_linear(
                aligned,
                gathered_a,
                index_a,
                gathered_b,
                index_b,
                index_c,
                weights,
                bias,
                index_a_sorted,
                precomputed_p0=precomputed_p0,
            )
            if fused_first is None:
                return None
            core_first, gate_first = fused_first
            core_x = self._run_after_first_activation(self.mlp_core, core_first)
            gate_x = self._run_after_first_activation(self.mlp_gate, gate_first)
            return self._finish_gate(core_x, gate_x)

        params = self._merged_first_linear_params(rows, dim, 4, aligned.device, aligned.dtype)
        if params is None:
            return None
        weight, bias = params
        if _INDEXED_CAT_LINEAR_BACKEND == "cutedsl_dense_silu":
            from .op.triton_gather_cat import aligned_gather_cat4
            from .op.cutedsl_dense_silu_linear import cutedsl_dense_silu_linear

            merged_input = aligned_gather_cat4(
                aligned,
                gathered_a,
                index_a,
                gathered_b,
                index_b,
                index_c,
            )
            fused_first = cutedsl_dense_silu_linear(merged_input, weight, bias)
            if fused_first is None:
                return None
            core_first, gate_first = fused_first
            core_x = self._run_after_first_activation(self.mlp_core, core_first)
            gate_x = self._run_after_first_activation(self.mlp_gate, gate_first)
            return self._finish_gate(core_x, gate_x)
        return None

    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas (Tensor): shape (feas_num, input_dim)
        Returns:
            output: shape (feas_num, output_dim)
        """
        core_x = self.mlp_core(feas)
        gate_x = self.mlp_gate(feas)
        return self._finish_gate(core_x, gate_x)


class MOE_Layer(nn.Module):
    
    def __init__(
        self,
        num_expert: int = 64,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MOE layer.

        Args:
            
        """
        super().__init__()
        
        raise NotImplementedError
         
    def forward(self, feas: Tensor) -> Tensor:
        return None


class GraphPooling(nn.Module):
    def __init__(self, average: bool = False) -> None:
        
        super().__init__()
        self.average = average

    def forward(self, node_feat: Tensor, segment: Tensor, num_segment: int | None = None) -> Tensor:
        """
        Args:
            atom_feat (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, node_feat_dim or 1]
            segment (Tensor): graph indices for each atom.
                [num_batch_atoms]
            num_segment (int): number of output segments (e.g. n_graphs). When
                provided, avoids a host max()-read (D2H sync).

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, node_feat_dim or 1]
        """
        if self.average:
            if num_segment is not None:
                bin_count = segment_count(segment, num_segment)  # sync-free
            else:
                bin_count = torch.bincount(segment)
            bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
            n_seg = bin_count.shape[0]
        else:
            # average=False only needs the output size; avoid the host max()-read.
            n_seg = num_segment if num_segment is not None else int(segment.max()) + 1

        output = node_feat.new_zeros([n_seg, node_feat.shape[1]])
        output = output.index_add_(0, segment, node_feat)
        if self.average:
            output = (output.T / bin_count).T
        return output


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        change_mat = torch.tensor(
            [
                [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                [
                    -(6 ** (-0.5)),
                    0,
                    0,
                    0,
                    2 * 6 ** (-0.5),
                    0,
                    0,
                    0,
                    -(6 ** (-0.5)),
                ],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
            ],
            device=device,
        ).detach()

    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1

    return total


def reshape_stress(L0out, L2out, batch_size=1):
    _max_rank = 2
    pred_irreps = torch.zeros(
        (batch_size, irreps_sum(_max_rank)),
        device = L0out.device,
    )
    # L=0
    L=0
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L0out.view(batch_size, -1)
    
    L=2
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L2out.view(batch_size, -1) 
    
    pred = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(_max_rank, device = L0out.device),
        pred_irreps,
    )
    
    return pred.view(batch_size, 3,3)


class Sphere(nn.Module):
    
    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.lmax = lmax
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_0_0 = torch.ones_like(x)
        if lmax == 0:
            return torch.stack([ sh_0_0, ], dim=-1)
        
        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)
