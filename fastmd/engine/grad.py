"""P>1 forces: exchanges as autograd Functions on a token chain (design §5.2/§5.3; steps 4.1/4.2).

``AutogradBackend`` is the P>1 ``GradBackend`` (``fastmd.engine.runtime``);
since step 4.2 the only one (the StageTape is deleted). The forward is the
prepared sequence of local stages and exchanges; the backward is ONE
``torch.autograd.grad`` over the whole forward:

* ``_ExchangeCopy.apply(x, tok, stage) -> (y, tok')``: COPY forward into a new
  buffer (INV-8), reverse SUM backward, both through the prepared binding's
  ``ExchangePlan._communicate``. ``gy`` None (an unused output or a dead tail)
  sends zeros: the collective always runs.
* the token chain: a 0-dim ``token_leaf`` (requires grad) is threaded through
  every exchange and ends in ``_Anchor(E, tok_K)``, and ``token_leaf`` is an
  input of the ``autograd.grad`` call. Every rank therefore creates every
  exchange node (its ``tok`` input requires grad even where ``x`` does not),
  exec_info never prunes one (each is on the path E -> token_leaf), and the
  exchange backwards run in exactly the reverse forward order (node k needs the
  token cotangent of node k+1), whatever the rank-local graph between them:
  every rank issues the same collectives in the same order (design §5.3).
  Token values are uninitialized and never read; their cotangent is the energy
  seed, passed on without a kernel.
* ``_Boundary``: an identity at every differentiable local-stage input. Its
  backward receives the stage's summed cotangent for that input, so
  cross-stage accumulation is grouped per stage (the grouping of the former
  StageTape, bitwise equal to it under deterministic kernels), and a
  cotangent that is a view of a larger buffer (a CatBackward slice) is copied
  rather than pinning that buffer in a long-lived consumer's InputBuffer.
* guards (design §5.6): an exchange forward inside a backward (a checkpoint
  recompute) raises before any collective; the exchange apply is
  ``torch._dynamo.disable``d, so torch.compile always breaks the graph there
  and runs it eagerly in program order.
* the backward runs single-threaded (``set_multithreading_enabled(False)``):
  every node, and so every NCCL call, is issued by the calling thread (the
  capturing thread in a CUDA graph capture).

Values are opaque ``GradValue``s (``.tensor``; registered inputs get ``.grad``
from ``backward``): ``result.backend.backward(result.energy)`` then
``result.positions.grad``. A schedule (the prevalidated bindings) is required.
This module imports nothing from ``fastmd.distributed``.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor

# The report's per-stage timing buckets (``stats["tape_timing"]``, kept for
# report compatibility); this backend has no per-stage host timing: its
# backward is one autograd call.
_TIMING_KEYS = ("forward_input_ms", "forward_local_ms", "forward_exchange_ms", "backward_setup_ms",
                "backward_local_ms", "backward_exchange_ms", "backward_finish_ms")


class GradValue:
    """An opaque stage value of the autograd backend."""

    __slots__ = ("tensor", "name", "grad")

    def __init__(self, tensor: Tensor, name: str = ""):
        self.tensor, self.name, self.grad = tensor, name, None


class _Cursor:
    """Schedule position of one forward; the backward must walk it back exactly (design §5.3).

    Tensor-free and without a reference to its backend: autograd nodes hold
    it (through ``ctx.stage``), and the C++ nodes are not visited by the
    garbage collector, so anything they reach that points back to a tensor of
    the forward would form an uncollectable cycle.
    """

    __slots__ = ("position", "reverse")

    def __init__(self):
        self.position = 0
        self.reverse: list[str] = []

    def reverse_step(self, stage: "_Stage") -> None:
        # Node k runs only after node k+1 returned its token cotangent.
        self.reverse.append(stage.name)
        if self.position != stage.position + 1:
            raise RuntimeError(f"reverse exchange {stage.name!r} out of order")
        self.position = stage.position


class _Stage:
    """One prepared exchange of an evaluation: its schedule position, plan, binding and cursor."""

    __slots__ = ("name", "position", "plan", "binding", "cursor", "rows")

    def __init__(self, name, position, plan, binding, cursor):
        self.name, self.position, self.plan, self.binding, self.cursor = name, position, plan, binding, cursor
        self.rows = len(plan.requested_keys)

    def zeros_output(self) -> Tensor:
        binding = self.binding
        return torch.zeros((self.rows, *binding.trailing_shape), dtype=binding.dtype, device=binding.device)


class _ExchangeCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tok, stage):
        if torch._C._current_graph_task_id() != -1:
            # A checkpoint (or any) recompute inside a backward would issue this
            # collective again on this rank only (design §5.6).
            raise RuntimeError(f"exchange {stage.name!r}: forward inside a backward (a recompute) "
                               "would re-issue its collective")
        ctx.stage = stage
        ctx.set_materialize_grads(False)
        y = stage.plan._communicate(x, reverse=False, _prepared=stage.binding)
        return y, tok.new_empty(tok.shape)

    @staticmethod
    def backward(ctx, gy, gtok):
        stage = ctx.stage
        stage.cursor.reverse_step(stage)
        if gy is None:
            gy = stage.zeros_output()
        return stage.plan._communicate(gy, reverse=True, _prepared=stage.binding), gtok, None


@torch._dynamo.disable
def _exchange(x: Tensor, tok: Tensor, stage: _Stage):
    return _ExchangeCopy.apply(x, tok, stage)


class _Anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, energy, tok):
        return energy.clone()

    @staticmethod
    def backward(ctx, g):
        # The 0-dim token's cotangent is the seed itself: no kernel.
        return g, g.reshape(())


class _Boundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None
        whole = (g.is_contiguous() and g.storage_offset() == 0
                 and g.numel() * g.element_size() == g.untyped_storage().nbytes())
        return g if whole else g.clone()


def _differentiable(tensor: Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


class AutogradBackend:
    """``GradBackend`` of one P>1 forward on autograd (design §5.7).

    ``schedule`` is the evaluation's ``PrevalidatedSchedule``: ``exchange``
    checks name and plan against it in order, and the backward must visit it
    in exact reverse (checked after the grad). ``differentiable`` names the
    inputs whose adjoint ``backward`` returns. Every floating input becomes a
    leaf requiring grad, so the local stages see one ``requires_grad`` state
    (and kernel choices) whatever the input; inputs outside
    ``differentiable`` are pruned from the backward. Single use.
    """

    def __init__(self, schedule, *, device, differentiable: Sequence[str] = ("owned_positions",),
                 retain_outputs: Sequence[str] = ("coordinates",)):
        if schedule is None:
            raise ValueError("the P>1 autograd backend requires a prevalidated schedule")
        self.schedule = schedule
        self.device = torch.device(device)
        self._differentiable = frozenset(differentiable)
        # Received rows kept past the backward for ``exchange_output`` (the
        # topology epoch's ghost-coordinate check); the others are released
        # with the autograd graph.
        self._retain = frozenset(retain_outputs)
        self._cursor = _Cursor()
        self._outputs: dict[str, Tensor] = {}
        self.inputs: list[GradValue] = []
        # A fresh token per evaluation; inside a capture it lives in the graph pool.
        self.token_leaf = torch.empty((), dtype=torch.float32, device=self.device, requires_grad=True)
        self._token = self.token_leaf
        self._anchored = self._finished = False
        self.stats = {**dict.fromkeys(_TIMING_KEYS, 0.0), "timing_mode": "autograd_no_stage_timing",
                      "stages": {}, "control_ms": {}, "payload_ms": {}}

    # --- GradBackend ---

    def input(self, x: Tensor, name: str = "") -> GradValue:
        if self._finished or self._anchored:
            raise RuntimeError("cannot append to a finished autograd backend")
        leaf = x.detach()
        if _differentiable(leaf):
            leaf.requires_grad_(True)
        value = GradValue(leaf, name)
        self.inputs.append(value)
        return value

    def local(self, fn: Callable, *values: GradValue, name: str = ""):
        if self._finished or self._anchored:
            raise RuntimeError("cannot append to a finished autograd backend")
        with torch.enable_grad():
            # One boundary per distinct value (a repeated argument shares it).
            leaves = {}
            for value in values:
                if id(value) not in leaves:
                    tensor = value.tensor
                    leaves[id(value)] = _Boundary.apply(tensor) if tensor.requires_grad else tensor
            result = fn(*(leaves[id(value)] for value in values))
        if isinstance(result, (tuple, list)):
            return tuple(GradValue(tensor, f"{name}:{index}") for index, tensor in enumerate(result))
        return GradValue(result, f"{name}:0")

    def exchange(self, value: GradValue, plan, name: str = "") -> GradValue:
        if self._finished or self._anchored:
            raise RuntimeError("cannot append to a finished autograd backend")
        calls = self.schedule.exchanges
        position = self._cursor.position
        if position >= len(calls):
            raise RuntimeError("more exchanges than the prepared schedule")
        expected, binding = calls[position]
        if name != expected or binding.plan is not plan:
            raise RuntimeError("exchange order/name/plan differs from the prepared schedule")
        stage = _Stage(name, position, plan, binding, self._cursor)
        self._cursor.position += 1
        y, self._token = _exchange(value.tensor, self._token, stage)
        if name in self._retain:
            # Detached: the retained rows must not keep the autograd graph alive.
            self._outputs[name] = y.detach()
        return GradValue(y, name)

    def energy_anchor(self, energy: GradValue) -> GradValue:
        if self._anchored:
            raise RuntimeError("the energy is already anchored")
        if energy.tensor.numel() != 1:
            raise ValueError("the anchored P>1 energy must be one local scalar")
        self._anchored = True
        return GradValue(_Anchor.apply(energy.tensor, self._token), energy.name)

    # --- the forward's owner surface (evaluator, graph runner, topology epoch) ---

    def backward(self, energy: GradValue, inputs: Sequence[GradValue] | None = None) -> dict[int, Tensor]:
        """One VJP of the anchored local energy seed; never all-reduce that seed.

        Sets ``grad`` on ``inputs`` (default: the inputs named in
        ``differentiable``); every exchange runs its reverse SUM on every rank.
        """
        if self._finished:
            raise RuntimeError("the autograd backend is single-use")
        if not self._anchored:
            raise RuntimeError("the energy must be anchored (energy_anchor) before backward")
        if self._cursor.position != len(self.schedule.exchanges):
            raise RuntimeError("fewer exchanges than the prepared schedule")
        self._finished = True
        targets = (tuple(value for value in self.inputs if value.name in self._differentiable)
                   if inputs is None else tuple(inputs))
        seed = torch.ones_like(energy.tensor)
        with torch.autograd.set_multithreading_enabled(False):
            grads = torch.autograd.grad(energy.tensor, tuple(value.tensor for value in targets) + (self.token_leaf,),
                                        grad_outputs=seed, materialize_grads=True)
        # The graph is released: drop the last token (an output of the last exchange node).
        self._token = None
        if self._cursor.position != 0:
            raise RuntimeError(f"the backward left {self._cursor.position} exchanges unexecuted "
                               f"(reverse order {self._cursor.reverse})")
        for value, grad in zip(targets, grads):
            value.grad = grad
        return {index: value.grad for index, value in enumerate(targets)}

    @property
    def reverse_order(self) -> tuple[str, ...]:
        """Names of the exchanges in the order their backward ran (G-D4)."""
        return tuple(self._cursor.reverse)

    def exchange_output(self, name: str) -> Tensor:
        """The received rows of the forward exchange ``name`` (one of ``retain_outputs``)."""
        try:
            return self._outputs[name]
        except KeyError:
            raise RuntimeError(f"no exchange {name!r} in this forward") from None
