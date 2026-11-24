"""The node-halo exchange of a model that declares its shard (``ShardSpec``): hooks, halo COPY, sharded derivative.

A model whose layers sum their messages at the receiving atom and read only
per-atom features of the senders (MACE, MACE-SOYO, SevenNet, TACE) runs at P>1 on [owned | ghost] rows:
a rank keeps the edges of its owned centres (the receiver rule of the
edges-only ghost builder), every atom within the halo (cutoff + skin) of its
domain is a ghost row holding its owner's wrapped coordinates, and before
each layer that reads the senders' features (the adapter's
``ShardSpec.exchange_points``) the ghost rows of those features are replaced
by a COPY of the owners' rows. Nothing of the upstream model changes: the
runtime installs a hook on each declared module (:class:`ExchangeHooks`),
which replaces the ghost rows only inside the strategy's forward (a context
variable) and passes through everywhere else, so P=1 calls of the same
module in the same process are untouched.

A model whose layers send their messages to both ends of an edge (ORB,
``message_flow="both_ends"``) keeps every edge with an owned end: an edge
between two domains is computed on both ranks from the same COPY'd node
features, so the owned rows' sums are complete. A model whose energy pools
the atoms' features before its head (``readout="pooled"``) runs its own layer
loop (``ShardedTap.sharded_energy``) with a :class:`ShardContext`: it calls the
COPYs itself, pools its owned rows, sums the pooled values over the ranks in
FP64 (:func:`global_sum`, whose backward sums the ranks' gradients) and counts
the head of the sum on rank 0 only.

The derivative is the engine's (:class:`ShardedEnergyDerivative`): the
owned rows' coordinates are the autograd leaf; the ghost rows' coordinates
are their owners' through one differentiable COPY (:func:`halo_copy`), so
one backward sends every ghost row's coordinate and feature gradients back
to its owner (the COPY's reverse SUM); the energy is the FP64 sum of the
owned real rows' energies (a select, never a product with a mask), or the
rank's share of a pooled readout, summed over the ranks after the backward.

The COPY's layout (:class:`HaloLayout`) is the topology's fixed per-peer
blocks (rank-uniform capacities): one ``all_to_all_single`` of fixed split
sizes on the data group, the same call eager and inside a CUDA graph.
"""
from __future__ import annotations

import contextlib
import contextvars
import copy
import math
import weakref
from collections.abc import MutableMapping

import torch
import torch.distributed as dist
from torch import Tensor

from fastmd.engine.energy_derivative import _times, cell_volume
from . import shutdown_registry
from .partition import AtomState, DomainPartition, face_spacings, migrate_atoms

# The exchanges of the running P>1 forward (an _Exchanges), or None: every declared hook passes through.
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("matris_node_halo_exchanges", default=None)
# Only the forward that installs trimming may activate its hooks.
_TRIMMING: contextvars.ContextVar = contextvars.ContextVar("fastmd_node_halo_trimming", default=None)
# ShardSpec.message_flow -> the topologies' edge rule: the edges of the owned centres, or every edge with an owned end
# (an edge between two domains on both ranks).
EDGE_RULES = {"receiver": "receiver", "both_ends": "either", "line_graph": "either"}  # (line_graph: line_partial)
# The count of ShardContext.counted_once calls among a forward's exchanges: once per forward for a pooled readout.
COUNTED_ONCE = "counted_once()"


def expected_calls(spec) -> dict:
    """The calls of one forward of ``spec``: each exchange point and each global sum once, and for a pooled readout
    one ``counted_once`` (the head of the summed values, counted on one rank)."""
    calls = {point.name: 1 for point in (*spec.exchange_points, *spec.global_reductions)}
    if spec.readout == "pooled":
        calls[COUNTED_ONCE] = 1
    return calls


class HaloLayout:
    """Which owned rows each peer receives as ghost rows, in fixed per-peer blocks (one topology refresh).

    ``send_index`` (``[S]`` int64): the owned row (``[0, n_cap)``) each send
    slot carries, the slots grouped by receiving peer in rank order, each
    peer's block padded to the rank-uniform capacity ``send_splits[peer]``
    (a padded slot reads row 0); ``reverse_index``: the same rows with every
    padded slot at row ``n_cap``, a discarded row (the reverse SUM never adds
    a padded slot's gradient to a real row). ``recv_splits[peer]``: the ghost
    rows from each peer (its block, real rows first). ``group``: the data
    process group (None: one rank, no ghost rows).
    """

    def __init__(self, *, n_cap: int, send_index: Tensor, reverse_index: Tensor, send_splits, recv_splits,
                 group, world_size: int):
        self.n_cap = int(n_cap)
        self.send_index, self.reverse_index = send_index, reverse_index
        self.send_splits, self.recv_splits = [int(v) for v in send_splits], [int(v) for v in recv_splits]
        self.ghost_rows = sum(self.recv_splits)
        self.group, self.world_size = group, int(world_size)

    @property
    def rows(self) -> int:
        """Owned slots plus ghost rows: the rows of the model's input."""
        return self.n_cap + self.ghost_rows

    def _all_to_all(self, output: Tensor, input: Tensor, output_splits, input_splits) -> None:
        if self.world_size > 1:
            dist.all_to_all_single(output, input, output_split_sizes=output_splits, input_split_sizes=input_splits,
                                   group=self.group)

    def copy(self, owned: Tensor) -> Tensor:
        """The ghost rows (``[ghost_rows, ...]``, a new tensor) of the owners' ``owned`` rows (``[n_cap, ...]``)."""
        sent = owned.index_select(0, self.send_index)
        received = owned.new_empty((self.ghost_rows, *owned.shape[1:]))
        self._all_to_all(received, sent, self.recv_splits, self.send_splits)
        return received

    def reverse_sum(self, ghost: Tensor) -> Tensor:
        """The owners' sum (``[n_cap, ...]``) of the ghost rows' ``ghost`` values (the COPY's adjoint)."""
        sent = ghost.new_empty((self.send_index.shape[0], *ghost.shape[1:]))
        self._all_to_all(sent, ghost.contiguous(), self.send_splits, self.recv_splits)
        owned = ghost.new_zeros((self.n_cap + 1, *ghost.shape[1:]))
        owned.index_add_(0, self.reverse_index, sent)
        return owned[: self.n_cap]


class _HaloCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owned, layout):
        if torch._C._current_graph_task_id() != -1:
            # A recompute inside a backward would issue this collective again on this rank only.
            raise RuntimeError("node-halo COPY: forward inside a backward (a recompute) would re-issue its collective")
        ctx.layout = layout
        ctx.like = (owned.shape[1:], owned.dtype, owned.device)
        ctx.set_materialize_grads(False)
        return layout.copy(owned)

    @staticmethod
    def backward(ctx, grad):
        layout = ctx.layout
        if grad is None:
            # An unused output: zeros, so that the collective still runs on every rank.
            trailing, dtype, device = ctx.like
            grad = torch.zeros((layout.ghost_rows, *trailing), dtype=dtype, device=device)
        return layout.reverse_sum(grad), None


@torch._dynamo.disable
def halo_copy(owned: Tensor, layout: HaloLayout) -> Tensor:
    """The ghost rows of ``owned`` (differentiable: the gradient of a ghost row is summed into its owner's row)."""
    return _HaloCopy.apply(owned, layout)


class _GlobalSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, layout):
        if torch._C._current_graph_task_id() != -1:
            raise RuntimeError("global sum: forward inside a backward (a recompute) would re-issue its collective")
        ctx.layout = layout
        ctx.like = (values.shape, values.dtype, values.device)
        ctx.set_materialize_grads(False)
        total = values.clone()
        if layout.world_size > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=layout.group)
        return total

    @staticmethod
    def backward(ctx, grad):
        layout = ctx.layout
        if grad is None:
            # An undefined cotangent (the sum is in this rank's graph, its gradient was not materialized): zeros.
            # Every rank's share depends on the sum (a pooled readout's counted_once is a select), so every rank
            # runs this collective.
            shape, dtype, device = ctx.like
            grad = torch.zeros(shape, dtype=dtype, device=device)
        total = grad.clone()
        if layout.world_size > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM, group=layout.group)
        return total, None


@torch._dynamo.disable
def global_sum(values: Tensor, layout: HaloLayout) -> Tensor:
    """The sum over the ranks of ``values`` (every rank's own; differentiable: the backward all-reduces the sum's
    gradient, an undefined one as zeros, so each rank's values get the sum over the ranks of the gradients; every
    rank's share must depend on the sum, or that rank never runs the backward's collective)."""
    return _GlobalSum.apply(values, layout)


class _Exchanges:
    """The exchanges of one P>1 forward, which the declared hooks call (``count_only``: count, change nothing but
    the zero ghost rows of a tensor that holds the owned slots only)."""

    def __init__(self, layout: HaloLayout | None, *, count_only: bool = False):
        self.layout, self.count_only = layout, count_only
        self.fired: dict[str, int] = {}
        self.shapes: dict[str, tuple] = {}

    @property
    def owned_rows(self) -> int:
        """The owned slots (the rows before the ghost rows; without a layout: none known, 0)."""
        return 0 if self.layout is None else self.layout.n_cap

    @property
    def ghost_rows(self) -> int:
        """The ghost rows, after the owned slots (without a layout: none)."""
        return 0 if self.layout is None else self.layout.ghost_rows

    def exchange(self, point, tensor):
        self.fired[point.name] = self.fired.get(point.name, 0) + 1
        if not isinstance(tensor, Tensor):
            if self.count_only:
                self.shapes[point.name] = (type(tensor).__name__,)
                return tensor
            raise TypeError(f"exchange point {point.name!r}: the hooked value is a {type(tensor).__name__}, "
                            "not a tensor")
        layout = self.layout
        # The owned slots only (the modules before it ran on them only, ShardSpec.owned_slot_inputs): the COPY
        # appends the ghost rows (zeros in the counting forward), the same rows as from every row.
        owned_only = (layout is not None and layout.ghost_rows > 0 and tensor.ndim > 0
                      and tensor.shape[0] == layout.n_cap)
        self.shapes[point.name] = (((layout.rows if owned_only else int(tensor.shape[0])), math.prod(tensor.shape[1:]))
                                   if tensor.ndim else (-1, -1))
        if self.count_only:
            if owned_only:
                return torch.cat((tensor, tensor.new_zeros((layout.ghost_rows, *tensor.shape[1:]))))
            return tensor
        if not owned_only and (tensor.ndim == 0 or tensor.shape[0] != layout.rows):
            raise ValueError(f"exchange point {point.name!r}: {tuple(tensor.shape)} rows, the layout has {layout.rows}")
        owned = tensor if owned_only else tensor[: layout.n_cap]
        return torch.cat((owned, halo_copy(owned, layout)))


class ShardContext(_Exchanges):
    """The context of one P>1 forward of an adapter's own layer loop (``fastmd.adapters.protocol.ShardedTap``).

    ``exchange(point, x)``: the COPY of the owners' rows into the ghost rows
    (as a declared hook's); ``global_sum(point, values)``: the FP64 sum over
    the ranks of ``point.width`` values (:func:`global_sum`);
    ``counted_once(value)``: a value the same on every rank, kept on rank 0
    and a zero on the others (a select: every rank's backward still runs
    through it, so that each rank's global sums run their backward
    collective; counted among the forward's calls, once per forward of a
    pooled readout). ``owned``: ``[rows]`` bool, the owned real rows;
    ``n_total``: the atoms of the whole system (FP64 scalar tensor);
    ``first``: ``[]`` bool, this rank is rank 0 (device tensors, read by a
    captured step). ``owned_rows`` and ``ghost_rows``: the layout's owned
    slots (the rows before the ghost rows) and ghost rows, host ints, for a
    layer loop that runs its per-atom work on the owned slots only.
    ``world_size``: the Python rank count, or None for a memory probe
    (including a synthetic layout, which must preserve the P>1 forward).
    ``reverse_sum(x)``: the adjoint of ``exchange`` (one collective), for an
    upstream exchange that is an autograd function of its own, whose forward
    calls ``exchange`` (outside autograd) and whose backward calls this.
    ``count_only``: the counting forward (and the memory probe, with a probe
    layout where a ghost-row trimming is declared): every call
    is counted and returns its input (no collective), except an ``exchange``
    of a tensor that holds the owned slots only (a layout with ghost rows),
    which returns it with zero ghost rows appended. ``exchange(point, x)`` may
    get the owned slots only (a trimmed module's output, ``ShardSpec.
    owned_slot_inputs``): the COPY appends the ghost rows.
    """

    def __init__(self, layout: HaloLayout | None, *, owned: Tensor, n_total: Tensor, first: Tensor,
                 count_only: bool = False):
        super().__init__(layout, count_only=count_only)
        self.owned, self.n_total, self.first = owned, n_total, first
        self.world_size = None if layout is None else layout.world_size

    @property
    def owned_rows(self) -> int:
        """The owned slots (the atoms, then the unused slots): the rows before the ghost rows (without a layout, a
        memory probe's context without trimming, every row)."""
        return int(self.owned.shape[0]) if self.layout is None else self.layout.n_cap

    def reverse_sum(self, grad: Tensor) -> Tensor:
        """The adjoint of :meth:`exchange` (``[rows, ...]``, a new tensor): ``grad``'s owned rows plus the sum of its
        ghost rows' values into their owners' rows, zeros on the ghost rows (the COPY's reverse SUM, one
        collective)."""
        if self.count_only:
            return grad
        layout = self.layout
        if grad.ndim == 0 or grad.shape[0] != layout.rows:
            raise ValueError(f"reverse sum: {tuple(grad.shape)} rows, the layout has {layout.rows}")
        owned = grad[: layout.n_cap] + layout.reverse_sum(grad[layout.n_cap:])
        return torch.cat((owned, grad.new_zeros((layout.ghost_rows, *grad.shape[1:]))))

    def global_sum(self, point, values: Tensor) -> Tensor:
        self.fired[point.name] = self.fired.get(point.name, 0) + 1
        self.shapes[point.name] = (int(values.numel()),)
        if self.count_only:
            return values
        if values.shape != (point.width,) or values.dtype != torch.float64:
            raise ValueError(f"global sum {point.name!r}: {tuple(values.shape)} {values.dtype} values, declared "
                             f"({point.width},) float64")
        return global_sum(values, self.layout)

    def counted_once(self, value: Tensor) -> Tensor:
        self.fired[COUNTED_ONCE] = self.fired.get(COUNTED_ONCE, 0) + 1
        if self.count_only:
            return value
        return torch.where(self.first, value, value.new_zeros(()))


class _InputHook:
    """Forward pre-hook (with kwargs) of one declared input: holds the point and the id of the module it was
    installed on, nothing else (deepcopy-safe; a deep copy's hook passes through). An entry of a mapping
    argument (``arg=(argument, key)``) is replaced in a shallow copy of the mapping."""

    __slots__ = ("point", "target")

    def __init__(self, point, target):
        self.point, self.target = point, id(target)

    def __call__(self, module, args, kwargs):
        exchanges = _ACTIVE.get()
        if exchanges is None or id(module) != self.target:
            return None
        arg, key = self.point.arg if isinstance(self.point.arg, tuple) else (self.point.arg, None)
        if isinstance(arg, str):
            missing = None if arg in kwargs else f"{arg!r} is not a keyword of the call"
        else:
            missing = None if -len(args) <= arg < len(args) else f"no positional argument {arg}"
        value = None if missing else (kwargs[arg] if isinstance(arg, str) else args[arg])
        if missing is None and key is not None and not (isinstance(value, MutableMapping) and key in value):
            missing = f"argument {arg!r} is not a mutable mapping with the entry {key!r}"
        if missing is not None:
            if exchanges.count_only:
                exchanges.shapes[self.point.name] = ("missing",)
                return None
            raise ValueError(f"exchange point {self.point.name!r}: {missing}")
        if key is None:
            value = exchanges.exchange(self.point, value)
        else:
            value = copy.copy(value)
            value[key] = exchanges.exchange(self.point, value[key])
        if isinstance(arg, str):
            return args, {**kwargs, arg: value}
        args = list(args)
        args[arg] = value
        return tuple(args), kwargs


class _OutputHook:
    """Forward hook of one declared output: holds the point and the id of the module it was installed on (as
    ``_InputHook``)."""

    __slots__ = ("point", "target")

    def __init__(self, point, target):
        self.point, self.target = point, id(target)

    def __call__(self, module, args, output):
        exchanges = _ACTIVE.get()
        if exchanges is None or id(module) != self.target:
            return None
        return exchanges.exchange(self.point, output)


class ExchangeHooks:
    """The hooks of a ``ShardSpec``'s exchange points on one module, shared by the evaluators of that module.

    ``ExchangeHooks.acquire(module, points)`` installs them once per module
    (another set of points on the same module is an error) and counts the
    users; ``release()`` removes them with the last user. The hooks hold
    only their point and their module's id and act only inside
    :meth:`exchanging` (the strategy's forward), so the module stays
    deep-copyable (a copy's hooks pass through) and its other callers (P=1
    evaluations in the same process) are unaffected. Registered in the
    shutdown registry (hooks after the graphs that run them).
    """

    _BY_MODULE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

    def __init__(self, module, points):
        self.points = tuple(points)
        self._handles = []
        self.users = 0
        try:
            for point in self.points:
                try:
                    target = module.get_submodule(point.module)
                except AttributeError as error:
                    raise ValueError(f"exchange point {point.name!r}: no module {point.module!r} in the model") from error
                if point.hook == "input":
                    handle = target.register_forward_pre_hook(_InputHook(point, target), with_kwargs=True)
                else:
                    handle = target.register_forward_hook(_OutputHook(point, target))
                self._handles.append(handle)
        except BaseException:
            self.remove()
            raise

    @classmethod
    def acquire(cls, module, points) -> "ExchangeHooks":
        hooks = cls._BY_MODULE.get(module)
        if hooks is None:
            hooks = cls(module, points)
            cls._BY_MODULE[module] = hooks
            shutdown_registry.register(hooks, "hooks")
        elif hooks.points != tuple(points):
            raise ValueError("the module already carries the hooks of other exchange points")
        hooks.users += 1
        return hooks

    def release(self) -> None:
        """One user fewer; the last one removes the hooks."""
        self.users -= 1
        if self.users <= 0:
            self.close()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def close(self) -> None:
        """Remove the hooks (idempotent; the shutdown registry's call)."""
        self.remove()
        self.users = 0
        for module, hooks in list(self._BY_MODULE.items()):
            if hooks is self:
                del self._BY_MODULE[module]
        shutdown_registry.unregister(self, "hooks")

    @staticmethod
    @contextlib.contextmanager
    def exchanging(exchanges: _Exchanges):
        """The declared hooks call ``exchanges`` inside (this thread and context only)."""
        token = _ACTIVE.set(exchanges)
        try:
            yield exchanges
        finally:
            _ACTIVE.reset(token)


def qualified_name(cls) -> str:
    """``"package.module.Class"`` of ``cls``: how a trimming declaration names a class."""
    return f"{cls.__module__}.{cls.__qualname__}"

def declared_modules(module, names) -> list:
    """The submodules of ``module`` that ``names`` declare, in order and each once: a dotted path in ``module``
    (``"layers.0.gate"``), else a class (``"package.module.Class"``: every submodule of exactly that class); a
    ``(module, address)`` entry names its module so. A name that matches no module, or a TorchScript module (which
    takes no hooks), raises ``ValueError``."""
    return [target for target, _ in _declared_entries(module, names)]


def _declared_entries(module, names) -> list:
    """``[(submodule, addresses)]`` of a declaration, each submodule once in declaration order: ``addresses`` None for
    a module named bare (every tensor argument, or the whole output), else the tuple of its entries' addresses."""
    found: list = []
    for name in names:
        selector, address = name if isinstance(name, tuple) else (name, None)
        try:
            matches = [module.get_submodule(selector)]
        except AttributeError:
            matches = [submodule for submodule in module.modules() if qualified_name(type(submodule)) == selector]
        if not matches:
            raise ValueError(f"ShardSpec ghost-row trimming: {selector!r} is neither a module path in the model nor the "
                             "class of one of its modules")
        if any(isinstance(match, torch.jit.ScriptModule) for match in matches):
            raise ValueError(f"ShardSpec ghost-row trimming: {selector!r} is a TorchScript module, which takes no "
                             "hooks")
        for match in matches:
            index = next((i for i, (target, _) in enumerate(found) if target is match), None)
            if index is None:
                found.append((match, None if address is None else (address,)))
            elif (found[index][1] is None) != (address is None):
                # a bare name cuts the tensor arguments (pads a tensor or tuple output) only, not a mapping's entries
                raise ValueError(f"ShardSpec ghost-row trimming: {selector!r} selects a module also declared "
                                 f"{'by entries' if address is None else 'bare'}; declare each module one way")
            elif address is not None:
                found[index] = (match, found[index][1] + (address,))
    return found

def _trimmed_rows(module, target: int, owner: int):
    """(owned slots, every row) of the running P>1 forward for a trimming hook of the module with id ``target``
    registered by the trimming with id ``owner``, or None: outside that forward (another thread's call, a forward
    whose trimming did not register it, one CUDA rank), on a copy of the module, or a rank (or memory probe) without
    ghost rows."""
    exchanges, trimming = _ACTIVE.get(), _TRIMMING.get()
    if exchanges is None or trimming is None or id(trimming) != owner or id(module) != target:
        return None
    ghost = exchanges.ghost_rows
    if not ghost:
        return None
    owned = exchanges.owned_rows
    return owned, owned + ghost + trimming.trailing_rows


def _cut(value, owned: int, every: int, where: str):
    """An addressed argument cut to the owned slots: a tensor with every row (or the owned slots already)."""
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] in (every, owned):
        return value[:owned]
    shape = tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
    raise ValueError(f"ghost-row trimming: {where} is {shape}, not a tensor of the rank's {every} rows or its {owned} "
                     "owned slots")


def _padded(value, owned: int, every: int, where: str | None = None):
    """``value`` with zero rows up to every row where it holds the owned slots (an addressed part: also accepted with
    every row already; else ``ValueError``)."""
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == owned:
        return torch.cat((value, value.new_zeros((every - owned, *value.shape[1:]))))
    if where is not None and not (isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == every):
        shape = tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
        raise ValueError(f"ghost-row trimming: {where} is {shape}, not a tensor of the rank's {owned} owned slots or "
                         f"its {every} rows")
    return value

class _OwnedSlotsHook:
    """Forward pre-hook (with kwargs) of a module of ``ShardSpec.owned_slot_inputs``: inside a P>1 forward, each
    tensor argument (positional or keyword) with the rank's rows (owned slots, ghost rows, trailing rows) keeps the
    owned slots only, every other argument passing as it is; with ``addresses``, only the addressed arguments (a
    positional index, a keyword name, or (one of them, the key of an entry of that mapping argument: replaced in a
    shallow copy)), each of which must hold every row or the owned slots already. Holds the ids of its module and its
    trimming and the addresses only (a deep copy's hook passes). Called without kwargs (another thread's call while
    the forward that registered it removes it: torch then calls it as a plain pre-hook), it returns the arguments
    only."""

    __slots__ = ("target", "owner", "addresses")

    def __init__(self, target, owner, addresses=None):
        self.target, self.owner, self.addresses = id(target), id(owner), addresses

    def __call__(self, module, args, kwargs=None):
        rows = _trimmed_rows(module, self.target, self.owner)
        if rows is None:
            return None
        owned, every = rows
        if self.addresses is None:
            def cut(value):
                return value[:owned] if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == every else value

            args = tuple(cut(arg) for arg in args)
            return args if kwargs is None else (args, {name: cut(value) for name, value in kwargs.items()})
        args, named = list(args), dict(kwargs or {})
        for address in self.addresses:
            arg, key = address if isinstance(address, tuple) else (address, None)
            where = f"{type(module).__name__} argument {arg!r}" + ("" if key is None else f"[{key!r}]")
            if isinstance(arg, str):
                if arg not in named:
                    raise ValueError(f"ghost-row trimming: {where} is not a keyword of the call")
                value = named[arg]
            else:
                if not -len(args) <= arg < len(args):
                    raise ValueError(f"ghost-row trimming: {where}: no positional argument {arg}")
                value = args[arg]
            if key is None:
                value = _cut(value, owned, every, where)
            else:
                if not isinstance(value, MutableMapping) or key not in value:
                    raise ValueError(f"ghost-row trimming: {where} is not an entry of a mutable mapping")
                value = copy.copy(value)
                value[key] = _cut(value[key], owned, every, where)
            if isinstance(arg, str):
                named[arg] = value
            else:
                args[arg] = value
        return tuple(args) if kwargs is None else (tuple(args), named)

class _ZeroGhostHook:
    """Forward hook of a module of ``ShardSpec.zero_ghost_outputs``: inside a P>1 forward, its output (a tensor, or
    each tensor of a tuple or list) holding the owned slots' rows gets zero rows appended up to every row (ghost rows,
    trailing rows), for a consumer that reads every row (those rows' values are never used); with ``addresses``, only
    the addressed parts (a mapping output's entries, in a shallow copy, or a tuple output's elements), each holding
    the owned slots or every row already. Holds the ids of its module and its trimming and the addresses only."""

    __slots__ = ("target", "owner", "addresses")

    def __init__(self, target, owner, addresses=None):
        self.target, self.owner, self.addresses = id(target), id(owner), addresses

    def __call__(self, module, args, output):
        rows = _trimmed_rows(module, self.target, self.owner)
        if rows is None:
            return None
        owned, every = rows
        if self.addresses is None:
            if isinstance(output, Tensor):
                return _padded(output, owned, every)
            if isinstance(output, (tuple, list)):
                values = [_padded(value, owned, every) for value in output]
                return type(output)(*values) if hasattr(output, "_fields") else type(output)(values)
            return None
        for part in self.addresses:
            where = f"{type(module).__name__} output [{part!r}]"
            if isinstance(part, str):
                if not isinstance(output, MutableMapping) or part not in output:
                    raise ValueError(f"ghost-row trimming: {where} is not an entry of a mutable mapping output")
                output = copy.copy(output)
                output[part] = _padded(output[part], owned, every, where)
            else:
                if not isinstance(output, (tuple, list)) or not -len(output) <= part < len(output):
                    raise ValueError(f"ghost-row trimming: {where} is not an element of a tuple or list output")
                values = list(output)
                values[part] = _padded(values[part], owned, every, where)
                output = type(output)(*values) if hasattr(output, "_fields") else type(output)(values)
        return output

class OwnedSlotHooks:
    """The ghost-row trimming of a ``ShardSpec`` (``owned_slot_inputs``, ``zero_ghost_outputs``) on one module.

    The declared modules are resolved once (:func:`declared_modules`: a name
    that matches no module raises ``ValueError``, as does a module whose inputs
    are an input exchange point's; ``paths`` lists them). :meth:`installed` registers
    the hooks for one P>1 forward and removes them after it, also when the
    forward raises; they act only in the context of that forward (a context
    variable it sets, with the running exchanges', ``ExchangeHooks.exchanging``)
    and where its layout holds ghost rows, so another thread's call of a module
    while they are registered passes (also another evaluator's P>1 forward of
    the same module, trimmed or not), and any hook registered before them (e.g.
    a test's recorder) sees the rows a module computes before its output is
    padded. The owned slots and ghost
    rows are the layout's host ints (a new layout is a new captured step).
    """

    def __init__(self, module, spec):
        self.trailing_rows = int(spec.trailing_rows)
        self.cut_entries = _declared_entries(module, spec.owned_slot_inputs)
        self.pad_entries = _declared_entries(module, spec.zero_ghost_outputs)
        self.cut = [target for target, _ in self.cut_entries]
        self.pad = [target for target, _ in self.pad_entries]
        # An input exchange point's module reads the owners' rows in its ghost rows: its inputs are never cut (the
        # exchange's hook runs first, the cut would then drop the COPY).
        for point in spec.exchange_points:
            if point.module is None or point.hook != "input":
                continue
            try:
                exchanged = module.get_submodule(point.module)
            except AttributeError:
                continue  # (refused by the exchange hooks with the reason)
            if any(exchanged is target for target in self.cut):
                raise ValueError(f"ShardSpec ghost-row trimming: {point.module!r} is the input exchange point "
                                 f"{point.name!r}; its inputs cannot also be cut to the owned slots")
        paths = {id(submodule): name for name, submodule in module.named_modules()}
        # The declared modules' paths (a class declares every submodule of it): what config_summary reports.
        self.paths = {"owned_slot_inputs": [paths[id(target)] for target in self.cut],
                      "zero_ghost_outputs": [paths[id(target)] for target in self.pad]}

    @contextlib.contextmanager
    def installed(self):
        handles = []
        token = _TRIMMING.set(self)
        try:
            for target, addresses in self.cut_entries:
                handles.append(target.register_forward_pre_hook(_OwnedSlotsHook(target, self, addresses),
                                                                with_kwargs=True))
            for target, addresses in self.pad_entries:
                handles.append(target.register_forward_hook(_ZeroGhostHook(target, self, addresses)))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            _TRIMMING.reset(token)


class ShardedEnergyDerivative:
    """The engine-owned derivative at P>1 over a node-halo layout: forces, the rank's energy and its virial.

    ``derivative(inputs, rows, cell, owned_mask, layout, stress=False)``:
    ``rows`` (``[rows, 3]``) are the builder's published coordinates (owned
    slots, then ghost rows, each ghost row its owner's wrapped coordinates),
    ``inputs`` the adapter's ``prepare`` of the same batch, ``owned_mask``
    (``[rows]`` bool) the owned real rows. Returns ``(totals, forces [n_cap,
    3])``: ``totals`` (float64) the FP64 sum of this rank's owned real rows'
    energies (``[1]``), followed with ``stress`` by this rank's nine strain
    gradients (``[10]``; the caller sums the vector over the ranks in one
    all-reduce after the backward, and the stress is its last nine values over
    the cell volume), and minus the gradient of the whole system's energy with
    respect to the owned slots' coordinates. The ghost coordinates and the
    declared node features reach the model through COPYs, so one backward
    returns every ghost row's gradient to its owner. For the stress a zero
    ``[3, 3]`` strain leaf deforms this rank's owned rows and its cell by its
    symmetric part (the arithmetic of the single-GPU ``EnergyDerivative``)
    before the coordinates' COPY, so every row is strained exactly once, by
    its owner (a ghost row is its owner's strained row) and the cell-dependent
    terms read the rank's strained cell. The backward runs single-threaded
    (every collective issued by the calling thread, the capturing one in a
    CUDA graph capture).

    An adapter's own layer loop (``spec.adapter_called``:
    ``ShardedTap.sharded_energy``) returns the rank's share of the energy
    itself, through a :class:`ShardContext`: a per-atom readout the FP64 sum
    of its owned real rows' energies; a pooled readout (``spec.readout ==
    "pooled"``; ``n_total``, ``first``: its device scalars) its owned rows'
    pooled sums through the global sums, the head of the summed values
    counted on rank 0 only, and the global sums' backward sums every rank's
    gradient of them, so each rank's backward gives the whole system's
    gradient on its rows. The declared ghost-row trimming (``trim``, an
    :class:`OwnedSlotHooks` or None) is registered around each forward of
    :meth:`local_energy`: the step's, the counting forward's and the memory
    probe's (a probe layout of owned slots and ghost rows: the probe measures
    the trimmed peak).
    """

    def __init__(self, adapter, spec):
        self.adapter = adapter
        self.tap = spec.adapter_called
        self.expected = expected_calls(spec)
        # The declared ghost-row trimming (its modules resolved now: a name that matches none raises ValueError).
        self.trim = (OwnedSlotHooks(adapter.module, spec) if spec.owned_slot_inputs or spec.zero_ghost_outputs
                     else None)
        if self.trim is not None and getattr(adapter, "compiled", False):
            # The hooks are registered and removed around every forward, which would invalidate compiled guards.
            raise ValueError(f"model {getattr(adapter, 'name', type(adapter).__name__)!r}: ghost-row trimming needs "
                             "eager modules (the adapter runs a compiled module)")

    def context(self, layout, owned_mask, n_total, first, *, count_only: bool = False):
        """The exchanges of one forward: a :class:`ShardContext` for the adapter's own layer loop, else the
        declared hooks' (``count_only``: the counting forward's)."""
        if self.tap:
            return ShardContext(layout, owned=owned_mask, n_total=n_total, first=first, count_only=count_only)
        return _Exchanges(layout, count_only=count_only)

    def local_energy(self, inputs, positions: Tensor, cell: Tensor, owned_mask: Tensor, exchanges) -> Tensor:
        """This rank's share of the energy (FP64 scalar) at ``positions`` and ``cell`` (``exchanges`` active), with
        the declared ghost-row trimming's hooks registered for this forward. An energy tap may return the owned
        slots' energies only (its readout trimmed) or every row's."""
        with contextlib.nullcontext() if self.trim is None else self.trim.installed():
            if self.tap:
                return self.adapter.sharded_energy(inputs, positions, cell, exchanges).to(torch.float64).sum()
            energies = self.adapter.node_energy(inputs, positions, cell).to(torch.float64)
        if energies.shape[0] != owned_mask.shape[0]:
            # The owned slots' energies only (the readout ran on them only, ShardSpec.owned_slot_inputs).
            owned = exchanges.owned_rows
            if not (exchanges.ghost_rows and energies.shape[0] == owned):
                raise ValueError(f"node_energy returned {energies.shape[0]} rows; the rank has "
                                 f"{owned_mask.shape[0]} rows ({owned} owned slots)")
            owned_mask = owned_mask[:owned]
        # A select, not a product: a padded row's energy may be anything, NaN included.
        return torch.where(owned_mask, energies, energies.new_zeros(())).sum()

    def __call__(self, inputs, rows: Tensor, cell: Tensor, owned_mask: Tensor, layout: HaloLayout, *,
                 stress: bool = False, n_total: Tensor | None = None, first: Tensor | None = None):
        exchanges = self.context(layout, owned_mask, n_total, first)
        with torch.enable_grad(), ExchangeHooks.exchanging(exchanges):
            leaf = rows[: layout.n_cap].detach().requires_grad_(True)
            owned, leaves = leaf, (leaf,)
            if stress:
                strain = torch.zeros((3, 3), dtype=leaf.dtype, device=leaf.device, requires_grad=True)
                symmetric = 0.5 * (strain + strain.transpose(0, 1))
                owned = leaf + _times(leaf, symmetric)
                cell = cell + _times(cell, symmetric)
                leaves = (leaf, strain)
            positions = torch.cat((owned, halo_copy(owned, layout)))
            local = self.local_energy(inputs, positions, cell, owned_mask, exchanges)
            if exchanges.fired != self.expected:
                # Agreed before the first exchanging forward, so a difference here is the same on every rank (the
                # hooks were removed, e.g. by a closed evaluator): no rank enters the backward's collectives.
                raise RuntimeError(f"node-halo forward: declared exchanges called {exchanges.fired}, expected "
                                   f"{self.expected} (closed evaluator or removed hooks?)")
            with torch.autograd.set_multithreading_enabled(False):
                grads = torch.autograd.grad(local, leaves)
        totals = local.detach().reshape(1)
        if stress:
            totals = torch.cat((totals, grads[1].to(torch.float64).reshape(9)))
        return totals, -grads[0]


def summed_stress(totals: Tensor, cell: Tensor, dtype: torch.dtype) -> Tensor | None:
    """The stress (``[3, 3]`` eV/Å³, in ``dtype``) of ``totals`` summed over the ranks (``[10]``: the energy, then
    the strain gradients): their FP64 sum over the cell volume; None for ``totals`` without strain gradients."""
    if totals.shape[0] == 1:
        return None
    return (totals[1:] / cell_volume(cell.to(torch.float64))).reshape(3, 3).to(dtype)


def count_exchanges(derivative: ShardedEnergyDerivative, inputs, rows: Tensor, cell: Tensor, layout: HaloLayout,
                    owned_mask: Tensor, n_total: Tensor | None = None,
                    first: Tensor | None = None) -> tuple[dict, dict]:
    """One forward without exchanges (no gradient) that counts the declared exchanges' calls and shapes (the hooks',
    or those of the adapter's own layer loop): what every rank must agree before the first forward that exchanges
    (an upstream branch that skips a layer on one rank would otherwise leave its peers in a collective)."""
    exchanges = derivative.context(layout, owned_mask, n_total, first, count_only=True)
    with torch.no_grad(), ExchangeHooks.exchanging(exchanges):
        # A view: upstream marks its positions as requiring grad.
        derivative.local_energy(inputs, rows.detach(), cell, owned_mask, exchanges)
    return exchanges.fired, exchanges.shapes


def agree_exchanges(context, spec, fired: dict, shapes: dict, rows: int) -> None:
    """Every rank's counting forward called each declared exchange once, on a tensor of its ``rows`` rows and the
    declared width, each global sum once on its declared width, and for a pooled readout ``counted_once`` once
    (collective; an error on any rank raises on every rank)."""
    expected = expected_calls(spec)
    error = None
    if fired != expected:
        error = f"declared exchanges called {fired}, expected once each ({sorted(expected)})"
    else:
        for point in spec.exchange_points + spec.global_reductions:
            shape = (rows, point.width) if point.kind == "copy" else (point.width,)
            if shapes.get(point.name) != shape:
                error = (f"exchange point {point.name!r}: rows and width {shapes.get(point.name)}, expected {shape}"
                         if point.kind == "copy" else
                         f"global sum {point.name!r}: values {shapes.get(point.name)}, expected {shape}")
                break
    context.collective_check(error)


def _prepared(adapter, batch):
    """``(adapter.prepare(batch), None)``, or ``(None, the error)`` for the caller to agree collectively."""
    try:
        return adapter.prepare(batch), None
    except (ValueError, TypeError, RuntimeError) as error:
        return None, f"{type(error).__name__}: {error}"


def md_step_flags(md, totals: Tensor, forces: Tensor, status, positions: Tensor) -> Tensor:
    """The in-graph MD step's post-force half and its flags (``ShardedEvaluator.md_step``; device only, inside the
    captured step after the energy's sum over the ranks and before the flags' MAX). ``md`` is the MD driver's hook
    (``post(forces, energy, rows)``, ``flags(rows)``), ``positions`` the ``[rows, 3]`` owned slots its pre-force half
    wrote and ``status`` the step's build. Returns ``[capacity overflow, largest displacement (float32 bits, a
    nonnegative float's bits order as its value), non-finite step positions, the integrator's non-finite positions or
    momenta]``: the overflow of a capacity, not of used-up candidates (which the stale flag reports)."""
    rows = positions.shape[0]
    with torch.no_grad():
        md.post(forces[:rows], totals[:1], rows)
        capacity = status.overflow.reshape(-1)[:1] & status.candidate_valid.reshape(-1)[:1]
        displacement = status.max_displacement.reshape(1).to(torch.float32).view(torch.int32).to(torch.int64)
        return torch.cat((capacity.long(), displacement, (~torch.isfinite(positions).all()).long().reshape(1),
                          md.flags(rows).reshape(1).long()))


class NodeHaloStrategy:
    """The node-halo exchange of one rank at P>1 on CUDA: the topology, the declared exchanges and the derivative.

    ``refresh(state)`` (collective, outside any CUDA graph): the
    ``NodeHaloTopology`` refresh (the shard's edge rule: the edges of the
    owned centres, or every edge with an owned end), the ghost rows' COPY
    layout and the adapter's input over the builder's batch, prepared once
    per ``key`` (a refresh that keeps the buffers runs ``prepare`` for its
    checks only and keeps the input a captured step reads).
    ``set_positions(positions)``: the owned coordinates
    (outside a graph). ``step(stress)`` (device only, collective: what a
    captured step replays): the ghost coordinates' COPY and the exact filter,
    the model with its exchanges and the backward (``ShardedEnergyDerivative``),
    the energy's (and strain gradients') sum and the step flags' MAX over the
    ranks; returns ``(energy [1] float64, forces [n_cap, 3], flags [4] ([5] with a neighbour cap),
    stress)`` (flags: overflow, used-up candidates, non-finite results, a
    rank's new state: ``new_state``, set by the caller; for a model with a
    neighbour cap (``TopologySpec.max_neighbors``) a fifth, the most
    neighbours of an atom in the builds since the caller last zeroed
    ``neighbor_count`` (before each step); ``stress``: the [3, 3]
    stress in eV/Å³ in the positions' dtype, None without ``stress``).
    ``step(stress, md)``: the in-graph MD step (``ShardedEvaluator.md_step``)
    with the MD driver's hook ``md``: its pre-force half writes the owned
    slots before the build, its post-force half runs after the energy's sum,
    and the four flags of :func:`md_step_flags` follow the step's own (one
    MAX); it zeroes ``neighbor_count`` itself, first, inside the captured
    step. ``refresh(state, floor)``: the topology's capacity floor.
    ``key``: changes when the buffers a captured step reads are replaced.
    The edge rule follows ``spec.message_flow`` (:data:`EDGE_RULES`).
    ``positions_dtype`` and ``cell``: the geometry (``NodeHaloTopology``'s):
    float64 positions, COPY and published cell for float64 geometry.
    """

    def __init__(self, adapter, spec, partition, context, *, skin: float, device,
                 positions_dtype: torch.dtype = torch.float32, cell=None):
        from .node_halo_topology import NodeHaloTopology

        self.adapter, self.context = adapter, context
        topology_spec = adapter.topology_spec()
        # The builds' maximum of an atom's edges, for a model with a neighbour cap (an owned atom's edges are all on
        # its rank, a ghost row's at most all: the MAX over the ranks is the whole system's); the evaluator zeroes it
        # before each step.
        self.neighbor_count = (None if topology_spec.max_neighbors is None
                               else torch.zeros(1, dtype=torch.int64, device=device))
        self.topology = NodeHaloTopology(partition, context, cutoff=topology_spec.cutoff, skin=skin,
                                         edges=EDGE_RULES[spec.message_flow], padding=adapter.padding_rule(),
                                         pad_species=spec.pad_species, max_neighbor_count=self.neighbor_count,
                                         capacity_step=spec.capacity_step, positions_dtype=positions_dtype, cell=cell)
        self.derivative = ShardedEnergyDerivative(adapter, spec)
        self.inputs = self.layout = self._inputs_key = None
        self.n_owned = 0
        # Set by the caller before a step (outside the graph): this rank's input is not the refreshed state.
        self.new_state = torch.zeros(1, dtype=torch.long, device=device)
        # The atoms of the whole system (set at each refresh) and whether this is rank 0: what an adapter's own
        # layer loop reads (ShardContext), device scalars a captured step reads.
        self.n_total = torch.zeros((), dtype=torch.float64, device=device)
        self.first = torch.full((), context.rank == 0, dtype=torch.bool, device=device)

    def refresh(self, state: AtomState, floor: dict | None = None) -> AtomState:
        topology = self.topology
        state = topology.refresh(state, floor)
        self.layout = topology.layout()
        inputs, error = _prepared(self.adapter, topology.builder.batch)
        # The adapter's checks (e.g. an element outside the model) fail on the ranks holding the atom only.
        self.context.collective_check(error)
        if self._inputs_key != self.key:
            # The adapter's input over these buffers, prepared once as a single-system builder does: a step captured
            # over them reads its tensors, so a refresh that keeps the buffers keeps it (its prepare only checks).
            self.inputs, self._inputs_key = inputs, self.key
        self.n_owned = topology.n_owned
        self.n_total.fill_(topology.n_total)
        return state

    def set_positions(self, positions: Tensor) -> None:
        self.topology.set_positions(positions)

    def step(self, stress: bool = False, md=None):
        topology = self.topology
        builder = topology.builder
        with torch.no_grad():
            if md is not None:
                if self.neighbor_count is not None:
                    self.neighbor_count.zero_()  # this step's builds only (inside the captured MD step)
                positions = md.pre(topology.n_cap)
                topology.load_owned_slots(positions)
            _, status = topology.build_rows()
        batch = builder.batch
        totals, forces = self.derivative(self.inputs, batch.positions, batch.cell, builder.owned_mask, self.layout,
                                         stress=stress, n_total=self.n_total, first=self.first)
        flags = torch.cat((status.overflow.reshape(-1)[:1].long(), (~status.candidate_valid.reshape(-1)[:1]).long(),
                           (~(torch.isfinite(totals).all() & torch.isfinite(forces).all())).long().reshape(1),
                           self.new_state, *(() if self.neighbor_count is None else (self.neighbor_count,))))
        group = self.context.group
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=group)
        if md is not None:
            flags = torch.cat((flags, md_step_flags(md, totals, forces, status, positions)))
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
        return totals[:1], forces, flags, summed_stress(totals, batch.cell, forces.dtype)

    def count(self):
        builder = self.topology.builder
        batch = builder.batch
        return count_exchanges(self.derivative, self.inputs, batch.positions, batch.cell, self.layout,
                               builder.owned_mask, self.n_total, self.first)

    def extents(self) -> dict:
        """This rank's sizes of the current layout (the memory admission's extents)."""
        layout = self.layout
        return {"owned": self.n_owned, "owned_slots": layout.n_cap, "rows": layout.rows,
                "ghost_rows": layout.ghost_rows, "send_rows": int(layout.send_index.shape[0]),
                "edges": int(self.topology.builder.e_capacity)}

    @property
    def key(self):
        topology = self.topology
        return (topology.layouts, topology.builders, id(topology.builder))

    def close(self):
        self.topology.close()
        self.inputs = self.layout = self._inputs_key = None


# --- the CPU reference topology (Gloo; exact, rebuilt at every evaluation) -------------------------------


class _ReferenceBatch:
    """The edges-only graph of one rank's rows on the CPU (the fields an energy tap reads)."""

    def __init__(self, positions, cell, species, edge_index, edge_image, cutoff):
        from fastmd.adapters.protocol import GraphBatch

        self.batch = GraphBatch(positions=positions, frac_coords=positions.new_zeros(positions.shape), cell=cell,
                                species=species, n_real=int(positions.shape[0]), edge_index=edge_index,
                                edge_image=edge_image, n_edges=None, cutoff=cutoff, padding=None)


def _image_range(cell: Tensor, cutoff: float, spacings: Tensor | None = None) -> Tensor:
    """Every integer image whose cell translation may hold a neighbour within ``cutoff`` of a wrapped row (a pair's
    distance exceeds its fractional offset along an axis times the cell's face spacing there). ``spacings``: the
    face spacings (``DomainPartition.face_spacings``, an orthogonal cell's lengths); None: computed from ``cell``."""
    spacings = face_spacings(cell) if spacings is None else spacings
    reach = [int(math.ceil(cutoff / float(spacing))) + 1 for spacing in spacings]
    axes = [torch.arange(-k, k + 1, dtype=torch.float64) for k in reach]
    return torch.cartesian_prod(*axes)


def reference_edges(positions: Tensor, owned: int, cell: Tensor, cutoff: float,
                    edges: str = "receiver", spacings: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """The directed edges (centre, neighbour) and images of the owned centres (rows ``[0, owned)``) to every row
    within ``cutoff`` (exact, any image; not a row to itself at image 0), centre-major, on the CPU; ``edges="either"``
    adds the edges of the other rows to the owned ones (the reverse of each owned centre's edge to a row at or past
    ``owned``, its image negated)."""
    images = _image_range(cell, cutoff, spacings).to(positions.dtype)
    shifts = images @ cell.to(positions.dtype)
    centres = positions[:owned]
    # [owned, rows, images, 3]: neighbour + shift - centre
    vectors = positions[None, :, None, :] + shifts[None, None, :, :] - centres[:, None, None, :]
    distance = torch.linalg.vector_norm(vectors, dim=-1)
    keep = distance < cutoff
    zero = int(torch.nonzero((images == 0).all(dim=1))[0])
    rows = torch.arange(positions.shape[0])
    keep[rows[:owned], rows[:owned], zero] = False
    centre, neighbour, image = torch.nonzero(keep, as_tuple=True)
    edge_image = images[image]
    if edges == "either":
        other = neighbour >= owned
        order = torch.argsort(neighbour[other], stable=True)
        centre, neighbour = (torch.cat((centre, neighbour[other][order])),
                             torch.cat((neighbour, centre[other][order])))
        edge_image = torch.cat((edge_image, -edge_image[other][order]))
    edge_index = torch.stack((centre, neighbour), dim=1).to(torch.int32)
    return edge_index, edge_image.to(torch.float32)


class NodeHaloReferenceTopology:
    """The node-halo rows and edges of one rank on the CPU: exact, rebuilt at every evaluation (Gloo).

    The rows of :class:`~fastmd.distributed.node_halo_topology.NodeHaloTopology`
    without padding: the owned atoms (ID order), then each sending rank's
    atoms within the halo of this domain (its rows in its order), each with
    its owner's wrapped coordinates; the edges are those of the owned centres
    to every row within the cutoff, at every image (a brute-force search: the
    reference of the CUDA builder, for small systems), and with
    ``edges="either"`` also those of the ghost rows to the owned ones.
    ``dtype``: the rows' (the geometry's); ``cell``: the cell of the rows'
    wrap and edge vectors (float64 geometry of a float32 model: the exact
    float64 cell; default the partition's), the halo selected from the
    coordinates in the partition's dtype.
    """

    def __init__(self, partition: DomainPartition, context, *, cutoff: float, dtype: torch.dtype,
                 pad_species: int = 1, edges: str = "receiver", cell=None):
        if not (math.isfinite(cutoff) and cutoff > 0):
            raise ValueError("cutoff must be finite and positive")
        if edges not in ("receiver", "either"):
            raise ValueError(f"edges must be 'receiver' or 'either', got {edges!r}")
        self.partition, self.context, self.cutoff, self.dtype = partition, context, float(cutoff), dtype
        self.pad_species, self.edges = int(pad_species), edges
        # The wrap of the rows: the partition's, or one of the same grid over ``cell`` (its own cell in ``dtype``).
        self.wrap = partition if cell is None else DomainPartition(torch.as_tensor(cell).detach().cpu().to(dtype),
                                                                   partition.world_size, partition.domain_grid)
        self.layout = self.batch = self.owned_mask = None
        self.n_owned = 0
        self.stats = {}

    def refresh(self, state: AtomState) -> AtomState:
        """Migrate ``state`` and build this evaluation's rows, layout and edges (collective)."""
        context, partition = self.context, self.partition
        # (Owners decided on the coordinates in the partition's dtype, as the halo.)
        state = migrate_atoms(state, partition, context, owner_dtype=None if self.wrap is partition
                              else partition.cell.dtype).sorted()
        world, rank = context.world_size, context.rank
        n = len(state.ids)
        cell = self.wrap.cell.to(self.dtype)
        wrapped = self.wrap.wrap_positions(state.positions.to(self.dtype))
        if world > 1:
            destinations = partition.halo_destinations(state.positions.to(partition.cell.dtype), self.cutoff)
            destinations[rank] = destinations[rank][:0]
        else:
            destinations = [torch.zeros(0, dtype=torch.int64)]
        send_counts = [int(rows.numel()) for rows in destinations]
        counts = torch.tensor(send_counts, dtype=torch.int64)
        received = torch.zeros(world, dtype=torch.int64)
        if world > 1:
            dist.all_to_all_single(received, counts, group=context.group)
        recv_counts = received.tolist()
        send_rows = torch.cat(destinations) if destinations else torch.zeros(0, dtype=torch.int64)
        records = torch.stack((state.ids.long()[send_rows], state.numbers.long()[send_rows]), dim=1)
        incoming = records.new_empty((sum(recv_counts), 2))
        if world > 1:
            dist.all_to_all_single(incoming, records, output_split_sizes=recv_counts, input_split_sizes=send_counts,
                                   group=context.group)
        # At least one owned slot: a rank without atoms keeps one unused row (pad species, at the origin, masked
        # out), so that the model never runs on zero rows.
        n_cap = max(n, 1)
        owned = torch.cat((wrapped, wrapped.new_zeros((n_cap - n, 3))))
        self.layout = HaloLayout(n_cap=n_cap, send_index=send_rows, reverse_index=send_rows, send_splits=send_counts,
                                 recv_splits=recv_counts, group=context.group, world_size=world)
        rows = torch.cat((owned, self.layout.copy(owned)))
        pads = torch.full((n_cap - n,), self.pad_species, dtype=torch.long)
        species = torch.cat((state.numbers.long(), pads, incoming[:, 1])).to(torch.int32)
        edge_index, edge_image = reference_edges(rows, n, cell, self.cutoff, self.edges, partition.face_spacings)
        self.batch = _ReferenceBatch(rows, cell, species, edge_index, edge_image, self.cutoff).batch
        self.owned_mask = torch.arange(rows.shape[0]) < n
        self.atom_ids = torch.cat((state.ids.long(), torch.full((n_cap - n,), -1, dtype=torch.long), incoming[:, 0]))
        self.n_owned = n
        self.stats = {"owned_atoms": n, "ghost_atoms": int(incoming.shape[0]), "edges": int(edge_index.shape[0])}
        return state
