"""Bucketed CUDA-graph execution for MatRIS MD (static-shape forward+backward).

The model forward is D2H-sync-free, so a fixed-shape forward+backward can be
captured with torch.cuda.graph and replayed each MD step (collapsing ~3300
kernel launches into one replay -> removes launch-gap GPU idle).

MD edge/triplet counts fluctuate, so shapes are made static by padding to fixed
CAPACITY BUCKETS: padding edges/triplets are routed to dummy SINK atoms (index
>= N) that are masked out of the energy/reference via the model's n_real arg, so
results are numerically identical to the ragged model on the real atoms. One
graph is captured per (U_cap, T_cap) bucket and selected at replay time from the
current real counts (host-known .shape, no sync); MD counts cluster into a few
buckets, so after warmup every step is a cache hit.

Note: this does NOT make the ragged neighbor-graph BUILD sync-free (the sparse
three-body line graph cannot be densified without exploding compute); it removes
the forward's launch-gap idle, which is the dominant, addressable component.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .._nvtx import nvtx_range
from ..graph.radiusgraph import RadiusGraph
from ._config import scoped_call, scoped_init
from ._graph_capture import capture_graph


def pad_radius_graph(g: RadiusGraph, u_max: int, t_max: int, n_dummy: int = 64):
    """Pad a single ragged RadiusGraph to fixed capacity (e_max=2*u_max, t_max)
    with n_dummy dummy SINK atoms. Returns (padded_graph, n_real).

    Padding self-loops use a large periodic-image shift so lengths are finite
    (radial basis -> 0) and unit vectors well-defined. Padding edges/triplets are
    spread across the dummy atoms and padding undirected edges to avoid atomic-add
    scatter contention.
    """
    with nvtx_range("pad_radius_graph"):
        dev = g.lattice.device
        n = g.atomic_number.shape[0]
        e = g.atom_graph.shape[0]
        u = g.undirected2directed.shape[0]
        t = g.line_graph.shape[0]
        e_max = 2 * u_max
        n_pad_u = u_max - u
        assert e <= e_max and n_pad_u >= 1 and t <= t_max, (e, u, t, e_max, u_max, t_max)
        D = n_dummy  # fixed -> total atom count (N+D) is static across steps

        atomic_number = torch.cat([g.atomic_number, g.atomic_number.new_ones(D)])
        atom_frac_coord = torch.cat([g.atom_frac_coord, g.atom_frac_coord.new_zeros(D, 3)])

        e_pad = e_max - e
        pad_edge_pair = torch.arange(e_pad, device=dev) // 2
        pad_edge_sink = n + (pad_edge_pair * D // n_pad_u)
        atom_graph = g.atom_graph.new_zeros(e_max, 2)
        atom_graph[:e] = g.atom_graph
        atom_graph[e:, 0] = pad_edge_sink
        atom_graph[e:, 1] = pad_edge_sink

        neighbor_image = g.neighbor_image.new_zeros(e_max, 3)
        neighbor_image[:e] = g.neighbor_image
        neighbor_image[e:] = 3.0

        pad_u_ids = torch.arange(u, u_max, device=dev)
        directed2undirected = g.directed2undirected.new_zeros(e_max)
        directed2undirected[:e] = g.directed2undirected
        directed2undirected[e:] = pad_u_ids.repeat_interleave(2)
        undirected2directed = g.undirected2directed.new_zeros(u_max)
        undirected2directed[:u] = g.undirected2directed
        undirected2directed[u:] = torch.arange(e, e_max, 2, device=dev)

        line_graph = g.line_graph.new_zeros(t_max, 5)
        if t > 0:
            line_graph[:t] = g.line_graph
        t_pad = t_max - t
        pad_triplet_atom = torch.arange(t_pad, device=dev) * D // max(t_pad, 1)
        pu = torch.arange(t_pad, device=dev) % n_pad_u
        line_graph[t:, 0] = n + pad_triplet_atom
        line_graph[t:, 1] = u + pu
        line_graph[t:, 2] = e + 2 * pu
        line_graph[t:, 3] = u + pu
        line_graph[t:, 4] = e + 2 * pu

        pg = RadiusGraph(
            atomic_number=atomic_number, atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph, neighbor_image=neighbor_image,
            directed2undirected=directed2undirected, undirected2directed=undirected2directed,
            line_graph=line_graph, lattice=g.lattice, graph_id=None, mp_id=None,
            composition=g.composition, atom_graph_cutoff=g.atom_graph_cutoff,
            line_graph_cutoff=g.line_graph_cutoff,
            atom_target_sorted=getattr(g, "atom_target_sorted", False),
            line_atom_sorted=getattr(g, "line_atom_sorted", False),
            isolated_atom_count=g.isolated_atom_count,
        )
        return pg, n


@dataclass
class _PaddedGraphWorkspace:
    pg: RadiusGraph
    n_real: int
    u_max: int
    t_max: int
    n_dummy: int
    edge_pair: torch.Tensor
    u_ids: torch.Tensor
    twice_u_ids: torch.Tensor
    triplet_ids: torch.Tensor

    @classmethod
    def create(
        cls,
        pg: RadiusGraph,
        n_real: int,
        u_max: int,
        t_max: int,
        n_dummy: int,
    ) -> "_PaddedGraphWorkspace":
        dev = pg.lattice.device
        e_max = 2 * u_max
        edge_pair = torch.arange(e_max, device=dev, dtype=pg.atom_graph.dtype) // 2
        u_ids = torch.arange(u_max, device=dev, dtype=pg.undirected2directed.dtype)
        return cls(
            pg=pg,
            n_real=n_real,
            u_max=u_max,
            t_max=t_max,
            n_dummy=n_dummy,
            edge_pair=edge_pair,
            u_ids=u_ids,
            twice_u_ids=2 * u_ids,
            triplet_ids=torch.arange(t_max, device=dev, dtype=pg.line_graph.dtype),
        )

    def update(self, g: RadiusGraph) -> None:
        with nvtx_range("static input copy"):
            pg = self.pg
            n = g.atomic_number.shape[0]
            e = g.atom_graph.shape[0]
            u = g.undirected2directed.shape[0]
            t = g.line_graph.shape[0]
            e_max = 2 * self.u_max
            n_pad_u = self.u_max - u
            assert n == self.n_real, (n, self.n_real)
            assert e <= e_max and n_pad_u >= 1 and t <= self.t_max, (
                e,
                u,
                t,
                e_max,
                self.u_max,
                self.t_max,
            )

            pg.atomic_number[:n].copy_(g.atomic_number)
            pg.atom_frac_coord[:n].copy_(g.atom_frac_coord)
            pg.lattice.copy_(g.lattice)
            pg.atom_graph[:e].copy_(g.atom_graph)
            pg.neighbor_image[:e].copy_(g.neighbor_image)
            pg.directed2undirected[:e].copy_(g.directed2undirected)
            pg.undirected2directed[:u].copy_(g.undirected2directed)
            if t > 0:
                pg.line_graph[:t].copy_(g.line_graph)

            if e < e_max:
                e_pad = e_max - e
                pad_edge_sink = n + (
                    self.edge_pair[:e_pad] * self.n_dummy // n_pad_u
                )
                pg.atom_graph[e:, 0].copy_(pad_edge_sink)
                pg.atom_graph[e:, 1].copy_(pad_edge_sink)
                pg.neighbor_image[e:].fill_(3.0)
                pg.directed2undirected[e:e_max:2].copy_(self.u_ids[u:self.u_max])
                pg.directed2undirected[e + 1 : e_max : 2].copy_(self.u_ids[u:self.u_max])
                pg.undirected2directed[u:self.u_max].copy_(self.twice_u_ids[u:self.u_max])

            if t < self.t_max:
                t_pad = self.t_max - t
                triplet_ids = self.triplet_ids[:t_pad]
                pu = triplet_ids % n_pad_u
                pg.line_graph[t:, 0].copy_(
                    n + (triplet_ids * self.n_dummy // max(t_pad, 1))
                )
                pg.line_graph[t:, 1].copy_(u + pu)
                pg.line_graph[t:, 2].copy_(e + 2 * pu)
                pg.line_graph[t:, 3].copy_(u + pu)
                pg.line_graph[t:, 4].copy_(e + 2 * pu)


def _cap_for(u, t, u_step, t_step, min_pad_u):
    u_cap = ((u // u_step) + 1) * u_step
    if u_cap - u < min_pad_u:
        u_cap += u_step
    t_cap = ((t // t_step) + 1) * t_step
    return int(u_cap), int(t_cap)


def _smallest_fitting_key(keys, u, t, u_step, t_step):
    fitting = [key for key in keys if key[0] > u and key[1] >= t]
    if not fitting:
        return None
    return min(
        fitting,
        key=lambda key: (
            (key[0] - u) / u_step + (key[1] - t) / t_step,
            key[0],
            key[1],
        ),
    )


class BucketedGraphRunner:
    """Captures one CUDA graph per capacity bucket; replays the matching one."""

    @scoped_init
    def __init__(self, model, task="efsm", u_step=512, t_step=8192,
                 n_dummy=64, min_pad_u=256, warmup=3,
                 enable_model_fusions=True, *, config=None):
        self.model = model
        self.task = task
        self.u_step, self.t_step = u_step, t_step
        self.n_dummy, self.min_pad_u, self.warmup = n_dummy, min_pad_u, warmup
        self.enable_model_fusions = bool(enable_model_fusions)
        if config is not None and config.optimization_profile != "legacy":
            self.enable_model_fusions = config.optimization_profile != "generic"
        # Bucket graphs are mutually exclusive and replayed on one stream, so
        # their private allocations can safely reuse the same graph pool.
        self.pool = torch.cuda.graph_pool_handle()
        self.cache = {}
        self.captures = 0
        self.cache_fit_reuses = 0
        self._capture_exact_capacity = False

    @contextmanager
    def exact_capacity_capture(self):
        previous = self._capture_exact_capacity
        self._capture_exact_capacity = True
        try:
            yield
        finally:
            self._capture_exact_capacity = previous

    @scoped_call
    def _capture(self, u_cap, t_cap, sample_g):
        pg, n_real = pad_radius_graph(sample_g, u_cap, t_cap, self.n_dummy)
        workspace = _PaddedGraphWorkspace.create(
            pg=pg,
            n_real=n_real,
            u_max=u_cap,
            t_max=t_cap,
            n_dummy=self.n_dummy,
        )
        from ..model.interaction_block import (
            restore_fused_graph_optimizations,
            set_fused_graph_optimizations,
        )
        from ..model.feature_embed import (
            restore_fused_feature_graph_optimizations,
            set_fused_feature_graph_optimizations,
        )
        previous = set_fused_graph_optimizations(self.enable_model_fusions)
        previous_feature = set_fused_feature_graph_optimizations(
            self.enable_model_fusions
        )
        try:
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self.model([pg], task=self.task, is_training=False, n_real=n_real)
            torch.cuda.current_stream().wait_stream(side)
            gr = torch.cuda.CUDAGraph()
            with capture_graph(gr, pool=self.pool):
                out = self.model([pg], task=self.task, is_training=False, n_real=n_real)
        finally:
            restore_fused_feature_graph_optimizations(previous_feature)
            restore_fused_graph_optimizations(previous)
        self.cache[(u_cap, t_cap)] = dict(
            pg=pg,
            graph=gr,
            out=out,
            n_real=n_real,
            workspace=workspace,
        )
        self.captures += 1

    def _bucket(self, g: RadiusGraph):
        u, t = g.undirected2directed.shape[0], g.line_graph.shape[0]
        return _cap_for(u, t, self.u_step, self.t_step, self.min_pad_u)

    def ensure(self, sample_g: RadiusGraph, key):
        """Capture the (u_cap, t_cap) bucket if not already cached. sample_g must
        fit (its real counts <= capacity)."""
        if key not in self.cache:
            self._capture(key[0], key[1], sample_g)

    @scoped_call
    def precapture(self, sample_g: RadiusGraph, proactive: bool = True):
        """Pre-capture the bucket for sample_g and (optionally) the next-larger U/T
        neighbor buckets, without replaying — so later run() calls are pure replay.
        Larger buckets always fit sample_g (just more padding)."""
        u0, t0 = self._bucket(sample_g)
        keys = [(u0, t0)]
        if proactive:
            keys += [(u0 + self.u_step, t0),
                     (u0, t0 + self.t_step),
                     (u0 + self.u_step, t0 + self.t_step)]
        for k in keys:
            self.ensure(sample_g, k)

    @scoped_call
    def run(self, g: RadiusGraph):
        """Return (out_dict, n_real). out_dict tensors are valid until the next run().

        NB: grad must stay enabled — the model computes conservative forces via an
        internal torch.autograd.grad; the copy_ of inputs is wrapped in no_grad.
        """
        with nvtx_range("BucketedGraphRunner.run"):
            u, t = g.undirected2directed.shape[0], g.line_graph.shape[0]
            requested_key = _cap_for(
                u, t, self.u_step, self.t_step, self.min_pad_u
            )
            key = None
            if not self._capture_exact_capacity:
                key = _smallest_fitting_key(
                    self.cache, u, t, self.u_step, self.t_step
                )
            if key is None:
                key = requested_key
                if key not in self.cache:
                    self._capture(*key, g)
            elif key != requested_key:
                self.cache_fit_reuses += 1
            ent = self.cache[key]
            with torch.no_grad():
                ent["workspace"].update(g)
            with nvtx_range("cudaGraphLaunch"):
                ent["graph"].replay()
            return ent["out"], ent["n_real"]

    def stats(self) -> dict[str, object]:
        return {
            "captures": self.captures,
            "cache_fit_reuses": self.cache_fit_reuses,
            "cached_buckets": [list(key) for key in sorted(self.cache)],
            "enable_model_fusions": self.enable_model_fusions,
            "shared_graph_pool": True,
        }
