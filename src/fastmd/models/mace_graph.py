"""Capture GPU neighbor maintenance plus MACE, with checked capacity growth.

A runner owns a fixed composition and cell. ModelBackend invalidates it when
those change. No integration takes place here: retries evaluate the same input.
"""
import math

import numpy as np
import torch

from fastmd._vendor.mace_opt import tensor_batch as tb


def padded_capacity(required, step, headroom):
    return max(step, math.ceil(max(1, required) * headroom / step) * step)


class MACEGraphRunner:
    def __init__(self, backend, atoms):
        from fastmd._vendor.mace_opt.neighbors import FixedCapacityNeighbors
        self.backend = backend
        self.config = backend.config
        info = backend.info
        self.step = self.config.edge_capacity_step or 256
        self.headroom = backend.capacity_headroom
        attrs = tb.one_hot_node_attrs(atoms.numbers, info.z_table, info.dtype, info.device)
        # One host neighbor build estimates capacity; steady-state lists stay on GPU.
        edges, shifts = tb.neighbor_list_matscipy(atoms.positions, atoms.cell.array, atoms.pbc,
                                                 info.r_max + backend.neighbor_skin + .002)
        vectors = atoms.positions[edges[1]] - atoms.positions[edges[0]] + shifts @ atoms.cell.array
        active_edges = int(np.count_nonzero(np.einsum("ij,ij->i", vectors, vectors) < (info.r_max+.001)**2))
        candidate_capacity = padded_capacity(edges.shape[1], self.step, self.headroom)
        # Keep the skin's extra candidates out of the model's padded edge budget.
        self.e_cap = padded_capacity(active_edges, self.step, self.headroom)
        self.neighbors = FixedCapacityNeighbors(
            len(atoms), atoms.cell.array, info.r_max, backend.neighbor_skin,
            node_attrs=attrs, head=info.head_index, model_dtype=info.dtype,
            c_cap=candidate_capacity, e_caps=[self.e_cap], device=info.device,
        )
        self.positions = torch.as_tensor(np.array(atoms.positions), dtype=torch.float64,
                                         device=info.device).contiguous()
        self.fast_model = None
        if backend.variant != "plain":
            from fastmd._vendor.mace_opt.fast_forward import FastMace, VARIANT_FLAGS
            with tb.default_dtype(info.dtype):
                self.fast_model = FastMace(backend.model, attrs, VARIANT_FLAGS[backend.variant],
                                          head=info.head_index)
        self.cache = {}
        self.captures = 0
        self.replays = 0
        self.capacity_growths = 0
        self.last_neighbor_stats = {}

    def _evaluate(self, compute_stress):
        self.neighbors.reset_stats()
        tier = self.neighbors.step(self.positions, self.e_cap)
        with tb.default_dtype(self.backend.info.dtype):
            if self.fast_model is None:
                return tb.mace_forward(self.backend.model, tier.inputs, compute_stress)
            energy, forces, stress = self.fast_model(tier.inputs, compute_stress)
            return {"energy": energy.detach(), "forces": forces.detach(),
                    "stress": stress.detach() if stress is not None else None}

    def _check_and_grow(self):
        report = self.neighbors.host_stats()
        self.last_neighbor_stats = report
        if report["nl_err"]:
            raise RuntimeError(f"MACE GPU neighbor count/fill mismatch: {report}")
        candidate_overflow = report["cand_max"] > self.neighbors.c_cap
        edge_overflow = report["edges_max"] > self.e_cap
        if not candidate_overflow and not edge_overflow:
            return False
        # Graphs retain pointers to candidate buffers. Release before reallocating.
        self.cache.clear()
        if candidate_overflow:
            self.neighbors.set_candidate_capacity(
                padded_capacity(report["cand_max"], self.step, self.headroom))
        if edge_overflow:
            self.e_cap = padded_capacity(report["edges_max"], self.step, self.headroom)
            self.neighbors.add_tier(self.e_cap)
        self.neighbors.tiers = {self.e_cap: self.neighbors.tiers[self.e_cap]}
        # A truncated candidate list must NEVER be reused on the retry.
        self.neighbors.request_rebuild()
        self.capacity_growths += 1
        return True

    def _capture(self, compute_stress):
        # Validate capacities before warmup: both branches of the GPU rebuild
        # decision are compiled by Triton even when the first list already fits.
        for _ in range(5):
            self.neighbors.reset_stats()
            self.neighbors.step(self.positions, self.e_cap)
            if not self._check_and_grow():
                break
        else:
            raise RuntimeError("MACE neighbor capacities did not stabilize before capture")
        if len(self.cache) >= self.config.max_cached_graphs:
            self.cache.clear()
        stream = torch.cuda.Stream(device=self.backend.device)
        current = torch.cuda.current_stream(self.backend.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(self.config.warmup_steps):
                self._evaluate(compute_stress)
        current.wait_stream(stream)
        # Catch unexpected count/fill failures before recording a graph.
        if self._check_and_grow():
            return self._capture(compute_stress)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            result = self._evaluate(compute_stress)
        current.wait_stream(stream)
        self.cache[(self.e_cap, compute_stress)] = (graph, result)
        self.captures += 1

    def run(self, positions, *, compute_stress):
        self.positions.copy_(torch.as_tensor(np.array(positions), dtype=torch.float64,
                                             device=self.backend.device))
        for _ in range(5):
            key = (self.e_cap, compute_stress)
            if key not in self.cache:
                self._capture(compute_stress)
            graph, result = self.cache[(self.e_cap, compute_stress)]
            graph.replay()
            self.replays += 1
            if not self._check_and_grow():
                return result
        raise RuntimeError("MACE neighbor capacities did not stabilize; no prediction returned")

    def stats(self):
        return dict(captures=self.captures, replays=self.replays, cached_graphs=len(self.cache),
                    capacity_growths=self.capacity_growths, edge_capacity=self.e_cap,
                    candidate_capacity=self.neighbors.c_cap,
                    capture_scope="neighbors+model+forces/stress", neighbors=self.last_neighbor_stats,
                    compiled_modules=self.fast_model.compiled_module_report() if self.fast_model else {})
