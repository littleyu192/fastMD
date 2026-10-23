"""Subprocess worker: whole-step graphs whose real edge count hits e_capacity.

Run by ``test_fixed_capacity_boundary.py`` in a child process, because the
pre-fix failure mode is a CUDA device-side assert that poisons the CUDA
context of the whole process.  Covers the initial evaluation, the replay
rollback (single and batched runner) and the GPU MD grow paths.  Prints one
JSON line and exits 0 on success.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fastmd.interfaces.gpu_md as gpu_md_module  # noqa: E402
from fastmd.engine.capacity_policy import (  # noqa: E402
    CapacityKey,
    JointCapacityPolicy,
)
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics  # noqa: E402
from fastmd.execution.gpu_integrator import GPUIntegrator  # noqa: E402
from fastmd.execution.whole_step_cuda_graph import (  # noqa: E402
    BatchedWholeStepGraphRunner,
    FixedCapacityOverflow,
    WholeStepGraphRunner,
)
from fastmd.config import CapacityConfig, InferenceConfig  # noqa: E402
from fastmd.graph import FixedCapacityGraphBuilder, atoms_to_graph_gpu  # noqa: E402

from test_isolated_atom_handling import make_model  # noqa: E402

CUTOFF, LINE_CUTOFF = 2.0, 1.5  # make_model cutoffs
BASE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5]]
# atom 3 moves within CUTOFF of atom 0 only (1.8 A, not a line-graph edge),
# so the undirected edge count grows by exactly one and triplets do not change
DENSE = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.8], [0, 0, 2.8]]


def atoms_of(positions) -> Atoms:
    return Atoms("H5", positions=positions, cell=[20.0] * 3, pbc=True)


def counts(atoms: Atoms) -> tuple[int, int]:
    graph = atoms_to_graph_gpu(
        atoms, atom_graph_cutoff=CUTOFF, line_graph_cutoff=LINE_CUTOFF,
        check_isolated_atoms=False,
    )
    return int(graph.undirected2directed.shape[0]), int(graph.line_graph.shape[0])


def as_cuda(positions) -> torch.Tensor:
    return torch.tensor(positions, dtype=torch.float32, device="cuda")


def make_builder(u_capacity: int, t_capacity: int) -> FixedCapacityGraphBuilder:
    return FixedCapacityGraphBuilder(
        cell=torch.eye(3) * 20.0,
        atomic_numbers=torch.ones(5, dtype=torch.int32),
        u_capacity=u_capacity,
        t_capacity=t_capacity,
        atom_graph_cutoff=CUTOFF,
        line_graph_cutoff=LINE_CUTOFF,
        candidate_cutoff=CUTOFF + 0.5,
        n_dummy=4,
        device="cuda",
    )


def make_integrator(dt_fs: float = 0.0) -> GPUIntegrator:
    return GPUIntegrator(
        torch.ones(5, device="cuda"), dt_fs=dt_fs, ensemble="nve", fix_com=False
    )


def reference(model, positions) -> dict:
    return model(
        [atoms_to_graph_gpu(atoms_of(positions), atom_graph_cutoff=CUTOFF,
                            line_graph_cutoff=LINE_CUTOFF)],
        task="ef", is_training=False,
    )


def expect_overflow(make, what: str) -> FixedCapacityOverflow:
    try:
        make()
    except FixedCapacityOverflow as overflow:
        return overflow
    raise AssertionError(f"{what}: edges == e_capacity was accepted")


def initial_evaluation(model, u_base: int, u_dense: int, t: int, result: dict) -> None:
    """A runner whose initial state is exactly at the boundary must not run on
    the substitute (all-sink) topology with silently wrong forces/energy."""
    overflow = expect_overflow(
        lambda: WholeStepGraphRunner(
            model=model, builder=make_builder(u_dense, t + 32),
            integrator=make_integrator(), positions=as_cuda(DENSE),
            momenta=torch.zeros(5, 3, device="cuda"), warmup=1,
        ),
        "single runner initial evaluation",
    )
    assert overflow.edge_count == 2 * u_dense, (overflow.edge_count, u_dense)
    positions = torch.stack([as_cuda(BASE), as_cuda(DENSE)])
    overflow = expect_overflow(
        lambda: BatchedWholeStepGraphRunner(
            model=model,
            builders=[make_builder(u_base + 1, t + 32), make_builder(u_dense, t + 32)],
            integrator=make_integrator(), positions=positions,
            momenta=torch.zeros_like(positions), warmup=1,
        ),
        "batched runner initial evaluation",
    )
    assert overflow.edge_count == 2 * u_dense, (overflow.edge_count, u_dense)
    result["initial_overflow_edges"] = overflow.edge_count


def replay_rollback(model, u_base: int, t: int, result: dict) -> None:
    """An overflowing window restores positions, momenta, forces and energy
    bitwise (single runner, 1 and 3 steps per launch)."""
    momenta = torch.full((5, 3), 1e-3, device="cuda")
    momenta[1] *= -1
    for steps_per_launch in (1, 3):
        runner = WholeStepGraphRunner(
            model=model, builder=make_builder(u_base + 1, t + 32),
            integrator=make_integrator(0.5), positions=as_cuda(BASE),
            momenta=torch.zeros(5, 3, device="cuda"), warmup=1,
            steps_per_launch=steps_per_launch,
        )
        runner.replay(strict=True)
        runner.reset(
            positions=as_cuda(DENSE), momenta=momenta,
            forces=torch.linspace(-1e-3, 1e-3, 15, device="cuda").reshape(5, 3),
            potential_energy=torch.tensor([1.25], device="cuda"),
        )
        before = [tensor.clone() for tensor in runner.state.__dict__.values()]
        overflow = expect_overflow(
            lambda runner=runner: runner.replay(strict=True), "rollback replay"
        )
        assert overflow.edge_count == runner.builder.e_capacity
        after = list(runner.state.__dict__.values())
        assert all(torch.equal(a, b) for a, b in zip(before, after)), steps_per_launch
    result["rollback_steps_per_launch"] = [1, 3]


def batched_replay(model, u_base: int, t: int, result: dict) -> None:
    """One replica reaching the boundary rolls the whole batch back."""
    positions = torch.stack([as_cuda(BASE), as_cuda(BASE)])
    runner = BatchedWholeStepGraphRunner(
        model=model,
        builders=[make_builder(u_base + 1, t + 32), make_builder(u_base + 1, t + 32)],
        integrator=make_integrator(), positions=positions,
        momenta=torch.zeros_like(positions), warmup=1,
    )
    runner.replay(strict=True)
    runner.positions[1].copy_(as_cuda(DENSE))
    runner.candidate_refresher.refresh(runner.positions)
    before = [tensor.clone() for tensor in runner.state.__dict__.values()]
    overflow = expect_overflow(lambda: runner.replay(strict=True), "batched replay")
    assert overflow.edge_count == runner.builders[1].e_capacity, overflow.edge_count
    after = list(runner.state.__dict__.values())
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    result["batched_overflow_edges"] = overflow.edge_count


class BoundaryStartPolicy(JointCapacityPolicy):
    """Sizes the first capture exactly at the real edge count (no padding
    pair), as a fixed/dynamic builder count mismatch at the cutoff would."""

    def start(self, u: int, t: int) -> CapacityKey:
        key = CapacityKey(int(u), self.key_for(u, t).t_capacity)
        self.capacity_history.append(key)
        return key


def md_initial_grow(model, u_dense: int, capacity: CapacityConfig, result: dict) -> None:
    gpu_md_module.JointCapacityPolicy = BoundaryStartPolicy
    try:
        md = GPUResidentMolecularDynamics(
            atoms_of(DENSE), model=model, task="ef", ensemble="nve", timestep=0.0,
            config=InferenceConfig(execution="whole_step", capacity=capacity, report=False),
        )
    finally:
        gpu_md_module.JointCapacityPolicy = JointCapacityPolicy
    try:
        expected = reference(model, DENSE)
        torch.testing.assert_close(
            md.state.potential_energy.reshape(-1)[0], expected["e"][0] * 5,
            rtol=2e-4, atol=2e-5,
        )
        torch.testing.assert_close(md.state.forces, expected["f"][0], rtol=2e-4, atol=2e-5)
        stats = md.whole_step_stats()
        assert stats["capacity_history"] == [(u_dense, 64), (u_dense + 1, 64)], stats
        assert md.whole_step_runner.builder.u_capacity == u_dense + 1
        md.run(1)
        result["md_initial_capacity_history"] = stats["capacity_history"]
    finally:
        md.close()


def main() -> None:
    torch.manual_seed(0)
    model = make_model(num_layers=1).cuda()
    u_base, t_base = counts(atoms_of(BASE))
    u_dense, t_dense = counts(atoms_of(DENSE))
    assert u_dense == u_base + 1 and t_dense == t_base, (u_base, u_dense, t_base, t_dense)
    result = {"u_base": u_base, "u_dense": u_dense, "t": t_base}

    # 1) raw runner: one padding pair at capture, none after the move
    builder = make_builder(u_base + 1, t_base + 32)  # leaves padding triplets
    positions = as_cuda(BASE)
    runner = WholeStepGraphRunner(
        model=model,
        builder=builder,
        integrator=make_integrator(),
        positions=positions,
        momenta=torch.zeros_like(positions),
        warmup=1,
    )
    runner.positions.copy_(as_cuda(DENSE))
    runner.builder.refresh_candidates(runner.positions)
    try:
        runner.replay(strict=True)
    except FixedCapacityOverflow as overflow:
        result["runner_overflow_edges"] = overflow.edge_count
    else:
        raise AssertionError(
            "edges == e_capacity was accepted: "
            f"overflow={bool(runner.window_overflow.item())} "
            f"edges={int(builder.edge_offsets[-1].item())} e_capacity={builder.e_capacity} "
            f"max line u={int(builder.line_graph[:, [1, 3]].max().item())} "
            f"u_capacity={builder.u_capacity}"
        )
    assert result["runner_overflow_edges"] == builder.e_capacity, result
    torch.testing.assert_close(runner.positions, as_cuda(DENSE), rtol=0, atol=0)

    initial_evaluation(model, u_base, u_dense, t_base, result)
    replay_rollback(model, u_base, t_base, result)
    batched_replay(model, u_base, t_base, result)

    # 2) GPU MD: key_for(u_base) = u_base + 1 with u_step=1/min_pad_u=1, then
    # the move must roll back, grow to u_dense + 1 and recapture
    capacity = CapacityConfig(
        u_step=1, t_step=64, min_pad_u=1, min_pad_t=8, n_dummy=4,
        warmup=1, candidate_skin=0.5,
    )
    md = GPUResidentMolecularDynamics(
        atoms_of(BASE),
        model=model,
        task="ef",
        ensemble="nve",
        timestep=0.0,
        config=InferenceConfig(execution="whole_step", capacity=capacity, report=False),
    )
    try:
        md.run(1)
        start = md.whole_step_stats()["capacity_history"]
        assert start == [(u_base + 1, 64)], start
        md.whole_step_runner.positions.copy_(as_cuda(DENSE))
        md.whole_step_runner.builder.refresh_candidates(md.whole_step_runner.positions)
        state = md.run(1)
        stats = md.whole_step_stats()
        expected = reference(model, DENSE)
        torch.testing.assert_close(
            state.potential_energy.reshape(-1)[0], expected["e"][0] * 5,
            rtol=2e-4, atol=2e-5,
        )
        torch.testing.assert_close(state.forces, expected["f"][0], rtol=2e-4, atol=2e-5)
        assert stats["capacity_history"] == [(u_base + 1, 64), (u_dense + 1, 64)], stats
        assert stats["capacity_recaptures"] == 1, stats
        assert md.whole_step_runner.builder.u_capacity == u_dense + 1
        result["md_capacity_history"] = stats["capacity_history"]
        result["md_max_edge_count"] = stats["max_edge_count"]
    finally:
        md.close()
    # 3) GPU MD whose first capacity has no padding pair at the initial state
    md_initial_grow(model, u_dense, capacity, result)
    torch.cuda.synchronize()
    result["ok"] = True
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
