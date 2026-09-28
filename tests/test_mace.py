"""MACE reference parity and opt-in CUDA replay/overflow coverage."""
import os

import numpy as np
import pytest
import torch
from ase.build import bulk

from fastmd import FastMDCalculator


def checkpoint_for(kind):
    path = os.environ.get(f"FASTMD_MACE_{kind.upper()}_CHECKPOINT")
    if kind == "mpa0":
        path = path or os.environ.get("FASTMD_MACE_CHECKPOINT")
    if not path:
        pytest.skip(f"Set FASTMD_MACE_{kind.upper()}_CHECKPOINT to a local checkpoint")
    pytest.importorskip("mace")
    return path


@pytest.fixture(params=["mp0", "mpa0"])
def mace_cpu(request):
    torch.set_num_threads(2)
    dtype = torch.get_default_dtype()
    calc = FastMDCalculator("mace", checkpoint=checkpoint_for(request.param), device="cpu", cuda_graph=False)
    assert torch.get_default_dtype() == dtype
    return calc


@pytest.mark.integration
def test_mace_matches_official_and_derivatives(mace_cpu):
    from mace.calculators import MACECalculator
    from fastmd._vendor.mace_opt.tensor_batch import default_dtype
    calc = mace_cpu
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions[0] += [.02, -.01, .015]
    calc.calculate(atoms, ["energy", "forces", "stress"])
    result = {k: np.array(v, copy=True) for k, v in calc.results.items()}
    # Upstream calculator mutates the process default dtype; contain it in the test.
    with default_dtype(torch.float64):
        reference = MACECalculator(model_paths=calc.backend.info.path, device="cpu", default_dtype="float64")
        reference.calculate(atoms, ["energy", "forces", "stress"])
    for key in ("energy", "forces", "stress"):
        np.testing.assert_allclose(result[key], reference.results[key], rtol=1e-9, atol=1e-9)
    h = 1e-4
    atoms.calc = calc
    atoms.positions[0, 0] += h
    plus = atoms.get_potential_energy()
    atoms.positions[0, 0] -= 2 * h
    minus = atoms.get_potential_energy()
    np.testing.assert_allclose(result["forces"][0, 0], -(plus-minus)/(2*h), rtol=2e-4, atol=1e-5)
    atoms.positions[0, 0] += h
    cell = atoms.cell.copy()
    volume = atoms.get_volume()
    strain = np.eye(3)
    strain[0, 0] += h
    atoms.set_cell(cell @ strain, scale_atoms=True)
    plus = atoms.get_potential_energy()
    strain[0, 0] -= 2*h
    atoms.set_cell(cell @ strain, scale_atoms=True)
    minus = atoms.get_potential_energy()
    np.testing.assert_allclose(result["stress"][0], (plus-minus)/(2*h*volume), rtol=2e-4, atol=1e-6)
    assert calc.stats()["mode"] == "eager"


@pytest.mark.integration
def test_mace_padding_and_unfused_fast_forward(mace_cpu):
    pytest.importorskip("triton")
    from fastmd._vendor.mace_opt import tensor_batch as tb
    from fastmd._vendor.mace_opt.fast_forward import FastMace, FastFlags
    backend = mace_cpu.backend
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions[0] += [.02, -.01, .015]
    inputs = tb.inputs_from_atoms(atoms, backend.info)
    with tb.default_dtype(backend.info.dtype):
        reference = tb.mace_forward(backend.model, inputs)
        edges, shifts = tb.pad_edges_self_loop(inputs["edge_index"], inputs["unit_shifts"],
                                               inputs["edge_index"].shape[1]+73, inputs["cell"], backend.info.r_max)
        padded = tb.build_input_dict(inputs["positions"], inputs["cell"], edges, shifts, inputs["node_attrs"])
        result = tb.mace_forward(backend.model, padded)
        fm = FastMace(backend.model, inputs["node_attrs"], FastFlags.off())
        energy, forces, stress = fm(padded, True)
    for key in reference:
        torch.testing.assert_close(result[key], reference[key], rtol=1e-9, atol=1e-9)
    for actual, expected in ((energy, reference["energy"].reshape(())),
                             (forces, reference["forces"]), (stress, reference["stress"].reshape(3, 3))):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


def test_missing_mace_checkpoint_does_not_download(tmp_path):
    from fastmd._vendor.mace_opt.tensor_batch import resolve_model_path
    with pytest.raises(FileNotFoundError):
        resolve_model_path(str(tmp_path / "missing.model"))


def test_dtype_scope_restores_on_error():
    from fastmd._vendor.mace_opt.tensor_batch import default_dtype
    previous = torch.get_default_dtype()
    with pytest.raises(RuntimeError), default_dtype(torch.float64):
        raise RuntimeError("failed load")
    assert torch.get_default_dtype() == previous


@pytest.mark.parametrize("cand,edges,overflow", [(64, 64, False), (65, 64, True), (64, 65, True), (65, 65, True)])
def test_overflow_invalidates_all_captures_and_forces_rebuild(cand, edges, overflow):
    from fastmd.models.mace_graph import MACEGraphRunner
    class Neighbors:
        c_cap = 64
        tiers = {64: object()}
        rebuilt = False
        def host_stats(self):
            return dict(cand_max=cand, edges_max=edges, nl_err=0)
        def set_candidate_capacity(self, capacity):
            self.c_cap = capacity
        def add_tier(self, capacity):
            self.tiers[capacity] = object()
        def request_rebuild(self):
            self.rebuilt = True
    runner = MACEGraphRunner.__new__(MACEGraphRunner)
    runner.neighbors = Neighbors()
    runner.cache = {(64, False): object(), (64, True): object()}
    runner.e_cap, runner.step, runner.headroom, runner.capacity_growths = 64, 64, 1.25, 0
    assert runner._check_and_grow() is overflow
    assert runner.neighbors.rebuilt is overflow
    assert len(runner.cache) == (0 if overflow else 2)
    assert runner.neighbors.c_cap >= cand
    assert runner.e_cap >= edges


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("kind", ["mp0", "mpa0"])
@pytest.mark.parametrize("variant", ["plain", "fast", "fast_cm"])
def test_mace_cuda_replay_and_invalidation(kind, variant):
    path = checkpoint_for(kind)
    eager = FastMDCalculator("mace", checkpoint=path, device="cuda", cuda_graph=False)
    graph = FastMDCalculator("mace", checkpoint=path, device="cuda", cuda_graph=True,
                             model_kwargs={"variant": variant})
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    rng = np.random.default_rng(12)
    saved = None
    for step in range(7):
        atoms.positions += rng.normal(0, .02, atoms.positions.shape)
        if step == 2:
            # Genuine runtime overflow: same cell/composition, reallocated tiny list.
            runner = graph.backend.runner
            runner.cache.clear()
            runner.e_cap = 1
            runner.neighbors.add_tier(1)
            runner.neighbors.set_candidate_capacity(1)
        if step == 3:
            atoms.positions[0] += atoms.cell[0]  # cross a periodic boundary
        if step == 4:
            cell = atoms.cell.array.copy()
            cell[1, 0] += .3
            atoms.set_cell(cell, scale_atoms=True)
        if step == 5:
            atoms.numbers[0] = 6
        if step == 6:
            atoms = atoms.repeat((2, 1, 1))
        for calc in (eager, graph):
            calc.calculate(atoms, ["energy", "forces", "stress"])
        for key in ("energy", "forces", "stress"):
            np.testing.assert_allclose(graph.results[key], eager.results[key], rtol=2e-5, atol=2e-6)
        if step == 0:
            saved = graph.results["forces"]
            copy = saved.copy()
        if step == 2:
            assert graph.stats()["cache"]["capacity_growths"] > 0
    np.testing.assert_array_equal(saved, copy)
    assert graph.stats()["mode"] == "cuda_graph"
    assert graph.stats()["invalidations"] == 3
    assert graph.stats()["graph_calls"] == 7


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("kind", ["mp0", "mpa0"])
def test_mace_overflow_after_successful_replay(kind):
    """Move isolated atoms into a cluster without changing the capture signature."""
    from ase import Atoms
    path = checkpoint_for(kind)
    eager = FastMDCalculator("mace", checkpoint=path, device="cuda", cuda_graph=False)
    graph = FastMDCalculator("mace", checkpoint=path, device="cuda", cuda_graph=True)
    grid = np.indices((2, 2, 2)).reshape(3, -1).T.astype(float)
    atoms = Atoms("Si8", positions=grid*12, cell=[40, 40, 40], pbc=True)
    graph.calculate(atoms)
    runner = graph.backend.runner
    runner.cache.clear()
    runner.e_cap = 1
    runner.neighbors.add_tier(1)
    runner.neighbors.set_candidate_capacity(1)
    graph.calculate(atoms)  # Successful capture and replay with a valid one-slot list.
    assert (1, False) in runner.cache
    assert runner.capacity_growths == 0
    replays = runner.replays
    atoms.positions = grid*1.7 + np.random.default_rng(8).normal(0, .03, (8, 3))
    eager.calculate(atoms)
    graph.calculate(atoms)  # Overflow must be detected AFTER replay, then recovered.
    assert graph.backend.runner is runner
    assert runner.capacity_growths > 0
    assert runner.replays >= replays + 2
    assert (1, False) not in runner.cache
    for key in ("energy", "forces"):
        np.testing.assert_allclose(graph.results[key], eager.results[key], rtol=2e-5, atol=2e-6)
