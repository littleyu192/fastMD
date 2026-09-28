"""Variable-cell MatRIS: numerical checks on CPU and strict CUDA replay tests."""
import os

import numpy as np
import pytest
import torch
from ase import units
from ase.build import bulk
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fastmd import CUDAGraphConfig, FastMDCalculator


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("FASTMD_MATRIS_CHECKPOINT")
    if not path:
        pytest.skip("Set FASTMD_MATRIS_CHECKPOINT")
    pytest.importorskip("pymatgen")
    torch.set_num_threads(2)
    return path


def calculator(checkpoint, device="cpu", graph=False):
    return FastMDCalculator(
        "matris", checkpoint=checkpoint, device=device,
        cuda_graph=CUDAGraphConfig(enabled=graph, edge_capacity_step=64,
                                   triplet_capacity_step=128),
        model_kwargs={"compute_stress": True})


def structure():
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[0] += [.02, -.01, .015]
    return atoms


@pytest.mark.integration
def test_matris_efs_cache_and_strain_derivative(checkpoint):
    atoms = structure()
    atoms.calc = calculator(checkpoint)
    atoms.get_forces()
    calls = atoms.calc.stats()["calls"]
    stress = atoms.get_stress()
    atoms.get_potential_energy()
    assert atoms.calc.stats()["calls"] == calls
    assert atoms.calc.backend.calculator.task == "efs"
    # Symmetric shear convention: half of the engineering shear on each side.
    cell = atoms.cell.array.copy()
    scaled = atoms.get_scaled_positions(wrap=False)
    volume = atoms.get_volume()
    # Use the local derivative regime; convergence was checked at h=2e-4 too.
    h = 5e-4
    for component, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))):
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = 1 if i == j else .5
        energies = []
        for sign in (1, -1):
            atoms.set_cell(cell @ (np.eye(3) + sign * h * strain))
            atoms.set_scaled_positions(scaled)
            energies.append(atoms.get_potential_energy())
        derivative = (energies[0] - energies[1]) / (2 * h * volume)
        np.testing.assert_allclose(derivative, stress[component], atol=3e-4, rtol=.04)
    assert atoms.calc.stats()["invalidations"] == 0


@pytest.mark.integration
def test_matris_workspace_updates_cell_and_topology(checkpoint):
    from pymatgen.io.ase import AseAtomsAdaptor
    from fastmd._vendor.matris.applications.cuda_graph import pad_radius_graph, _PaddedGraphWorkspace

    backend = calculator(checkpoint).backend
    frames = []
    for deformation in (np.eye(3), np.eye(3) * .88, np.eye(3) * 1.08,
                        np.array([[1., .12, 0], [0, .96, .05], [0, 0, 1.04]])):
        atoms = structure()
        atoms.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
        frames.append(atoms)

    def build(atoms):
        return backend.model.graph_converter(AseAtomsAdaptor.get_structure(atoms))

    graphs = [build(atoms) for atoms in frames]
    counts = [(len(g.undirected2directed), len(g.line_graph)) for g in graphs]
    assert len(set(counts)) > 1  # Compression/expansion actually changes topology.
    u_cap = max(u for u, _ in counts) + 64
    t_cap = max(t for _, t in counts) + 64
    padded, n = pad_radius_graph(build(frames[0]), u_cap, t_cap)
    workspace = _PaddedGraphWorkspace.create(padded, n, u_cap, t_cap, 64)
    pointers = {key: getattr(padded, key).data_ptr() for key in
                ("lattice", "atom_frac_coord", "atom_graph", "neighbor_image", "line_graph")}
    with backend._graph_config.scope():
        for graph in graphs:
            expected = backend.model([graph], task="efs", is_training=False)
            with torch.no_grad():
                workspace.update(graph)
            actual = backend.model([padded], task="efs", is_training=False, n_real=n)
            torch.testing.assert_close(actual["e"], expected["e"], atol=1e-4, rtol=1e-5)
            torch.testing.assert_close(actual["f"][0][:n], expected["f"][0], atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(actual["s"][0], expected["s"][0], atol=1e-3, rtol=1e-3)
            assert all(getattr(padded, key).data_ptr() == ptr for key, ptr in pointers.items())


@pytest.mark.integration
@pytest.mark.parametrize("graph", [False, pytest.param(True, marks=[pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")])])
def test_matris_ase_npt(checkpoint, graph):
    device = "cuda" if graph else "cpu"
    atoms = structure()
    atoms.calc = calculator(checkpoint, device, graph)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))
    initial_cell = atoms.cell.array.copy()
    atoms.calc.warmup(atoms)
    runner = atoms.calc.backend.runner
    reference = calculator(checkpoint, device) if graph else None
    with NPTBerendsen(atoms, timestep=.5 * units.fs, temperature_K=300,
                      pressure_au=1 * units.GPa, taut=100 * units.fs,
                      taup=1000 * units.fs, compressibility_au=1 / (100 * units.GPa)) as dyn:
        for _ in range(4):
            dyn.run(1)
            calls = atoms.calc.stats()["calls"]
            assert np.isfinite(atoms.get_stress()).all()
            assert atoms.calc.stats()["calls"] == calls
            if graph:
                reference.calculate(atoms, ["energy", "forces", "stress"])
                for key in ("energy", "forces", "stress"):
                    np.testing.assert_allclose(atoms.calc.results[key], reference.results[key],
                                               atol=3e-3 if key != "stress" else 3e-5, rtol=1e-3)
                assert atoms.calc.backend.runner is runner
    assert np.max(np.abs(atoms.cell.array - initial_cell)) > 1e-8
    assert atoms.calc.stats()["invalidations"] == 0
    assert atoms.calc.stats()["mode"] == ("cuda_graph" if graph else "eager")


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_matris_variable_cell_replay_and_capacity_growth(checkpoint):
    graph = calculator(checkpoint, "cuda", True)
    eager = calculator(checkpoint, "cuda")
    atoms = structure()
    atoms.calc = graph
    graph.warmup(atoms)
    runner = graph.backend.runner
    captures = runner.captures
    saved_forces = atoms.get_forces()
    saved_copy = saved_forces.copy()
    # Small isotropic/anisotropic/shear changes must reuse the first capture.
    for deformation in (np.eye(3) * 1.0001, np.diag([1.0001, .9999, 1.]),
                        np.array([[1., .0001, 0], [0, 1., 0], [0, 0, 1.]]),
                        np.eye(3) * .80, np.eye(3) * 1.08):
        frame = structure()
        frame.set_cell(frame.cell.array @ deformation, scale_atoms=True)
        frame.positions[0] += frame.cell.array[0]  # Periodic boundary crossing.
        frame.calc = graph
        frame.get_forces()  # Must also produce stress without a second call.
        calls = graph.stats()["calls"]
        frame.get_stress()
        assert graph.stats()["calls"] == calls
        eager.calculate(frame, ["energy", "forces", "stress"])
        for key in ("energy", "forces", "stress"):
            np.testing.assert_allclose(graph.results[key], eager.results[key],
                                       atol=3e-3 if key != "stress" else 3e-5, rtol=1e-3)
        assert graph.backend.runner is runner
        if np.max(np.abs(deformation - np.eye(3))) < .001:
            assert runner.captures == captures
    assert runner.captures > captures  # Larger topology needs another capacity bucket.
    assert graph.stats()["invalidations"] == 0
    assert graph.stats()["eager_calls"] == 0
    np.testing.assert_array_equal(saved_forces, saved_copy)
