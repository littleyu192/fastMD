"""Variable-cell neighbor coverage, stress derivatives and ASE NPT parity."""
import numpy as np
import pytest
import torch
from ase import units
from ase.build import bulk
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from fastmd import FastMDCalculator
from test_mace import checkpoint_for


def test_cell_buffers_stay_live_and_image_range_only_grows():
    pytest.importorskip("triton")
    from fastmd._vendor.mace_opt.neighbors import FixedCapacityNeighbors
    neighbors = FixedCapacityNeighbors(1, np.eye(3)*12, 5., 1.,
        node_attrs=torch.ones(1, 1, dtype=torch.float64), device="cpu", c_cap=64, e_caps=[64])
    pointers = [t.data_ptr() for t in (neighbors.prm32, neighbors.prm_m, neighbors.cell_m)]
    original = neighbors.cell_np.copy()
    small = np.eye(3)*4
    small[1, 0] = .4
    assert neighbors.update_cell(small)
    bounds = neighbors.n_img.copy()
    small[0, 0] = 9  # The internal cell must not alias the caller's array.
    assert neighbors.cell_np[0, 0] == 4
    assert neighbors.update_cell(original) is False
    assert neighbors.n_img == bounds
    assert pointers == [t.data_ptr() for t in (neighbors.prm32, neighbors.prm_m, neighbors.cell_m)]
    torch.testing.assert_close(neighbors.tiers[64].inputs['cell'], torch.eye(3, dtype=torch.float64)*12)
    assert neighbors.ctrl[1] == 1
    with pytest.raises(ValueError, match="nonsingular"):
        neighbors.update_cell(np.zeros((3, 3)))


def calculator(kind, graph, variant="fast", dtype="float64"):
    return FastMDCalculator("mace", checkpoint=checkpoint_for(kind), device="cuda",
        cuda_graph=graph, model_kwargs=dict(variant=variant, default_dtype=dtype, compute_stress=True))


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("kind", ["mp0", "mpa0"])
@pytest.mark.parametrize("variant", ["plain", "fast", "fast_cm"])
def test_variable_cell_efs_and_image_growth(kind, variant):
    eager, graph = calculator(kind, False), calculator(kind, True, variant)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions += np.random.default_rng(42).normal(0, .02, atoms.positions.shape)
    base = atoms.copy()
    runner = None
    for scale, shear in [(1, 0), (.99, .03), (1.03, -.04), (.52, .02), (1., 0)]:
        atoms = base.copy()
        deformation = np.eye(3)*scale
        deformation[1, 0] = shear
        atoms.set_cell(base.cell.array @ deformation, scale_atoms=True)
        atoms.positions[0] += atoms.cell[0]  # Unwrapped periodic coordinate.
        for calc in (eager, graph):
            atoms.calc = calc
            calc.reset()
            atoms.get_forces()  # Stress must already be in the ASE result cache.
            calls = calc.stats()["calls"]
            atoms.get_stress()
            assert calc.stats()["calls"] == calls
        for key in ("energy", "forces", "stress"):
            np.testing.assert_allclose(graph.results[key], eager.results[key], rtol=2e-5, atol=2e-6)
        if runner is None:
            runner = graph.backend.runner
            initial_captures = runner.captures
        else:
            assert graph.backend.runner is runner
        if scale in (.99, 1.03):
            assert runner.captures == initial_captures
    assert runner.image_range_growths >= 1
    assert graph.stats()["invalidations"] == 0
    assert graph.stats()["eager_calls"] == 0


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("kind", ["mp0", "mpa0"])
def test_fused_stress_finite_strain(kind):
    calc = calculator(kind, True)
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    cell = atoms.cell.array.copy()
    cell[1, 0] += .2
    atoms.set_cell(cell, scale_atoms=True)
    atoms.positions[0] += [.04, -.02, .03]
    atoms.calc = calc
    stress = atoms.get_stress(voigt=False).copy()
    volume = atoms.get_volume()
    initial = atoms.copy()
    h = 1e-5
    for i, j in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        energies = []
        for sign in (1, -1):
            deformed = initial.copy()
            strain = np.eye(3)
            strain[i, j] += sign*h
            deformed.set_cell(cell @ strain, scale_atoms=True)
            deformed.calc = calc
            energies.append(deformed.get_potential_energy())
        np.testing.assert_allclose((energies[0]-energies[1])/(2*h*volume),
                                   stress[i, j], rtol=3e-4, atol=2e-6)


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("kind", ["mp0", "mpa0"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_npt_trajectory_matches_eager(kind, dtype):
    initial = bulk("Si", "diamond", a=5.43, cubic=True)
    initial.positions += np.random.default_rng(2026).normal(0, .01, initial.positions.shape)
    MaxwellBoltzmannDistribution(initial, temperature_K=300, rng=np.random.default_rng(42))
    Stationary(initial)
    trajectories = []
    for use_graph in (False, True):
        calc = calculator(kind, use_graph, dtype=dtype)
        atoms = initial.copy()
        atoms.calc = calc
        calc.warmup(atoms)
        before = calc.stats()["cache"].get("captures", 0)
        frames = []
        with NPTBerendsen(atoms, timestep=.5*units.fs, temperature_K=300,
            pressure_au=units.GPa, compressibility_au=1/(100*units.GPa),
            taut=100*units.fs, taup=1000*units.fs) as dyn:
            for _ in range(20):
                dyn.run(1)
                frames.append(np.concatenate([atoms.positions.ravel(), atoms.cell.array.ravel(),
                                               atoms.get_momenta().ravel()]))
        trajectories.append(frames)
        assert calc.stats()["calls"] == 41
        assert np.max(np.abs(atoms.cell.array-initial.cell.array)) > 1e-7
        if use_graph:
            assert calc.stats()["cache"]["captures"] == before
            assert calc.stats()["invalidations"] == calc.stats()["eager_calls"] == 0
        calc.clear_cache()
    np.testing.assert_allclose(*trajectories, rtol=1e-5, atol=2e-4 if dtype=="float32" else 2e-6)
