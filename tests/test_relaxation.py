"""Relaxation semantics: cell convergence, independent structures and dispatch."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from fastmd import FastMDCalculator, CUDAGraphConfig
from fastmd.models.base import ModelBackend
from fastmd.models.chgnet import CHGNetModel
from fastmd.relaxation import RelaxationConfig, make_relaxation_calculator, relax


class Quadratic(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": float((self.atoms.positions ** 2).sum() / 2),
                        "forces": -self.atoms.positions.copy(), "stress": np.zeros(6)}


def test_fixed_cell_relaxation_preserves_input_and_owns_results():
    atoms = Atoms("H2", positions=[[0.4, 0, 0], [0, .6, 0]], cell=[5, 5, 5], pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    original_calc = EMT()
    atoms.calc = original_calc
    before = atoms.copy()
    calculator = Quadratic()
    result = relax(atoms, calculator, RelaxationConfig(relax_cell=False, fmax=1e-4))
    assert result.converged and result.steps > 0 and result.elapsed_seconds > 0
    assert result.final_energy < result.initial_energy
    assert result.max_force < 1e-4 and result.max_filter_force == result.max_force
    assert result.stress is None
    np.testing.assert_array_equal(atoms.positions, before.positions)
    np.testing.assert_array_equal(atoms.cell, before.cell)
    assert atoms.calc is original_calc
    np.testing.assert_array_equal(result.atoms.positions[0], before.positions[0])
    np.testing.assert_allclose(result.atoms.get_forces(), result.forces)
    saved = result.atoms.get_forces().copy()
    other = Atoms("H", positions=[[1, 2, 3]], cell=[4, 4, 4], pbc=True)
    relax(other, calculator, RelaxationConfig(relax_cell=False, steps=1))
    np.testing.assert_array_equal(result.atoms.get_forces(), saved)
    result.forces[:] = 99
    np.testing.assert_array_equal(result.atoms.get_forces(), saved)


def test_variable_cell_matches_direct_ase_and_converges_cell():
    initial = bulk("Cu", "fcc", a=3.8, cubic=True)
    initial.positions[0] += [.02, -.01, .015]
    config = RelaxationConfig(fmax=.01, steps=150)
    result = relax(initial, EMT(), config)
    reference = initial.copy()
    reference.calc = EMT()
    with FIRE(FrechetCellFilter(reference), logfile=None) as optimizer:
        optimizer.run(fmax=config.fmax, steps=config.steps)
        steps = optimizer.nsteps
    assert result.converged and result.steps == steps
    assert result.final_energy < result.initial_energy
    assert not np.array_equal(result.atoms.cell, initial.cell)
    np.testing.assert_allclose(result.atoms.cell, reference.cell, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.atoms.positions, reference.positions, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.stress, reference.get_stress(), rtol=0, atol=1e-12)
    assert result.max_filter_force < config.fmax


def test_cell_convergence_is_not_atomic_force_convergence():
    atoms = bulk("Cu", "fcc", a=4.0, cubic=True)
    result = relax(atoms, EMT(), RelaxationConfig(steps=0))
    assert result.steps == 0 and not result.converged
    assert result.max_force < 1e-10
    assert result.max_filter_force > .05
    np.testing.assert_array_equal(result.atoms.cell, atoms.cell)


def test_zero_steps_initial_convergence_and_fresh_optimizers():
    atoms = Atoms("H", positions=[[.2, 0, 0]], cell=[4, 4, 4], pbc=True)
    calc = Quadratic()
    config = RelaxationConfig(relax_cell=False, steps=2, fmax=1e-8)
    first, second = relax(atoms, calc, config), relax(atoms, calc, config)
    np.testing.assert_array_equal(first.atoms.positions, second.atoms.positions)
    assert first.steps == second.steps == 2 and not first.converged
    atoms.positions[:] = 0
    result = relax(atoms, calc, RelaxationConfig(relax_cell=False, steps=0))
    assert result.converged and result.steps == 0


def test_bfgs_and_unsupported_cell_calculator():
    atoms = Atoms("H", positions=[[.2, 0, 0]], cell=[4, 4, 4], pbc=True)
    assert relax(atoms, Quadratic(), RelaxationConfig(relax_cell=False, optimizer="BFGS")).converged
    calc = Quadratic()
    calc.implemented_properties = ["energy", "forces"]
    with pytest.raises(PropertyNotImplementedError, match="stress"):
        relax(atoms, calc)


@pytest.mark.parametrize("field", ["energy", "forces", "stress"])
def test_nonfinite_predictions_fail(field):
    class Broken(Quadratic):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results[field] = np.asarray(self.results[field]) * np.nan
    atoms = Atoms("H", positions=[[.2, 0, 0]], cell=[4, 4, 4], pbc=True)
    with pytest.raises(ValueError, match="non-finite"):
        relax(atoms, Broken())


@pytest.mark.parametrize("kwargs", [
    {"fmax": 0}, {"fmax": float("nan")}, {"steps": -1}, {"steps": True},
    {"optimizer": "unknown"}, {"relax_cell": "yes"},
    {"optimizer_kwargs": {"restart": "old.json"}},
    {"relax_cell": False, "filter_kwargs": {"mask": [1] * 6}},
])
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        RelaxationConfig(**kwargs)


@pytest.mark.parametrize("model", ["matris", "mace", "chgnet", "alignn", "MACE", "CHGNet"])
def test_calculator_factory_uses_supported_paths(monkeypatch, model):
    import fastmd.relaxation as module
    monkeypatch.setattr(module, "FastMDCalculator", lambda name, **kwargs: SimpleNamespace(model=name, **kwargs))
    graph = CUDAGraphConfig(warmup_steps=7, enable_fusions=False)
    source_kwargs = {"arbitrary": 42}
    fixed = make_relaxation_calculator(model, cuda_graph=graph, relax_cell=False, model_kwargs=source_kwargs)
    assert fixed.model == model.lower()
    assert fixed.cuda_graph == graph and fixed.model_kwargs == source_kwargs
    assert source_kwargs == {"arbitrary": 42}
    if model == "alignn":
        with pytest.raises(PropertyNotImplementedError, match="stress"):
            make_relaxation_calculator(model)
    else:
        variable = make_relaxation_calculator(model, cuda_graph=graph)
        assert variable.model_kwargs["compute_stress"] is True
        assert variable.cuda_graph.enabled == (False if model.lower() == "chgnet" else "auto")
        assert variable.cuda_graph.warmup_steps == 7 and not variable.cuda_graph.enable_fusions
        with pytest.raises(ValueError, match="compute_stress"):
            make_relaxation_calculator(model, model_kwargs={"compute_stress": False})
    if model.lower() == "chgnet":
        with pytest.raises(ValueError, match="eager stress"):
            make_relaxation_calculator(model, cuda_graph=True)


def test_chgnet_stress_policy_and_cache(monkeypatch):
    # Test the adapter's EFS task contract without optional CHGNet dependencies.
    module = SimpleNamespace(AseAtomsAdaptor=SimpleNamespace(get_structure=lambda atoms: atoms))
    monkeypatch.setitem(sys.modules, "pymatgen.io.ase", module)
    graph = SimpleNamespace(to=lambda device: graph)
    tasks = []
    def predict(graph, task):
        tasks.append(task)
        return {"e": -1., "f": np.zeros((1, 3)), "s": np.eye(3)}
    backend = CHGNetModel.__new__(CHGNetModel)
    ModelBackend.__init__(backend, device="cpu", cuda_graph=False)
    backend.compute_stress = True
    backend.model = SimpleNamespace(graph_converter=lambda atoms: graph,
                                    predict_graph=predict, is_intensive=True)
    backend.runner = None
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.calc = FastMDCalculator(backend)
    atoms.get_forces()
    atoms.get_stress()
    atoms.get_potential_energy()
    assert tasks == ["efs"] and atoms.calc.stats()["calls"] == 1
    backend.device = torch.device("cuda")
    assert "stress" in backend.graph_unavailable_reason({"energy", "forces"})


@pytest.mark.parametrize("relax_cell", [False, True])
def test_strict_graph_factory_rejects_global_environment_errors(monkeypatch, relax_cell):
    import fastmd.relaxation as module
    properties_seen = []
    def unavailable(properties):
        properties_seen.append(properties)
        return "CUDA Graph requires a CUDA device"
    calculator = SimpleNamespace(backend=SimpleNamespace(graph_unavailable_reason=unavailable))
    monkeypatch.setattr(module, "FastMDCalculator", lambda *args, **kwargs: calculator)
    with pytest.raises(RuntimeError, match="required.*CUDA device"):
        make_relaxation_calculator("matris", device="auto", cuda_graph=True, relax_cell=relax_cell)
    assert properties_seen == [frozenset({"energy", "forces", "stress"} if relax_cell
                                        else {"energy", "forces"})]
    # Auto/eager remain legitimate CPU modes and do not enforce CUDA readiness.
    properties_seen.clear()
    for setting in ("auto", False):
        assert make_relaxation_calculator("matris", cuda_graph=setting, relax_cell=relax_cell) is calculator
    assert properties_seen == []
    calculator.backend.graph_unavailable_reason = lambda properties: None
    assert make_relaxation_calculator("matris", cuda_graph=True, relax_cell=relax_cell) is calculator


@pytest.mark.integration
def test_chgnet_real_cell_relaxation_and_stress_cache():
    pytest.importorskip("pymatgen")
    torch.set_num_threads(2)
    atoms = bulk("Si", "diamond", a=5.5)
    atoms.positions[0] += [.02, -.01, .015]
    calculator = make_relaxation_calculator("chgnet", device="cpu")
    atoms.calc = calculator
    atoms.get_forces()
    atoms.get_stress()
    assert calculator.stats()["calls"] == 1
    result = relax(atoms, calculator, RelaxationConfig(steps=3, fmax=1e-5))
    assert result.steps == 3
    assert result.final_energy < result.initial_energy
    assert np.isfinite(result.stress).all()
    assert not np.array_equal(result.atoms.cell, atoms.cell)
    assert result.stats["mode"] == "eager" and result.stats["graph_calls"] == 0
