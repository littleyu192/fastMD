"""Batch bookkeeping tests independent of any MLIP runtime or GPU."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from fastmd.discovery import run as runner
from fastmd.relaxation import RelaxationConfig


class FakeCalculator:
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self):
        self.calls = 0
        self.clears = 0

    def stats(self):
        return {"mode": "eager", "calls": self.calls, "eager_calls": self.calls, "captures": 0}

    def clear_cache(self):
        self.clears += 1


def fake_relax(atoms, calculator, config):
    calculator.calls += 1
    output = atoms.copy()
    output.positions += 0.01
    energy = -float(len(atoms)) * 3.0
    forces = np.zeros((len(atoms), 3))
    stress = np.zeros(6) if config.relax_cell else None
    values = dict(energy=energy, forces=forces)
    if stress is not None:
        values["stress"] = stress
    output.calc = SinglePointCalculator(output, **values)
    return SimpleNamespace(atoms=output, initial_energy=energy + 1.0, final_energy=energy,
        forces=forces, stress=stress, converged=True, steps=4, elapsed_seconds=0.125,
        max_force=0.0, max_filter_force=0.0, stats=calculator.stats())


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    ids = [f"wbm-1-{i}" for i in range(7)]
    manifest = {"schema_version": 1, "population": "unique_prototypes", "seed": 20261011,
                "population_count": 70, "fraction": 0.1, "sample_count": len(ids), "material_ids": ids,
                "selection_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text(json.dumps(manifest))
    structures = tmp_path / "initial.db"
    db = connect(structures)
    for index, material_id in enumerate(ids):
        atoms = bulk("Si", "diamond", a=5.43).repeat((index % 2 + 1, 1, 1))
        db.write(atoms, material_id=material_id, data={"atoms_info": {"origin": "initial", "item": index}})
    db.metadata = {"selection_sha256": manifest["selection_sha256"], "source_kind": "wbm_initial_atoms", "sample_count": len(ids)}
    monkeypatch.setattr(runner, "relax", fake_relax)
    return manifest_path, structures, tmp_path / "out", ids


def invoke(inputs, **kwargs):
    manifest, structures, output, _ = inputs
    return runner.run_selection(manifest, structures, output, "test", device="cpu", cuda_graph=False,
                                calculator=kwargs.pop("calculator", FakeCalculator()), **kwargs)


def records(inputs):
    return [json.loads(line) for line in (inputs[2] / "results.jsonl").read_text().splitlines()]


def test_batch_preserves_inputs_and_records_units_and_actual_mode(inputs):
    initial_bytes = inputs[1].read_bytes()
    summary = invoke(inputs)
    assert summary["status"] == "complete"
    assert summary["converged"] == 7
    assert summary["actual_modes"] == {"eager": 7}
    assert summary["limited"] is False
    assert inputs[1].read_bytes() == initial_bytes
    results = records(inputs)
    assert [record["material_id"] for record in results] == inputs[3]
    for record in results:
        assert record["final_energy_per_atom"] == -3.0
        assert record["final_energy"] == -3.0 * record["n_atoms"]
        assert record["counter_delta"]["calls"] == 1
        assert record["actual_mode"] == "eager"
    initial = connect(inputs[1]).get(material_id=inputs[3][0]).toatoms()
    output = connect(inputs[2] / "relaxed.db").get(material_id=inputs[3][0])
    np.testing.assert_allclose(output.positions, initial.positions + 0.01)
    assert output.data["atoms_info"]["origin"] == "initial"
    assert output.energy == results[0]["final_energy"]
    metadata = json.loads((inputs[2] / "run_metadata.json").read_text())
    assert metadata["units"]["energy"] == "eV total"
    assert metadata["parameters"]["structures_sha256"] == hashlib.sha256(initial_bytes).hexdigest()


def test_sharding_then_smoke_limit_does_not_change_manifest(inputs):
    original = inputs[0].read_bytes()
    summary = invoke(inputs, shard_index=1, num_shards=2, limit=2)
    assert summary["selected_count"] == 7
    assert summary["shard_count"] == 3
    assert summary["requested_count"] == 2
    assert summary["limited"] is True
    assert [record["material_id"] for record in records(inputs)] == inputs[3][1::2][:2]
    assert inputs[0].read_bytes() == original


def test_failed_and_unconverged_ids_are_retained_and_not_replaced(inputs, monkeypatch):
    def sometimes_fails(atoms, calculator, config):
        if atoms.info["material_id"] == inputs[3][1]:
            raise RuntimeError("test neighbor overflow")
        result = fake_relax(atoms, calculator, config)
        if atoms.info["material_id"] == inputs[3][2]:
            result.converged = False
            result.max_filter_force = 1.0
        return result
    monkeypatch.setattr(runner, "relax", sometimes_fails)
    calculator = FakeCalculator()
    summary = invoke(inputs, calculator=calculator)
    assert (summary["failed"], summary["unconverged"], summary["converged"]) == (1, 1, 5)
    assert calculator.clears == 1
    assert records(inputs)[1]["error_type"] == "RuntimeError"
    assert records(inputs)[1]["final_energy"] is None
    assert records(inputs)[2]["status"] == "unconverged"
    assert connect(inputs[2] / "relaxed.db").count() == 6
    prior_calls = calculator.calls
    resumed = invoke(inputs, calculator=calculator, resume=True)
    assert resumed["skipped_on_resume"] == 7
    assert calculator.calls == prior_calls


def test_resume_after_interrupt_and_torn_jsonl_line(inputs, monkeypatch):
    def interrupted(atoms, calculator, config):
        if atoms.info["material_id"] == inputs[3][2]:
            raise KeyboardInterrupt()
        return fake_relax(atoms, calculator, config)
    monkeypatch.setattr(runner, "relax", interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(inputs)
    assert len(records(inputs)) == 2
    with (inputs[2] / "results.jsonl").open("ab") as stream:
        stream.write(b'{"material_id": "wbm-1-2", "statu')
    # Mimic a crash after the SQLite write but before the JSONL commit marker.
    atoms = connect(inputs[1]).get(material_id=inputs[3][2]).toatoms()
    connect(inputs[2] / "relaxed.db").write(atoms, material_id=inputs[3][2], status="converged")
    monkeypatch.setattr(runner, "relax", fake_relax)
    calculator = FakeCalculator()
    summary = invoke(inputs, calculator=calculator, resume=True)
    assert summary["status"] == "complete"
    assert summary["skipped_on_resume"] == 2
    assert calculator.calls == 5
    assert len(records(inputs)) == 7
    assert connect(inputs[2] / "relaxed.db").count() == 7


def test_resume_rejects_configuration_or_input_changes(inputs):
    invoke(inputs, limit=1)
    with pytest.raises(ValueError, match="Cannot resume"):
        invoke(inputs, limit=2, resume=True)
    with pytest.raises(ValueError, match="Cannot resume"):
        invoke(inputs, limit=1, resume=True, config=RelaxationConfig(fmax=0.1))
    connect(inputs[1]).update(1, extra="changed")
    with pytest.raises(ValueError, match="Cannot resume"):
        invoke(inputs, limit=1, resume=True)


def test_resume_rejects_changed_checkpoint(inputs, tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"first version")
    invoke(inputs, checkpoint=checkpoint, limit=1)
    (checkpoint / "model.bin").write_bytes(b"different version")
    with pytest.raises(ValueError, match="Cannot resume"):
        invoke(inputs, checkpoint=checkpoint, limit=1, resume=True)


def test_stress_incompatibility_and_factory_errors_are_global(inputs, monkeypatch):
    calculator = FakeCalculator()
    calculator.implemented_properties = ["energy", "forces"]
    with pytest.raises(PropertyNotImplementedError, match="stress"):
        invoke(inputs, calculator=calculator)
    assert not (inputs[2] / "results.jsonl").exists()
    def unavailable(*args, **kwargs):
        raise ImportError("missing model runtime")
    monkeypatch.setattr(runner, "make_relaxation_calculator", unavailable)
    with pytest.raises(ImportError, match="runtime"):
        runner.run_selection(inputs[0], inputs[1], inputs[2].parent / "factory-error", "test", calculator=None)


def test_nan_results_are_failures_and_json_never_contains_nan(inputs, monkeypatch):
    def invalid(atoms, calculator, config):
        result = fake_relax(atoms, calculator, config)
        result.final_energy = float("nan")
        return result
    monkeypatch.setattr(runner, "relax", invalid)
    summary = invoke(inputs, limit=1)
    assert summary["failed"] == 1
    record = records(inputs)[0]
    assert record["error_type"] == "ValueError"
    assert record["final_energy"] is None
    assert ":NaN" not in (inputs[2] / "results.jsonl").read_text()


def test_overwrite_and_missing_inputs_are_rejected(inputs):
    invoke(inputs, limit=1)
    with pytest.raises(FileExistsError, match="resume"):
        invoke(inputs, limit=1)
    connect(inputs[1]).delete([7])
    with pytest.raises(ValueError, match="missing 1 selected IDs"):
        invoke(inputs, resume=True)


def test_no_stress_needed_for_fixed_cell(inputs):
    calculator = FakeCalculator()
    calculator.implemented_properties = ["energy", "forces"]
    invoke(inputs, calculator=calculator, config=RelaxationConfig(relax_cell=False), limit=1)
    assert records(inputs)[0]["stress"] is None


def test_resume_rejects_dependency_changes(inputs, monkeypatch):
    invoke(inputs, limit=1)
    monkeypatch.setattr(runner, "_dependency_versions", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="Cannot resume"):
        invoke(inputs, limit=1, resume=True)


def test_runner_replacement_keeps_cache_snapshots_without_invalid_deltas(inputs, monkeypatch):
    class GraphCalculator(FakeCalculator):
        def __init__(self):
            super().__init__()
            self.captures = 0
            self.invalidations = 0

        def stats(self):
            return {"mode": "cuda_graph", "calls": self.calls, "graph_calls": self.calls,
                    "invalidations": self.invalidations, "cache": {"captures": self.captures}}

    def with_runner_replacement(atoms, calculator, config):
        result = fake_relax(atoms, calculator, config)
        # A different atom count creates a new graph runner. Its local counter
        # falls from four to one even though the second sample did capture.
        calculator.captures = 4 if calculator.calls == 1 else 1
        calculator.invalidations = calculator.calls - 1
        result.stats = calculator.stats()
        return result

    monkeypatch.setattr(runner, "relax", with_runner_replacement)
    summary = invoke(inputs, calculator=GraphCalculator(), limit=2)
    second = records(inputs)[1]
    assert second["stats_before"]["cache"]["captures"] == 4
    assert second["stats"]["cache"]["captures"] == 1
    assert second["counter_delta"] == {"calls": 1, "graph_calls": 1, "invalidations": 1}
    assert "runner-local snapshots" in summary["statistics_note"]
    assert summary["structure_seconds"] == 0.25
    assert summary["actual_modes"] == {"cuda_graph": 2}


@pytest.mark.parametrize("metadata_change, message", [
    ({}, "selection_sha256"),
    ({"source_kind": "wbm_initial_atoms"}, "selection_sha256"),
    ({"source_kind": None}, "initial, unrelaxed"),
    ({"source_kind": "wbm_dft_relaxed_atoms"}, "initial, unrelaxed"),
    ({"sample_count": 6}, "sample_count"),
])
def test_input_provenance_is_required(inputs, metadata_change, message):
    database = connect(inputs[1])
    if "selection_sha256" in message:
        database.metadata = metadata_change
    else:
        metadata = dict(database.metadata)
        metadata.update(metadata_change)
        if metadata.get("source_kind") is None:
            metadata.pop("source_kind")
        database.metadata = metadata
    with pytest.raises(ValueError, match=message):
        invoke(inputs, limit=1)
    assert not inputs[2].exists()


@pytest.mark.parametrize("options", [{"limit": 1}, {"shard_index": 1, "num_shards": 2}])
def test_complete_manifest_database_required_even_for_smoke_or_shard(inputs, options):
    # The final material is outside both the requested smoke and requested shard.
    # It must still be present so all runs share the same complete prepared DB.
    connect(inputs[1]).delete([7])
    with pytest.raises(ValueError, match="missing 1 selected IDs"):
        invoke(inputs, **options)
    assert not inputs[2].exists()
