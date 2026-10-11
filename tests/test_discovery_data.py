"""Reproducible selection and atomic extraction of unrelaxed WBM structures."""
import csv
from copy import deepcopy
import gzip
import io
import json
import zipfile

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write

from fastmd.discovery.data import (
    download_verified, extract_subset, file_hash, load_manifest, sample_manifest,
    selection_hash, write_manifest,
)


def summary_file(path, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["material_id", "unique_prototype", "e_form_dft"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def sample_rows(count=100):
    return [dict(material_id=f"wbm-1-{index:04}", unique_prototype=index % 5 != 0,
                 e_form_dft=-index / 10) for index in range(count)]


def test_selection_is_row_order_independent_and_target_blind(tmp_path):
    rows = sample_rows()
    original = sample_manifest(summary_file(tmp_path / "original.csv.gz", rows), seed=17)
    changed = deepcopy(rows[::-1])
    for row in changed:
        row["e_form_dft"] = "a completely different or missing DFT label"
    reordered = sample_manifest(summary_file(tmp_path / "changed.csv", changed), seed=17)
    assert original["material_ids"] == reordered["material_ids"]
    assert original["selection_sha256"] == reordered["selection_sha256"]
    assert original["summary"]["sha256"] != reordered["summary"]["sha256"]
    assert original["sample_count"] == 8
    assert original["population_count"] == 80 and original["full_test_count"] == 100
    assert original["material_ids"] == sorted(set(original["material_ids"]))


def test_unique_filter_and_full_population_are_distinct(tmp_path):
    rows = sample_rows(20)
    path = summary_file(tmp_path / "summary.csv", rows)
    unique = sample_manifest(path, fraction=1)
    full = sample_manifest(path, population="full_test_set", fraction=1)
    expected = {row["material_id"] for row in rows if row["unique_prototype"]}
    assert set(unique["material_ids"]) == expected
    assert set(full["material_ids"]) == {row["material_id"] for row in rows}
    assert unique["sample_count"] == 16 and full["sample_count"] == 20


def test_manifest_roundtrip_immutable_existing_selection(tmp_path):
    source = summary_file(tmp_path / "summary.csv", sample_rows())
    manifest = sample_manifest(source)
    path = tmp_path / "selection.json"
    assert write_manifest(manifest, path) == path
    assert load_manifest(path) == manifest
    assert write_manifest(manifest, path) == path
    previous = path.read_bytes()
    other = deepcopy(manifest)
    other["seed"] += 1
    with pytest.raises(FileExistsError, match="different subset"):
        write_manifest(other, path)
    assert path.read_bytes() == previous


@pytest.mark.parametrize("change", ["checksum", "duplicate", "count", "population", "fraction"])
def test_manifest_tampering_is_rejected(tmp_path, change):
    manifest = sample_manifest(summary_file(tmp_path / "summary.csv", sample_rows()))
    if change == "checksum":
        manifest["material_ids"][0] = "wbm-tampered"
    elif change == "duplicate":
        manifest["material_ids"][1] = manifest["material_ids"][0]
        manifest["selection_sha256"] = selection_hash(manifest["material_ids"])
    elif change == "count":
        manifest["sample_count"] += 1
    elif change == "population":
        manifest["population_count"] = 70
    else:
        manifest["fraction"] = .2
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize("problem", ["duplicate", "bad_flag", "wrong_md5"])
def test_invalid_summary_is_rejected(tmp_path, problem):
    rows = sample_rows()
    if problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "bad_flag":
        rows[0]["unique_prototype"] = "perhaps"
    path = summary_file(tmp_path / "summary.csv", rows)
    kwargs = {"expected_md5": "0" * 32} if problem == "wrong_md5" else {}
    with pytest.raises(ValueError):
        sample_manifest(path, **kwargs)


@pytest.fixture
def extraction_input(tmp_path):
    rows = [dict(material_id="wbm-1-0001", unique_prototype=True, e_form_dft=-2),
            dict(material_id="wbm-1-0002", unique_prototype=True, e_form_dft=-3)]
    manifest = sample_manifest(summary_file(tmp_path / "summary.csv", rows), fraction=1)
    manifest_path = write_manifest(manifest, tmp_path / "selection.json")
    return manifest_path, manifest, tmp_path / "initial.zip", tmp_path / "selected.db"


def frame(material_id, displacement=0):
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.positions[0, 0] += displacement
    atoms.info.update(material_id=material_id, source_note="initial geometry")
    # Labels in the input must never be reused as model results after extraction.
    atoms.calc = SinglePointCalculator(atoms, energy=-99.0, forces=np.ones((len(atoms), 3)))
    stream = io.StringIO()
    write(stream, atoms, format="extxyz")
    return stream.getvalue(), atoms


def test_zip_multiframe_uses_metadata_ids_and_drops_labels(extraction_input):
    manifest_path, manifest, archive, destination = extraction_input
    first, first_atoms = frame("wbm-1-0001", .03)
    second, second_atoms = frame("wbm-1-0002", .07)
    unselected, _ = frame("wbm-unselected")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        # Both chosen structures are in one unrelated member. A filename that
        # looks selected deliberately contains a different, unselected ID.
        package.writestr("arbitrary/trajectory.extxyz", first + "\n" + second)
        package.writestr("wbm-1-0001.extxyz", unselected)
        package.writestr("README.txt", "not an extxyz structure")
    result = extract_subset(manifest_path, archive, destination, expected_md5=None)
    assert result["count"] == 2
    with connect(destination) as database:
        assert len(database) == 2
        assert database.metadata["selection_sha256"] == manifest["selection_sha256"]
        assert database.metadata["source_sha256"] == file_hash(archive)
        for original in (first_atoms, second_atoms):
            row = database.get(material_id=original.info["material_id"])
            extracted = row.toatoms()
            np.testing.assert_allclose(extracted.positions, original.positions, rtol=0, atol=1e-8)
            np.testing.assert_allclose(extracted.cell, original.cell, rtol=0, atol=1e-8)
            assert extracted.pbc.all() and extracted.calc is None
            assert "energy" not in row and "forces" not in row
            assert row.data["atoms_info"]["source_note"] == "initial geometry"
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        extract_subset(manifest_path, archive, destination, expected_md5=None)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("problem", ["missing", "duplicate"])
def test_missing_or_duplicate_selected_id_never_publishes_partial_db(extraction_input, problem):
    manifest_path, _, archive, destination = extraction_input
    first, _ = frame("wbm-1-0001")
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("first.extxyz", first)
        if problem == "duplicate":
            package.writestr("second.extxyz", first)
    with pytest.raises(ValueError, match="Missing|Duplicate"):
        extract_subset(manifest_path, archive, destination, expected_md5=None)
    assert not destination.exists()
    assert not destination.with_name(destination.stem + ".part.db").exists()


def test_wrong_archive_source_checksum_is_rejected(extraction_input):
    manifest_path, _, archive, destination = extraction_input
    first, _ = frame("wbm-1-0001")
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("first.extxyz", first)
    with pytest.raises(ValueError, match="verified official"):
        extract_subset(manifest_path, archive, destination)
    assert not destination.exists()
    assert not destination.with_name(destination.stem + ".part.db").exists()


@pytest.mark.parametrize("field", ["positions", "cell"])
def test_nonfinite_initial_geometry_does_not_publish_database(extraction_input, field):
    manifest_path, _, archive, destination = extraction_input
    cell = "nan 0 0 0 5 0 0 0 5" if field == "cell" else "5 0 0 0 5 0 0 0 5"
    position = "nan 0 0" if field == "positions" else "0 0 0"
    text = (f'1\nLattice="{cell}" Properties=species:S:1:pos:R:3 '
            f'pbc="T T T" material_id=wbm-1-0001\nSi {position}\n')
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("initial.extxyz", text)
    with pytest.raises(ValueError, match="Invalid periodic"):
        extract_subset(manifest_path, archive, destination, expected_md5=None)
    assert not destination.exists()
    assert not destination.with_name(destination.stem + ".part.db").exists()


def test_download_rejects_corrupt_payload_and_cached_source(tmp_path, monkeypatch):
    import fastmd.discovery.data as module
    path = tmp_path / "source.zip"
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="Downloaded file checksum"):
        download_verified("https://example.invalid/source.zip", path, "0" * 32)
    assert not path.exists() and not path.with_name(path.name + ".part").exists()
    path.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="Cached file checksum"):
        download_verified("https://example.invalid/source.zip", path, "0" * 32)
