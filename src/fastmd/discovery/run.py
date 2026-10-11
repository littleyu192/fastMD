"""Resumable, deterministic Discovery relaxation batches in ASE units.

Each output directory belongs to exactly one shard and immutable configuration.
A failed material remains in the results: it is never replaced by another sample.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.db import connect

from fastmd.relaxation import RelaxationConfig, make_relaxation_calculator, relax

from .data import load_manifest


def _json_value(value):
    """Convert metadata/results without emitting JavaScript's invalid NaN tokens."""
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Torch dtypes/devices appear in model options, but arbitrary stateful objects
    # cannot be given a reproducible identity by their repr/address.
    if type(value).__module__ == "torch" and type(value).__name__ in {"dtype", "device"}:
        return str(value)
    raise TypeError(f"Cannot serialize run parameter of type {type(value).__name__}")


def _canonical(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path):
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return {"path": str(path), "sha256": _sha256(path)}
    if path.is_dir():
        files = [{"path": str(file.relative_to(path)), "sha256": _sha256(file)}
                 for file in sorted(path.rglob("*")) if file.is_file()]
        return {"path": str(path), "sha256": hashlib.sha256(_canonical(files).encode()).hexdigest(),
                "files": files}
    raise FileNotFoundError(f"Checkpoint does not exist: {path}")


def _dependency_versions():
    packages = ("fastmd-mlip", "ase", "numpy", "torch", "scipy", "pymatgen", "mace-torch",
                "e3nn", "triton", "dgl", "jarvis-tools", "warp-lang", "nvalchemi-toolkit-ops")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _implementation_digest():
    root = Path(__file__).resolve().parents[1]
    files = [{"path": str(path.relative_to(root)), "sha256": _sha256(path)}
             for path in sorted(root.rglob("*.py"))]
    return hashlib.sha256(_canonical(files).encode()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(_json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _output_lock(directory):
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this output directory; use one directory per shard") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read_records(path):
    """Recover only a torn final JSONL line; reject corruption in earlier lines."""
    if not path.exists():
        return []
    records = []
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if stream.read(1):
                    raise ValueError(f"Corrupt JSONL record before end of file: {path}")
                stream.truncate(start)
                break
            if not isinstance(value, dict):
                raise ValueError(f"Invalid JSONL record in {path}")
            records.append(value)
            if not line.endswith(b"\n"):
                stream.seek(0, 2)
                stream.write(b"\n")
                break
        stream.flush()
        os.fsync(stream.fileno())
    return records


def _stats(calculator):
    return deepcopy(calculator.stats()) if hasattr(calculator, "stats") else {}


def _safe_stats(calculator):
    try:
        return _stats(calculator)
    except Exception:
        return {}


def _counter_delta(before, after):
    # Only these backend counters survive runner replacement. Cache captures and
    # replays restart at zero when composition/capacity invalidates a runner, so
    # subtracting their snapshots can hide recaptures or produce negative counts.
    counters = {"calls", "graph_calls", "eager_calls", "invalidations"}
    return {name: after[name] - before.get(name, 0) for name in counters
            if isinstance(after.get(name), (int, float))}


def _accumulate(aggregate, record):
    aggregate["statuses"][record["status"]] += 1
    aggregate["modes"][record.get("actual_mode", "unknown")] += 1
    aggregate["count"] += 1
    aggregate["seconds"] += record["elapsed_seconds"]


def _aggregate_records(records):
    aggregate = {"statuses": Counter(), "modes": Counter(), "count": 0, "seconds": 0.0}
    for record in records:
        _accumulate(aggregate, record)
    return aggregate


def _validate_result(result, atom_count):
    if len(result.atoms) != atom_count:
        raise ValueError("Relaxation changed the atom count")
    for name in ("initial_energy", "final_energy", "forces", "elapsed_seconds", "max_force", "max_filter_force"):
        if not np.isfinite(getattr(result, name)).all():
            raise ValueError(f"Relaxation returned non-finite {name}")
    if result.stress is not None and not np.isfinite(result.stress).all():
        raise ValueError("Relaxation returned non-finite stress")
    if not np.isfinite(result.atoms.positions).all() or not np.isfinite(result.atoms.cell.array).all():
        raise ValueError("Relaxation returned non-finite geometry")
    if np.asarray(result.forces).shape != (atom_count, 3):
        raise ValueError("Relaxation returned an invalid forces shape")


def _summary(aggregate, metadata, load_seconds, started, skipped):
    counts = aggregate["statuses"]
    modes = aggregate["modes"]
    return {
        "schema_version": 1, "run_fingerprint": metadata["run_fingerprint"],
        "population": metadata["parameters"]["population"],
        "selected_count": metadata["selected_count"], "shard_count": metadata["shard_count"],
        "requested_count": metadata["requested_count"], "limited": metadata["limited"],
        "completed_count": aggregate["count"], "converged": counts["converged"],
        "unconverged": counts["unconverged"], "failed": counts["failed"],
        "status": "complete" if aggregate["count"] == metadata["requested_count"] else "incomplete",
        "checkpoint_verification": metadata["checkpoint_verification"],
        "skipped_on_resume": skipped, "actual_modes": dict(modes),
        "statistics_note": metadata["statistics_note"],
        "model_load_seconds_this_session": load_seconds,
        "wall_seconds_this_session": perf_counter() - started,
        "structure_seconds": aggregate["seconds"],
        "timing_note": "Structure timings include initial evaluation, graph capture and all optimization steps; model loading is separate.",
        "energy_note": "Raw total and per-atom model energies in eV; no formation-energy or convex-hull correction.",
    }


def run_selection(manifest_path, structures_path, output_dir, model, checkpoint=None,
                  device="auto", cuda_graph="auto", config=None, model_kwargs=None,
                  limit=None, shard_index=0, num_shards=1, resume=False, calculator=None):
    """Relax a manifest selection from an ASE SQLite database.

    Sharding uses the original manifest order, then ``limit`` truncates only that
    shard for a smoke run. Resume requires byte-identical inputs, weights, source
    implementation and parameters. Completed failed records are also skipped.
    Pass an explicit checkpoint to verify weight identity on resume: default or
    named model downloads cannot be hashed before loading, and their identifiers
    alone do not guarantee identical weights. ``calculator`` allows an already
    initialized ASE calculator for validation; its weights are not fingerprinted.
    """
    config = RelaxationConfig() if config is None else config
    if not isinstance(config, RelaxationConfig):
        raise TypeError("config must be a RelaxationConfig")
    if type(num_shards) is not int or num_shards < 1 or type(shard_index) is not int or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    manifest_path = Path(manifest_path).expanduser().resolve()
    structures_path = Path(structures_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not structures_path.is_file():
        raise FileNotFoundError(structures_path)
    # Neither input may alias a file this runner writes, even with unusual names.
    output_paths = {output_dir / name for name in
                    ("relaxed.db", "run_metadata.json", "results.jsonl", "summary.json", ".run.lock")}
    if structures_path in output_paths or manifest_path in output_paths:
        raise ValueError("Input files cannot also be output files")
    manifest = load_manifest(manifest_path)
    material_ids = manifest["material_ids"]
    selection_digest = manifest["selection_sha256"]
    shard_ids = material_ids[shard_index::num_shards]
    selected = shard_ids if limit is None else shard_ids[:limit]
    selected_set = set(selected)
    input_db = connect(str(structures_path))
    db_metadata = input_db.metadata
    if db_metadata.get("selection_sha256") != selection_digest:
        raise ValueError("Input database selection_sha256 does not match the manifest")
    if db_metadata.get("source_kind") != "wbm_initial_atoms":
        raise ValueError("Discovery relaxation requires initial, unrelaxed WBM structures")
    if "sample_count" in db_metadata and (
            type(db_metadata["sample_count"]) is not int or db_metadata["sample_count"] != len(material_ids)):
        raise ValueError("Input database sample_count does not match the complete manifest")
    rows_by_material = {}
    for row in input_db.select(columns=["id", "key_value_pairs"], include_data=False):
        material_id = row.get("material_id")
        if material_id in rows_by_material:
            raise ValueError(f"Duplicate material_id in input database: {material_id}")
        rows_by_material[material_id] = row.id
    missing = set(material_ids) - rows_by_material.keys()
    if missing:
        raise ValueError(f"Input database is missing {len(missing)} selected IDs, including {sorted(missing)[:3]}")
    if "sample_count" in db_metadata and db_metadata["sample_count"] != len(rows_by_material):
        raise ValueError("Input database sample_count does not match its row count")
    parameters = {
        "model": model, "checkpoint": _path_identity(checkpoint), "device": str(device),
        "cuda_graph": cuda_graph, "relaxation": asdict(config), "model_kwargs": model_kwargs or {},
        "population": manifest["population"], "selection_sha256": selection_digest,
        "manifest_sha256": _sha256(manifest_path), "structures_sha256": _sha256(structures_path),
        "source_sha256": _implementation_digest(), "dependencies": _dependency_versions(), "limit": limit,
        "shard_index": shard_index, "num_shards": num_shards,
        "calculator_class": (f"{type(calculator).__module__}.{type(calculator).__qualname__}" if calculator is not None else None),
    }
    parameters = _json_value(parameters)
    fingerprint = hashlib.sha256(_canonical(parameters).encode()).hexdigest()
    metadata = {
        "schema_version": 1, "run_fingerprint": fingerprint, "parameters": parameters,
        "manifest_path": str(manifest_path), "structures_path": str(structures_path),
        "selected_count": len(material_ids), "shard_count": len(shard_ids),
        "requested_count": len(selected), "limited": limit is not None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_verification": ("explicit-checkpoint-sha256" if checkpoint is not None and calculator is None
            else "unverified-injected-calculator" if calculator is not None else "unverified-default-weights"),
        "checkpoint_note": "Use an explicit checkpoint for weight-verified resume; default identifiers and injected calculators do not establish weight identity.",
        "statistics_note": "counter_delta contains only lifetime backend counters. stats.cache and stats_before.cache are runner-local snapshots: replacement resets them, so their differences are not capture/replay totals.",
        "units": {"energy": "eV total", "energy_per_atom": "eV/atom", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_dir):
        metadata_path = output_dir / "run_metadata.json"
        results_path = output_dir / "results.jsonl"
        database_path = output_dir / "relaxed.db"
        if metadata_path.exists():
            if not resume:
                raise FileExistsError("Output already contains a run; use resume=True or a new directory")
            previous = json.loads(metadata_path.read_text())
            if previous.get("run_fingerprint") != fingerprint:
                raise ValueError("Cannot resume: configuration, inputs, checkpoint or implementation changed")
            metadata = previous
        else:
            if any(path.exists() for path in (results_path, database_path)):
                raise ValueError("Existing outputs have no run_metadata.json; refusing to overwrite")
            _atomic_json(metadata_path, metadata)
        records = _read_records(results_path)
        completed = set()
        for record in records:
            material_id = record.get("material_id")
            if material_id not in selected_set or material_id in completed or record.get("status") not in {"converged", "unconverged", "failed"}:
                raise ValueError("Results contain duplicate/unexpected IDs or invalid status")
            completed.add(material_id)
        output_db = connect(str(database_path))
        # A crash between DB commit and JSONL append leaves an orphan row. Delete
        # only those rows and retry the same ID; JSONL is the commit record.
        saved = {}
        orphan_ids = []
        for row in output_db.select(columns=["id", "key_value_pairs"], include_data=False):
            material_id = row.get("material_id")
            if material_id not in completed:
                orphan_ids.append(row.id)
            elif material_id in saved:
                raise ValueError("Duplicate material_id in output database")
            else:
                saved[material_id] = row.id
        if orphan_ids:
            output_db.delete(orphan_ids)
        expected_saved = {record["material_id"] for record in records if record["status"] != "failed"}
        if set(saved) != expected_saved:
            raise ValueError("Results and relaxed.db disagree; a recorded final structure is missing or unexpected")
        started = perf_counter()
        load_started = perf_counter()
        pending = [material_id for material_id in selected if material_id not in completed]
        if pending and calculator is None:
            calculator = make_relaxation_calculator(model, checkpoint=checkpoint, device=device,
                cuda_graph=cuda_graph, relax_cell=config.relax_cell, model_kwargs=model_kwargs)
        load_seconds = perf_counter() - load_started if pending else 0.0
        if pending and config.relax_cell and "stress" not in calculator.implemented_properties:
            raise PropertyNotImplementedError("Cell relaxation requires a calculator providing stress")
        skipped = len(completed)
        aggregate = _aggregate_records(records)
        with results_path.open("a") as results_stream:
            for material_id in pending:
                row = input_db.get(id=rows_by_material[material_id])
                atoms = row.toatoms()
                atoms.info.update(deepcopy(row.data.get("atoms_info", row.data.get("info", {}))))
                atoms.info["material_id"] = material_id
                before = _safe_stats(calculator)
                sample_started = perf_counter()
                result = None
                try:
                    result = relax(atoms, calculator, config)
                    _validate_result(result, len(atoms))
                    after = result.stats
                    record = {
                        "material_id": material_id, "status": "converged" if result.converged else "unconverged",
                        "n_atoms": len(atoms), "initial_energy": result.initial_energy,
                        "initial_energy_per_atom": result.initial_energy / len(atoms),
                        "final_energy": result.final_energy, "final_energy_per_atom": result.final_energy / len(atoms),
                        "steps": result.steps, "elapsed_seconds": result.elapsed_seconds,
                        "max_force": result.max_force, "max_filter_force": result.max_filter_force,
                        "stress": result.stress, "actual_mode": after.get("mode", "unknown"),
                        "stats": after, "stats_before": before, "counter_delta": _counter_delta(before, after),
                    }
                    record = _json_value(record)
                except Exception as exc:
                    result = None
                    after = _safe_stats(calculator)
                    record = {
                        "material_id": material_id, "status": "failed", "n_atoms": len(atoms),
                        "initial_energy": None, "initial_energy_per_atom": None,
                        "final_energy": None, "final_energy_per_atom": None, "steps": None,
                        "elapsed_seconds": perf_counter() - sample_started,
                        "max_force": None, "max_filter_force": None, "stress": None,
                        "actual_mode": after.get("mode", "unknown"), "stats": after,
                        "stats_before": before, "counter_delta": _counter_delta(before, after),
                        "error_type": type(exc).__name__, "error": str(exc)[:2000],
                        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=3))[-4000:],
                    }
                    if hasattr(calculator, "clear_cache"):
                        calculator.clear_cache()
                    elif hasattr(calculator, "reset"):
                        calculator.reset()
                # Persistence failures abort: they must never become model errors.
                if result is not None:
                    output_db.write(result.atoms, material_id=material_id, status=record["status"],
                                    data={"result": record, "atoms_info": deepcopy(result.atoms.info)})
                results_stream.write(_canonical(record) + "\n")
                results_stream.flush()
                os.fsync(results_stream.fileno())
                _accumulate(aggregate, record)
                _atomic_json(output_dir / "summary.json", _summary(aggregate, metadata, load_seconds, started, skipped))
        summary = _summary(aggregate, metadata, load_seconds, started, skipped)
        _atomic_json(output_dir / "summary.json", summary)
        return summary
