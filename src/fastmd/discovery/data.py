"""Select WBM IDs without inspecting target energies; extract initial structures."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request
import zipfile

SUMMARY_URL = "https://ndownloader.figshare.com/files/64706751"
SUMMARY_MD5 = "1555268d3aa59d20262cb9d01ce0994b"
INITIAL_URL = "https://ndownloader.figshare.com/files/48169597"
INITIAL_MD5 = "2a292211ca6acb30ed8416178d644098"
DEFAULT_SEED = 20261011
DEFAULT_MANIFEST = "benchmarks/discovery/unique-prototypes-10pct-seed20261011.json"


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def download_verified(url, destination, expected_md5):
    """Download a published file atomically; never accept partial/corrupt caches."""
    destination = Path(destination)
    if destination.exists():
        if file_hash(destination, "md5") != expected_md5:
            raise ValueError(f"Cached file checksum differs from the official release: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "fastMD-discovery"})
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
        if file_hash(temporary, "md5") != expected_md5:
            raise ValueError(f"Downloaded file checksum differs from the official release: {url}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def sample_manifest(summary, *, population="unique_prototypes", fraction=0.1,
                    seed=DEFAULT_SEED, expected_md5=None):
    """Seeded SHA256 ranking gives a reproducible sample without replacement.

    Hash each (seed, material_id), select the lowest floor(N*fraction) ranks,
    then sort IDs for a stable execution order. This does not depend on CSV row
    order, random-library versions, chemistry, DFT energies or model results.
    """
    if population not in {"unique_prototypes", "full_test_set"}:
        raise ValueError("population must be unique_prototypes or full_test_set")
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be finite and in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    summary = Path(summary)
    md5 = file_hash(summary, "md5")
    if expected_md5 is not None and md5 != expected_md5:
        raise ValueError("Summary does not match the requested official release")
    opener = gzip.open if summary.suffix == ".gz" else open
    all_ids, eligible = set(), []
    with opener(summary, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"material_id"} | ({"unique_prototype"} if population == "unique_prototypes" else set())
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Summary needs columns: {sorted(required)}")
        for row in reader:
            material_id = row["material_id"]
            if not material_id or material_id in all_ids or "\n" in material_id:
                raise ValueError(f"Missing, invalid or duplicate material_id: {material_id!r}")
            all_ids.add(material_id)
            if population == "unique_prototypes":
                flag = row["unique_prototype"].strip().lower()
                if flag not in {"true", "false", "1", "0"}:
                    raise ValueError(f"Invalid unique_prototype flag for {material_id}: {flag!r}")
                if flag in {"false", "0"}:
                    continue
            eligible.append(material_id)
    count = math.floor(len(eligible) * fraction)
    if count == 0:
        raise ValueError("The requested fraction selects no structures")
    ranks = sorted(eligible, key=lambda value: (
        hashlib.sha256(f"{seed}\0{value}".encode()).digest(), value))
    ids = sorted(ranks[:count])
    return dict(schema_version=1, population=population, population_count=len(eligible),
                full_test_count=len(all_ids), fraction=fraction, seed=seed, sample_count=count,
                selection_algorithm="sha256-rank-v1", selection_sha256=selection_hash(ids),
                summary=dict(filename=summary.name, url=SUMMARY_URL, md5=md5,
                             sha256=file_hash(summary)),
                initial_structures=dict(url=INITIAL_URL, md5=INITIAL_MD5), material_ids=ids)


def load_manifest(path):
    result = json.loads(Path(path).read_text())
    ids = result.get("material_ids")
    if result.get("schema_version") != 1 or not isinstance(ids, list) or not ids:
        raise ValueError("Invalid or empty discovery manifest")
    if any(not isinstance(value, str) or not value or "\n" in value for value in ids):
        raise ValueError("Invalid material IDs in manifest")
    if len(set(ids)) != len(ids) or result.get("selection_sha256") != selection_hash(ids):
        raise ValueError("Manifest has duplicate IDs or an invalid selection checksum")
    if result.get("sample_count") != len(ids):
        raise ValueError("Manifest sample_count does not match material IDs")
    if result.get("population") not in {"unique_prototypes", "full_test_set"}:
        raise ValueError("Invalid manifest population")
    if "population_count" in result or "fraction" in result:
        population_count, fraction = result.get("population_count"), result.get("fraction")
        if (type(population_count) is not int or population_count < len(ids)
                or isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 < fraction <= 1
                or math.floor(population_count * fraction) != len(ids)):
            raise ValueError("Manifest population_count/fraction does not match sample_count")
    return result


def write_manifest(manifest, destination):
    """An existing manifest is immutable unless its complete contents agree."""
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise FileExistsError(f"Refusing to replace a different subset: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    return destination


def _extxyz_frames(stream):
    """Read one frame at a time, including ZIP members containing trajectories."""
    while True:
        count_line = stream.readline()
        if not count_line:
            return
        if not count_line.strip():
            continue
        try:
            count = int(count_line)
        except ValueError as exc:
            raise ValueError("Invalid initial extxyz atom count") from exc
        if count <= 0:
            raise ValueError("Initial structure must contain atoms")
        header = stream.readline()
        if not header:
            raise ValueError("Truncated extxyz header")
        rows = [stream.readline() for _ in range(count)]
        if any(not line for line in rows):
            raise ValueError("Truncated extxyz atom records")
        yield header, count_line + header + "".join(rows)


def extract_subset(manifest_path, archive, destination, *, expected_md5=INITIAL_MD5):
    """Store only selected, unrelaxed structures in an ASE database atomically.

    ID matching uses frame metadata, never ZIP member names. All selected IDs
    must appear exactly once; missing/duplicate IDs fail without publishing a
    partial database. No relaxed structures or target energies are read.
    """
    from ase.db import connect
    from ase.io import read
    from ase.io.extxyz import key_val_str_to_dict
    import numpy as np
    manifest = load_manifest(manifest_path)
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing structures: {destination}")
    if expected_md5 is not None and file_hash(archive, "md5") != expected_md5:
        raise ValueError("Archive is not the verified official WBM initial structure file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part.db")
    if temporary.exists():
        raise FileExistsError(f"Remove an incomplete extraction before retrying: {temporary}")
    selected, seen = set(manifest["material_ids"]), set()
    try:
        with connect(temporary, type="db") as database, zipfile.ZipFile(archive) as package:
            database.metadata = dict(schema_version=1, source_kind="wbm_initial_atoms",
                selection_sha256=manifest["selection_sha256"], source_sha256=file_hash(archive),
                source_md5=file_hash(archive, "md5"), sample_count=len(selected))
            for member in package.infolist():
                if member.is_dir() or not member.filename.endswith(".extxyz"):
                    continue
                with package.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
                    for header, frame in _extxyz_frames(stream):
                        material_id = key_val_str_to_dict(header).get("material_id")
                        if material_id not in selected:
                            continue
                        if material_id in seen:
                            raise ValueError(f"Duplicate selected material_id in archive: {material_id}")
                        atoms = read(io.StringIO(frame), format="extxyz")
                        if (not atoms.pbc.all() or not np.isfinite(atoms.positions).all()
                                or not np.isfinite(atoms.cell.array).all() or atoms.cell.volume <= 0):
                            raise ValueError(f"Invalid periodic initial structure: {material_id}")
                        atoms.calc = None
                        database.write(atoms, material_id=material_id,
                                       data={"atoms_info": dict(atoms.info)})
                        seen.add(material_id)
            missing = selected - seen
            if missing:
                raise ValueError(f"Missing {len(missing)} selected initial structures: {sorted(missing)[:5]}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dict(structures=str(destination), count=len(seen),
                selection_sha256=manifest["selection_sha256"])
