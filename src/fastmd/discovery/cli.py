"""Prepare a reproducible WBM subset and run ASE relaxations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import (DEFAULT_MANIFEST, DEFAULT_SEED, INITIAL_MD5, INITIAL_URL,
                   SUMMARY_MD5, SUMMARY_URL, download_verified, extract_subset,
                   load_manifest, sample_manifest, write_manifest)

DEFAULT_STRUCTURES = ".cache/discovery/unique-prototypes-10pct-seed20261011.db"


def _graph_mode(value):
    return {"auto": "auto", "true": True, "false": False}[value]


def _json_object(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("Expected a JSON object") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Select IDs from the verified official WBM summary")
    sample.add_argument("--summary", type=Path)
    sample.add_argument("--cache-dir", type=Path, default=Path(".cache/discovery"))
    sample.add_argument("--population", choices=["unique_prototypes", "full_test_set"], default="unique_prototypes")
    sample.add_argument("--fraction", type=float, default=0.1)
    sample.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sample.add_argument("--output", type=Path, default=Path(DEFAULT_MANIFEST))
    for name, description in (("extract", "Extract selected initial structures from the official ZIP"),
                              ("prepare", "Prepare the default subset, downloading verified data as needed")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
        command.add_argument("--initial-archive", type=Path)
        command.add_argument("--output", type=Path, default=Path(DEFAULT_STRUCTURES))
        command.add_argument("--cache-dir", type=Path, default=Path(".cache/discovery"))
        if name == "prepare":
            command.add_argument("--summary", type=Path)
    run = commands.add_parser("run", help="Relax a prepared subset (no formation-energy or hull scoring)")
    run.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    run.add_argument("--structures", type=Path, default=Path(DEFAULT_STRUCTURES))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", choices=["matris", "chgnet", "alignn", "mace"], required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--device", default="auto")
    run.add_argument("--cuda-graph", choices=["auto", "true", "false"], default="auto")
    run.add_argument("--fixed-cell", action="store_true", help="Relax atomic positions only; auxiliary protocol")
    run.add_argument("--optimizer", choices=["FIRE", "BFGS"], default="FIRE")
    run.add_argument("--fmax", type=float, default=0.05)
    run.add_argument("--steps", type=int, default=500)
    run.add_argument("--model-kwargs", type=_json_object, default={})
    run.add_argument("--limit", type=int, help="Smoke run: process only this many IDs within the shard")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    return parser


def _summary_path(args):
    if args.summary is not None:
        return args.summary
    return download_verified(SUMMARY_URL, args.cache_dir / "2023-12-13-wbm-summary.csv.gz", SUMMARY_MD5)


def _extract(args, *, reuse=False):
    manifest = load_manifest(args.manifest)
    if reuse and args.output.exists():
        from ase.db import connect
        database = connect(str(args.output))
        metadata = database.metadata
        rows = list(database.select(columns=["id", "key_value_pairs"], include_data=False))
        if (metadata.get("selection_sha256") != manifest["selection_sha256"]
                or metadata.get("source_md5") != INITIAL_MD5
                or metadata.get("source_kind") != "wbm_initial_atoms"
                or len(rows) != manifest["sample_count"]
                or {row.get("material_id") for row in rows} != set(manifest["material_ids"])):
            raise ValueError("Existing structures do not match the verified initial subset; choose another output")
        return dict(structures=str(args.output), count=len(rows), reused=True,
                    selection_sha256=manifest["selection_sha256"])
    archive = args.initial_archive
    if archive is None:
        archive = download_verified(INITIAL_URL, args.cache_dir / "2024-08-04-wbm-initial-atoms.extxyz.zip", INITIAL_MD5)
    return extract_subset(args.manifest, archive, args.output)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            manifest = sample_manifest(_summary_path(args), population=args.population,
                                       fraction=args.fraction, seed=args.seed, expected_md5=SUMMARY_MD5)
            write_manifest(manifest, args.output)
            result = {key: manifest[key] for key in ("population", "population_count", "sample_count", "seed", "selection_sha256")}
            result["manifest"] = str(args.output)
        elif args.command in {"extract", "prepare"}:
            if args.command == "prepare" and not args.manifest.exists():
                manifest = sample_manifest(_summary_path(args), expected_md5=SUMMARY_MD5)
                write_manifest(manifest, args.manifest)
            result = _extract(args, reuse=args.command == "prepare")
        else:
            from fastmd.relaxation import RelaxationConfig
            from .run import run_selection
            result = run_selection(
                args.manifest, args.structures, args.output, args.model,
                checkpoint=args.checkpoint, device=args.device, cuda_graph=_graph_mode(args.cuda_graph),
                config=RelaxationConfig(relax_cell=not args.fixed_cell, optimizer=args.optimizer,
                                        fmax=args.fmax, steps=args.steps),
                model_kwargs=args.model_kwargs, limit=args.limit, shard_index=args.shard_index,
                num_shards=args.num_shards, resume=args.resume)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get("failed", 0) else 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"fastmd-discovery: {exc}", file=sys.stderr)
        return 2
