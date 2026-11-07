"""Compare MD's migrated StructOptimizer ASE/GPU paths on the unrelaxed WBM fixture.

The default is a five-step numerical smoke comparison, with fixed and variable
cells. ``--full`` changes the default step limit to 500; it does not assert that
either run converges. One MACE checkpoint is loaded once and its adapter is reused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter


REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE = REPOSITORY / "tests/data/wbm-1-12.initial.extxyz"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="local MACE .model checkpoint")
    parser.add_argument("--device", default="cuda", help="CUDA device, for example cuda:0")
    parser.add_argument("--model-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--profile", choices=("topology", "generic"), default="topology")
    parser.add_argument("--steps", type=int, help="maximum FIRE steps (default: 5, or 500 with --full)")
    parser.add_argument("--full", action="store_true", help="use a 500-step limit unless --steps is supplied")
    parser.add_argument("--fmax", type=float, default=0.05, help="ASE optimizer force tolerance, eV/Angstrom")
    parser.add_argument("--output", type=Path, default=Path("wbm-md-relax-comparison.json"))
    parser.add_argument("--energy-atol", type=float, default=1e-3, help="total energy tolerance, eV")
    parser.add_argument("--force-atol", type=float, default=1e-3, help="largest force-component error, eV/Angstrom")
    parser.add_argument("--position-atol", type=float, default=1e-3, help="largest Cartesian-coordinate error, Angstrom")
    parser.add_argument("--cell-atol", type=float, default=1e-3, help="largest cell-matrix error, Angstrom")
    return parser


def _run(adapter, initial, *, engine, relax_cell, args):
    import numpy as np
    import torch
    from fastmd import CapacityConfig, InferenceConfig, StructOptimizer

    atoms = initial.copy()
    config = InferenceConfig(
        execution="eager" if engine == "ase" else "auto",
        optimization_profile=args.profile, checkpoint="off", report=False,
        topology_reuse="off" if engine == "ase" else "auto",
        capacity=CapacityConfig(warmup=1, candidate_skin=1.0), matmul_precision="highest",
    )
    torch.cuda.synchronize(args.device)
    started = perf_counter()
    with StructOptimizer(model=adapter, device=args.device, task="efs", optimizer="FIRE",
                         enable_compile=False, config=config, engine=engine) as optimizer:
        if optimizer.calculator.adapter is not adapter:
            raise RuntimeError("The comparison must reuse the same loaded adapter")
        torch.cuda.synchronize(args.device)
        constructed = perf_counter()
        result = optimizer.relax(atoms, fmax=args.fmax, steps=args.steps, relax_cell=relax_cell,
                                 loginterval=1, verbose=False, assign_magmoms=False, engine=engine,
                                 logfile=None)
        torch.cuda.synchronize(args.device)
        finished = perf_counter()
        last = deepcopy(optimizer.last_relaxation)
        if last["path"] != engine:
            raise RuntimeError(f"Requested engine={engine!r}, observed {last!r}")
        observer = result["trajectory"]
        # MD records every FIRE step, including step zero, then appends the terminal frame.
        steps = len(observer.energies) - 2
        if not 0 <= steps <= args.steps:
            raise RuntimeError(f"Unexpected trajectory length {len(observer.energies)} for {args.steps} steps")
        final = {
            "energy_eV": float(observer.energies[-1]),
            "forces_eV_per_A": np.asarray(observer.forces[-1], dtype=float).copy(),
            "positions_A": np.asarray(observer.atom_positions[-1], dtype=float).copy(),
            "cell_A": np.asarray(observer.cells[-1], dtype=float).copy(),
            "stress_eV_per_A3": np.asarray(observer.stresses[-1], dtype=float).copy(),
        }
        if not all(np.isfinite(value).all() for value in final.values()):
            raise RuntimeError(f"{engine} produced a non-finite terminal property")
        return {
            "engine": engine, "last_relaxation": last,
            "steps": steps, "step_count_source": "loginterval=1 trajectory, excluding initial and appended final frames",
            "stop": "threshold_reached_before_limit" if steps < args.steps else "limit_reached_or_threshold_at_limit",
            "initial_energy_eV": float(observer.energies[0]),
            "max_atomic_force_eV_per_A": float(np.linalg.norm(final["forces_eV_per_A"], axis=1).max()),
            "seconds": {
                "total_including_construction": finished - started,
                "interface_construction": constructed - started,
                "relax_call_including_gpu_construction": finished - constructed,
                "gpu_engine_construction": last.get("seconds", {}).get("construction"),
                "gpu_relaxation_and_collection": last.get("seconds", {}).get("relaxation"),
            },
            "final": {key: value.tolist() if hasattr(value, "tolist") else value for key, value in final.items()},
        }


def _compare(ase, gpu, args):
    import numpy as np

    fields = {
        "energy_eV": args.energy_atol,
        "forces_eV_per_A": args.force_atol,
        "positions_A": args.position_atol,
        "cell_A": args.cell_atol,
    }
    errors = {name: float(np.max(np.abs(np.asarray(gpu["final"][name]) - np.asarray(ase["final"][name]))))
              for name in fields}
    return {
        "max_absolute_errors": errors, "absolute_tolerances": fields, "relative_tolerance": 0.0,
        "within_tolerance": {name: error <= fields[name] for name, error in errors.items()},
        "passed": all(error <= fields[name] for name, error in errors.items()),
        "same_step_count": ase["steps"] == gpu["steps"],
        "initial_energy_absolute_difference_eV": abs(ase["initial_energy_eV"] - gpu["initial_energy_eV"]),
        "stress_max_absolute_difference_eV_per_A3": float(np.max(np.abs(
            np.asarray(gpu["final"]["stress_eV_per_A3"]) - np.asarray(ase["final"]["stress_eV_per_A3"])))),
    }


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    args.steps = (500 if args.full else 5) if args.steps is None else args.steps
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    for name in ("fmax", "energy_atol", "force_atol", "position_atol", "cell_atol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not args.model_path.is_file():
        parser.error("--model-path must name an existing local MACE checkpoint")
    if args.output.resolve() in (args.model_path.resolve(), FIXTURE.resolve()):
        parser.error("--output must differ from the checkpoint and input structure")

    # --help and argument validation work without importing torch, ASE, or an optional model package.
    sys.path.insert(0, str(REPOSITORY))
    import numpy as np
    import torch
    from ase.io import read
    from fastmd import load_adapter

    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("this strict ASE/GPU comparison requires an available CUDA device")
    initial = read(FIXTURE)
    positions, cell = initial.positions.copy(), initial.cell.array.copy()
    report = {
        "fixture": str(FIXTURE), "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "material_id": "wbm-1-12", "formula": initial.get_chemical_formula(), "atoms": len(initial),
        "checkpoint": str(args.model_path.resolve()), "model_dtype": args.model_dtype,
        "device": args.device, "device_name": torch.cuda.get_device_name(args.device),
        "profile": args.profile, "max_steps": args.steps, "fmax_eV_per_A": args.fmax,
        "mode": "extended_comparison" if args.full else "short_numerical_comparison",
        "convergence_claim": "No convergence assertion; --full increases the step limit only.",
        "timing_scope": "One loaded adapter; per-run times include interface and GPU graph construction, exclude model loading. "
                        "Runs execute ASE then GPU for each cell mode in one process; timings are not an unbiased benchmark.",
        "comparisons": {},
    }
    adapter = None
    try:
        adapter = load_adapter("mace", model_path=str(args.model_path), device=args.device,
                               model_dtype=args.model_dtype, optimization_profile=args.profile)
        torch.cuda.synchronize(args.device)
        for relax_cell in (False, True):
            name = "variable_cell" if relax_cell else "fixed_cell"
            runs = {engine: _run(adapter, initial, engine=engine, relax_cell=relax_cell, args=args)
                    for engine in ("ase", "gpu")}
            comparison = _compare(runs["ase"], runs["gpu"], args)
            report["comparisons"][name] = {**runs, "comparison": comparison}
            print(f"{name}: ASE {runs['ase']['steps']} steps, GPU {runs['gpu']['steps']} steps; "
                  f"numerical_match={comparison['passed']}; errors={comparison['max_absolute_errors']}")
        np.testing.assert_array_equal(initial.positions, positions)
        np.testing.assert_array_equal(initial.cell.array, cell)
        report["passed"] = all(value["comparison"]["passed"] for value in report["comparisons"].values())
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if close is not None:
                close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
