"""Generate a small periodic input, or evaluate it with explicitly selected local weights.

Run ``python examples/quickstart.py --help`` from the checkout. Model evaluation
requires the selected family's environment. No model is substituted when a
package is absent; model/capture failures propagate to the command's exit code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs", help="write a small periodic silicon cell without loading a model")
    inputs.add_argument("--output", type=Path, required=True)
    calculator = commands.add_parser("calculator", help="evaluate energy, forces and stress")
    calculator.add_argument("--structure", type=Path, required=True)
    calculator.add_argument("--model", required=True, help="model name or family accepted by FastMD")
    calculator.add_argument("--model-path", type=Path, required=True, help="existing local checkpoint; no named download")
    calculator.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    calculator.add_argument("--execution", choices=("eager", "model_graph", "auto"), default="eager")
    calculator.add_argument("--profile", choices=("generic", "topology"), default="generic")
    args = parser.parse_args(argv)
    if args.command == "inputs":
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
        from ase.build import bulk
        from ase.io import write

        atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
        atoms.rattle(0.01, seed=0)
        write(args.output, atoms, format="extxyz")
        print(f"Wrote {len(atoms)} atoms to {args.output}")
        return 0
    if not args.model_path.is_file():
        parser.error(f"local checkpoint does not exist: {args.model_path}")
    if not args.structure.is_file():
        parser.error(f"input structure does not exist: {args.structure}")
    from ase.io import read
    import numpy as np
    from fastmd import InferenceConfig, MLIPCalculator

    atoms = read(args.structure)
    config = InferenceConfig(execution=args.execution, optimization_profile=args.profile)
    with MLIPCalculator(model=args.model, model_path=str(args.model_path), device=args.device,
                        task="efs", config=config) as calc:
        atoms.calc = calc
        energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        if not all(np.isfinite(value).all() for value in (energy, forces, stress)):
            raise FloatingPointError("model evaluation produced a non-finite energy, force or stress")
        print(json.dumps({"natoms": len(atoms), "energy_eV": float(energy),
                          "forces_eV_per_A": forces.tolist(), "stress_eV_per_A3": stress.tolist(),
                          "configuration": calc.config_summary()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
