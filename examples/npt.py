"""MatRIS or MACE + ASE Berendsen pressure equilibration.

Berendsen coupling is useful for equilibration; it does not reproduce the exact
NPT fluctuations. The compressibility below is an illustrative silicon value.
"""
import argparse

import numpy as np
from ase import units
from ase.build import bulk
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from fastmd import FastMDCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("matris", "mace"), default="matris")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eager", action="store_true", help="Disable CUDA Graph (also needed for CPU)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=300, help="Temperature in kelvin")
    parser.add_argument("--pressure-gpa", type=float, default=0, help="Positive means compression")
    parser.add_argument("--bulk-modulus-gpa", type=float, default=100,
                        help="Inverse compressibility for the barostat; choose for your material")
    args = parser.parse_args()
    if args.bulk_modulus_gpa <= 0:
        parser.error("--bulk-modulus-gpa must be positive")

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    atoms.calc = FastMDCalculator(
        args.model, checkpoint=args.checkpoint, device=args.device,
        cuda_graph=not args.eager, model_kwargs={"compute_stress": True})
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature,
                                rng=np.random.default_rng(42))
    Stationary(atoms)
    # Includes stress even though warmup's default request is energy + forces.
    atoms.calc.warmup(atoms)
    with NPTBerendsen(
        atoms, timestep=units.fs, temperature_K=args.temperature,
        pressure_au=args.pressure_gpa * units.GPa,
        compressibility_au=1 / (args.bulk_modulus_gpa * units.GPa),
        taut=100 * units.fs, taup=1000 * units.fs,
        trajectory="npt.traj", logfile="npt.log", loginterval=10,
    ) as dynamics:
        dynamics.run(args.steps)
    print(atoms.calc.stats())


if __name__ == "__main__":
    main()
