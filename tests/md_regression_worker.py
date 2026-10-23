"""Subprocess worker: short NVE / Berendsen-NVT GPU-resident MD in the eager,
model_graph and whole_step paths with a deterministic analytic force field.
Saves the per-step state (positions, momenta, forces, energy) and the logged
trajectory frames, for bitwise comparisons between two source trees (run by
``test_gpu_nhc.NVEBerendsenRegressionTest``).

The MatRIS network itself is not bitwise reproducible run to run (atomic
scatter-adds; ~1e-4 A after 40 steps), so its forward is replaced by
periodic forces of the graph's fractional coordinates (elementwise, no
atomics): the integrators, graph builders, capture/replay and logging are
compared exactly.

Usage: md_regression_worker.py SOURCE_ROOT OUT.npz  (fastmd must be imported
from SOURCE_ROOT; this worker only uses APIs that predate ``nvt_nhc``).
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import ase.io
import numpy as np
import torch
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fastmd  # noqa: E402
from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics  # noqa: E402
from fastmd.config import InferenceConfig  # noqa: E402

from test_isolated_atom_handling import make_model  # noqa: E402

N_STEPS = 40


def system() -> Atoms:
    rng = np.random.default_rng(123)
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    atoms = Atoms(
        "H32C32", positions=grid * 1.3 + rng.uniform(-0.1, 0.1, (64, 3)),
        cell=[5.2] * 3, pbc=True,
    )
    atoms.set_momenta(rng.normal(0.0, 0.3, (64, 3)))
    return atoms


def deterministic_forces(model) -> None:
    phase = torch.tensor(
        np.random.default_rng(7).uniform(0.0, 1.0, (4096, 3)),
        dtype=torch.float32, device="cuda",
    )

    def forward(graphs, task="ef", is_training=False, n_real=None, **_):
        frac = graphs[0].atom_frac_coord
        arg = 2 * math.pi * (frac - phase[: frac.shape[0]])
        energy = (0.8 / (2 * math.pi) * torch.cos(arg)).sum().reshape(1)
        return {"e": energy, "f": [0.8 * torch.sin(arg)]}

    model.forward = forward


def main(root: str, out: str) -> None:
    source = Path(fastmd.__file__).resolve()
    assert source.is_relative_to(Path(root).resolve()), (source, root)
    torch.manual_seed(0)
    model = make_model(num_layers=1).cuda()
    deterministic_forces(model)
    result = {}
    for ensemble in ("nve", "nvt"):
        for execution in ("eager", "model_graph", "whole_step"):
            with tempfile.TemporaryDirectory() as tmp:
                trajectory = Path(tmp) / "md.traj"
                md = GPUResidentMolecularDynamics(
                    system(), model=model, task="ef", ensemble=ensemble,
                    temperature=500.0, timestep=0.5, taut=20.0,
                    trajectory=str(trajectory), loginterval=5,
                    config=InferenceConfig(execution=execution, report=False),
                )
                states = []
                try:
                    for _ in range(N_STEPS):
                        state = md.step()
                        states.append(
                            torch.cat([
                                state.positions.reshape(-1), state.momenta.reshape(-1),
                                state.forces.reshape(-1),
                                state.potential_energy.reshape(-1).float(),
                            ]).detach().cpu().numpy()
                        )
                finally:
                    md.close()
                frames = ase.io.read(trajectory, ":")
            key = f"{ensemble}_{execution}"
            result[key] = np.array(states)
            result[key + "_trajectory"] = np.array([
                np.concatenate([
                    frame.get_positions().ravel(), frame.get_momenta().ravel(),
                    frame.get_forces().ravel(), [frame.get_potential_energy()],
                ])
                for frame in frames
            ])
    np.savez(out, **result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
