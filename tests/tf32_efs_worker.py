"""Subprocess worker: MatRIS-10M-OAM E/F/S on the S01 frames under the TF32
default of this process. ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`` is read by torch
at start-up and sets the default of ``torch.backends.cuda.matmul.allow_tf32``
(1: True, unset/0: False); the worker keeps that default, i.e. it evaluates
exactly what an MD script started under the same environment computes.

Two evaluation paths per frame: the ASE calculator (dynamic GPU graph builder)
and the fixed-capacity builder used by the whole-step CUDA graph.
Usage: tf32_efs_worker.py OUT.npz
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastmd.interfaces.calculator import MLIPCalculator  # noqa: E402
from fastmd.graph import FixedCapacityGraphBuilder  # noqa: E402

from matris_test_assets import CHECKPOINT, load_s01_frames  # noqa: E402


def main(out: str) -> None:
    frames, _ = load_s01_frames()
    calc = MLIPCalculator(model_path=str(CHECKPOINT), device="cuda")
    model = calc.model
    n = len(frames[0])
    result = {key: [] for key in ("e", "f", "s", "e_fixed", "f_fixed", "s_fixed",
                                  "counts", "position_error")}
    for atoms in frames:
        atoms = atoms.copy()
        atoms.calc = calc
        result["e"].append(atoms.get_potential_energy())
        result["f"].append(atoms.get_forces())
        result["s"].append(atoms.get_stress(voigt=False))

        cell64 = atoms.cell.array
        builder = FixedCapacityGraphBuilder(
            cell=torch.tensor(cell64, dtype=torch.float32),
            atomic_numbers=torch.tensor(atoms.numbers, dtype=torch.int32),
            u_capacity=1536,
            t_capacity=24576,
            atom_graph_cutoff=model.graph_converter.atom_graph_cutoff,
            line_graph_cutoff=model.graph_converter.line_graph_cutoff,
            candidate_cutoff=model.graph_converter.atom_graph_cutoff + 2.0,
            device="cuda",
        )
        positions = torch.tensor(atoms.positions, dtype=torch.float32, device="cuda")
        graph, status = builder.build(positions)
        assert not bool(status.overflow.item())
        output = model([graph], task="efs", is_training=False, n_real=n)
        forces = output["f"][0] if isinstance(output["f"], (list, tuple)) else output["f"]
        stress = output["s"]
        stress = stress[0] if isinstance(stress, (list, tuple)) else stress.reshape(-1, 3, 3)[0]
        result["e_fixed"].append(float(output["e"].reshape(-1)[0].item()) * n)
        result["f_fixed"].append(forces[:n].detach().cpu().numpy())
        result["s_fixed"].append(stress.detach().cpu().numpy())  # GPa
        result["counts"].append(
            (int(status.edge_count.item()), int(status.triplet_count.item()))
        )
        # fp32 input positions -> builder fractional coordinates, fp64 comparison
        pos32 = atoms.positions.astype(np.float32).astype(np.float64)
        frac64 = pos32 @ np.linalg.inv(cell64.astype(np.float32).astype(np.float64))
        delta = builder.frac_coords[:n].double().cpu().numpy() - frac64
        delta -= np.round(delta)
        result["position_error"].append(float(np.abs(delta @ cell64).max()))
    np.savez(
        out,
        tf32_override=os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", ""),
        allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        matris_file=__import__("fastmd").__file__,
        **{key: np.asarray(value) for key, value in result.items()},
    )


if __name__ == "__main__":
    main(sys.argv[1])
