"""GPU-resident single-system MatRIS E/F baseline for FastMD.

Rebuilds the graph on every call. No ASE result cache, fixed-topology replay-only
timing, thermalization, or COM removal. Run each configuration in a fresh process.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def initial_snapshot(evaluator, positions, force, energy):
    """Serialize the evaluated state outside timing, with independent JSON-compatible arrays.

    Read the actual FP32 device state rather than the higher-precision ASE
    input. Python lists own the snapshot even if later MD mutates a tensor.
    """
    numbers = evaluator.atomic_numbers
    frame = {"ids": list(range(len(positions))),
             "numbers": numbers.detach().cpu().tolist(),
             "positions": positions.detach().cpu().tolist(),
             "masses": evaluator.masses.detach().cpu().tolist(),
             "momenta": evaluator.state.momenta.detach().cpu().tolist(),
             "forces": force.detach().cpu().tolist()}
    reference = evaluator.model.reference_energy
    reference_energy = (0.0 if reference is None else
                        float(reference.fc.weight[0, numbers.long() - 1].sum().detach()))
    total_energy = float(energy.detach())
    return frame, {"energy_eV": total_energy,
                   "energy_eV_per_atom": total_energy / len(positions),
                   "reference_energy_eV": reference_energy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", choices=("eager", "model_graph"), default="model_graph")
    parser.add_argument("--profile", choices=("generic", "legacy", "topology"), default="topology")
    parser.add_argument("--precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--md-steps", type=int, default=0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--precision-force-rmse-tolerance", type=float,
                        help="also gate the oracle's force RMSE against 'highest' (eV/A); reported only by default")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    if args.threads < 1 or args.steps < 1 or min(args.warmup, args.md_steps) < 0:
        parser.error("threads/steps must be positive; warmup/md-steps nonnegative")
    if not math.isfinite(args.dt_fs) or args.dt_fs <= 0:
        parser.error("dt-fs must be finite and positive")
    tolerance = args.precision_force_rmse_tolerance
    if tolerance is not None and not (math.isfinite(tolerance) and tolerance >= 0):
        parser.error("precision-force-rmse-tolerance must be finite and nonnegative")
    import numpy as np
    import torch
    from ase.io import read

    from fastmd.execution.gpu_integrator import GPUIntegrator
    from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
    from fastmd.config import InferenceConfig, resolve_config
    from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu
    from fastmd.models.matris.model import MatRIS
    from fastmd.precision import compute_references, oracle_report, precision_record, warn_implicit_tf32

    if not torch.cuda.is_available():
        parser.error("a CUDA device is required; select one with CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() != 1:
        parser.error("select exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision(args.precision)
    warn_implicit_tf32(args.precision, "benchmark_single_ef")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    atoms = read(args.input)
    if not len(atoms) or not all(atoms.pbc):
        parser.error("requires a nonempty, fully periodic system")
    started = time.perf_counter()
    model = MatRIS.load(model_path=args.model, device=str(device),
                        enable_compile=False, enable_checkpoint=False).eval().requires_grad_(False)
    # Compute and release the original generic oracle before preparing the
    # optimized runner, which may install fused projection caches on the model.
    # References use 'highest' GEMMs, plus the run precision if that is TF32.
    generic = resolve_config(InferenceConfig(execution="eager", optimization_profile="generic",
                                            checkpoint="off", report=False))

    def predict():
        graph = atoms_to_graph_gpu(atoms, device=device,
                                   atom_graph_cutoff=model.config["pairwise_cutoff"],
                                   line_graph_cutoff=model.config["three_body_cutoff"],
                                   check_isolated_atoms=False)
        original = model([graph], task="ef", is_training=False, handle_isolated_atoms=True)
        return float(original["e"][0].detach()), original["f"][0].detach().cpu()

    with generic.scope():
        references = compute_references(predict, args.precision)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    oracle_and_load_s = time.perf_counter() - started
    started = time.perf_counter()
    evaluator = GPUResidentMolecularDynamics(
        atoms, model=model, task="ef", device=device, ensemble="nve", timestep=args.dt_fs,
        config=InferenceConfig(execution=args.execution, optimization_profile=args.profile,
                               checkpoint="off", isolated_atoms="reference", report=False,
                               matmul_precision=args.precision))
    positions = evaluator.state.positions

    def evaluate(pos):
        force, energy = evaluator.evaluate(pos)
        # Return independent snapshots, including for graph-captured predictions.
        # This cost belongs inside timing, including for static graph outputs.
        return force.clone(), energy.clone()

    force, energy = evaluate(positions)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - started
    # Snapshot only after the cold-start timer stops, and before any warmup,
    # fixed-frame samples or MD. No diagnostics enter their timing/memory scope.
    initial_frame, initial_energy = initial_snapshot(evaluator, positions, force, energy)
    oracle = oracle_report(float(energy) / len(atoms), force.cpu(), references,
                           precision_force_rmse_tolerance=tolerance)
    if not oracle["passed"]:
        raise AssertionError(oracle)
    del references
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    for _ in range(args.warmup):
        force, energy = evaluate(positions)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    captures_before_timing = evaluator.runner.captures if evaluator.runner is not None else 0
    samples = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        before = time.perf_counter()
        force, energy = evaluate(positions)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - before) * 1000)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    captures_after_timing = evaluator.runner.captures if evaluator.runner is not None else 0
    md = None
    if args.md_steps:
        integrator = GPUIntegrator(evaluator.masses, args.dt_fs, ensemble="nve", fix_com=False)
        momenta = evaluator.state.momenta.clone()
        initial_e = energy.clone()
        initial_k = (0.5 * momenta.square() / evaluator.masses[:, None]).sum()
        md_samples = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        def force_fn(pos):
            nonlocal energy
            result, energy = evaluate(pos)
            return result

        for _ in range(args.md_steps):
            torch.cuda.synchronize()
            before = time.perf_counter()
            positions, momenta, force = integrator.step(positions, momenta, force, force_fn)
            torch.cuda.synchronize()
            md_samples.append((time.perf_counter() - before) * 1000)
        final_k = (0.5 * momenta.square() / evaluator.masses[:, None]).sum()
        md = {"ensemble": "nve", "fix_com": False, "steps": args.md_steps,
              "dt_fs": args.dt_fs, "samples_ms": md_samples,
              "initial_total_energy_eV": float(initial_e + initial_k),
              "final_total_energy_eV": float(energy + final_k),
              "positions": positions.cpu().tolist(), "momenta": momenta.cpu().tolist(),
              "forces": force.cpu().tolist(), "potential_energy_eV": float(energy)}
        md.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  captures_before=captures_after_timing,
                  captures_after=evaluator.runner.captures if evaluator.runner is not None else 0)
    configuration = evaluator.config_summary()
    report = {"world_size": 1, "natoms": len(atoms), "torch": torch.__version__,
              "cuda": torch.version.cuda, "precision": args.precision, "threads": args.threads,
              "precision_record": precision_record(args.precision, device),
              "model": str(Path(args.model).resolve()), "input": str(Path(args.input).resolve()),
              "oracle_and_load_seconds": oracle_and_load_s,
              "initialization_and_first_EF_seconds": cold_seconds,
              "initial_energy": initial_energy, "initial_frame": initial_frame,
              "configuration": configuration, "oracle": oracle,
              "warmup": args.warmup, "samples_ms": samples, "median_ms": statistics.median(samples),
              "p95_ms": float(np.percentile(samples, 95)),
              "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
              "peak_memory_scope": "post-warmup fixed-frame E/F only; MD has separate peaks; excludes cold-start peak",
              "fixed_frame_captures_before": captures_before_timing,
              "fixed_frame_captures_after": captures_after_timing,
              "captures": evaluator.runner.captures if evaluator.runner is not None else 0,
              "memory_protocol": "oracle freed before runner; GC/sync/empty_cache before warmup; peak reset after warmup",
              "timing_protocol": "GPU-resident positions, graph rebuild + E/F + output snapshots + device sync; no I/O or oracle in timing",
              "md": md}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in
                      ("natoms", "median_ms", "p95_ms", "oracle", "captures")}), flush=True)


if __name__ == "__main__":
    main()
