"""Compare eager, capture, and fusion on identical frames, including ASE transfers."""
import argparse
import gc
import json
import platform
import time

import numpy as np
import torch
from ase.build import bulk
from ase.io import read
from fastmd import CUDAGraphConfig, FastMDCalculator, __version__


def evaluate(calc, frames, properties):
    torch.cuda.synchronize(calc.backend.device)
    start = time.perf_counter()
    values = []
    for atoms in frames:
        # Explicit calculate avoids ASE reusing an identical frame's result.
        calc.calculate(atoms, properties)
        values.append({key: np.array(calc.results[key], copy=True) for key in properties})
    torch.cuda.synchronize(calc.backend.device)
    return values, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="chgnet")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--stress", action="store_true", help="Also validate/time stress (MatRIS/MACE)")
    parser.add_argument("--compile", action="store_true", help="Add compiled MatRIS/MACE variant")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64", help="MACE only")
    parser.add_argument("--cueq", action="store_true", help="MACE: apply cuEquivariance to every variant")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        parser.error("This comparison requires an available CUDA GPU")
    if args.frames < 1:
        parser.error("--frames must be positive")
    if (args.stress or args.compile) and args.model not in {"matris", "mace"}:
        parser.error("--stress and --compile are supported here only for MatRIS/MACE")
    atoms = read(args.input) if args.input else bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(args.frames):
        frame = atoms.copy()
        frame.positions += rng.normal(0, .01, frame.positions.shape)
        frames.append(frame)
    properties = ["energy", "forces"] + (["stress"] if args.stress else [])
    variants = [("eager", False, False), ("graph_plain", True, False), ("graph_fused", True, True)]
    if args.compile:
        variants.append(("graph_compiled", True, True))
    report = dict(fastmd=__version__, torch=torch.__version__, python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(args.device), model=args.model,
                  checkpoint=args.checkpoint, atoms=len(atoms), frames=len(frames),
                  properties=properties, dtype=args.dtype if args.model == "mace" else "float32",
                  cueq=args.cueq if args.model == "mace" else False, variants={})
    reference = None
    # Release each model and graph pool before loading the next variant.
    for mode, enabled, fusions in variants:
        kwargs = {}
        if args.model == "mace":
            kwargs = dict(default_dtype=args.dtype, enable_cueq=args.cueq,
                          variant="fast_cm" if mode == "graph_compiled" else "fast")
        elif args.model == "matris":
            kwargs = dict(compile_lowerings=mode == "graph_compiled")
        start = time.perf_counter()
        calc = FastMDCalculator(args.model, checkpoint=args.checkpoint, device=args.device,
                                 cuda_graph=CUDAGraphConfig(enabled=enabled, enable_fusions=fusions),
                                 model_kwargs=kwargs)
        load = time.perf_counter()-start
        _, setup = evaluate(calc, frames, properties)
        predictions, elapsed = evaluate(calc, frames, properties)
        errors = {key: 0.0 for key in properties}
        if reference is None:
            reference = predictions
            eager_time = elapsed
        else:
            for actual, expected in zip(predictions, reference):
                for key in properties:
                    # Defaults account for FP32 MatRIS; use tighter MACE FP64 checks.
                    atol, rtol = ((2e-6, 2e-5) if args.model == "mace" and args.dtype == "float64"
                                  else (3e-3, 1e-3))
                    np.testing.assert_allclose(actual[key], expected[key], atol=atol, rtol=rtol,
                                               err_msg=f"{mode}: {key}")
                    errors[key] = max(errors[key], float(np.max(np.abs(actual[key]-expected[key]))))
        row = dict(load_seconds=load, setup_seconds=setup, milliseconds_per_frame=elapsed*1000/len(frames),
                   speedup_vs_eager=eager_time/elapsed, max_absolute_error=errors, stats=calc.stats())
        report["variants"][mode] = row
        print(f"{mode}: load={load:.2f}s setup={setup:.2f}s, {row['milliseconds_per_frame']:.3f} ms/frame, "
              f"{row['speedup_vs_eager']:.2f}x vs eager; max errors={errors}")
        calc.clear_cache()
        del calc
        gc.collect()
        torch.cuda.empty_cache()
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print("PASS: all measured variants agree within the declared tolerances.")


if __name__ == "__main__":
    main()
