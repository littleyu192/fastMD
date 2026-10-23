#!/usr/bin/env python3
"""Time a model through GPU-resident MD: ms/step in fresh processes, with the precision state.

Runs ``fastmd.interfaces.gpu_md.GPUResidentMolecularDynamics`` with any model that
``fastmd.adapters.load_adapter`` resolves (MatRIS, MACE-MP-0, MACE-MPA-0, SevenNet-0, ORB v3, CHGNet 0.3.0, a
checkpoint) on one GPU, in the execution modes eager, model_graph and whole_step (or auto). Every repeat of every mode
is a fresh process
(this script with ``--child``): the parent interleaves the modes (reversed on even repeats), keeps each
process's JSON and log in ``--out`` and reports the median, min and max of the per-process ms/step
(``summary.json`` and a table).

One process: momenta from the Maxwell-Boltzmann distribution at ``--temperature`` (seeded), the model load and
the engine's construction, ``--warmup`` untimed steps (captures happen there), then ``gc.collect()``,
``gc.freeze()``, ``gc.disable()`` and ``--steps`` timed steps in five near-equal chunks with a CUDA synchronize
after each (the splits). ms/step is the wall time from the synchronize before the first chunk to the one after
the last, over ``--steps``. ``--dtype float32`` sets IEEE FP32 math in cuBLAS and cuDNN itself (``allow_tf32``
off, ``highest``); a run during which TF32 got enabled ends with status ``precision_changed``.

Usage (from the checkout root, ``PYTHONPATH=.``)::

    python -B benchmarks/multimodel/run.py --model mace_mp_0 --dtype float32 --structure rdx_1344_1x1x1 \\
        --mode whole_step --ensemble nvt_nhc --repeats 3 --out DIR
    python -B benchmarks/multimodel/run.py --model matris --checkpoint CKPT --structure frames.npz \\
        --mode eager model_graph whole_step --temperature 500 --timestep 0.25 --out DIR
    python -B benchmarks/multimodel/run.py --model sevennet_0 --structure rdx_1344_1x1x1 --mode whole_step eager \\
        --kernel cueq_fused_tp --repeats 3 --out DIR

``--kernel OPTION`` keeps one option of the model's kernel site (an A/B of kernels the negotiation would rank
otherwise): the adapter's kernel choice is narrowed to that option (and its reference) before the engine is
constructed, so the negotiation installs it in every execution mode.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import socket
import statistics
import subprocess
import sys
import time
import traceback

MODES = ("eager", "model_graph", "whole_step", "auto")
STRUCTURE_SUFFIXES = (".extxyz", ".xyz", ".npz", ".h5")
# Array names of an .npz/.h5 structure, first match wins; each may have a leading frame axis.
ARRAY_KEYS = {"numbers": ("numbers", "atomic_numbers", "Z"), "positions": ("positions", "pos", "R"),
              "cell": ("cell", "lattice"), "pbc": ("pbc",)}
# Removed from the child's environment: one process on one GPU (no launcher ranks); MATRIS_STRUCTURES is resolved
# by the parent (matris's configuration warns about MATRIS_* names it does not register).
DROPPED_ENVIRONMENT = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                       "MATRIS_STRUCTURES")
PACKAGES = ("mace-torch", "sevenn", "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
            "cuequivariance-ops-torch-cu13", "e3nn", "orb-models", "chgnet", "nvalchemi-toolkit-ops", "warp-lang",
            "triton", "ase", "numpy")
# Engine statistics that count events: their change over the timed window is reported, and the one-off events
# (captures, capacity growth, fallbacks), which the warm-up should have absorbed, are warned about.
ONE_OFF_EVENTS = ("captures", "capacity_recaptures", "pool_resets", "shrinks", "fallback_windows", "recovery_replays")
COUNTERS = ("launches", "candidate_refreshes", "warmup_runs") + ONE_OFF_EVENTS
TF32_OVERRIDE = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="mace_mp_0, mace_mpa_0, a MACE checkpoint path, sevennet_0, "
                        "orb_v3, chgnet_0_3_0 (CHGNet 0.3.0, float32), or matris (the default MatRIS model; a MatRIS "
                        "name works too, MatRIS checkpoints go in --checkpoint)")
    parser.add_argument("--checkpoint", help="a checkpoint file in place of the named model's (model_path)")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                        help="model dtype (MatRIS, CHGNet: float32); float32 runs IEEE FP32 cuBLAS/cuDNN math")
    parser.add_argument("--profile", choices=("topology", "generic"),
                        help="optimization profile (None: topology); generic runs the reference kernels")
    parser.add_argument("--kernel", help="keep this option of the model's kernel site in every execution mode "
                        "(e.g. SevenNet's cueq_fused_tp); None: the negotiated kernels")
    parser.add_argument("--matmul-precision", choices=("highest", "high"),
                        help="InferenceConfig.matmul_precision of the engine's calls (None: the process state, "
                             "which --dtype float32 sets to IEEE FP32); high runs float32 GEMMs in TF32 (an opt-in)")
    parser.add_argument("--structure", required=True,
                        help="a structure file (ASE-readable, or .npz/.h5 arrays) or a name in --structures-dir")
    parser.add_argument("--structures-dir", help="where structure names resolve (None: $MATRIS_STRUCTURES)")
    parser.add_argument("--frame", type=int, default=0, help="frame of a multi-frame structure file")
    parser.add_argument("--mode", nargs="+", choices=MODES, default=["whole_step"],
                        help="execution modes (InferenceConfig.execution), interleaved per repeat")
    parser.add_argument("--ensemble", choices=("nvt_nhc", "nvt", "nve"), default="nvt_nhc")
    parser.add_argument("--temperature", type=float, default=300.0, help="K: initial momenta and thermostat")
    parser.add_argument("--timestep", type=float, default=1.0, help="fs (the 135-atom CsSnI3 frame: 0.25)")
    parser.add_argument("--steps", type=int, default=1000, help="timed steps per process")
    parser.add_argument("--warmup", type=int, default=100, help="untimed steps before the timed ones")
    parser.add_argument("--capacity-step", type=int, default=0,
                        help="capacity tier step in undirected pairs (CapacityConfig u_step, and min_pad_u unless "
                             "--capacity-pad; 0: the defaults 512 and 256, or the model's declared step); 32 (64-edge "
                             "tiers) suits small edges-only systems")
    parser.add_argument("--capacity-pad", type=int,
                        help="minimum slack of a pair tier (CapacityConfig min_pad_u), with --capacity-step "
                             "(None: --capacity-step)")
    parser.add_argument("--triplet-capacity-step", type=int, default=0,
                        help="capacity tier step in triplets, for line-graph models (CapacityConfig t_step, and "
                             "min_pad_t unless --triplet-capacity-pad; 0: the defaults 8192 and 4096); CHGNet on "
                             "the 135-atom CsSnI3 frame: 1024, pad 512, with pair step 128, pad 256")
    parser.add_argument("--triplet-capacity-pad", type=int,
                        help="minimum slack of a triplet tier (CapacityConfig min_pad_t; None: "
                             "--triplet-capacity-step, or the default without it)")
    parser.add_argument("--precapture-steps", type=int, default=0,
                        help="whole_step: steps run and rolled back at construction, so that capacity growth "
                             "is captured before the warm-up")
    parser.add_argument("--steps-per-launch", type=int, default=1,
                        help="steps per step() call (one whole-step replay); divides --steps and --warmup")
    parser.add_argument("--repeats", type=int, default=3, help="fresh processes per mode")
    parser.add_argument("--seed", type=int, default=0, help="seed of the Maxwell-Boltzmann momenta")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per process")
    parser.add_argument("--out", required=True, help="output directory (fresh: refuses an existing summary)")
    parser.add_argument("--child", help=argparse.SUPPRESS)  # internal: measure one mode, write the JSON here
    args = parser.parse_args(argv)
    if args.child is None:
        launch = args.steps_per_launch
        if launch < 1 or args.steps < 1 or args.warmup < 0 or args.repeats < 1 or args.steps % launch \
                or args.warmup % launch:
            parser.error("--steps (>= 1) and --warmup (>= 0) must be multiples of --steps-per-launch (>= 1); "
                         "--repeats >= 1")
        if any(value is not None and value < 0 for value in (args.capacity_step, args.capacity_pad,
                                                             args.triplet_capacity_step, args.triplet_capacity_pad)):
            parser.error("--capacity-step, --capacity-pad, --triplet-capacity-step and --triplet-capacity-pad "
                         "must be >= 0")
        if args.capacity_pad is not None and not args.capacity_step:
            # (a pad alone would keep the default step and drop a model's declared capacity step)
            parser.error("--capacity-pad needs --capacity-step")
        if os.sep not in args.model and args.model.lower().startswith("matris") and args.dtype != "float32":
            parser.error("MatRIS runs float32 on one GPU: --dtype float32")
        if args.dtype == "float64" and args.ensemble != "nvt_nhc":
            parser.error("float64 models run --ensemble nvt_nhc (the integrator with a float64 state)")
        args.mode = list(dict.fromkeys(args.mode))
        args.structure = str(resolve_structure(parser, args.structure, args.structures_dir))
    return args


def resolve_structure(parser, spec: str, directory: str | None) -> Path:
    """``spec`` if it is a file, else ``<directory>/<spec><suffix>`` for the first suffix that exists."""
    if Path(spec).is_file():
        return Path(spec).absolute()
    directory = directory or os.environ.get("MATRIS_STRUCTURES")
    if not directory:
        parser.error(f"--structure {spec!r} is not a file; a name resolves in --structures-dir or "
                     "$MATRIS_STRUCTURES, and neither is set")
    for suffix in STRUCTURE_SUFFIXES:
        if (path := Path(directory, spec + suffix)).is_file():
            return path.absolute()
    parser.error(f"--structure {spec!r}: no {spec}{{{','.join(STRUCTURE_SUFFIXES)}}} in {directory}")


def read_structure(path: str, frame: int = 0):
    """ASE atoms of frame ``frame`` of a structure file, and what an .npz/.h5 file says about itself: its frame
    count, scalar arrays and per-frame vectors (such as a temperature or the source frame index)."""
    import numpy as np
    from ase import Atoms
    from ase.io import read

    suffix = Path(path).suffix.lower()
    if suffix not in (".npz", ".h5", ".hdf5"):
        return read(path, index=frame), {}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            data = {key: handle[key] for key in handle.files}
    else:
        import h5py  # only .h5 structures need it

        with h5py.File(path, "r") as handle:
            data = {key: np.asarray(item[()]) for key, item in handle.items() if isinstance(item, h5py.Dataset)}

    def take(name: str, rank: int):
        key = next((key for key in ARRAY_KEYS[name] if key in data), None)
        value = None if key is None else np.asarray(data.pop(key))
        return value[frame] if value is not None and value.ndim == rank + 1 else value

    frames = next((len(data[key]) for key in ARRAY_KEYS["positions"] if key in data and data[key].ndim == 3), 1)
    if not -frames <= frame < frames:
        raise IndexError(f"{path} has {frames} frame(s); --frame {frame}")
    numbers, positions, cell, pbc = take("numbers", 1), take("positions", 2), take("cell", 2), take("pbc", 1)
    if numbers is None or positions is None:
        raise ValueError(f"{path}: no atomic numbers or positions among its arrays {sorted(data)}")
    periodic = cell is not None and bool(np.any(cell)) if pbc is None else pbc
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=periodic)
    info = {"frames": frames, **{key: value.item() for key, value in data.items() if value.ndim == 0}}
    info.update({key: value[frame].item() for key, value in data.items()
                 if frames > 1 and value.ndim == 1 and len(value) == frames})
    return atoms, info


def file_digest(path: str) -> dict:
    with open(path, "rb") as handle:
        return {"path": str(Path(path).absolute()), "sha256": hashlib.sha256(handle.read()).hexdigest()}


def environment() -> dict:
    from importlib import metadata

    import fastmd
    import torch

    versions = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "pid": os.getpid(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "matris": str(Path(fastmd.__file__).parent),
            "versions": versions}


def gpu_apps() -> list[str] | str:
    """The other compute processes on this process's GPU (nvidia-smi, best effort): they skew a timed run."""
    import torch

    uuid = str(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", ""))
    try:
        lines = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60, check=True).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable ({type(exc).__name__})"
    return [line.strip() for line in lines if uuid in line and line.split(",")[0].strip() != str(os.getpid())]


def thermo(md, ensemble: str) -> dict:
    """Step, potential energy, temperature (and the NHC conserved energy) of the MD state, on the host."""
    energy, host = md.state.potential_energy, lambda tensor: float(tensor.detach().reshape(-1)[0])
    out = {"step": md.nsteps, "potential_energy_eV": None if energy is None else host(energy),
           "temperature_K": host(md.temperature())}
    if ensemble == "nvt_nhc" and energy is not None:
        out["conserved_energy_eV"] = host(md.conserved_energy())
    return out


def keep_kernel(adapter, option: str) -> None:
    """Narrow the adapter's kernel site that offers ``option`` to it (and its reference), in every execution
    mode: the negotiation then installs ``option`` (an instance attribute over ``kernel_backends``)."""
    from dataclasses import replace

    choices = list(adapter.kernel_backends()) if hasattr(adapter, "kernel_backends") else []
    if not any(option in choice.options for choice in choices):
        raise SystemExit(f"--kernel {option}: the model's kernel sites offer "
                         f"{[list(choice.options) for choice in choices]}")
    def narrow(choice):
        options = tuple(dict.fromkeys((option, choice.reference)))
        return replace(choice, options=options, by_execution={},
                       accuracy={name: cls for name, cls in choice.accuracy.items() if name in options})
    narrowed = tuple(narrow(choice) if option in choice.options else choice for choice in choices)
    site = next(choice.site for choice in choices if option in choice.options)
    try:  # installed now (outside any graph): an option the loaded model cannot switch to fails here, clearly
        adapter.select_kernel(site, option)
    except Exception as error:  # noqa: BLE001 - the harness exits with the adapter's reason
        raise SystemExit(f"--kernel {option}: {type(error).__name__}: {error} (an option chosen when the model is "
                         "loaded needs the profile that loads it: --profile generic for e3nn, the default profile "
                         "for cuEquivariance)") from None
    adapter.kernel_backends = lambda: narrowed


def engine_stats(md, capacity_plan: bool = True) -> dict | None:
    if getattr(md, "whole_step_runner", None) is not None:
        return md.whole_step_stats(capacity_plan=capacity_plan)
    if getattr(md, "runner", None) is not None:
        return md.model_graph_stats()
    return None


def capacity_fields(args) -> dict:
    """The ``CapacityConfig`` fields the capacity flags set (none: the engine's defaults): the tier step and
    minimum slack of the undirected pairs and of the triplets, the slack defaulting to the step."""
    fields = {}
    for step, pad, (step_field, pad_field) in (
            (args.capacity_step, args.capacity_pad, ("u_step", "min_pad_u")),
            (args.triplet_capacity_step, args.triplet_capacity_pad, ("t_step", "min_pad_t"))):
        if step:
            fields[step_field] = step
        if step or pad is not None:
            fields[pad_field] = step if pad is None else pad
    return fields


def measure(args, result: dict) -> None:
    """One process: load, construct, warm up, time; fills ``result``."""
    import torch

    mode, float32, notes = args.mode[0], args.dtype == "float32", result["warnings"]
    if float32:  # IEEE FP32 math in cuBLAS and cuDNN, set before anything is loaded
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if os.environ.get(TF32_OVERRIDE, "0") not in ("", "0"):
            notes.append(f"{TF32_OVERRIDE}={os.environ[TF32_OVERRIDE]} makes TF32 the cuBLAS default (torch < 2.9: "
                         "even under 'highest'); unset it for IEEE FP32 timings")
    if args.warmup == 0:
        notes.append("no warm-up: first evaluations, compilation and captures fall inside the timed window")
    from fastmd.adapters import load_adapter
    from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
    from fastmd.execution.gpu_nhc import maxwell_boltzmann_momenta
    from fastmd.config import CapacityConfig
    from fastmd.config import InferenceConfig
    from fastmd.precision import precision_state

    atoms, info = read_structure(args.structure, args.frame)
    # Bitwise ASE's MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(seed)).
    atoms.set_momenta(maxwell_boltzmann_momenta(atoms.get_masses(), args.temperature, seed=args.seed))
    result["structure"] = {"path": args.structure, "frame": args.frame, "n_atoms": len(atoms),
                           "formula": atoms.get_chemical_formula(), "cell_lengths": atoms.cell.lengths().tolist(),
                           "pbc": atoms.pbc.tolist(), "file": info}
    result["protocol"] = {"ensemble": args.ensemble, "temperature_K": args.temperature, "timestep_fs": args.timestep,
                          "tdamp_fs": 100.0 * args.timestep if args.ensemble == "nvt_nhc" else None,
                          "momenta": f"maxwell_boltzmann_momenta(seed={args.seed}), ASE MaxwellBoltzmannDistribution's",
                          "warmup": args.warmup, "steps": args.steps, "steps_per_launch": args.steps_per_launch}
    result["environment"] = environment()
    precision = result["precision"] = {"model_dtype": args.dtype, "harness_set_ieee": float32,
                                       "matmul_precision": args.matmul_precision,
                                       TF32_OVERRIDE: os.environ.get(TF32_OVERRIDE), "at_start": precision_state()}
    result["load_adapter_model"] = name = "matris_10m_oam" if args.model == "matris" else args.model
    started = time.perf_counter()
    adapter = load_adapter(name, model_path=args.checkpoint, device="cuda", model_dtype=args.dtype,
                           optimization_profile=args.profile)
    if args.kernel is not None:
        keep_kernel(adapter, args.kernel)
    fields = capacity_fields(args)
    capacity = {"capacity": CapacityConfig(**fields)} if fields else {}
    md = GPUResidentMolecularDynamics(
        atoms, model=adapter, device="cuda", ensemble=args.ensemble, temperature=args.temperature,
        timestep=args.timestep, steps_per_launch=args.steps_per_launch, precapture_steps=args.precapture_steps,
        config=InferenceConfig(execution=mode, optimization_profile=args.profile,
                               matmul_precision=args.matmul_precision, **capacity))
    torch.cuda.synchronize()
    result["load_and_construct_s"] = time.perf_counter() - started
    result["adapter"] = {key: getattr(adapter, key, None) for key in ("name", "kernel", "kernel_reason", "checkpoint")}
    if not result["adapter"]["checkpoint"] and args.checkpoint:
        result["adapter"]["checkpoint"] = file_digest(args.checkpoint)
    precision["kernel"] = result["adapter"]["kernel"]
    result["thermo_initial"] = thermo(md, args.ensemble)
    started = time.perf_counter()
    md.run(args.warmup)
    torch.cuda.synchronize()
    result["warmup_s"] = time.perf_counter() - started
    precision["after_warmup"] = precision_state()
    result["thermo_start"] = thermo(md, args.ensemble)
    before = engine_stats(md, capacity_plan=False)
    result["gpu_apps_before"] = apps = gpu_apps()
    if apps:
        notes.append(f"other processes on this GPU before timing: {apps}")
    result["loadavg_before"] = os.getloadavg()
    result["peak_memory_setup_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    windows = args.steps // args.steps_per_launch  # five near-equal chunks of whole windows
    bounds = sorted({round(k * windows / 5) * args.steps_per_launch for k in range(6)})
    chunks = [end - start for start, end in zip(bounds, bounds[1:])]
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        torch.cuda.synchronize()
        marks = [time.perf_counter()]
        for steps in chunks:
            md.run(steps)
            torch.cuda.synchronize()
            marks.append(time.perf_counter())
    finally:
        gc.enable()
        gc.unfreeze()
    result["ms_per_step"] = (marks[-1] - marks[0]) / args.steps * 1e3
    splits = [(end - start) / steps * 1e3 for start, end, steps in zip(marks, marks[1:], chunks)]
    result["splits_ms_per_step"], result["split_steps"] = splits, chunks
    result["split_spread"] = (max(splits) - min(splits)) / statistics.median(splits)
    result["peak_memory_timed_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    result["loadavg_after"] = os.getloadavg()
    result["thermo_end"] = thermo(md, args.ensemble)
    values = [value for key in ("thermo_start", "thermo_end") for value in result[key].values() if value is not None]
    result["status"] = "ok" if all(math.isfinite(value) for value in values) else "non_finite"
    try:  # the timing stands whatever the report does
        report(md, args, result, before)
    except Exception:  # noqa: BLE001 - recorded
        result["report_error"] = traceback.format_exc()
        notes.append("the report after the timed window failed: see report_error")
    states = [precision[key] for key in ("at_start", "after_warmup", "at_end") if key in precision]
    if float32 and any(state["fp32_gemm"] != "ieee" or state["cudnn_allow_tf32"] is not False for state in states):
        result["status"] = "precision_changed" if result["status"] == "ok" else result["status"]
        notes.append("float32 run, but TF32 was enabled at some point: see precision")


def report(md, args, result: dict, before: dict | None) -> None:
    """After the timed window: the precision at the end, the engine's statistics and their change over the
    window, its configuration summary and plan record."""
    from fastmd.precision import precision_record

    requested = args.matmul_precision or ("highest" if args.dtype == "float32" else None)
    result["precision"]["at_end"] = precision_record(requested, device="cuda")
    result["engine_stats"] = after = engine_stats(md)
    if before is not None and after is not None:
        counters = result["timed_counters"] = {
            key: after[key] - before[key] for key in COUNTERS if isinstance(before.get(key), int)}
        events = {key: value for key, value in counters.items() if value and key in ONE_OFF_EVENTS}
        if events:
            result["warnings"].append(f"one-off engine events inside the timed window {events}: lengthen --warmup")
    summary = result["config_summary"] = md.config_summary()
    result["execution"] = summary["effective"]["execution"]
    if result["execution"] != args.mode[0] and args.mode[0] != "auto":
        result["warnings"].append(f"execution {args.mode[0]!r} ran as {result['execution']!r}")
    if hasattr(md, "plan_record"):
        result["plan_record"] = md.plan_record().as_dict()
    md.close()


def plain(value):
    """``value`` in JSON types: string keys, lists, finite floats (others null), arrays and tensors as lists,
    anything else as its string."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if callable(getattr(value, "items", None)):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(item) for item in value]
    return plain(value.tolist()) if hasattr(value, "tolist") else str(value)


def child(args) -> int:
    result = {"status": "error", "mode": args.mode[0], "model": args.model, "args": vars(args), "warnings": [],
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        measure(args, result)
    except Exception:  # noqa: BLE001 - recorded in the JSON, reported by the parent
        result["error"] = traceback.format_exc()
        traceback.print_exc()
    for note in result["warnings"]:
        print(f"WARNING: {note}", file=sys.stderr)
    print(line(result), flush=True)
    Path(args.child).write_text(json.dumps(plain(result), indent=1))
    return 0 if result["status"] == "ok" else 1


def line(result: dict) -> str:
    """One line per process: ms/step and splits, the execution that ran, thermodynamics and precision."""
    tag = f"[{result['mode']}{' r%s' % result['repeat'] if 'repeat' in result else ''}]"
    status, error = result.get("status"), (result.get("error") or "").strip().splitlines()
    warned = "".join(f"\n    WARNING: {note}" for note in result.get("warnings") or ())
    try:
        start, end, splits = result["thermo_start"], result["thermo_end"], result["splits_ms_per_step"]
        state, n_atoms = result["precision"].get("at_end") or result["precision"]["after_warmup"], \
            result["structure"]["n_atoms"]
        energy = "" if start["potential_energy_eV"] is None else \
            f"E/atom {start['potential_energy_eV'] / n_atoms:.5f} -> {end['potential_energy_eV'] / n_atoms:.5f} eV, "
        counters = {key: value for key, value in (result.get("timed_counters") or {}).items()
                    if value and key not in ("launches", "warmup_runs")}
        dots = sorted({value for site, value in state["triton_dot"].items() if site != "requested"})
        return (f"{tag} {'' if status == 'ok' else status.upper() + ' '}{result['ms_per_step']:.4f} ms/step (splits "
                f"{min(splits):.4f}-{max(splits):.4f}, spread {result['split_spread']:.1%}) "
                f"{result.get('execution')}, {n_atoms} atoms, {energy}T {start['temperature_K']:.1f} -> "
                f"{end['temperature_K']:.1f} K | {result['precision']['model_dtype']}, fp32_gemm {state['fp32_gemm']} "
                f"(cuda.matmul.allow_tf32 {state['cuda_matmul_allow_tf32']}, cudnn.allow_tf32 "
                f"{state['cudnn_allow_tf32']}, float32_matmul_precision {state['float32_matmul_precision']}), "
                f"triton dots {state['triton_dot']['requested']} {dots}, kernel {result['precision']['kernel']}, "
                f"{TF32_OVERRIDE}={result['precision'][TF32_OVERRIDE]}"
                + (f", engine calls at matmul_precision {result['precision']['matmul_precision']!r}"
                   if result["precision"].get("matmul_precision") else "")
                + (f" | timed-window events {counters}" if counters else "") + warned)
    except (KeyError, TypeError):  # no complete measurement
        return f"{tag} {status}: {error[-1] if error else ''}{warned}"


def run_process(args, argv: list[str], mode: str, repeat: int, out: Path) -> dict:
    """One fresh process measuring ``mode`` (this command line; the last ``--structure``/``--mode`` win);
    its JSON, or a stub when it wrote none."""
    stem = out / f"{mode}_r{repeat}"
    json_path, log_path = stem.with_suffix(".json"), stem.with_suffix(".log")
    json_path.unlink(missing_ok=True)  # a stale file must not stand in for a process that writes none
    command = [sys.executable, "-B", str(Path(__file__).absolute()), *argv, "--structure", args.structure,
               "--mode", mode, "--child", str(json_path)]
    env = {key: value for key, value in os.environ.items() if key not in DROPPED_ENVIRONMENT}
    with open(log_path, "w") as log:
        log.write(f"# {shlex.join(command)}\n")
        log.flush()
        try:
            returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                        timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
    try:
        result = json.loads(json_path.read_text())
    except (OSError, ValueError):
        result = {"status": "timeout" if returncode == "timeout" else "error", "mode": mode,
                  "error": f"no JSON (return code {returncode}); see {log_path}"}
    result.update(repeat=repeat, returncode=returncode, json=str(json_path), log=str(log_path))
    return result


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    if args.child is not None:
        return child(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        sys.exit(f"{out / 'summary.json'} exists: use a new --out")
    print(f"model {args.model} ({args.dtype}, profile {args.profile}), {args.structure} frame {args.frame}, "
          f"modes {args.mode}, {args.ensemble} {args.temperature} K dt {args.timestep} fs, {args.warmup} warm-up + "
          f"{args.steps} timed steps x{args.repeats} processes -> {out}", flush=True)
    processes = []
    for repeat in range(1, args.repeats + 1):
        for mode in args.mode if repeat % 2 else args.mode[::-1]:
            processes.append(run_process(args, argv, mode, repeat, out))
            print(line(processes[-1]), flush=True)
    rows = {}
    print(f"\n{'mode':<12} {'ok':>5} {'median':>9} {'min':>9} {'max':>9} {'spread':>7}  ms/step per process")
    for mode in args.mode:
        mine = [result for result in processes if result["mode"] == mode]
        times = [result["ms_per_step"] for result in mine if result.get("status") == "ok"]
        row = rows[mode] = {"ok": len(times), "processes": len(mine), "ms_per_step": times,
                            "statuses": [result.get("status") for result in mine],
                            "execution": sorted({result.get("execution") for result in mine} - {None})}
        if times:
            row.update(median=statistics.median(times), min=min(times), max=max(times))
            row["spread"] = (row["max"] - row["min"]) / row["median"]
            print(f"{mode:<12} {len(times):>2}/{len(mine):<2} {row['median']:>9.4f} {row['min']:>9.4f} "
                  f"{row['max']:>9.4f} {row['spread']:>7.1%}  " + " ".join(f"{value:.4f}" for value in times))
        else:
            print(f"{mode:<12} {0:>2}/{len(mine):<2} {'-':>9} {'-':>9} {'-':>9} {'-':>7}  {row['statuses']}")
    keys = ("mode", "repeat", "status", "returncode", "ms_per_step", "execution", "split_spread", "timed_counters",
            "warnings", "json", "log")
    summary = {"args": vars(args), "rows": rows,
               "processes": [{key: result.get(key) for key in keys} for result in processes]}
    (out / "summary.json").write_text(json.dumps(plain(summary), indent=1))
    return 0 if all(result.get("status") == "ok" for result in processes) else 1


if __name__ == "__main__":
    sys.exit(main())
