"""torchrun launcher for distributed eager, automatically padded model_graph, or a size-based choice.

With ``--md-steps``: MD after the E/F timing with the one MD driver GPUResidentMolecularDynamics (nve,
Berendsen nvt or Nose-Hoover chain nvt_nhc; heating ramp; md_pyrolysis-style thermo, EXTXYZ and restart
files; resume at any rank count). Several CUDA ranks (and CPU runs) integrate on the evaluator
(``from_evaluator``, engine "domain"); one CUDA rank runs the native single-GPU engine (engine "native":
whole_step / model_graph with the single-CTA Nose-Hoover chain), see ``--md-engine``.
"""
# The entry point behind ``python -m fastmd.distributed``, whose ``__main__`` only calls main() (the docstring above
# is the command line's help text).
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist

from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.execution.md_protocol import (
    ENSEMBLES, RESTART_FILE, THERMO_FILE, TRAJECTORY_FILE, HeatingRamp, constant_pressure_error, initialize_momenta,
)
from fastmd.config import InferenceConfig, resolve_config
from fastmd.adapters import load_adapter
from fastmd.models import DEFAULT, MODELS, check_load_options, model_family
from fastmd.precision import (
    NOISE_FLOOR_SHIFT_A, compute_references, matmul_precision_scope, oracle_report, precision_record,
    warn_implicit_tf32,
)
from fastmd.distributed.affinity import ENVIRONMENT, MODES, bind_local_rank
from fastmd.distributed.auto_execution import AutoExecution
from fastmd.distributed.config import ParallelConfig, single_rank_profile
from fastmd.distributed.evaluator import DistributedMatRISEvaluator, _objects, collective_check, use_single_rank_engine
from fastmd.distributed.api import DistributedEvaluator
from fastmd.distributed.launch import fail_fast, initialize, shutdown
from fastmd.distributed.partition import AtomState
from fastmd.distributed.watchdog import ProgressWatchdog

# The single-GPU engine builds its graph with the reference's own builder and
# its reference runs the same profile (G-D1), so only E/F execution differs:
# gate IEEE runs at the K3/G-D1 1e-5 limits (TF32 unchanged).
SINGLE_ENGINE_TOLERANCE = {"ieee": {"energy_error_eV_per_atom": 1e-5, "force_RMSE_eV_A": 1e-5,
                                    "force_max_error_eV_A": 1e-5}}


def removed_choice_error(args) -> str | None:
    """A removed CLI choice, kept one version so that it names its replacement, or None."""
    if args.control_plane == "gloo":
        return ("--control-plane gloo was removed (the pre-3.1a Gloo object rounds, kept one version as the "
                "rollback of the NCCL control plane): nccl, the default, is the only multi-rank control plane")
    if args.single_rank_engine == "plugin_loopback":
        return ("--single-rank-engine plugin_loopback was removed (the multi-rank protocol at one CUDA rank, kept "
                "one version as the rollback of unified step 1): one CUDA rank always runs the single-GPU "
                "engine; run the multi-rank protocol with 2+ ranks, or with --device cpu")
    if args.md_driver == "nve_legacy":
        return ("--md-driver nve_legacy was removed (DistributedNVE, the pre-3.3 NVE driver): use the default "
                "driver (GPUResidentMolecularDynamics.from_evaluator) with --ensemble nve --no-fix-com, the same "
                "trajectory bitwise")
    return None


def md_arguments_error(args) -> str | None:
    """The first invalid MD option combination, or None."""
    temperatures = [value for value in (args.temp_start, args.temp_target) if value is not None]
    if any(not (math.isfinite(value) and value > 0) for value in temperatures):
        return "temperatures must be finite and positive"
    if not all(math.isfinite(value) and value > 0 for value in (args.tdamp_fs, args.taut_fs)):
        return "tdamp-fs and taut-fs must be finite and positive"
    if min(args.tchain, args.tloop, args.update_interval) < 1:
        return "tchain, tloop and update-interval must be positive"
    if min(args.thermo_interval, args.xyz_interval, args.restart_interval, args.stop_at_step) < 0:
        return "output intervals and stop-at-step must be nonnegative"
    if args.temp_start is None and (args.temp_target is not None or args.seed is not None
                                    or args.ensemble != "nve"):
        return "--temp-target, --seed, nvt and nvt_nhc need --temp-start"
    if args.ensemble == "nve" and args.temp_target not in (None, args.temp_start):
        return "a heating ramp needs a thermostat (--ensemble nvt or nvt_nhc)"
    outputs = args.xyz_interval or args.restart_interval or args.resume or args.stop_at_step
    if outputs and args.md_output_dir is None:
        return "--xyz-interval, --restart-interval, --resume and --stop-at-step need --md-output-dir"
    if args.steps_per_launch < 1:
        return "--steps-per-launch must be positive"
    window = {"--md-steps": args.md_steps, "--thermo-interval": args.thermo_interval,
              "--xyz-interval": args.xyz_interval, "--restart-interval": args.restart_interval,
              "--stop-at-step": args.stop_at_step}
    off = [name for name, value in window.items() if value % args.steps_per_launch]
    if off:
        return f"--steps-per-launch {args.steps_per_launch} must divide {', '.join(off)}"
    return None


def model_and_checkpoint(args) -> tuple[str, str | None]:
    """Resolve CLI model arguments using the same spelling for preflight and loading.

    The original ``--model /path/to/matris.pth.tar`` selects MatRIS with that
    checkpoint; MACE's ``.model`` paths go through the model registry unchanged.
    """
    if args.checkpoint is None and Path(args.model).is_file() and not args.model.endswith(".model"):
        return "matris", args.model
    return args.model, args.checkpoint


def model_source(args) -> str:
    """The registry family used by both preflight and loading, including checkpoint path dispatch."""
    return model_family(*model_and_checkpoint(args))


def ensemble_name(value: str) -> str:
    """Reject constant-pressure ensembles with the MD driver's fixed-cell explanation."""
    refusal = constant_pressure_error(value)
    if refusal is not None:
        raise argparse.ArgumentTypeError(refusal)
    return value


def matris_only_options(args, parser) -> list[str]:
    """Explicit MatRIS scheduling options that would have no effect on a declared-shard evaluator."""
    names = ("auto_eager_triplet_rows", "auto_single_rank_eager_triplet_rows", "saved_tensor_recompute")
    return ["--" + name.replace("_", "-") for name in names
            if getattr(args, name) != parser.get_default(name)]


def sharded_configs(args) -> tuple[InferenceConfig, ParallelConfig]:
    """The same declared-shard settings for early validation and evaluator construction."""
    config = InferenceConfig(execution=args.execution, optimization_profile=args.profile, report=False,
                             matmul_precision=args.precision, saved_tensor_recompute=args.saved_tensor_recompute,
                             triton_dot_precision=args.triton_dot_precision, recompute_policy=args.recompute_policy)
    parallel = ParallelConfig(domain_grid=tuple(args.grid) if args.grid else "auto", local_backend=args.local_backend,
                              exchange_self_rows=args.exchange_self_rows, payload_transport=args.payload_transport,
                              legacy_capacity_policy=args.legacy_capacity_policy, exchange_scheme=args.exchange_scheme,
                              topology_engine=args.topology_engine, topology_skin=args.topology_skin,
                              grad_mechanism=args.grad_mechanism, debug_replay_check=args.debug_replay_check)
    return config, parallel


def model_options_error(args, family: str, parser) -> str | None:
    """Reject unsupported loading/evaluation options before binding CPUs or starting ranks and thread pools."""
    from fastmd.adapters.protocol import NegotiationError

    model_name, checkpoint = model_and_checkpoint(args)
    if (args.model in {row.name for row in MODELS} - {DEFAULT} and not Path(args.model).exists()
            and (checkpoint is None or family == DEFAULT)):
        return (f"--model {args.model}: a model family's name loads the checkpoint file given with --checkpoint "
                "where its loader reads one; otherwise give a pretrained name of the registry "
                "(e.g. mace_mp_0, sevennet_0, orb_v3, chgnet_0_3_0)")
    try:
        check_load_options(model_name, model_path=checkpoint, model_dtype=args.dtype,
                           optimization_profile=args.profile)
    except (ValueError, TypeError, NegotiationError) as exc:
        return f"--model {args.model} ({family}): {exc}"
    if family == DEFAULT:
        return None
    from fastmd.distributed.sharded_evaluator import check_device, unsupported_settings

    given = matris_only_options(args, parser)
    if given:
        return (f"{', '.join(given)}: MatRIS's scheduling options; the {family} model runs on ShardedEvaluator, "
                "which has no such option")
    try:
        config, parallel = sharded_configs(args)
        if args.dtype == "float64":
            check_device(torch.device(args.device), torch.float64, int(os.environ.get("WORLD_SIZE", "1")))
    except (ValueError, TypeError, NegotiationError) as exc:
        return f"--model {args.model} ({family}): {exc}"
    if args.topology_engine == "epoch" and args.device == "cpu":
        return "--topology-engine epoch needs CUDA: CPU ranks rebuild their reference topology every evaluation"
    unsupported = unsupported_settings(config, parallel)
    if unsupported:
        return (f"the {family} model runs on ShardedEvaluator, which does not run {', '.join(unsupported)}: "
                "leave those options at their defaults")
    return None


def load_on_ranks(load, context, *, root_first: bool):
    """Load named weights on rank zero first to avoid racing a shared download cache; agree every failure."""
    adapter = error = None
    if root_first:
        if context.rank == 0:
            try:
                adapter = load()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        collective_check(context, error)
    error = None
    if not root_first or context.rank != 0:
        try:
            adapter = load()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    collective_check(context, error)
    return adapter


def write_report(path: Path, report: dict) -> None:
    """Write ``report`` as JSON to the new file ``path``, whole or not at all: serialized first (a non-finite
    float, e.g. in a failed oracle's errors, as its string with a note), then linked from a temporary file
    (never overwriting)."""
    try:
        text = json.dumps(report, indent=2, allow_nan=False)
    except ValueError:
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {key: finite(item) for key, item in value.items()}
            return [finite(item) for item in value] if isinstance(value, (list, tuple)) else value
        text = json.dumps({**finite(report), "non_finite_values": "written as strings ('nan', 'inf')"},
                          indent=2, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def neighbor_pairs(atoms, cutoff: float) -> int:
    """Directed pairs within ``cutoff`` of ``atoms`` (ASE's FP64 neighbour list: an independent edge count)."""
    from ase.neighborlist import neighbor_list
    return len(neighbor_list("i", atoms, cutoff))


def same_pairs(first, second) -> bool:
    """Whether two canonical topologies have the same directed atom pairs (with multiplicity, any image) and as
    many triplets: a rigid shift may rewrap images but must not move a pair across a cutoff."""
    from collections import Counter

    def pairs(topology):
        return Counter((key[0], key[1]) for key in topology.directed_keys)
    return pairs(first) == pairs(second) and len(first.triplet_keys) == len(second.triplet_keys)


def sharded_edge_count(evaluator, owned_ids, context) -> int | None:
    """Directed edges at owned centres of a declared-shard evaluator's last evaluation, summed over the ranks
    (collective); None where its engine does not list them (one CUDA rank; a CPU line-graph shard)."""
    count = None
    try:  # every rank reaches the gather below
        if evaluator.device.type != "cuda":  # exact rows and edges; a receiver shard lists the owned centres' edges
            if evaluator.spec.message_flow == "receiver":
                count = int(evaluator.last_stats["edges"])
        elif not evaluator.single:
            keys = evaluator.edge_keys()
            count = int(torch.isin(keys[:, 0], owned_ids.to(keys.device)).sum())
    except Exception:  # no count: no evidence, the absolute gate decides
        count = None
    counts = _objects(context, count)
    return None if None in counts else sum(counts)


def edge_evidence(distributed: int | None, atoms, cutoff: float) -> dict:
    """The oracle's topology evidence for an evaluator without canonical topology parity: its edge count against
    ASE's at the same frame (``parity`` None: the engine does not list its edges)."""
    if distributed is None:
        return {"parity": None, "reason": "this engine does not list its edges"}
    reference = neighbor_pairs(atoms, cutoff)
    return {"parity": distributed == reference, "distributed_edges": distributed, "reference_edges": reference,
            "reference": "ASE neighbour list"}


def md_engine(args, world_size: int, device) -> tuple[str, str]:
    """(engine, reason) of the MD: "native" (GPUResidentMolecularDynamics on one GPU: whole_step /
    model_graph / eager with the single-CTA Nose-Hoover chain) or "domain" (``from_evaluator``: the
    evaluator's E/F with the decomposed integrators, the multi-rank numerics). ``--md-engine auto``: native
    at one CUDA rank, except where only the domain engine has the requested behavior (nvt_nhc with
    ``--fix-com`` or a non-default decomposed-chain option)."""
    if args.md_engine == "domain":
        if args.steps_per_launch > 1:
            raise ValueError("--steps-per-launch needs the native engine (whole-step windows)")
        return "domain", "--md-engine domain"
    # --deterministic: the native eager, model_graph and whole_step paths repeat bitwise under the
    # switch, also through whole-step capacity growth and candidate refreshes (measured, GB200).
    nhc = args.ensemble == "nvt_nhc"
    blocker = ("several ranks" if world_size > 1 else "CPU device" if device.type != "cuda" else
               "nvt_nhc --fix-com: the native chain has no centre-of-mass correction"
               if nhc and args.fix_com else
               "--nhc-scalar-sync local / --nhc-ke2-update recompute are options of the decomposed chain"
               if nhc and (args.nhc_scalar_sync, args.nhc_ke2_update) != ("broadcast", "scaled") else None)
    if blocker is None:
        return "native", "one CUDA rank"
    if args.md_engine == "native" or args.steps_per_launch > 1:
        option = "--md-engine native" if args.md_engine == "native" else "--steps-per-launch"
        raise ValueError(f"{option}: {blocker}; the native engine runs one CUDA rank")
    return "domain", blocker


def build_native_dynamics(args, evaluator, model, atoms, resume: bool):
    """One CUDA rank: the native GPU-MD engine, from the input atoms (with any --seed momenta) or the restart file.

    ``--execution auto`` leaves the mode to GPUResidentMolecularDynamics's execution auto (whole_step where
    it applies), except that it stays eager where the single-GPU engine's auto chose eager for the E/F
    timing (``--auto-single-rank-eager-triplet-rows``, models outside the memory calibration). The
    evaluator's captured graphs are released first: the two engines never hold graph pools at once.
    """
    from fastmd.execution.md_protocol import read_restart
    ramp = heating_ramp(args)
    engine = evaluator._engine
    custom_strategy = hasattr(evaluator, "_resolved")
    selected_execution = engine.execution if custom_strategy else evaluator.execution
    if args.execution != "auto":
        execution, reason = args.execution, f"--execution {args.execution}"
    elif selected_execution == "eager":
        execution, reason = "eager", "the single-GPU engine's execution auto chose eager"
    else:
        execution, reason = "auto", None
    if custom_strategy:
        engine.close()  # the evaluator is not evaluated again
        evaluator.last_graph = None
    else:
        evaluator.close()
    gc.collect()
    torch.cuda.empty_cache()
    profile = evaluator._resolved.optimization_profile if custom_strategy else args.profile
    config = InferenceConfig(execution=execution, optimization_profile=profile,
                             report=False, matmul_precision=args.precision,
                             saved_tensor_recompute=args.saved_tensor_recompute,
                             triton_dot_precision=args.triton_dot_precision, recompute_policy=args.recompute_policy)
    options = dict(model=model, task="ef", config=config, ensemble=args.ensemble, timestep=args.dt_fs,
                   tdamp=args.tdamp_fs, taut=args.taut_fs, tchain=args.tchain, tloop=args.tloop,
                   fix_com=args.fix_com, ramp=ramp, output_dir=args.md_output_dir,
                   thermo_interval=args.thermo_interval, xyz_interval=args.xyz_interval,
                   restart_interval=args.restart_interval, total_steps=args.md_steps,
                   steps_per_launch=args.steps_per_launch)
    if args.temp_start is not None:
        options["temperature"] = args.temp_start
    if not resume:
        dynamics = GPUResidentMolecularDynamics(atoms, **options)
    else:
        path = args.md_output_dir / RESTART_FILE
        data = read_restart(path)
        # The checks of evaluator.distribute_restart (the domain engine's resume).
        cell = torch.as_tensor(data.cell).to(evaluator.dtype)
        if not torch.equal(cell, evaluator.partition.cell.to(evaluator.dtype)) or not data.pbc.all():
            raise ValueError("restart cell/pbc differs from the evaluator's")
        if not np.array_equal(data.numbers, atoms.get_atomic_numbers()):
            raise ValueError("restart atomic numbers do not match the input structure")
        saved = data.parameters["ensemble"]
        if saved != args.ensemble:
            raise ValueError(f"--ensemble {args.ensemble} cannot resume a {saved} restart file")
        if data.temperature_K is not None:
            if ramp is None and data.parameters["ramp"] is None and data.temperature_K != args.temp_start:
                raise ValueError(f"--temp-start {args.temp_start} cannot resume a restart file at "
                                 f"{data.temperature_K} K")
            options["temperature"] = data.temperature_K
        # from_restart rejects any other option that differs from the file's (except --md-steps without a ramp).
        dynamics = GPUResidentMolecularDynamics.from_restart(path, **options)
    dynamics.engine_reason = reason
    return dynamics


def native_report(dynamics) -> tuple[dict, dict]:
    """The native engine's report frame (``evaluator.gather``'s keys, ID order) and energies."""
    fields = dynamics.md_fields()
    frame = {"ids": fields["ids"], "positions": dynamics.state.positions, "numbers": fields["numbers"],
             "forces": fields["forces"], "masses": fields["masses"], "momenta": fields["momenta"]}
    if dynamics.thermostat is not None:
        frame["positions64"] = fields["positions"]
    energy = float(fields["energy"])
    energies = {"natoms": dynamics.n_atoms, "energy_eV": energy, "energy_eV_per_atom": energy / dynamics.n_atoms,
                "reference_energy_eV": float(fields["reference_energy"])}
    return {key: value.detach().cpu() for key, value in frame.items()}, energies


def heating_ramp(args) -> HeatingRamp | None:
    return (HeatingRamp(args.temp_start, args.temp_target, args.md_steps, args.update_interval)
            if args.temp_target not in (None, args.temp_start) else None)


def build_dynamics(args, evaluator, initial, atoms, resume: bool):
    """The domain MD driver of the CLI options (all ranks; ``atoms`` on rank 0 only)."""
    ramp = heating_ramp(args)
    options = dict(ensemble=args.ensemble, timestep=args.dt_fs, temperature=args.temp_start,
                   tdamp=args.tdamp_fs, taut=args.taut_fs, tchain=args.tchain, tloop=args.tloop,
                   fix_com=args.fix_com, ramp=ramp, nhc_scalar_sync=args.nhc_scalar_sync,
                   nhc_ke2_update=args.nhc_ke2_update, output_dir=args.md_output_dir,
                   thermo_interval=args.thermo_interval, xyz_interval=args.xyz_interval,
                   restart_interval=args.restart_interval, total_steps=args.md_steps,
                   integrator_in_graph=args.md_integrator_in_graph)
    if not resume:
        # nvt_nhc integrates the float64 input state (distributed from root's atoms); nve/nvt the float32 one.
        if args.ensemble == "nvt_nhc":
            return GPUResidentMolecularDynamics.from_evaluator(evaluator, atoms=atoms, **options)
        return GPUResidentMolecularDynamics.from_evaluator(evaluator, initial.state, **options)
    restart = evaluator.distribute_restart(args.md_output_dir / RESTART_FILE, atoms=atoms)
    saved = restart["parameters"]["ensemble"]
    if saved != args.ensemble:  # replicated: every rank raises
        raise ValueError(f"--ensemble {args.ensemble} cannot resume a {saved} restart file")
    temperature = restart["temperature_K"]
    if temperature is not None:
        # A constant target is not in "parameters" (a ramp's start and end are):
        # the saved one must be --temp-start.
        if ramp is None and restart["parameters"]["ramp"] is None and temperature != args.temp_start:
            raise ValueError(f"--temp-start {args.temp_start} cannot resume a restart file at {temperature} K")
        options["temperature"] = temperature
    # from_evaluator rejects any other option that differs from the file's (except --md-steps without a ramp).
    return GPUResidentMolecularDynamics.from_evaluator(evaluator, restart=restart, **options)


def md_thermostat(args, dynamics) -> dict | None:
    """The report's thermostat settings."""
    if args.ensemble == "nve":
        return None
    settings = {"temperature_start_K": args.temp_start,
                "temperature_target_K": args.temp_start if args.temp_target is None else args.temp_target,
                "update_interval": args.update_interval}
    if args.ensemble == "nvt":
        return {**settings, "kind": "berendsen", "taut_fs": args.taut_fs}
    return {**settings, "kind": "nose_hoover_chain", "tdamp_fs": args.tdamp_fs, "tchain": args.tchain,
            "tloop": args.tloop, "scalar_sync": args.nhc_scalar_sync, "ke2_update": args.nhc_ke2_update,
            "backend": dynamics.thermostat.backend,
            # "single" (the native engine; scalar_sync and ke2_update are the decomposed protocol's) or "decomposed"
            "protocol": dynamics.thermostat.protocol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="model registry name (MatRIS, CHGNet, MACE, MACE-SOYO, SevenNet, ORB, TACE), or MatRIS checkpoint")
    parser.add_argument("--checkpoint", help="local weights for the selected model (CHGNet uses bundled weights)")
    parser.add_argument("--dtype", "--weights-dtype", dest="dtype", choices=("float32", "float64"), default=None,
                        help="model dtype; float64 is supported only by models that declare it")
    parser.add_argument("--input", required=True, help="ASE-readable fixed-cell full-PBC structure")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--execution", choices=("auto", "eager", "model_graph"), default="auto",
                        help="auto: model_graph below, eager at or above --auto-eager-triplet-rows "
                             "(max per-rank triplet rows, decided at the first evaluation with "
                             "hysteresis; the P=1 single-GPU engine uses --auto-single-rank-eager-"
                             "triplet-rows on its own triplet rows); under memory admission the "
                             "topology and legacy profiles (P=1) and topology (P>1) take model_graph "
                             "whenever it fits instead, and models outside the memory calibration run "
                             "eager; a mode that does not fit device memory is never chosen "
                             "and an explicit one fails fast (MATRIS_MEMORY_ADMISSION=off disables); "
                             "CPU runs keep eager")
    parser.add_argument("--auto-eager-triplet-rows", type=int,
                        default=AutoExecution.eager_min_triplet_rows)
    parser.add_argument("--auto-single-rank-eager-triplet-rows", type=int,
                        default=AutoExecution.single_rank_eager_min_triplet_rows,
                        help="the same choice for the P=1 single-GPU engine, on its own triplet rows")
    parser.add_argument("--cpu-affinity", choices=MODES,
                        help=f"auto (default, or {ENVIRONMENT}): pin each rank to CPUs local to its GPU, "
                             "split among local ranks; kept if already narrower than the cgroup cpuset")
    parser.add_argument("--grid", nargs=3, type=int)
    parser.add_argument("--local-backend", choices=("torch_reference", "native"), default=None,
                        help="deprecated alias of --profile (native = topology, torch_reference = generic); "
                             "with the old default --profile generic it decides, otherwise it must agree")
    parser.add_argument("--profile", choices=("topology", "generic", "legacy"), default=None,
                        help="optimization_profile, the one kernel knob at every rank count: topology (fused "
                             "kernels; the default), generic (plain torch reference), legacy (single GPU only)")
    parser.add_argument("--single-rank-engine", choices=("single", "plugin_loopback"), default="single",
                        help="one CUDA rank runs the single-GPU engine; 'plugin_loopback' (the multi-rank "
                             "protocol at one rank) was removed and is rejected")
    parser.add_argument("--precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--deterministic", action="store_true",
                        help="torch.use_deterministic_algorithms(True), the deterministic switch: MatRIS "
                             "orders its sums that otherwise add with atomics, so repeated runs are bitwise "
                             "at a fixed GPU type, rank count and software (pin --execution and "
                             "--saved-tensor-recompute: auto decides from free memory); slower. Sets "
                             "CUBLAS_WORKSPACE_CONFIG=:4096:8 unless it is :4096:8 or :16:8")
    parser.add_argument("--saved-tensor-recompute", choices=("off", "on", "deep", "auto"), default=None,
                        help="rebuild cheap saved activations in the backward (InferenceConfig."
                             "saved_tensor_recompute; default: MATRIS_SAVED_TENSOR_RECOMPUTE, else auto)")
    parser.add_argument("--triton-dot-precision", choices=("auto", "ieee", "tf32x3", "tf32"), default=None,
                        help="input precision of the Triton FP32 dots (InferenceConfig.triton_dot_precision; "
                             "default: MATRIS_TRITON_DOT_PRECISION, else auto)")
    parser.add_argument("--recompute-policy", choices=("safe", "fast"), default=None,
                        help="how close to the device the memory decisions go (InferenceConfig.recompute_policy; "
                             "default: MATRIS_RECOMPUTE_POLICY, else fast)")
    parser.add_argument("--exchange-self-rows", choices=("local", "nccl"), default="local",
                        help="self rows as an on-device gather-scatter (local) or through NCCL (rollback)")
    parser.add_argument("--exchange-scheme", choices=("rows", "bond_partial"), default="bond_partial",
                        help="P>1 model exchange: the symmetric boundary-bond partials (bond_partial, the only "
                             "scheme; 'rows' was removed in step 4.2 and is rejected)")
    parser.add_argument("--topology-engine", choices=("auto", "rebuild", "epoch"), default="auto",
                        help="P>1 topology: built once per skin epoch and filtered on the device every step "
                             "(epoch; CUDA), or auto (default: the epoch on CUDA, above 4 ranks with the skin "
                             "clamped to the box; CPU runs use an internal reference topology); 'rebuild' "
                             "was removed in step 4.2 and is rejected")
    parser.add_argument("--topology-skin", type=float, default=1.0,
                        help="skin (Angstrom) of the topology epoch")
    parser.add_argument("--grad-mechanism", choices=("tape", "autograd"), default="autograd",
                        help="P>1 forces: one autograd backward over exchange Functions on a token chain "
                             "(autograd, the only mechanism; 'tape' was removed in step 4.2 and is rejected)")
    parser.add_argument("--debug-replay-check", action="store_true",
                        help="debug: agree every P>1 model_graph replay's identity across ranks after it")
    parser.add_argument("--control-plane", choices=("gloo", "nccl"), default="nccl",
                        help="multi-rank host control agreement: fixed-shape rounds on the data group, Gloo "
                             "only for error strings (nccl, the only control plane; 'gloo', the pre-3.1a "
                             "rollback, was removed and is rejected)")
    parser.add_argument("--payload-transport", choices=("auto", "all_to_all", "neighbor_p2p"), default="auto",
                        help="per-peer-split payload exchanges: all_to_all_single, or one grouped P2P over "
                             "the neighbor domains; auto: all_to_all up to 4 ranks, neighbor_p2p above unless "
                             "every rank's neighbor domains are all other ranks")
    parser.add_argument("--legacy-capacity-policy", action="store_true",
                        help="removed in step 4.2 (the pre-3.2 multi-rank model_graph capacity policy, which "
                             "required the rows exchange) and rejected; model_graph uses the single-GPU tiers")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3, help="fixed-frame E/F timing repetitions")
    parser.add_argument("--stress", action="store_true",
                        help="include stress in initial, warmup and timed fixed-frame evaluations; report the initial "
                             "stress. MatRIS domain MD may retain the stress capture; native MD evaluates E/F")
    parser.add_argument("--md-steps", type=int, default=0,
                        help="total MD steps of the schedule (the heating ramp's denominator); "
                             "--resume continues from the saved step")
    parser.add_argument("--dt-fs", type=float, default=0.2,
                        help="MD time step (fs; CLI default 0.2 as md_pyrolysis, the Python API "
                             "GPUResidentMolecularDynamics defaults to 1.0)")
    md_options = parser.add_argument_group("MD (the one driver, GPUResidentMolecularDynamics; "
                                           "option names of the external md_pyrolysis script)")
    md_options.add_argument("--md-engine", choices=("auto", "native", "domain"), default="auto",
                            help="native: the single-GPU engine (one CUDA rank; whole_step / model_graph / "
                                 "eager by --execution, the single-CTA Nose-Hoover chain; faster, other "
                                 "numerics than domain); domain: the evaluator's E/F with the decomposed "
                                 "integrators (from_evaluator; every rank count). auto: native at one CUDA "
                                 "rank (except nvt_nhc with --fix-com, --nhc-scalar-sync local or "
                                 "--nhc-ke2-update recompute), else domain")
    md_options.add_argument("--md-driver", choices=("thin", "nve_legacy"), default="thin",
                            help="GPUResidentMolecularDynamics.from_evaluator, the only driver ('nve_legacy', "
                                 "the pre-3.3 DistributedNVE, was removed and is rejected)")
    md_options.add_argument("--ensemble", type=ensemble_name, choices=ENSEMBLES, default="nve",
                            help="nve (CLI default; the Python API's is nvt), nvt (Berendsen) or nvt_nhc "
                                 "(ASE Nose-Hoover chain, float64 state)")
    md_options.add_argument("--temp-start", type=float,
                            help="thermostat target (K) at step 0 and the --seed temperature")
    md_options.add_argument("--temp-target", type=float,
                            help="target (K) at --md-steps: linear heating ramp (default: --temp-start)")
    md_options.add_argument("--tdamp-fs", type=float, default=20.0,
                            help="nvt_nhc damping time (CLI default 20 as md_pyrolysis; the Python API's is "
                                 "100 time steps)")
    md_options.add_argument("--taut-fs", type=float, default=100.0, help="nvt (Berendsen) coupling time")
    md_options.add_argument("--tchain", type=int, default=3)
    md_options.add_argument("--tloop", type=int, default=1)
    md_options.add_argument("--fix-com", action=argparse.BooleanOptionalAction, default=False,
                            help="remove the total momentum every step (CLI default off, as md_pyrolysis and "
                                 "utils/benchmark_single_ef.py; the Python API GPUResidentMolecularDynamics "
                                 "defaults on for nve/nvt)")
    md_options.add_argument("--steps-per-launch", type=int, default=1,
                            help="native engine: MD steps per whole-step graph launch (one host check per "
                                 "window; ramps keep per-step targets); must divide --md-steps and the output "
                                 "intervals; samples_ms stays per MD step")
    md_options.add_argument("--update-interval", type=int, default=1,
                            help="steps between ramp target updates (the chunk midpoint, as md_pyrolysis)")
    md_options.add_argument("--seed", type=int,
                            help="md_pyrolysis initial momenta at --temp-start (default: keep the input momenta)")
    md_options.add_argument("--md-output-dir", type=Path,
                            help=f"{THERMO_FILE}, {TRAJECTORY_FILE} and {RESTART_FILE} (rank 0)")
    md_options.add_argument("--thermo-interval", type=int, default=100, help="steps between thermo lines")
    md_options.add_argument("--xyz-interval", type=int, default=0, help="steps between EXTXYZ frames (0 = off)")
    md_options.add_argument("--restart-interval", type=int, default=0,
                            help="steps between restart files (0 = off; else also at the last step)")
    md_options.add_argument("--resume", action="store_true",
                            help=f"continue from --md-output-dir/{RESTART_FILE} if it exists (any rank count; "
                                 "integrator and thermostat options must match the file's, only a run "
                                 "without a ramp may change --md-steps)")
    md_options.add_argument("--stop-at-step", type=int, default=0,
                            help="stop (with a restart file) at this step")
    md_options.add_argument("--nhc-scalar-sync", choices=("broadcast", "local"), default="broadcast")
    md_options.add_argument("--nhc-ke2-update", choices=("scaled", "recompute"), default="scaled")
    md_options.add_argument("--md-integrator-in-graph", action=argparse.BooleanOptionalAction, default=True,
                            help="model_graph at P>1: the integrator inside the epoch whole-step graph (default); "
                                 "--no-md-integrator-in-graph runs it outside, around each evaluation (one "
                                 "version)")
    parser.add_argument("--oracle", action="store_true",
                        help="independent original-model E/F check ('highest' GEMMs; generic eager kernels, "
                             "or the resolved profile's eager kernels for the single-GPU engine)")
    parser.add_argument("--oracle-final", action="store_true",
                        help="also compare the final MD frame with a fresh unsharded generic eager model")
    parser.add_argument("--precision-force-rmse-tolerance", type=float,
                        help="also gate the oracle's force RMSE against 'highest' (eV/A); reported only by default")
    parser.add_argument("--output", type=Path, help="new JSON output file (will not overwrite)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--watchdog-seconds", type=float, default=600.0,
                        help="progress deadline for initialization, oracle, capture and teardown; 0 disables")
    parser.add_argument("--watchdog-step-seconds", type=float, default=60.0,
                        help="minimum per-step deadline once steps run (at least 20x the last step)")
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.md_steps) < 0 or args.threads < 1:
        parser.error("step counts must be nonnegative and threads positive")
    if args.watchdog_seconds < 0 or args.watchdog_step_seconds <= 0:
        parser.error("watchdog seconds must be nonnegative and step seconds positive")
    tolerance = args.precision_force_rmse_tolerance
    if tolerance is not None and not (math.isfinite(tolerance) and tolerance >= 0):
        parser.error("precision-force-rmse-tolerance must be finite and nonnegative")
    md_error = removed_choice_error(args) or md_arguments_error(args)
    if md_error:
        parser.error(md_error)
    family = model_source(args)
    model_error = model_options_error(args, family, parser)
    if model_error:
        parser.error(model_error)
    if args.md_steps:
        try:  # an MD engine the options exclude fails before the model, oracle and E/F timing
            md_engine(args, int(os.environ.get("WORLD_SIZE", "1")), torch.device(args.device))
            # Only MatRIS has its own exchange strategy; other auto engines use the domain driver.
            if (args.steps_per_launch > 1 and args.md_engine == "auto"
                    and family != DEFAULT):
                raise ValueError("declared model shards with the domain driver require --steps-per-launch 1")
        except ValueError as exc:
            parser.error(str(exc))
    if args.deterministic:
        # Before cuBLAS starts; a rank re-executed by its CPU binding inherits it.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            parser.error("--deterministic needs CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 (unset: :4096:8)")
    try:
        auto_execution = (AutoExecution(
            eager_min_triplet_rows=args.auto_eager_triplet_rows,
            single_rank_eager_min_triplet_rows=args.auto_single_rank_eager_triplet_rows)
                          if args.execution == "auto" and family == DEFAULT else None)
        # Before thread pools and process groups start; a binding re-executes
        # this rank once so that all of its memory is first touched locally.
        affinity = bind_local_rank(args.cpu_affinity, device=args.device, reexec=True)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision(args.precision)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        warn_implicit_tf32(args.precision, "fastmd.distributed")
    context = initialize(device=args.device, timeout_seconds=args.timeout)
    # c10d timeouts do not end a job whose peer failed; the watchdog and
    # fail_fast() below are the application-level replacement.
    watchdog = ProgressWatchdog(args.watchdog_seconds, step_seconds=args.watchdog_step_seconds,
                                rank=context.rank, output=args.output).start()
    watchdog.phase("initialize")
    evaluator = None
    try:
        device = torch.device("cuda", torch.cuda.current_device()) if args.device == "cuda" else torch.device("cpu")
        stress_options = {"stress": True} if args.stress else {}
        atoms, metadata, error = None, None, None
        if context.rank == 0:
            try:
                from ase.io import read
                atoms = read(args.input)
                if family != DEFAULT:
                    from fastmd.distributed.partition import DomainPartition
                    try:
                        DomainPartition(torch.as_tensor(atoms.cell.array), 1, "auto",
                                        tuple(bool(axis) for axis in atoms.pbc))
                    except ValueError as exc:
                        raise ValueError(f"--input {args.input} (pbc {atoms.pbc.tolist()}): {exc}") from None
                # md_pyrolysis: --resume without a restart file is a fresh start.
                resume = bool(args.md_steps and args.resume and (args.md_output_dir / RESTART_FILE).exists())
                if args.md_steps and args.seed is not None and not resume:
                    initialize_momenta(atoms, args.temp_start, args.seed)
                metadata = {"cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist(), "resume": resume}
                if args.output is not None and args.output.exists():
                    raise ValueError(f"refusing to overwrite {args.output}")
                final_npz = None if args.output is None else args.output.with_suffix(".final.npz")
                if (args.oracle_final and final_npz is not None
                        and (final_npz.exists() or final_npz.is_symlink())):
                    raise ValueError(f"refusing to overwrite {final_npz} (the --oracle-final frame of --output)")
                # fail_fast never overwrites; a stale record would pass for this run's.
                stale = [] if args.output is None else sorted(args.output.parent.glob(
                    glob.escape(args.output.name) + ".rank*.error.json"))
                if stale:
                    raise ValueError(f"refusing to reuse --output: {stale[0]} exists")
            except Exception as exc:
                error = str(exc)
        collective_check(context, error)
        metadata = _objects(context, metadata)[0]
        started = time.perf_counter()
        model_name, checkpoint = model_and_checkpoint(args)
        # The report's names, a file (--model or --checkpoint) resolved: a relative path keeps its location.
        recorded = {key: str(Path(value).resolve()) if value is not None and Path(value).is_file() else value
                    for key, value in (("model", model_name), ("checkpoint", checkpoint))}
        def load():
            return load_adapter(model_name, model_path=checkpoint, device=str(device), enable_compile=False,
                                enable_checkpoint=False, model_dtype=args.dtype, optimization_profile=args.profile)

        # Named weights can populate a shared cache. Explicit checkpoints and MACE .model paths are local loads.
        root_first = (context.world_size > 1 and checkpoint is None and not Path(model_name).is_file()
                      and Path(model_name).parent == Path(".") and not model_name.endswith(".model"))
        adapter = load_on_ranks(load, context, root_first=root_first)
        checkpoint_metadata = json.loads(json.dumps(getattr(adapter, "checkpoint", None), default=str))
        from fastmd.adapters.protocol import ShardDeclaration
        custom_strategy = isinstance(adapter, ShardDeclaration) and adapter.custom_strategy() is not None
        model = adapter.module.eval().requires_grad_(False)

        def synchronize():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def shifted(frame):
            """``frame`` (Atoms or AtomState) rigidly shifted by NOISE_FLOOR_SHIFT_A: an oracle's noise floor."""
            if isinstance(frame, AtomState):
                shift = torch.tensor(NOISE_FLOOR_SHIFT_A, dtype=frame.positions.dtype, device=frame.positions.device)
                return AtomState(frame.ids, frame.positions + shift, frame.numbers)
            frame = frame.copy()
            frame.positions += NOISE_FLOOR_SHIFT_A
            return frame

        def adapter_reference(atoms):
            """Unsharded eager E/F on the whole system ``atoms``, before shard allocation."""
            from fastmd.interfaces.calculator import MLIPCalculator
            reference_adapter = load_adapter(
                model_name, model_path=checkpoint, device=str(device), model_dtype=args.dtype,
                optimization_profile="generic", enable_compile=False, enable_checkpoint=False)
            calculator = MLIPCalculator(model=reference_adapter, task="ef", device=str(device),
                                        config=InferenceConfig(execution="eager", optimization_profile="generic",
                                                               checkpoint="off", report=False))
            def predict():
                from ase.calculators.calculator import all_changes
                calculator.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
                return (float(calculator.results["energy"]) / len(atoms),
                        torch.as_tensor(calculator.results["forces"]).cpu())
            try:
                return None, compute_references(predict, args.precision, args.triton_dot_precision)
            finally:
                calculator.close()

        def calculator_floor(atoms):
            """The calculator's references on ``atoms`` shifted. Its graph is rebuilt there and not compared (the
            calculator does not expose it): a pair at the cutoff that rounds across it enters with the model's
            vanishing cutoff envelope."""
            return adapter_reference(shifted(atoms))[1]

        def original_reference(build_graph, topology=True):
            """Canonical topology (``topology``: unless the single-GPU engine) and E/F references of the
            original model.

            Reference numerics are independent of the tested path: eager
            kernels of ``reference_profile`` and 'highest' GEMMs, plus a
            same-precision rerun when the tested GEMMs are not IEEE (see
            fastmd.precision). The distributed paths run generic; the
            single-GPU engine is compared with single GPU under its own
            resolved profile (G-D1), not with generic-vs-topology kernel
            differences inside the 1e-5 gates.
            """
            from fastmd.distributed.graph import frozen_topology_from_radius_graph

            def predict():
                original = model([build_graph()], task="ef", is_training=False)
                return float(original["e"][0].detach()), original["f"][0].detach().cpu()

            resolved = resolve_config(InferenceConfig(execution="eager", optimization_profile=reference_profile,
                                                     checkpoint="off", report=False))
            with resolved.scope():
                original_topology = None
                if topology and not single_engine:  # its graph comes from this same builder
                    with matmul_precision_scope("highest"):
                        original_topology = frozen_topology_from_radius_graph(build_graph())
                return original_topology, compute_references(predict, args.precision, args.triton_dot_precision)

        parallel = ParallelConfig(domain_grid=tuple(args.grid) if args.grid else "auto",
                                  local_backend=args.local_backend,
                                  auto_execution=auto_execution if custom_strategy else None,
                                  exchange_self_rows=args.exchange_self_rows,
                                  payload_transport=args.payload_transport,
                                  legacy_capacity_policy=args.legacy_capacity_policy,
                                  exchange_scheme=args.exchange_scheme,
                                  topology_engine=args.topology_engine, topology_skin=args.topology_skin,
                                  grad_mechanism=args.grad_mechanism,
                                  debug_replay_check=args.debug_replay_check)
        single_engine = use_single_rank_engine(context.world_size, device)
        reference_profile = (single_rank_profile(args.profile, args.local_backend)[0] if single_engine and custom_strategy
                             else "generic")

        original = None
        noise_floor = None  # the reference on the shifted system, see oracle_report
        reference_s = 0.0
        if args.oracle and (device.type == "cuda" or not custom_strategy):
            # As in benchmark_single_ef, compute and release the single-GPU
            # reference before the evaluator's shards and graph pools occupy
            # rank 0: the oracle peak is then the larger of the two, not their
            # sum (the generic reference needs ~21 KB per triplet, 225 GiB for
            # 16,128 RDX atoms).
            error = None
            if context.rank == 0:
                reference_started = time.perf_counter()
                try:
                    from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu

                    def gpu_graph(frame):
                        return lambda: atoms_to_graph_gpu(
                            frame, atom_graph_cutoff=model.config["pairwise_cutoff"],
                            line_graph_cutoff=model.config["three_body_cutoff"], device=device,
                            check_isolated_atoms=False)
                    original = adapter_reference(atoms) if not custom_strategy else original_reference(gpu_graph(atoms))
                    if not custom_strategy:
                        noise_floor = lambda: calculator_floor(atoms)
                    elif not single_engine:
                        # Called only if the absolute gate fails (then beside the evaluator's shards).
                        def noise_floor():
                            shifted_topology, references = original_reference(gpu_graph(shifted(atoms)))
                            if not same_pairs(shifted_topology, original[0]):
                                raise ValueError("the shifted frame's graph has other pairs or triplets")
                            return references
                    gc.collect()
                    synchronize()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:
                    error = str(exc)
                reference_s = time.perf_counter() - reference_started
            collective_check(context, error)
        if custom_strategy:
            evaluation_config = InferenceConfig(
                execution=None if auto_execution else args.execution, optimization_profile=args.profile, report=False,
                matmul_precision=args.precision, saved_tensor_recompute=args.saved_tensor_recompute,
                triton_dot_precision=args.triton_dot_precision, recompute_policy=args.recompute_policy)
        else:
            evaluation_config, parallel = sharded_configs(args)
        evaluator = DistributedEvaluator(adapter, metadata["cell"], pbc=metadata["pbc"], context=context,
                                         config=evaluation_config, parallel=parallel)
        if custom_strategy and single_engine != (evaluator.engine == "single"):
            raise RuntimeError(f"expected the single-GPU engine={single_engine}, got {evaluator.engine}")
        if custom_strategy and single_engine and evaluator._resolved.optimization_profile != reference_profile:
            raise RuntimeError(f"oracle reference profile {reference_profile!r} differs from the engine's "
                               f"{evaluator._resolved.optimization_profile!r}")
        state = evaluator.distribute(atoms)
        initial = evaluator.evaluate(state, **stress_options)
        initial_gathered = evaluator.gather(initial)
        initial_stress = None if not args.stress else initial.stress.detach().cpu().tolist()
        initial_energy = {"energy_eV": float(initial.energy),
                          "energy_eV_per_atom": float(initial.energy_per_atom),
                          "reference_energy_eV": float(initial.reference_energy)}
        # Excludes the rank-0 reference above; other ranks waited for it.
        initialization_s = time.perf_counter() - started - reference_s
        watchdog.phase("oracle")
        oracle = None
        error = None
        topology = edges = None
        if args.oracle and not custom_strategy:
            edges = sharded_edge_count(evaluator, initial.state.ids, context)
        if args.oracle and not single_engine and custom_strategy:
            from fastmd.distributed.graph import gather_topology
            topology = gather_topology(evaluator.last_graph, context)
        if args.oracle and context.rank == 0:
            oracle_started = time.perf_counter()
            try:
                if original is None:
                    # Independent original model, consuming precisely the frozen
                    # distributed topology rather than legacy CPU cutoff rules.
                    from fastmd.distributed.graph import freeze_topology, radius_graph_from_frozen
                    if context.world_size == 1:
                        frozen, system = freeze_topology([evaluator.last_graph]), initial.state
                    else:
                        # CPU P>1: an independent one-domain build of the
                        # gathered system is both the topology parity
                        # reference and the reference model's graph.
                        from fastmd.distributed.graph import build_local_graph
                        from fastmd.distributed.partition import DomainPartition
                        # Wrapped, like the engine's halo coordinates: the
                        # frozen images refer to them (forces are unchanged).
                        whole = DomainPartition(evaluator.partition.cell, 1)
                        system = AtomState(initial_gathered["ids"], whole.wrap_positions(initial_gathered["positions"]),
                                           initial_gathered["numbers"])
                        local = build_local_graph(system, system, whole, 0, model.config["pairwise_cutoff"],
                                                  model.config["three_body_cutoff"])
                        frozen = freeze_topology([local])
                    original = original_reference(lambda: radius_graph_from_frozen(
                        frozen, system, evaluator.partition.cell))
                    # The same frozen images: a rigid shift needs no rewrap.
                    noise_floor = lambda: original_reference(lambda: radius_graph_from_frozen(
                        frozen, shifted(system), evaluator.partition.cell), topology=False)[1]
                original_topology, references = original
                evidence = None
                if single_engine or not custom_strategy:
                    parity = None
                    oracle = {"canonical_graph_parity": None, "canonical_graph_parity_reason": (
                        "E/F compared to unsharded eager model; canonical MatRIS topology comparison "
                        "does not apply to this engine")}
                    if not custom_strategy:
                        oracle["edge_count_parity"] = edge_evidence(edges, atoms, adapter.topology_spec().cutoff)
                        evidence = oracle["edge_count_parity"]["parity"]
                else:
                    parity = (topology.directed_keys == original_topology.directed_keys
                              and topology.triplet_keys == original_topology.triplet_keys)
                    oracle = {"canonical_graph_parity": parity}
                    evidence = parity
                oracle.update(oracle_report(
                    float(initial.energy_per_atom), initial_gathered["forces"], references,
                    precision_force_rmse_tolerance=tolerance,
                    implementation_tolerance=SINGLE_ENGINE_TOLERANCE if single_engine and custom_strategy else None,
                    reference_profile=reference_profile,
                    # The floor only with positive topology evidence (else a missing edge could hide in it).
                    noise_floor=noise_floor if evidence is True else None))
                oracle["passed"] = parity is not False and oracle["passed"]
                # Other ranks wait in collective_check (--timeout) meanwhile.
                oracle["rank0_seconds"] = reference_s + time.perf_counter() - oracle_started
                if parity is False:
                    raise AssertionError(f"distributed canonical topology differs from original builder: {oracle}")
                if not oracle["passed"]:
                    raise AssertionError(f"original single-GPU oracle failed: {oracle}")
            except Exception as exc:
                error = str(exc)
        collective_check(context, error)
        # The independent oracle is diagnostic only. Do not keep its full GPU
        # graph or saved tensors alive during sharded timing or memory reporting.
        original = original_topology = topology = references = noise_floor = None
        watchdog.phase("warmup")
        gc.collect()
        synchronize()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for _ in range(args.warmup):
            initial = evaluator.evaluate(initial.state, **stress_options)
            watchdog.beat()
        watchdog.phase("timed", long=False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        per_step = []
        for _ in range(args.steps):
            if context.world_size > 1:
                dist.barrier(group=context.group)
            synchronize()
            before = time.perf_counter()
            initial = evaluator.evaluate(initial.state, **stress_options)
            synchronize()
            elapsed = (time.perf_counter() - before) * 1000
            initial.stats["wall_ms"] = elapsed
            from fastmd.distributed.timing import wall_breakdown
            if custom_strategy:
                initial.stats["wall_breakdown_ms"] = wall_breakdown(initial.stats)
            samples.append(max(_objects(context, elapsed)))
            per_step.append(_objects(context, initial.stats))
            watchdog.beat()
        # Native MD and the final oracle release the evaluator. Keep the fixed-frame plan while it is live.
        record_plan = getattr(evaluator, "plan_record", None)
        plan_record = record_plan().as_dict() if callable(record_plan) else None
        final = initial
        md = None
        native_final = None
        if args.md_steps:
            watchdog.phase("md")
            engine, engine_reason = md_engine(args, context.world_size, device)
            if not custom_strategy and args.md_engine == "auto":
                engine, engine_reason = "domain", "shared evaluator driver for a declared model shard"
                if args.steps_per_launch > 1:
                    raise ValueError("declared model shards with the domain driver require --steps-per-launch 1")
            if engine == "native":
                initial = final = None  # the evaluator's graphs and prediction go before the native capture
                dynamics = build_native_dynamics(args, evaluator, adapter, atoms, metadata["resume"])
            else:
                dynamics = build_dynamics(args, evaluator, initial, atoms, metadata["resume"])
                final = dynamics.prediction
            start_step = dynamics.nsteps
            first_thermo = dynamics.thermodynamics()
            md_samples = []
            md_graph_events = []
            md_auto_events = []
            md_gloo_calls = []
            stop_reason = "completed"
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            window = dynamics.steps_per_launch  # MD steps per step() call (native multi-step windows)
            for md_step in range(start_step, args.md_steps, window):
                synchronize()
                before = time.perf_counter()
                step = dynamics.step()
                synchronize()
                elapsed = max(_objects(context, (time.perf_counter() - before) * 1000))
                md_samples.extend([elapsed / window] * window)
                if engine == "native":  # no evaluator (no collectives): runner statistics at the end
                    md_gloo_calls.extend([0] * window)
                else:
                    final = step
                    # Reporting is outside the timed MD step. These are runner
                    # decisions, not replayed Python/kernel invocation counters.
                    event = final.stats.get("model_graph")
                    if event is not None:
                        # Distributed counters are per runner (an auto switch back to model_graph
                        # restarts them); the single engine's are per engine (runner_resets counts drops).
                        md_graph_events.append({"md_step": md_step, **{key: event[key] for key in (
                            "generation", "capture_count", "recapture_count", "replay_count",
                            "captured_this_call", "capacity_changed", "capture_ms", "warmup_ms") if key in event}})
                    auto = final.stats.get("auto_execution")
                    if auto is not None and (not md_auto_events
                                             or auto["decisions"] != md_auto_events[-1]["decisions"]):
                        md_auto_events.append({"md_step": md_step, **auto})
                    # Gloo calls of a step's evaluate: only the control slow paths; the MD
                    # step adds none (the timing gather above is this harness's).
                    communication = final.stats.get("communication", {})
                    md_gloo_calls.append(communication.get("control_slow_path_calls", 0))
                dynamics.write_outputs()
                if args.stop_at_step and dynamics.nsteps >= args.stop_at_step:
                    dynamics.save_restart()
                    stop_reason = "stop_at_step"
                    watchdog.beat()
                    break
                watchdog.beat()
            last_thermo = dynamics.thermodynamics()
            native = None
            if engine == "native":
                native_final = native_report(dynamics)
                effective = dynamics.config_summary()["effective"]
                native = {"execution": effective.get("execution"), "auto_execution": effective.get("auto_execution"),
                          "requested_by": dynamics.engine_reason,
                          "whole_step": dynamics.whole_step_stats(capacity_plan=False),
                          "model_graph": dynamics.model_graph_stats()}
            dynamics.close()
            # "driver" is always "thin" (the one MD driver) since the rollback cleanup; kept for report stability.
            md = {"driver": args.md_driver, "engine": engine, "engine_reason": engine_reason, "native": native,
                  "steps_per_launch": window,
                  "ensemble": args.ensemble, "fix_com": args.fix_com,
                  "dt_fs": args.dt_fs, "steps": args.md_steps, "start_step": start_step,
                  "end_step": dynamics.nsteps, "stop_reason": stop_reason,
                  "thermostat": md_thermostat(args, dynamics), "outputs": None if args.md_output_dir is None else {
                      "directory": str(args.md_output_dir.resolve()), "resumed": metadata["resume"],
                      "thermo_interval": args.thermo_interval, "xyz_interval": args.xyz_interval,
                      "restart_interval": args.restart_interval},
                  "initial": first_thermo,
                  "final": last_thermo, "samples_ms": md_samples,
                  "gloo_calls_per_step_by_rank": _objects(context, md_gloo_calls),
                  "graph_events_by_rank": _objects(context, md_graph_events),
                  # First MD step's auto state, then every step with a new decision (switch).
                  "auto_execution_events_by_rank": _objects(context, md_auto_events),
                  "peak_memory_by_rank": _objects(context, {
                      "allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "reserved_bytes": torch.cuda.max_memory_reserved(device),
                  } if device.type == "cuda" else None),
                  "memory_scope": "MD steps only; peaks reset after the initial MD evaluation; allocator reserved includes graph pool"}
        watchdog.phase("finalize")
        if native_final is None:
            gathered = evaluator.gather(final)
            energies = {"natoms": final.natoms, "energy_eV": float(final.energy),
                        "energy_eV_per_atom": float(final.energy_per_atom),
                        "reference_energy_eV": float(final.reference_energy)}
        else:
            gathered, energies = native_final
        configurations = _objects(context, evaluator.config_summary())
        affinities = _objects(context, affinity)
        final_oracle = None
        final_error = None  # raised after the report is written: a finished run keeps its report
        if args.oracle_final:
            if context.rank == 0 and args.output is not None:
                # Keep a self-contained snapshot even if the independent
                # reference fails, so it can be checked in a fresh process.
                import numpy as np
                frame_path = args.output.with_suffix(".final.npz")
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                with frame_path.open("xb") as stream:
                    np.savez_compressed(stream, cell=metadata["cell"], pbc=metadata["pbc"],
                                        **{key: value.cpu().numpy() for key, value in gathered.items()},
                                        **energies)
            # The final frame's topology evidence (collective, before close): the gathered canonical topology
            # (MatRIS) or the shards' edges; none for one CUDA rank, whose graph is the reference builder's kind.
            final_edges = None
            if native_final is None and custom_strategy and not single_engine:
                from fastmd.distributed.graph import gather_topology
                final_topology = gather_topology(evaluator.last_graph, context)
                final_edges = None if final_topology is None else len(final_topology.directed_keys)
                final_topology = None
            elif native_final is None and not custom_strategy:
                final_edges = sharded_edge_count(evaluator, final.state.ids, context)
            # Release all distributed graphs/hooks collectively before loading
            # the whole-system reference on rank 0. This check is not timed.
            evaluator.close()
            if context.rank == 0:
                try:
                    atoms = atoms.copy()
                    atoms.positions[:] = gathered["positions"].cpu().numpy()
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    _, final_references = adapter_reference(atoms)
                    evidence = edge_evidence(final_edges, atoms, model.config["pairwise_cutoff"] if custom_strategy
                                             else adapter.topology_spec().cutoff)
                    # Two FP32 builders on the same frame: the noise floor applies as in --oracle, with the
                    # same requirement of positive topology evidence.
                    final_oracle = oracle_report(
                        energies["energy_eV_per_atom"], gathered["forces"], final_references,
                        precision_force_rmse_tolerance=tolerance, reference_profile="generic",
                        noise_floor=(lambda: calculator_floor(atoms)) if evidence["parity"] is True else None)
                    final_oracle["edge_count_parity"] = evidence
                    if not final_oracle["passed"]:
                        final_error = f"final-frame single-GPU oracle failed: {final_oracle}"
                except Exception as exc:
                    final_error = str(exc)
                    final_oracle = {"passed": False, "error": final_error}
        if context.rank == 0:
            from fastmd.distributed.timing import summarize_wall_time
            serial = lambda data: {key: value.tolist() for key, value in data.items()}
            report = {"world_size": context.world_size, "natoms": energies["natoms"],
                      "initialization_seconds": initialization_s, "torch": torch.__version__,
                      "cuda": torch.version.cuda, "configurations": configurations,
                      "requested_execution": args.execution, "cpu_affinity_by_rank": affinities,
                      "precision_record": precision_record(args.precision, device, triton_dot_precision=args.triton_dot_precision),
                      **recorded,
                      "model_family": family, "checkpoint_metadata": checkpoint_metadata,
                      "plan_record": plan_record, "plan_record_scope": "fixed-frame evaluations",
                      "stress_eV_A3": initial_stress,
                      "adapter": adapter.name, "input": str(Path(args.input).resolve()),
                      "energy_eV": energies["energy_eV"], "energy_eV_per_atom": energies["energy_eV_per_atom"],
                      "reference_energy_eV": energies["reference_energy_eV"],
                      "oracle": oracle, "final_oracle": final_oracle,
                      "warmup": args.warmup, "samples_ms": samples,
                      "memory_protocol": "oracle tensors released, GC/synchronize/empty_cache before warmup; timed peak reset after warmup",
                      "median_ms": statistics.median(samples) if samples else None,
                      "rank_stats": per_step, "md": md,
                      "wall_timing": summarize_wall_time(per_step) if custom_strategy else {
                          "note": "node/line-halo evaluators report total wall time and their own rank_stats"},
                      "initial_energy": initial_energy,
                      "initial_frame": serial(initial_gathered), "final_frame": serial(gathered)}
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_report(args.output, report)
            print(json.dumps({key: report[key] for key in ("world_size", "natoms", "median_ms", "oracle", "md")}), flush=True)
        if args.oracle_final:
            collective_check(context, final_error)
        watchdog.phase("teardown")
        if evaluator is not None:
            evaluator.close()
        shutdown()
    except BaseException as exc:
        # Never run close()/destroy_process_group() after a failure: both can
        # wait forever on a failed peer (measured 480-1,200 s silent hangs).
        fail_fast(exc, rank=context.rank, phase=watchdog.phase_name, output=args.output)
    watchdog.stop()


if __name__ == "__main__":
    main()
