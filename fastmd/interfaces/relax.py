""" Modified from CHGNet: https://github.com/CederGroupHub/chgnet """

import inspect
import math
import sys
import io
import contextlib
import time
import warnings
from collections.abc import Sequence
from ase import Atoms, units
from ase.optimize.optimize import Optimizer
import ase.filters as filter_classes
from ase.filters import Filter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure

from ..config import InferenceConfig
from ..engine.entry_config import reject_removed_arguments
from .calculator import (
    OPTIMIZERS,
    MLIPCalculator,
    TrajectoryObserver,
    CrystalFeasObserver
)

# The relaxation engines: "gpu" relaxes in whole-step CUDA graphs (FIRE and the cell filter on the device,
# fastmd.interfaces.multi_system_relax), "ase" runs ASE's optimizer on the host with the calculator, "auto" takes the
# GPU path wherever it computes what ASE's would.
ENGINES = ("auto", "gpu", "ase")
# FIRE's keyword arguments that the GPU path takes (FIREParameters; ASE's maxstep None is its default 0.2), and those
# it takes at the values that change nothing it computes (no log file, trajectory file, restart file or downhill check).
GPU_FIRE_ARGUMENTS = ("dt", "maxstep", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a")
GPU_NO_OP_ARGUMENTS = {"logfile": None, "trajectory": None, "restart": None, "downhill_check": False,
                       "position_reset_callback": None, "append_trajectory": False}
# The GPU path's trajectory records are preallocated on the device for every step a structure may take (48 bytes per
# atom and 160 per structure per record, SystemRecords): at most this fraction of the device's memory. A structure
# whose records alone would exceed it relaxes with ASE's optimizer; a list's batch holds fewer slots.
RECORD_MEMORY_FRACTION = 1.0 / 16.0
# The exceptions of a GPU relaxation that relaxes with ASE's optimizer without a warning: a negotiation, a construction
# check or the relaxation's own limits (ValueError, nvalchemiops' NeighborOverflowError), out of memory or a failed
# growth (RuntimeError); any other is warned about (RuntimeWarning) before ASE's optimizer takes over.
EXPECTED_FAILURES = (ValueError, RuntimeError)
# The lean path of one structure: its union's first capture runs after LEAN_WARMUP side-stream warmup window (the
# count of a capture again) instead of capacity.warmup's three, and its windows hold LEAN_STEPS_PER_LAUNCH step. The
# warmups and the capture run their windows' steps at eager speed, which a short relaxation does not pay back (MatRIS
# 10M OAM, 12-48 atoms, warm process: 2.7-3.5 s of construction with three warmups of 5-step windows, 0.61 s with one
# of 1-step windows), and a window's host work is small against a step (MACE-MPA-0, FrechetCellFilter, 100 atoms,
# 301 steps: 0.868 s with 1-step windows, 0.864 s with 5-step ones; SevenNet-0, ORB v3, MatRIS alike).
LEAN_WARMUP = 1
LEAN_STEPS_PER_LAUNCH = 1
# The parts of a MatRIS interaction block that enable_compile compiles (at its first call).
COMPILED_BLOCK_PARTS = ("attn_block_atom_graph", "attn_block_line_graph", "refine_block_atom_graph",
                        "refine_block_line_graph")


class StructOptimizer:
    """This class is used for Sturcture Optimization."""

    def __init__(
        self,
        model_path: str = None,
        model: str = "matris_10m_oam",
        task: str = "efs",
        optimizer: str = "FIRE",
        device: str = "cpu",
        enable_compile: bool = True,
        config: InferenceConfig | None = None,
        *,
        auto_policy: str = "latency",
        engine: str = "auto",
    ) -> None:
        """
        Args:
            model_path (str): Local checkpoint path. If set, this checkpoint is loaded.
            model (MatRIS): Instance of a MatRIS model. If set to None, the default MatRIS is loaded.
            task (str): The prediction task. Can be 'e', 'em', 'ef', 'efs', 'efsm'.
            optimizer (Optimizer,str): choose optimizer from ASE.
            device (str): The device to be used for predictions,
            enable_compile (bool): Whether to compile interaction blocks.
            config: Unified inference settings forwarded to MLIPCalculator.
                Defaults to automatic execution and checkpointing off.
            auto_policy: Defaults to "latency" to skip uncached online memory probes.
                Use "throughput" to permit calibration for longer tasks.
            engine (str): How :meth:`relax` relaxes: "auto" (default) on the GPU path wherever it computes
                what ASE's optimizer would (see :meth:`relax`), else with ASE's optimizer; "ase" always with
                ASE's optimizer; "gpu" on the GPU path or a ValueError naming why not.
        """

        if optimizer in OPTIMIZERS:
            optimizer = OPTIMIZERS[optimizer]
        else:
            raise ValueError(
                f"Optimizer {optimizer} not found. Select from {list(OPTIMIZERS)}"
            )
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")

        self.optimizer: Optimizer = optimizer
        self.engine = engine
        self._config = config
        # The last relax call's path and why (config_summary()["relaxation"]).
        self.last_relaxation: dict | None = None

        self.calculator = MLIPCalculator(
            model_path=model_path,
            model=model,
            task=task,
            device=device,
            enable_compile=enable_compile,
            config=config,
            auto_policy=auto_policy,
        )

    def config_summary(self) -> dict:
        """Return the relaxation calculator's effective inference settings, and under ``"relaxation"`` the
        requested engine and the last relax call's path with its reason."""
        return {**self.calculator.config_summary(),
                "relaxation": {"engine": self.engine, "last": self.last_relaxation}}

    def close(self) -> None:
        """Close the calculator this optimizer made (``MLIPCalculator.close``: its graphs, and the release
        of its model adapter); context exit closes too."""
        self.calculator.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def relax(
        self,
        atoms: Structure | Atoms | Sequence[Structure | Atoms],
        fmax: float = 0.05,
        steps: int = 500,
        relax_cell: bool = True,
        ase_filter: str = "FrechetCellFilter",
        save_path: str = None,
        loginterval: int = 1,
        crystal_feas_save_path: str = None,
        verbose: bool = True,
        assign_magmoms: bool = True,
        engine: str | None = None,
        **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver] | list[dict[str, Structure | TrajectoryObserver]]:
        """
        Args:
            atoms (Structure | Atoms | list): A Structure or Atoms object to relax, or a list of them (one
                result per structure, in order).
            fmax (float): The maximum force tolerance for relaxation.
            steps (int): The maximum number of steps for relaxation.
            relax_cell (bool): Whether to relax the cell as well.
            ase_filter (str): The filter to apply to the atoms object for relaxation.
            save_path (str): The path to save the trajectory (a list of structures: a name with
                ``{index}``, formatted with each structure's index).
            loginterval (int): Interval for logging trajectory and crystal feas.
            crystal_feas_save_path (str): Path to save crystal feature vectors
                which are logged at a loginterval rage (a list: a name with ``{index}``)
            verbose (bool): Whether to print the output of the ASE optimizer (the GPU path: a summary).
            assign_magmoms (bool): Whether to assign magnetic moments to the final
                structure.
            engine (str): "auto", "gpu" or "ase" for this call (default: the optimizer's ``engine``).
            **kwargs: Additional parameters for the optimizer.

        Returns ``{"final_structure": Structure, "trajectory": TrajectoryObserver}`` (a list of them for a list).
        Atoms given are relaxed in place, with the calculator attached.

        The GPU path (engine "auto" or "gpu") relaxes in whole-step CUDA graphs
        (:class:`~fastmd.interfaces.multi_system_relax.MultiSystemRelaxation`: ASE's FIRE per structure, and
        ASE's FrechetCellFilter, on the device) when it computes what ASE's optimizer would: a CUDA calculator
        with execution auto, no activation checkpointing and ``capacity.warmup >= 1``; FIRE with ASE's defaults
        or valid keyword arguments of :class:`~fastmd.execution.system_relax.FIREParameters` (``logfile``,
        ``trajectory``, ``restart`` and ``position_reset_callback`` only as None, ``downhill_check`` and
        ``append_trajectory`` only as False); a fixed cell or ``FrechetCellFilter``; a structure periodic in all
        three directions without constraints; no ``crystal_feas_save_path``; a task with energies and forces; a
        model that the multi-system entry points negotiate (MatRIS, and MACE's ``ScaleShiftMACE``, SevenNet, ORB v3
        and CHGNet); and trajectory records that fit ``RECORD_MEMORY_FRACTION`` of the device's memory. Anything
        else, or a GPU relaxation that cannot start (the model's negotiation, a structure's construction checks),
        stops or fails (a non-finite energy or force), takes ASE's optimizer as before (``engine="gpu"`` raises
        ValueError instead); ``config_summary()["relaxation"]`` and the verbose output name the path and why. On
        the GPU path the trajectory has the same fields at the same steps (energies, forces, stresses with an "s"
        task, positions, cells; magnetic moments only in its last record, from one calculator call on the final
        structure with an "m" task), recorded on the device; MatRIS's interaction blocks run uncompiled there
        (``enable_compile`` applies to ASE's path), and a list relaxes its structures together in the slots of one
        batch.
        """

        reject_removed_arguments(kwargs)
        engine = self.engine if engine is None else engine
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
        valid_filter_names = [
            name
            for name, cls in inspect.getmembers(filter_classes, inspect.isclass)
            if issubclass(cls, Filter)
        ]

        if isinstance(ase_filter, str):
            if ase_filter in valid_filter_names:
                ase_filter = getattr(filter_classes, ase_filter)
            else:
                raise ValueError(
                    f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                )

        options = dict(fmax=fmax, steps=steps, relax_cell=relax_cell, ase_filter=ase_filter, loginterval=loginterval,
                       verbose=verbose, assign_magmoms=assign_magmoms)
        if isinstance(atoms, (Structure, Atoms)):
            atoms = AseAtomsAdaptor().get_atoms(atoms) if isinstance(atoms, Structure) else atoms
            reason, fire = self._gpu_path_reason(atoms, engine, crystal_feas_save_path, kwargs, **options)
            if reason is None:
                results, reasons, seconds = self._relax_on_gpu([atoms], fire, **options)
                reason = reasons[0]
                if reason is None:
                    self._note("gpu", "every condition of the GPU path holds", 1, seconds)
                    if save_path is not None:
                        results[0]["trajectory"].save(save_path)
                    return results[0]
            if engine == "gpu":
                raise ValueError(f"StructOptimizer: engine='gpu' cannot relax this structure: {reason}")
            self._note("ase", reason, 1)
            if verbose:
                print(f"StructOptimizer: ASE optimizer ({reason})")
            return self._relax_with_ase(atoms, save_path=save_path, crystal_feas_save_path=crystal_feas_save_path,
                                        **options, **kwargs)
        return self._relax_list(list(atoms), engine, save_path, crystal_feas_save_path, kwargs, options)

    def _record_bytes(self, atoms: int, *, steps: int, loginterval: int) -> int:
        """The device memory of the trajectory records of ``atoms`` atoms in one slot (SystemRecords)."""
        records = steps // loginterval + 2 if loginterval > 0 else 2
        return records * (48 * atoms + 160)

    def _record_limit(self) -> int:
        import torch
        device = self.calculator._model_device()
        return int(RECORD_MEMORY_FRACTION * torch.cuda.get_device_properties(device).total_memory)

    def _relax_list(self, structures: list, engine: str, save_path, crystal_feas_save_path, kwargs: dict,
                    options: dict) -> list[dict]:
        """``relax`` of a list: the structures the GPU path takes relaxed together, the others one by one with
        ASE's optimizer; one result per structure, in order."""
        for name, path in (("save_path", save_path), ("crystal_feas_save_path", crystal_feas_save_path)):
            if path is not None and "{index" not in str(path):
                raise ValueError(f"StructOptimizer.relax of a list: {name} must name each structure's file with "
                                 "'{index}'")
        save_path = None if save_path is None else str(save_path)
        crystal_feas_save_path = None if crystal_feas_save_path is None else str(crystal_feas_save_path)
        if not all(isinstance(atoms, (Structure, Atoms)) for atoms in structures):
            raise ValueError("StructOptimizer.relax: atoms must be a Structure, an Atoms, or a list of them")
        if not structures:
            self._note(None, "no structures", 0)
            return []
        structures = [AseAtomsAdaptor().get_atoms(atoms) if isinstance(atoms, Structure) else atoms
                      for atoms in structures]
        reasons, fire = [], None
        for index, atoms in enumerate(structures):
            reason, fire_parameters = self._gpu_path_reason(
                atoms, engine, crystal_feas_save_path, kwargs, **options)
            reasons.append(reason)
            fire = fire_parameters if reason is None else fire
        on_gpu = [index for index, reason in enumerate(reasons) if reason is None]
        results: list = [None] * len(structures)
        seconds = None
        if engine == "gpu" and len(on_gpu) < len(structures):
            index = next(index for index, reason in enumerate(reasons) if reason is not None)
            raise ValueError(f"StructOptimizer: engine='gpu' cannot relax structure {index}: {reasons[index]}")
        if on_gpu:
            seconds = {"construction": 0.0, "relaxation": 0.0}
            for index, result, failure in zip(on_gpu, *self._relax_batch([structures[i] for i in on_gpu], fire,
                                                                         seconds, options)):
                results[index], reasons[index] = result, failure
            if engine == "gpu" and any(reasons[index] is not None for index in on_gpu):
                raise ValueError(f"StructOptimizer: engine='gpu' cannot relax the structures: "
                                 f"{next(reasons[index] for index in on_gpu if reasons[index] is not None)}")
            if save_path is not None:
                for index in on_gpu:
                    if reasons[index] is None:
                        results[index]["trajectory"].save(save_path.format(index=index))
        for index, reason in enumerate(reasons):
            if reason is None:
                continue
            if options["verbose"]:
                print(f"StructOptimizer: structure {index}: ASE optimizer ({reason})")
            results[index] = self._relax_with_ase(
                structures[index],
                save_path=None if save_path is None else save_path.format(index=index),
                crystal_feas_save_path=(None if crystal_feas_save_path is None
                                        else crystal_feas_save_path.format(index=index)),
                **options, **kwargs)
        gpu = sum(reason is None for reason in reasons)
        ase = {reason for reason in reasons if reason is not None}
        self._note("gpu" if gpu == len(structures) else "ase" if not gpu else "both",
                   "every condition of the GPU path holds" if not ase else "; ".join(sorted(ase)), len(structures),
                   seconds, gpu=gpu)
        return results

    def _relax_batch(self, structures: list, fire, seconds: dict, options: dict) -> tuple[list, list]:
        """``(results, reasons)`` of ``structures`` relaxed together on the GPU path (``seconds`` accumulates
        the parts' times); a batch that cannot start (one structure's construction check fails the batch) is
        halved until the structures it holds start, so that only those that cannot start alone take ASE's
        optimizer."""
        results, reasons, parts = self._relax_on_gpu(structures, fire, **options)
        for name in seconds:
            seconds[name] += parts[name]
        if len(structures) == 1 or not all(reason is not None and reason.startswith("the GPU relaxation cannot start")
                                           for reason in reasons):
            return results, reasons
        half = len(structures) // 2
        first, second = (self._relax_batch(part, fire, seconds, options) for part in (structures[:half],
                                                                                      structures[half:]))
        return first[0] + second[0], first[1] + second[1]

    def _note(self, path: str, reason: str, structures: int, seconds: dict | None = None, **counts) -> None:
        self.last_relaxation = {"path": path, "reason": reason, "structures": structures, **counts,
                                **({} if seconds is None else {"seconds": seconds})}

    # ------------------------------------------------------------------ the GPU path
    def _gpu_path_reason(self, atoms: Atoms, engine: str, crystal_feas_save_path, kwargs: dict, *, fmax, steps,
                         relax_cell, ase_filter, loginterval, **_) -> tuple[str | None, object]:
        """Why ``relax`` takes ASE's optimizer for ``atoms`` (None: the GPU path applies), and the GPU path's
        FIREParameters."""
        import torch
        from ..adapters.negotiate import check_system_cells
        from ..execution.system_relax import FIREParameters
        calculator = self.calculator
        if engine == "ase":
            return "engine='ase'", None
        if torch.device(calculator.device).type != "cuda" or not torch.cuda.is_available():
            return f"device {calculator.device!r} (the GPU path runs on CUDA)", None
        if self.optimizer is not OPTIMIZERS["FIRE"]:
            return f"optimizer {self.optimizer.__name__} (the GPU path runs FIRE)", None
        other = sorted(name for name, value in kwargs.items() if name not in GPU_FIRE_ARGUMENTS and not (
            name in GPU_NO_OP_ARGUMENTS and value is GPU_NO_OP_ARGUMENTS[name]))
        if other:
            return f"FIRE arguments {other} (the GPU path takes {list(GPU_FIRE_ARGUMENTS)})", None
        parameters = {name: value for name, value in kwargs.items()
                      if name in GPU_FIRE_ARGUMENTS and not (name == "maxstep" and value is None)}
        try:
            fire = FIREParameters(**parameters)
        except (TypeError, ValueError) as error:
            return f"FIRE arguments outside the GPU path's ({error})", None
        if relax_cell and ase_filter is not filter_classes.FrechetCellFilter:
            return (f"cell filter {getattr(ase_filter, '__name__', repr(ase_filter))} (the GPU path runs "
                    "FrechetCellFilter)"), None
        if crystal_feas_save_path:
            return "crystal_feas_save_path (crystal features come from the calculator's calls)", None
        if not all(atoms.get_pbc()):
            return "not periodic in all three directions", None
        if atoms.constraints:
            return f"constraints ({', '.join(type(c).__name__ for c in atoms.constraints)})", None
        if len(atoms) == 0:
            return "no atoms", None
        if not ("e" in calculator.task and "f" in calculator.task):
            return f"task {calculator.task!r} (FIRE needs energies and forces)", None
        if (isinstance(steps, bool) or not isinstance(steps, int) or steps < 0 or isinstance(fmax, bool)
                or not isinstance(fmax, (int, float)) or not math.isfinite(fmax) or fmax <= 0):
            return f"steps={steps!r}, fmax={fmax!r} (the GPU path takes an integer >= 0 and a positive fmax)", None
        if isinstance(loginterval, bool) or not isinstance(loginterval, int):
            return f"loginterval={loginterval!r} (the GPU path takes an integer)", None
        if not calculator._execution_auto:
            return f"execution={calculator._inference_config.execution!r} requested", None
        if calculator._inference_config.checkpoint_enabled is not False:
            return "activation checkpointing (not validated inside CUDA graph capture)", None
        if calculator._inference_config.capacity.warmup < 1:
            return "capacity.warmup=0 (a capture needs a warmup)", None
        needed = self._record_bytes(len(atoms), steps=steps, loginterval=loginterval)
        if needed > self._record_limit():
            return (f"trajectory records of {needed / 2**30:.1f} GiB on the device (steps={steps}, "
                    f"loginterval={loginterval}; at most {RECORD_MEMORY_FRACTION:.4g} of the device's memory)"), None
        try:
            check_system_cells(calculator.adapter)
        except ValueError as error:
            return f"the model's negotiation: {error}", None
        return None, fire

    @contextlib.contextmanager
    def _uncompiled(self):
        """MatRIS's interaction blocks with ``enable_compile`` run eagerly in the scope, as a model loaded without
        it (whole-step CUDA graphs capture the eager modules): not compiled at their first call, and blocks
        compiled already (by ASE's path) with their original modules and without the compiled blocks' flags;
        everything is put back at the scope's end."""
        undo = []
        for block in getattr(self.calculator.adapter.module, "interaction_block", ()):
            if getattr(block, "enable_compile", False):
                undo.append((block, "enable_compile", True))
                block.enable_compile = False
            if getattr(block, "_is_compiled", False):  # its layer wrappers run compiled regions while it is set
                undo.append((block, "_is_compiled", True))
                block._is_compiled = False
            for name in COMPILED_BLOCK_PARTS:
                part = getattr(block, name, None)
                original = getattr(part, "_orig_mod", None)
                if original is None:
                    continue
                undo.append((block, name, part))
                setattr(block, name, original)
                if getattr(original, "_compiled_block", False):
                    undo.append((original, "_compiled_block", True))
                    original._compiled_block = False
        try:
            yield
        finally:
            for owner, name, value in reversed(undo):
                setattr(owner, name, value)

    def _relax_on_gpu(self, structures: list[Atoms], fire, *, fmax, steps, relax_cell, ase_filter, loginterval,
                      verbose, assign_magmoms) -> tuple[list, list, dict | None]:
        """The structures relaxed together by MultiSystemRelaxation: ``(results, reasons, seconds)``, one result
        or None and one reason or None per structure: why a structure takes ASE's optimizer after all (the
        relaxation cannot start, or stopped before the structure finished: a cell compressed beyond what its
        union holds, a capacity growth that failed). One structure: one slot of its own atoms, no self-check of
        the model's union declaration (the union is the structure) and no draining."""
        import gc
        import torch
        from ..config import current_config
        from ..execution.replica_batch import DEFAULT_UNION_ROWS
        from .multi_system_relax import MultiSystemRelaxation
        calculator = self.calculator
        options = dict(fmax=float(fmax), max_steps=int(steps), cell_filter="frechet" if relax_cell else None,
                       fire=fire, config=self._config, device=calculator._model_device(),
                       trajectory_interval=int(loginterval), trajectory_stress="s" in calculator.task)
        if len(structures) == 1:
            options.update(slots=1, drain=False, self_check=False, steps_per_launch=LEAN_STEPS_PER_LAUNCH)
            if self._config is not None or current_config() is None:  # (an outer configured scope's settings stay)
                from dataclasses import replace
                config = self._config if self._config is not None else InferenceConfig(report=False)
                capacity = replace(config.capacity, warmup=max(1, min(config.capacity.warmup, LEAN_WARMUP)))
                options["config"] = replace(config, capacity=capacity)
        else:
            # The batch's slots hold the records of their rows: fewer rows where those of every structure (at most
            # one default union's) would exceed the records' share of the device's memory.
            rows = min(sum(len(atoms) for atoms in structures), DEFAULT_UNION_ROWS)
            records = self._record_bytes(0, steps=int(steps), loginterval=int(loginterval)) // 160
            per_row = records * (48 + 160 / min(len(atoms) for atoms in structures))  # a slot's 160 B over its rows
            if rows * per_row > self._record_limit():
                options["union_rows"] = max(max(len(atoms) for atoms in structures),
                                            int(self._record_limit() // per_row))
        failure, relaxed, stats, relaxation, unexpected = None, [None] * len(structures), None, None, False
        started = constructed = time.perf_counter()
        with self._uncompiled():
            try:
                relaxation = MultiSystemRelaxation(calculator.adapter, structures, **options)
            except Exception as error:  # noqa: BLE001 - a negotiation, a construction check, out of memory: ASE's path
                failure = (f"the GPU relaxation cannot start: {type(error).__name__}: "
                           f"{str(error).splitlines()[0][:300] if str(error) else ''}")
                unexpected = (not isinstance(error, EXPECTED_FAILURES)
                              and type(error).__name__ != "NeighborOverflowError")
            if relaxation is not None:
                constructed = time.perf_counter()
                try:
                    relaxed = relaxation.run()
                except Exception as error:  # noqa: BLE001 - e.g. a cell compressed beyond its union: ASE's path
                    failure = (f"the GPU relaxation stopped: {type(error).__name__}: "
                               f"{str(error).splitlines()[0][:300] if str(error) else ''}")
                    unexpected = (not isinstance(error, EXPECTED_FAILURES)
                                  and type(error).__name__ != "NeighborOverflowError")
                    relaxed = list(relaxation.results)
                finally:
                    try:
                        stats = relaxation.stats()
                    except Exception:  # noqa: BLE001 - statistics of a relaxation that failed are not needed
                        stats = None
                    finally:
                        relaxation.close()
                del relaxation
            if failure is not None:  # outside the except blocks: a traceback would pin the failed graphs
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:  # (a CUDA error that ASE's path then meets too)
                    pass
                if unexpected:
                    warnings.warn(f"StructOptimizer: {failure}; relaxing with ASE's optimizer", RuntimeWarning,
                                  stacklevel=3)
            reasons = [failure if result is None else
                       f"the GPU relaxation failed at step {result.steps} (a non-finite energy or force)"
                       if result.status == "failed" else None for result in relaxed]
            results = [None if reason is not None else
                       self._gpu_result(atoms, result, relax_cell=relax_cell, assign_magmoms=assign_magmoms)
                       for atoms, result, reason in zip(structures, relaxed, reasons)]
        ran = time.perf_counter()
        seconds = {"construction": constructed - started, "relaxation": ran - constructed}
        if verbose and stats is not None:
            what = "FIRE + FrechetCellFilter" if relax_cell else "FIRE, fixed cell"
            done = [result for result, reason in zip(relaxed, reasons) if reason is None]
            if len(structures) == 1 and done:
                result = done[0]
                print(f"StructOptimizer: GPU relaxation ({what}): {result.status} in {result.steps} steps, fmax "
                      f"{result.fmax:.4g} eV/A, energy {result.energy:.6f} eV; {ran - started:.3f} s "
                      f"(construction {seconds['construction']:.3f} s)")
            elif done:
                counts = ", ".join(f"{count} {name}" for name, count in stats["finished"].items() if count)
                print(f"StructOptimizer: GPU relaxation ({what}) of {len(structures)} structures: {counts}; steps "
                      f"{min(r.steps for r in done)}-{max(r.steps for r in done)}; {ran - started:.3f} s "
                      f"(construction {seconds['construction']:.3f} s)")
        return results, reasons, seconds

    def _gpu_result(self, atoms: Atoms, result, *, relax_cell: bool, assign_magmoms: bool) -> dict:
        """``relax``'s result of a structure relaxed on the GPU path: ``atoms`` relaxed in place with the
        calculator attached, its trajectory from the device's records plus the final state (as ASE's path
        records it after the run), the final structure with the model's magnetic moments (an "m" task: one
        calculator call on it)."""
        from ase.stress import full_3x3_to_voigt_6_stress
        calculator = self.calculator
        final = result.atoms
        if relax_cell:
            atoms.set_cell(final.cell.array, scale_atoms=False)
        atoms.set_positions(final.positions)
        atoms.calc = calculator
        magmoms = None
        if "m" in calculator.task:
            magmoms = calculator.get_magnetic_moments(atoms)
        records = result.trajectory
        observer = TrajectoryObserver(atoms)
        stresses = "s" in calculator.task
        cell = atoms.get_cell()[:]
        for k in range(len(records["steps"])):
            observer.energies.append(float(records["energies"][k]))
            observer.forces.append(records["forces"][k])
            observer.stresses.append(full_3x3_to_voigt_6_stress(records["stresses"][k]) if stresses else None)
            observer.magmoms.append(None)
            observer.atom_positions.append(records["positions"][k])
            observer.cells.append(records["cells"][k] if relax_cell else cell.copy())
        observer.energies.append(float(result.energy))
        observer.forces.append(result.forces.copy())
        observer.stresses.append(full_3x3_to_voigt_6_stress(records["final_stress"]) if stresses else None)
        observer.magmoms.append(magmoms)
        observer.atom_positions.append(atoms.get_positions())
        observer.cells.append(cell.copy())
        struct = AseAtomsAdaptor.get_structure(atoms)
        if assign_magmoms and magmoms is not None:
            for key in struct.site_properties:
                struct.remove_site_property(property_name=key)
            struct.add_site_property("magmom", [float(magmom) for magmom in magmoms])
        return {"final_structure": struct, "trajectory": observer}

    # ------------------------------------------------------------------ ASE's optimizer
    def _relax_with_ase(self, atoms, *, fmax, steps, relax_cell, ase_filter, save_path, loginterval,
                        crystal_feas_save_path, verbose, assign_magmoms, **kwargs) -> dict:
        """``relax`` of one structure with ASE's optimizer (the path of every relaxation before the GPU path)."""
        atoms.calc = self.calculator

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)

            if relax_cell:
                atoms = ase_filter(atoms)
            with self.optimizer(atoms, **kwargs) as optimizer:
                optimizer.attach(obs, interval=loginterval)

                if crystal_feas_save_path:
                    optimizer.attach(cry_obs, interval=loginterval)

                optimizer.run(fmax=fmax, steps=steps)
                obs()

        if save_path is not None:
            obs.save(save_path)

        if crystal_feas_save_path:
            cry_obs.save(crystal_feas_save_path)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)

        if assign_magmoms:
            if atoms.get_magnetic_moments() is not None:
                for key in struct.site_properties:
                    struct.remove_site_property(property_name=key)
                struct.add_site_property(
                    "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
                )

        return {"final_structure": struct, "trajectory": obs}
