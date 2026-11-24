"""Single-system spatially sharded inference, with lazily loaded runtime components."""

from fastmd import _lazy

_EXPORTS = {
    "fastmd.distributed.api": ("DistributedEvaluator",),
    "fastmd.execution.md_protocol": ("HeatingRamp",),
    "fastmd.distributed.auto_execution": ("AutoExecution",),
    "fastmd.distributed.config": ("ParallelConfig",),
    "fastmd.distributed.evaluator": ("DistributedMatRISEvaluator", "DistributedPrediction"),
    "fastmd.distributed.exchange": ("ExchangeContext",),
    "fastmd.distributed.partition": ("AtomState", "DomainPartition"),
    "fastmd.distributed.sharded_evaluator": ("ShardedEvaluator",),
}
__all__ = sorted(name for names in _EXPORTS.values() for name in names)
_getattr, __dir__ = _lazy.lazy_attributes(
    __name__, {name: module for module, names in _EXPORTS.items() for name in names})

# One version: a removed name raises its migration. MD at any P is the one driver,
# GPUResidentMolecularDynamics.from_evaluator (unified step 3.5); restart files are unchanged.
_REMOVED = {
    "DistributedMD": (
        "DistributedMD was removed (one MD driver, step 3.5): use "
        "fastmd.applications.GPUResidentMolecularDynamics.from_evaluator(evaluator, state, ensemble=..., "
        "timestep=<dt_fs>, temperature=<temperature_K>, tdamp=<tdamp_fs>, taut=<taut_fs>, "
        "scalar_sync=/ke2_update= -> nhc_scalar_sync=/nhc_ke2_update=, ...); a continuation "
        "(prediction=/start_step=/nhc_state=) -> restart=evaluator.distribute_restart(path); restart "
        "files are unchanged"),
    "load_restart": (
        "load_restart was removed (one MD driver, step 3.5): use evaluator.distribute_restart(path) and "
        "GPUResidentMolecularDynamics.from_evaluator(evaluator, restart=...); restart files are unchanged"),
    "DistributedNVE": (
        "DistributedNVE (the pre-3.3 NVE driver, --md-driver nve_legacy) was removed: use "
        "fastmd.applications.GPUResidentMolecularDynamics.from_evaluator(evaluator, state, ensemble=\"nve\", "
        "timestep=<dt_fs>, fix_com=False), the same trajectory bitwise"),
}


def __getattr__(name):
    if name in _REMOVED:
        raise ImportError(_REMOVED[name], name=name)
    return _getattr(name)
