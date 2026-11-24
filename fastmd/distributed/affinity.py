"""NUMA-local CPU affinity for the ranks of the distributed launcher.

Measured on 4x GB300 (1,344 atoms, model_graph native): unpinned ranks took
109.45 ms/step against 102.39 ms with each rank pinned to CPUs local to its GPU,
the 7% spread across every host phase. The launcher therefore pins each rank by
default: the CPUs local to its GPU (sysfs ``local_cpulist`` of the PCI device),
restricted to the CPUs this process may use, split into disjoint whole physical
cores among the local ranks whose GPUs share those CPUs. Memory follows by
first touch, so the launcher then re-executes the rank once (same PID and
argv): memory touched before the binding (imports, CUDA initialization) kept
in-process binding 1.0% slower than ``taskset`` at exec (101.28 vs 100.29 ms).

Nothing is changed for CPU runs, when disabled (``--cpu-affinity off`` or
``MATRIS_DISTRIBUTED_CPU_AFFINITY=off``), when OpenMP thread placement is
configured, or when the inherited affinity is already narrower than the cgroup
cpuset (a launcher or user binding is kept).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ENVIRONMENT = "MATRIS_DISTRIBUTED_CPU_AFFINITY"
MODES = ("auto", "off")
# Carries the binding record across the launcher's one re-exec.
REEXEC_ENVIRONMENT = "MATRIS_DISTRIBUTED_CPU_AFFINITY_REEXEC"


def parse_cpulist(text: str) -> set[int]:
    """Parse a kernel CPU list such as ``0-35,72,80-81``."""
    cpus = set()
    for part in text.strip().split(","):
        if part:
            low, _, high = part.partition("-")
            cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def format_cpulist(cpus) -> str:
    runs, cpus = [], sorted(cpus)
    for cpu in cpus:
        if runs and cpu == runs[-1][1] + 1:
            runs[-1][1] = cpu
        else:
            runs.append([cpu, cpu])
    return ",".join(f"{low}" if low == high else f"{low}-{high}" for low, high in runs)


def cgroup_cpus(root: Path = Path("/")) -> set[int] | None:
    """Effective cpuset of this process's cgroup (v2, then v1); None if unknown.

    A v2 cgroup has cpuset files only where its parent enables the controller;
    the nearest ancestor that has them constrains it. Walking up also reaches
    the mount root when a container without a cgroup namespace lists a host
    path that does not exist inside it.
    """
    try:
        lines = (root / "proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    candidates = []
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        parts = [part for part in path.split("/") if part]
        if hierarchy == "0" and not controllers:
            base, names = root / "sys/fs/cgroup", ("cpuset.cpus.effective",)
        elif "cpuset" in controllers.split(","):
            base, names = root / "sys/fs/cgroup/cpuset", ("cpuset.effective_cpus", "cpuset.cpus")
        else:
            continue
        for depth in range(len(parts), -1, -1):
            candidates += [base.joinpath(*parts[:depth], name) for name in names]
    for candidate in candidates:
        try:
            cpus = parse_cpulist(candidate.read_text())
        except (OSError, ValueError):
            continue
        if cpus:
            return cpus
    return None


def gpu_pci_address(device_index: int) -> str:
    import torch
    props = torch.cuda.get_device_properties(device_index)
    return f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"


def pci_local_cpus(address: str, root: Path = Path("/")) -> tuple[set[int], int]:
    device = root / "sys/bus/pci/devices" / address
    cpus = parse_cpulist((device / "local_cpulist").read_text())
    return cpus, int((device / "numa_node").read_text())


def cpu_cores(cpus, root: Path = Path("/")) -> list[set[int]] | None:
    """Physical cores (SMT sibling sets) covering ``cpus``; None if unknown."""
    cores = {}
    try:
        for cpu in cpus:
            siblings = parse_cpulist((root / f"sys/devices/system/cpu/cpu{cpu}/topology/"
                                      "thread_siblings_list").read_text())
            cores[min(siblings | {cpu})] = siblings | {cpu}
    except (OSError, ValueError):
        return None
    return list(cores.values())


def plan_cpus(local_rank: int, device_cpus: list[set[int]], allowed: set[int],
              cores: list[set[int]] | None = None):
    """Pure placement: (cpus or None, local ranks sharing them, reason).

    Local ranks whose GPUs have the same allowed local CPUs split them into
    contiguous, disjoint, balanced chunks of whole physical cores (``cores``;
    each CPU its own core if None) in local-rank order, so no two ranks share
    an SMT core.
    """
    local = [cpus & allowed for cpus in device_cpus]
    mine = local[local_rank]
    if not mine:
        return None, [], "no allowed CPU is local to this rank's GPU"
    sharing = [rank for rank, cpus in enumerate(local) if cpus == mine]
    groups = [core & mine for core in cores or () if core & mine]
    covered = set().union(*groups)
    groups = sorted(groups + [{cpu} for cpu in mine - covered], key=min)
    if len(groups) < len(sharing):
        return set(mine), sharing, "fewer local cores than ranks sharing them; not split"
    size, extra = divmod(len(groups), len(sharing))
    index = sharing.index(local_rank)
    start = index * size + min(index, extra)
    return (set().union(*groups[start:start + size + (index < extra)]), sharing,
            f"GPU-local cores split among local ranks {sharing}")


def _set_all_threads(cpus: set[int]) -> None:
    """Like ``taskset -a``: threads already started (CUDA, OpenMP) move too."""
    os.sched_setaffinity(0, cpus)
    for task in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(task), cpus)
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            pass  # exited meanwhile, or not ours to move


def resolve_mode(requested: str | None) -> str:
    mode = requested if requested is not None else os.environ.get(ENVIRONMENT, "auto")
    if mode not in MODES:
        raise ValueError(f"CPU affinity must be one of {MODES}, got {mode!r}"
                         + ("" if requested is not None else f" from {ENVIRONMENT}"))
    return mode


def _reexec(record: dict) -> dict:
    """Restart this process image on its new CPUs; returns only on failure."""
    argv = getattr(sys, "orig_argv", None)
    if not argv or not sys.executable:
        return {**record, "method": "in-process", "reexec": "unavailable: no sys.orig_argv or sys.executable"}
    try:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None:
                stream.flush()
        os.environ[REEXEC_ENVIRONMENT] = json.dumps(record)
        os.execv(sys.executable, [sys.executable, *argv[1:]])
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        os.environ.pop(REEXEC_ENVIRONMENT, None)
        return {**record, "method": "in-process", "reexec": f"failed: {exc!r}"}


def bind_local_rank(requested: str | None = None, *, device: str = "cuda",
                    root: Path = Path("/"), reexec: bool = False) -> dict:
    """Pin this torchrun rank to CPUs local to its GPU; returns a JSON record.

    Call before process groups and thread pools start. The rank's GPU is CUDA
    device ``LOCAL_RANK``, as in ``launch.initialize``. Failures to discover
    the topology leave the affinity unchanged and are reported, never raised.
    With ``reexec`` a successful binding re-executes the process (same argv),
    whose own call then returns the carried record.
    """
    carried = os.environ.pop(REEXEC_ENVIRONMENT, None)
    mode = resolve_mode(requested)
    record = {"mode": mode, "applied": False}
    try:
        current = os.sched_getaffinity(0)
    except (AttributeError, OSError) as exc:
        return {**record, "reason": f"CPU affinity unavailable: {exc}"}
    if carried is not None:
        try:
            record = json.loads(carried)
        except ValueError:
            record = {}
        if record.get("applied") and record.get("cpus") == format_cpulist(current):
            return {**record, "method": "re-exec"}
        record = {"mode": mode, "applied": False}
    record["inherited_cpus"] = format_cpulist(current)

    def keep(reason):
        return {**record, "cpus": format_cpulist(current), "reason": reason}

    if mode == "off":
        return keep(f"disabled (--cpu-affinity off or {ENVIRONMENT}=off)")
    if device != "cuda":
        return keep("no GPU: CPU device")
    # libgomp computes its places from the affinity at library load (before
    # this call), so it would pin later workers outside the rank's CPUs.
    placement = [name for name in ("OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")
                 if os.environ.get(name, "false").strip().lower() not in ("", "false")]
    if placement:
        return keep(f"OpenMP thread placement is configured ({', '.join(placement)}); "
                    "bind with taskset/numactl before launch instead")
    allowed = cgroup_cpus(root)
    if allowed is not None:
        # Isolated CPUs are in the cpuset but never in a default affinity.
        try:
            allowed -= parse_cpulist((root / "sys/devices/system/cpu/isolated").read_text())
        except (OSError, ValueError):
            pass
    record["cgroup_cpus"] = None if allowed is None else format_cpulist(allowed)
    if allowed is not None and current != allowed:
        return keep("inherited affinity is narrower than the cgroup cpuset; launcher binding kept")
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1" if "LOCAL_RANK" not in os.environ
                                         else os.environ.get("WORLD_SIZE", "1")))
        import torch
        if not 0 <= local_rank < local_world <= torch.cuda.device_count():
            return keep(f"local rank {local_rank} of {local_world} does not map to one of "
                        f"{torch.cuda.device_count()} visible GPUs")
        addresses = [gpu_pci_address(index) for index in range(local_world)]
        topology = [pci_local_cpus(address, root) for address in addresses]
    except (OSError, RuntimeError, ValueError, AttributeError) as exc:
        return keep(f"GPU CPU locality unavailable: {exc}")
    cpus, sharing, reason = plan_cpus(local_rank, [cpus for cpus, _ in topology], current,
                                      cpu_cores(current, root))
    record.update(gpu_pci=addresses[local_rank], numa_node=topology[local_rank][1],
                  gpu_local_cpus=format_cpulist(topology[local_rank][0]),
                  shared_with_local_ranks=sharing)
    if cpus is None:
        return keep(reason)
    if cpus == current:
        return keep(f"{reason}; equals the inherited affinity, unchanged")
    try:
        _set_all_threads(cpus)
    except OSError as exc:
        return keep(f"sched_setaffinity failed: {exc}")
    record = {**record, "applied": True, "cpus": format_cpulist(os.sched_getaffinity(0)),
              "reason": reason}
    return _reexec(record) if reexec else {**record, "method": "in-process"}
