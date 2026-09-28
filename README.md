# fastMD

**Keep using ASE's `Atoms`, optimizers, and molecular dynamics. Just change the calculator.**

fastMD provides a common interface to CUDA Graph inference backends for MatRIS,
CHGNet, ALIGNN, and MACE. On a GPU, it attempts to capture and replay energy and force
calculations by default. On a CPU, or for unsupported properties, it uses eager
inference and reports the reason. Capacity buckets, warmup, and kernel fusion
have defaults, so everyday use requires no CUDA Graph tuning.

```python
from ase.build import bulk
from fastmd import FastMDCalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = FastMDCalculator("chgnet")

print(atoms.get_potential_energy())  # Total energy, eV
print(atoms.get_forces())            # (N, 3), eV/Å
```

This package is independent of the adjacent `MatRIS-09bk` directory. Installation
and use do not require switching old branches, changing `PYTHONPATH`, or
installing overlapping source repositories. CHGNet 0.3.0 weights are
bundled, so the example above can run offline.

## 1. Installation

Python 3.11 or newer is required. Install an appropriate PyTorch build in your
environment first, then install the dependencies for your model:

```bash
cd fastMD
python -m pip install -e '.[chgnet]'
# Alternatively, install dependencies for other models
python -m pip install -e '.[matris]'
python -m pip install -e '.[alignn]'
python -m pip install -e '.[mace]'
```

CUDA Graph requires an NVIDIA GPU, CUDA-enabled PyTorch 2.8 or newer, and a matching
Triton version. MatRIS, CHGNet, and ALIGNN also use the GPU neighbor-list operators:

```bash
python -m pip install -e '.[matris,chgnet,cuda]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For MACE, the GPU neighbor list is implemented directly in Triton. Warp,
NVIDIA neighbor operators, DGL, and TorchSim are not needed:

```bash
python -m pip install -e '.[mace-cuda]'
```

The MACE extra pins `mace-torch==0.3.16` and `e3nn==0.4.4`, matching the migrated
forward implementation. MatRIS's `cuda` extra pins `nvalchemi-toolkit-ops==0.2.0`
and `warp-lang==1.10.0`; do not replace these with the newer versions used by the
original MACE benchmark's TorchSim environment. MACE and MatRIS can use the same
fastMD environment without TorchSim.

The `cuda` extra does not guarantee that an existing CPU-only PyTorch installation
will be replaced with the appropriate CUDA build. Install
[PyTorch](https://pytorch.org/get-started/locally/) for your machine first.
CUDA Graph itself is provided by PyTorch. CPU inference does not require Warp,
Triton, or NVIDIA operators, and does not require compiling the old Cython/CUDA
extensions.

ALIGNN also requires DGL. The declared dependency uses the DGL 1.x interface; CUDA
execution requires a DGL build compatible with your PyTorch/CUDA environment.
Validate these dependencies in a separate ALIGNN environment before attempting
to combine existing model environments.

## 2. Models and checkpoints

All models use `FastMDCalculator(model, checkpoint=..., device=...)`.

| Model | Default or local checkpoint | Eager properties | CUDA Graph properties |
| --- | --- | --- | --- |
| `matris` | Downloads `matris_10m_oam` by default; or a local `.pth.tar` file | Energy, forces, stress, magnetic moments | Energy, forces, stress, magnetic moments |
| `chgnet` | Bundled 0.3.0 weights; or a local `.pth.tar` file | Energy, forces, stress, magnetic moments | Energy, forces |
| `alignn` | Requires a directory containing `config.json` and `best_model.pt` | Energy, forces | Energy, forces |
| `mace` | Downloads MACE-MPA-0 medium by default; or a local `.model` file | Energy, forces, stress | Energy, forces, stress for supported MP-0/MPA-0 architectures |

This table describes the implemented interfaces. See the
[validation record](docs/validation.md) for what has been tested on the current
node. The CUDA paths were migrated from the original branches and still need
the numerical consistency checks below on your target GPU.

```python
from fastmd import FastMDCalculator

# Local MatRIS checkpoint, fully offline
calc = FastMDCalculator("matris", checkpoint="../checkpoint/MatRIS_10M_OAM.pth.tar")

# CHGNet 0.3.0 is bundled by default
calc = FastMDCalculator("chgnet")
# Your own trained CHGNet checkpoint
calc = FastMDCalculator("chgnet", checkpoint="checkpoints/my_chgnet.pth.tar")

# ALIGNN expects a directory, not a single .pt file
calc = FastMDCalculator("alignn", checkpoint="checkpoints/v12.2.2024_dft_3d_307k")

# MACE: local weights, double precision, fused GPU inference by default
calc = FastMDCalculator("mace", checkpoint="checkpoints/mace-mpa-0-medium.model")
# Or let upstream MACE download MP-0 medium
calc = FastMDCalculator("mace", model_kwargs={"model_name": "medium"})
```

MatRIS retains the original download mechanism, with a cache at `~/.cache/matris`.
Use `model_kwargs={"model_name": "matris_10m_mp"}` to select the other supported
default model. For other CHGNet versions, provide `checkpoint`; only 0.3.0 is
bundled.

The ALIGNN adapter currently supports `alignn_atomwise` potentials. Capture
requires the `radius_graph` neighbor strategy, at least one ALIGNN layer, and a
supported model configuration. Unsupported configurations explicitly fall back
to eager inference in `auto` mode.

ALIGNN retains the original branch's total-energy and force-scaling conventions.
For other training configurations, use `model_kwargs` to set `intensive`,
`force_multiplier`, `force_mult_natoms`, and `force_mult_batchsize`, and check the
scaling against your reference forces. Their defaults are `True`, `1.0`, `False`,
and `True`, respectively.

The local `v12.2.2024_dft_3d_307k` configuration has `batch_size=6`. With the
original interface's defaults, the returned forces are six times the unscaled
negative energy gradient. fastMD retains this behavior for compatibility with
the original branch. To obtain forces consistent with the returned energy's
negative gradient, explicitly set
`model_kwargs={"force_mult_batchsize": False}` and verify the checkpoint's
training and inference conventions. In particular, do not assume that the
original default scaling guarantees energy conservation in NVE simulations.

## 3. Using standard ASE workflows

### Single-point calculations

```python
from ase.io import read
from fastmd import FastMDCalculator

atoms = read("POSCAR")
atoms.calc = FastMDCalculator("chgnet", device="auto")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
# CHGNet stress and magnetic moments use eager inference in default auto mode
stress = atoms.get_stress()                 # xx, yy, zz, yz, xz, xy; eV/Å³
magmoms = atoms.get_magnetic_moments()       # μB
print(atoms.calc.stats())                   # Actual mode, fallback reason, cache statistics
```

All backends return total energy; you do not need to multiply it by the atom
count. `free_energy` equals `energy`, supporting ASE optimizers that request
`force_consistent=True`. No additional electronic-temperature free-energy
correction is applied.

The four adapters currently target fully periodic materials with a nonzero
cell volume (`pbc=True`). Nonperiodic molecules and partially periodic systems
raise an explicit error. The adapters do not silently alter your vacuum spacing,
cell, or PBC settings.

### Molecular dynamics

```python
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from fastmd import FastMDCalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms.calc = FastMDCalculator("chgnet")
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))

# Optional: warm up and capture the current geometry before timing. Atoms do not move.
atoms.calc.warmup(atoms)

dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300,
               friction=0.01 / units.fs, trajectory="md.traj",
               logfile="md.log", loginterval=10)
dyn.run(1000)
print(atoms.calc.stats())
```

Use standard ASE units: multiply a time step in femtoseconds by `units.fs`, and
pass temperature in kelvin through `temperature_K`. ASE constraints,
`dyn.attach()`, trajectory output, and NVE/NVT workflows work as usual.

The acceleration here applies to **energy/force inference**. MACE also captures its GPU neighbor-list maintenance. ASE still runs the
integrator and Python loop, and each call still involves host/device data
transfers. This release does not expose the original branches' whole-step GPU MD
as ASE MD or capture the entire MD loop in a CUDA Graph. Initial capture has a
setup cost, so short jobs may not benefit. Benchmark your own system.

### MatRIS and MACE with ASE NPT

Enable `compute_stress` so every evaluation returns energy, forces and stress
together. ASE can then reuse these results when its integrator requests them
separately, without switching between force-only and stress captures:

```python
from ase.md.nptberendsen import NPTBerendsen

# Reuse the initialized periodic atoms and velocities from the MD example above.
atoms.calc = FastMDCalculator(
    "matris", device="cuda", cuda_graph=True,  # Use "mace" for MACE-MPA-0 medium
    model_kwargs={"compute_stress": True},
)
atoms.calc.warmup(atoms)
dyn = NPTBerendsen(
    atoms, timestep=units.fs, temperature_K=300,
    pressure_au=0.0 * units.GPa,
    compressibility_au=1 / (100 * units.GPa),  # Illustrative value for Si
    taut=100 * units.fs, taup=1000 * units.fs,
)
dyn.run(1000)
```

MatRIS rebuilds neighbors using the current cell and copies the new cell,
coordinates and topology into fixed-address graph inputs. Cell changes reuse
captures while the topology fits a cached capacity bucket; larger topologies
capture another bucket. Changes in composition, atom count or PBC still
invalidate captures. With stable capacity, `calc.stats()["cache"]["captures"]`
should stop increasing as the cell evolves.

MACE supports the same `compute_stress=True` option. It updates the cell,
inverse cell, periodic shifts and padding in fixed-address buffers and rebuilds
the GPU candidate list whenever the cell changes. Existing captures remain
usable while the neighbor capacity and periodic-image loop bounds suffice.
Compression or shear that requires a larger image range triggers recapture;
capacity overflow grows buffers and retries the same geometry. Atom count,
species or PBC changes still invalidate the runner.

The barostat and integration run in ASE; kinetic stress is added by ASE. The
calculator returns potential stress in eV/Å³. Berendsen coupling is useful for
pressure equilibration but does not reproduce exact NPT fluctuations. Choose
the compressibility and coupling times for your material and workflow.

Run `python examples/npt.py --checkpoint /path/to/MatRIS.pth.tar` on a CUDA GPU,
or add `--device cpu --eager` for a CPU check. Variable-cell capture/replay and
short ASE NPT runs have been checked on an H100 with MatRIS 10M OAM; see the
[NPT GPU report](matris_npt.md) for measured speedups and the model's hard-cutoff
limitation. This option also works in eager mode and for
cell relaxation. Force-only MatRIS workflows keep their existing default cost.

For MACE, run:

```bash
python examples/npt.py --model mace --checkpoint /path/to/mace-mpa-0-medium.model
```

Omit `--checkpoint` to use the default MPA-0 medium model. MACE uses double
precision and the fused `fast` variant by default; no graph tuning is required.
Use the [MACE NPT validation script](examples/validate_mace_npt.py) to compare
MP-0/MPA-0 medium with the official calculator and measure end-to-end ASE timings.

### Geometry and cell relaxation

```python
from ase.optimize import FIRE

# Fixed cell: supported by all four backends
FIRE(atoms, trajectory="relax.traj").run(fmax=0.05, steps=500)

# Variable cell: use MatRIS, MACE, or CHGNet, which provide stress
from ase.filters import FrechetCellFilter
FIRE(FrechetCellFilter(atoms)).run(fmax=0.05, steps=500)
```

MatRIS and MACE can capture stress calculations. CHGNet computes stress eagerly; use
`cuda_graph=False` for cell relaxation to avoid unnecessary capture overhead.
ALIGNN currently does not expose stress through this interface and cannot be
used for NPT or cell relaxation here. Cell changes invalidate captures for backends
other than MatRIS and MACE, so their variable-cell workflows can trigger frequent
recapture and may perform better with eager inference.

## 4. CUDA Graph defaults

Most users can keep the defaults:

```python
calc = FastMDCalculator("chgnet")                    # auto, the default
calc = FastMDCalculator("chgnet", cuda_graph=False)  # Eager inference for comparison
calc = FastMDCalculator("chgnet", cuda_graph=True)   # Require capture; raise if unavailable
calc = FastMDCalculator("chgnet", device="cpu")      # Explicit CPU execution
calc = FastMDCalculator("chgnet", device="cuda:1")   # Select a GPU
```

`auto` falls back only for known capability limitations or missing dependencies
and reports the reason. Unexpected runtime errors, out-of-memory errors, and
invalid model outputs propagate to the caller. With `True`, requesting stress
or magnetic moments that the backend cannot capture also raises an error.

For manual tuning:

```python
from fastmd import FastMDCalculator, CUDAGraphConfig

calc = FastMDCalculator("chgnet", cuda_graph=CUDAGraphConfig(
    enabled="auto",
    warmup_steps=3,
    max_cached_graphs=8,
    edge_capacity_step=None,       # None = model-specific default
    triplet_capacity_step=None,
    enable_fusions=True,
))
```

| Setting | MatRIS | CHGNet | ALIGNN | MACE |
| --- | --- | --- | --- | --- |
| Warmup iterations | 3 | 3 | 3 | 3 |
| Edge capacity increment | 512 undirected | 128 undirected | 1024 directed | 256 directed |
| Triplet increment | 8192 | 1024 | 16384 | Not used |
| Cache | Up to 8 buckets | Up to 8 buckets | One capacity | Up to 8 graphs |
| Fusion | Topology profile | Enabled | Enabled | `fast` |
| Captured work | Model + derivatives | Model + forces | Model + forces | Neighbor update + model + derivatives |

Larger capacity increments can reduce recapture frequency but increase memory
use and padded computation. When MatRIS or CHGNet exceeds the cache limit, it
finishes the current evaluation, clears captures, and captures again on the next
call. This limit is not a hard GPU memory quota.

These public settings apply to each calculator instance. You do not need to set
`MATRIS_*` or `CHGNET_*` environment variables. The migrated kernels retain a few
internal experimental environment variables, which ordinary workflows should
not depend on.

Changes to atom count, species, species order, or PBC automatically invalidate
existing captures. Cell changes also invalidate captures except for MatRIS,
which updates the cell in place and reuses compatible capacity buckets.
Neighbor lists are updated for new positions on each evaluation. MACE reuses a
candidate list while displacements stay within the skin threshold, filtering
active edges on the GPU each time; it rebuilds the candidates when needed. When edge or triplet counts exceed capacity, the backend
increases capacity and captures again. Arrays returned to ASE own their storage
and are not overwritten by the next replay.

```python
print(calc.stats())     # mode: not-run / eager / cuda_graph
calc.clear_cache()      # Release captures and ASE results; keep model weights
```

Use one calculator/backend per serial workflow; do not share an instance across
threads. `warmup()` prepares the cache for the current geometry only. Subsequent
changes in temperature or neighbor counts can still trigger new captures.

## 5. Optimized MatRIS and MACE

### MatRIS

The bundled MatRIS engine now comes from the optimized `matris-09bk` snapshot
in `reb_package_20260927`, including all five supplied fix patches. The default
**topology** profile enables eligible indexed projections, merged frozen-weight
projections, segmented attention/reductions, paired edge operations and fused
basis/envelope/residual operations. Kernel selection still depends on device,
graph topology and tensor size; enabling a profile does not mean every operation
uses a fused kernel on every system. Activation checkpointing is disabled.

Ordinary ASE code needs no changes. Optional operator compilation is available:

```python
calc = FastMDCalculator(
    "matris", checkpoint="checkpoints/MatRIS_10M_OAM.pth.tar",
    model_kwargs={"compile_lowerings": True},
)
```

Compilation adds startup cost. Advanced kernel experiments can use
`model_kwargs={"expert_overrides": {"MATRIS_INDEXED_CAT_LINEAR_MIN_ROWS": 0}}`.
The underlying typed configuration validates these options and scopes them to
this backend. `calc.stats()` reports the requested profile and configuration.
Optional Blackwell CuTeDSL implementations are retained, but are not selected
by default and require their own compatible toolchain. The default GEMM backend
is PyTorch; no CuTeDSL installation is needed for ordinary use.

### MACE

```python
from ase.build import bulk
from fastmd import FastMDCalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = FastMDCalculator(
    "mace", checkpoint="checkpoints/mace-mpa-0-medium.model",
)
print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())
```

MP-0 medium and MPA-0 medium are the initial supported CUDA architectures.
This does not imply support for every MACE foundation model or fine-tuned
architecture. An unsupported capture architecture reports an eager fallback in
`auto` mode and raises in strict mode. Provide `model_kwargs={"head": "name"}`
when a multi-head checkpoint needs an explicit head.

| `model_kwargs` option | Default | Meaning |
| --- | --- | --- |
| `model_name` | `medium-mpa-0` | Used only when `checkpoint` is omitted; `medium` selects MP-0 |
| `default_dtype` | `float64` | Set `float32` explicitly if appropriate for your accuracy requirements |
| `compute_stress` | `False` | Set `True` for NPT/cell relaxation to return energy, forces and stress together |
| `variant` | `fast` | Fused edge geometry/SH/radial basis and applicable ZBL, density and force/stress tails |
| `neighbor_skin` | `1.0` Å | Candidate-list reuse distance |
| `capacity_headroom` | `1.25` | Initial/growth capacity margin |
| `enable_cueq` | `False` | Optional cuEquivariance conversion; install compatible cuEquivariance packages separately |

Use `variant="plain"` for captured upstream operations or `variant="fast_cm"`
for fused inference plus optional compilation of the original GEMM-level
submodules. `fast_cm` takes longer to prepare and reports compiled/fallback
modules in `calc.stats()["cache"]["compiled_modules"]`. `enable_fusions=False`
selects `plain`. CPU and `cuda_graph=False` use upstream eager MACE regardless
of the chosen variant, providing a reference path.

GPU capture includes candidate-list rebuild/filter kernels, model execution,
and coordinate/strain derivatives. Each replay checks capacity and neighbor
consistency before returning a prediction. Overflow releases affected graphs,
grows buffers and retries the same geometry. Changes to composition, atom count
or PBC discard the entire runner. Cell changes update buffers and rebuild neighbors;
only larger periodic-image ranges or capacity requirements trigger recapture.
The source neighbor builder uses quadratic
pair enumeration during rebuilds; benchmark large systems before assuming it
will outperform a cell-list implementation.

The source package's whole-step NHC MD timing results include work outside this
ASE Calculator interface. They are **not fastMD benchmark results**. Compare on
your GPU and structures using the script below; startup and replay are reported
separately.

## 6. Examples, numerical checks, and timing

After installation, run these commands from this directory:

```bash
python examples/single_point.py --model chgnet
python examples/md.py --model chgnet --steps 100
python examples/compare.py --model chgnet
python examples/compare.py --model matris --checkpoint ../checkpoint/MatRIS_10M_OAM.pth.tar --compile
python examples/compare.py --model mace --checkpoint checkpoints/mace-mpa-0-medium.model --stress --compile --output mace-benchmark.json
```

`compare.py` compares eager, graph without fusions, and fused graph inference on
the same perturbed geometries. `--compile` adds the optional compiled variant;
`--stress` includes stress validation. It reports model loading, setup/warmup,
and subsequent evaluation time separately, including ASE host/device transfers. It exits explicitly when no GPU is available;
CPU execution is not reported as a CUDA acceleration result.

```bash
python -m pip install -e '.[chgnet,test]'
python -m pytest -q
# Include tests for your local checkpoints
FASTMD_MATRIS_CHECKPOINT=/absolute/path/model.pth.tar \
FASTMD_ALIGNN_CHECKPOINT=/absolute/path/alignn_directory \
FASTMD_MACE_MP0_CHECKPOINT=/absolute/path/mace-mp-0.model \
FASTMD_MACE_MPA0_CHECKPOINT=/absolute/path/mace-mpa-0.model \
python -m pytest -q
```

CUDA tests are skipped when no GPU is available. Model integration tests are
skipped when their local checkpoints are missing. CUDA tests cover replay,
position perturbations, cell changes, composition changes, atom-count changes,
and the lifetime of returned arrays.

Assess performance and numerical tolerances for your model, system, GPU, and
trajectory length. Long MD trajectories are not expected to remain bitwise
identical across execution modes.

## 7. Adding more models for WBM workflows

The [model integration guide](docs/adding_models.md) provides a runnable
registration example and an integration checklist. Users keep the same API:
`FastMDCalculator("new_model", checkpoint=...)`. Developers put model loading,
graph representations, and capture details in `models/<name>.py`; the ASE
calculator does not need to change.

The architecture draws on the common capability declarations and adapter layer
in [torch-sim's model interface](https://github.com/TorchSim/torch-sim/blob/main/torch_sim/models/interface.py).
It does not depend on torch-sim or implement its batching interface. The current
API handles one ASE `Atoms` object at a time. For WBM screening, you can initially
reuse a calculator across structures, with captures rebuilt according to the
invalidation rules. True batching and capacity scheduling across structures
remain separate future work.

Performance on WBM does not establish that a model provides forces, stress, or
safe CUDA capture. Declare each model's capabilities explicitly and validate
total energy, forces, stress sign, and units against its original calculator.

## 8. Repository layout and provenance

```text
fastMD/
├── src/fastmd/
│   ├── calculator.py      # Common ASE entry point
│   ├── config.py          # Defaults for each calculator instance
│   ├── models/            # Backend contract, registry, and four adapters
│   └── _vendor/           # Migrated models, graph builders, and optimized kernels
├── examples/              # Single-point, ASE MD, and CUDA accuracy/timing comparisons
├── tests/                 # ASE contract and real-model integration tests
├── docs/                  # Architecture, integration guide, provenance, and validation
└── licenses/              # Original project licenses
```

The [migration notes](docs/migration.md) describe the original branches'
responsibilities and the fixes made during this refactor. The
[source manifest](docs/sources.json) records source commits, fix-patch hashes and
migrated files. Private namespaces avoid conflicts with `matris`, `chgnet`, and `alignn`
packages already installed in your environment. MACE optimizations also live in
a private namespace; the upstream `mace` dependency is used for checkpoint classes. Original license and citation
requirements continue to apply; see [NOTICE](NOTICE) and `licenses/`.
