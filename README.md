# FastMD

Single-device and distributed molecular simulation with machine-learned interatomic potentials.
Use one model interface for ASE calculations, structure relaxation, and molecular dynamics, with CPU execution
or GPU-resident integrators and CUDA Graphs.

[Installation](#installation) · [Quick start](#quick-start) · [Distributed](#distributed-simulation) ·
[Relaxation](#structure-relaxation) · [Models](#supported-models) · [Benchmarks](#s01-benchmark-results) · [Development](#development)

## Installation

Use **Python 3.10+** and run from the repository root:

```bash
# Base package and bundled MatRIS implementation
python -m pip install -e .

# CUDA graph builders and MACE
python -m pip install -e '.[gpu-graph,mace]'
```

Choose the model extra you need: `mace`, `sevennet`, `orb`, `chgnet`, or `nequip`. Install `.[gpu-graph]` for MatRIS on
CUDA, or `.[mace]` for MACE on CPU. Prepare a PyTorch build compatible with your CUDA environment first.

Distributed GPU runs use the same `gpu-graph` and model extras, with NCCL and Gloo support in PyTorch.
Launch one process per GPU with `torchrun`; no separate distributed extra is needed.

<details>
<summary>Dependency and acceleration notes</summary>

- The base package requires PyTorch ≥ 2.6. GPU graph builders require PyTorch ≥ 2.8,
  `nvalchemi-toolkit-ops[torch]>=0.4.1,<0.5`, and `warp-lang>=1.13.0`.
- MACE and SevenNet use e3nn with `optimization_profile="generic"`. Their `topology` profiles can use
  cuEquivariance; the CUDA 13 extras are `.[gpu-graph,mace,cueq-cu13]` (substitute `sevennet` as needed).
  Keep MACE and SevenNet in separate environments because their e3nn requirements differ.
- MACE-SOYO is a separate CUDA-only family whose upstream package requires Python 3.12–3.14.
  Its adapter targets upstream `mace-soyo` 0.1.0 at commit
  `387b5114c974fb04b8aceffac82d1d8f10ea4ac0`, with cuEquivariance 0.12.0 and torch-sim 0.6.1.
  [requirements-mace-soyo.txt](requirements-mace-soyo.txt) records the upstream Linux aarch64,
  CPython 3.12, CUDA 13 environment; those binary hashes do not install an x86_64 or CUDA 12 environment.
  Use an isolated environment with wheels verified for its platform. The existing `cueq-cu13` extra
  targets MACE/SevenNet's cuEquivariance 0.9.1 and does not provide the MACE-SOYO dependency stack.
- TACE uses upstream 0.2.0 at commit `81f65a4c188bd09cec8d1419388f7afdcc1b6fd0`,
  with e3nn 0.5.9 and torch-geometric 2.7.0. The PyPI release with the same version has different model code.
  [requirements-tace.txt](requirements-tace.txt) records the source archive and upstream Linux aarch64,
  CPython 3.12 wheel hashes; other platforms need their own verified binary hashes and a separate environment.
  The file supplements an existing inference environment and does not install every transitive dependency.
  OpenEquivariance 0.6.4 is optional: `generic` selects e3nn; the default attempts OEQ and reports any fallback
  in `kernel_reason`. OEQ's build must use the installed Torch headers and libraries instead of its default
  LibTorch download. Torch versions below 2.10 need the official legacy JIT path, with a matching CUDA toolkit
  and C++ compiler; the requirements file describes the build settings. Installed packages alone do not verify
  that their GPU kernels compile and run.
- NequIP uses `.[nequip]`, which pins NequIP 0.14.0 and PyTorch ≥ 2.8, < 2.10. Supply a local
  `.nequip.zip` package or current `.ckpt`; the adapter does not download weights.
- Legacy eSEN needs its own environment and local weights; follow the [eSEN guide](docs/esen.md).
  There is no `esen` extra. Experimental M3GNet uses `.[m3gnet]`.
- [pyproject.toml](pyproject.toml) defines installation dependencies. The `requirements-*.txt` files record
  upstream test environments, including platform-specific hashes; they are not complete dependency lockfiles.

</details>

## Quick start

### Energy, forces, and stress

Attach `MLIPCalculator` to an ASE structure:

```python
from ase.build import bulk
from fastmd import MLIPCalculator, InferenceConfig

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = MLIPCalculator(
    model="chgnet_0_3_0", device="cuda", task="efs",
    config=InferenceConfig(execution="eager", optimization_profile="generic"),
)
print(atoms.get_potential_energy())  # eV, total energy
print(atoms.get_forces())           # eV/Å
print(atoms.get_stress())           # eV/Å³, ASE Voigt order
```

For CPU use, select `model="matris_10m_oam"` and `device="cpu"`. Model adapters handle energy normalization,
derivatives, and units behind the same ASE interface.

### Molecular dynamics

```bash
python examples/simulate.py --structure input.extxyz \
  --model chgnet_0_3_0 --driver gpu --device cuda \
  --ensemble nvt_nhc --temperature 300 --timestep 0.5 \
  --execution auto --steps 100
```

`GPUResidentMolecularDynamics` keeps integration on the GPU; `MolecularDynamics` uses ASE's integrator.
Use `GPUResidentMolecularDynamics` as a context manager to finish writing logs and trajectories.
Automatic execution selects among `eager`, `model_graph`, and `whole_step` according to model capabilities
and memory. Inspect the proposed plan with:

```bash
fastmd-explain --structure input.extxyz --model chgnet_0_3_0 --device cuda --json
```

`GPUResidentMolecularDynamics.from_restart(...)` also reads compatible MD restart files
(`matris.distributed.md/1`). Supply the original model and weights; simulation parameters default to the
saved values. FastMD continues to write its own format (`fastmd.distributed.md/1`).

## Distributed simulation

The experimental runtime partitions one periodic system across GPUs for **MatRIS, CHGNet, MACE,
MACE-SOYO, TACE, SevenNet, and conservative ORB v3**. It requires full periodicity and a fixed, nonsingular cell;
triclinic cells are supported. Model families and versions without an audited shard declaration are rejected.

ORB keeps the nearest neighbors up to the checkpoint's cap on one rank, including CPU evaluation.
With multiple ranks it retains the radius graph and rejects an evaluation on every rank if any atom
exceeds that cap: a rank cannot select the nearest neighbors of its ghosts from incomplete neighborhoods.

```bash
torchrun --standalone --nproc_per_node=2 -m fastmd.distributed \
  --model matris --checkpoint /path/to/MatRIS_10M_OAM.pth.tar \
  --input input.extxyz --execution model_graph \
  --profile topology --precision highest --oracle --oracle-final \
  --warmup 5 --steps 50 --md-steps 100 --md-engine domain \
  --temp-start 600 --seed 0 --dt-fs 0.2 --output result.json
```

Use `--model mace`, `mace_soyo`, `sevennet`, `tace`, or `orb_v3` with the corresponding `--checkpoint`. For CHGNet, use
`--model chgnet_0_3_0 --deterministic` and omit `--checkpoint` to load its bundled weights. Deterministic
CHGNet execution controls repeated atomic-sum noise in the strict FP32 checks; their limits are unchanged.
Use separate model environments as described in [Installation](#installation). `--profile generic` uses
reference local operators; `topology` selects the model's optimized operators.

Model loading and evaluator options are checked before distributed initialization. The selected family
and checkpoint use the same resolution for validation and loading; a MACE `.model` path remains a MACE model.

| Option | Meaning |
| --- | --- |
| `--weights-dtype` / `--dtype` | Model weights and arithmetic; `--dtype` remains compatible. Supported choices depend on the family. Multiple CUDA ranks require FP32 weights. |
| `--geometry-dtype float32` | Explicit model geometry for compatible FP32 models, including the native MD configuration. It does not convert weights. |
| `--geometry-dtype float64` | FP64 positions, cell, coordinate exchange, edge vectors, forces and MD state at every rank count for supporting shard adapters. Model weights and feature exchanges retain their own dtype. MatRIS refuses this option. |
| `--stress` | Also evaluate fixed-frame stress and report a tension-positive 3 × 3 tensor in eV/Å³. This does not enable a barostat or NPT. |
| `--oracle` / `--oracle-final` | Compare the initial / final MD frame with a separate unsharded reference. Reports retain the absolute gate and any measured-noise gate separately. |

Eligible FP32 external models default to FP64 geometry in both single-rank and multi-rank evaluators
and their MD drivers. `--geometry-dtype float32` restores FP32 geometry; when the option is unset,
a resumed run follows the saved restart geometry. An explicit change from that saved dtype warns.
MatRIS retains its FP32 geometry. FP64 weights cannot be combined with explicit FP32 geometry.
The CUDA neighbor search, ownership and halo selection still use FP32 decisions; FP64 geometry
preserves the coordinates used to evaluate the selected edges, with the same lattice shifts.

The JSON report records the resolved model family, checkpoint and precision. For declared shards,
`plan_record` describes the fixed-frame evaluations (`plan_record_scope`); native MD has its own execution
record under `md.native`. Evaluation success, oracle agreement and timing are separate results; inspect the oracle gates before treating a run
as a correctness check. Existing output files are never overwritten.

Preview a declared shard's plan with `fastmd-explain --structure input.extxyz --model chgnet_0_3_0
--device cuda --world-size 2 --json`. This estimates the partition in one process; it does not launch ranks
or measure memory. For MatRIS, inspect the distributed evaluator's `config_summary()`.

<details>
<summary>Python API and runtime details</summary>

The public API is `fastmd.DistributedEvaluator` with `fastmd.ParallelConfig`;
`fastmd.distributed.DistributedMatRISEvaluator` remains available for MatRIS. Run this script with
`torchrun --standalone --nproc_per_node=2 example.py`. `initialize()` binds each local GPU and creates the
data and control groups. Without an initialized process group, the evaluator runs locally with one rank.
Each rank must call collective methods in the same order.
Pass a loaded FastMD adapter or an uncompiled MatRIS instance to `DistributedEvaluator`.
For other models, first use `load_adapter("mace", model_path="/path/to/model", device="cuda")`
or the appropriate model family; the evaluator does not load model names, paths, or bare upstream modules.

```python
from ase.io import read
from fastmd import DistributedEvaluator, InferenceConfig, ParallelConfig, load_adapter
from fastmd.distributed.launch import fail_fast, initialize, shutdown

context = initialize(device="cuda")
try:
    atoms = read("input.extxyz")  # same cell metadata on every rank
    adapter = load_adapter(model_path="/path/to/MatRIS_10M_OAM.pth.tar", device="cuda")
    evaluator = DistributedEvaluator(
        adapter, atoms.cell.array, context=context, parallel=ParallelConfig(),
        config=InferenceConfig(execution="model_graph", matmul_precision="highest"),
    )
    state = evaluator.distribute(atoms if context.rank == 0 else None)
    prediction = evaluator.evaluate(state)
    frame = evaluator.gather(prediction)
    if context.rank == 0:
        print(float(prediction.energy), frame["forces"])
    evaluator.close()
    adapter.close()
    shutdown()
except BaseException as error:
    fail_fast(error, rank=context.rank)
```

The runtime uses spatial ownership, ghost exchanges and reverse gradient sums, reusable neighbor-list
epochs, and collective CUDA Graph capture. Model weights are replicated; atom ownership and model work
are partitioned. MatRIS uses boundary-bond partial exchanges, CHGNet uses line-graph partial exchanges,
and MACE, MACE-SOYO, TACE, SevenNet, and ORB use their declared node-halo exchanges.
MACE, MACE-SOYO, SevenNet, and energy-only TACE checkpoints trim declared per-atom work to owned
slots at multiple ranks; the shared runtime validates those declarations when ghost rows first appear.
SevenNet trims selected dictionary entries, and TACE restores its trailing sink row after exchange;
unknown SevenNet module classes and TACE checkpoints with additional per-atom heads keep the untrimmed path.
Memory admission probes the declared trimmed computation while preserving the owned/ghost row split.

`GPUResidentMolecularDynamics.from_evaluator(evaluator, atoms=...)` uses the domain driver at any rank
count, supporting NVE, Berendsen NVT (`nvt`), and Nose–Hoover chain NVT (`nvt_nhc`), with restart continuation.
At one CUDA rank, force evaluation uses a single-device path without halo exchange: MatRIS delegates to
its existing single-GPU engine; the external adapters use the fixed-capacity builder and engine-owned
derivative. This is separate from the MD driver choice.

`--md-engine domain` keeps the domain integrator even at one rank. CLI `auto` selects the native MD driver
for eligible one-rank MatRIS runs and the domain driver for the external models. Explicit
`--md-engine native` also supports eligible external adapters at one CUDA rank; it releases the fixed-frame
evaluator before constructing the native MD runner. CPU or multi-rank runs and incompatible thermostat
options reject that choice. Domain MD advances one step per launch. With multiple ranks and `model_graph`,
the supported models can capture the distributed integrator together with the model. Topology refreshes occur at committed positions, and rejected
steps restore the integrator before retrying or temporarily using integration outside the graph.
The Python option `integrator_in_graph=False` keeps integration outside the captured model step.

</details>

See [migration scope and validation coverage](docs/migration.md) for model-specific limitations and
validation records. Multi-GPU speedups depend on system size.

## Structure relaxation

`StructOptimizer` accepts an ASE `Atoms`, a pymatgen `Structure`, or a list of either.
This **MACE-MPA-0** example relaxes atomic positions and the cell with FIRE. Replace the checkpoint path with
your local `.model` file; the `generic` profile uses e3nn on CPU or CUDA without requiring cuEquivariance.

```python
import torch
from ase.build import bulk
from fastmd import InferenceConfig, StructOptimizer

atoms = bulk("Si", "diamond", a=5.5, cubic=True)
atoms.rattle(0.03, seed=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

with StructOptimizer(
    model="mace", model_path="/path/to/mace-mpa-0-medium.model",
    device=device, task="efs", optimizer="FIRE", engine="auto",
    config=InferenceConfig(optimization_profile="generic"),
) as optimizer:
    result = optimizer.relax(
        atoms, fmax=0.05, steps=500, relax_cell=True,
        ase_filter="FrechetCellFilter", assign_magmoms=False, verbose=False,
    )
    print(result["trajectory"].energies[-1])  # Final total energy in eV
    print(result["final_structure"].lattice.abc)
    print(optimizer.last_relaxation)
```

For a fixed cell, use `task="ef"` and `relax_cell=False`. Variable-cell relaxation uses `task="efs"`;
`fmax` includes the cell filter's force rows, so it is not a separate stress tolerance.
Pass optimizer options directly to `relax(...)`, for example `dt=0.05`.

| Engine | Behavior |
| --- | --- |
| `"auto"` | GPU FIRE when eligible; otherwise ASE. |
| `"gpu"` | Require GPU relaxation; raise with the reason if the GPU path is unavailable or fails. |
| `"ase"` | ASE optimizer on the host; model evaluation uses the configured device. |

Set `engine` on the constructor or override it in `relax(...)`. `optimizer.last_relaxation` records the path
and reason. Results contain `final_structure` and `trajectory`; lists preserve input order. Supplied `Atoms`
are modified in place with the calculator attached; pymatgen inputs are copied through conversion to `Atoms`.

More examples and configuration details:

- [MatRIS relaxation](examples/README.md#matris-relaxation)
- [Batch GPU relaxation](examples/README.md#batch-gpu-relaxation)
- [GPU eligibility, trajectories, and execution policies](examples/README.md#relaxation-options)

## Supported models

Checkpoint links point to official weights or model pages. Registration identifies a loader; it does not
by itself promise graph capture, joint evaluation of several systems, or distributed execution.

| Model | Checkpoints / entry point | Backend |
| --- | --- | --- |
| MatRIS | [matris_10m_oam](https://ndownloader.figshare.com/files/59142728), [matris_10m_mp](https://ndownloader.figshare.com/files/59143058) | Bundled; separate weights |
| MACE | [mace_mp_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model), [mace_mpa_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model) | `mace-torch` |
| MACE-SOYO | `mace_soyo_oam` (default head `salexmp`), `mace_soyo_mptrj` | `mace-soyo` and cuEquivariance; CUDA only |
| TACE | `tace_oam_l`, `tace_oam_rra_preview`, `tece_oam_rra_1_0` | Pinned spherical TACE; e3nn or OpenEquivariance |
| SevenNet | [sevennet_0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth), [7net-0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth) | `sevenn` |
| ORB | [orb_v3](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt) | `orb-models` |
| CHGNet | [chgnet_0_3_0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) | `chgnet`; includes weights |
| NequIP | `nequip`, `nequip-oam-l`, `nequip-oam-xl`, `nequip-mp-l` + `model_path` | `nequip==0.14.0`; local `.nequip.zip` / current `.ckpt` |
| eSEN | [esen](https://huggingface.co/facebook/OMAT24#model-checkpoints) + `model_path` | Legacy fairchem; [separate environment](docs/esen.md) |

The capabilities below apply to the supported checkpoint architectures and dependency versions. Each
loaded adapter is checked before using a path; custom subclasses and additional heads can reduce support.
“Batched systems” means independent systems evaluated together on one GPU, not a spatially divided system.

| Family | Weights | Single-device CUDA capture | Batched systems | Spatially distributed system |
| --- | --- | --- | --- | --- |
| MatRIS | FP32 | Supported | Supported with per-atom reference energies | Boundary-bond exchanges |
| MACE | FP32 / FP64 | Supported | Audited `ScaleShiftMACE`, without joint embeddings | Declared node halo; FP32 on multiple CUDA ranks |
| MACE-SOYO | FP32 / FP64; CUDA only | Supported | Supported | Declared node halo; FP32 on multiple CUDA ranks |
| TACE | FP32 | Supported spherical checkpoints | Supported | Declared node halo |
| SevenNet | FP32 | Supported exact-cutoff checkpoints | Supported union; fixed modality per union | Declared node halo and source checks |
| ORB | FP32 | Audited conservative v3 | Per-system pooling for audited conservative v3 | Conservative v3 node halo; neighbor-cap overflow is refused |
| CHGNet | FP32 | Supported | Audited CHGNet union with line graph | Audited 0.3.0 line-graph exchanges |
| NequIP | Saved FP32 / FP64 | Opt-in `fixed_shape=True`, after padding checks | No declaration | No declaration |
| Legacy eSEN | FP32 | Supported after source-pinned patches | No declaration | No declaration |

NequIP and eSEN retain their single-device calculator and Python MD interfaces; the distributed evaluator
requires a shard declaration even with one rank. NequIP preserves FP64 geometry and the checkpoint's saved
weight precision. For opt-in fixed-shape or cuEquivariance execution, call its family loader and pass the
returned adapter to the calculator or MD runner:

```python
from fastmd.models.nequip import load_nequip

adapter = load_nequip(
    "nequip", model_path="/path/to/model.nequip.zip", device="cuda",
    fixed_shape=True, kernel="upstream",  # kernel="cueq" requires its optional CUDA dependencies
)
```

`fixed_shape` and `kernel` are NequIP family-loader options, not generic `load_adapter` arguments. Legacy
eSEN is limited to conservative, envelope-enabled fairchem 1.x models; it does not imply support for UMA
or direct-force heads. SevenNet's `allow_eager=True` variants and ORB direct/v2 adapters keep their separate
single-device behavior; a registered family does not grant those variants the matrix's captured or
multi-rank capabilities.

For local weights, pass `model_path=...`; MACE and SevenNet also accept `model="mace"` and `model="sevennet"`
respectively. A MACE `.model` path can be used as `model` directly. Loaded adapters can be reused via `model=`.

MACE-SOYO names download the upstream release checkpoint once and verify its SHA256; for local weights,
use `load_adapter("mace_soyo", model_path="/path/to/model.pth", device="cuda")`.
The adapter runs upstream's energy function with engine-owned forces and stress. It supports float32 or
float64 weights, FP64 geometry at every rank count, graph capture, union batching, and the declared node-halo
shard. Multiple CUDA ranks require FP32 weights. Both `generic` and `topology` run its cuEquivariance kernels;
there is no CPU or e3nn fallback. Charge/spin-conditioned models are rejected. Pinned source checks reject
unreviewed upstream implementations at multiple ranks.

Select the OAM checkpoint's other head through its family loader, then pass the returned adapter to a
calculator, MD runner, or distributed evaluator:

```python
from fastmd.models.mace_soyo import load_mace_soyo

adapter = load_mace_soyo("mace_soyo_oam", head="omat_pbe", device="cuda", model_dtype="float32")
```

TACE names download SHA256-checked weights from a fixed upstream Hugging Face revision. For local weights,
use `load_adapter("tace", model_path="/path/to/model.pt", device="cuda")`. The adapter supports the audited
spherical `e3nnTACE` checkpoints, FP32 weights with optional FP64 geometry, graph capture, union batches,
and declared node-halo exchanges. Batched systems retain their own cells and isolated-atom behavior;
padding is routed to a separate sink atom. The Cartesian TACE-v1 family, long-range or universal-embedding
models, and capped-neighbor models are rejected. Multi-rank source checks reject unreviewed implementations;
multi-rank TF32 is declared only for the pinned TACE-OAM-L weights. Use `optimization_profile="generic"`
for explicit e3nn execution, and inspect `adapter.kernel` when requesting OEQ. Checkpoints contain pickled
model objects and should come from a trusted source.

## S01 benchmark results

**41 checkpoints · 5 model families · NVIDIA H100 80 GB**

S01 uses 135-atom CsSnI₃ at 500 K: NVT Nose–Hoover chain, IEEE FP32, 80,000 steps of 0.25 fs (20 ps).
Speedup is official / FastMD full-run wall time, including trajectory output and excluding model loading,
compilation, capture, and post-run validation. It describes this workload and setup; its five-family scope
is separate from the current model inventory.

<!-- s01-results:begin -->

| Model family | Completed checkpoint pairs | Recorded speedup range |
| --- | ---: | ---: |
| MatRIS | 2 | 5.13–5.52× |
| MACE | 15 | 12.88–24.73× |
| SevenNet | 8 | 14.87–24.47× |
| ORB | 13 | 3.23–7.19× |
| CHGNet | 3 | 18.45–21.05× |

See the [benchmark protocol and all 41 checkpoint results](examples/README.md#s01-benchmark-details)
for the complete measurements.

## Development

| Resource | Contents |
| --- | --- |
| [Examples and detailed results](examples/README.md) | MatRIS, batch relaxation, and S01 benchmarks |
| [Model adapters](docs/model_adapters.md) | Model contracts, energies, derivatives, and extension points |
| [Additional checkpoints](docs/additional_checkpoints.md) | Model-specific loading options |
| [Distributed migration](docs/migration.md) | Supported models, runtime boundaries, and validation coverage |
| [Regression tests](tests/README.md) | CPU/GPU checks and test assets |
| [Validation tools](tools/README.md) | Architecture checks and GPU validation |
| [Model validation lanes](tools/ci/README.md) | MACE, SevenNet, ORB, and CHGNet environments |

Runtime code is organized into [models](fastmd/models), [engine](fastmd/engine), [execution](fastmd/execution),
[interfaces](fastmd/interfaces), [graph](fastmd/graph), [ops](fastmd/ops), and
[distributed](fastmd/distributed).

## Contact

Welcome to FastMD! For technical questions, please contact [husiyu@ict.ac.cn](mailto:husiyu@ict.ac.cn).
