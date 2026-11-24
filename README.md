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

Choose the model extra you need: `mace`, `sevennet`, `orb`, or `chgnet`. Install `.[gpu-graph]` for MatRIS on
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
SevenNet, and conservative ORB v3**. It requires full periodicity and a fixed, nonsingular cell;
triclinic cells are supported. Model families and versions without an audited shard declaration are rejected.

Multi-rank ORB keeps all radius-graph edges; when a node exceeds the checkpoint's neighbor cap,
this differs from single-device nearest-k evaluation.

```bash
torchrun --standalone --nproc_per_node=2 -m fastmd.distributed \
  --model matris --checkpoint /path/to/MatRIS_10M_OAM.pth.tar \
  --input input.extxyz --execution model_graph \
  --profile topology --precision highest --oracle --oracle-final \
  --warmup 5 --steps 50 --md-steps 100 --md-engine domain \
  --temp-start 600 --seed 0 --dt-fs 0.2 --output result.json
```

Use `--model mace`, `sevennet`, or `orb_v3` with the corresponding `--checkpoint`. For CHGNet, use
`--model chgnet_0_3_0 --deterministic` and omit `--checkpoint` to load its bundled weights. Deterministic
CHGNet execution controls repeated atomic-sum noise in the strict FP32 checks; their limits are unchanged.
Use separate model environments as described in [Installation](#installation). `--profile generic` uses
reference local operators; `topology` selects the model's optimized operators.

<details>
<summary>Python API and runtime details</summary>

The public API is `fastmd.DistributedEvaluator` with `fastmd.ParallelConfig`;
`fastmd.distributed.DistributedMatRISEvaluator` remains available for MatRIS. Run this script with
`torchrun --standalone --nproc_per_node=2 example.py`. `initialize()` binds each local GPU and creates the
data and control groups. Without an initialized process group, the evaluator runs locally with one rank.
Each rank must call collective methods in the same order.

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
and MACE, SevenNet, and ORB use their declared node-halo exchanges.

`GPUResidentMolecularDynamics.from_evaluator(evaluator, atoms=...)` uses the domain driver at any rank
count, supporting NVE, Berendsen NVT (`nvt`), and Nose–Hoover chain NVT (`nvt_nhc`), with restart continuation.
At one CUDA rank, force evaluation uses a single-device path without halo exchange: MatRIS delegates to
its existing single-GPU engine; the other four models use the fixed-capacity builder and engine-owned
derivative. This is separate from the MD driver choice.

`--md-engine domain` keeps the domain integrator even at one rank. CLI `auto` selects the native MD driver
for eligible one-rank MatRIS runs and the domain driver for the other four models. Domain MD advances one
step per launch. Only MatRIS currently captures the distributed integrator together with the model;
the other models capture energy and forces and run integration outside that graph.

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

Checkpoint links point to official weights or model pages.

| Model | Checkpoints / entry point | Backend |
| --- | --- | --- |
| MatRIS | [matris_10m_oam](https://ndownloader.figshare.com/files/59142728), [matris_10m_mp](https://ndownloader.figshare.com/files/59143058) | Bundled; separate weights |
| MACE | [mace_mp_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model), [mace_mpa_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model) | `mace-torch` |
| SevenNet | [sevennet_0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth), [7net-0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth) | `sevenn` |
| ORB | [orb_v3](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt) | `orb-models` |
| CHGNet | [chgnet_0_3_0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) | `chgnet`; includes weights |
| eSEN | [esen](https://huggingface.co/facebook/OMAT24#model-checkpoints) + `model_path` | Legacy fairchem; [separate environment](docs/esen.md) |

For local weights, pass `model_path=...`; MACE and SevenNet also accept `model="mace"` and `model="sevennet"`
respectively. A MACE `.model` path can be used as `model` directly. Loaded adapters can be reused via `model=`.

## S01 benchmark results

**41 checkpoints · 5 model families · NVIDIA H100 80 GB**

S01 uses 135-atom CsSnI₃ at 500 K: NVT Nose–Hoover chain, IEEE FP32, 80,000 steps of 0.25 fs (20 ps).
Speedup is official / FastMD full-run wall time, including trajectory output and excluding model loading,
compilation, capture, and post-run validation. It describes this workload and setup.

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
