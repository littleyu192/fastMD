<div align="center">

<h1>FastMD</h1>

<p><strong>Molecular simulation with machine-learned interatomic potentials</strong></p>
<p>One interface for energy, forces, relaxation, and molecular dynamics.</p>

<p>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square" alt="Python 3.10 or newer"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?style=flat-square" alt="PyTorch 2.6 or newer"></a>
  <a href="#molecular-dynamics"><img src="https://img.shields.io/badge/Execution-CPU%20%7C%20CUDA-76B900?style=flat-square" alt="CPU and CUDA execution"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-64748B?style=flat-square" alt="BSD 3-Clause license"></a>
</p>

<p>
  <a href="#installation">Install</a> &nbsp;·&nbsp;
  <a href="#unified-calculator">Quick start</a> &nbsp;·&nbsp;
  <a href="#supported-models">Models</a> &nbsp;·&nbsp;
  <a href="#s01-benchmark-results">Benchmarks</a> &nbsp;·&nbsp;
  <a href="#molecular-dynamics">MD</a> &nbsp;·&nbsp;
  <a href="#structure-relaxation">Relaxation</a>
</p>

</div>

---

FastMD is a **single-device simulation framework** for machine-learned interatomic
potentials (MLIPs). It connects model backends to a shared simulation runtime, with
`eager`, `model_graph`, and `whole_step` execution modes.

| Calculate | Simulate | Relax |
| :--- | :--- | :--- |
| Energy, forces, and stress through a common ASE calculator | ASE-driven and GPU-resident molecular dynamics | Single-structure and batched structure optimization |
| [`MLIPCalculator`](#unified-calculator) | [`MolecularDynamics` / `GPUResidentMolecularDynamics`](#molecular-dynamics) | [`StructOptimizer` / `MultiSystemRelaxation`](#structure-relaxation) |

## Supported models

Checkpoint names link to official weight files or model pages.

| Model | Name or entry point | Model implementation |
| --- | --- | --- |
| **MatRIS** | [matris_10m_oam](https://ndownloader.figshare.com/files/59142728), [matris_10m_mp](https://ndownloader.figshare.com/files/59143058), or `model_path=...` | Bundled single-device implementation; weights loaded separately |
| **MACE** | [mace_mp_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model), [mace_mpa_0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model), `mace` + `model_path`, or a `.model` path | Optional `mace-torch` package |
| **SevenNet** | [sevennet_0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth), [7net-0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth), or `sevennet` + `model_path` | Optional `sevenn` package |
| **ORB** | [orb_v3](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt), with optional `model_path` | Optional `orb-models` package |
| **CHGNet** | [chgnet_0_3_0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) | Pretrained weights supplied by the optional `chgnet` package |
| **eSEN** | [esen](https://huggingface.co/facebook/OMAT24#model-checkpoints) + `model_path` | Legacy `fairchem-core==1.10.0`; isolated environment and local weights required |

## S01 benchmark results

**41 checkpoints · 5 model families · NVIDIA H100 80 GB**

S01: 135-atom CsSnI₃ at 500 K, run for 80,000 steps (20 ps) on one GPU.
Every checkpoint has a complete official/FastMD pair.

> **Timing basis:** speedup = official full-run time / FastMD full-run time.
> Timings include trajectory output; model loading, compilation, capture, and
> post-run validation are excluded. Results apply to the S01 setup below.

<!-- s01-results:begin -->

| Model family | Completed checkpoint pairs | Recorded speedup range |
| --- | ---: | ---: |
| MatRIS | 2 | 5.13–5.52× |
| MACE | 15 | 12.88–24.73× |
| SevenNet | 8 | 14.87–24.47× |
| ORB | 13 | 3.23–7.19× |
| CHGNet | 3 | 18.45–21.05× |

<details>
<summary><strong>Benchmark setup and timing protocol</strong></summary>

| Setting | Value |
| --- | --- |
| System | S01: `CsSnI3_500K_Ivor_VASP`, 135 atoms |
| Hardware | NVIDIA H100 80 GB HBM3; one GPU per trajectory |
| Ensemble | NVT Nose–Hoover chain at 500 K |
| Integration | 0.25 fs × 80,000 steps = 20 ps |
| Thermostat | `tdamp=25 fs`, `tchain=3`, `tloop=1` |
| Recording | One frame every 10 steps; 8,001 frames including the initial state |
| Precision | FP32 IEEE |
| Initial state | Shared coordinates, cell, and momenta; seed 0; no center-of-mass momentum removal |
| Timing | GPU-synchronized wall time for the complete MD run, including trajectory output; model loading, compilation, capture, and post-run validation excluded |

**Speedup = official full-run time / FastMD full-run time.** The displayed step times are the complete MD wall times divided by 80,000.

</details>

<details>
<summary><strong>MatRIS</strong> — 2 checkpoints</summary>

### MatRIS

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [matris_10m_oam](https://ndownloader.figshare.com/files/59142728) | 154.5975 | 28.0180 | 5.5178× |
| [matris_10m_mp](https://ndownloader.figshare.com/files/59143058) | 164.6329 | 32.0987 | 5.1290× |

</details>

<details>
<summary><strong>MACE</strong> — 15 checkpoints</summary>

### MACE

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [mace-small](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model) | 31.8842 | 1.5295 | 20.8460× |
| [mace-medium](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model) | 43.0477 | 1.7405 | 24.7327× |
| [mace-large](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model) | 40.2385 | 2.1759 | 18.4924× |
| [mace-small-0b](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model) | 31.5469 | 2.0938 | 15.0665× |
| [mace-medium-0b](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_medium.model) | 39.0663 | 2.3195 | 16.8429× |
| [mace-small-0b2](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-small-density-agnesi-stress.model) | 28.2565 | 2.1946 | 12.8752× |
| [mace-medium-0b2](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model) | 44.8428 | 2.5230 | 17.7738× |
| [mace-large-0b2](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-large-density-agnesi-stress.model) | 38.3883 | 2.8898 | 13.2842× |
| [mace-medium-0b3](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model) | 52.3621 | 2.5283 | 20.7101× |
| [mace-medium-mpa-0](https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model) | 43.0448 | 2.3935 | 17.9840× |
| [mace-small-omat-0](https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-small.model) | 41.8795 | 2.1667 | 19.3284× |
| [mace-medium-omat-0](https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model) | 41.1945 | 2.4880 | 16.5574× |
| [mace-mace-matpes-pbe-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model) | 51.4085 | 2.5274 | 20.3405× |
| [mace-mace-matpes-r2scan-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model) | 36.5004 | 2.4031 | 15.1891× |
| [mace-large-2024-01-07](https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model) | 36.9483 | 2.1359 | 17.2991× |

</details>

<details>
<summary><strong>SevenNet</strong> — 8 checkpoints</summary>

### SevenNet

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [7net-0](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth) | 53.6565 | 2.5952 | 20.6754× |
| [7net-0_22may2024](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_0__22May2024/checkpoint_sevennet_0.pth) | 62.0417 | 2.5770 | 24.0756× |
| [7net-l3i5](https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/pretrained_potentials/SevenNet_l3i5/checkpoint_l3i5.pth) | 80.1659 | 3.2757 | 24.4730× |
| [7net-omat](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.11.0.cp/checkpoint_sevennet_omat.pth) | 59.0921 | 3.8197 | 15.4705× |
| [7net-nano-4.5](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_4.5.pth) | 34.0357 | 1.4519 | 23.4430× |
| [7net-nano-5.0](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_5.0.pth) | 20.3308 | 1.3669 | 14.8732× |
| [7net-nano-5.5](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_5.5.pth) | 32.0208 | 1.4654 | 21.8514× |
| [7net-nano-6.0](https://github.com/MDIL-SNU/SevenNet/releases/download/v0.13.1.cp/checkpoint_7net_nano_6.0.pth) | 23.2114 | 1.3616 | 17.0477× |

</details>

<details>
<summary><strong>ORB</strong> — 13 checkpoints</summary>

### ORB

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [orb-v3-conservative-20-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-omat-20250404.ckpt) | 34.3849 | 5.2446 | 6.5562× |
| [orb-v3-conservative-inf-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-omat-20250404.ckpt) | 36.7809 | 5.1188 | 7.1855× |
| [orb-v3-direct-20-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt) | 8.8192 | 2.5000 | 3.5277× |
| [orb-v3-direct-inf-omat](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-inf-omat-20250404.ckpt) | 9.6175 | 2.7153 | 3.5420× |
| [orb-v3-conservative-20-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-20-mpa-20250404.ckpt) | 28.9682 | 4.7892 | 6.0487× |
| [orb-v3-conservative-inf-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-conservative-inf-mpa-20250404.ckpt) | 33.8396 | 5.0916 | 6.6461× |
| [orb-v3-direct-20-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-mpa-20250404.ckpt) | 9.0028 | 2.7347 | 3.2921× |
| [orb-v3-direct-inf-mpa](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-inf-mpa-20250404.ckpt) | 8.9940 | 2.7543 | 3.2654× |
| [orb-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v2-20241011.ckpt) | 11.1864 | 3.4686 | 3.2251× |
| [orb-d3-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-v2-20241011.ckpt) | 11.5781 | 3.4350 | 3.3706× |
| [orb-d3-sm-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-sm-v2-20241011.ckpt) | 8.6669 | 2.5232 | 3.4349× |
| [orb-d3-xs-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-d3-xs-v2-20241011.ckpt) | 6.7959 | 1.7014 | 3.9943× |
| [orb-mptraj-only-v2](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt) | 10.9139 | 3.2807 | 3.3267× |

</details>

<details>
<summary><strong>CHGNet</strong> — 3 checkpoints</summary>

### CHGNet

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [chgnet-0.3.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) | 53.3781 | 2.7588 | 19.3484× |
| [chgnet-0.2.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar) | 52.0884 | 2.4750 | 21.0455× |
| [chgnet-r2scan](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/r2scan/chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar) | 51.5925 | 2.7956 | 18.4548× |

</details>

<!-- s01-results:end -->

## Installation

Use **Python 3.10+** and run installation commands from the repository root.
Dependencies and optional backends are defined in [`pyproject.toml`](pyproject.toml).

```bash
python -m pip install -e .
```

Choose the extra for your workload:

| Workload | Install command |
| :--- | :--- |
| GPU neighbor graphs | `python -m pip install -e '.[gpu-graph]'` |
| CHGNet + GPU graphs | `python -m pip install -e '.[gpu-graph,chgnet]'` |
| MACE + GPU graphs | `python -m pip install -e '.[gpu-graph,mace]'` |
| SevenNet + GPU graphs | `python -m pip install -e '.[gpu-graph,sevennet]'` |
| ORB + GPU graphs | `python -m pip install -e '.[gpu-graph,orb]'` |
| Experimental M3GNet integration | `python -m pip install -e '.[m3gnet]'` |

> **GPU setup:** the base package requires Torch ≥ 2.6; `gpu-graph` requires
> Torch ≥ 2.8, `nvalchemi-toolkit-ops[torch]>=0.4.1,<0.5`, and `warp-lang>=1.13.0`.
> Prepare a Torch build matching your CUDA environment first, because installing
> the extra can upgrade Torch. MACE and SevenNet need separate environments due
> to their different e3nn requirements.

<details>
<summary><strong>cuEquivariance acceleration and dependency notes</strong></summary>

The MACE and SevenNet `topology` profiles use cuEquivariance. The upstream CUDA 13 dependency set is available through `python -m pip install -e '.[gpu-graph,mace,cueq-cu13]'`; substitute `sevennet` for `mace` as needed. Use `optimization_profile="generic"` without that acceleration environment. MACE and SevenNet require different e3nn versions and should use separate environments.

`requirements-{mace,sevennet,orb,chgnet}.txt` preserve upstream test-environment versions and hashes. They do not replace the base installation, contain Python/platform-specific wheel hashes, and are not complete dependency lockfiles. Use the extras above for normal installation. The upstream CI scripts mentioned in those records are not included here.

The former `graph/setup.py` and `fastmd_op` build are no longer required. Use
`-e .` and `-e '.[m3gnet]'` in place of the old `requirements.txt` and
`requirements-m3gnet.txt` installation paths.

</details>

<details>
<summary><strong>eSEN: isolated legacy environment</strong></summary>

### eSEN environment

Legacy eSEN uses an isolated environment because fairchem 1.10.0 conflicts with the base package's dependency constraints. The recorded environment uses Python 3.11, fairchem-core 1.10.0, Torch 2.8.0+cu128, NumPy 1.26.4, SciPy 1.14.1, e3nn 0.5.9, ASE 3.26.0, nvalchemi-toolkit-ops 0.2.0, and warp-lang 1.10.0, with PyG extensions matching Torch/CUDA. Torch 2.8 is an explicit override of fairchem's declared Torch 2.4 requirement. There is no `esen` extra: use the prepared environment and load this checkout through `PYTHONPATH`, for example `PYTHONPATH="$PWD" python your_script.py` from the repository root. Supply an [eSEN-30M-OAM](https://huggingface.co/facebook/OMAT24#model-checkpoints) checkpoint explicitly; fairchem 2.x/UMA and direct-force heads are not supported by this adapter.

</details>

## Unified calculator

Evaluate energy, forces, and stress with ASE conventions:

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

For a CPU example, use `model="matris_10m_oam"` and `device="cpu"`. The adapters and shared derivative layer handle model energy normalization, gradients, and stress units while preserving ASE conventions at the interface.

## Molecular dynamics

Run GPU-resident NVT dynamics from the command line:

```bash
python examples/simulate.py --structure input.extxyz \
  --model chgnet_0_3_0 --driver gpu --device cuda \
  --ensemble nvt_nhc --temperature 300 --timestep 0.5 \
  --execution auto --steps 100
```

You can also instantiate `GPUResidentMolecularDynamics` directly. Use it as a context manager to finish writing logs and trajectories on exit. `MolecularDynamics` provides ASE-driven MD, and `StructOptimizer` provides structure relaxation. `execution="auto"` selects a path using model capabilities, available memory, and applicable constraints; it does not guarantee whole-step capture for every model.

Inspect the selected execution plan:

```bash
fastmd-explain --structure input.extxyz --model chgnet_0_3_0 --device cuda --json
```

## Structure relaxation

`StructOptimizer` shares model adapters and inference settings with the
calculator and MD drivers. It selects GPU or ASE optimization for each workload.

```python
from ase.io import read
from fastmd import InferenceConfig, StructOptimizer

atoms = read("input.extxyz")
with StructOptimizer(
    model="chgnet_0_3_0",
    device="cuda",
    task="efs",
    engine="auto",
    config=InferenceConfig(execution="auto", optimization_profile="generic"),
) as optimizer:
    result = optimizer.relax(
        atoms, fmax=0.05, steps=500, relax_cell=True,
        verbose=False, assign_magmoms=False,
    )
    print(result["trajectory"].energies[-1])
    print(optimizer.config_summary()["relaxation"]["last"])
    final_structure = result["final_structure"]
```

| Engine | Behavior |
| :--- | :--- |
| `auto` | Use the GPU path when supported; otherwise use ASE and record the reason |
| `gpu` | Require the GPU path; raise an error when it cannot run |
| `ase` | Use ASE optimization |

Use `relax_cell=False` for a fixed cell. Keep `InferenceConfig(execution="auto")`
for GPU relaxation; explicit `eager` or `model_graph` requests select the ASE path
under `engine="auto"`, or raise an error under `engine="gpu"`.
The engine can also be selected per `relax(..., engine=...)` call.

> **Input and output:** ASE `Atoms` are relaxed **in place** with the optimizer's
> calculator attached. Each result contains `final_structure` and `trajectory`;
> a list of inputs returns a list of results in the same order.

<details>
<summary><strong>Relax multiple structures</strong></summary>

Compatible structures run together in GPU slots; structures requiring ASE are
handled separately. Use an `{index}` placeholder when saving multiple trajectories.

```python
structures = read("inputs.extxyz", index=":")
with StructOptimizer(
    model="mace", model_path="/path/to/model.model",
    device="cuda", engine="auto",
    config=InferenceConfig(execution="auto", optimization_profile="generic"),
) as optimizer:
    results = optimizer.relax(
        structures, steps=500, fmax=0.02, relax_cell=True,
        loginterval=5, save_path="relax-{index}.pkl",
        verbose=False, assign_magmoms=False,
    )
```

`save_path` writes the trajectory observer's pickle format. `loginterval` controls
recording, and the final frame is always included. Final magnetic moments require
an `m` prediction task.

</details>

<details>
<summary><strong>GPU requirements and execution details</strong></summary>

The GPU path uses FIRE and whole-step CUDA Graphs. It requires CUDA, full periodic
boundary conditions, unconstrained atoms, no activation checkpointing, and the
default `FrechetCellFilter` for variable cells. Model support includes MatRIS,
MACE's `ScaleShiftMACE`, SevenNet, ORB v3, and CHGNet, subject to adapter checks.
MatRIS variable-cell relaxation requires a conservative model.
Trajectory records must fit the GPU memory budget; unsupported workloads use ASE
under `engine="auto"`.

Capture and setup cost time, so short relaxations are not guaranteed to be faster.
`StructOptimizer` defaults to `auto_policy="latency"`, which skips uncached online
memory probes; `auto_policy="throughput"` permits calibration for longer workloads.
Inspect `config_summary()["relaxation"]["last"]` for the selected path and, for
GPU runs, timings.

For lower-level batching and cell-filter control, use `MultiSystemRelaxation`:

```python
from ase.io import read
from fastmd import InferenceConfig, MultiSystemRelaxation

with MultiSystemRelaxation(
    model="mace", model_path="/path/to/model.model",
    structures=read("inputs.extxyz", ":"), device="cuda",
    cell_filter="frechet", fmax=0.05, max_steps=500,
    config=InferenceConfig(execution="whole_step", optimization_profile="topology"),
) as runner:
    results = runner.run()
```

Use `cell_filter=None` for fixed cells. The `topology` profile in this example
requires the [cuEquivariance environment](#installation). Batched execution retains
its model capability checks.

For ORB multi-system relaxation, set `InferenceConfig(matmul_precision="highest")`
for IEEE FP32 and keep `self_check=True`. Explicit `matmul_precision="high"`
enables TF32; its rounding can depend on batch shape and fail the check against
single-system results. Match precision settings when comparing with an official
ORB baseline, whose `float32-high` default differs from IEEE FP32.

Near MatRIS's hard triplet cutoff, small geometry differences can change neighbor
membership and the relaxation trajectory. Check convergence when comparing
relaxed structures across execution paths.

</details>

<details>
<summary><strong>Migration from earlier relaxation APIs</strong></summary>

The functional `relax` / `RelaxationConfig` API, `make_relaxation_calculator`,
`fastmd-relax` CLI, and TorchSim `relax_batch` have been removed. Use
`StructOptimizer` for relaxation. The compatibility import
`fastmd.applications.relax.StructOptimizer` resolves to the same implementation as
`fastmd.interfaces.relax.StructOptimizer`.

</details>

## Layout and extension points

```text
fastmd/
  models/            Model adapters, capabilities, patches, and the MatRIS implementation
  engine/            Configuration, model protocols, derivatives, capacity, and caches
  execution/         Integrators, thermostats, CUDA Graphs, logging, and batched execution
  interfaces/        ASE calculators, MD, relaxation, and multi-system interfaces
  graph/             Neighbor lists, atom/line graphs, fixed-capacity layouts, and topology
  ops/               Shared PyTorch, Triton, and CUDA operators
  integrations/      Experimental M3GNet topology integration
  adapters/          Shared adapter helpers and compatibility imports
  applications/      Compatibility forwarding for legacy application imports
  config.py          Configuration compatibility entry point
  precision.py       Precision configuration and records
```

To add an MLIP, implement `ModelAdapter` (available through `adapters/protocol.py`) and expose an `EnergyTap`, `SystemEnergyTap`, or `ModelOwnedDerivative` interface. Pass the adapter as `model=` to reuse the simulation runtime. Declare fixed-shape capture, padding, and cutoff-continuity capabilities accurately so the runtime can select a supported execution path.

## Development and license

| Resource | Description |
| :--- | :--- |
| [Tests](tests/README.md) | Regression coverage and validation commands |
| [Developer tools](tools/README.md) | Layering checks and backend validation |
| [License](LICENSE) · [Third-party notices](NOTICE) | BSD 3-Clause license and dependency notices |
