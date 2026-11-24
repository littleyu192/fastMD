# Examples and benchmark results

[← FastMD](../README.md) · [MatRIS](#matris-relaxation) · [Batch relaxation](#batch-gpu-relaxation) ·
[Options](#relaxation-options) · [Benchmarks](#s01-benchmark-details)

Follow the [installation instructions](../README.md#installation) and run commands from the repository root.
Model weights are supplied separately. The main README includes a [MACE relaxation example](../README.md#structure-relaxation)
and [molecular dynamics usage](../README.md#molecular-dynamics) for [simulate.py](simulate.py).

## Local quick start

[quickstart.md](quickstart.md) shows installation inspection, deterministic input generation and
a calculator run with explicit local weights. [quickstart.py](quickstart.py) runs these examples
without depending on the optional documentation directory. Model and kernel validation remain
separate from package discovery.

## MatRIS relaxation

Replace the path with your local MatRIS checkpoint:

```python
import torch
from ase.build import bulk
from fastmd import StructOptimizer

atoms = bulk("Cu", "fcc", a=3.7, cubic=True)
atoms.rattle(0.05, seed=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

with StructOptimizer(
    model="matris_10m_oam", model_path="/path/to/MatRIS_10M_OAM.pth.tar",
    device=device, task="efs", optimizer="FIRE", enable_compile=False,
    engine="auto",
) as optimizer:
    result = optimizer.relax(
        atoms, fmax=0.05, steps=500, relax_cell=True,
        ase_filter="FrechetCellFilter", assign_magmoms=False, verbose=False,
    )
    print(result["trajectory"].energies[-1])  # Final total energy in eV
    print(result["final_structure"].lattice.abc)  # pymatgen Structure
    print(optimizer.last_relaxation)  # Selected path and its reason
```


For a fixed cell, use `task="ef"` and `relax_cell=False`. The [relaxation API overview](../README.md#structure-relaxation)
describes engine selection, return values, and input handling.

## Batch GPU relaxation

Use concurrent GPU slots for multiple structures:

```python
from ase.build import bulk
from fastmd import MultiSystemRelaxation

structures = [bulk("Cu", "fcc", a=a, cubic=True) for a in (3.5, 3.6, 3.7)]
with MultiSystemRelaxation(
    model="matris_10m_oam", model_path="/path/to/MatRIS_10M_OAM.pth.tar",
    structures=structures, device="cuda", slots=2,
    cell_filter="frechet", fmax=0.05, max_steps=500,
) as runner:
    results = runner.run()
    for result in results:
        print(result.index, result.status, result.steps, result.energy)
        print(result.atoms.cell.lengths())
```

This entry point requires CUDA, fully periodic unconstrained `Atoms`, and a compatible adapter supporting
union or segment batching and per-edge shifts. It copies inputs and returns one `RelaxationResult` per
structure, in order. Status is `converged`, `max_steps`, or `failed`; `converged` is also a boolean property.
Each result's `atoms` has a `SinglePointCalculator` with final energy, forces, and variable-cell stress.
Use `cell_filter=None` for fixed cells and `trajectory_interval` to retain records.
This direct GPU entry point has no ASE fallback. Unsupported inputs raise; completed results remain in
`runner.results` if execution later stops with an error.


## Relaxation options

These settings apply to `StructOptimizer`; the direct `MultiSystemRelaxation` entry point follows the
batch API described above.

**GPU path.** Requires automatic calculator execution, checkpointing off, `capacity.warmup >= 1`,
fully periodic nonempty structures without constraints, energies and forces, and no `crystal_feas_save_path`.
Use FIRE with its defaults or valid `dt`, `maxstep`, `dtmax`, `Nmin`, `finc`, `fdec`, `astart`, `fa`, and `a`
options, plus a fixed cell or `FrechetCellFilter`. Custom restart, logging, trajectory or position-reset
callbacks and downhill checks use ASE. Supported adapters include MatRIS, MACE's `ScaleShiftMACE`, SevenNet,
ORB v3, and CHGNet, subject to capability checks. Trajectory buffers must fit within 1/16 of device memory;
lists may use fewer concurrent slots to meet that budget.

With `engine="auto"`, construction failures or failed/non-finite GPU results fall back to ASE from the
affected structure's input geometry; successful GPU results in a list are retained. CPU, other optimizers,
and explicit `eager` or `model_graph` calculator execution use ASE. Available optimizers are FIRE, BFGS,
LBFGS, BFGSLineSearch, LBFGSLineSearch, and MDMin. `last_relaxation` reports `gpu`, `ase`, or `both`, counts,
and GPU construction/relaxation times when available; it is also in `config_summary()["relaxation"]["last"]`.

**Trajectories.** Records contain energies, forces, `atom_positions`, cells, and optional stresses/magnetic
moments. `loginterval` sets the interval; a terminal record is appended even if that step was recorded,
so record count is not optimizer step count. `save_path` writes a pickle; for lists, both it and
`crystal_feas_save_path` must contain `{index}`. On the GPU path, magnetic moments for an `m` task are computed
only for the final record. MatRIS interaction blocks run uncompiled there; `enable_compile` applies to ASE.

**Calculator policy.** Keyword-only `auto_policy="latency"` reuses compatible measurements and skips uncached
online calibration on the automatic ASE path. `"throughput"` permits calibration; experimental `"latency_trial"`
attempts model-graph capture using eager peak memory and device headroom, falling back to eager on rejection
or failure. Its bound is heuristic and capture still costs time. Measurements are specific to the model,
configuration, and exact geometry, subject to mutation and capacity checks. Inspect
`optimizer.calculator.memory_probe_summary()` or `capture_trial_summary()` for details.
These policies do not disable GPU FIRE's own captures or sizing work.


## S01 benchmark details

The S01 campaign contains **41 checkpoint pairs across five model families**.
**Speedup = official full-run time / FastMD full-run time.** These measurements describe the workload and
setup below; the [main README summarizes the results](../README.md#s01-benchmark-results).

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

Displayed step times are the complete MD wall times divided by 80,000. Each checkpoint has a complete
official/FastMD pair.

### MatRIS

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [matris_10m_oam](https://ndownloader.figshare.com/files/59142728) | 154.5975 | 28.0180 | 5.5178× |
| [matris_10m_mp](https://ndownloader.figshare.com/files/59143058) | 164.6329 | 32.0987 | 5.1290× |

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

### CHGNet

| Checkpoint | Official (ms/step) | FastMD (ms/step) | Speedup |
| --- | ---: | ---: | ---: |
| [chgnet-0.3.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar) | 53.3781 | 2.7588 | 19.3484× |
| [chgnet-0.2.0](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar) | 52.0884 | 2.4750 | 21.0455× |
| [chgnet-r2scan](https://github.com/CederGroupHub/chgnet/blob/main/chgnet/pretrained/r2scan/chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar) | 51.5925 | 2.7956 | 18.4548× |
