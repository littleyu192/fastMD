# Validation record

Date: 2026-09-27. This records local evidence, not a GPU performance claim.

## Environments

- Existing adapters: Python 3.11, PyTorch 2.8.0+cu128, ASE 3.23.0, DGL 1.1.1.
- MACE: a temporary virtual environment using the same PyTorch, with
  mace-torch 0.3.16, e3nn 0.4.4, ASE 3.29.0 and matscipy 1.3.0.
- No NVIDIA GPU/driver is available on this node. Existing conda environments
  were not changed; additional MACE dependencies were installed in a temporary venv.
- Checkpoints: MatRIS 10M OAM, bundled CHGNet 0.3.0, local ALIGNN
  `v12.2.2024_dft_3d_307k`, and the supplied MP-0/MPA-0 medium checkpoints.

## Completed numerical checks

The original adapter suite passed **29 tests**, with **3 CUDA tests skipped**.
The additional MACE and MatRIS-configuration suite passed **12 tests**, with
**8 CUDA tests skipped**.

Coverage includes:

- ASE energy, forces, stress Voigt ordering, free_energy, result ownership,
  invalidation, warmup, constraints, MD integration and error handling.
- Real MatRIS, CHGNet and ALIGNN CPU evaluation and force finite differences.
- MatRIS/CHGNet stress, magnetic moments and sink-padding equivalence.
- ALIGNN static masks versus dynamic DGL inputs and original Calculator parity.
- MP-0 and MPA-0 energy/forces/stress versus upstream MACECalculator in FP64,
  at `rtol=atol=1e-9`.
- MACE force and diagonal-strain stress finite differences, including sign and units.
- MACE safe-edge padding and the unfused FastMace numerical forward versus
  upstream MACE on CPU, at `rtol=atol=1e-9`.
- MACE capacity overflow control: candidate overflow, model-edge overflow,
  exact-capacity acceptance, clearing captured pointers and forced rebuild.
- Missing local MACE checkpoints raise without an unintended download;
  dtype scopes restore state on exceptions.
- MatRIS generic/topology CPU parity and correct scoped binding/restoration
  of private module constants.

ALIGNN retains its source Calculator's training-batch force multiplier. For the
local checkpoint it is 6; the finite-difference comparison accounts for that
multiplier. See the README before using its default force scaling for MD.

## Pending GPU validation

CUDA capture/replay, fused Triton execution, neighbor-list rebuild/reuse and
capacity retries on real hardware, optional Inductor/cuEquivariance/CuTeDSL
paths, multi-device selection, peak memory, and speedups remain unverified here.
CPU equivalence checks do not validate those GPU kernels. No reproduction-bundle
speedup is claimed as a measured fastMD result.

Packaging checks also passed: the 0.2.0 wheel contains the new runtimes,
source licenses and bundled CHGNet checkpoint. It was installed into a separate
temporary directory and imported from outside this repository. Installed MACE
ran FP64 and FP32 single-point/stress evaluation and two ASE VelocityVerlet steps;
installed MatRIS ran with the topology profile. Importing `fastmd` did not import
MACE, DGL or pymatgen eagerly. No adjacent source directory was on PYTHONPATH.

The CUDA tests require actual capture (`cuda_graph=True`); they cannot pass by
silently running eager. MACE tests cover both checkpoints and plain/fast/fast_cm,
replay on changing positions, forced tiny-capacity recovery, periodic boundary
crossing, a skewed cell, changed composition/atom count and returned-array lifetime.

On the target GPU:

```bash
python -m pip install -e '.[matris,chgnet,cuda,mace,test]'
export FASTMD_MATRIS_CHECKPOINT=/absolute/path/MatRIS_10M_OAM.pth.tar
export FASTMD_MACE_MP0_CHECKPOINT=/absolute/path/mace-mp-0.model
export FASTMD_MACE_MPA0_CHECKPOINT=/absolute/path/mace-mpa-0-medium.model
# Add FASTMD_ALIGNN_CHECKPOINT in a compatible DGL environment.
python -m pytest -q
python examples/compare.py --model matris --checkpoint "$FASTMD_MATRIS_CHECKPOINT" --stress --compile
python examples/compare.py --model mace --checkpoint "$FASTMD_MACE_MPA0_CHECKPOINT" --stress --compile --output mace.json
```

The benchmark reports load/setup separately and compares eager, graph without
fusions, fused graph and optional compiled graph on identical frames. Timing
includes ASE host/device transfers and result validation. Validate representative
production structures and short NVE trajectories before interpreting throughput.
