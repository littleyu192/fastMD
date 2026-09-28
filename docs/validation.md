# Validation record

Date: 2026-09-27. This records local evidence, not a GPU performance claim.

## MatRIS ASE NPT update (2026-09-28)

MatRIS now opts into variable-cell graph reuse. `model_kwargs={"compute_stress":
True}` returns energy, forces and stress from each evaluation, including eager
inference, so ASE can cache all three properties for NPT.

The focused regression suite passed **37 tests**, with **4 CUDA tests skipped**
and 4 unrelated ALIGNN cases deselected, using the local MatRIS 10M OAM checkpoint:

```bash
export FASTMD_MATRIS_CHECKPOINT=/absolute/path/MatRIS_10M_OAM.pth.tar
python -m pytest -q tests/test_calculator.py tests/test_matris_config.py \
  tests/test_matris_npt.py tests/test_models.py -k 'not alignn'
```

New CPU coverage verifies all six stress components against strain finite
differences (step 5e-4; convergence also checked at 2e-4), energy/force/stress
caching, and four ASE NPTBerendsen steps with an evolving cell. It also compares
the address-stable padded workspace against fresh ragged graphs after compression,
expansion and shear, including changing edge/triplet counts. Other backends retain
cell-triggered invalidation; composition, atom count, ordering and PBC still
invalidate captures for variable-cell backends.

`examples/npt.py --device cpu --eager --steps 2` also completed for its default
64-atom Si cell, writing the ASE log and trajectory with zero cache invalidations.

The new CUDA tests require `cuda_graph=True` and check eager/replay agreement,
capture reuse for small cell changes, growth into another capacity bucket,
periodic boundary crossing, returned-array ownership and a short ASE NPT run.
The initial checks above ran on the management node without CUDA. A subsequent
run on **g08 / NVIDIA H100 80GB**, in job **102928** after its training task ended,
passed **37 tests with no skips** (108.331 seconds), including these CUDA tests.
The [NPT GPU report](../matris_npt.md) records the full environment, raw artifacts,
64/216-atom Si comparisons and measured default-Graph speedups of **3.957×/2.641×**
over default eager inference for end-to-end ASE NPT. No captures occurred in the
timed runs. The report also retains an initial stress finite-difference failure
across the model's hard three-body cutoff, reproduced in eager mode; checks away
from that cutoff passed without relaxing tolerances. This validates Graph/eager
equivalence, not long-time ensemble statistics or a globally smooth potential.
Run `python -m pytest -q tests/test_matris_npt.py` on the target GPU before
production use. The example uses Berendsen coupling for pressure equilibration;
it is not a validation of exact NPT fluctuation statistics.

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

Beyond the MatRIS NPT checks recorded above, CUDA capture/replay, fused Triton
execution, neighbor-list rebuild/reuse and capacity retries for the other
backends, optional Inductor/cuEquivariance/CuTeDSL
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
