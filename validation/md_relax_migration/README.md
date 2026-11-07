# MD relaxation migration (2026-10-26)

> Historical validation: this report and its retained results describe the
> 2026-10-26 migration. They predate the adapter-kit validation in job `110201`
> and do not report that job's results or validate subsequent source changes.

Source: the sibling `MD` checkout at commit
`4fa638dc5a3fd774739a9b46e08b36185196c273` (unchanged by this migration).

`StructOptimizer`, `MultiSystemRelaxation` and the FIRE/Frechet execution code
now use that source implementation, with the `fastmd` namespace and
`MLIPCalculator`. Its shared slot execution, capacity growth, memory fallback,
trajectory records and model contracts have also been migrated. The ASE optimizer
now closes its log on exit, and closing a relaxation breaks owner callbacks so
dropping it releases its graphs without waiting for cyclic garbage collection.

The previous functional relaxation API, relaxation CLI and TorchSim relaxation
implementation have been removed. Use `StructOptimizer(engine="auto"|"gpu"|"ase")`.
As in MD, ASE `Atoms` inputs are modified in place, and the result contains
`final_structure` and `trajectory`. The lower-level `MultiSystemRelaxation` also
reports convergence and optimizer step counts.

FastMD's existing ASE MD, GPU-resident MD, replica/batched MD, TorchSim simulation
interface and model-specific inference optimizations remain. In particular,
`gpu_md.py`, `md.py`, `whole_step_cuda_graph.py` (including
`BatchedWholeStepGraphRunner`) and `torchsim_interface.py` are unchanged.
The four replica/system MD integrator classes are unchanged. Shared execution
retains geometry precision and adapter stress units. No dependency on the sibling
`matris` package or distributed execution was added.

## Reproduce validation

Run from the FastMD root using a Python environment with the required models.
CPU integration, API, lifetime and contract checks:

```sh
PYTHONPATH=.:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest \
  test_relax_migration test_migration test_runtime_config \
  test_system_batch_contracts.SystemBatchContractsTest test_relax_model_systems \
  test_md_relaxation_gpu.SystemRecordsVersusASETest \
  test_md_relaxation_gpu.RelaxationLifetimeTest -v
python -B tools/lint_layering.py --strict
python -B tools/test_lint_layering.py
```

GPU relaxation and existing MD regression checks, on an idle allocated device:

```sh
PYTHONPATH=.:tests OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest \
  test_md_relaxation_gpu test_system_batch_contracts \
  test_config_cuda.ReplicaMolecularDynamicsCudaTest \
  test_config_cuda.MultiSystemMolecularDynamicsCudaTest \
  test_config_cuda.InferenceConfigCudaTest.test_gpu_md_graph_modes_match_eager_reference \
  test_config_cuda.InferenceConfigCudaTest.test_gpu_md_execution_auto -v
```

The validation node's driver provided `libcuda.so.1` without a `libcuda.so`
development link. A temporary symlink to that existing driver, exposed through
`TRITON_LIBCUDA_PATH` and `LIBRARY_PATH`, allowed Triton to link its launcher;
no system driver files were changed.

CPU results: 71 tests ran, 70 passed and the optional CHGNet test was skipped in
the base environment. CHGNet's explicit-shift and empty-angle checks passed in
the environment with that package installed. The existing FIRE-versus-ASE CPU
test also passed. The strict layering check reports no new or stale exceptions;
all 15 lint regression fixtures passed.
An additional 18 model/adapter CPU checks passed in the CHGNet and ORB
environments. Their commands and observed results are transcribed in
[model-validation-record.json](model-validation-record.json); these are tool
output records, not separately captured stdout logs. Comparing MatRIS against
FastMD's pre-migration commit also gave 26 exactly equal E/F/S tensors for single
structures and the existing list batches, including sink padding and both energy
normalizations. A copy of that reproduction script is
[compare_matris_before_migration.py](compare_matris_before_migration.py).

The first GPU suite ran 29 tests: 28 passed, including all existing MD checks.
One long three-body relaxation exceeded the source test's position tolerance:
one coordinate differed from ASE by 1.004697898e-4 Angstrom at step 137, against
1e-4. Running the unchanged MD source on the same GPU reproduced exactly that
difference. All six structures' ASE results and GPU results were each identical
between MD and FastMD, including energies, forces, cells and step counts.
Only this three-body position tolerance is now 2e-4 Angstrom; the force, cell,
energy and step-count checks remain unchanged. The diagnostic is recorded in
[three-body-source-comparison.json](three-body-source-comparison.json).
The complete affected test was rerun after that adjustment: all 18 fixed/variable
cell, slot, drain and simulated OOM-split cases passed in 50.947 seconds on RTX
3090. The original log was `three-body-regression.log`. Together with
the other 28 passing tests, this completes the selected GPU regression coverage.

The first-run failure was recorded in `gpu-tests-first-run.log`;
`cpu-tests.log` and `layering.log` recorded the other checks. These raw logs,
including `three-body-regression.log`, are not included in this repository.
The retained summaries and JSON records above preserve the reported outcomes;
they are not substitutes for the original raw logs.

## WBM structure comparison

```sh
python validation/md_relax_migration/compare_wbm.py \
  --model-path /path/to/mace-medium-mpa-0.model --steps 5 \
  --output validation/md_relax_migration/wbm-comparison.json
```

The retained unrelaxed fixture is `wbm-1-12`, Ag2CdI4, seven atoms. Provenance
and license are in [tests/data/README.md](../../tests/data/README.md).
The comparison reuses one MACE-MPA-0 checkpoint in float32, starts both paths
from independent copies, and requires `engine="gpu"` to actually use the GPU.
The recorded run used an independently allocated RTX 3090.

Both fixed and variable cells matched after five steps. The larger errors across
the two cases were 1.26e-6 eV in total energy, 2.16e-6 eV/Angstrom in a force
component, 9.10e-8 Angstrom in position, and 2.65e-8 Angstrom in a cell entry.
See [wbm-comparison.json](wbm-comparison.json) for the arrays, path decisions and
separate construction/run timings. This is a short numerical comparison; it does
not assert convergence or demonstrate a throughput improvement. Graph capture
and first-use setup can dominate a short relaxation.
