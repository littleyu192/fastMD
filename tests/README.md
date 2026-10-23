# Single-device regression tests

The additional low-level tests are adapted from MD commit
`3dc3743d4c1e8617e26004c9b84cc7698e9ffd8c`. They use FastMD's current implementations
and coexist with its existing model, relaxation and TorchSim tests. They do not
require an MD checkout or import its distributed runtime.

Use Python 3.10 or newer in a suitable existing FastMD environment. From the
checkout root, include both the checkout and `tests` on the Python import path.
The test suite uses `unittest`; pytest is not required.

## Coverage

| Area | Modules |
| --- | --- |
| Operator selection and recording | `test_op_selection_recorder`, `test_op_selection_snapshot`, `op_selection_snapshots.py`, single-device golden files |
| Fused operators, forward and gradients | `test_fused_kernel_accuracy`, `test_triton_layernorm_act`, `test_csr_segment_sum`, `test_segment_partials`, `test_zero_skip_index` |
| Index widths and layouts | `test_int64_offsets`, `test_triton_large_rows`, `test_kernel_layout_fixes`, `test_kernel_policy` |
| Memory and capture | `test_saved_recompute`, `test_cuda_graph_capacity`, `test_projection_cache_provenance` |
| Shared MatRIS forward and precision | `test_shared_embedding_readout`, `test_shared_forward`, `test_shared_forward_guards`, `test_single_device_reference`, `test_tf32_geometry` |
| Benchmark state recording | `test_benchmark_single_ef` |

Mixed upstream files retain their applicable single-device cases. Cross-rank
communication, partitioning and distributed-evaluator cases are excluded. Shared
test fixtures are extracted into `single_gpu_fixtures.py` rather than importing
distributed test modules. Existing FastMD adapter tests remain unchanged.

The static int64 source check covers the migrated MatRIS, shared engine and graph
kernels. It is not a source audit of FastMD-specific ORB/eSEN kernels.

Two upstream expectations need adaptation for the current code and PyTorch 2.8:
the pre-refactor FP64 golden values remain unchanged, with an absolute tolerance
of `1e-8` for their E/F comparison (the same upstream checkout differs by up to
`8.23e-9` on this environment); independent stage/VJP and finite-difference
checks remain separate. Projection-cache tests follow the current contract:
Dynamo tracing does not retain Python cache holders, while eager capture retains
them and replay rejects modified sources. These adaptations do not change model
or runtime implementations.

## CPU checks

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MATRIS_RUN_PERF_TIER=1 python -B -m unittest -v \
  test_benchmark_single_ef test_csr_segment_sum test_cuda_graph_capacity \
  test_fused_kernel_accuracy test_int64_offsets test_kernel_layout_fixes \
  test_kernel_policy test_op_selection_recorder test_op_selection_snapshot \
  test_projection_cache_provenance test_saved_recompute test_segment_partials \
  test_shared_embedding_readout test_shared_forward test_shared_forward_guards \
  test_single_device_reference test_tf32_geometry test_triton_large_rows \
  test_triton_layernorm_act test_zero_skip_index
```

CUDA-only cases report skips when CUDA is unavailable; they are not counted as
numerically validated. `MATRIS_RUN_PERF_TIER=1` enables the CPU recorder and golden
format checks as well as optional GPU structural guards. These guards inspect
executed paths rather than asserting wall-clock performance. Existing `MATRIS_*`
configuration and asset variable names are retained for compatibility.

## GPU checks and assets

Run the same module list on one idle, allocated GPU, preferably through
`utils/run_gpu_validation.py` as described in [the tool guide](../tools/README.md).
Most numerical kernel tests use seeded toy models and need no model downloads.
The following larger checks are optional:

- `MATRIS_RUN_LARGE_ROW_TESTS=1`: int64 tests with large allocations; use an idle
  GPU with enough free memory. The individual tests also check their memory needs.
- `MATRIS_RUN_OP_SELECTION_SNAPSHOTS=1`: the full single-device golden matrix.
- `MATRIS_TEST_ASSETS`: directory containing `rdx_checkpoint.pth.tar`,
  `rdx_1344_1x1x1.extxyz`, and `rdx_10752_2x2x2.extxyz` for RDX cases.
- `MATRIS_TEST_CHECKPOINT` and `MATRIS_TEST_S01_FRAMES`: local MatRIS-10M-OAM weights
  and the S01 frames NPZ for real-model TF32/EFS checks. See
  `matris_test_assets.py` for the NPZ schema.

Missing optional assets are reported as skips. The golden harness can list its
single-device configurations without loading a model:

```bash
PYTHONPATH=.:tests python -B tests/op_selection_snapshots.py --list
```

The 78 retained single-device golden files were originally recorded on
GB300/PyTorch 2.12. A GPU/compiler change can legitimately change kernel selection;
inspect an observed difference before accepting any updated golden. CPU checks
validate the registry and serialization, not those GPU execution choices.

Model-family validation lanes additionally require the matching optional model
environment and cached weights; see [their guide](../tools/ci/README.md).

## Migration validation, 2026-10-23

The combined CPU run on Python 3.11 / PyTorch 2.8 enabled the structural perf tier
and included all 20 new regression modules, the existing migration smoke tests,
15 architecture-lint fixtures and 16 validation-tool integration tests:
**175 cases: 89 passed, 86 skipped, zero failures or errors**. The 86 skips require
CUDA, optional real-model assets, or the opt-in large-row allocation tests.

Strict architecture lint also passed with no new or stale violations against its
42 precisely documented existing baseline entries. All 33 added Python files
passed syntax checks; shell syntax and CLI help checks passed. SHA256 comparison
confirmed that all 287 pre-existing tracked/untracked files in the task snapshot,
including the ignored local benchmark, remained unchanged.

CHGNet and ORB environment/checkpoint preflight checks passed using local weights.
MACE and SevenNet preflight correctly reported that their existing CUDA 12
environments do not satisfy the CUDA 13 package pins in the original requirements
files. No dependencies were installed or changed.

GPU numerical tests were not run: the allocated GPUs were occupied by existing
workloads. The CPU results do not certify the skipped CUDA kernels or regenerate
the GPU golden matrix. The commands and asset requirements above remain available
for an idle allocated GPU.
