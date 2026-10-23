# Single-device operator-selection snapshots

These 78 goldens come from MD commit `3dc3743d4c1e8617e26004c9b84cc7698e9ffd8c`.
They retain the upstream observed operator choices and parameters, with the
`fastmd.op_selection_golden/1` schema namespace. The eight distributed
configurations were removed; remaining cross-file references were expanded and
re-encoded without changing the recorded single-device selections.

`tests/op_selection_snapshots.py` records kernel selections, fallbacks, and
per-site ATen operator counts. It compares them with these files to detect
unexpected changes in optimized execution paths. It does not measure latency.
Each configuration uses one GPU; multiple independent configurations can run
on separately allocated GPUs.

From the checkout root:

```bash
PYTHONPATH=.:tests python -B tests/op_selection_snapshots.py --list
PYTHONPATH=.:tests python -B tests/op_selection_snapshots.py --check --all --gpus 0
MATRIS_RUN_PERF_TIER=1 PYTHONPATH=.:tests python -B -m unittest test_op_selection_snapshot -v
```

Twenty `synth_*` cases need no model download. The remaining 58 configurations
require some of these files in `$MATRIS_TEST_ASSETS`:

| File | Content |
| --- | --- |
| `rdx_checkpoint.pth.tar` | MatRIS-1M fine-tuned on RDX, widths 128/32/32 |
| `rdx_1344_1x1x1.extxyz` | RDX crystal, 1,344 atoms |
| `rdx_10752_2x2x2.extxyz` | RDX crystal, 10,752 atoms |

Missing assets are reported as skips, never as successful GPU validations.
CPU tests validate the registry, canonical encoding, comparison behavior, and
recording determinism without these assets.

The upstream recordings were produced on GB300 with CUDA 13.2, PyTorch 2.12,
and Triton 3.7. GPU architecture and software changes can alter tile choices or
ATen sequences. Inspect differences before deliberately updating a baseline:

```bash
PYTHONPATH=.:tests python -B tests/op_selection_snapshots.py --check --all --out-dir /tmp/fastmd-opsel-run --gpus 0
PYTHONPATH=.:tests python -B tests/op_selection_snapshots.py --write --all --from /tmp/fastmd-opsel-run
```

`--strict` also treats configuration-resolution and runner-counter differences
as errors; ordinary checks already fail on changed operator selections,
outcomes, compiled programs, and inconsistent repeated warm evaluations.
The shared `/tmp/matris_gpu.lock` remains the default advisory lock so these
tests coordinate with the original MD harness when both are used locally.
