# Single-device validation tools

These tools are adapted from MD commit
`3dc3743d4c1e8617e26004c9b84cc7698e9ffd8c` for the `fastmd` package.
Use Python 3.10 or newer and the appropriate existing model environment.
Run commands from the FastMD checkout root.

## Architecture checks

```bash
python -B tools/lint_layering.py --strict
python -B tools/test_lint_layering.py
```

The linter reads Python syntax without importing model dependencies. It checks
layer direction, model/engine boundaries, eager optional imports, runtime/capture
ownership, and compatibility paths. `lint_layering_allowlist.txt` records precise
pre-existing exceptions in the published source tree.
Strict mode rejects new violations and stale entries. This baseline does not mean
that the existing source is free of coupling; it allows checking future changes
without changing the current model implementations.

The tool integration tests run entirely on CPU, using simulated GPU status and
real child processes:

```bash
python -B utils/test_validation_tools.py
```

## GPU execution wrapper

On an idle GPU allocated to you:

```bash
PYTHONPATH=.:tests python -B utils/run_gpu_validation.py \
  --gpu GPU-UUID --prefix /new/output/kernel-check --timeout 900 -- \
  python -B -m unittest -v test_fused_kernel_accuracy test_saved_recompute
```

Replace `GPU-UUID` with the allocated device UUID. The wrapper checks only the
selected GPU, records production source hashes and preflight state, and saves
`.log` and `.result.json` files. It rejects an occupied device or an existing
output prefix. Timeout cleanup targets its own child process group. A successful
child whose production source changed during the run is reported as a failure.
The wrapper does not allocate a GPU, stop other workloads, or install packages.

## Model validation lanes

The four entry points in [ci/README.md](ci/README.md) cover MACE, SevenNet, ORB and
CHGNet. They reuse existing environments and local weights, verify dependency
versions and checkpoint hashes, and run the existing adapter tests. Optional
benchmark runs use the included `benchmarks/multimodel/run.py` entry point. Other
local benchmark assets remain ignored by Git. The single-device lanes do not
include upstream's multi-GPU tier.

## Rebuild-on-every-call energy/force baseline

```bash
python -B utils/benchmark_single_ef.py \
  --model /path/to/MatRIS_10M_OAM.pth.tar \
  --input /path/to/periodic.extxyz --output /new/output/baseline.json \
  --execution model_graph --profile topology --steps 30 --warmup 3
```

This MatRIS-specific baseline rebuilds topology on every evaluation and checks
energy/force precision against a reference path before timing. It records the
actual device initial state, performance and optional short MD results. It uses
an existing checkpoint and refuses to overwrite output. It is not an additional
model adapter or a replacement for the multi-model benchmark.

See [the regression test guide](../tests/README.md) for CPU checks, optional assets
and GPU-only tests.
