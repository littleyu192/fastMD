# Single-GPU validation lanes

These four lanes retain MD's single-GPU adapter checks and benchmark presets:

```bash
bash tools/ci/mace_lane.sh --venv /path/to/mace-env --check-only
bash tools/ci/sevennet_lane.sh --python /path/to/python --gpu 0 --tier 1
bash tools/ci/orb_lane.sh --venv /path/to/orb-env --gpu 0 --tier 1,2 \
  --checkpoint /path/to/orb-v3-conservative-inf-mpa-20250404.ckpt \
  --large-structure /path/to/large.extxyz --small-structure /path/to/small.extxyz \
  --out /new/path/orb-results
```

The lanes **reuse existing environments and local weights**. They do not install
dependencies, download models, or change model adapters. The upstream multi-GPU
tier and environment creation commands are deliberately omitted. The four entry
points share `_single_gpu_lane.py` to keep their validation rules consistent.

Before testing, the selected interpreter must match every exact version in
`requirements-FAMILY.txt`. Package locations and available installation provenance
are printed, FastMD must import from this checkout, and checkpoint files must match
the adapter's pinned SHA256. Installed package versions are checked; wheel archive
hashes cannot be reconstructed from installed packages. The original requirements
files include CUDA 13 / aarch64 wheel pins. For another platform or an intentionally
different environment, supply an independently reviewed exact-version requirements
file with `--requirements FILE`. A mismatch is an error, never an automatic upgrade.
`--check-only` checks imports and weights without executing GPU tests.

MACE uses its existing `$XDG_CACHE_HOME/mace` cache; SevenNet and CHGNet use packaged
weights. ORB accepts `--checkpoint`, `MATRIS_ORB_CHECKPOINT`, or an existing
cached-path cache entry. Tier 2 receives explicit verified checkpoint paths;
CHGNet keeps its name-based loading of verified packaged weights. A benchmark run
does not fetch missing weights. `MATRIS_*` test-asset variables retain
their existing names because FastMD's tests still consume those names.

Tier 1 runs the existing family adapter test module and fails on any skipped test,
failure, error, or zero passing tests. It includes the optional cuEquivariance
checks for MACE/SevenNet. Tier 2 runs only after tier 1 passes when both are selected:

| Family | Presets |
| --- | --- |
| MACE | MP-0 FP32 large structure; MP-0 and MPA-0 FP64 small structure |
| SevenNet | SevenNet-0 FP32 large and small structures |
| ORB | ORB v3 conservative FP32 large and small structures |
| CHGNet | CHGNet 0.3.0 FP32 large and small structures |

These are the upstream baseline presets, not a replacement for FastMD's broader
model/checkpoint support. Small-structure runs use 500 K, 0.25 fs and ten steps per
launch. The original capacity, warmup and precapture settings are retained.
Results must use a fresh output directory. Tier 2 uses the included
`benchmarks/multimodel/run.py` entry point; structure files and model weights must
be provided separately. Other local benchmark assets remain ignored by Git.
A missing harness is reported before any validation subprocess starts.

All GPU executions require `--gpu` with one index or UUID. The child environment
removes inherited distributed-launcher variables and TF32 override, uses this
checkout and disables user site packages. `--cpus` optionally applies `taskset`.
Run lanes under `utils/run_gpu_validation.py` to add selected-GPU idle checking,
process-group timeout cleanup, source hashes and persistent logs.
