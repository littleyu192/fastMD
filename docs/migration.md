# Inference migration

fastMD exposes one ASE Calculator backed by lazy model adapters. The original
repositories and the supplied reproduction bundle remain independent of the
installed package. No training utilities, platform-specific binaries, benchmark
HDF5 datasets, or MACE/MatRIS checkpoints are required from adjacent directories.

## Sources

| Backend | Source | Integration |
| --- | --- | --- |
| MatRIS | Optimized `reb/third_party/matris-09bk` at `ab9ece74`, plus all five bundled patches through `1faa60f9` | Topology profile, typed scoped settings, bucketed graph runner |
| CHGNet | `origin/CHGNet_CG` at `1d362271` | Model graph runner connected directly to ASE; composition-energy correction |
| ALIGNN | `origin/ALIGNN_CG` at `b2ac35d5` | Minimal inference context for the model graph runner |
| MACE | `reb/mace_opt`, targeting `mace-torch==0.3.16` | Tensor inputs, Triton neighbor lists, fused forward/backward and optional compiled modules |

The complete file inventory, source hashes, patch hashes and local adaptations
are in [sources.json](sources.json). Retained source licenses and attribution
are listed in [../NOTICE](../NOTICE).

## Public boundaries

- `calculator.py`: ASE caching, requested properties, result ownership, units,
  finite-value/shape checks and `free_energy` compatibility.
- `models/base.py`: device selection, declared capabilities, explicit fallback,
  and capture invalidation on species/order, atom count, cell or PBC changes.
- `models/registry.py`: lazy factories; adding a model does not add branches to
  the common Calculator or eagerly import unrelated optional dependencies.
- `models/<model>.py`: checkpoint loading, native inputs, graph runner ownership,
  extensive energy/ASE stress conversion and statistics.
- `_vendor/`: private numerical implementations and source-level attribution.
  Import paths, scoped MatRIS module bindings, the tuning subprocess and MatRIS
  custom-op registrations use private namespaces.

One backend belongs to one serial workflow. Returned NumPy arrays own their
storage, so later graph replays cannot overwrite previously returned values.
Known capability limitations can fall back in `auto`; unexpected runtime
errors propagate. Failed calculations clear ASE results before inference.

## Optimized MatRIS

The `topology` profile is now the default whenever `enable_fusions=True`.
It connects verified graph topology to indexed projections, merged frozen
projections, segmented attention/reductions and the other eligible Triton
lowerings. Selection remains conditional on row counts, device, dtype and
feature widths. `enable_fusions=False` selects the generic profile.
Activation checkpointing is disabled for coordinate/strain inference.

The five source patches cover:

1. Fixed-capacity edge overflow at the sentinel boundary.
2. Lattice/fractional-coordinate matrix multiplication without TF32 loss.
3. GPU logging producer/consumer stream ordering.
4. Native ASE-order Nose–Hoover chain integration.
5. Thread-local CUDA capture and logger failure handling.

Single-GPU application sources carrying these fixes are preserved privately.
Only inference is exposed through the public Calculator. Distributed experiments
and the separate M3GNet integration are excluded. Optional Blackwell CuTeDSL
sources and operator compilation are retained; default inference uses the
portable PyTorch GEMM backend and does not require CuTeDSL.

## MACE

The migrated numerical runtime comprises `tensor_batch.py`, `neighbors.py`,
`fast_forward.py`, and `fusion/`. Benchmark-specific MD drivers, whole-step NHC
capture, frame loggers and HDF5 fixtures are outside the ASE inference adapter.

The public runner captures both neighbor maintenance and model derivatives.
Fixed species/cell data remain on the GPU. A skin allows candidate-list reuse;
Triton kernels filter active edges on every evaluation. Every replay checks
neighbor consistency and capacity. Overflow invalidates captured pointers,
grows storage, forces a complete candidate rebuild and retries the same input.
Predictions from truncated lists are never returned.

`plain` captures upstream MACE, `fast` enables the migrated edge and derivative
fusions, and `fast_cm` also compiles original GEMM-level submodules. The latter
reports per-module eager fallbacks. CPU and explicitly eager inference use
upstream MACE for a reference. MP-0 medium and MPA-0 medium are the initial
supported CUDA architectures; this is not a claim of universal MACE support.

Local adaptations remove the optimizer package's import-time loading environment
mutation, restore default dtype on exceptions, validate checkpoint paths, support
explicit heads, scope compiler limits, and select indexed CUDA devices correctly
when enabling optional cuEquivariance convolution fusion.

## Scope and performance

ASE still owns integration, constraints and callbacks. MACE's captured neighbor
update reduces work inside an inference call, but it does not make ASE's MD loop
a whole-step CUDA graph. The reproduction bundle's NHC throughput numbers are
not transferred to fastMD as measured speedups. See [validation.md](validation.md)
and use the comparison script on the target GPU and actual structures.

TorchSim inspired the adapter/capability boundary; it is not a dependency. The
current API handles one ASE structure at a time. Future WBM model adapters and
batched tensor interfaces can extend that boundary independently.
