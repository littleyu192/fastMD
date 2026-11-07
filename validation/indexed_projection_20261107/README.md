# CHGNet / ORB indexed projection

These opt-in inference paths apply the identity
`cat(E, X[i], X[j]) W.T = E We.T + (X Wi.T)[i] + (X Wj.T)[j]`.
Projecting source nodes before gathering avoids repeating their matrix products
for every edge. Dense products retain PyTorch's precision policy and GEMM
implementation; the Triton candidate fuses gather/add/SiLU and its input VJP.

Select the rewrite once, before preparing inputs or constructing a calculator
or MD driver:

```python
from fastmd import load_adapter

chgnet = load_adapter("chgnet_0_3_0", device="cuda")
chgnet.enable_operator_rewrites("triton")  # "torch" is also available

orb = load_adapter("orb_v3", device="cuda")
orb.enable_operator_rewrites("torch")
# On a fresh adapter, test the additional projection epilogue separately:
# orb.enable_operator_rewrites("torch", projection_backend="triton")
```

CHGNet combines the first core/gate projections, projects undirected bond
features before gathering directed edges, and preserves normalization order
for output-only angle updates. When projecting complete source tables would
cost at least as much as projecting gathered rows, it retains the original
concat/Linear. This matters for sparse and padded angle graphs. Its Triton
path supports FP32 first derivatives for forces/stress; select Torch for
`create_graph=True` or Hessians.

ORB already had the Torch decomposition. The new `projection_backend` selects
the extra epilogue independently of the aggregation backend. CPU, FP64,
unsupported layouts and deterministic policies use Torch operations; a
differentiable backward handles higher derivatives. This option cannot be
combined with `steady_options`.

Installation checks upstream versions/source hashes and acts on model
instances. Weights stay frozen; `close()` restores patched methods. Defaults
remain upstream. These choices are separate from MatRIS's global fusion
switches.

## Measured scope: S01, 2026-11-07

Job 110201, H100 80GB, PyTorch 2.8/CUDA 12.8, CHGNet-0.3.0 and
ORB-v3-conservative-inf-mpa. All measurements use FP32/highest with TF32 off.
The earlier S01 CHGNet 1.001x result was a negative control: neither arm
dispatched these operators. MatRIS's published fusion ratios switch several
operators together and cannot be attributed to this pattern alone.

Whole-step MD uses 135 atoms, FP64 integration, 500 K NVT NHC, 0.25 fs,
10 steps per launch, recomputation off and E/F/S output every 10 steps.
Each cell contains five independent 1000-step processes; discard the highest
and lowest duration and average the remaining three. Timing excludes initial
preparation but includes recapture, output, close/flush and final synchronization.

| Path | Whole-step ms/step | Upstream / path |
| --- | ---: | ---: |
| CHGNet upstream | 2.4932 | 1.000x |
| CHGNet Torch | 2.5588 | 0.974x |
| CHGNet Triton | 2.4534 | 1.016x |
| ORB upstream | 7.2457 | 1.000x |
| ORB Torch | 6.7347 | 1.076x |
| ORB Torch + Triton projection | 6.4842 | 1.117x |

The separate 1350-atom fixed-geometry E/F/S benchmark takes the median of five
samples of 100 CUDA Graph replays. It includes coordinate/strain derivatives,
and excludes graph building, integration and output:

| Model | Upstream ms | Torch ms | Triton ms | Upstream / Torch | Upstream / Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| CHGNet | 2.1783 | 2.1565 | 1.8273 | 1.010x | 1.192x |
| ORB | 26.2025 | 22.6093 | 22.6934 | 1.159x | 1.155x |

With the same real second-layer features and input cotangent, the 1350-atom
gated/edge MLP forward plus VJP microbenchmark gives 1.316x for CHGNet and
1.252x for ORB (upstream versus Triton). These use eager operators captured in
CUDA Graphs and exclude attention, aggregation and other model work.

ORB's upstream-to-Torch model comparison also includes existing aggregation
and attention execution-order rewrites. Only Torch versus the extra Triton
projection holds those choices fixed. The extra epilogue did not improve
compiled fixed-geometry E/F/S. Whole-step candidate timing ranges overlap;
capacity growth and recapture contribute substantial variation. For example,
one default-capacity ORB run rebuilt runners twice, spending 1.20 seconds in
their construction, warmup and capture. Increasing headroom did not improve
the diagnostic runs, so default capacities remain unchanged.

All 30 formal trajectories passed 101-frame integrity checks and 17 same-geometry
generic-eager E/F/S comparisons per trajectory. Limits are 0.01 eV total energy,
0.001 eV/Angstrom force and 0.0001 eV/Angstrom^3 stress (the larger static test
scales the energy limit by atom count). This checks implementation consistency,
not DFT accuracy or an independent official graph builder. Three dense-angle
Cu configurations additionally verified actual CHGNet cat4 forward/backward
execution. The five ORB projection GPU unit tests passed, including derivatives,
fallbacks and lifecycle behavior. GPU process monitors were clean.

Raw results, frozen source, scripts and hash manifests are retained in the
development workspace under `records2610/indexed_fusion_20261107/`, outside this
Git repository; this document carries the portable result summary. The timings
refer to that frozen workspace snapshot. The isolated Git changes also undergo
adapter and architecture regression checks before submission.
