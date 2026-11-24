# Running a local example

Run these commands from the repository root after following the
[installation instructions](../README.md#installation). Use a separate environment for
families whose upstream dependencies conflict.

```bash
python -m fastmd.check --json
python -m fastmd.check --family mace --import
python examples/quickstart.py inputs --output silicon.extxyz
```

The check covers the nine families in the loader registry: MatRIS, MACE, MACE-SOYO,
SevenNet, ORB, CHGNet, TACE, NequIP and legacy eSEN. Package discovery, distribution
versions and actual imports are separate fields. A matching version does not validate
source hashes or prove that a CUDA extension compiles or runs. The report does not load
a model or execute a kernel, and does not download weights. Its `--family` exit status
checks package presence and, with `--import`, whether the import succeeds; version
differences are reported separately. A checkout's bundled MatRIS needs no separate
distribution. M3GNet uses a separate experimental API and is outside this registry.

Evaluate the generated 64-atom silicon structure with a local MACE checkpoint:

```bash
python examples/quickstart.py calculator --structure silicon.extxyz \
  --model mace --model-path /path/to/mace.model --device cpu
```

This prints total energy in eV, forces in eV/Å, ASE Voigt stress in eV/Å³, and the
resolved calculator configuration. CUDA execution uses `--device cuda`; request a
captured model with `--execution model_graph` if the family supports it. Change
`--model` and `--model-path` together for another family. NequIP and legacy eSEN need
their respective local packages/checkpoints; MACE-SOYO requires CUDA. Model-loading
errors remain errors: the example does not substitute another model. This small cell
is a usage example, not a benchmark or a correctness oracle.

The existing MD example accepts the same local checkpoint:

```bash
python examples/simulate.py --structure silicon.extxyz \
  --model mace --model-path /path/to/mace.model --driver gpu --device cuda \
  --ensemble nvt_nhc --temperature 300 --timestep 0.5 --steps 20
```

Use separate output directories when running several examples. For relaxation and
concurrent systems, see the runnable API snippets in [examples](README.md) and the
[relaxation API](../README.md#structure-relaxation). For one system on multiple GPUs,
follow the [distributed CLI and API](../README.md#distributed-simulation): one rank
per GPU, identical settings on every rank, full periodicity, and an audited shard
declaration are required. NequIP and eSEN currently use the single-device interfaces.
Inspect the actual kernel and execution configuration after loading; installation
checks alone cannot select or validate a model's accelerated kernel.
