# Discovery relaxation with fastMD

This workflow adapts fastMD's existing MatRIS, MACE, CHGNet and ALIGNN inference
backends to ASE geometry optimization. The default workload follows the
Matbench Discovery relaxation settings: FIRE, FrechetCellFilter, `fmax=0.05`
eV/Å, and at most 500 optimizer steps. Atomic positions, cell shape and volume
are relaxed at zero external pressure.

## The shared 10% subset

The committed [manifest](unique-prototypes-10pct-seed20261011.json) selects
**21,548 of the 215,488 unique WBM prototypes**. This is the integer floor of
10% of the population used by the default Discovery leaderboard view. The
complete WBM test set contains 256,963 structures; it is a different population.

Selection uses `sha256-rank-v1`: rank IDs by the SHA-256 digest of the decimal
seed, a NUL separator, and the material ID, encoded as UTF-8. With seed `20261011`,
take the first 21,548 without replacement, then sort the selected IDs for execution.
This makes selection independent
of input row order and random-number-library versions. The manifest records the
selection and its provenance so all checkpoints and inference modes can run the
same structures. Keep this manifest when reproducing a comparison.

The eligibility filter is the official summary's `unique_prototype` field.
Selection does not use predicted energies, convergence, runtime, or hardware
success. A failed relaxation keeps its material ID and failure record; it is
never replaced by an easier structure.

## Prepare the inputs

Install fastMD and your model's dependencies as described in the
[main README](../../README.md#1-installation). Run the following commands from
the repository root:

```bash
python -m fastmd.discovery prepare
```

This reuses the committed manifest, downloads and verifies the official
initial-structure ZIP, and writes the selected initial structures to:

```text
.cache/discovery/unique-prototypes-10pct-seed20261011.db
```

If the manifest is missing, `prepare` first downloads the official WBM summary
and generates the default selection. An existing matching structure cache is
reused. The structure cache is an ASE database; downloaded inputs and run outputs stay
outside Git. The material-ID manifest is the small, version-controlled input to
the benchmark. `prepare` prepares data only and does not launch relaxations.

The official inputs are:

| Input | Official file | Download |
| --- | --- | --- |
| Population and IDs | `2023-12-13-wbm-summary.csv.gz` | [Figshare](https://figshare.com/files/64706751) |
| Unrelaxed structures | `2024-08-04-wbm-initial-atoms.extxyz.zip` | [Figshare](https://figshare.com/files/48169597) |

The summary provides `material_id` and `unique_prototype`; the initial structures
carry `material_id` in their ASE `Atoms.info`. DFT-relaxed structures are reference
data and must not be substituted for these initial geometries.

To use an existing initial-structure download:

```bash
python -m fastmd.discovery prepare \
  --initial-archive /path/to/2024-08-04-wbm-initial-atoms.extxyz.zip
```

For a separate sample, create its own manifest and database rather than replacing
the committed default. For example:

```bash
python -m fastmd.discovery sample --seed 42 --fraction 0.1 \
  --output .cache/discovery/seed42.json
python -m fastmd.discovery extract --manifest .cache/discovery/seed42.json \
  --output .cache/discovery/seed42.db
```

Run custom inputs with `run --manifest ... --structures ...` and label the
different selection in any comparison. Inspect all command-specific options with:

```bash
python -m fastmd.discovery prepare --help
python -m fastmd.discovery sample --help
python -m fastmd.discovery extract --help
```

## Run a model

First check loading, inference and output on a small part of the same manifest:

```bash
python -m fastmd.discovery run --model matris \
  --checkpoint checkpoints/MatRIS_10M_OAM.pth.tar \
  --device cuda --output runs/matris-smoke --limit 2 --steps 3
```

A three-step smoke run tests the workflow; it is not a convergence study or a
full-subset result. Use a separate output directory for the full run:

```bash
python -m fastmd.discovery run --model matris \
  --checkpoint checkpoints/MatRIS_10M_OAM.pth.tar \
  --device cuda --output runs/matris-discovery

python -m fastmd.discovery run --model mace \
  --checkpoint checkpoints/mace-mpa-0-medium.model \
  --device cuda --output runs/mace-discovery

python -m fastmd.discovery run --model chgnet \
  --device cuda --output runs/chgnet-discovery
```

MACE MP-0 medium and MPA-0 medium are the supported capture architectures. Use
the same checkpoint for eager/Graph numerical comparisons. Model options can be
passed as JSON, for example `--model-kwargs '{"default_dtype":"float64"}'` for
MACE. The usual model defaults remain in effect when an option is omitted.

| Backend | Default position + cell relaxation | With `--fixed-cell` |
| --- | --- | --- |
| MatRIS | CUDA Graph energy/force/stress inference and default kernel fusion | CUDA Graph inference |
| MACE | CUDA Graph energy/force/stress inference and default kernel fusion, when the architecture is supported | CUDA Graph inference |
| CHGNet | Eager energy/force/stress inference | CUDA Graph energy/force inference |
| ALIGNN | Rejected because stress is unavailable | CUDA Graph energy/force inference |

`--cuda-graph auto` is the default. `--cuda-graph true` requires capture and
rejects an unsupported path, including CHGNet cell relaxation.
`--cuda-graph false` selects eager inference. Actual mode and cache statistics
are recorded in the results; requesting Graph mode alone is not evidence of
graph replay. CPU runs can use `--device cpu --cuda-graph false`.

ALIGNN can run a separate fixed-cell experiment:

```bash
python -m fastmd.discovery run --model alignn \
  --checkpoint checkpoints/alignn-model-directory \
  --device cuda --fixed-cell --output runs/alignn-fixed-cell
```

This optimizes positions only and does not implement the default Discovery
relaxation protocol. Check the ALIGNN checkpoint's force-scaling conventions
documented in the [main README](../../README.md#2-models-and-checkpoints).

## Resume and split work

Repeat the same run command with `--resume` after an interruption:

```bash
python -m fastmd.discovery run --model matris \
  --checkpoint checkpoints/MatRIS_10M_OAM.pth.tar \
  --device cuda --output runs/matris-discovery --resume
```

Keep the checkpoint, selection, optimizer settings and inference configuration
the same. Resume also checks source implementation and dependency versions, so
code or environment changes require a new output directory. Pass an explicit
local checkpoint to verify its weight hash; a default or named download is marked
as having unverified weight identity. Already recorded failed and unconverged
attempts are kept and skipped on resume. Use a new output directory to rerun them
or compare a different configuration.

For independent processes, assign each a zero-based shard and a separate output
directory. For example, the first of four shards is:

```bash
python -m fastmd.discovery run --model matris \
  --checkpoint checkpoints/MatRIS_10M_OAM.pth.tar \
  --device cuda --num-shards 4 --shard-index 0 \
  --output runs/matris-discovery/shard-0
```

Run indices 1, 2 and 3 with the same `--num-shards 4` and their own directories.
Assign GPUs through your scheduler or `CUDA_VISIBLE_DEVICES`. The CLI does not
allocate GPUs or submit scheduler jobs.

## Results and timing

Each output directory contains:

| File | Purpose |
| --- | --- |
| `run_metadata.json` | Input selection and run configuration for reproduction and resume |
| `results.jsonl` | Per-material final energies, convergence, steps, timing, inference statistics and failures |
| `relaxed.db` | Available terminal geometries as ASE database records |
| `summary.json` | Aggregate completion, convergence and performance information |

`summary.json` is updated after each material, so it can be inspected while a
long run is in progress. `counter_delta` contains cumulative backend-counter
differences; the cache snapshots are local to a graph runner and may reset when
the structure changes. Do not subtract cache snapshots to estimate total captures.

Unconverged finite geometries are useful results and are identified separately
from inference failures. Reaching the step limit is not equivalent to satisfying
the convergence criterion. `fmax` is checked against the forces supplied by
FrechetCellFilter, including its scaled cell degrees of freedom. The reported
atomic maximum force is therefore not sufficient on its own to establish cell
convergence.

The command exits nonzero when a material fails or setup is invalid. A completed
run may still contain unconverged structures, so inspect the convergence counts
even if the command exits successfully.

Per-structure relaxation timing includes the initial evaluation, graph
preparation/capture, optimizer steps, and final evaluation. Model loading is
outside that interval. ASE runs the optimizer and cell filter on the CPU; the
accelerated work is model inference, with the backend's existing neighbor and
kernel optimizations. **The complete relaxation step is not captured in a CUDA
Graph.** Changing neighbor capacity or periodic-image requirements may cause
recapture, and different structures may need different graphs.

For a fair comparison, use the same manifest, checkpoint, precision, optimizer,
cell settings, stopping criteria and hardware in every mode. Compare terminal
energy, atomic forces, stress and geometry on identical inputs before measuring
the full sample. Report failed and unconverged counts, total time per attempted
structure and optimizer step counts as well as milliseconds per step. Early
convergence changes the number of evaluations, and short runs can be dominated
by graph preparation. Measure eager and Graph in separate output directories;
do not treat smoke-test timing as a representative speedup.

## What this measures

This is a reproducible sample of the Discovery **relaxation workload**. It
produces final structures and raw model energies. It does not apply model-specific
energy corrections, compute formation energies or convex-hull distances, or
calculate Discovery F1. A raw total energy divided by atom count is not a
formation energy. Models can also use different training-energy conventions,
so a single unconditional correction is not appropriate for every checkpoint.

Official Discovery classification compares predicted formation energies against
a fixed Materials Project **DFT** convex hull. Geometry-optimization metrics
additionally compare ML-relaxed and DFT-relaxed structures. Those evaluations
require their respective reference data and protocols. Results on this selected
10% must be labelled as subset results, not as the full leaderboard score.

Sources:

- [Official data registry](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/data-files.yml)
- [Discovery default population and task description](https://github.com/janosh/matbench-discovery/blob/main/site/src/routes/tasks/discovery/+page.svelte)
- [Shared relaxation protocol](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/discovery.py)
- [MACE-MPA-0 model settings](https://github.com/janosh/matbench-discovery/blob/main/models/mace/mace-mpa-0.yml)
- [Fixed DFT convex-hull definition](https://github.com/janosh/matbench-discovery/blob/main/site/src/routes/tasks/discovery/hull-construction-note.md)
