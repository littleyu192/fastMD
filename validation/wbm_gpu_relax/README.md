> Historical result: this 2026-10-23 report describes the superseded functional relaxation implementation. On 2026-10-26 it was replaced by the `MD` repository’s `StructOptimizer` path; use the current FastMD README for the supported API.
>
> These records predate the adapter-kit validation in job `110201` and do not report that job's results or validate subsequent source changes.

# GPU-resident relaxation validation — 2026-10-23

`relax` and `fastmd-relax` now dispatch eligible CUDA FIRE tasks through the existing MD whole-step engine, using a single slot of `MultiSystemRelaxation`. FIRE state, Frechet cell variables and convergence stay on the GPU; candidate refresh and capacity-overflow recovery reuse the existing engine. Initial energy is retained on the device and collected with the final scalars.

## Input and environment

- Structure: `wbm-1-12`, Ag2CdI4, 7 atoms, unchanged official WBM initial geometry.
- Source and license: [fixture provenance](../../tests/data/README.md); [official Figshare file](https://ndownloader.figshare.com/files/48169597), CC BY 4.0.
- Checkpoint: MACE-MP-0 medium, SHA256 `01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845`.
- GPU: NVIDIA H100 80GB HBM3; PyTorch 2.8.0+cu128; ASE 3.26.0; torch.float32 weights.
- Profile: `topology`; matmul precision `highest`; FIRE defaults; fmax 0.05 eV/Å; step limit 500; 5 steps per launch.
- Both paths borrow the same loaded adapter and start from independent copies. No model download was required.

## Final-code measurements

| Mode | ASE/GPU steps | ASE seconds | GPU seconds | Final energy difference (eV) | Position difference (Å) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_cell | 7/7 | 2.5739 | 6.1894 | 1.19209e-07 | 3.83297e-08 |
| variable_cell | 6/6 | 0.5106 | 2.5900 | 1.54972e-06 | 9.38788e-08 |

All four relaxations converged. The five-step strict comparisons also passed (atol=1e-4, rtol=1e-5), comparing energies, coordinates, cells, forces, stress, step counts and convergence. These compare execution paths, not DFT accuracy.

Times include initial inference, setup, self-check and graph capture, but exclude model loading. This seven-atom task finishes in only 6–7 steps: capture dominates, so total GPU-path latency is higher. These single-run timings are not a throughput or speedup claim; warmup/cache order differs between runs.

## Transfers and ownership

| Mode | ASE calculator calls | GPU-path ASE calls | Captured windows | Terminal collections |
| --- | ---: | ---: | ---: | ---: |
| fixed_cell | 8 | 0 | 2 | 1 |
| variable_cell | 7 | 0 | 2 | 1 |

The window loop reads convergence/overflow flags instead of uploading positions and reading forces through ASE every step. Startup negotiation, self-check and candidate construction still involve host work; terminal collection copies several arrays once. This is not a measurement claiming zero transfers or only two transfers for the entire task. Both runs needed no fallback window or recapture.

Assertions verify input arrays, cell and calculator are unchanged; the adapter is shared; and terminal properties come from an independent `SinglePointCalculator`. The preflight checks adapter capabilities without repeating negotiation or its model-weight fingerprint readback.

## Regression checks

- 46 CPU tests passed: existing relaxation/CLI/StructOptimizer behavior, routing and configuration preservation, initial-energy rollback/slot reuse, and GPU FIRE mathematics against ASE.
- 3 CUDA tests (8 fixed/Frechet scenarios) passed on H100: multiple windows, initial convergence, zero steps, exact step limits, and tiny-skin recovery. The recovery test forces invalid windows and verifies the restored initial energy and final geometry against ASE.
- CLI end-to-end check passed: `--execution auto` selected `whole_step`, converged in 6 steps, and wrote [relaxed.extxyz](relaxed.extxyz) plus [cli.json](cli.json). Re-reading the structure preserved finite forces/stress and the reported energy.
- Syntax compilation and `git diff --check` passed.

## Historical commands

The following commands were used with the historical FastMD checkout and its
CUDA/model dependencies. `examples/relax_wbm.py` and the
`test_gpu_relaxation_cuda` module were removed with the superseded implementation;
these commands cannot be run from the current checkout. They are retained as
provenance, not as current reproduction instructions.

```sh
PYTHONPATH=. python examples/relax_wbm.py \
  --model-path /path/to/mace-medium.model \
  --steps 500 --fmax 0.05 --steps-per-launch 5 --strict-steps 5 \
  --output validation/wbm_gpu_relax/comparison.json

PYTHONPATH=.:tests python -m unittest test_gpu_relaxation_cuda -v
```

The actual environment used `../records2610/environments/mace/bin/python` and checkpoint `../records2610/checkpoints/mace-medium.model` inside an existing H100 allocation. No existing campaign was resumed or modified.

## Supported scope

Whole-step relaxation currently requires a capture-safe edges-only adapter declaring union batching and per-edge shifts (MACE/SevenNet), full PBC, FIRE, no ASE constraints, and the default Frechet filter for variable cells. Other models/options use ASE under `auto`; explicit `whole_step` rejects them. Per-step ASE logs/trajectories use the ASE path. `StructOptimizer` retains ASE scheduling. This change does not make MatRIS/CHGNet relaxation GPU-resident.

Raw results: [comparison.json](comparison.json). The original regression logs
`cpu_tests.log` and `cuda_tests.log` are not included in this repository; the
results above are retained historical summaries.
