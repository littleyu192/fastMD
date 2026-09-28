"""Render the benchmark's recorded measurements without estimating missing rows."""
import json
from pathlib import Path

out=Path(__file__).resolve().parent
repo=out.parents[1]
reports=[]
for path in sorted(out.glob('*_results.json')):
    if path.name.startswith('setup_attempt_'):
        continue
    reports.append(json.loads(path.read_text()))
lines=['# MatRIS c5728925 MD benchmark', '',
       'Exact revision: `c5728925f0fe13503ad5aa081d2e4db3733040d5`.', '',
       'The revision was exported separately; the current uncommitted NPT implementation was not changed.',
       'All 132 exported package source/data files were checked against their Git blob hashes.', '',
       'Si diamond, a=5.43 Å, cubic repeats 2×2×2 and 3×3×3 (64/216 atoms).',
       'Initial position noise: seed 2026, σ=0.01 Å; Maxwell-Boltzmann velocities: seed 42, 300 K, stationary COM.',
       'MatRIS 10M OAM; default topology fusions; activation checkpointing off; TF32 off; two CPU threads.',
       'NPT: ASE NPTBerendsen, dt=0.5 fs, 300 K, 1 GPa, taut=100 fs, taup=1000 fs, bulk modulus=100 GPa.',
       'NVE, if present: ASE VelocityVerlet, fixed cell, dt=0.5 fs, same initial structure and velocities.', '',
       'Each variant uses a 50-step warmup trajectory, then three independent 50-step timed trajectories from the same initial state.',
       'The timer includes initial evaluation, ASE integration, neighbor construction, transfers, and all recaptures required by this revision.',
       'Graph buffers are not cleared manually between timed trajectories; ASE results are reset at the start of each trajectory.',
       'There is no trajectory/log output in the timing region. Kernel disk caches were not cleared; warmup is not a clean-install cold-start measurement.',
       'The timing-only Calculator subclass counts captures using weak references, without extending old graph lifetimes or changing inference.',
       'No `compute_stress=True` override or variable-cell reuse was backported: this revision does not support those NPT-adapter options.', '']
reference=json.loads((repo/'validation/matris_npt_20260928/results.json').read_text())
for report in reports:
    env=report['environment']
    lines += [f"## Run: {report['started']}", '',
              f"Status: **{report['status']}**. Job/step `{env['job']}.{env['step']}`, node `{env['node']}`.",
              f"GPU: {env['gpu']}; identity/driver: `{env['gpu_info']}`.",
              f"Python {env['python']}; PyTorch {env['torch']}; CUDA {env['cuda']}; ASE {env['ase']}.",
              f"Checkpoint SHA256: `{report['checkpoint_sha256']}`.", '']
    for ensemble,sizes in report['results'].items():
        lines += [f'### {ensemble.upper()}', '',
                  '| Atoms | Mode | Warmup ms/step | Median ms/step | Min–max | Model calls per 50 steps | Captures per 50 steps |',
                  '|---:|---|---:|---:|---:|---|---|']
        for n,modes in sizes.items():
            for mode,row in modes.items():
                trials=row['trials']; values=[t['ms_per_step'] for t in trials]
                lines.append(f"| {n} | {mode} | {row['warmup']['ms_per_step']:.3f} | {row['median_ms_per_step']:.3f} | {min(values):.3f}–{max(values):.3f} | {[t['model_calls'] for t in trials]} | {[t['captures_during_run'] for t in trials]} |")
        if ensemble=='npt':
            lines += ['', 'Comparison with the existing NPT-adapted workspace report (same GPU and protocol, measured earlier, not rerun here):', '',
                      '| Atoms | Mode | c5728925 ms/step | NPT-adapted ms/step | Old/new time ratio |',
                      '|---:|---|---:|---:|---:|']
            for n,modes in sizes.items():
                for mode,row in modes.items():
                    current=reference['results'][n][mode]['median_ms_per_step']
                    lines.append(f"| {n} | {mode} | {row['median_ms_per_step']:.3f} | {current:.3f} | {row['median_ms_per_step']/current:.3f}× |")
        lines += ['', 'Numerical check: a separate 20-step graph trajectory is compared to the corresponding eager trajectory.',
                  'Absolute tolerances: positions 2e-4 Å, cell 2e-5 Å, momenta 2e-3 ASE units; rtol=1e-5.', '']
        for n,modes in sizes.items():
            row=modes.get('graph_default',{})
            if 'trajectory_errors' in row:
                lines.append(f"- {n} atoms max absolute differences: `{row['trajectory_errors']}`.")
    if report.get('error'):
        lines += ['', '```text', report['error'], '```']
lines += ['', '## Reproduction and raw records', '',
          'Raw results and counters: `validation/c5728925_comparison_20260928/*_results.json`.',
          'Logs: `*_run.log`. Source provenance: `source_verification.json` and `commit.txt`.',
          'A preliminary setup attempt was cancelled before collecting a full benchmark to remove a strong-reference effect from the counter; its log is retained as `setup_attempt_*` and is excluded from all reported results.', '',
          '```bash',
          'srun --jobid=102928 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \\',
          '  bash validation/c5728925_comparison_20260928/run.sh npt',
          '```', '',
          'The launcher requires an idle allocated GPU. The archived source path is recorded in `source_path.txt`;',
          'if removed, regenerate it with `git archive c5728925f0fe13503ad5aa081d2e4db3733040d5` and update that path.']
(repo/'matris_c5728925_benchmark.md').write_text('\n'.join(lines)+'\n')
print(repo/'matris_c5728925_benchmark.md')
