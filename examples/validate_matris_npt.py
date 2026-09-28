"""Reproducible GPU validation and end-to-end ASE NPT timing for MatRIS."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import ase
from ase import units
from ase.build import bulk
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list
import numpy as np
import torch

from fastmd import CUDAGraphConfig, FastMDCalculator


VARIANTS = {
    "eager_generic": (False, False),
    "eager_default": (False, True),
    "graph_generic": (True, False),
    "graph_default": (True, True),
}
PROPERTIES = ("energy", "forces", "stress")
TOLERANCES = {"energy": (3e-3, 1e-5), "forces": (3e-3, 1e-3), "stress": (3e-5, 1e-3)}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def initial_atoms(repeat):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((repeat,) * 3)
    atoms.positions += np.random.default_rng(2026).normal(0, .01, atoms.positions.shape)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))
    Stationary(atoms)
    return atoms


def dynamics(atoms):
    return NPTBerendsen(atoms, timestep=.5 * units.fs, temperature_K=300,
                        pressure_au=units.GPa, taut=100 * units.fs,
                        taup=1000 * units.fs, compressibility_au=1 / (100 * units.GPa))


def make_calculator(checkpoint, variant):
    graph, fused = VARIANTS[variant]
    return FastMDCalculator("matris", checkpoint=checkpoint, device="cuda",
                            cuda_graph=CUDAGraphConfig(enabled=graph, enable_fusions=fused),
                            model_kwargs={"compute_stress": True})


def snapshot(calc):
    stats = calc.stats()
    return {key: stats.get(key) for key in
            ("mode", "calls", "graph_calls", "eager_calls", "invalidations", "cache")}


def capture_count(calc):
    return calc.stats().get("cache", {}).get("captures", 0)


def predictions(calc, frames):
    values = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for atoms in frames:
        calc.calculate(atoms, PROPERTIES)
        values.append({key: np.array(calc.results[key], copy=True) for key in PROPERTIES})
    torch.cuda.synchronize()
    return values, time.perf_counter() - start


def errors(actual, reference):
    maximum = {key: 0.0 for key in PROPERTIES}
    for index, (a, b) in enumerate(zip(actual, reference, strict=True)):
        for key in PROPERTIES:
            atol, rtol = TOLERANCES[key]
            np.testing.assert_allclose(a[key], b[key], atol=atol, rtol=rtol,
                                       err_msg=f"frame {index}, {key}")
            maximum[key] = max(maximum[key], float(np.max(np.abs(a[key] - b[key]))))
    return maximum


def trajectory(calc, initial, steps, *, save=False):
    atoms = initial.copy()
    atoms.calc = calc
    # Include the initial E/F/S evaluation in the timed end-to-end run.
    calc.reset()
    saved = []
    start_captures = capture_count(calc)
    start_calls = calc.stats()["calls"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    with dynamics(atoms) as dyn:
        if save:
            for _ in range(steps):
                dyn.run(1)
                saved.append({"positions": atoms.positions.copy(), "cell": atoms.cell.array.copy(),
                              "momenta": atoms.get_momenta().copy()})
        else:
            dyn.run(steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert np.isfinite(atoms.positions).all() and np.isfinite(atoms.cell.array).all()
    assert np.max(np.abs(atoms.cell.array - initial.cell.array)) > 1e-8
    assert calc.stats()["invalidations"] == 0
    return {"seconds": elapsed, "ms_per_step": elapsed * 1000 / steps,
            "captures_during_run": capture_count(calc) - start_captures,
            "model_calls": calc.stats()["calls"] - start_calls,
            "final_volume": atoms.get_volume(), "final_temperature": atoms.get_temperature(),
            "cell_change_max_A": float(np.max(np.abs(atoms.cell.array - initial.cell.array)))}, saved


def trajectory_error(actual, expected):
    limits = {"positions": 2e-4, "cell": 2e-5, "momenta": 2e-3}
    out = {}
    for key, atol in limits.items():
        a = np.stack([row[key] for row in actual])
        b = np.stack([row[key] for row in expected])
        np.testing.assert_allclose(a, b, atol=atol, rtol=1e-5, err_msg=key)
        out[key] = float(np.max(np.abs(a - b)))
    return out


def stress_derivative(calc, initial):
    atoms = initial.copy()
    # The OAM checkpoint has a hard 4.5 A three-body cutoff, close to a Si
    # neighbor shell. A derivative check must stay on one smooth topology branch.
    atoms.set_cell(atoms.cell * 1.03, scale_atoms=True)
    converter = calc.backend.model.graph_converter
    cutoffs = (converter.atom_graph_cutoff, converter.line_graph_cutoff)
    distances = neighbor_list('d', atoms, max(cutoffs) + .1)
    margins = [float(np.min(np.abs(distances - cutoff))) for cutoff in cutoffs]
    h = 5e-4
    assert min(margins) > 2 * h * max(cutoffs), (cutoffs, margins)
    atoms.calc = calc
    reference = atoms.get_stress()
    cell = atoms.cell.array.copy()
    fractional = atoms.get_scaled_positions(wrap=False)
    volume = atoms.get_volume()
    finite = []
    for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = 1. if i == j else .5
        energies = []
        for sign in (1, -1):
            atoms.set_cell(cell @ (np.eye(3) + sign * h * strain))
            atoms.set_scaled_positions(fractional)
            energies.append(atoms.get_potential_energy())
        finite.append((energies[0] - energies[1]) / (2 * h * volume))
    np.testing.assert_allclose(finite, reference, atol=3e-4, rtol=.04)
    return {"analytic": reference.tolist(), "finite_difference": finite,
            "strain_step": h, "cell_scale": 1.03, "cutoffs_A": list(cutoffs),
            "cutoff_margins_A": margins,
            "max_error_eV_A3": float(np.max(np.abs(finite - reference)))}


def write_report(report, path):
    env = report["environment"]
    status = {"PASS_WITH_MODEL_LIMITATION": "Graph 一致性验证通过；存在模型截断限制",
              "PASS": "通过", "RUNNING": "运行中", "FAIL": "未通过"}.get(report['status'], report['status'])
    lines = ["# MatRIS + ASE NPT GPU 验证", "", f"- 状态：**{status}**",
             f"- 开始：{report['started']}；最近更新：{now()}",
             f"- Slurm job/step：{env['job']} / {env['step']}；节点：{env['host']}",
             f"- GPU：{env['gpu']}；PyTorch {env['torch']}；CUDA {env['cuda']}；ASE {env['ase']}",
             f"- Python：{env['python']}；线程数：{env['threads']}",
             f"- GPU UUID / driver：`{env['gpu_identity']}`；TF32：关闭。",
             f"- 权重：`{report['checkpoint']}`",
             f"- 权重 SHA256：`{report['checkpoint_sha256']}`",
             f"- Git HEAD：`{env['git_head']}`；测试工作区包含未提交的 NPT 适配。", "",
             "## 方法", "",
             "Si diamond，a=5.43 Å，随机扰动 0.01 Å；初始速度固定随机种子。",
             "ASE NPTBerendsen：300 K、1 GPa、dt=0.5 fs、taut=100 fs、taup=1000 fs、体积模量 100 GPa。",
             "Berendsen 用于压力平衡，本报告不验证严格 NPT 涨落分布。",
             "所有变体均在同一 GPU 上顺序运行，使用 compute_stress=True；不允许 Graph 静默回退。",
             "计时包含 ASE 积分、变胞邻居重建、CPU/GPU 传输与结果验证；没有轨迹/日志写盘。",
             f"每变体先跑 {report['steps']} 步完整预热轨迹，再从相同初态重复 {report['repeats']} 次，每次 {report['steps']} 步；表中采用中位数。",
             "generic 关闭模型融合；default 使用默认 topology 融合。首次完整轨迹单列，不计入稳态加速比。", "",
             "首次轨迹指该变体在本轮进程内的第一次完整运行；没有清理 Triton/Warp 磁盘缓存，且回归测试先于基准执行，因此它不代表全新环境的冷启动总耗时。", "",
             "## 正确性", "",
             "相同输入帧比较覆盖压缩、膨胀、剪切、周期边界跨越；参考是 GPU eager_generic。",
             "容差（atol, rtol）：E=(3e-3 eV, 1e-5)，F=(3e-3 eV/Å, 1e-3)，stress=(3e-5 eV/Å³, 1e-3)。",
             "独立运行 20 步 NPT 后逐帧比较位置、晶胞和动量；绝对容差分别 2e-4 Å、2e-5 Å、2e-3 ASE 动量单位，rtol=1e-5。", "",
             "| 原子数 | 变体 | 最大 ΔE (eV) | 最大 ΔF (eV/Å) | 最大 Δstress (eV/Å³) | 20 步最大位置偏差 (Å) |",
             "|---:|---|---:|---:|---:|---:|"]
    for size, variants in report["results"].items():
        for name, row in variants.items():
            err = row["frame_errors"]
            lines.append(f"| {size} | {name} | {err['energy']:.3g} | {err['forces']:.3g} | {err['stress']:.3g} | {row['trajectory_errors'].get('positions', 0):.3g} |")
    for size, variants in report["results"].items():
        derivative = variants.get("graph_default", {}).get("stress_derivative")
        if derivative:
            lines += ["", f"{size} 原子默认 CUDA Graph 的六分量应力有限差分通过：晶胞先均匀放大 1.03 倍，确认所有邻居距离远离截断后取 h=5e-4，atol=3e-4 eV/Å³、rtol=0.04；最大误差 {derivative['max_error_eV_A3']:.3g} eV/Å³。",
                      f"原子/三体 cutoff 分别为 {derivative['cutoffs_A']} Å，最近距离余量分别为 {derivative['cutoff_margins_A']} Å。"]
    lines += ["", "## 性能", "", "| 原子数 | 变体 | 加载 (s) | 首次 NPT 轨迹 (s) | 稳态 ms/step（中位数） | 最小–最大 | 相对默认 eager | 计时内捕获数 |",
              "|---:|---|---:|---:|---:|---:|---:|---|"]
    for size, variants in report["results"].items():
        reference = variants.get("eager_default", {}).get("median_ms_per_step")
        for name, row in variants.items():
            samples = [trial["ms_per_step"] for trial in row["trials"]]
            ratio = f"{reference / row['median_ms_per_step']:.3f}×" if reference else "—"
            lines.append(f"| {size} | {name} | {row['load_seconds']:.2f} | {row['cold_run']['seconds']:.2f} | {row['median_ms_per_step']:.3f} | {min(samples):.3f}–{max(samples):.3f} | {ratio} | {[t['captures_during_run'] for t in row['trials']]} |")
    for size, variants in report["results"].items():
        if "graph_default" in variants and "eager_default" in variants:
            ratio = variants["eager_default"]["median_ms_per_step"] / variants["graph_default"]["median_ms_per_step"]
            lines += ["", f"{size} 原子：默认 CUDA Graph / 默认 eager 的端到端加速比为 **{ratio:.3f}×**。", ""]
        if "graph_generic" in variants and "eager_generic" in variants:
            ratio = variants["eager_generic"]["median_ms_per_step"] / variants["graph_generic"]["median_ms_per_step"]
            lines += [f"{size} 原子：关闭融合时，CUDA Graph / eager 的加速比为 **{ratio:.3f}×**。", ""]
    lines += ["", "## 回归测试与原始记录", "",
              "GPU 回归测试结果见 `validation/matris_npt_20260928/pytest.log`。",
              "原始数值、每次计时、Graph 状态及环境见 `validation/matris_npt_20260928/results.json`。",
              "GPU 空闲检查和启动时间见同目录 `gpu_run.log`；原训练步骤退出后才启动验证。",
              "本结果只覆盖指定权重、体系、步数和硬件；短程数值一致不代表长程轨迹逐点一致。"]
    if report.get("handoff"):
        handoff = report["handoff"]
        lines += ["", f"原任务 {handoff['previous_step']} 于 {handoff['previous_exit_time']} 正常退出（exit={handoff['previous_exit_code']}）；{handoff['gpu_idle_verified_at']} 确认 GPU 无计算进程且显存为 0 MiB 后，开始回归测试。"]
    if report.get("pytest"):
        checks = report['pytest']
        lines += ["", f"实际回归结果：共 {checks['tests']} 项，失败 {checks['failures']}，错误 {checks['errors']}，跳过 {checks['skipped']}；耗时 {checks['time']} 秒。"]
    if report.get("resumed_from"):
        lines += ["", f"本次分阶段运行：此前 step={report['resumed_from']['step']} 的已完成计时被保留；当前 step={env['step']} 补完剩余项目。"]
    if report.get("cutoff_diagnosis"):
        lines += ["", "## 已确认的模型限制：跨三体截断的应力差分", "",
                  "初始 64 原子 Si 结构在 h=5e-4 应变差分中未通过应力检查：最大误差约 0.01183 eV/Å³。",
                  "独立诊断表明，关闭 Graph、关闭融合的 eager 同样出现该偏差；差分时无向原子边数保持 1472，但三体项数量在 26498–28060 之间变化。",
                  "缩小步长至 2e-4、1e-4、5e-5 仍可跨越三体硬截断，并未消除该问题。",
                  "因此该结果不能归因于本次 Graph 复用改动；它是此权重/构型在截断附近的非光滑性，不能用跨拓扑的中心差分检验单点导数。",
                  "本报告保留初始失败，不修改势函数、不放宽误差阈值。远离截断的导数检查另列；NPT 性能和轨迹对照仍使用原始未放大结构。",
                  "Graph/eager 短程一致不等于证明长期能量守恒或严格 NPT 采样正确。",
                  "详见 `validation/matris_npt_20260928/initial_results.json`、`initial_benchmark.log` 和 `stress_diagnosis.json`。"]
    lines += ["", "## 复现", "", "```bash",
              "export FASTMD_MATRIS_CHECKPOINT=/share/home/husiyu/software/NEP/MatRIS_BL/checkpoint/MatRIS_10M_OAM.pth.tar",
              "# 在已有 GPU 分配中，确认没有其他计算任务后执行：",
              "bash validation/matris_npt_20260928/gpu_run.sh", "```"]
    if "error" in report:
        lines += ["", "## 未完成原因", "", "```text", report["error"], "```"]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("matris_npt.md"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Keep completed variants from the output JSON")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA GPU is required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(args.checkpoint).resolve()
    report = {"status": "RUNNING", "started": now(), "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "steps": args.steps, "repeats": args.repeats, "results": {},
              "environment": {"host": platform.node(), "job": os.getenv("SLURM_JOB_ID"),
                  "step": os.getenv("SLURM_STEP_ID"), "gpu": torch.cuda.get_device_name(),
                  "python": platform.python_version(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "ase": ase.__version__, "threads": torch.get_num_threads(),
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "gpu_identity": subprocess.check_output(
                      ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("warp-lang", "nvalchemi-toolkit-ops", "triton")}}}
    pytest_xml = args.output.parent / "pytest.xml"
    if pytest_xml.exists():
        suite = ET.parse(pytest_xml).getroot().find("testsuite")
        if suite is not None:
            report["pytest"] = {key: suite.get(key) for key in ("tests", "failures", "errors", "skipped", "time")}
    if args.resume:
        previous = json.loads(args.output.read_text())
        for key in ("checkpoint_sha256", "steps", "repeats"):
            assert previous[key] == report[key], key
        report["results"] = previous["results"]
        report["resumed_from"] = {"step": previous["environment"]["step"], "started": previous["started"]}
    diagnosis = args.output.parent / "stress_diagnosis.json"
    if diagnosis.exists():
        report["cutoff_diagnosis"] = json.loads(diagnosis.read_text())

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        write_report(report, args.report)

    save()
    try:
        for repeat in (2, 3):
            initial = initial_atoms(repeat)
            size = str(len(initial))
            report["results"].setdefault(size, {})
            frames = []
            for scale, shear in ((1., 0), (.999, 0), (1.001, 0), (.97, .02), (1.03, -.03), (1., .04)):
                frame = initial.copy()
                deformation = np.eye(3) * scale
                deformation[0, 1] = shear
                frame.set_cell(initial.cell.array @ deformation, scale_atoms=True)
                frame.positions[0] += frame.cell.array[0]
                frames.append(frame)
            reference_predictions = reference_trajectory = None
            if report["results"][size]:
                calc = make_calculator(checkpoint, "eager_generic")
                reference_predictions, _ = predictions(calc, frames)
                _, reference_trajectory = trajectory(calc, initial, 20, save=True)
                calc.clear_cache()
                del calc
                gc.collect()
                torch.cuda.empty_cache()
            for variant in VARIANTS:
                if variant in report["results"][size]:
                    print(f"{now()} KEEP size={size} variant={variant}", flush=True)
                    continue
                print(f"{now()} START size={size} variant={variant}", flush=True)
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                calc = make_calculator(checkpoint, variant)
                load = time.perf_counter() - start
                cold, _ = trajectory(calc, initial, args.steps)
                trial_rows = []
                for _ in range(args.repeats):
                    row, _ = trajectory(calc, initial, args.steps)
                    trial_rows.append(row)
                _, saved = trajectory(calc, initial, 20, save=True)
                values, frame_seconds = predictions(calc, frames)
                if reference_predictions is None:
                    reference_predictions, reference_trajectory = values, saved
                frame_errors = errors(values, reference_predictions)
                traj_errors = trajectory_error(saved, reference_trajectory)
                derivative = stress_derivative(calc, initial) if variant == "graph_default" else None
                stats = snapshot(calc)
                assert stats["invalidations"] == 0
                if VARIANTS[variant][0]:
                    assert stats["mode"] == "cuda_graph" and stats["eager_calls"] == 0
                    assert stats["cache"]["captures"] > 0
                row = {"load_seconds": load, "cold_run": cold, "trials": trial_rows,
                       "median_ms_per_step": statistics.median(r["ms_per_step"] for r in trial_rows),
                       "frame_errors": frame_errors, "trajectory_errors": traj_errors,
                       "frame_seconds": frame_seconds, "stats": stats,
                       "stress_derivative": derivative,
                       "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30}
                report["results"][size][variant] = row
                print(f"{now()} DONE size={size} variant={variant} ms/step={row['median_ms_per_step']:.3f} errors={frame_errors}", flush=True)
                save()
                calc.clear_cache()
                del calc
                gc.collect()
                torch.cuda.empty_cache()
        report["status"] = "PASS_WITH_MODEL_LIMITATION" if report.get("cutoff_diagnosis") else "PASS"
    except Exception:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["finished"] = now()
        save()


if __name__ == "__main__":
    main()
