#!/usr/bin/env bash
set -euo pipefail
cd /share/home/husiyu/software/NEP/MatRIS_BL/fastMD
out=validation/matris_npt_20260928
exec > >(tee -a "$out/gpu_run.log") 2>&1
trap 'rc=$?; printf "%s exit=%s\n" "$(date -Is)" "$rc" >> "$out/gpu_exit.txt"' EXIT
date -Is
hostname
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total --format=csv
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active" ]]; then
    printf 'GPU still has compute processes; refusing to overlap: %s\n' "$active"
    exit 2
fi
printf 'GPU_IDLE_CONFIRMED %s\n' "$(date -Is)"
cat > matris_npt.md <<EOF
# MatRIS + ASE NPT GPU 验证

状态：**正在运行 GPU 回归测试**。

原任务步骤已退出，$(date -Is) 确认所分配 GPU 无计算进程后启动。
当前 job=$SLURM_JOB_ID，step=$SLURM_STEP_ID，节点=$(hostname)。
详细进度见 \`validation/matris_npt_20260928/gpu_run.log\`。
回归通过后将继续运行 NPT 正确性与性能对照，并在此保存最终结果。
EOF
export PYTHONPATH="$PWD/src:$PWD/.venv/npt-test-deps"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export FASTMD_MATRIS_CHECKPOINT=/share/home/husiyu/software/NEP/MatRIS_BL/checkpoint/MatRIS_10M_OAM.pth.tar
py=/share/home/husiyu/anaconda3/envs/matris-opt/bin/python
git diff > "$out/source.diff"
"$py" -c 'import torch; assert torch.cuda.is_available(); print("CUDA", torch.__version__, torch.cuda.get_device_name())'
"$py" -m pytest -q -rs tests/test_calculator.py tests/test_matris_npt.py tests/test_matris_config.py \
    'tests/test_models.py::test_cuda_replay_matches_eager[matris]' \
    'tests/test_models.py::test_real_model_cpu[matris]' \
    'tests/test_models.py::test_real_model_magmoms_and_stress[matris]' \
    'tests/test_models.py::test_sink_padding_preserves_predictions_on_cpu[matris]' \
    --junitxml="$out/pytest.xml" 2>&1 | tee "$out/pytest.log"
"$py" examples/validate_matris_npt.py --checkpoint "$FASTMD_MATRIS_CHECKPOINT" \
    --output "$out/results.json" --report matris_npt.md --steps 50 --repeats 3 \
    2>&1 | tee "$out/benchmark.log"
