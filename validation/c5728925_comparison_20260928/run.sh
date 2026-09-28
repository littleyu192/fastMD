#!/usr/bin/env bash
set -euo pipefail
cd /share/home/husiyu/software/NEP/MatRIS_BL/fastMD
out="$PWD/validation/c5728925_comparison_20260928"
ensemble=${1:-npt}
exec > >(tee -a "$out/${ensemble}_run.log") 2>&1
trap 'rc=$?; printf "%s exit=%s\n" "$(date -Is)" "$rc" >> "$out/${ensemble}_exit.txt"' EXIT
date -Is
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active" ]]; then
    printf 'GPU has active compute processes, benchmark not started: %s\n' "$active"
    exit 2
fi
source_root=$(cat "$out/source_path.txt")
export PYTHONPATH="$source_root/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
/share/home/husiyu/anaconda3/envs/matris-opt/bin/python "$out/benchmark.py" \
    --source "$source_root" --ensemble "$ensemble" --steps 50 --repeats 3 \
    --checkpoint /share/home/husiyu/software/NEP/MatRIS_BL/checkpoint/MatRIS_10M_OAM.pth.tar \
    --output "$out/${ensemble}_results.json"
