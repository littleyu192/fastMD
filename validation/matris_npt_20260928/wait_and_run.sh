#!/usr/bin/env bash
set -euo pipefail
cd /share/home/husiyu/software/NEP/MatRIS_BL/fastMD
out=validation/matris_npt_20260928
exec > >(tee -a "$out/wait.log") 2>&1
exec 9>"$out/watcher.lock"
flock -n 9 || { echo 'A validation watcher is already running'; exit 1; }
while true; do
    state=$(squeue -h -j 102928 -o '%T')
    [[ "$state" == RUNNING ]] || { echo "Allocation is not RUNNING: $state"; exit 1; }
    steps=$(squeue -h --steps -j 102928 -o '%i')
    if printf '%s\n' "$steps" | rg -q '^102928\.[0-9]+$'; then
        printf '%s waiting for active task steps: %s\n' "$(date -Is)" "$steps"
        sleep 30
        continue
    fi
    printf '%s Original task steps have exited; starting validation.\n' "$(date -Is)"
    tail -5 /share/home/husiyu/software/NEP/MatRIS_BL/Kaifang_0921/results/rec2_n128_e32_t32/runner.log
    break
done
srun --jobid=102928 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --gres=gpu:1 --time=04:00:00 bash "$out/gpu_run.sh"
