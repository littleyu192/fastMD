#!/usr/bin/env bash
# Offline single-GPU validation; see --help and tools/ci/README.md.
set -euo pipefail
LANE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$LANE_DIR/_single_gpu_lane.py" mace "$@"
