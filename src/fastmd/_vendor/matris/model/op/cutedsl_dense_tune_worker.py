"""Subprocess worker for isolated CUPTI dense GEMM tactic profiling."""

from __future__ import annotations

import argparse
import json
import os
import warnings

import torch

from . import cutedsl_dense_linear as dense_linear


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()

    os.environ["_MATRIS_CUTEDSL_TUNE_WORKER"] = "1"
    os.environ["MATRIS_CUTEDSL_TUNE_TIMER"] = "cupti_inline"

    torch.cuda.set_device(args.device)
    dtype = _dtype_from_name(args.dtype)
    device = torch.device("cuda", args.device)
    x = torch.empty((args.m, args.k), device=device, dtype=dtype)
    weight = torch.empty((args.n, args.k), device=device, dtype=dtype)
    bias = torch.empty((args.n,), device=device, dtype=dtype) if args.bias else None
    out = torch.empty((args.m, args.n), device=device, dtype=dtype)

    valid = dense_linear._valid_tactics(x, weight, out)
    if not valid:
        raise RuntimeError(f"no valid CuTeDSL dense GEMM tactics for {(args.m, args.n, args.k)}")
    scores = []
    for tactic in valid:
        elapsed_ms = dense_linear._profile_tactic(x, weight, tactic, bias)
        scores.append((elapsed_ms, tactic))
    best_elapsed_ms, best = min(scores, key=lambda item: item[0])
    payload = {
        "best": best,
        "best_ms": best_elapsed_ms,
        "scores": [{"ms": ms, "tactic": tactic} for ms, tactic in scores],
    }
    print(f"{dense_linear._TUNE_RESULT_PREFIX}{json.dumps(payload)}", flush=True)


if __name__ == "__main__":
    main()
