"""Inspect installed model packages without loading or downloading a model.

    python -m fastmd.check [--family NAME ...] [--import] [--json]

Package discovery and version matches do not validate upstream source pins, model
compatibility or GPU kernel execution. ``--import`` additionally imports the
model packages and reports import failures. An explicitly required family must
have its package metadata and module; with ``--import`` its import must succeed.
MatRIS is bundled with FastMD and needs no separate package. No weights are loaded.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import platform
import sys

# Distribution, import module, adapter version attribute, installation reference.
# Keep these rows aligned with fastmd.models.MODELS. M3GNet has a separate API
# and is not a member of that registry.
FAMILIES = {
    "nequip": ("nequip", "nequip", None, ".[nequip]"),
    "esen": ("fairchem-core", "fairchem.core", None, "fairchem-core==1.10.0; local eSEN weights"),
    "mace": ("mace-torch", "mace", "_SHARD_PIN_VERSIONS", ".[mace]"),
    "mace_soyo": ("mace-soyo", "mace_soyo", "_SHARD_PIN_VERSIONS", "requirements-mace-soyo.txt"),
    "sevennet": ("sevenn", "sevenn", "SEVENN_VERSIONS", ".[sevennet]"),
    "orb": ("orb-models", "orb_models", "ORB_VERSIONS", ".[orb]"),
    "chgnet": ("chgnet", "chgnet", "CHGNET_VERSIONS", ".[chgnet]"),
    "tace": ("tace", "tace", "TACE_VERSIONS", "requirements-tace.txt"),
}
# NequIP's extra and the legacy eSEN loader specify these versions; neither
# adapter publishes the version-tuple attribute used by the other families.
_VALIDATED = {"nequip": ("0.14.0",), "esen": ("1.10.0",)}
KERNELS = ("cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu13",
           "cuequivariance-ops-torch-cu12", "openequivariance", "e3nn")
CORE = ("torch", "numpy", "ase", "pymatgen")
GPU_GRAPH = ("nvalchemi-toolkit-ops", "warp-lang", "triton")


def version(distribution: str) -> str | None:
    """Installed distribution metadata, not proof that its module imports."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def family_record(name: str, imports: bool = False) -> dict:
    from fastmd.models import MODELS

    distribution, module, attribute, how = FAMILIES[name]
    record = {"package": distribution, "module": module, "installed": version(distribution),
              "install": how, "imports_checked": imports, "source_pins_checked": False,
              "kernel_execution_checked": False}
    # Looking up a dotted module imports its parent. Discover only the top-level
    # package here; --import checks the complete module (including fairchem.core).
    discovery = module.split(".", 1)[0]
    record["discovery_module"] = discovery
    try:
        record["found"] = importlib.util.find_spec(discovery) is not None
    except (ImportError, ValueError, AttributeError) as error:
        record["found"] = False
        record["discovery_error"] = f"{type(error).__name__}: {error}"[:300]
    try:
        where = next(row.module for row in MODELS if row.name == name)
        record["validated"] = list(_VALIDATED[name] if attribute is None else
                                   getattr(importlib.import_module(where), attribute))
    except Exception as error:
        record["validated"] = []
        record["adapter_error"] = f"{type(error).__name__}: {error}"[:300]
    installed, validated = record["installed"], record["validated"]
    record["version_matches"] = installed in validated if installed is not None and validated else None
    if record["version_matches"] is False:
        record["note"] = "Version differs from the adapter's validated versions; model loading and source pins remain unchecked."
    if imports:
        record["imports"] = False
        if not record["found"]:
            record["import_error"] = f"module {discovery!r} was not found"
        else:
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                try:
                    importlib.import_module(module)
                    record["imports"] = True
                except Exception as error:
                    record["import_error"] = f"{type(error).__name__}: {error}"[:300]
            if output.getvalue() or errors.getvalue():
                record["import_output"] = {"stdout": output.getvalue(), "stderr": errors.getvalue()}
    return record


def gpu_graph_builder() -> dict:
    """Inspect import/capability flags; do not build a graph or execute a kernel."""
    record = {name: version(name) for name in GPU_GRAPH}
    record.update(available=False, execution_checked=False)
    try:
        import torch
        from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
        from fastmd.graph.gpu_graph_builder import op_available

        record.update(cuda_available=bool(torch.cuda.is_available()),
                      neighbor_list_import_available=bool(op_available),
                      fixed_builder_import_available=bool(_fixed_builder_available))
        record["available"] = all(record[key] for key in
                                  ("cuda_available", "neighbor_list_import_available", "fixed_builder_import_available"))
        if not record["available"]:
            record["reason"] = "CUDA device or graph-builder imports unavailable"
    except Exception as error:
        record["reason"] = f"{type(error).__name__}: {error}"[:300]
    return record


def report(imports: bool = False) -> dict:
    import fastmd
    from fastmd.models import DEFAULT, MODELS

    data = {"python": platform.python_version(),
            "fastmd": {"version": version("fastmd") or fastmd.__version__, "path": fastmd.__file__},
            "core": {name: version(name) for name in CORE},
            "kernels": {name: found for name in KERNELS if (found := version(name)) is not None},
            "source_pins_checked": False, "kernel_execution_checked": False}
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        data["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": cuda,
                         "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
                         if cuda else []}
    except Exception as error:
        data["torch"] = {"version": version("torch"), "cuda": None, "cuda_available": False, "gpus": [],
                         "import_error": f"{type(error).__name__}: {error}"[:300]}
    data["gpu_graph_builder"] = gpu_graph_builder()
    built_in = {"package": "fastmd", "module": next(row.module for row in MODELS if row.name == DEFAULT),
                "installed": version("fastmd"), "built_in": True,
                "found": True, "imports_checked": imports,
                "source_pins_checked": False, "kernel_execution_checked": False}
    if imports:
        try:
            importlib.import_module(built_in["module"])
            built_in["imports"] = True
        except Exception as error:
            built_in["imports"] = False
            built_in["import_error"] = f"{type(error).__name__}: {error}"[:300]
    data["families"] = {"matris": built_in, **{name: family_record(name, imports) for name in FAMILIES}}
    return data


def text(data: dict) -> str:
    torch = data["torch"]
    lines = [f"FastMD {data['fastmd']['version']} ({data['fastmd']['path']}), Python {data['python']}",
             f"torch {torch['version']}, CUDA {torch['cuda']}, GPUs: {', '.join(torch['gpus']) or 'none'}",
             "core: " + ", ".join(f"{name} {value or 'missing'}" for name, value in data["core"].items()),
             "GPU graph builder: " + ("capability flags available" if data["gpu_graph_builder"]["available"]
                                      else f"unavailable ({data['gpu_graph_builder']['reason']})")
             + "; execution not checked",
             "kernel libraries: " + (", ".join(f"{name} {value}" for name, value in data["kernels"].items()) or "none"),
             "model families:"]
    for name, record in data["families"].items():
        if record.get("built_in"):
            state = "built in; model loading not checked"
            if "imports" in record:
                state += "; imports" if record["imports"] else f"; IMPORT FAILED: {record['import_error']}"
        else:
            state = f"{record['package']} {record['installed'] or 'missing'}; module " + ("found" if record["found"] else "missing")
            state += f"; validated versions: {', '.join(record['validated']) or 'unknown'}"
            if "imports" in record:
                state += "; imports" if record["imports"] else f"; IMPORT FAILED: {record['import_error']}"
            if record.get("adapter_error"):
                state += f"; adapter inspection failed: {record['adapter_error']}"
            if record.get("note"):
                state += "; " + record["note"]
            if record["installed"] is None:
                state += f"; install: {record['install']}"
        lines.append(f"  {name:10s} {state}")
    lines.append("Source pins, model loading and kernel execution are not checked by this report.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fastmd.check", description=__doc__.split("\n\n")[0])
    parser.add_argument("--family", action="append", default=[], choices=["all", "matris", *FAMILIES],
                        help="require an installed family (repeatable); all requires every registry family")
    parser.add_argument("--import", dest="imports", action="store_true", help="also import each discovered model package")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    output, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        data = report(args.imports)
    if output.getvalue() or errors.getvalue():
        data["inspection_output"] = {"stdout": output.getvalue(), "stderr": errors.getvalue()}
    print(json.dumps(data, indent=2) if args.json else text(data))
    required = list(data["families"]) if "all" in args.family else args.family
    missing = [name for name in required if data["families"][name].get("imports") is False or
               (not data["families"][name].get("built_in") and
                (data["families"][name]["installed"] is None or not data["families"][name]["found"]))]
    if missing:
        print(f"missing or broken: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
