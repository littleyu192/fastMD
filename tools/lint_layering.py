"""Layering lint: the engine reaches models only through the adapter interface, and a layer imports only the layers
below it.

Ported from MD's architecture checks for the single-device FastMD layout.
Reads a checkout's ``fastmd/`` with ``ast`` (nothing is imported) and reports:

* ``model``: an import statement in the engine (``fastmd.engine``, ``graph``,
  ``execution``, ``interfaces``, ``accuracy``, ``relaxation``, ``relax_cli`` and
  the ``applications`` re-exports) or in the adapter packages (``fastmd/adapters``, the model
  contract, and ``fastmd/models`` except MatRIS's own package) of a model
  (``MODEL``) or a submodule, wherever it sits (functions, ``try``,
  ``TYPE_CHECKING``). Relative imports resolve against the importing package;
  ``from x import a, b`` counts once, ``from .. import model`` counts, and
  import text inside strings does not. The model registry itself
  (``fastmd.models``: its names, not its submodules) is not a model, and a
  foreign model may import its own family.
* ``adapter``: an engine import of ``fastmd.adapters`` other than the names in
  ``INTERFACE`` or ``fastmd.adapters.protocol``: never a concrete adapter.
  ``load_adapter`` (which loads checkpoints), ``entry_plan_record`` (an
  entry point's engine plan record) and the negotiation of another model's
  plan (``negotiated``, ``negotiate``, ``renegotiate_execution``,
  ``can_capture``, and the replica batch's ``check_system_batching``,
  ``batch_systems`` and ``size_unions`` from ``fastmd.adapters.negotiate``,
  ``MeasuredMemoryModel`` from ``fastmd.adapters.memory``) belong to the
  ``ENTRY_POINTS``.
* ``runtime-class`` and ``capture``: a class in the adapter packages named like
  a runtime (``RUNTIME_WORDS``; the records in ``RECORDS``, e.g.
  ``EnginePlanRecord``, are not runtimes) or deriving from a name imported from the
  engine, ASE or torch-sim, and any CUDA-graph capture API used there. Graph
  capture and the MD runtime belong to the engine.
* ``layer``: an import of a higher layer. The layers (``LAYERS``; ``fastmd.`` left out), bottom to top:

      leaf         the root package, config, precision, _nvtx, _lazy, adapters (its __init__),
                   adapters.protocol, adapters.memory, adapters.patches, models (the model registry: its __init__,
                   which imports no model: it loads one by name)
      graph        graph
      engine       engine
      contract     adapters.negotiate
      execution    execution
      interfaces   interfaces (including TorchSim), explain, relaxation, relax_cli,
                   accuracy, a package's __main__ (its command line), applications

  FastMD's ``models.orb_graph`` and ``models.orb_ops`` belong to the ORB family;
  ``models.esen_ops`` belongs to eSEN. NequIP follows the ordinary model rules.
  Compatibility modules in ``ops`` retain their actual destination's layer.

  A model module (``fastmd.models.<family>``, MatRIS's package included) may import leaf, graph, engine and its own
  family, and MatRIS's plan record (``PLAN_RECORD``) also the contract layer, whose record of the decisions the runtime
  makes for MatRIS it fills; no other module imports one (the registry loads models by name), and in the files the
  ``model`` rule lints, imports of ``MODEL`` are left to that rule. Every import statement counts, wherever it sits:
  once per module it loads above the file's layer (``from x import y`` loads ``x.y`` if that is a module), or else for
  the innermost package above it whose ``__init__`` importing that module runs (``engine`` importing
  ``interfaces.calculator`` runs ``interfaces/__init__.py``; the registry's, which importing any model module
  runs, does not count). An old import path is in the layer of what it stands for (``OLD_PATHS``).
  A name a package forwards lazily (``from fastmd.adapters import as_adapter``: the dict literals of its
  ``__init__``, read by ``forwarded_names``) also loads the module that defines it, for this rule and the
  ``model`` and ``adapter`` rules.
* ``init``: in a package ``__init__``, an import that runs when the package is imported (not in a function or
  class, nor under ``if TYPE_CHECKING:``) of anything but the standard library and ``fastmd._lazy``: a package
  re-exports lazily, and importing it loads neither a model nor an optional dependency.
* ``string``: outside ``fastmd/models``, a model module named in a string: ``fastmd.models.<module>`` (the
  registry, ``fastmd.models``, is not one), an old path of a model with or without a submodule
  (``fastmd.model``, ``fastmd.integrations``, ``fastmd.adapters.<family>``, ``fastmd.graph.converter``,
  model-specific aliases in ``fastmd.ops``), and in
  ``config.py`` the relative ``OPTION_REGISTRY`` bindings (``"models.matris.functions"``). Docstrings do not
  count; the constant parts of f-strings do.
* ``old-path``: a file at an old import path (``OLD_PATHS``) that is not a generated stub or compatibility
  package: those start with a docstring "Moved..." and name a module of the checkout in ``_NEW``. The facade
  ``fastmd/applications/__init__.py``, which re-exports the entry points, is neither and is not checked.
* ``matris-shaped``: a foreign model's module (``fastmd/models/<family>.py``, or a module of its package)
  importing an engine module that carries MatRIS's parts (``MATRIS_SHAPED``): the engine's own modules, not API
  for a new model.
* ``--similarity`` (report only): an adapter file more than 30% similar
  (``difflib`` ratio over non-blank, non-comment lines) to an engine or model
  file; copy-paste ports are not allowed. The stubs at the old import paths of
  moved modules have no code of their own and are not compared.

The packages of the layout before the move are listed with the current ones:
``applications`` with the engine packages (``execution`` and ``interfaces``
hold its modules now), and the old import paths ``fastmd.model`` (MatRIS) and
``fastmd.integrations`` (M3GNet) with the models (``fastmd.models``), so an
import through an old path is still checked; their modules are stubs at the
old paths now. A package that does not exist is skipped. MatRIS's own packages
(``MODEL_PACKAGES``) may import the model, and the similarity check counts
their files as model files. The rules from ``layer`` on read every file of
``fastmd/`` but the facade and the stubs and compatibility packages, which
only ``old-path`` reads.

``tools/lint_layering_allowlist.txt`` lists the violations of each rule when it
was introduced, per (file, rule, target) with a count, and
may only shrink: ``--strict`` exits 1 on a violation beyond it or on an entry
above the current count (stale). Without ``--strict`` the exit status is 0.

Usage (from the checkout root)::

    python -B tools/lint_layering.py [--root DIR] [--strict] [--similarity] [--verbose]
    python -B tools/lint_layering.py --write-allowlist    # after removing violations
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import difflib
import functools
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = "tools/lint_layering_allowlist.txt"
ENGINE = ("applications", "engine", "graph", "execution", "interfaces", "accuracy", "relaxation", "relax_cli")
# The packages linted as model adapters, and MatRIS's own packages (the model, which may import itself).
ADAPTER_PACKAGES = ("adapters", "models")
MODEL_PACKAGES = ("model", "models/matris")
# The models: every package under models (MatRIS's included), and the old import paths of MatRIS's package and of the
# M3GNet integration.
MODEL = tuple(f"fastmd.{package}" for package in ("model", "models", "integrations"))
ADAPTERS = "fastmd.adapters"
PROTOCOL = "fastmd.adapters.protocol"
INTERFACE = frozenset({"as_adapter", "ModelAdapter", "ModelOwnedDerivative", "TopologySpec"})
ENTRY_POINTS = frozenset([*(f"fastmd/applications/{name}.py" for name in ("base", "gpu_md", "md", "relax",
                                                                          "torchsim_interface")),
                          *(f"fastmd/interfaces/{name}.py" for name in ("calculator", "gpu_md", "md",
                                                                        "multi_system_md", "multi_system_relax",
                                                                        "relax", "replica_md", "torchsim_interface",
                                                                        "torchsim_batch", "torchsim_native")),
                          "fastmd/relaxation.py", "fastmd/relax_cli.py"])
ENTRY_INTERFACE = frozenset({"load_adapter", "entry_plan_record"})
# The entry points' negotiation of another model's plan, imported from these modules.
NEGOTIATION_INTERFACE = {"fastmd.adapters.negotiate": frozenset({"negotiated", "negotiate", "renegotiate_execution",
                                                                 "can_capture", "check_system_batching",
                                                                 "check_system_cells", "batch_systems",
                                                                 "size_unions", "contract_of"}),
                         "fastmd.adapters.memory": frozenset({"MeasuredMemoryModel"})}
RECORDS = frozenset({"EnginePlanRecord"})
RUNTIME_WORDS = ("Runner", "Integrator", "Builder", "Engine", "Thermostat", "Logger", "Calculator",
                 "Dynamics", "Evaluator", "Communicator", "Optimizer", "Scheduler", "Driver")
RUNTIME_MODULES = tuple(f"fastmd.{package}" for package in ENGINE) + ("ase", "torch_sim")
CAPTURE_NAMES = frozenset({"CUDAGraph", "capture_graph", "make_graphed_callables"})
SIMILARITY = 0.30
# The layers, bottom to top: a module may import its own layer and the layers below it. A name stands for the module
# and its submodules, except the packages in INIT_ONLY (their __init__ only: the root package, the model contract's
# package, which forwards its names lazily, and the model registry, which loads a model by its module's name).
LAYERS = (("leaf", ("fastmd", "fastmd.config", "fastmd.precision", "fastmd._nvtx", "fastmd._lazy", "fastmd.adapters",
                    "fastmd.adapters.protocol", "fastmd.adapters.memory", "fastmd.adapters.patches", "fastmd.models",
                    "fastmd.ops")),
          ("graph", ("fastmd.graph",)),
          ("engine", ("fastmd.engine",)),
          ("contract", ("fastmd.adapters.negotiate",)),
          ("execution", ("fastmd.execution",)),
          ("interfaces", ("fastmd.interfaces", "fastmd.explain", "fastmd.applications", "fastmd.accuracy",
                          "fastmd.relaxation", "fastmd.relax_cli")))
INIT_ONLY = frozenset({"fastmd", "fastmd.adapters", "fastmd.models", "fastmd.ops"})
RANK = {layer: rank for rank, (layer, _) in enumerate(LAYERS)}
REGISTRY = "fastmd.models"
# A model module (any family's, MatRIS's included) may import the layers up to the engine, and its own family.
MODELS, MODEL_TOP = "models", "engine"
# MatRIS's plan record fills the negotiation's record with the decisions the runtime makes for MatRIS by its own rules
# (the negotiation records them, it does not make them): it may also import the contract layer.
PLAN_RECORD, PLAN_RECORD_LAYER = "fastmd.models.matris.plan_record", "contract"
# The old import paths of moved code (stubs and compatibility packages now) and what each stands for: its layer and,
# for a model, the family. The facade at FACADE re-exports the entry points; it is not a stub.
OLD_PATHS = {"fastmd.model": (MODELS, "matris"), "fastmd.integrations": (MODELS, "m3gnet"),
             "fastmd.applications": ("interfaces", None), "fastmd.graph.converter": (MODELS, "matris"),
             **{f"fastmd.adapters.{family}": (MODELS, family)
                for family in ("chgnet", "mace", "matris", "orb", "sevennet", "esen", "orb_graph",
                               "operator_rewrites")}}
# FastMD kept these compatibility modules when kernels moved into their owners.
# List the routes explicitly: a new implementation at an old path must still fail.
OLD_PATHS.update({f"fastmd.ops.{name}": ("engine", None) for name in (
    "decomposed_indexed_cat_silu_linear", "merged_frozen_projections", "row_count", "row_tiles", "sink_atomics",
    "triton_row_gather_scatter", "triton_row_gemm", "triton_segment_softmax", "triton_wide_gemm",
    "triton_zero_skip_index")})
OLD_PATHS.update({f"fastmd.ops.{name}": (MODELS, "matris") for name in (
    "chunked_gather_linear", "compiled_lowerings", "fuse_sigmoid_op", "fused_silu_op", "segment_partials",
    "triton_act_linear", "triton_csr_segment_sum", "triton_directed_pair_aggregate", "triton_edge_ffn",
    "triton_gated_ln", "triton_gated_tail", "triton_gather_cat", "triton_layernorm_act", "triton_line_envelope",
    "triton_residual_add", "triton_segment_attention", "triton_threebody_fourier", "triton_weighted_segment_sum")})
OLD_PATHS.update({f"fastmd.ops.{name}": (MODELS, family) for family, names in (
    ("orb", ("orb_inference", "orb_energy", "orb_steady", "triton_orb_inference")),
    ("esen", ("esen_inference", "triton_esen_nonlinear", "triton_esen_rotation"))) for name in names})
# A model may use its own auxiliary package, but not another model's operators.
FAMILY_ALIASES = {"orb_graph": "orb", "orb_ops": "orb", "esen_ops": "esen"}
OLD_PATHS["fastmd.adapters.orb_graph"] = (MODELS, "orb")
FACADE = "fastmd/applications/__init__.py"
# What a package __init__ may import when the package is imported, besides the standard library.
INIT_IMPORTS = ("fastmd._lazy",)
# The engine modules that carry MatRIS-specific parts: MatRIS's operator dispatch layer (dispatch, layout, runtime,
# policy), the K6 memory table (memory_admission), the saved-tensor recompute recipe (saved_recompute), MatRIS's
# activation checkpoint configuration (entry_config) and the bond-partial capture slots (capture). The engine imports
# them as its own modules, but they are not generic API: a new model does not import them.
MATRIS_SHAPED = tuple(f"fastmd.engine.{name}" for name in (
    "dispatch", "layout", "runtime", "policy", "memory_admission", "saved_recompute", "entry_config", "capture"))
# A model module named in a string: fastmd.models.<module> (the registry itself is not one), or an old path of a model
# (with or without a submodule); in OPTIONS also a relative OPTION_REGISTRY binding ("models.matris.functions").
MODEL_NAME = re.compile(r"(?<![\w.])(?:fastmd\.models\.[A-Za-z_]\w*"
                        + "".join(rf"|{re.escape(old)}\b" for old, (layer, _) in OLD_PATHS.items() if layer == MODELS)
                        + r")(?:\.[A-Za-z_]\w*)*")
OPTIONS, OPTION_BINDING = "fastmd/config.py", re.compile(r"models?(?:\.[A-Za-z_]\w*)+")
MODULE_PATH = re.compile(r"fastmd(?:\.[A-Za-z_]\w*)+")
HEADER = """\
# Layering violations tolerated by tools/lint_layering.py: <file>  <rule>  <target>  <count>.
# Entries may only be removed, never added: a new violation fails --strict, and so does an
# entry above the current count (stale: lower or delete it, e.g. with --write-allowlist).
"""


# Filled by lint() for the checkout it reads: its modules, and the names each package forwards lazily
# ({package: {name: defining module}}, forwarded_names).
_MODULES: frozenset[str] = frozenset()
_FORWARDS: dict[str, dict[str, str]] = {}


def _under(name: str, package: str | tuple[str, ...]) -> bool:
    if isinstance(package, tuple):
        return any(_under(name, one) for one in package)
    return name == package or name.startswith(package + ".")


def python_files(root: Path, package: str) -> list[Path]:
    path = root / "fastmd" / package
    return sorted(path.rglob("*.py")) if path.is_dir() else ([path.with_suffix(".py")]
                                                           if path.with_suffix(".py").is_file() else [])


@functools.cache
def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), path.as_posix())


def module_name(rel: Path) -> str:
    """The module of a file (``rel``: relative to the checkout)."""
    parts = rel.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def model_file(root: Path, path: Path) -> bool:
    """A file of MatRIS's own packages."""
    return any(path.is_relative_to(root / "fastmd" / package) for package in MODEL_PACKAGES)


def present(root: Path, packages: tuple[str, ...]) -> list[str]:
    """The packages that exist in the checkout."""
    return [package for package in packages if python_files(root, package)]


def resolve(node: ast.ImportFrom, package: str) -> str:
    """The absolute module of a ``from`` import made in a module of ``package``."""
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = ".".join(parts[:max(len(parts) - node.level + 1, 0)])
    return ".".join(part for part in (base, node.module) if part)


def loads(statement: ast.Import | ast.ImportFrom, package: str) -> list[list[str]]:
    """Per ``import`` alias, or once per ``from`` import: the module, then each ``module.name``, then the module
    defining each name the module forwards lazily."""
    if isinstance(statement, ast.Import):
        return [[alias.name] for alias in statement.names]
    module = resolve(statement, package)
    forwards = _FORWARDS.get(module, {})
    return [[module, *(f"{module}.{alias.name}" for alias in statement.names),
             *(forwards[alias.name] for alias in statement.names if alias.name in forwards)]]


def loaded(statement: ast.Import | ast.ImportFrom, package: str, modules: frozenset[str]) -> list[str]:
    """The modules an import statement loads: ``from x import y`` loads ``x.y`` if that is a module of the checkout,
    else ``x`` and, for a name ``x`` forwards lazily, the module defining it."""
    if isinstance(statement, ast.Import):
        return [alias.name for alias in statement.names]
    module = resolve(statement, package)
    forwards = _FORWARDS.get(module, {})
    return list(dict.fromkeys(name for alias in statement.names for name in (
        (f"{module}.{alias.name}",) if f"{module}.{alias.name}" in modules
        else (module, *((forwards[alias.name],) if alias.name in forwards else ())))))


def forwarded_names(root: Path) -> dict[str, dict[str, str]]:
    """``{package: {name: defining module}}``: the names a package ``__init__`` forwards lazily, read from its dict
    literals, ``{name: module}`` (``_lazy.lazy_attributes``' and the root package's maps) or ``{module: (name,
    ...)}`` (``fastmd.graph``'s)."""
    forwards = {}
    for path in sorted((root / "fastmd").rglob("__init__.py")):
        names = {}
        for node in ast.walk(parse(path)):
            if not isinstance(node, ast.Dict):
                continue
            try:
                value = ast.literal_eval(node)
            except ValueError:
                continue
            for key, item in value.items():
                if isinstance(key, str) and isinstance(item, str) and MODULE_PATH.fullmatch(item):
                    names[key] = item
                elif isinstance(key, str) and MODULE_PATH.fullmatch(key) and isinstance(item, tuple):
                    names.update(dict.fromkeys((name for name in item if isinstance(name, str)), key))
        if names:
            forwards[module_name(path.relative_to(root))] = names
    return forwards


def registry_name(name: str) -> bool:
    """The model registry, or one of its names (not a submodule): not a model."""
    return name == REGISTRY or (name.startswith(REGISTRY + ".") and "." not in name[len(REGISTRY) + 1:]
                                and name not in _MODULES)


def packages(module: str) -> list[str]:
    """``module``, then each package whose ``__init__`` importing it runs, innermost first."""
    parts = module.split(".")
    return [".".join(parts[:end]) for end in range(len(parts), 0, -1)]


def adapter_target(statement: ast.Import | ast.ImportFrom, names: list[str], entry_point: bool) -> str | None:
    """The target of an import of ``fastmd.adapters`` beyond its interface, else None."""
    module = names[0]
    if isinstance(statement, ast.ImportFrom) and _under(module, ADAPTERS):
        imports = {alias.name for alias in statement.names}
        if module == ADAPTERS:
            allowed = INTERFACE | {"protocol"} | (ENTRY_INTERFACE if entry_point else set())
            return None if imports <= allowed else module
        if entry_point and module in NEGOTIATION_INTERFACE:
            return None if imports <= NEGOTIATION_INTERFACE[module] else module
        return None if _under(module, PROTOCOL) and "*" not in imports else module
    return next((name for name in names if _under(name, ADAPTERS) and not _under(name, PROTOCOL)), None)


def bindings(tree: ast.AST, package: str) -> dict[str, str]:
    """``{name an import binds: the dotted name it refers to}`` over a whole file."""
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name if alias.asname else alias.name.split(".")[0]
                bound[alias.asname or name] = name
        elif isinstance(node, ast.ImportFrom):
            module = resolve(node, package)
            bound.update({alias.asname or alias.name: f"{module}.{alias.name}" for alias in node.names})
    return bound


def dotted(node: ast.AST, bound: dict[str, str]) -> str | None:
    """A Name/Attribute chain as a dotted name, its root resolved through the file's imports."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([bound.get(node.id, node.id), *reversed(parts)]) if isinstance(node, ast.Name) else None


def runtime_base(base: ast.expr, bound: dict[str, str]) -> str | None:
    """The first name in a base class expression that comes from the engine, ASE or torch-sim."""
    for node in ast.walk(base):
        name = dotted(node, bound) if isinstance(node, (ast.Name, ast.Attribute)) else None
        if name and any(_under(name, module) for module in RUNTIME_MODULES):
            return name
    return None


def engine_violations(rel: str, tree: ast.AST, package: str) -> tuple[list[tuple], int]:
    """The ``model`` and ``adapter`` violations of an engine file, and its interface imports."""
    found, interface = [], 0
    for statement in ast.walk(tree):
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        groups = loads(statement, package)
        model = next((name for names in groups for name in names if _under(name, MODEL) and not registry_name(name)),
                     None)
        adapter = next(filter(None, (adapter_target(statement, names, rel in ENTRY_POINTS) for names in groups)),
                       None)
        for rule, target in (("model", model), ("adapter", adapter)):
            if target:
                found.append((rel, statement.lineno, rule, target, ast.unparse(statement)))
        if not adapter and any(_under(name, ADAPTERS) for names in groups for name in names):
            interface += 1
    return found, interface


def adapter_violations(rel: str, tree: ast.AST, package: str, protocols: frozenset[str] = frozenset()) -> list[tuple]:
    """The ``runtime-class`` and ``capture`` violations of an adapter file; a base in ``protocols`` is allowed."""
    bound, found = bindings(tree, package), []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            reasons = ["runtime name"] if any(word in node.name for word in RUNTIME_WORDS) and \
                node.name not in RECORDS else []
            bases = [*node.bases, *(keyword.value for keyword in node.keywords)]
            reasons += [f"base {name}" for name in (runtime_base(base, bound) for base in bases)
                        if name and name not in protocols]
            if reasons:
                found.append((rel, node.lineno, "runtime-class", node.name, ", ".join(reasons)))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [name for group in loads(node, package) for name in group]
        elif isinstance(node, (ast.Name, ast.Attribute)):
            names = [dotted(node, bound) or node.attr]
        else:
            continue
        found += [(rel, node.lineno, "capture", name, "") for name in names
                  if name.split(".")[-1] in CAPTURE_NAMES or name.split(".")[-2:] == ["cuda", "graph"]]
    return found


def layer_of(module: str) -> tuple[str, str | None] | None:
    """The layer of a ``fastmd`` module and, for a model module, its family; None if no layer lists it. An old
    import path is in the layer of what it stands for; a command line (``python -m <package>``, the package's
    ``__main__``) is an entry point."""
    old = next((path for path in OLD_PATHS if _under(module, path)), None)
    if old:
        return OLD_PATHS[old]
    if module.endswith(".__main__"):
        return "interfaces", None
    if module != REGISTRY and _under(module, MODEL):
        family = module.split(".")[2]
        return MODELS, FAMILY_ALIASES.get(family, family)
    matches = [(len(name), layer) for layer, names in LAYERS for name in names
               if module == name or (name not in INIT_ONLY and module.startswith(name + "."))]
    return (max(matches)[1], None) if matches else None


def above(source: tuple[str, str | None], name: str, imported: bool, extra: str | None = None) -> str | None:
    """``"<layer> -> <layer>"`` (a model's: ``models.<family>``) if a module in ``source`` (layer, family) may not
    load the module ``name`` (``imported``: by its name, else as a package of the module imported), else None.
    ``extra``: a layer the module may load besides (MatRIS's plan record: the contract layer)."""
    target = layer_of(name)
    if target is None or (name == REGISTRY and not imported):  # the registry is the package of every model module
        return None
    (layer, family), (target_layer, target_family) = source, target
    if target_layer == MODELS:  # only its own family imports a model module (the registry loads one by name)
        allowed = layer == MODELS and target_family == family
    else:
        allowed = RANK[target_layer] <= RANK[MODEL_TOP if layer == MODELS else layer] or target_layer == extra
    return None if allowed else " -> ".join(f"{one}.{one_family}" if one == MODELS else one
                                            for one, one_family in (source, target))


def imports_and_strings(tree: ast.AST) -> tuple[list[ast.Import | ast.ImportFrom], list[ast.Constant]]:
    """A file's import statements, and its string constants but the docstrings."""
    imports, strings, docstrings = [], [], set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                strings.append(node)
        elif isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr):
            docstrings.add(id(node.body[0].value))
    return imports, [node for node in strings if id(node) not in docstrings]


def layer_violations(rel: str, imports: list[ast.Import | ast.ImportFrom], package: str, modules: frozenset[str],
                     model_rule: bool) -> list[tuple]:
    """The ``layer`` violations of a file: per import statement, each module it loads above the file's layer, or
    else the innermost package above it whose ``__init__`` that runs. In a file the ``model`` rule lints
    (``model_rule``), the models are left to that rule."""
    importer = module_name(Path(rel))
    source, found = layer_of(importer), []
    if source is None:
        return found
    extra = PLAN_RECORD_LAYER if _under(importer, PLAN_RECORD) else None
    for statement in imports:
        targets = {}
        for module in loaded(statement, package, modules):
            for name in packages(module):
                edge = None if model_rule and _under(name, MODEL) else above(source, name, name == module, extra)
                if edge:
                    targets.setdefault(name, edge)
                    break
        found += [(rel, statement.lineno, "layer", name, edge) for name, edge in targets.items()]
    return found


def import_time_imports(tree: ast.Module):
    """The import statements a module runs when it is imported: not in a function or class, nor under
    ``if TYPE_CHECKING:``."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If) and dotted(node.test, {}) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
            pending += node.orelse
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pending += ast.iter_child_nodes(node)


def init_violations(rel: str, tree: ast.Module, package: str, modules: frozenset[str]) -> list[tuple]:
    """The ``init`` violations of a package ``__init__``: the modules it loads when imported beyond the standard
    library and ``INIT_IMPORTS``."""
    found = []
    for statement in import_time_imports(tree):
        found += [(rel, statement.lineno, "init", module, ast.unparse(statement))
                  for module in loaded(statement, package, modules)
                  if module.split(".")[0] not in sys.stdlib_module_names and not _under(module, INIT_IMPORTS)]
    return found


def string_violations(rel: str, strings: list[ast.Constant]) -> list[tuple]:
    """The ``string`` violations of a file: the model modules its strings (docstrings aside) name."""
    found = []
    for node in strings:
        names = ([node.value] if rel == OPTIONS and OPTION_BINDING.fullmatch(node.value)
                 else MODEL_NAME.findall(node.value))
        found += [(rel, node.lineno, "string", name, "") for name in names]
    return found


def matris_shaped_violations(rel: str, imports: list[ast.Import | ast.ImportFrom], package: str,
                             modules: frozenset[str]) -> list[tuple]:
    """The ``matris-shaped`` violations of a foreign model's module: its imports of ``MATRIS_SHAPED`` modules."""
    found = []
    for statement in imports:
        found += [(rel, statement.lineno, "matris-shaped", module, ast.unparse(statement))
                  for module in loaded(statement, package, modules) if _under(module, MATRIS_SHAPED)]
    return found


def old_path_files(root: Path) -> list[Path]:
    """The files at the old import paths (``OLD_PATHS``) but the facade."""
    files = []
    for old in OLD_PATHS:
        path = root.joinpath(*old.split("."))
        files += sorted(path.rglob("*.py")) if path.is_dir() else [path.with_suffix(".py")]
    return [path for path in files if path.is_file() and path.relative_to(root).as_posix() != FACADE]


def lint(root: Path) -> tuple[list[tuple], dict[str, int], dict[str, list[str]]]:
    """Every violation ``(file, line, rule, target, detail)``; interface imports per engine package; the files each
    of the rules from ``layer`` on read, and the files outside the layers."""
    global _MODULES, _FORWARDS
    parse.cache_clear()  # Repeated lint runs must observe edits, including fixture mutations.
    paths = sorted((root / "fastmd").rglob("*.py"))
    modules = _MODULES = frozenset(module_name(path.relative_to(root)) for path in paths)
    _FORWARDS = forwarded_names(root)
    violations, interface = [], dict.fromkeys(ENGINE, 0)
    for package in (*ENGINE, *ADAPTER_PACKAGES):
        for path in python_files(root, package):
            rel = path.relative_to(root)
            source = (rel.as_posix(), parse(path), ".".join(rel.parts[:-1]))
            if package in ADAPTER_PACKAGES:
                violations += adapter_violations(*source)
                if not model_file(root, path):
                    # The model contract does not import a model either (the registry loads one by name); a foreign
                    # model may import its own family.
                    owner = layer_of(module_name(rel)) if package == "models" else None
                    violations += [violation for violation in engine_violations(*source)[0]
                                   if violation[2] == "model" and not
                                   (owner and owner[0] == MODELS and layer_of(violation[3]) == owner)]
                continue
            found, count = engine_violations(*source)
            violations += found
            interface[package] += count
    read = {rule: [] for rule in ("layer", "init", "string", "old-path", "matris-shaped", "outside the layers")}
    for path in old_path_files(root):
        rel = path.relative_to(root)
        read["old-path"].append(rel.as_posix())
        new = moved_to(path)
        if new is None or new not in modules:
            violations.append((rel.as_posix(), 1, "old-path", module_name(rel),
                               "not a generated stub" if new is None else f"_NEW names no module: {new}"))
    for path in paths:
        rel = path.relative_to(root)
        file, tree, package = rel.as_posix(), parse(path), ".".join(rel.parts[:-1])
        if file == FACADE or moved_to(path) is not None:
            continue
        imports, strings = imports_and_strings(tree)
        model_rule = Path(rel.parts[1]).stem in ENGINE or (rel.parts[1] in ADAPTER_PACKAGES and not model_file(root, path))
        violations += layer_violations(file, imports, package, modules, model_rule)
        layer = layer_of(module_name(rel))
        read["layer" if layer else "outside the layers"].append(file)
        if rel.name == "__init__.py":
            violations += init_violations(file, tree, package, modules)
            read["init"].append(file)
        if rel.parts[1] != "models":
            violations += string_violations(file, strings)
            read["string"].append(file)
        if layer and layer[0] == MODELS and layer[1] != "matris":  # a foreign model's module
            violations += matris_shaped_violations(file, imports, package, modules)
            read["matris-shaped"].append(file)
    return sorted(violations), interface, read


def read_allowlist(path: Path) -> dict[tuple, int] | None:
    """``{(file, rule, target): count}``, or None without an allowlist file."""
    if not path.is_file():
        return None
    allowed = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if fields and (len(fields) != 4 or not fields[3].isdigit() or tuple(fields[:3]) in allowed):
            raise SystemExit(f"{path.name}:{number}: expected a unique '<file>  <rule>  <target>  <count>'")
        if fields:
            allowed[tuple(fields[:3])] = int(fields[3])
    return allowed


def code_lines(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def moved_to(path: Path) -> str | None:
    """What a generated file at an old import path stands for, else None. A stub at the old path of a moved module
    and a compatibility package (an old package split over new ones) both start with a docstring "Moved..." and
    name the new module or package in ``_NEW``."""
    tree = parse(path)
    if not (ast.get_docstring(tree) or "").startswith("Moved"):
        return None
    return next((node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 and [getattr(target, "id", None) for target in node.targets] == ["_NEW"]
                 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)), None)


def similar_files(root: Path) -> tuple[list[tuple[str, str, float]], int]:
    """Adapter/core file pairs more than ``SIMILARITY`` alike, and the number of pairs compared."""
    core = {path: code_lines(path) for package in (*ENGINE, *MODEL_PACKAGES) for path in python_files(root, package)
            if moved_to(path) is None}
    adapters = [path for package in ADAPTER_PACKAGES for path in python_files(root, package)
                if not model_file(root, path) and moved_to(path) is None]
    pairs = []
    for adapter in adapters:
        lines = code_lines(adapter)
        for path, other in core.items():
            matcher = difflib.SequenceMatcher(None, lines, other, autojunk=False)
            if (matcher.real_quick_ratio() > SIMILARITY and matcher.quick_ratio() > SIMILARITY
                    and (ratio := matcher.ratio()) > SIMILARITY):
                pairs.append((adapter.relative_to(root).as_posix(), path.relative_to(root).as_posix(), ratio))
    return pairs, len(adapters) * len(core)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="checkout to lint (default: this one)")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 on a violation beyond the allowlist or a stale allowlist entry")
    parser.add_argument("--write-allowlist", action="store_true",
                        help=f"rewrite ROOT/{ALLOWLIST} from the current violations")
    parser.add_argument("--similarity", action="store_true",
                        help="also report adapter files more than 30%% similar to an engine or model file")
    parser.add_argument("--verbose", action="store_true", help="list every violation, not only new ones")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()
    if not (root / "fastmd").is_dir():
        raise SystemExit(f"--root {args.root}: no fastmd/ package")
    violations, interface, read = lint(root)
    counts = {}
    for file, _, rule, target, _ in violations:
        counts[file, rule, target] = counts.get((file, rule, target), 0) + 1
    if args.write_allowlist:
        previous = read_allowlist(root / ALLOWLIST)
        if previous is not None and any(count > previous.get(key, 0) for key, count in counts.items()):
            raise SystemExit("--write-allowlist may only shrink the existing baseline; fix new violations first")
        (root / ALLOWLIST).parent.mkdir(exist_ok=True)
        (root / ALLOWLIST).write_text(HEADER + "".join(f"{'  '.join(key)}  {count}\n"
                                                       for key, count in sorted(counts.items())))
        print(f"[lint-layering] wrote {ALLOWLIST}: {len(counts)} entries")
    print("[lint-layering] import statements of a model package (engine and adapters), and of fastmd.adapters "
          "beyond and within its interface (engine)")
    print(f"  {'package':14s} {'model':>6s} {'adapter':>8s} {'interface':>10s}")
    for package in (*present(root, ENGINE), *present(root, ADAPTER_PACKAGES), "total"):
        scoped = [v for v in violations if package in ("total", Path(v[0].split("/")[1]).stem)]
        model, adapter = (sum(v[2] == rule for v in scoped) for rule in ("model", "adapter"))
        allowed_imports = sum(interface.values()) if package == "total" else interface.get(package, 0)
        print(f"  {package:14s} {model:6d} {adapter:8d} {allowed_imports:10d}")
    runtime, capture = (sum(v[2] == rule for v in violations) for rule in ("runtime-class", "capture"))
    adapter_packages = ", ".join(f"fastmd/{package}" for package in present(root, ADAPTER_PACKAGES))
    print(f"[lint-layering] {adapter_packages or 'fastmd/adapters'}: "
          + (f"{runtime} runtime classes, {capture} capture API uses" if adapter_packages else "absent"))
    found = {rule: [v for v in violations if v[2] == rule] for rule in read}
    files = {rule: len({v[0] for v in found[rule]}) for rule in read}
    edges = Counter(v[4] for v in found["layer"])
    outside = read["outside the layers"]
    print(f"[lint-layering] layer: {len(found['layer'])} imports of a higher layer in {files['layer']} of "
          f"{len(read['layer'])} files" + "".join(f"; {edge}: {count}" for edge, count in sorted(edges.items()))
          + (f"; outside the layers (not checked): {', '.join(outside)}" if outside else ""))
    print(f"[lint-layering] init: {len(found['init'])} imports beyond the standard library and fastmd._lazy at "
          f"import time in {files['init']} of {len(read['init'])} package __init__ files")
    print(f"[lint-layering] string: {len(found['string'])} model modules named in strings in {files['string']} of "
          f"{len(read['string'])} files outside fastmd/models")
    print(f"[lint-layering] old-path: {len(found['old-path'])} of {len(read['old-path'])} files at old import paths "
          "are not generated stubs")
    print(f"[lint-layering] matris-shaped: {len(found['matris-shaped'])} imports of MatRIS-shaped engine modules in "
          f"{files['matris-shaped']} of {len(read['matris-shaped'])} foreign model modules")
    allowed = read_allowlist(root / ALLOWLIST)
    entries = "missing" if allowed is None else f"{len(allowed)} entries"
    allowed = allowed or {}
    new = {key: count for key, count in counts.items() if count > allowed.get(key, 0)}
    stale = {key: count for key, count in allowed.items() if counts.get(key, 0) < count}
    excess = sum(counts[key] - allowed.get(key, 0) for key in new)
    print(f"[lint-layering] allowlist {ALLOWLIST} ({entries}): {excess} new violations, "
          f"{len(stale)} stale entries")
    for file, line, rule, target, detail in violations:
        key = (file, rule, target)
        if key in new or args.verbose:
            mark = f"NEW ({counts[key]} found, {allowed.get(key, 0)} allowed) " if key in new else ""
            print(f"  {mark}{file}:{line}  {rule}  {target}  {detail}".rstrip())
    for key, count in sorted(stale.items()):
        print(f"  STALE {'  '.join(key)}  ({counts.get(key, 0)} found, {count} allowed): lower the entry")
    if args.similarity:
        pairs, compared = similar_files(root)
        print(f"[lint-layering] similarity: {len(pairs)} of {compared} adapter/core file pairs above "
              f"{SIMILARITY:.2f}")
        for adapter, core, ratio in pairs:
            print(f"  {adapter} ~ {core}  {ratio:.2f}")
    if args.strict and (new or stale):
        print("[lint-layering] --strict: failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
