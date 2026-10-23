"""CPU-only architecture-lint regression fixtures; no ML dependencies are imported.

Run with Python >= 3.10: ``python -B tools/test_lint_layering.py``.
Fixtures are isolated temporary checkouts; the real source tree is never edited.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("fastmd_layering_lint", Path(__file__).with_name("lint_layering.py"))
assert SPEC is not None and SPEC.loader is not None
lint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lint)


class LayeringFixtures(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="fastmd-layering-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write("fastmd/__init__.py", "")
        self.write("fastmd/models/nequip.py", "")

    def write(self, name, source):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def violations(self):
        return {(file, rule, target) for file, _, rule, target, _ in lint.lint(self.root)[0]}

    def cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            result = lint.main(["--root", str(self.root), *args])
        return result, output.getvalue()

    def test_valid_layers_are_read_without_importing_models(self):
        self.write("fastmd/adapters/protocol.py", "class ModelAdapter: pass\n")
        self.write("fastmd/engine/kernels/segment.py", "VALUE = 1\n")
        self.write("fastmd/engine/runtime.py", "from fastmd.adapters.protocol import ModelAdapter\n")
        self.write("fastmd/models/nequip.py", "from fastmd.engine.kernels import segment\nraise RuntimeError('must not import')\n")
        self.write("fastmd/models/matris/model.py", "from fastmd.engine import runtime\n")
        self.assertEqual(self.cli("--strict")[0], 0)
        self.assertEqual(self.violations(), set())
        self.assertEqual(lint.layer_of("fastmd.models.matris.model"), ("models", "matris"))

    def test_model_auxiliary_packages_keep_their_own_family(self):
        self.write("fastmd/models/orb.py", "from fastmd.models.orb_ops import orb_energy\n")
        self.write("fastmd/models/orb_graph.py", "from fastmd.models.orb import ORBAdapter\n")
        self.write("fastmd/models/orb_ops/orb_energy.py", "from fastmd.models.orb_graph import graph\n")
        self.write("fastmd/models/esen.py", "from fastmd.models.esen_ops import esen_inference\n")
        self.write("fastmd/models/esen_ops/esen_inference.py", "")
        self.assertEqual(self.cli("--strict")[0], 0)

    def test_cross_family_import_still_fails(self):
        self.write("fastmd/models/esen.py", "from fastmd.models.orb_ops.orb_energy import install\n")
        self.assertIn(("fastmd/models/esen.py", "model", "fastmd.models.orb_ops.orb_energy"), self.violations())
        self.assertEqual(self.cli("--strict")[0], 1)

    def test_engine_and_new_top_level_entry_cannot_import_model(self):
        for name in ("engine/runtime", "interfaces/torchsim_native", "relaxation", "relax_cli", "accuracy/check"):
            self.write(f"fastmd/{name}.py", "def later():\n    from fastmd.models.nequip import NequIPAdapter\n")
        violations = self.violations()
        for name in ("engine/runtime", "interfaces/torchsim_native", "relaxation", "relax_cli", "accuracy/check"):
            self.assertIn((f"fastmd/{name}.py", "model", "fastmd.models.nequip"), violations)
        self.assertEqual(self.cli("--strict")[0], 1)

    def test_lazy_root_export_is_resolved_to_model(self):
        self.write("fastmd/__init__.py", "_PUBLIC = {'MatRIS': 'fastmd.models.matris.model'}\n")
        self.write("fastmd/engine/runtime.py", "from fastmd import MatRIS\n")
        self.assertIn(("fastmd/engine/runtime.py", "model", "fastmd.models.matris.model"), self.violations())

    def test_optional_eager_import_fails_even_in_try_block(self):
        self.write("fastmd/models/__init__.py", "try:\n    import nequip\nexcept ImportError:\n    pass\n")
        self.assertIn(("fastmd/models/__init__.py", "init", "nequip"), self.violations())
        self.assertEqual(self.cli("--strict")[0], 1)

    def test_type_checking_import_is_not_eager_but_model_rule_still_applies(self):
        self.write("fastmd/models/__init__.py", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import nequip\n")
        self.assertEqual(self.cli("--strict")[0], 0)
        self.write("fastmd/engine/runtime.py", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from fastmd.models.nequip import NequIPAdapter\n")
        self.assertIn(("fastmd/engine/runtime.py", "model", "fastmd.models.nequip"), self.violations())

    def test_graph_cannot_import_interfaces(self):
        self.write("fastmd/graph/geometry.py", "from fastmd.interfaces.calculator import MLIPCalculator\n")
        self.assertIn(("fastmd/graph/geometry.py", "layer", "fastmd.interfaces.calculator"), self.violations())

    def test_adapter_cannot_own_runtime_or_capture(self):
        self.write("fastmd/models/nequip.py", "import torch\nclass NequIPRunner: pass\ndef run():\n    return torch.cuda.CUDAGraph()\n")
        violations = self.violations()
        self.assertIn(("fastmd/models/nequip.py", "runtime-class", "NequIPRunner"), violations)
        self.assertIn(("fastmd/models/nequip.py", "capture", "torch.cuda.CUDAGraph"), violations)

    def test_foreign_model_cannot_import_matris_engine_internals(self):
        self.write("fastmd/models/nequip.py", "from fastmd.engine.policy import policy\n")
        self.assertIn(("fastmd/models/nequip.py", "matris-shaped", "fastmd.engine.policy"), self.violations())

    def test_strings_are_checked_but_docstrings_are_not(self):
        self.write("fastmd/engine/runtime.py", '"""See fastmd.models.nequip."""\n')
        self.assertEqual(self.cli("--strict")[0], 0)
        self.write("fastmd/engine/runtime.py", 'MODULE = "fastmd.models.nequip"\n')
        self.assertIn(("fastmd/engine/runtime.py", "string", "fastmd.models.nequip"), self.violations())

    def test_legacy_ops_routes_retain_layer_and_check_stubs(self):
        self.write("fastmd/ops/orb_energy.py", "VALUE = 1\n")
        self.assertIn(("fastmd/ops/orb_energy.py", "old-path", "fastmd.ops.orb_energy"), self.violations())
        self.write("fastmd/engine/runtime.py", "from fastmd.ops.orb_energy import install\n")
        self.assertIn(("fastmd/engine/runtime.py", "layer", "fastmd.ops.orb_energy"), self.violations())

    def test_exact_allowlist_catches_increased_count(self):
        line = "from fastmd.models.nequip import NequIPAdapter\n"
        self.write("fastmd/engine/runtime.py", line)
        self.write(lint.ALLOWLIST, "fastmd/engine/runtime.py  model  fastmd.models.nequip  1\n")
        self.assertEqual(self.cli("--strict")[0], 0)
        self.write("fastmd/engine/runtime.py", line + line)
        result, output = self.cli("--strict")
        self.assertEqual(result, 1)
        self.assertIn("1 new violations", output)

    def test_stale_allowlist_and_repeated_runs_observe_edits(self):
        self.write("fastmd/engine/runtime.py", "from fastmd.models.nequip import NequIPAdapter\n")
        self.write(lint.ALLOWLIST, "fastmd/engine/runtime.py  model  fastmd.models.nequip  1\n")
        self.assertEqual(self.cli("--strict")[0], 0)
        self.write("fastmd/engine/runtime.py", "")
        result, output = self.cli("--strict")
        self.assertEqual(result, 1)
        self.assertIn("1 stale entries", output)
        self.assertEqual(self.cli("--write-allowlist", "--strict")[0], 0)

    def test_write_allowlist_cannot_accept_new_violations(self):
        self.write(lint.ALLOWLIST, "")
        self.write("fastmd/models/__init__.py", "import torch\n")
        with self.assertRaisesRegex(SystemExit, "may only shrink"):
            self.cli("--write-allowlist")
        self.assertEqual((self.root / lint.ALLOWLIST).read_text(), "")


if __name__ == "__main__":
    unittest.main()
