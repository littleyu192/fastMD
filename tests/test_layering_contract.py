"""The public negotiation probe is restricted to interface entry points."""
import ast
import importlib.util
from pathlib import Path
import unittest


class NegotiationImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "tools/lint_layering.py"
        spec = importlib.util.spec_from_file_location("layering_contract_lint", path)
        cls.lint = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.lint)

    def test_public_probe_is_allowed_only_at_entry_points(self):
        module = "fastmd.adapters.negotiate"
        node = ast.parse(f"from {module} import dry_run").body[0]
        self.assertIsNone(self.lint.adapter_target(node, [module], True))
        self.assertEqual(self.lint.adapter_target(node, [module], False), module)

    def test_private_names_and_wildcards_remain_rejected(self):
        module = "fastmd.adapters.negotiate"
        for name in ("_dry_run", "*"):
            with self.subTest(name=name):
                node = ast.parse(f"from {module} import {name}").body[0]
                self.assertEqual(self.lint.adapter_target(node, [module], True), module)


if __name__ == "__main__":
    unittest.main()
