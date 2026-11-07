"""Parity and lifecycle gates for opt-in CHGNet indexed first projections."""
from __future__ import annotations

import copy
import importlib.util
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(importlib.util.find_spec("chgnet"), "optional CHGNet unavailable")
class CHGNetIndexedProjectionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)

    def test_first_projection_values_and_vjp_including_output_only_angle(self):
        from chgnet.model.functions import GatedMLP
        from fastmd.models.chgnet_fusion import _Projection

        for hidden, norm, rows in ((7, "layer", 19), (0, "layer", 19), (7, "layer", 2), (7, None, 0)):
            with self.subTest(hidden=hidden, norm=norm, rows=rows):
                gated = GatedMLP(16, 4, hidden_dim=hidden, norm=norm).double().eval()
                gated.requires_grad_(False)
                tables = [torch.randn(n, 4, dtype=torch.float64, requires_grad=True) for n in (5, 8, rows)]
                i, j, a = (torch.randint(n, (rows,)) for n in (8, 8, 5))
                sources, indices = (tables[1], tables[1], tables[2], tables[0]), (i, j, None, a)
                expected = gated(torch.cat((tables[1][i], tables[1][j], tables[2], tables[0][a]), dim=1))
                projection = _Projection(gated, (4, 4, 4, 4), "torch")
                with patch.object(gated, "forward", wraps=gated.forward) as original:
                    actual = projection(sources, indices, "angle")
                    self.assertEqual(original.call_count, int(rows < 3))
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                cotangent = torch.randn_like(actual)
                want = torch.autograd.grad(expected, tables, cotangent, retain_graph=True)
                got = torch.autograd.grad(actual, tables, cotangent, retain_graph=True)
                for g, w in zip(got, want):
                    torch.testing.assert_close(g, w, rtol=1e-10, atol=1e-11)
                # In-place checkpoint updates must refresh the frozen split cache.
                with torch.no_grad():
                    projection.core.weight.add_(0.01)
                changed = projection(sources, indices, "angle")
                reference = gated(torch.cat((tables[1][i], tables[1][j], tables[2], tables[0][a]), dim=1))
                torch.testing.assert_close(changed, reference, rtol=1e-11, atol=1e-12)
                projection.core.weight.requires_grad_(True)
                with self.assertRaisesRegex(RuntimeError, "frozen inference"):
                    projection(sources, indices, "angle")
                projection.core.weight.requires_grad_(False)
                gated.train()
                with self.assertRaisesRegex(RuntimeError, "before training"):
                    projection(sources, indices, "angle")

    def test_model_energy_force_stress_and_revert(self):
        from chgnet.model.model import CHGNet
        from pymatgen.io.ase import AseAtomsAdaptor
        from fastmd.models.chgnet import CHGNetAdapter
        from fastmd.engine.energy_derivative import EnergyDerivative
        from adapters.test_chgnet_empty_angles import batch_of, structures

        base = CHGNet.load(model_name="0.3.0", use_device="cpu", verbose=False)
        baseline = CHGNetAdapter(base)
        candidate = CHGNetAdapter(copy.deepcopy(base))
        self.addCleanup(baseline.close)
        self.addCleanup(candidate.close)
        candidate.enable_operator_rewrites("torch")
        self.assertEqual(candidate.system_inputs(), {"operator_backend": "torch"})
        self.assertEqual(len(candidate._operator_patches), 12)
        self.assertTrue(all(p.applied for p in candidate._operator_patches))
        for atoms in structures():
            graph = base.graph_converter(AseAtomsAdaptor.get_structure(atoms))
            batch = batch_of(graph)
            results = [EnergyDerivative(adapter)(batch, adapter.prepare(batch), task="efs", n_real=batch.n_real)
                       for adapter in (baseline, candidate)]
            for key, atol in (("e", 2e-4), ("f", 2e-5), ("s", 2e-6)):
                for actual, expected in zip(results[1][key], results[0][key]):
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-5)
        with self.assertRaisesRegex(RuntimeError, "before preparing"):
            candidate.enable_operator_rewrites("torch")
        candidate.close()
        self.assertNotIn("forward", vars(candidate.module.atom_conv_layers[0]))
        self.assertFalse(hasattr(candidate.module.atom_conv_layers[0], "_fastmd_chgnet_projection"))

    def test_declined_source_pin_rolls_back_and_allows_retry(self):
        from chgnet.model.model import CHGNet
        from fastmd.models.chgnet import CHGNetAdapter
        from fastmd.models import chgnet_fusion

        adapter = CHGNetAdapter(CHGNet.load(model_name="0.3.0", use_device="cpu", verbose=False))
        self.addCleanup(adapter.close)
        with patch.dict(chgnet_fusion._HASHES, {"BondConv": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "refused"):
                adapter.enable_operator_rewrites("torch")
        self.assertEqual(adapter.operator_backend, "upstream")
        self.assertEqual(adapter._operator_patches, [])
        self.assertNotIn("forward", vars(adapter.module.atom_conv_layers[0]))
        self.assertFalse(hasattr(adapter.module.atom_conv_layers[0], "_fastmd_chgnet_projection"))
        adapter.enable_operator_rewrites("torch")


if __name__ == "__main__":
    unittest.main()
