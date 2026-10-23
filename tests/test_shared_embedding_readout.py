"""CPU contracts for the graph-independent original embedding/readout entrypoints."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from fastmd.models.matris.basis_function import (
    BesselExpansion, FourierExpansion, GaussianExpansion, PolynomialEnvelope,
    SphericalExpansion, angular_basis, radial_basis,
)
from fastmd.models.matris.feature_embed import AtomTypeEmbedding, EdgeBasisEmbedding, ThreebodyEmbedding
from fastmd.models.matris.functions import aggregate
from fastmd.models.matris.model import MatRIS


def ordinary_module(module, value):
    return module(value)


def tiny_model(reference="demo"):
    return MatRIS(num_layers=0, node_feat_dim=4, edge_feat_dim=4,
                  three_body_feat_dim=4, mlp_hidden_dims=(4,), num_radial=3,
                  num_angular=5, max_n=2, max_l=2, norm_type="layer",
                  three_body_expansion="fourier", reference_energy=reference).double()


def compare_vjp(case, actual_fn, expected_fn, values, parameters=()):
    actual_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    expected_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    actual, expected = actual_fn(*actual_inputs), expected_fn(*expected_inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    case.assertEqual(len(actual), len(expected))
    probes = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
        probes.append(torch.full_like(got, (index + 1) / 7))
    actual_grads = torch.autograd.grad(actual, [*actual_inputs, *parameters], probes, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, [*expected_inputs, *parameters], probes, allow_unused=True)
    for got, want in zip(actual_grads, expected_grads):
        if want is None:
            case.assertIsNone(got)
        else:
            torch.testing.assert_close(got, want, atol=1e-11, rtol=1e-11)


class SharedEmbeddingReadoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(143)

    def test_atom_embedding_policy_and_atomic_number_entrypoint(self):
        module = AtomTypeEmbedding(4).double()
        types = torch.tensor([0, 5, 7, 0], dtype=torch.int32)
        expected = module.atom_swish_layer(module.embedding(types))
        actual = module(types, module_forward=ordinary_module)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        model = tiny_model()
        model.atom_embedding = module
        torch.testing.assert_close(model.embed_atoms(types + 1), expected, atol=0, rtol=0)

    def test_basis_wrappers_call_original_modules_and_preserve_training_vjp(self):
        modules = [PolynomialEnvelope(), BesselExpansion(num_radial=3, learnable=True),
                   GaussianExpansion(num_gaussians=3)]
        lengths = torch.tensor([0.7, 1.5, 3.0], dtype=torch.float64)
        for module in modules:
            module.double()
            with self.subTest(kind=type(module).__name__):
                compare_vjp(self, lambda x: radial_basis(module, x), module,
                            [lengths], tuple(module.parameters()))
        angles = torch.tensor([0.2, 1.0, 2.8], dtype=torch.float64)
        for module in (FourierExpansion(max_f=2, learnable=True),
                       SphericalExpansion(max_n=2, max_l=2)):
            module.double()
            distance = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64)
            with self.subTest(kind=type(module).__name__):
                expected = ((lambda a, d: module(a)) if isinstance(module, FourierExpansion)
                            else (lambda a, d: module(d, a)))
                compare_vjp(self, lambda a, d: angular_basis(module, a, d), expected,
                            [angles, distance], tuple(module.parameters()))
        with self.assertRaises(TypeError):
            radial_basis(nn.Identity(), lengths)
        with self.assertRaises(TypeError):
            angular_basis(nn.Identity(), angles, lengths)
        with self.assertRaises(ValueError):
            angular_basis(SphericalExpansion(max_n=1, max_l=1), angles, None)

    def test_edge_lengths_match_original_formula_and_all_parameter_vjps(self):
        lengths = torch.tensor([0.8, 1.4, 2.1], dtype=torch.float64)
        for expansion in ("Bessel", "Gaussian"):
            module = EdgeBasisEmbedding(num_radial=3, edge_feat_dim=4,
                                        distance_expansion=expansion, learnable=True).double()

            def original(x):
                atom = module.pairwise_rbf_expansion(x)
                line = module.threebody_rbf_expansion(x)
                features = module.edge_init_norm(module.edge_linear2(module.edge_linear1(atom)))
                return module.swish_layer(features), atom, line

            with self.subTest(expansion=expansion):
                compare_vjp(self, lambda x: module.forward_lengths(x, module_forward=ordinary_module),
                            original, [lengths], tuple(module.parameters()))

    def test_edge_graph_adapter_preserves_directed_legacy_and_empty_rows(self):
        module = EdgeBasisEmbedding(num_radial=3, edge_feat_dim=4).double()
        for rows in (0, 3):
            lengths = torch.linspace(0.8, 2.0, rows, dtype=torch.float64)
            directed_map = torch.arange(rows).repeat_interleave(2)
            graphs = {"edge_lengths": lengths.repeat_interleave(2),
                      "undirected2directed": torch.arange(rows) * 2,
                      "directed2undirected": directed_map}
            for undirected in (False, True):
                with self.subTest(rows=rows, undirected=undirected), patch(
                    "fastmd.models.matris.feature_embed._UNDIRECTED_EDGE_INIT", undirected
                ):
                    features, smooth = module(graphs)
                    atom = module.pairwise_rbf_expansion(lengths)
                    features0 = module.edge_linear1(atom)
                    if not undirected:
                        features0 = features0[directed_map]
                    expected = module.edge_init_norm(module.edge_linear2(features0))
                    if not undirected:
                        expected = aggregate(expected, directed_map, average=True, num_segment=rows)
                    expected = module.swish_layer(expected)
                    torch.testing.assert_close(features, expected, atol=0, rtol=0)
                    torch.testing.assert_close(smooth["atom graph"], atom, atol=0, rtol=0)

    def test_indexed_and_vector_triplets_match_original_fourier_sh_vjps(self):
        for expansion in ("fourier", "sh"):
            module = ThreebodyEmbedding(num_angular=5, max_n=2, max_l=2,
                                        three_body_feat_dim=4, three_body_expansion=expansion).double()
            for rows in (0, 3):
                units = torch.nn.functional.normalize(torch.randn(5, 3, dtype=torch.float64), dim=1)
                target, source = torch.tensor([0, 1, 3][:rows]), torch.tensor([1, 3, 0][:rows])
                target, source = target.long(), source.long()

                def original(unit):
                    left, right = unit[target], unit[source]
                    angle = torch.acos((left * right).sum(1) * (1 - 1e-6))
                    basis = (module.expansion(right.norm(dim=1), angle) if expansion == "sh"
                             else module.expansion(angle))
                    return module.swish_layer(module.angle_embedding(basis))

                with self.subTest(expansion=expansion, rows=rows):
                    compare_vjp(self, lambda x: module.forward_indexed(
                        x, target, source, module_forward=ordinary_module, allow_fused=False),
                        original, [units], tuple(module.parameters()))
                    graph = {"unit_edge_vectors": units, "line_graph_dict":
                             {"target_DE_index": target, "source_DE_index": source}}
                    torch.testing.assert_close(module(graph), original(units), atol=0, rtol=0)

    def test_per_atom_rows_and_graph_sums(self):
        model = tiny_model("demo")
        nodes = torch.randn(5, 4, dtype=torch.float64)
        numbers = torch.tensor([1, 6, 8, 6, 1])
        active = torch.tensor([True, False, True, True, True])
        real = torch.tensor([True, True, True, True, False])
        weights = model.reference_energy.fc.weight[0, numbers - 1]

        def original(value):
            learned = model.energy_head.energy_head(model.readout_norm(value)).reshape(-1).masked_fill(~active, 0)
            return torch.where(real, learned + weights, 0)

        # Isolated rows keep their reference; dummy rows contribute nothing.
        compare_vjp(self, lambda x: model.atomic_energy_rows(model.readout_norm(x), numbers, active, real_mask=real,
                                                             module_forward=ordinary_module),
                    original, [nodes], tuple(p for p in model.parameters() if p.requires_grad))
        rows = original(nodes)
        batch = {"num_graphs": 2, "atom_segment": torch.tensor([0, 0, 0, 1, 1])}
        torch.testing.assert_close(model._graph_sum(rows, batch), torch.stack((rows[:3].sum(), rows[3:].sum())))
        torch.testing.assert_close(model._graph_sum(rows, {"num_graphs": 1}), rows.sum().reshape(1))
        # FP32 rows of reference magnitude: per-graph sums accumulate in FP64,
        # so each is the FP32 rounding of the (FP64) exact sum, and trailing
        # zero (dummy) rows do not change a single graph's bits.
        generator = torch.Generator().manual_seed(0)
        rows32 = (model.reference_energy.fc.weight[0, torch.randint(0, 8, (3000,), generator=generator)].float()
                  + 0.01 * torch.randn(3000, generator=generator)).float()
        segment = (torch.arange(3000) >= 1800).long()
        summed = model._graph_sum(rows32, {"num_graphs": 2, "atom_segment": segment})
        self.assertEqual(summed.dtype, torch.float64)
        exact = torch.stack((rows32[:1800].double().sum(), rows32[1800:].double().sum()))
        torch.testing.assert_close(summed, exact, atol=1e-9, rtol=0)
        single = model._graph_sum(rows32[:1800], {"num_graphs": 1})
        padded = model._graph_sum(torch.cat((rows32[:1800], rows32.new_zeros(64))), {"num_graphs": 1})
        torch.testing.assert_close(single, exact[:1], atol=1e-9, rtol=0)
        torch.testing.assert_close(padded.float(), single.float(), atol=0, rtol=0)

    def test_owned_energy_keeps_isolated_references_and_unscaled_total(self):
        for reference in (None, "demo"):
            for intensive in (False, True):
                model = tiny_model(reference)
                model.is_intensive = intensive
                nodes = torch.randn(3, 4, dtype=torch.float64)
                numbers = torch.tensor([1, 6, 8])
                active = torch.tensor([True, False, True])

                def original(value):
                    learned = model.energy_head.energy_head(model.readout_norm(value)).reshape(-1)
                    learned = learned.masked_fill(~active, 0)
                    if model.reference_energy is not None:
                        learned = learned + model.reference_energy.fc.weight[0, numbers - 1]
                    return learned.sum()

                with self.subTest(reference=reference, intensive=intensive):
                    compare_vjp(self, lambda x: model.atomic_energy(x, numbers, active,
                                                                  module_forward=ordinary_module),
                                original, [nodes], tuple(p for p in model.parameters() if p.requires_grad))
                if reference is not None:
                    all_isolated = model.atomic_energy(nodes, numbers, torch.zeros_like(active))
                    torch.testing.assert_close(all_isolated, model.reference_energy.fc.weight[0, numbers - 1].sum())


if __name__ == "__main__":
    unittest.main()
