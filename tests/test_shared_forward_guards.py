"""CPU Dynamo tracing and E/F/VJP parity of the single-device shared forward."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from single_gpu_fixtures import frozen_fixture, hand_built_dispatcher, toy_model


class SharedForwardCompileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_existing_enable_compile_wraps_shared_layers_and_preserves_energy_force(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        # Each shared layer alone is fullgraph-traceable (eager backend), with
        # unchanged outputs and VJPs.
        model = toy_model(reference=None, smoothed=True, dtype=torch.float64)
        block = model.interaction_block[0]
        target = torch.tensor([0, 1, 0, 2, 2, 3])
        source = torch.tensor([1, 0, 2, 0, 3, 2])
        pairs = torch.tensor([0, 0, 1, 1, 2, 2])
        graph = {"source_index": source, "target_index": target, "num_undirected": 3,
                 "source_bincount": torch.bincount(source, minlength=4).clamp_min(1),
                 "target_bincount": torch.bincount(target, minlength=4).clamp_min(1),
                 "_ops": hand_built_dispatcher()}
        for kind in ("attention", "refinement"):
            with self.subTest(layer=kind):
                layer = (block.attn_block_atom_graph if kind == "attention"
                         else block.refine_block_atom_graph)
                counter = CompileCounterWithBackend("eager")
                compiled = torch.compile(layer, backend=counter, fullgraph=True)
                values = (torch.randn(4, 6, dtype=torch.float64),
                          torch.randn(3, 6, dtype=torch.float64),
                          torch.randn(3, 3, dtype=torch.float64))
                outputs, inputs = [], []
                for function in (layer, compiled):
                    nodes, edges, radial = [value.clone().requires_grad_(True) for value in values]
                    result = (function(nodes, edges, graph, pairs) if kind == "attention"
                              else function(nodes, edges, radial, graph, pairs))
                    inputs.append((nodes, edges, radial))
                    outputs.append(result)
                self.assertGreaterEqual(counter.frame_count, 1)  # The compiled layer was traced.
                cotangents = tuple(torch.randn_like(value) for value in outputs[0])
                for expected, actual in zip(*outputs):
                    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)
                gradients = [torch.autograd.grad(output, arguments, cotangents, allow_unused=True)
                             for output, arguments in zip(outputs, inputs)]
                for expected, actual in zip(*gradients):
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-11)

        # The model's existing enable_compile path: energy and forces unchanged.
        with self.subTest(model="enable_compile"):
            plain = toy_model(reference=None, smoothed=True, dtype=torch.float64)
            compiled = deepcopy(plain)
            for block in compiled.interaction_block:
                block.enable_compile = True
            real_compile = torch.compile
            counter = CompileCounterWithBackend("eager")

            def eager_compile(module, **kwargs):
                return real_compile(module, backend=counter, fullgraph=True, **kwargs)

            _, graph = frozen_fixture(dtype=torch.float64, periodic=True)
            expected = plain([graph], task="ef")
            with patch("torch.compile", side_effect=eager_compile):
                _, graph = frozen_fixture(dtype=torch.float64, periodic=True)
                actual = compiled([graph], task="ef")
            self.assertGreaterEqual(counter.frame_count, 1)  # enable_compile did compile.
            torch.testing.assert_close(actual["e"], expected["e"], atol=2e-11, rtol=2e-10)
            torch.testing.assert_close(actual["f"][0], expected["f"][0], atol=2e-11, rtol=2e-10)


if __name__ == "__main__":
    unittest.main()
