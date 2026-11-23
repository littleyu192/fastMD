"""CPU public-constructor compatibility, with the calculator loader mocked."""
import inspect
import unittest
from unittest.mock import patch

from ase.optimize import BFGS

from fastmd import InferenceConfig, StructOptimizer


class StructOptimizerPolicyCompatibilityTest(unittest.TestCase):
    def test_default_keeps_main_latency_policy(self):
        with patch("fastmd.interfaces.relax.MLIPCalculator") as calculator:
            optimizer = StructOptimizer()
            self.assertEqual(calculator.call_args.kwargs["auto_policy"], "latency")
            self.assertEqual(optimizer.engine, "auto")
            optimizer.close()
            calculator.return_value.close.assert_called_once()

    def test_explicit_policy_and_config_reach_calculator(self):
        config = InferenceConfig(report=False)
        for policy in ("latency", "throughput", "latency_trial"):
            with self.subTest(policy=policy), patch("fastmd.interfaces.relax.MLIPCalculator") as calculator:
                optimizer = StructOptimizer(config=config, auto_policy=policy, engine="ase")
                self.assertEqual(calculator.call_args.kwargs["auto_policy"], policy)
                self.assertIs(calculator.call_args.kwargs["config"], config)
                self.assertEqual(optimizer.engine, "ase")
                optimizer.close()

    def test_legacy_seven_positional_arguments_keep_their_meaning(self):
        config = InferenceConfig(report=False)
        model = object()
        with patch("fastmd.interfaces.relax.MLIPCalculator") as calculator:
            optimizer = StructOptimizer("local-checkpoint", model, "ef", "BFGS", "cpu", False, config)
            calculator.assert_called_once_with(model_path="local-checkpoint", model=model, task="ef",
                                               device="cpu", enable_compile=False, config=config,
                                               auto_policy="latency")
            self.assertIs(optimizer.optimizer, BFGS)
            self.assertEqual(optimizer.engine, "auto")
            optimizer.close()

    def test_engine_keeps_md_position_and_policy_is_keyword_only(self):
        for engine in ("auto", "gpu", "ase"):
            with self.subTest(engine=engine), patch("fastmd.interfaces.relax.MLIPCalculator") as calculator:
                # MD's positional call: the eighth argument is the engine.
                optimizer = StructOptimizer(None, "matris_10m_oam", "efs", "FIRE", "cpu", True, None, engine)
                self.assertEqual(optimizer.engine, engine)
                self.assertEqual(calculator.call_args.kwargs["auto_policy"], "latency")
                optimizer.close()
        parameters = inspect.signature(StructOptimizer).parameters
        self.assertEqual(list(parameters)[:8], ["model_path", "model", "task", "optimizer", "device",
                                                "enable_compile", "config", "engine"])
        self.assertEqual(parameters["engine"].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(parameters["auto_policy"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["auto_policy"].default, "latency")
        with patch("fastmd.interfaces.relax.MLIPCalculator") as calculator:
            # A policy passed in the engine's place is rejected before anything loads.
            with self.assertRaisesRegex(ValueError, "engine must be one of"):
                StructOptimizer(None, "matris_10m_oam", "efs", "FIRE", "cpu", True, None, "throughput")
            with self.assertRaises(TypeError):
                StructOptimizer(None, "matris_10m_oam", "efs", "FIRE", "cpu", True, None, "ase", "throughput")
            calculator.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
