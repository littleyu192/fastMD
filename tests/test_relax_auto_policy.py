"""CPU public-constructor compatibility, with the calculator loader mocked."""
import inspect
import unittest
from unittest.mock import patch

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

    def test_policy_and_new_engine_are_keyword_only(self):
        parameters = inspect.signature(StructOptimizer).parameters
        for name in ("auto_policy", "engine"):
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["auto_policy"].default, "latency")
        with patch("fastmd.interfaces.relax.MLIPCalculator") as calculator:
            with self.assertRaises(TypeError):
                StructOptimizer(None, "matris_10m_oam", "efs", "FIRE", "cpu", True, None, "throughput")
            calculator.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
