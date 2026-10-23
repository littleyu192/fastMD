"""CPU-only checks for single/distributed benchmark initial-frame parity."""
from types import SimpleNamespace
import unittest

import torch

from utils.benchmark_single_ef import initial_snapshot


class SingleBenchmarkInitialFrameTest(unittest.TestCase):
    def fixture(self, with_reference=True):
        # These values cannot all be represented exactly by FP32. Recording
        # ASE's FP64 originals would therefore produce different initial JSON.
        original_positions = torch.tensor([[0.123456789, 1.123456789, 2.123456789],
                                           [3.123456789, 4.123456789, 5.123456789]], dtype=torch.float64)
        positions = original_positions.float()
        momenta = torch.tensor([[0.234567891] * 3, [-0.345678912] * 3], dtype=torch.float32)
        masses = torch.tensor([1.007940001, 12.010700001], dtype=torch.float32)
        numbers = torch.tensor([1, 6], dtype=torch.int32)
        weights = torch.arange(1, 95, dtype=torch.float32).reshape(1, -1) * -0.125
        reference = SimpleNamespace(fc=SimpleNamespace(weight=weights)) if with_reference else None
        evaluator = SimpleNamespace(
            atomic_numbers=numbers, masses=masses,
            # The function must use the actual evaluate argument, not this
            # intentionally different placeholder or an ASE-side source.
            state=SimpleNamespace(positions=positions + 10, momenta=momenta),
            model=SimpleNamespace(reference_energy=reference))
        forces = torch.tensor([[0.4, -0.5, 0.6], [0.7, 0.8, -0.9]], dtype=torch.float32)
        energy = torch.tensor(-9.123456789, dtype=torch.float32)
        return evaluator, positions, forces, energy, original_positions

    def test_schema_and_fp32_evaluated_state_match_distributed_initial_frame(self):
        # A missing reference table serializes zero without changing the energy.
        for with_reference, reference_energy in ((True, -0.875), (False, 0.0)):
            with self.subTest(with_reference=with_reference):
                evaluator, positions, forces, energy, original_positions = self.fixture(with_reference)
                frame, energies = initial_snapshot(evaluator, positions, forces, energy)
                self.assertEqual(set(frame), {"ids", "numbers", "positions", "masses", "momenta", "forces"})
                self.assertEqual(frame["ids"], [0, 1])
                self.assertEqual(frame["numbers"], [1, 6])
                self.assertEqual(frame["positions"], positions.tolist())
                self.assertNotEqual(frame["positions"], original_positions.tolist())
                self.assertNotEqual(frame["positions"], evaluator.state.positions.tolist())
                self.assertEqual(frame["masses"], evaluator.masses.tolist())
                self.assertEqual(frame["momenta"], evaluator.state.momenta.tolist())
                self.assertEqual(frame["forces"], forces.tolist())
                self.assertEqual(energies, {"energy_eV": float(energy),
                                            "energy_eV_per_atom": float(energy) / 2,
                                            "reference_energy_eV": reference_energy})

    def test_snapshot_does_not_alias_tensors_modified_by_later_md(self):
        evaluator, positions, forces, energy, _ = self.fixture()
        frame, energies = initial_snapshot(evaluator, positions, forces, energy)
        import json
        before = json.dumps((frame, energies), allow_nan=False)
        for value in (positions, evaluator.atomic_numbers, evaluator.masses,
                      evaluator.state.momenta, forces, energy):
            value.add_(1)
        self.assertEqual(json.dumps((frame, energies), allow_nan=False), before)


if __name__ == "__main__":
    unittest.main()
