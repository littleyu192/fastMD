"""CPU regressions for short-task admission and memory-bound provenance."""
from __future__ import annotations

from dataclasses import replace
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from ase import Atoms
import torch

from fastmd.adapters.memory import MeasuredMemoryModel, MemoryPoint
from fastmd.adapters.protocol import CaptureContract, PaddingRule, TopologySpec
from fastmd.config import InferenceConfig, resolve_config
from fastmd.execution._memory_probe import measured_memory
from fastmd.execution._memory_probe_cache import (
    _CACHE, _MAX_ENTRIES, cached_memory, probe_cache_key, remember_memory,
)
from fastmd.interfaces.calculator import MLIPCalculator


def fitted(*, triplets=False):
    sizes = ((8, 4), (16, 8), (24, 24), (16, 32)) if triplets else ((8, 0), (16, 0), (32, 0))
    return MeasuredMemoryModel.from_points(
        [MemoryPoint(kind, edges, triples, 10000 + 100 * edges + 200 * triples)
         for kind in ("eager", "capture") for edges, triples in sizes], triplets=triplets)


class ProbeFixture:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2).eval()
        self.model.register_buffer("reference", torch.zeros(1))
        self.head = "default"
        self.adapter = SimpleNamespace(
            module=self.model,
            topology_spec=lambda: TopologySpec(cutoff=5.0, num_interactions=1, model_dtype="float32"),
            padding_rule=lambda: PaddingRule("far_self_loops"),
            capture_contract=lambda: CaptureContract(sync_free=True, exact_cutoff=True),
            system_inputs=lambda: {"head": self.head},
        )
        self.atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
        self.config = resolve_config(InferenceConfig(execution="auto", report=False))

    def key(self, *, atoms=None, config=None, task="efs", device="cpu"):
        return probe_cache_key(self.adapter, self.atoms if atoms is None else atoms,
                               self.config if config is None else config, task=task, device=device)

    def calculator(self, policy):
        # Exercise the real policy/cache method without loading a foreign model or CUDA.
        calculator = object.__new__(MLIPCalculator)
        calculator.model, calculator._adapter = self.model, self.adapter
        calculator.task = "efs"
        calculator.auto_policy = policy
        calculator._memory_probe_stats = {"cache_hits": 0, "measurements": 0, "skipped": 0,
                                          "last": "not_requested"}
        return calculator


class MemoryProbeCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ProbeFixture()
        self.memory = fitted()

    def remember(self):
        f = self.fixture
        key = f.key()
        self.assertIsNotNone(key)
        remember_memory(f.model, key, self.memory, 48)
        return key

    def test_reuses_only_inside_the_fits_supported_capacity_range(self):
        f, key = self.fixture, self.remember()
        self.assertIs(cached_memory(f.model, key, 48), self.memory)
        self.assertIs(cached_memory(f.model, key, 64), self.memory)
        self.assertIsNone(cached_memory(f.model, key, 65))
        self.assertIsNone(cached_memory(f.model, key, 48, 1))

    def test_triplet_bounds_reject_unmeasured_ratio_and_excess_capacity(self):
        f, memory = self.fixture, fitted(triplets=True)
        remember_memory(f.model, f.key(), memory, 24, 24)
        self.assertIs(cached_memory(f.model, f.key(), 24, 24), memory)
        self.assertIsNone(cached_memory(f.model, f.key(), 24, 1))
        self.assertIsNone(cached_memory(f.model, f.key(), 24, 100))

    def test_task_precision_configuration_and_adapter_inputs_are_separate(self):
        f = self.fixture
        self.remember()
        for key in (f.key(task="ef"), f.key(device="cpu:1"),
                    f.key(config=replace(f.config, matmul_precision="high")),
                    f.key(config=replace(f.config, topology_reuse="off"))):
            self.assertIsNone(cached_memory(f.model, key, 48))
        f.head = "different"
        self.assertIsNone(cached_memory(f.model, f.key(), 48))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertIsNone(cached_memory(f.model, f.key(), 48))

    def test_geometry_and_atom_count_changes_do_not_reuse_fit(self):
        f = self.fixture
        self.remember()
        changed = []
        atoms = f.atoms.copy(); atoms.positions[0, 0] += 0.001; changed.append(atoms)
        atoms = f.atoms.copy(); atoms.cell[0, 0] += 0.001; changed.append(atoms)
        atoms = f.atoms.copy(); atoms.numbers[0] = 2; changed.append(atoms)
        atoms = f.atoms.copy(); atoms.pbc[0] = False; changed.append(atoms)
        changed.append(f.atoms[:1])
        for atoms in changed:
            self.assertIsNone(cached_memory(f.model, f.key(atoms=atoms), 48))
        self.assertIs(cached_memory(f.model, f.key(atoms=f.atoms.copy()), 48), self.memory)

    def test_in_place_weight_edits_rebinding_buffers_and_dtype_invalidate(self):
        for change in (lambda m: m.weight.add_(1),
                       lambda m: setattr(m, "weight", torch.nn.Parameter(m.weight.clone())),
                       lambda m: m.reference.add_(1), lambda m: m.double(), lambda m: m.train(),
                       lambda m: setattr(m, "forward", lambda x: x),
                       lambda m: m.register_forward_hook(lambda *_: None)):
            f = ProbeFixture()
            key = f.key()
            remember_memory(f.model, key, self.memory, 48)
            with torch.no_grad():
                change(f.model)
            self.assertIsNone(cached_memory(f.model, key, 48))
            self.assertIsNone(cached_memory(f.model, f.key(), 48))

    def test_models_with_equal_weights_still_do_not_share_a_fit(self):
        f = self.fixture
        key = self.remember()
        other = ProbeFixture()
        other.model.load_state_dict(f.model.state_dict())
        self.assertIsNone(cached_memory(other.model, key, 48))
        self.assertIsNone(cached_memory(other.model, other.key(), 48))

    def test_failed_or_changed_during_measurement_results_are_not_cached(self):
        f, key = self.fixture, self.fixture.key()
        remember_memory(f.model, key, MeasuredMemoryModel.failed("OOM", triplets=False), 48)
        self.assertIsNone(cached_memory(f.model, key, 48))
        with torch.no_grad():
            f.model.weight.add_(1)
        remember_memory(f.model, key, self.memory, 48)
        self.assertIsNone(cached_memory(f.model, f.key(), 48))

    def test_cache_is_bounded_and_does_not_retain_model(self):
        f, key = self.fixture, self.fixture.key()
        for index in range(_MAX_ENTRIES + 1):
            remember_memory(f.model, replace(key, geometry=str(index)), self.memory, 48)
        self.assertEqual(len(_CACHE[f.model]), _MAX_ENTRIES)
        self.assertIsNone(cached_memory(f.model, replace(key, geometry="0"), 48))
        temporary = ProbeFixture()
        owner = weakref.ref(temporary.model)
        remember_memory(temporary.model, temporary.key(), self.memory, 48)
        del temporary
        gc.collect()
        self.assertIsNone(owner())


class LatencyPolicyTests(unittest.TestCase):
    def test_latency_miss_never_constructs_or_executes_probes(self):
        f = ProbeFixture()
        calc = f.calculator("latency")
        factory = Mock(side_effect=AssertionError("Short task must not create CUDA probes"))
        with patch.object(MeasuredMemoryModel, "measure", side_effect=AssertionError("No calibration")):
            memory = calc._system_memory(f.atoms, f.config, {"edges": 48}, factory)
        self.assertFalse(memory.valid)
        self.assertIn("latency policy", memory.reason)
        factory.assert_not_called()
        self.assertEqual(calc.memory_probe_summary(), {"auto_policy": "latency", "cache_hits": 0,
                         "measurements": 0, "skipped": 1, "last": "latency_cache_miss"})

    def test_throughput_calibrates_once_then_latency_reuses_same_bound(self):
        f = ProbeFixture()
        calc, memory = f.calculator("throughput"), fitted()
        factory = Mock(return_value={})
        with patch.object(MeasuredMemoryModel, "measure", return_value=memory) as measure:
            self.assertIs(calc._system_memory(f.atoms, f.config, {"edges": 48}, factory), memory)
            # A new calculator over the SAME adapter/model can use the calibration.
            short = f.calculator("latency")
            self.assertIs(short._system_memory(f.atoms.copy(), f.config, {"edges": 48}, factory), memory)
            self.assertIs(short._system_memory(f.atoms.copy(), f.config, {"edges": 64}, factory), memory)
            self.assertFalse(short._system_memory(f.atoms, f.config, {"edges": 65}, factory).valid)
        measure.assert_called_once()
        factory.assert_called_once()
        self.assertEqual(short.memory_probe_summary()["cache_hits"], 2)
        self.assertEqual(short.memory_probe_summary()["measurements"], 0)

    def test_low_level_probe_cache_is_opt_in_and_never_retries_invalid_fits_as_hits(self):
        f, memory = ProbeFixture(), fitted()
        with patch.object(MeasuredMemoryModel, "measure", return_value=memory) as measure:
            for _ in range(2):
                self.assertIs(measured_memory(MeasuredMemoryModel, {}, 48, "cpu",
                              cache_owner=f.model, cache_key=f.key()), memory)
            self.assertEqual(measure.call_count, 1)
            measured_memory(MeasuredMemoryModel, {}, 48, "cpu")
            self.assertEqual(measure.call_count, 2)

    def test_invalid_public_policy_fails_before_loading(self):
        with patch("fastmd.interfaces.calculator.load_adapter") as load:
            with self.assertRaisesRegex(ValueError, "auto_policy"):
                MLIPCalculator(auto_policy="unknown", config=InferenceConfig(report=False))
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
