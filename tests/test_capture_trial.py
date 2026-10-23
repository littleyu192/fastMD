"""CPU tests of capture admission, replay overhead and same-call fallback."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from ase import Atoms

from fastmd.adapters.negotiate import Decision
from fastmd.config import InferenceConfig, resolve_config
from fastmd.execution._capture_trial import (
    CaptureTrialRejected, guarded_bucket_captures, new_trial_stats, trial_estimate,
)
from fastmd.interfaces.calculator import MLIPCalculator


MIB = 2**20


class Runner:
    n_dummy = 0

    def __init__(self, error=None):
        self.calls = 0
        self.cache = {}
        self.error = error

    def _capture(self, pairs, triplets, sample):
        self.calls += 1
        self.cache[(pairs, triplets)] = 'captured buffers'
        if self.error is not None:
            raise self.error


def cuda_fixture(*, budget=1024 * MIB, peak=8 * MIB):
    stack = ExitStack()
    stack.enter_context(patch('fastmd.adapters.negotiate.capture_budget', return_value=budget))
    stack.enter_context(patch('fastmd.execution.cuda_graph._demand', return_value=(8, 32, 0)))
    stack.enter_context(patch.object(torch.cuda, 'memory_allocated', return_value=0))
    stack.enter_context(patch.object(torch.cuda, 'memory_reserved', return_value=0))
    stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=peak))
    stack.enter_context(patch.object(torch.cuda, 'max_memory_reserved', return_value=peak))
    stack.enter_context(patch.object(torch.cuda, 'reset_peak_memory_stats'))
    stack.enter_context(patch.object(torch.cuda, 'synchronize'))
    return stack


class CaptureTrialTests(unittest.TestCase):
    def test_missing_eager_measurement_cannot_admit_a_capture(self):
        for peak in (None, 0, float('nan'), float('inf')):
            self.assertIsNone(trial_estimate(peak, atoms=8, pairs=32, triplets=0,
                                          capacity_pairs=64, capacity_triplets=0, dummy_atoms=0))

    def test_current_budget_rejection_performs_no_capture(self):
        runner, stats = Runner(), new_trial_stats()
        with cuda_fixture(budget=MIB), guarded_bucket_captures(
                runner, device='cuda', eager_peak=MIB, stats=stats):
            with self.assertRaisesRegex(CaptureTrialRejected, 'headroom'):
                runner._capture(64, 0, None)
        self.assertEqual(runner.calls, 0)
        self.assertEqual(stats['attempts'], 0)
        self.assertEqual(stats['budget_rejections'], 1)
        self.assertNotIn('_capture', vars(runner))

    def test_capture_and_growth_are_measured_separately_but_replay_is_free(self):
        runner, stats = Runner(), new_trial_stats()
        with cuda_fixture(), patch.object(torch.cuda, 'synchronize') as sync:
            for capacity in (64, 96):
                with guarded_bucket_captures(runner, device='cuda', eager_peak=MIB, stats=stats):
                    runner._capture(capacity, 0, None)
            self.assertEqual(sync.call_count, 4)
            # A normal cached replay does not invoke _capture or add syncs.
            with guarded_bucket_captures(runner, device='cuda', eager_peak=MIB, stats=stats):
                self.assertEqual(runner.cache[(64, 0)], 'captured buffers')
            self.assertEqual(sync.call_count, 4)
        self.assertEqual(stats['attempts'], 2)
        self.assertEqual(stats['successes'], 2)
        self.assertEqual([r['capacity_edges'] for r in stats['records']], [128, 192])
        self.assertTrue(all(r['peak_increment_bytes'] == 8 * MIB for r in stats['records']))
        self.assertGreaterEqual(stats['total_seconds'], 0)

    def test_growth_rechecks_changed_free_memory(self):
        runner, stats = Runner(), new_trial_stats()
        with cuda_fixture(), patch('fastmd.adapters.negotiate.capture_budget', side_effect=[1024 * MIB, MIB]):
            with guarded_bucket_captures(runner, device='cuda', eager_peak=MIB, stats=stats):
                runner._capture(64, 0, None)
                with self.assertRaises(CaptureTrialRejected):
                    runner._capture(96, 0, None)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(stats['budget_rejections'], 1)

    def test_actual_peak_above_budget_is_rejected_after_capture(self):
        runner, stats = Runner(), new_trial_stats()
        with cuda_fixture(budget=16 * MIB, peak=20 * MIB), guarded_bucket_captures(
                runner, device='cuda', eager_peak=MIB, stats=stats):
            with self.assertRaisesRegex(CaptureTrialRejected, 'exceeded'):
                runner._capture(64, 0, None)
        self.assertEqual(stats['successes'], 0)
        self.assertEqual(stats['failures'], 1)
        self.assertEqual(stats['records'][0]['peak_increment_bytes'], 20 * MIB)

    def test_oom_and_budget_failure_return_current_eager_result_and_discard_graph(self):
        for error, budget, peak in ((torch.OutOfMemoryError('test OOM'), 1024 * MIB, 8 * MIB),
                                    (None, 16 * MIB, 20 * MIB), (None, MIB, 8 * MIB)):
            with self.subTest(error=error, budget=budget):
                runner, stats = Runner(error), new_trial_stats()
                calculator = object.__new__(MLIPCalculator)
                calculator._auto = {'eager_peak': MIB}
                calculator._auto_before_call = Mock()
                calculator._auto_after_call = Mock()
                calculator._graph_runner = runner
                calculator._trial_capture_enabled = True
                calculator._capture_trial_stats = stats
                calculator._model_device = lambda: 'cuda'
                modes = []

                def set_mode(mode, reason):
                    modes.append(mode)
                    runner.cache.clear()
                    calculator._graph_runner = None

                calculator._set_execution = set_mode
                expected = np.array([[1., 2., 3.]])

                def predict(structure, atoms=None):
                    if calculator._graph_runner is not None:
                        runner._capture(64, 0, None)
                        raise AssertionError('Failed graph must not supply a prediction')
                    return {'e': 7., 'f': expected.copy()}

                calculator._predict_structure_scoped = predict
                with cuda_fixture(budget=budget, peak=peak), self.assertWarnsRegex(RuntimeWarning, 'eager'):
                    result = calculator._predict_structure(None, atoms=object())
                np.testing.assert_array_equal(result['f'], expected)
                self.assertEqual(result['e'], 7.)
                expected[:] = 0
                self.assertEqual(result['f'][0, 0], 1.)
                self.assertEqual(modes, ['eager'])
                self.assertFalse(runner.cache)
                self.assertIsNone(calculator._graph_runner)
                self.assertTrue(calculator._auto['decided'])
                self.assertNotIn('_capture', vars(runner))

    def test_non_capture_error_propagates_and_restores_runner(self):
        runner, stats = Runner(ValueError('bad model')), new_trial_stats()
        with cuda_fixture(), guarded_bucket_captures(runner, device='cuda', eager_peak=MIB, stats=stats):
            with self.assertRaisesRegex(ValueError, 'bad model'):
                runner._capture(64, 0, None)
        self.assertEqual(stats['failures'], 1)
        self.assertNotIn('_capture', vars(runner))

    def test_successful_calculator_trial_records_cost_and_returns_graph_result(self):
        runner, stats = Runner(), new_trial_stats()
        calculator = object.__new__(MLIPCalculator)
        calculator._auto = {'eager_peak': MIB}
        calculator._graph_runner = runner
        calculator._trial_capture_enabled = True
        calculator._capture_trial_stats = stats
        calculator._model_device = lambda: 'cuda'
        calculator._engine_plan = (None, SimpleNamespace(decisions=[Decision('execution', 'model_graph', 'pending')]))

        def run():
            runner._capture(64, 0, None)
            return {'e': 7.}

        with cuda_fixture():
            result, ran = calculator._auto_capture(run)
        self.assertTrue(ran)
        self.assertEqual(result, {'e': 7.})
        self.assertEqual(stats['successes'], 1)
        self.assertIn('captured in', calculator._auto['reason'])
        self.assertNotIn('pending', calculator._engine_plan[1].decisions[0].reason)


class NegotiationTrialTests(unittest.TestCase):
    @staticmethod
    def calculator(policy='latency_trial', *, memory_valid=False):
        calc = object.__new__(MLIPCalculator)
        calc.auto_policy = policy
        calc._trial_capture_enabled = False
        calc.model = torch.nn.Linear(1, 1)
        calc._adapter = SimpleNamespace(module=calc.model)
        calc._inference_config = resolve_config(InferenceConfig(execution='auto', report=False))
        calc._engine_plan = (calc.adapter, object())
        calc._edges_only, calc._line_graph_batches = True, False
        calc._skin_static = 'trial uses bucket captures'
        calc._model_device = lambda: torch.device('cuda')
        calc._memory_probe_stats = {'last': 'cache_hit' if memory_valid else 'latency_cache_miss'}
        calc._system_memory = Mock(return_value=SimpleNamespace(valid=memory_valid))
        calc._per_call_graph = Mock(return_value=SimpleNamespace(
            edge_index=torch.zeros((64, 2), dtype=torch.int32), n_real=8,
            cell=torch.eye(3), species=torch.ones(8, dtype=torch.int32), positions=torch.zeros((8, 3))))
        return calc

    @staticmethod
    def plan(mode):
        plan = SimpleNamespace(decisions=[Decision('execution', mode, 'capability result')])
        plan.decision = lambda name: next(item for item in plan.decisions if item.id == name)
        return plan

    def test_trial_keeps_explicit_graph_capability_negotiation(self):
        atoms = Atoms('H8', positions=np.zeros((8, 3)), cell=[6, 6, 6], pbc=True)
        for chosen in ('model_graph', 'eager'):
            with self.subTest(chosen=chosen):
                calc = self.calculator()
                with patch('fastmd.graph.fixed_capacity_builder._fixed_builder_available', True), \
                     patch('fastmd.adapters.negotiate.can_capture', return_value=True), \
                     patch('fastmd.adapters.negotiate.renegotiate_execution',
                           side_effect=[self.plan('eager'), self.plan(chosen)]) as negotiate:
                    mode, reason = calc._negotiated_decide(atoms)
                self.assertEqual(mode, chosen)
                self.assertEqual(calc._trial_capture_enabled, chosen == 'model_graph')
                self.assertEqual(negotiate.call_count, 2)
                self.assertEqual(negotiate.call_args.args[2].execution, 'model_graph')
                self.assertEqual(calc._system_memory.call_count, 1)

    def test_cache_hit_preserves_bound_path_and_previous_latency_does_not_try_capture(self):
        atoms = Atoms('H8', positions=np.zeros((8, 3)), cell=[6, 6, 6], pbc=True)
        for policy, valid, chosen in (('latency', False, 'eager'), ('latency_trial', True, 'model_graph')):
            with self.subTest(policy=policy, memory_valid=valid):
                calc = self.calculator(policy, memory_valid=valid)
                with patch('fastmd.graph.fixed_capacity_builder._fixed_builder_available', True), \
                     patch('fastmd.adapters.negotiate.can_capture', return_value=True), \
                     patch('fastmd.adapters.negotiate.renegotiate_execution', return_value=self.plan(chosen)) as negotiate:
                    mode, _ = calc._negotiated_decide(atoms)
                self.assertEqual(mode, chosen)
                self.assertEqual(negotiate.call_count, 1)
                self.assertEqual(calc._trial_capture_enabled, policy == 'latency_trial')

    def test_odd_directed_edge_capacity_matches_runner_demand(self):
        from fastmd.engine.capacity_policy import CapturePolicy
        from fastmd.execution.cuda_graph import _demand
        from fastmd.adapters.protocol import GraphBatch

        atoms = Atoms('H8', positions=np.zeros((8, 3)), cell=[6, 6, 6], pbc=True)
        calc = self.calculator()
        # Pick an edge count on a bucket boundary so floor/ceil changes the
        # actual admitted capacity, not just an intermediate integer.
        policy = CapturePolicy.from_config(calc._inference_config.capacity)
        pad = calc._inference_config.capacity.min_pad_u
        count = next(n for n in range(1, 4096, 2)
                     if policy.bucket_key(n // 2 + pad, 0) != policy.bucket_key((n + 1) // 2 + pad, 0))
        batch = calc._per_call_graph.return_value
        batch.edge_index = torch.zeros((count, 2), dtype=torch.int32)
        graph = object.__new__(GraphBatch)
        graph.edge_index, graph.n_real = batch.edge_index, batch.n_real
        expected = 2 * policy.bucket_key(_demand(graph)[1] + pad, 0)[0]
        with patch('fastmd.graph.fixed_capacity_builder._fixed_builder_available', True), \
             patch('fastmd.adapters.negotiate.can_capture', return_value=True), \
             patch('fastmd.adapters.negotiate.renegotiate_execution', return_value=self.plan('eager')) as negotiate:
            calc._negotiated_decide(atoms)
        self.assertGreaterEqual(expected, count)
        self.assertEqual(calc._system_memory.call_args.args[2]['edges'], expected)
        self.assertTrue(all(call.kwargs['capacity']['edges'] == expected for call in negotiate.call_args_list))


if __name__ == '__main__':
    unittest.main()
