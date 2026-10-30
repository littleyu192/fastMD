"""Cached launches respect both the optional-hook and hook-collection Triton APIs."""
from types import SimpleNamespace
import unittest
from unittest import mock
import torch

from fastmd.engine.kernels import triton_wide_gemm as wg


class WideGemmHooksTest(unittest.TestCase):
    def test_shared_memory_retry_keeps_the_successful_stage_count(self):
        compiled = SimpleNamespace(run=mock.Mock(), function=object(), packed_metadata=object(),
                                   metadata=SimpleNamespace(num_stages=1))
        standard = mock.Mock(side_effect=[wg.OutOfResources(264232, 232448, 'shared memory'),
                                          wg.OutOfResources(240000, 232448, 'shared memory'), compiled, compiled])
        fn = mock.MagicMock()
        fn.__getitem__.return_value = standard
        args = (SimpleNamespace(base=SimpleNamespace(device=SimpleNamespace(index=0))),)
        hooks = SimpleNamespace(launch_enter_hook=lambda *_: None, launch_exit_hook=None)
        with mock.patch.object(wg, '_COMPILED', {}), \
             mock.patch.object(wg.triton, 'knobs', SimpleNamespace(runtime=hooks)):
            wg._run(fn, 2, args, (('K', 128),), 8, 3)
            wg._run(fn, 2, args, (('K', 128),), 8, 3)
        self.assertEqual([call.kwargs['num_stages'] for call in standard.call_args_list], [3, 2, 1, 1])
        for stages, resource in ((1, 'shared memory'), (3, 'registers')):
            standard.side_effect = wg.OutOfResources(10, 5, resource)
            with mock.patch.object(wg, '_COMPILED', {}), self.assertRaises(wg.OutOfResources):
                wg._run(fn, 2, args, (('K', 128),), 8, stages)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cached_gpu_launch_matches_standard_launch(self):
        # Both kernel variants, both precisions, varying row count under one cache key.
        # The hook fix changes dispatch only: the normal Triton launch is the exact reference.
        torch.manual_seed(31)
        for k, n in ((128, 128), (128, 512), (256, 128), (512, 128), (96, 64)):
            w = torch.randn(n, k, device='cuda') / k ** 0.5
            split = wg._split(w, False)
            bias = torch.randn(n, device='cuda')
            for precision in ('tf32x3', 'tf32'):
                for rows in (77, 70001):
                    with self.subTest(k=k, n=n, precision=precision, rows=rows):
                        x = torch.randn(rows, k, device='cuda')
                        wg._COMPILED.clear()
                        standard = wg._launch(x, split, precision, bias=bias)
                        cached = wg._launch(x, split, precision, bias=bias)
                        torch.testing.assert_close(cached, standard, rtol=0, atol=0)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            replayed = wg._launch(x, split, precision, bias=bias)
                        graph.replay()
                        torch.testing.assert_close(replayed, standard, rtol=0, atol=0)
                        # The accumulating backward has a different shared-memory layout.
                        # Forward-only coverage misses the H100 resource failure.
                        for with_bias in (None, bias):
                            c = torch.randn(rows, n, device='cuda')
                            reference = x.double() @ w.double().t() + c.double()
                            if with_bias is not None:
                                reference += with_bias.double()
                            got = wg._launch(x, split, precision, bias=with_bias, c=c, out=c)
                            rms = ((got.double()-reference).square().mean()/reference.square().mean()).sqrt()
                            self.assertLess(float(rms), 1e-6 if precision == 'tf32x3' else 1e-3)

    def test_cached_launch_preserves_profiler_hooks(self):
        callback = lambda *_: None
        cases = [(None, None, False), (callback, None, True), (None, callback, True),
                 (SimpleNamespace(calls=[]), SimpleNamespace(calls=[]), False),
                 (SimpleNamespace(calls=[callback]), SimpleNamespace(calls=[]), True),
                 (SimpleNamespace(calls=[]), SimpleNamespace(calls=[callback]), True)]
        for enter, leave, profiled in cases:
            with self.subTest(enter=enter, leave=leave):
                compiled = SimpleNamespace(run=mock.Mock(), function=object(), packed_metadata=object())
                standard = mock.Mock(return_value=compiled)
                fn = mock.MagicMock()
                fn.__getitem__.return_value = standard
                args = (SimpleNamespace(base=SimpleNamespace(device=SimpleNamespace(index=0))),)
                hooks = SimpleNamespace(launch_enter_hook=enter, launch_exit_hook=leave)
                with mock.patch.object(wg, '_COMPILED', {}), \
                     mock.patch.object(wg.triton, 'knobs', SimpleNamespace(runtime=hooks)), \
                     mock.patch.object(wg.torch._C, '_cuda_getCurrentRawStream', return_value=7):
                    wg._run(fn, 2, args, (('K', 128),), 8, 2)
                    self.assertEqual(standard.call_count, 1)
                    compiled.run.assert_not_called()
                    wg._run(fn, 2, args, (('K', 128),), 8, 2)
                    self.assertEqual(standard.call_count, 2 if profiled else 1)
                    self.assertEqual(compiled.run.call_count, 0 if profiled else 1)


if __name__ == '__main__':
    unittest.main()
