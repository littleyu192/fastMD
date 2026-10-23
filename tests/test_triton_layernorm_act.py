"""Fused LayerNorm + activation (triton_layernorm_act): gradients against torch.

The kernel computes in FP32 whatever the input dtype, so first-order results are
compared with an FP64 torch reference at FP32 tolerance, and the FP64 gradcheck
uses a finite-difference step above FP32 resolution. Double backward (force-loss
training differentiates forces, readout.py create_graph=is_training) and
non-contiguous weight/bias must match torch. The gated LayerNorm epilogue
(triton_gated_ln, 1-row and 2-D tile kernels) is checked the same way at first
order, plus bitwise run-to-run reproducibility; with the preceding linears'
biases folded in (D13a) also their gradients, and bitwise against adding the
biases to the inputs first (the FP32 sum the GEMM's bias epilogue rounds).
The fused narrow tail (triton_gated_tail: both second linears and the gated
LayerNorm, C2) against the FP64 reference and the unfused FP32 path, also on
strided inputs, writing its linear outputs over its inputs, and replayed from a
CUDA graph.
"""
from __future__ import annotations

from itertools import product
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from fastmd.models.matris.op import triton_layernorm_act as ln

ACTS = {ln.ACT_SILU: F.silu, ln.ACT_SIGMOID: torch.sigmoid, ln.ACT_NONE: lambda y: y}


def _reference(x, weight, bias, act):
    return ACTS[act](F.layer_norm(x, (x.shape[-1],), weight, bias, 1e-5))


def _fused(act):
    return lambda x, weight, bias: ln.fused_ln_act(x, weight, bias, 1e-5, act)


def _first_order(fn, inputs, seed, weight_grads=True, view=lambda t: t):
    """Output and gradients; ``view`` maps the weight/bias leaves to kernel operands."""
    leaves = [inputs[0].clone().requires_grad_(True),
              *(t.clone().requires_grad_(weight_grads) for t in inputs[1:])]
    out = fn(leaves[0], *(view(t) for t in leaves[1:]))
    wrt = leaves if weight_grads else leaves[:1]
    return (out.detach(), *torch.autograd.grad(out, wrt, seed))


def _assert_rel(test, actual, expected, tol):
    """Max error relative to the reference's largest magnitude (FP32 kernel scale)."""
    for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
        with test.subTest(output=index):
            err = (a.double() - e.double()).abs().max() / e.double().abs().max()
            test.assertLess(float(err), tol)


@unittest.skipUnless(torch.cuda.is_available(), "requires coordinator-assigned idle CUDA device")
class LayerNormActGPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_first_order_matches_fp64_reference(self):
        rows = 777
        # Weight/bias as contiguous vectors or stride-2 views of twice as long ones.
        layouts = {"contiguous": (1, lambda t: t), "stride 2": (2, lambda t: t[::2])}
        for act, dim, weight_grads, (scale, shift), layout in product(
                ACTS, (16, 64, 100, 128), (True, False), ((3.0, 0.5), (1e-2, 0.0)), layouts):
            with self.subTest(act=act, dim=dim, weight_grads=weight_grads, scale=scale, weight_layout=layout):
                x = torch.randn(rows, dim, device="cuda") * scale + shift
                if scale < 1:  # variance ~ EPS, plus constant rows (variance 0)
                    x[:32] = 0.25
                stride, view = layouts[layout]
                w, b = torch.randn(stride * dim, device="cuda"), torch.randn(stride * dim, device="cuda")
                seed = torch.randn(rows, dim, device="cuda")
                # First order stays in the Triton kernels; the torch path is
                # only for create_graph backward.
                with patch.object(ln, "_ln_act_torch", side_effect=AssertionError("torch path used")):
                    got = _first_order(_fused(act), (x, w, b), seed, weight_grads, view)
                ref = _first_order(lambda *a: _reference(*a, act),
                                   (x.double(), w.double(), b.double()), seed.double(),
                                   weight_grads, view)
                _assert_rel(self, got, ref, 1e-5)

    def test_double_backward_matches_torch(self):
        # E = sum(c * f(x)) + 0.5 |x|^2; F = dE/dx with create_graph; L = sum(r * F).
        rows, dim = 300, 32
        x = torch.randn(rows, 2 * dim, device="cuda")[:, ::2]  # non-contiguous input
        w, b = torch.randn(dim, device="cuda"), torch.randn(dim, device="cuda")
        c, r = torch.randn(rows, dim, device="cuda"), torch.randn(rows, dim, device="cuda")

        def second_order(fn, dtype):
            leaves = [t.detach().to(dtype).requires_grad_(True) for t in (x, w, b)]
            energy = (fn(*leaves) * c.to(dtype)).sum() + 0.5 * leaves[0].pow(2).sum()
            (force,) = torch.autograd.grad(energy, leaves[0], create_graph=True)
            return torch.autograd.grad((force * r.to(dtype)).sum(), leaves, allow_unused=True)

        for act in ACTS:
            with self.subTest(act=act):
                got = second_order(_fused(act), torch.float32)
                ref = second_order(lambda *a: _reference(*a, act), torch.float64)
                for name, g, e in zip(("x", "weight", "bias"), got, ref):
                    if e is None:  # e.g. the bias does not reach dE/dx without an activation
                        self.assertTrue(g is None or not g.any(), name)
                        continue
                    self.assertIsNotNone(g, name)
                    _assert_rel(self, (g,), (e,), 1e-5)

    def test_fp64_gradcheck_and_gradgradcheck(self):
        # x is rounded to FP32 in the kernel: a 1e-6 finite-difference step would only
        # measure that rounding, so the first-order check uses a step above FP32 resolution.
        # One program per row: 8 rows exercise the same kernels and the cross-row
        # weight/bias reduction as more, at a fraction of the finite-difference cost.
        inputs = (torch.randn(8, 16, device="cuda", dtype=torch.float64, requires_grad=True),
                  torch.randn(16, device="cuda", dtype=torch.float64, requires_grad=True),
                  torch.randn(16, device="cuda", dtype=torch.float64, requires_grad=True))
        for act in ACTS:
            with self.subTest(act=act):
                fn = lambda *a: ln._LNAct.apply(*a, 1e-5, act)
                self.assertTrue(torch.autograd.gradcheck(fn, inputs, eps=1e-3, atol=1e-3, rtol=1e-3))
                self.assertTrue(torch.autograd.gradgradcheck(fn, inputs))


@unittest.skipUnless(torch.cuda.is_available(), "requires coordinator-assigned idle CUDA device")
class GatedLayerNormGPUTest(unittest.TestCase):
    def test_gated_ln_matches_fp64_reference_and_is_deterministic(self):
        from fastmd.models.matris.op import triton_gated_ln as gated

        torch.manual_seed(1234)
        rows = 1031  # not a multiple of any 2-D tile height
        for dim, residual, tile_2d, bias in product((17, 32, 128), (False, True), (False, True), (False, True)):
            with self.subTest(dim=dim, residual=residual, tile_2d=tile_2d, bias=bias):
                inputs = [torch.randn(rows, dim, device="cuda") * 3 + 0.5 for _ in range(3)]
                inputs[0][:5] = 0.25  # constant rows: variance 0
                params = [torch.randn(dim, device="cuda") for _ in range(5)]
                biases = [torch.randn(dim, device="cuda") for _ in range(2)] if bias else []
                if bias:
                    inputs[0][:5] -= biases[0]  # still constant rows once the bias is added
                seed = torch.randn(rows, dim, device="cuda")

                def run(dtype, fused, fold=True):
                    core, gate, res = (t.detach().to(dtype).requires_grad_(True) for t in inputs)
                    cw, cb, gw, gb, rw = (t.to(dtype) for t in params)
                    lbs = [t.detach().to(dtype).requires_grad_(True) for t in biases]
                    x_core, x_gate, folded = core, gate, ()
                    if bias and fused and fold:
                        folded = tuple(lbs)  # D13a: the kernel adds them
                    elif bias:
                        x_core, x_gate = core + lbs[0], gate + lbs[1]
                    if fused:
                        with patch.object(gated, "_TILE_2D", tile_2d):
                            out = (gated.fused_gated_ln_residual(x_core, x_gate, cw, cb, gw, gb, res, rw,
                                                                 1e-5, 1e-5, *folded) if residual
                                   else gated.fused_gated_ln(x_core, x_gate, cw, cb, gw, gb, 1e-5, 1e-5, *folded))
                    else:
                        out = (F.silu(F.layer_norm(x_core, (dim,), cw, cb, 1e-5))
                               * torch.sigmoid(F.layer_norm(x_gate, (dim,), gw, gb, 1e-5)))
                        out = out + rw * res if residual else out
                    leaves = (core, gate, res) if residual else (core, gate)
                    return (out.detach(), *torch.autograd.grad(out, (*leaves, *lbs), seed.to(dtype)))

                got = run(torch.float32, True)
                _assert_rel(self, got, run(torch.float64, False), 1e-5)
                for index, (first, second) in enumerate(zip(got, run(torch.float32, True))):
                    self.assertTrue(torch.equal(first, second), f"output {index} not reproducible")
                if bias:
                    # The folded kernels see the same FP32 inputs as with the biases added first:
                    # bitwise outputs and input gradients (the bias gradients sum in another order).
                    unfolded = run(torch.float32, True, fold=False)
                    for index, (first, second) in enumerate(zip(got[:-2], unfolded[:-2])):
                        self.assertTrue(torch.equal(first, second), f"output {index} differs from unfolded")
                    _assert_rel(self, got[-2:], unfolded[-2:], 1e-6)

    def test_gated_tail_matches_fp64_unfused_and_capture(self):
        from fastmd.precision import fp32_gemm_mode, matmul_precision_scope

        # 3xTF32 dots where cuBLAS runs IEEE FP32 GEMMs, TF32 where it runs TF32 (~3e-4 relative).
        for precision, tol in (("highest", 1e-5), ("high", 5e-3)):
            with self.subTest(precision=precision), matmul_precision_scope(precision):
                if precision == "high" and fp32_gemm_mode() != "tf32":
                    continue  # e.g. NVIDIA_TF32_OVERRIDE=0
                self._gated_tail_checks(tol)

    def _gated_tail_checks(self, tol):
        from fastmd.models.matris.op import triton_gated_ln as gated
        from fastmd.models.matris.op import triton_gated_tail as tail

        torch.manual_seed(4321)
        rows = 1031
        for k, dim, bias in ((16, 32, False), (16, 32, True), (32, 32, False), (32, 32, True), (32, 16, True),
                             (32, 64, True)):
            with self.subTest(k=k, dim=dim, bias=bias):
                # The halves of one [rows, 2k] buffer, as a Cat3/Cat4 preact's activations may be.
                hidden = torch.randn(rows, 2 * k, device="cuda")
                weights = [torch.randn(dim, k, device="cuda") * k ** -0.5 for _ in range(2)]
                biases = [torch.randn(dim, device="cuda") * 0.3 for _ in range(2)] if bias else [None, None]
                norms = [torch.randn(dim, device="cuda") * 0.3 + (1 - i % 2) for i in range(4)]
                seed = torch.randn(rows, dim, device="cuda")

                def run(dtype, impl):
                    leaf = hidden.detach().to(dtype).requires_grad_(True)
                    core_h, gate_h = leaf[:, :k], leaf[:, k:]
                    w = [t.to(dtype) for t in weights]
                    b = [None if t is None else t.to(dtype) for t in biases]
                    cw, cb, gw, gb = (t.to(dtype) for t in norms)
                    if impl == "fused":
                        self.assertTrue(tail.gated_tail_supported(core_h, gate_h, *w))
                        out = tail.fused_gated_tail(core_h, gate_h, w[0], b[0], w[1], b[1], cw, cb, gw, gb)
                    else:
                        core, gate = F.linear(core_h, w[0], b[0]), F.linear(gate_h, w[1], b[1])
                        out = (gated.fused_gated_ln(core, gate, cw, cb, gw, gb) if impl == "unfused" else
                               F.silu(F.layer_norm(core, (dim,), cw, cb, 1e-5))
                               * torch.sigmoid(F.layer_norm(gate, (dim,), gw, gb, 1e-5)))
                    return (out.detach(), *torch.autograd.grad(out, leaf, seed.to(dtype)))

                got = run(torch.float32, "fused")
                _assert_rel(self, got, run(torch.float64, "reference"), tol)
                _assert_rel(self, got, run(torch.float32, "unfused"), tol)
                for index, (first, second) in enumerate(zip(got, run(torch.float32, "fused"))):
                    self.assertTrue(torch.equal(first, second), f"output {index} not reproducible")
                with torch.no_grad():  # nothing to save: the linear outputs are not stored
                    no_grad = tail.fused_gated_tail(hidden[:, :k], hidden[:, k:], weights[0], biases[0], weights[1],
                                                    biases[1], *norms)
                self.assertTrue(torch.equal(no_grad, got[0]))
                if k == dim:  # in place (inputs the caller no longer uses): the same results, inputs overwritten
                    leaf = hidden.detach().clone().requires_grad_(True)
                    core_in, gate_in = leaf[:, :k].clone(), leaf[:, k:].clone()
                    version = core_in._version
                    out = tail.fused_gated_tail(core_in, gate_in, weights[0], biases[0], weights[1], biases[1],
                                                *norms, inplace=True)
                    (grad,) = torch.autograd.grad(out, leaf, seed)
                    self.assertGreater(core_in._version, version)
                    self.assertTrue(torch.equal(out, got[0]) and torch.equal(grad, got[1]))
                    # Under other saved-tensor hooks (a layer checkpoint's) the inputs are left alone.
                    core_in, gate_in = leaf[:, :k].clone(), leaf[:, k:].clone()
                    version = core_in._version
                    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                        out = tail.fused_gated_tail(core_in, gate_in, weights[0], biases[0], weights[1], biases[1],
                                                    *norms, inplace=True)
                    (grad,) = torch.autograd.grad(out, leaf, seed)
                    self.assertEqual(core_in._version, version)
                    self.assertTrue(torch.equal(out, got[0]) and torch.equal(grad, got[1]))

                # Forward and backward captured in a CUDA graph replay the eager results.
                leaf = hidden.detach().clone().requires_grad_(True)
                args = (weights[0], biases[0], weights[1], biases[1], *norms)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):  # warm up (Triton compiles) outside the capture
                    tail.fused_gated_tail(leaf[:, :k], leaf[:, k:], *args).backward(seed)
                torch.cuda.current_stream().wait_stream(stream)
                leaf.grad = None
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = tail.fused_gated_tail(leaf[:, :k], leaf[:, k:], *args)
                    (grad,) = torch.autograd.grad(out, leaf, seed)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out, got[0]))
                self.assertTrue(torch.equal(grad, got[1]))


if __name__ == "__main__":
    unittest.main()
