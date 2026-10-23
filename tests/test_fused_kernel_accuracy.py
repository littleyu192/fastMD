"""Fused/custom kernels that had no accuracy test: forward and VJP against torch.

One parametrized test per kernel family, each against a plain-torch reference
and each proving that the kernel itself ran (the compiled lowerings return
None instead of raising, so a silent fallback would otherwise pass):

* Inductor lowerings (compiled_lowerings): line envelope and residual add.
* Jiterator SiLU / sigmoid (fused_silu, fused_sigmoid), first and second order.
* Paired segment attention with the identity CSR on side b, the single-GPU
  production layout (source perm CSR, sorted target).
* The line-graph materialize Triton kernel against its torch fallback.
* enable_compile: the Inductor-compiled attention/refinement layers against
  the uncompiled layers (EnableCompileLayersGPUTest).
"""
from __future__ import annotations

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from fastmd.graph import gpu_graph_builder as gb
from fastmd.models.matris.op import compiled_lowerings as cl
from fastmd.models.matris.op import fuse_sigmoid_op, fused_silu_op
from fastmd.models.matris.op.triton_segment_attention import paired_segment_attention

_CUDA = torch.cuda.is_available()
# FP32 kernels against an FP64 reference, as in test_triton_layernorm_act.
_FP32_TOL = 1e-5


def _assert_rel(test, named, tol):
    """Max error relative to the reference's largest magnitude, per output."""
    for name, actual, expected in named:
        with test.subTest(output=name):
            test.assertEqual(tuple(actual.shape), tuple(expected.shape))
            actual, expected = actual.detach().double(), expected.detach().double()
            err = (actual - expected).abs().max() / expected.abs().max()
            test.assertLess(float(err), tol)  # NaN fails too


# 1. Inductor lowerings -------------------------------------------------------


@unittest.skipUnless(_CUDA, "requires coordinator-assigned idle CUDA device")
class CompiledLoweringsGPUTest(unittest.TestCase):
    """Forward and VJP artifacts of the production compile mode vs eager formulas."""

    def setUp(self):
        torch.manual_seed(1234)
        # Opt in for this test only and leave no compiled artifact behind.
        self.enterContext(patch.object(cl, "COMPILED_LOWERINGS_ENABLED", True))
        for artifacts in (cl._ENVELOPE_ARTIFACTS, cl._RESIDUAL_ARTIFACTS):
            self.enterContext(patch.dict(artifacts._compiled))

    def test_line_envelope_and_residual_add_match_eager(self):
        # Rows, width, triplets; the int32 indices are strided columns of a
        # [T, 5] line-graph table, the int64 ones contiguous. Indices repeat, so
        # the VJP accumulates.
        for rows, width, triplets, index_dtype in ((37, 32, 211, torch.int64), (53, 64, 301, torch.int32)):
            with self.subTest(op="line_envelope", rows=rows, width=width, index_dtype=index_dtype):
                base = torch.randn(rows, width, device="cuda")
                table = torch.randint(0, rows, (triplets, 5), device="cuda", dtype=index_dtype)
                source, target = table[:, 1], table[:, 3]
                if index_dtype is torch.int64:
                    source, target = source.contiguous(), target.contiguous()
                seed = torch.randn(triplets, width, device="cuda")
                leaf, ref_leaf = base.clone().requires_grad_(True), base.double().requires_grad_(True)
                out = cl.compiled_line_envelope(leaf, source, target)
                self.assertIsNotNone(out, "compiled line envelope fell back")
                ref = ref_leaf[source.long()] * ref_leaf[target.long()]
                _assert_rel(self, (("out", out, ref),
                                   ("grad_base", torch.autograd.grad(out, leaf, seed)[0],
                                    torch.autograd.grad(ref, ref_leaf, seed.double())[0])), _FP32_TOL)

        # The second residual is a row-strided view (packed[:, :D]).
        for rows, width, strided in ((37, 32, False), (53, 64, True)):
            with self.subTest(op="residual_add", rows=rows, width=width, strided=strided):
                delta = torch.randn(rows, width, device="cuda")
                residual = torch.randn(rows, 2 * width if strided else width, device="cuda")[:, :width]
                weight = torch.randn(width, device="cuda")  # frozen: no weight gradient
                seed = torch.randn(rows, width, device="cuda")
                leaves = (delta.detach().requires_grad_(True), residual.detach().requires_grad_(True))
                ref_leaves = (delta.double().requires_grad_(True), residual.double().requires_grad_(True))
                out = cl.compiled_residual_add(*leaves, weight)
                self.assertIsNotNone(out, "compiled residual add fell back")
                ref = ref_leaves[0] + weight.double() * ref_leaves[1]
                grads = torch.autograd.grad(out, leaves, seed)
                ref_grads = torch.autograd.grad(ref, ref_leaves, seed.double())
                _assert_rel(self, (("out", out, ref), ("grad_delta", grads[0], ref_grads[0]),
                                   ("grad_residual", grads[1], ref_grads[1])), _FP32_TOL)


# 2. Jiterator activations ----------------------------------------------------

_ACTIVATIONS = {
    "silu": (fused_silu_op.fused_silu, F.silu, fused_silu_op.fusion_silu_fwd, fused_silu_op.fusion_silu_bwd),
    "sigmoid": (fuse_sigmoid_op.fused_sigmoid, torch.sigmoid, fuse_sigmoid_op.sigmoid_fwd,
                fuse_sigmoid_op.sigmoid_bwd),
}
_ACTIVATION_TOL = {torch.float32: _FP32_TOL, torch.float64: 1e-12}


def _activation_orders(fn, x, seed, weight):
    """y = f(x); g = VJP(seed) with create_graph; d/d(x, seed) of sum(weight * g).

    The second order is what force training differentiates (create_graph)."""
    x, seed = x.clone().requires_grad_(True), seed.clone().requires_grad_(True)
    y = fn(x)
    (grad,) = torch.autograd.grad(y, x, seed, create_graph=True)
    return (y, grad, *torch.autograd.grad((grad * weight).sum(), (x, seed)))


@unittest.skipUnless(_CUDA, "requires coordinator-assigned idle CUDA device")
class JiteratorActivationGPUTest(unittest.TestCase):
    def test_first_and_second_order_match_torch(self):
        torch.manual_seed(1234)
        x = torch.randn(1000, 64, device="cuda") * 4
        # Saturation both ways (exp overflow in 1/(1+exp(-x))), the SiLU minimum, 0.
        x[0, :7] = torch.tensor([-100.0, -30.0, -1.2785, 0.0, 1e-8, 30.0, 100.0])
        seed, weight = torch.randn_like(x), torch.randn_like(x)
        names = ("y", "grad_x", "second_x", "second_seed")
        for (name, (fused, reference, fwd_fn, bwd_fn)), dtype, transposed in product(
                _ACTIVATIONS.items(), _ACTIVATION_TOL, (False, True)):
            with self.subTest(act=name, dtype=dtype, transposed=transposed):
                inputs = [t.to(dtype) for t in (x, seed, weight)]
                if transposed:  # non-contiguous views
                    inputs = [t.t() for t in inputs]
                got = _activation_orders(fused, *inputs)
                # The jiterator Functions are on the tape, including the double backward.
                self.assertIs(type(got[0].grad_fn), fwd_fn._backward_cls)
                self.assertIs(type(got[1].grad_fn), bwd_fn._backward_cls)
                ref = _activation_orders(reference, *(t.double() for t in inputs))
                _assert_rel(self, zip(names, got, ref), _ACTIVATION_TOL[dtype])


# 3. Paired segment attention, identity CSR on side b -------------------------


def _torch_segment_attention(logits, value, segment, num_segment):
    """Plain segmented softmax attention; empty segments give 0."""
    dim = logits.shape[1]
    expanded = segment.unsqueeze(1).expand(-1, dim)
    peak = logits.new_full((num_segment, dim), float("-inf")).scatter_reduce(
        0, expanded, logits, reduce="amax", include_self=False)
    weight = (logits - peak[segment]).exp()
    denom = logits.new_zeros((num_segment, dim)).index_add(0, segment, weight)
    return logits.new_zeros((num_segment, dim)).index_add(0, segment, weight / denom[segment] * value)


def _poison_cuda_cache(*shapes, copies=8):
    """Leave NaN-filled freed blocks in the caching allocator, so a read of an
    unwritten output or statistic shows up as NaN."""
    blocks = [torch.full(shape, float("nan"), device="cuda") for shape in shapes for _ in range(copies)]
    del blocks


@unittest.skipUnless(_CUDA, "requires coordinator-assigned idle CUDA device")
class PairedSegmentAttentionIdentityBGPUTest(unittest.TestCase):
    """Side a (source) uses a perm CSR, side b (target) is sorted: identity CSR."""

    def test_identity_csr_on_side_b_matches_torch(self):
        segments = 41
        for seed, (name, dim, csr_a, off_b_dtype, skip) in enumerate((
                ("merged-alpha views, provided perm CSR on a", 32, "provided", torch.int64, False),
                ("width 128, computed CSR on a, int32 offsets on b", 128, "computed", torch.int32, False),
                ("private sink skipped on both sides", 32, "provided", torch.int64, True))):
            with self.subTest(name):
                generator = torch.Generator().manual_seed(100 + seed)
                lengths = torch.randint(0, 9, (segments,), generator=generator)
                lengths[[3, 17]] = 0  # empty segments on side b
                sink = segments - 1
                if skip:
                    lengths[sink] = 40
                seg_b = torch.repeat_interleave(torch.arange(segments), lengths)  # sorted
                rows = seg_b.numel()
                seg_a = torch.randint(0, sink, (rows,), generator=generator)
                seg_a[seg_a == 11] = 12  # an empty segment on side a
                if skip:  # sink on both sides, on b only (6 rows), on a only (6 rows)
                    seg_a[(seg_b == sink).nonzero().flatten()[:-6]] = sink
                    seg_a[:6] = sink
                logits_a = torch.randn(rows, dim, generator=generator) * 3
                logits_b = torch.randn(rows, dim, generator=generator) * 3
                if skip:  # conspicuous if the skipped side leaks
                    logits_a[seg_a == sink] += 30.0
                    logits_b[seg_b == sink] += 30.0
                value = torch.randn(rows, dim, generator=generator).cuda()
                seeds = (torch.randn(segments, dim, generator=generator).cuda(),
                         torch.randn(segments, dim, generator=generator).cuda())
                if dim == 32:  # row-strided views of a merged [rows, 128] alpha tensor
                    merged = torch.cat((torch.zeros(rows, 64), logits_a, logits_b), 1).cuda()
                    logits_a, logits_b = merged[:, 64:96], merged[:, 96:]
                else:
                    logits_a, logits_b = logits_a.cuda(), logits_b.cuda()
                seg_a, seg_b = seg_a.cuda(), seg_b.cuda()
                off_b = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(off_b_dtype).cuda()
                perm_a = off_a = None
                if csr_a == "provided":
                    perm_a = torch.argsort(seg_a, stable=True)
                    off_a = torch.searchsorted(seg_a[perm_a], torch.arange(segments + 1, device="cuda"))
                skip_segment = sink if skip else -1

                leaves = [t.detach().requires_grad_(True) for t in (logits_a, logits_b, value)]
                _poison_cuda_cache((segments, dim), (rows, dim))
                out = paired_segment_attention(
                    *leaves, seg_a, seg_b, segments, perm_a, off_a, None, off_b, False, True,
                    skip_segment=skip_segment)
                got = (*out, *torch.autograd.grad(out, leaves, seeds))

                ref_leaves = [t.double().requires_grad_(True) for t in (logits_a, logits_b, value)]
                ref = [_torch_segment_attention(ref_leaves[0], ref_leaves[2], seg_a, segments),
                       _torch_segment_attention(ref_leaves[1], ref_leaves[2], seg_b, segments)]
                if skip:  # skipped outputs are 0, so its rows get no gradient from that side
                    live = (torch.arange(segments, device="cuda") != sink)[:, None]
                    ref = [torch.where(live, r, 0.0) for r in ref]
                ref = (*ref, *torch.autograd.grad(ref, ref_leaves, tuple(s.double() for s in seeds)))
                _assert_rel(self, zip(("out_a", "out_b", "grad_logits_a", "grad_logits_b", "grad_value"),
                                      got, ref), _FP32_TOL)


# 4. Line-graph materialize kernel --------------------------------------------


def _structures():
    from ase import Atoms

    rng = np.random.default_rng(7)

    def cubic(n_side, spacing, jitter):
        grid = np.stack(np.meshgrid(*(np.arange(n_side),) * 3, indexing="ij"), -1).reshape(-1, 3)
        positions = (grid + 0.5) * spacing + rng.uniform(-jitter, jitter, grid.shape)
        return Atoms(numbers=rng.choice([1, 6, 7, 8], size=len(grid)), positions=positions,
                     cell=np.eye(3) * n_side * spacing, pbc=True)

    # Cell edges shorter than the cutoffs: triplets through periodic images.
    triclinic = Atoms(numbers=[8, 8, 1, 1, 6], positions=rng.uniform(0, 3, (5, 3)),
                      cell=[[3.1, 0, 0], [0.9, 2.8, 0], [0.4, 0.7, 3.3]], pbc=True)
    return (("cubic, 125 atoms", cubic(5, 1.6, 0.15), (3.0, 2.0)),
            ("triclinic, 5 atoms", triclinic, (4.0, 3.0)),
            ("batch of two", [cubic(3, 1.7, 0.2), triclinic], (3.5, 2.5)))


@unittest.skipUnless(_CUDA and gb.op_available and gb._triton_available,
                     "requires CUDA, nvalchemiops and Triton")
class LineGraphMaterializeGPUTest(unittest.TestCase):
    """MATRIS_LINE_GRAPH_TRITON=1 and =0 build identical triplets in the same order."""

    def test_triton_kernel_matches_torch_fallback(self):
        materialize = gb._materialize_line_graph_triton
        ran = []

        def spy(*args, **kwargs):
            result = materialize(*args, **kwargs)
            ran.append(result is not None)
            return result

        for name, atoms, (atom_cutoff, line_cutoff) in _structures():
            with self.subTest(name):
                line_graphs = {}
                for use_triton in (True, False):
                    ran.clear()
                    with patch.object(gb, "_LINE_GRAPH_TRITON", use_triton), \
                            patch.object(gb, "_materialize_line_graph_triton", spy):
                        graphs = gb.atoms_to_graph_gpu(atoms, atom_cutoff, line_cutoff, device="cuda")
                    self.assertEqual(set(ran), {use_triton})  # which implementation built the triplets
                    graphs = graphs if isinstance(graphs, list) else [graphs]
                    line_graphs[use_triton] = [graph.line_graph for graph in graphs]
                pairs = zip(line_graphs[True], line_graphs[False], strict=True)
                for index, (kernel, fallback) in enumerate(pairs):
                    with self.subTest(graph=index):
                        self.assertGreater(fallback.shape[0], 0)
                        self.assertEqual(kernel.dtype, fallback.dtype)
                        self.assertTrue(torch.equal(kernel, fallback), "triplets differ from the torch fallback")


# 5. enable_compile: Inductor-compiled interaction layers ---------------------

# enable_compile wraps four layers of two classes (Graph_Attention_Layer and
# Refinement, each on the atom and the line graph); one of each class, one per
# graph, keeps the Inductor compile time of this test down.
_COMPILED_LAYERS = ("attn_block_atom_graph", "refine_block_line_graph")


def _layer_arguments(model, graphs):
    """Keyword arguments of block 0's compiled layers in one uncompiled forward."""
    block, captured = model.interaction_block[0], {}

    def recorder(name):
        def record(module, args, kwargs):
            captured[name] = kwargs
        return record

    handles = [getattr(block, name).register_forward_pre_hook(recorder(name), with_kwargs=True)
               for name in _COMPILED_LAYERS]
    try:
        model(graphs, task="ef")
    finally:
        for handle in handles:
            handle.remove()
    return captured


@unittest.skipUnless(_CUDA, "requires coordinator-assigned idle CUDA device")
class EnableCompileLayersGPUTest(unittest.TestCase):
    """enable_compile=True wraps the attention and refinement layers of each
    interaction block in torch.compile(layer, dynamic=True), the default Inductor backend. On CUDA
    the jiterator activations graph-break, so fullgraph=True is not possible and
    Inductor compiles the fragments between them."""

    def test_compiled_layers_match_eager(self):
        from torch._dynamo.testing import CompileCounterWithBackend
        from single_gpu_fixtures import frozen_fixture, toy_model

        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        self.addCleanup(torch.set_float32_matmul_precision, previous)
        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(1234)
        model = toy_model(smoothed=True).cuda()
        block = model.interaction_block[0]
        counters = {name: CompileCounterWithBackend("inductor") for name in _COMPILED_LAYERS}
        compiled = {name: torch.compile(getattr(block, name), backend=counters[name], dynamic=True)
                    for name in _COMPILED_LAYERS}
        graph = frozen_fixture(periodic=True)[1].to("cuda")
        # The second system reruns the same compiled layers at other sizes.
        for system, graphs in (("frozen fixture", [graph]), ("batch of three", [graph] * 3)):
            arguments = _layer_arguments(model, graphs)
            for name in _COMPILED_LAYERS:
                with self.subTest(system=system, layer=name):
                    kwargs = arguments[name]
                    floats = {key: value.detach() for key, value in kwargs.items()
                              if isinstance(value, torch.Tensor) and value.is_floating_point()}
                    results = []
                    for function in (getattr(block, name), compiled[name]):
                        leaves = {key: value.clone().requires_grad_(True) for key, value in floats.items()}
                        results.append((function(**{**kwargs, **leaves}), leaves))
                    self.assertGreaterEqual(counters[name].frame_count, 1)  # Inductor compiled it.
                    (expected, ref_leaves), (actual, leaves) = results
                    seeds = tuple(torch.randn_like(output) for output in expected)
                    ref_grads = torch.autograd.grad(expected, tuple(ref_leaves.values()), seeds)
                    grads = torch.autograd.grad(actual, tuple(leaves.values()), seeds)
                    _assert_rel(self, ((f"out{index}", a, e) for index, (a, e) in enumerate(zip(actual, expected))),
                                _FP32_TOL)
                    _assert_rel(self, zip((f"grad_{key}" for key in leaves), grads, ref_grads), _FP32_TOL)


if __name__ == "__main__":
    unittest.main()
