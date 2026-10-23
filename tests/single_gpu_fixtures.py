"""Deterministic single-device MatRIS fixtures and independent VJP checks.

Extracted from MD tests; no distributed runtime is needed.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
from fastmd.engine.dispatch import current_policy, dispatcher_for
from fastmd.engine.layout import ExecutionLayout
from fastmd.graph.radiusgraph import RadiusGraph
from fastmd.models.matris.model import MatRIS


def toy_model(*, smoothed=False, norm="rms", reference="demo", angular="fourier", dtype=torch.float32):
    torch.manual_seed(47)
    model = MatRIS(num_layers=2, node_feat_dim=6, edge_feat_dim=6,
                   three_body_feat_dim=6, mlp_hidden_dims=(6,), num_radial=3,
                   num_angular=3, max_n=2, max_l=2, norm_type=norm,
                   three_body_expansion=angular, pairwise_cutoff=3., three_body_cutoff=2.,
                   reference_energy=reference, use_smoothed_for_delta_edge=smoothed,
                   use_bias=True, enable_compile=False, enable_checkpoint=False).to(dtype=dtype).eval()
    model.requires_grad_(False)
    return model

def _fixture_atoms(dtype, positions, periodic):
    """(positions, atomic numbers, 12 A cubic cell) of the fixtures below."""
    if positions is None:
        triad = ([[11.75, 1.13, 1.07], [.93, 1.29, 1.13], [1.17, 2.31, 1.32]]
                 if periodic else [[1., 1., 1.], [2.1, 1., 1.], [1.2, 2.2, 1.]])
        positions = torch.tensor([*triad,
                                  [5., 5., 5.], [6.1, 5., 5.], [8., 8., 8.]], dtype=dtype)
    numbers = torch.tensor([1, 6, 8, 1, 7, 8], dtype=torch.int64)[:len(positions)]
    return positions, numbers, torch.eye(3, dtype=dtype) * 12

def frozen_fixture(*, has_line=True, dtype=torch.float32, positions=None, periodic=False):
    """A triad, a bond with no incident triplets, and an isolated atom.

    The graph is hand-built: only two triad bonds (with two triplets when
    ``has_line``). The first result carries the atoms (positions, atomic
    numbers, cell) and their hand-built keys.
    """
    positions, numbers, cell = _fixture_atoms(dtype, positions, periodic)
    bonds = (((0, 1, 1, 0, 0), (1, 2, 0, 0, 0), (3, 4, 0, 0, 0)) if periodic else
             ((0, 1, 0, 0, 0), (0, 2, 0, 0, 0), (3, 4, 0, 0, 0)))
    directed = tuple(row for a, b, x, y, z in bonds for row in ((a, b, x, y, z), (b, a, -x, -y, -z)))
    center, left, right = (1, 1, 2) if periodic else (0, 0, 2)
    triplets = ((center, directed[left], directed[right]),
                (center, directed[right], directed[left])) if has_line else ()
    line_indices = [[center, left // 2, left, right // 2, right],
                    [center, right // 2, right, left // 2, left]] if has_line else []
    shard = SimpleNamespace(
        owned_atom_ids=tuple(range(6)), atom_ids=tuple(range(6)), positions=positions,
        atomic_numbers=numbers, cell=cell, owned_bond_keys=bonds,
        owned_triplet_keys=triplets, directed_keys=directed,
        center_directed_keys=tuple(sorted(directed)), line_triplet_keys=triplets,
        line_source_index=(1, 0) if has_line else (), line_target_index=(0, 1) if has_line else (),
    )
    graph = RadiusGraph(
        graph_id=None, mp_id=None, composition="toy", atomic_number=numbers,
        atom_frac_coord=positions / 12, lattice=cell,
        atom_graph=torch.tensor([key[:2] for key in directed], dtype=torch.int32),
        neighbor_image=torch.tensor([key[2:] for key in directed], dtype=dtype),
        atom_graph_cutoff=3., line_graph_cutoff=2.,
        directed2undirected=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
        undirected2directed=torch.tensor([0, 2, 4], dtype=torch.int32),
        line_graph=torch.tensor(line_indices, dtype=torch.int32).reshape(-1, 5),
    )
    return shard, graph

def hand_built_dispatcher():
    """The P=1 OpDispatcher for a hand-built graph dict (the layer sites require one).

    Its layout certifies nothing; process_graphs attaches the certified layout
    of its lowering plan instead.
    """
    return dispatcher_for(current_policy(), ExecutionLayout(pair_layout="generic", incidence="none"))

def assert_function_and_vjp(case, actual_fn, expected_fn, inputs, *, atol=0, rtol=0):
    """Outputs and every input VJP agree; the inputs are not modified in place."""
    actual_inputs = [value.detach().clone().requires_grad_(True) for value in inputs]
    expected_inputs = [value.detach().clone().requires_grad_(True) for value in inputs]
    before = [value.detach().clone() for value in actual_inputs]
    actual, expected = actual_fn(*actual_inputs), expected_fn(*expected_inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    case.assertEqual(len(actual), len(expected))
    gradients = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
        gradients.append(torch.full_like(got, (index + 1) / 3))
    actual_vjp = torch.autograd.grad(actual, actual_inputs, gradients, allow_unused=True)
    expected_vjp = torch.autograd.grad(expected, expected_inputs, gradients, allow_unused=True)
    for got, want in zip(actual_vjp, expected_vjp):
        if want is None:
            case.assertIsNone(got)
        else:
            torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    for value, saved in zip(actual_inputs, before):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)
