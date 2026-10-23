"""Lattice/fractional coordinate transforms must not depend on TF32.

With ``torch.backends.cuda.matmul.allow_tf32 = True`` cuBLAS runs float32
GEMMs in TF32 (10 mantissa bits). NGC PyTorch containers export
``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``, which makes ``allow_tf32`` default to
``True`` at start-up (on torch 2.12 it only sets the default: assigning
``False`` afterwards disables TF32), so MD scripts run with TF32 unless they
turn it off. Before the fix the small ``pos @ inv(cell)``, ``frac @ cell``,
``image @ lattice`` and strain products did, which moved S01 positions by
~9e-3 A, changed edge/triplet sets near the cutoffs and moved MatRIS-10M
forces by ~6e-2 eV/A and stress by ~30 MPa relative to strict FP32.
This port covers single-device graph builders, model geometry, strain gradients,
and GPU-resident MD. Domain partitioning is outside FastMD.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from ase import Atoms, units
from torch.utils._python_dispatch import TorchDispatchMode

from fastmd.interfaces.gpu_md import GPUResidentMolecularDynamics
from fastmd.config import InferenceConfig
from fastmd.graph import FixedCapacityGraphBuilder, RadiusGraph, TensorGraphBuilder
from fastmd.graph.fixed_capacity_builder import (
    BatchedCandidateRefresher,
    _fixed_builder_available,
)
from fastmd.graph.geometry import small_matmul
from fastmd.graph.gpu_graph_builder import atoms_to_graph_gpu, op_available
from fastmd.models.matris.model import MatRIS
from fastmd.models.matris.processgraph import process_graphs

from matris_test_assets import real_model_assets_available

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
CUDA = torch.cuda.is_available()


@contextmanager
def tf32_enabled():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _cells() -> dict[str, np.ndarray]:
    return {
        # S01-like cubic CsSnI3 cell
        "cubic": np.diag([18.6171] * 3),
        "triclinic": np.array([[12.3, 0.0, 0.0], [2.1, 11.7, 0.0], [-1.3, 1.9, 13.1]]),
    }


def _positions(cell: np.ndarray, n: int = 96, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # include atoms outside the cell so wrapping is exercised
    return rng.uniform(-0.3, 1.3, (n, 3)) @ cell


def _boundary_positions(cell: np.ndarray, n: int = 96, seed: int = 11) -> np.ndarray:
    """Positions with one fractional coordinate per atom 2e-5..4e-4 away from
    an integer (both sides): an exact transform wraps them correctly, a TF32
    one (~5e-4 fractional error at |x| ~ 20 A) wraps many of them wrongly."""
    rng = np.random.default_rng(seed)
    frac = rng.uniform(-0.3, 1.3, (n, 3))
    dims = rng.integers(0, 3, n)
    offsets = rng.uniform(2e-5, 4e-4, n) * rng.choice([-1.0, 1.0], n)
    frac[np.arange(n), dims] = rng.integers(-1, 3, n) + offsets
    return frac @ cell


def _exact_wraps(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    pos32 = positions.astype(np.float32).astype(np.float64)
    return np.floor(pos32 @ np.linalg.inv(cell.astype(np.float32).astype(np.float64)))


def _wrapped_error(frac32: torch.Tensor, positions: np.ndarray, cell: np.ndarray) -> float:
    """Max Cartesian error (A) of fp32 fractional coordinates vs an fp64 reference
    for the fp32-rounded inputs, modulo lattice translations."""
    pos = positions.astype(np.float32).astype(np.float64)
    cell32 = cell.astype(np.float32).astype(np.float64)
    delta = frac32.double().cpu().numpy() - pos @ np.linalg.inv(cell32)
    delta -= np.round(delta)
    return float(np.abs(delta @ cell32).max())


class SmallMatmulTest(unittest.TestCase):
    def test_matches_fp64_and_gradients(self) -> None:
        devices = ["cpu"] + (["cuda"] if CUDA else [])
        shapes = [((57, 3), (3, 3)), ((4, 31, 3), (4, 3, 3)), ((3, 3), (3, 3)),
                  ((5, 1, 3), (5, 3, 3))]
        for device in devices:
            for shape_a, shape_b in shapes:
                with self.subTest(device=device, a=shape_a, b=shape_b), tf32_enabled():
                    gen = torch.Generator().manual_seed(0)
                    a64 = (torch.rand(shape_a, generator=gen, dtype=torch.float64) * 40 - 20)
                    b64 = torch.rand(shape_b, generator=gen, dtype=torch.float64) * 20 - 10
                    a = a64.float().to(device).requires_grad_(True)
                    b = b64.float().to(device).requires_grad_(True)
                    out = small_matmul(a, b)
                    reference = torch.matmul(a.double(), b.double())
                    self.assertEqual(out.shape, reference.shape)
                    scale = reference.abs().max().item()
                    self.assertLess((out.double() - reference).abs().max().item(), 1e-6 * scale)
                    weight = torch.linspace(-1, 1, out.numel(), device=device).reshape(out.shape)
                    grad_a, grad_b = torch.autograd.grad((out * weight).sum(), (a, b))
                    a_ref = a.detach().double().requires_grad_(True)
                    b_ref = b.detach().double().requires_grad_(True)
                    ref_a, ref_b = torch.autograd.grad(
                        (torch.matmul(a_ref, b_ref) * weight.double()).sum(), (a_ref, b_ref)
                    )
                    for grad, ref in ((grad_a, ref_a), (grad_b, ref_b)):
                        self.assertLess(
                            (grad.double() - ref).abs().max().item(),
                            1e-6 * ref.abs().max().item(),
                        )


@unittest.skipUnless(
    CUDA and op_available and _fixed_builder_available, "CUDA graph builders are unavailable"
)
class TF32GeometryTest(unittest.TestCase):
    def test_graph_builders_keep_positions_exact(self) -> None:
        for name, cell in _cells().items():
            positions = _positions(cell)
            n = positions.shape[0]
            cell_t = torch.tensor(cell, dtype=torch.float32)
            pos_t = torch.tensor(positions, dtype=torch.float32, device="cuda")
            numbers = torch.full((n,), 14, dtype=torch.int32)
            with self.subTest(cell=name), tf32_enabled():
                fixed = FixedCapacityGraphBuilder(
                    cell=cell_t, atomic_numbers=numbers, u_capacity=8192,
                    t_capacity=65536, atom_graph_cutoff=4.0, line_graph_cutoff=3.0,
                    candidate_cutoff=4.5, n_dummy=4, device="cuda",
                )
                _, status = fixed.build(pos_t)
                self.assertFalse(bool(status.overflow.item()))
                self.assertLess(_wrapped_error(fixed.frac_coords[:n], positions, cell), 1e-5)
                # candidate refresh uses the same transform; its wraps must be exact
                fixed.refresh_candidates(pos_t)
                pos32 = positions.astype(np.float32).astype(np.float64)
                wraps = np.floor(pos32 @ np.linalg.inv(cell.astype(np.float32).astype(np.float64)))
                np.testing.assert_array_equal(fixed.reference_wraps.cpu().numpy(), wraps)

                dynamic = TensorGraphBuilder(
                    cell=cell_t, atomic_numbers=numbers, atom_graph_cutoff=4.0,
                    line_graph_cutoff=3.0, device="cuda", check_isolated_atoms=False,
                ).build(pos_t)
                self.assertLess(_wrapped_error(dynamic.atom_frac_coord, positions, cell), 1e-5)
                atoms = Atoms(numbers=[14] * n, positions=positions, cell=cell, pbc=True)
                from_atoms = atoms_to_graph_gpu(
                    atoms, atom_graph_cutoff=4.0, line_graph_cutoff=3.0,
                    check_isolated_atoms=False,
                )
                self.assertLess(_wrapped_error(from_atoms.atom_frac_coord, positions, cell), 1e-5)
                # identical topology to the fixed-capacity builder
                self.assertEqual(
                    int(from_atoms.atom_graph.shape[0]), int(status.edge_count.item())
                )
                self.assertEqual(
                    int(from_atoms.line_graph.shape[0]), int(status.triplet_count.item())
                )

    def test_batched_candidate_refresher_keeps_wraps_exact(self) -> None:
        # BatchedCandidateRefresher (batched whole-step runner) replaced a
        # torch.bmm; its wraps and candidate lists must match the exact
        # per-builder refresh under allow_tf32. On GB300 cuBLAS picks a TF32
        # kernel for the batched (B>=2, N, 3) x (3, 3) product from N = 512
        # (smaller shapes happen to get an exact kernel), hence 512 atoms in
        # 1.8x larger cells.
        cells = [1.8 * cell for cell in _cells().values()]
        n = 512
        positions = [_boundary_positions(cell, n, seed) for seed, cell in enumerate(cells)]

        def builder_for(cell):
            return FixedCapacityGraphBuilder(
                cell=torch.tensor(cell, dtype=torch.float32),
                atomic_numbers=torch.full((n,), 14, dtype=torch.int32),
                u_capacity=8192, t_capacity=65536, atom_graph_cutoff=4.0,
                line_graph_cutoff=3.0, candidate_cutoff=4.5, n_dummy=4, device="cuda",
            )

        batched = [builder_for(cell) for cell in cells]
        refresher = BatchedCandidateRefresher(batched)
        stacked = torch.tensor(np.stack(positions), dtype=torch.float32, device="cuda")
        with tf32_enabled():
            refresher.refresh(stacked)
            singles = []
            for cell, pos in zip(cells, positions):
                single = builder_for(cell)
                single.refresh_candidates(torch.tensor(pos, dtype=torch.float32, device="cuda"))
                singles.append(single)
        torch.cuda.synchronize()
        for replica, (cell, pos) in enumerate(zip(cells, positions)):
            with self.subTest(replica=replica):
                builder, single = batched[replica], singles[replica]
                np.testing.assert_array_equal(
                    builder.reference_wraps.cpu().numpy(), _exact_wraps(pos, cell)
                )
                np.testing.assert_array_equal(
                    builder.num_neighbors.cpu().numpy(), single.num_neighbors.cpu().numpy()
                )
                count = builder.num_neighbors.long().cpu()

                def pairs(b, atom, k):
                    index = b.neighbor_matrix[atom, :k].long().cpu().reshape(k, 1)
                    shift = b.neighbor_shifts[atom, :k].long().cpu().reshape(k, 3)
                    return sorted(torch.cat([index, shift], dim=1).tolist())

                for atom in range(n):
                    k = int(count[atom])
                    self.assertEqual(pairs(builder, atom, k), pairs(single, atom, k), atom)

    def test_process_graphs_geometry_and_strain_vjp_match_fp64(self) -> None:
        def graphs_for(cell, n, seed):
            atoms = Atoms(numbers=[14] * n, positions=_positions(cell, n=n, seed=seed),
                          cell=cell, pbc=True)
            graph32 = atoms_to_graph_gpu(
                atoms, atom_graph_cutoff=4.0, line_graph_cutoff=3.0,
                check_isolated_atoms=False,
            )
            graph64 = RadiusGraph(
                graph_id=None, mp_id=None, composition="",
                atomic_number=graph32.atomic_number,
                atom_frac_coord=graph32.atom_frac_coord.double(),
                lattice=graph32.lattice.double(),
                neighbor_image=graph32.neighbor_image.double(),
                atom_graph=graph32.atom_graph, atom_graph_cutoff=4.0,
                line_graph=graph32.line_graph, line_graph_cutoff=3.0,
                directed2undirected=graph32.directed2undirected,
                undirected2directed=graph32.undirected2directed,
            )
            return {"fp32": graph32, "fp64": graph64}

        cells = _cells()
        big = {name: graphs_for(cell, 64, 3) for name, cell in cells.items()}
        small = graphs_for(cells["triclinic"], 40, 5)
        cases = {
            # process_graphs has a single-graph, an equal-size batched and a
            # chunked (unequal sizes) branch for the coordinate/strain products
            **{f"single-{name}": lambda p, g=g: [g[p]] for name, g in big.items()},
            "batched-equal": lambda p: [big["cubic"][p], big["cubic"][p]],
            "batched-chunked": lambda p: [big["cubic"][p], small[p]],
        }
        for name, make in cases.items():
            results = {}
            for label in ("fp32", "fp64"):
                with tf32_enabled():
                    batch = process_graphs(make(label), compute_stress=True)
                    vectors = batch["unit_edge_vectors"] * batch["edge_lengths"][:, None]
                    # smooth pair energy of the (strained) geometry
                    energy = (torch.exp(-batch["edge_lengths"]) * batch["edge_lengths"] ** 2).sum()
                    grad_coords, grad_strain = torch.autograd.grad(
                        energy, [batch["batch_cart_coords"], batch["batch_strains"]]
                    )
                results[label] = (
                    batch["batch_cart_coords"].detach().double(), vectors.detach().double(),
                    batch["volumes"].detach().double(), grad_coords.double(),
                    grad_strain.double(),
                )
            with self.subTest(case=name):
                coords, vectors, volumes, grad_coords, grad_strain = results["fp32"]
                ref = results["fp64"]
                self.assertLess((coords - ref[0]).abs().max().item(), 1e-5)
                self.assertLess((vectors - ref[1]).abs().max().item(), 1e-5)
                self.assertLess(((volumes - ref[2]) / ref[2]).abs().max().item(), 1e-6)
                self.assertLess(
                    (grad_coords - ref[3]).abs().max().item(),
                    1e-5 * ref[3].abs().max().item(),
                )
                self.assertLess(
                    (grad_strain - ref[4]).abs().max().item(),
                    1e-5 * ref[4].abs().max().item(),
                )


def _orthogonal_cells() -> dict[str, np.ndarray]:
    # DomainPartition only admits orthogonal cells; RDX benchmark box (1,344 atoms)
    return {"cubic": _cells()["cubic"],
            "rdx": np.diag([26.3836268, 23.1877185, 21.35719065])}


def _toy_model() -> MatRIS:
    torch.manual_seed(5)
    # No layer is 3 wide, so every K = 3 cuBLAS product is a geometry transform.
    model = MatRIS(num_layers=2, node_feat_dim=8, edge_feat_dim=8, three_body_feat_dim=8,
                   mlp_hidden_dims=(8,), num_radial=5, num_angular=5, max_n=2, max_l=2,
                   three_body_expansion="fourier", pairwise_cutoff=3.0, three_body_cutoff=2.0,
                   reference_energy="demo", use_bias=True, enable_compile=False,
                   enable_checkpoint=False)
    return model.cuda().eval().requires_grad_(False)


def _rdx_like_atoms(seed: int = 3) -> Atoms:
    """Jittered grid filling the RDX box (many periodic-image bonds), with a
    third of the atoms shifted by a lattice vector (unwrapped MD positions)."""
    cell = _orthogonal_cells()["rdx"]
    rng = np.random.default_rng(seed)
    counts = (15, 13, 12)
    grid = np.stack(np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), -1).reshape(-1, 3)
    frac = (grid + 0.5 + rng.uniform(-0.1, 0.1, grid.shape)) / np.asarray(counts)
    frac += rng.integers(-1, 2, frac.shape) * (rng.random((len(frac), 1)) < 1 / 3)
    numbers = rng.choice([1, 6, 7, 8], len(frac))
    return Atoms(numbers=numbers, positions=frac @ cell, cell=cell, pbc=True)


class _SmallContractionRecorder(TorchDispatchMode):
    """Record cuBLAS-dispatched products with contraction length 3 on CUDA.

    einsum/matmul/linear decompose into these aten ops before dispatch, and
    autograd's backward ops are recorded as well."""

    _OPERAND = {torch.ops.aten.mm.default: 0, torch.ops.aten.bmm.default: 0,
                torch.ops.aten.addmm.default: 1, torch.ops.aten.baddbmm.default: 1,
                torch.ops.aten.mv.default: 0, torch.ops.aten.addmv.default: 1,
                torch.ops.aten.dot.default: 0, torch.ops.aten.vdot.default: 0,
                torch.ops.aten._addmm_activation.default: 1}

    def __init__(self):
        super().__init__()
        self.small, self.products = [], 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        index = self._OPERAND.get(func)
        if index is not None and args[index].is_cuda and args[index].is_floating_point():
            self.products += 1
            if args[index].shape[-1] == 3:
                self.small.append((str(func), [tuple(a.shape) for a in args
                                               if isinstance(a, torch.Tensor)]))
        return func(*args, **(kwargs or {}))


@unittest.skipUnless(
    CUDA and op_available and _fixed_builder_available, "CUDA graph builders are unavailable"
)
class TF32SingleDeviceGeometryTest(unittest.TestCase):

    def test_ef_paths_run_no_cublas_geometry_products(self) -> None:
        atoms, model = _rdx_like_atoms(), _toy_model()
        recorder = _SmallContractionRecorder()
        with tf32_enabled(), recorder:
            # single GPU: dynamic builder + model, GPU-resident MD (topology profile)
            graph = atoms_to_graph_gpu(atoms, atom_graph_cutoff=3.0, line_graph_cutoff=2.0,
                                       check_isolated_atoms=False)
            model([graph], task="ef", is_training=False)
            with GPUResidentMolecularDynamics(
                atoms, model=model, task="ef", device="cuda", ensemble="nve", timestep=0.2,
                config=InferenceConfig(execution="eager", optimization_profile="topology",
                                       checkpoint="off", report=False),
            ) as md:
                md.evaluate(md.state.positions)
            torch.cuda.synchronize()
        self.assertGreater(recorder.products, 0)
        self.assertEqual(recorder.small, [])


def _run_worker(out: Path, tf32_override: str) -> dict[str, np.ndarray]:
    env = dict(os.environ)
    env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = tf32_override
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(TESTS), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, str(TESTS / "tf32_efs_worker.py"), str(out)],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    if proc.returncode != 0:
        raise AssertionError(f"worker failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return dict(np.load(out))


@unittest.skipUnless(
    CUDA and op_available and _fixed_builder_available and real_model_assets_available(),
    "MatRIS-10M-OAM checkpoint / S01 frames unavailable",
)
class TF32MatRISS01Test(unittest.TestCase):
    """Full model in a process started with the TF32 override (allow_tf32
    defaults to True) vs a process started without it (strict FP32).

    Tier 2 (real model on one GPU, two worker processes): it runs only where
    the MatRIS-10M-OAM checkpoint and S01 frames are available.

    The model's own GEMMs legitimately stay in TF32 under the override, so a
    residual difference remains. Measured on GB300 (5 S01 frames, both paths):
    after the fix max|dF| 9.5e-4 eV/A, max|dE| 0.095 meV/atom, max|dS| 0.80
    MPa, identical edge/triplet counts; before it 6.2e-2 eV/A, 0.53 meV/atom,
    29.6 MPa and different triplet counts. Tolerances sit between the two.
    """

    def test_s01_efs_matches_strict_fp32_under_tf32_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strict = _run_worker(Path(tmp) / "strict.npz", "0")
            tf32 = _run_worker(Path(tmp) / "tf32.npz", "1")
        for result in (strict, tf32):
            self.assertTrue(str(result["matris_file"]).startswith(str(REPO)))
        self.assertFalse(bool(strict["allow_tf32"]))
        self.assertTrue(bool(tf32["allow_tf32"]))
        n_atoms = strict["f"].shape[1]
        np.testing.assert_array_equal(tf32["counts"], strict["counts"])
        self.assertLess(float(tf32["position_error"].max()), 1e-5)
        mpa = 1.0e3 / units.GPa  # calculator stress eV/A^3 -> MPa
        for suffix, stress_scale in (("", mpa), ("_fixed", 1.0e3)):  # fixed path: GPa
            with self.subTest(path=suffix or "calculator"):
                d_e = np.abs(tf32["e" + suffix] - strict["e" + suffix]).max() / n_atoms
                d_f = np.abs(tf32["f" + suffix] - strict["f" + suffix]).max()
                d_s = np.abs(tf32["s" + suffix] - strict["s" + suffix]).max() * stress_scale
                self.assertLess(d_f, 4e-3, "max |dF| eV/A")
                self.assertLess(d_e, 2.5e-4, "max |dE| eV/atom")
                self.assertLess(d_s, 4.0, "max |dS| MPa")


if __name__ == "__main__":
    unittest.main()
