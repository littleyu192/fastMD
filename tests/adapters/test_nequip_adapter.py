"""Native NequIP graph/derivative checks using a small actual upstream network.

CPU checks plus an optional CUDA precision regression, no downloads. Set FASTMD_NEQUIP_CHECKPOINT to additionally exercise
the official package loader and its packaged Python module identities.
"""
from __future__ import annotations

import dataclasses
import copy
import importlib.util
import io
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from fastmd.adapters.protocol import GraphBatch, PaddingRule
from fastmd.applications.energy_derivative import EnergyDerivative
from fastmd.models.nequip import NequIPAdapter, _apply_cueq, cpu_package_load_context, is_nequip_model, load_nequip


NEQUIP = importlib.util.find_spec("nequip") is not None
CUEQ = importlib.util.find_spec("cuequivariance_torch") is not None


def graph_for(atoms, adapter):
    centres, neighbours, images = neighbor_list("ijS", atoms, adapter.topology_spec().cutoff)
    return GraphBatch(positions=torch.tensor(atoms.positions, dtype=torch.float64),
                      frac_coords=torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64),
                      cell=torch.tensor(atoms.cell.array, dtype=torch.float64),
                      species=torch.tensor(atoms.numbers, dtype=torch.int32), n_real=len(atoms),
                      edge_index=torch.tensor(np.stack([centres, neighbours], axis=1), dtype=torch.int32),
                      edge_image=torch.tensor(images, dtype=torch.float64), n_edges=None,
                      cutoff=adapter.topology_spec().cutoff, padding=None)


def upstream_calculator(model, device="cpu"):
    from nequip.ase import NequIPCalculator
    from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform

    names = model.metadata["type_names"].split()
    return NequIPCalculator(model=model.eval(), device=device, transforms=[
        ChemicalSpeciesToAtomTypeMapper(names), NeighborListTransform(float(model.metadata["r_max"]))])


class NequIPNamesTest(unittest.TestCase):
    def test_cpu_nested_storage_mapping_preserves_values_and_restores_hook(self):
        original = torch.storage._load_from_bytes
        values = torch.tensor([1. / 3, -1e90, 1e-90], dtype=torch.float64)
        stream = io.BytesIO()
        # Tag CPU bytes as CUDA, reproducing the nested packaged FX storage
        # without allocating any GPU memory.
        with patch("torch.serialization.location_tag", return_value="cuda:0"):
            torch.save(values, stream)
        with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU loading must not initialize CUDA")):
            with cpu_package_load_context():
                outer = torch.storage._load_from_bytes
                with cpu_package_load_context():
                    actual = torch.storage._load_from_bytes(stream.getvalue())
                    self.assertEqual(actual.device.type, "cpu")
                    torch.testing.assert_close(actual, values, rtol=0, atol=0)
                self.assertIs(torch.storage._load_from_bytes, outer)
        self.assertIs(torch.storage._load_from_bytes, original)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with cpu_package_load_context():
                raise RuntimeError("test failure")
        self.assertIs(torch.storage._load_from_bytes, original)

    def test_explicit_names_and_offline_file_requirement(self):
        for name in ("nequip", "NequIP-OAM-L:0.1", "nequip_oam_xl", "NequIP-MP-L",
                     "nequip-oam-l-0.1", "nequip-oam-xl-0.1", "nequip-mp-l-0.1"):
            self.assertTrue(is_nequip_model(name))
        for name in ("allegro", "nequip-oam-s", "NequIP-MP-L:9.9", object()):
            self.assertFalse(is_nequip_model(name))
        with self.assertRaisesRegex(ValueError, "local model_path"):
            load_nequip("NequIP-OAM-L:0.1")
        with self.assertRaises(FileNotFoundError):
            load_nequip(model_path="/no/such/file.nequip.zip", device="cpu")
        with self.assertRaisesRegex(ValueError, "kernel"):
            load_nequip(model_path="/no/such/file.nequip.zip", kernel="auto")


@unittest.skipUnless(NEQUIP, "requires optional nequip==0.14.0")
class NequIPAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nequip.model import NequIPGNNModel
        from nequip.utils.global_state import set_global_state

        torch.set_num_threads(2)
        old_dtype = torch.get_default_dtype()
        try:
            set_global_state(allow_tf32=False)
            torch.set_default_dtype(torch.float64)
            # Deliberately non-atomic-number order and distinct energy shifts.
            cls.model = NequIPGNNModel(seed=17, model_dtype="float64", r_max=3.5,
                type_names=["O", "H", "C"], num_layers=2, num_features=4, l_max=1,
                radial_mlp_depth=1, radial_mlp_width=8, num_bessels=4, avg_num_neighbors=3.,
                per_type_energy_shifts={"O": -2., "H": -.5, "C": -1.},
                per_type_energy_scales={"O": 1.2, "H": .8, "C": .9}).eval()
        finally:
            torch.set_default_dtype(old_dtype)
        cls.adapter = NequIPAdapter(cls.model)

    def atoms(self):
        return Atoms("HCO", positions=[[.4, 1., 1.], [7.1, 1.25, 1.], [1.8, 2.1, 1.7]],
                     cell=[[8., 0., 0.], [1., 7., 0.], [.5, 1., 9.]], pbc=True)

    def evaluate(self, batch):
        return EnergyDerivative(self.adapter)(batch, self.adapter.prepare(batch), task="efs")

    def test_periodic_triclinic_graph_matches_upstream_ase_energy_forces_stress(self):
        from ase.stress import full_3x3_to_voigt_6_stress

        atoms = self.atoms()
        reference = upstream_calculator(self.model)
        previous_dtype = torch.get_default_dtype()
        try:
            # Upstream ASE from_dict converts NumPy geometry using this global
            # default, which NequIP initializes to float64 in normal use.
            torch.set_default_dtype(torch.float64)
            reference.calculate(atoms)
        finally:
            torch.set_default_dtype(previous_dtype)
        graph = graph_for(atoms, self.adapter)
        self.assertTrue(bool(graph.edge_image.ne(0).any()))
        actual = self.evaluate(graph)
        np.testing.assert_allclose(actual["e"][0], reference.results["energy"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual["f"][0], reference.results["forces"], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(full_3x3_to_voigt_6_stress(actual["s"][0]),
                                   reference.results["stress"], rtol=1e-11, atol=1e-13)
        self.assertEqual(actual["f"][0].dtype, torch.float64)

    def test_force_and_shear_stress_match_energy_finite_differences(self):
        graph = graph_for(self.atoms(), self.adapter)
        actual = self.evaluate(graph)
        prepared = self.adapter.prepare(graph)
        def energy(pos, cell):
            with torch.no_grad():
                return float(self.adapter.node_energy(prepared, pos, cell).sum())
        h = 1e-5
        perturb = torch.zeros_like(graph.positions)
        perturb[1, 2] = h
        force = -(energy(graph.positions + perturb, graph.cell) -
                  energy(graph.positions - perturb, graph.cell)) / (2 * h)
        self.assertAlmostEqual(force, float(actual["f"][0][1, 2]), delta=1e-8)
        shear = torch.zeros_like(graph.cell)
        shear[0, 1] = shear[1, 0] = h / 2
        plus, minus = torch.eye(3, dtype=torch.float64) + shear, torch.eye(3, dtype=torch.float64) - shear
        stress = (energy(graph.positions @ plus, graph.cell @ plus) -
                  energy(graph.positions @ minus, graph.cell @ minus)) / (2 * h * abs(np.linalg.det(graph.cell)))
        self.assertAlmostEqual(stress, float(actual["s"][0][0, 1]), delta=1e-10)

    def test_padding_and_isolated_element_reference(self):
        graph = graph_for(self.atoms(), self.adapter)
        expected = self.evaluate(graph)
        loops = torch.arange(9, dtype=torch.int32) % graph.n_real
        padded = dataclasses.replace(graph,
            edge_index=torch.cat([graph.edge_index, torch.stack([loops, loops], dim=1)]),
            edge_image=torch.cat([graph.edge_image, torch.tensor([[3., 0., 0.]]).repeat(9, 1)]),
            padding=self.adapter.padding_rule(), n_edges=torch.tensor([len(graph.edge_index)]))
        actual = self.evaluate(padded)
        for key in ("e", "f", "s"):
            torch.testing.assert_close(actual[key][0], expected[key][0], rtol=0, atol=0)
        isolated = Atoms("O", positions=[[1., 1., 1.]], cell=np.eye(3) * 12, pbc=True)
        result = self.evaluate(graph_for(isolated, self.adapter))
        self.assertAlmostEqual(float(result["e"][0]), -2., places=12)
        torch.testing.assert_close(result["f"][0], torch.zeros(1, 3, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(result["s"][0], torch.zeros(3, 3, dtype=torch.float64), rtol=0, atol=0)

    def test_inputs_remain_live_and_unknown_species_fail_before_forward(self):
        graph = graph_for(self.atoms(), self.adapter)
        prepared = self.adapter.prepare(graph)
        first = self.adapter.node_energy(prepared, graph.positions, graph.cell)
        graph.edge_image.fill_(5.)
        second = self.adapter.node_energy(prepared, graph.positions, graph.cell)
        self.assertFalse(torch.allclose(first, second))
        for numbers in (torch.tensor([2]), torch.tensor([999])):
            with self.assertRaisesRegex(ValueError, "species table"):
                self.adapter.prepare(dataclasses.replace(graph, species=numbers))
        self.assertFalse(self.adapter.capture_contract().sync_free)
        self.assertEqual(self.adapter.geometry_dtype, torch.float64)

    def test_fixed_shape_retains_padding_and_matches_periodic_derivatives(self):
        adapter = NequIPAdapter(self.model, fixed_shape=True)
        self.assertTrue(adapter.capture_contract().sync_free)
        self.assertTrue(adapter.padding_validation["validated"])
        self.assertEqual(adapter.capacity_step().u_step, 32)
        graph = graph_for(self.atoms(), adapter)
        expected = self.evaluate(graph)
        loops = torch.arange(17, dtype=torch.int32) % graph.n_real
        padded = dataclasses.replace(graph,
            edge_index=torch.cat([graph.edge_index, torch.stack([loops, loops], dim=1)]),
            edge_image=torch.cat([graph.edge_image, torch.tensor([[3., -2., 1.]]).repeat(17, 1)]),
            padding=adapter.padding_rule(), n_edges=torch.tensor([len(graph.edge_index)]))
        seen = []
        handle = adapter.module.register_forward_pre_hook(lambda module, args: seen.append(args[0]["edge_index"].shape[1]))
        try:
            actual = EnergyDerivative(adapter)(padded, adapter.prepare(padded), task="efs")
        finally:
            handle.remove()
        self.assertEqual(seen, [len(padded.edge_index)])
        for key in ("e", "f", "s"):
            torch.testing.assert_close(actual[key][0], expected[key][0], rtol=1e-12, atol=1e-13)
        # The lattice shifts of padding also enter strain differentiation.
        shear = torch.zeros_like(graph.cell)
        shear[0, 1] = shear[1, 0] = 5e-6
        plus, minus = torch.eye(3, dtype=torch.float64) + shear, torch.eye(3, dtype=torch.float64) - shear
        with torch.no_grad():
            difference = (adapter.node_energy(padded, graph.positions @ plus, graph.cell @ plus).sum() -
                          adapter.node_energy(padded, graph.positions @ minus, graph.cell @ minus).sum())
        self.assertAlmostEqual(float(difference / (2e-5 * torch.linalg.det(graph.cell))),
                               float(actual["s"][0][0, 1]), delta=1e-10)
        # Rewritten images must be consumed on every call, including after prepare.
        padded.edge_image[:len(graph.edge_index)].fill_(4.)
        changed = EnergyDerivative(adapter)(padded, padded, task="efs")
        self.assertFalse(torch.allclose(changed["e"][0], actual["e"][0]))

    def test_fixed_shape_empty_graph_and_cutoff_value_force_stress(self):
        adapter = NequIPAdapter(self.model, fixed_shape=True)
        isolated = Atoms("O", positions=[[1., 1., 1.]], cell=np.eye(3) * 12, pbc=True)
        graph = graph_for(isolated, adapter)
        for padded in (False, True):
            probe = dataclasses.replace(graph, edge_index=torch.zeros((4, 2), dtype=torch.int32),
                edge_image=torch.tensor([[1., 0., 0.]]).repeat(4, 1), padding=adapter.padding_rule()) if padded else graph
            result = EnergyDerivative(adapter)(probe, adapter.prepare(probe), task="efs")
            self.assertEqual(float(result["e"][0]), -2.)
            for key in ("f", "s"):
                self.assertEqual(int(torch.count_nonzero(result[key][0])), 0)
        # Keep two directed edges even at/outside r_max; dynamic reference removes them.
        for offset in (-1e-3, 0., 1e-3, 3.5):
            atoms = Atoms("OH", positions=[[0., 0., 0.], [3.5 + offset, 0., 0.]],
                          cell=np.eye(3) * 30, pbc=True)
            graph = dataclasses.replace(graph_for(atoms, adapter),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
                edge_image=torch.zeros(2, 3, dtype=torch.float64))
            actual = EnergyDerivative(adapter)(graph, adapter.prepare(graph), task="efs")
            expected = self.evaluate(graph)
            for key in ("e", "f", "s"):
                torch.testing.assert_close(actual[key][0], expected[key][0], rtol=1e-11, atol=1e-13)
            if offset >= 0:
                self.assertEqual(int(torch.count_nonzero(actual["f"][0])), 0)
                self.assertEqual(int(torch.count_nonzero(actual["s"][0])), 0)

    def test_fixed_shape_rejects_unsupported_network_and_padding(self):
        model = copy.deepcopy(self.model)
        next(part for part in model.modules() if type(part).__name__ == "InteractionBlock").edge_mlp.bias = True
        with self.assertRaisesRegex(ValueError, "bias-free"):
            NequIPAdapter(model, fixed_shape=True)
        adapter = NequIPAdapter(self.model, fixed_shape=True)
        graph = dataclasses.replace(graph_for(self.atoms(), adapter), padding=PaddingRule("sink_atoms"))
        with self.assertRaisesRegex(ValueError, "far_self_loops"):
            adapter.prepare(graph)
        with self.assertRaisesRegex(TypeError, "bool"):
            NequIPAdapter(self.model, fixed_shape="yes")

    def test_fixed_shape_forward_and_derivative_have_no_data_dependent_host_operations(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class NoHostOrDynamicShape(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func in (torch.ops.aten._local_scalar_dense.default, torch.ops.aten.nonzero.default,
                            torch.ops.aten.masked_select.default):
                    raise AssertionError(f"capture-incompatible operation: {func}")
                return func(*args, **(kwargs or {}))

        adapter = NequIPAdapter(self.model, fixed_shape=True)
        graph = graph_for(self.atoms(), adapter)
        prepared = adapter.prepare(graph)  # Species validation is explicitly outside capture.
        with NoHostOrDynamicShape():
            result = EnergyDerivative(adapter)(graph, prepared, task="efs")
        self.assertTrue(bool(torch.isfinite(result["f"][0]).all()))

    def test_plan_records_loaded_backend_and_cannot_switch_it(self):
        from fastmd import InferenceConfig
        from fastmd.adapters.negotiate import negotiate

        adapter = self.adapter
        plan = negotiate(adapter, InferenceConfig(execution="eager", optimization_profile="generic"),
                         entry="calculator", device="cpu")
        self.assertEqual(plan.decision("kernels.tensor_product_scatter").chosen, "upstream")
        self.assertEqual(adapter.kernel_backends()[0].options, ("upstream",))
        adapter.select_kernel("tensor_product_scatter", "upstream")
        with self.assertRaisesRegex(ValueError, "only the loaded"):
            adapter.select_kernel("tensor_product_scatter", "cueq_fused_tp")
        with self.assertRaisesRegex(ValueError, "only the loaded"):
            adapter.select_kernel("unknown", "upstream")


@unittest.skipUnless(NEQUIP and CUEQ, "requires optional NequIP and cuEquivariance")
class NequIPCuEqTest(unittest.TestCase):
    def test_official_modifier_preserves_state_and_cpu_padding_derivatives(self):
        from nequip.model import NequIPGNNModel, modify
        from nequip.utils.global_state import set_global_state

        previous_dtype = torch.get_default_dtype()
        try:
            set_global_state(allow_tf32=False)
            torch.set_default_dtype(torch.float64)
            torch.set_num_threads(2)
            model = NequIPGNNModel(seed=17, model_dtype="float64", r_max=3.5,
                type_names=["O", "H", "C"], num_layers=2, num_features=4, l_max=1,
                radial_mlp_depth=1, radial_mlp_width=8, num_bessels=4, avg_num_neighbors=3.,
                per_type_energy_shifts={"O": -2., "H": -.5, "C": -1.},
                per_type_energy_scales={"O": 1.2, "H": .8, "C": .9}).eval()
            with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU conversion must not initialize CUDA")):
                converted, metadata = _apply_cueq(copy.deepcopy(model))
                adapter = NequIPAdapter(converted, fixed_shape=True, kernel_metadata=metadata)
            self.assertEqual(metadata["tensor_products_converted"], 2)
            self.assertTrue(metadata["original_state_tensors_unchanged"])
            self.assertEqual(adapter.kernel, "cueq_naive")
            self.assertTrue(adapter.padding_validation["validated"])
            from fastmd import InferenceConfig
            from fastmd.adapters.negotiate import negotiate
            for profile in ("generic", "topology"):
                plan = negotiate(adapter, InferenceConfig(execution="eager", optimization_profile=profile),
                                 entry="calculator", device="cpu")
                self.assertEqual(plan.decision("kernels.tensor_product_scatter").chosen, "cueq_naive")
            self.assertEqual(adapter.kernel_backends()[0].options, ("cueq_naive",))
            with self.assertRaisesRegex(ValueError, "only the loaded"):
                adapter.select_kernel("tensor_product_scatter", "upstream")
            reference = NequIPAdapter(model)
            atoms = Atoms("HCO", positions=[[.4, 1., 1.], [7.1, 1.25, 1.], [1.8, 2.1, 1.7]],
                          cell=[[8., 0., 0.], [1., 7., 0.], [.5, 1., 9.]], pbc=True)
            graph = graph_for(atoms, adapter)
            loops = torch.arange(9, dtype=torch.int32) % graph.n_real
            padded = dataclasses.replace(graph,
                edge_index=torch.cat([graph.edge_index, torch.stack([loops, loops], dim=1)]),
                edge_image=torch.cat([graph.edge_image, torch.tensor([[3., -2., 1.]]).repeat(9, 1)]),
                padding=adapter.padding_rule())
            actual = EnergyDerivative(adapter)(padded, adapter.prepare(padded), task="efs")
            expected = EnergyDerivative(reference)(graph, reference.prepare(graph), task="efs")
            for key in ("e", "f", "s"):
                torch.testing.assert_close(actual[key][0], expected[key][0], rtol=1e-10, atol=1e-12)
            # N > 1 with zero edges triggers the upstream naive scatter bug;
            # the fixed-shape adapter's local far-loop branch must remain exact.
            isolated = Atoms("OHC", positions=[[1., 1., 1.], [8., 8., 8.], [16., 16., 16.]],
                             cell=np.eye(3) * 24, pbc=True)
            empty = graph_for(isolated, adapter)
            self.assertEqual(empty.edge_index.shape[0], 0)
            actual = EnergyDerivative(adapter)(empty, adapter.prepare(empty), task="efs")
            expected = EnergyDerivative(reference)(empty, reference.prepare(empty), task="efs")
            for key in ("e", "f", "s"):
                torch.testing.assert_close(actual[key][0], expected[key][0], rtol=0, atol=0)
            with patch("nequip.model.modify", side_effect=lambda model, **kwargs: model):
                with self.assertRaisesRegex(ValueError, "replaced 0/2"):
                    _apply_cueq(copy.deepcopy(model))
            def corrupt(model, **kwargs):
                result = modify(model, **kwargs)
                with torch.no_grad():
                    next(result.parameters()).add_(1.)
                return result
            with patch("nequip.model.modify", side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, "changed saved state"):
                    _apply_cueq(copy.deepcopy(model))
        finally:
            torch.set_default_dtype(previous_dtype)


@unittest.skipUnless(NEQUIP and torch.cuda.is_available(), "requires NequIP and CUDA")
class NequIPGPUPrecisionTest(unittest.TestCase):
    @unittest.skipUnless(CUEQ, "requires optional cuEquivariance")
    def test_cueq_fixed_shape_whole_step_keeps_double_geometry_and_matches_eager(self):
        self._check_fixed_shape_whole_step("cueq")

    def test_fixed_shape_whole_step_keeps_double_geometry_and_matches_eager(self):
        self._check_fixed_shape_whole_step("upstream")

    def _check_fixed_shape_whole_step(self, kernel):
        import tempfile
        from pathlib import Path
        from ase.io import Trajectory
        from nequip.model import NequIPGNNModel
        from nequip.utils.global_state import set_global_state
        from fastmd import GPUResidentMolecularDynamics, InferenceConfig

        previous_dtype = torch.get_default_dtype()
        try:
            set_global_state(allow_tf32=False)
            torch.set_default_dtype(torch.float64)
            model = NequIPGNNModel(seed=17, model_dtype="float32", r_max=3.5,
                type_names=["O", "H", "C"], num_layers=2, num_features=4, l_max=1,
                radial_mlp_depth=1, radial_mlp_width=8, num_bessels=4, avg_num_neighbors=3.,
                per_type_energy_shifts={"O": -2., "H": -.5, "C": -1.},
                per_type_energy_scales={"O": 1.2, "H": .8, "C": .9}).eval()
            reference_model = copy.deepcopy(model).cuda()
            if kernel == "cueq":
                model, _ = _apply_cueq(model)
            model.cuda()
            atoms = Atoms("HCO", positions=[[.4123456789, 1., 1.], [7.123456789, 1.25, 1.], [1.8, 2.1, 1.7]],
                          cell=[[8.123456789, 0, 0], [1., 7., 0], [.5, 1., 9.]], pbc=True)
            atoms.set_momenta([[.01, -.02, .03], [-.01, .01, -.02], [0., .01, -.01]])
            reference = upstream_calculator(reference_model, device="cuda")
            reference.calculate(atoms)
            trajectories = []
            with tempfile.TemporaryDirectory() as directory:
                for execution in ("eager", "whole_step"):
                    adapter = NequIPAdapter(copy.deepcopy(model), fixed_shape=True)
                    if kernel == "cueq":
                        self.assertEqual(adapter.kernel, "cueq_fused_tp")
                        if execution == "eager":
                            self._check_empty_cueq_capture(adapter, reference_model)
                    config = InferenceConfig(execution=execution, optimization_profile="generic", matmul_precision="highest")
                    path = Path(directory) / f"{execution}.traj"
                    dyn = GPUResidentMolecularDynamics(atoms.copy(), model=adapter, config=config, task="efs",
                        ensemble="nvt_nhc", temperature=300, timestep=.25, tdamp=25, fix_com=False,
                        trajectory=str(path), loginterval=2, log_stress=True, steps_per_launch=1)
                    try:
                        if execution == "whole_step":
                            runner = dyn.whole_step_runner
                            self.assertIsNotNone(runner)
                            for value in (runner.forces, runner.potential_energy, runner.stress,
                                          runner.builder.batch.positions, runner.builder.batch.cell):
                                self.assertEqual(value.dtype, torch.float64)
                            np.testing.assert_allclose(runner.forces.cpu(), reference.results["forces"], atol=1e-6, rtol=1e-5)
                            np.testing.assert_allclose(runner.potential_energy.cpu(), reference.results["energy"], atol=1e-6, rtol=1e-5)
                        dyn.run(4)
                        if execution == "whole_step":
                            self.assertGreater(dyn.whole_step_runner.launches, 0)
                    finally:
                        dyn.close()
                    with Trajectory(str(path)) as trajectory:
                        trajectories.append([frame.copy() for frame in trajectory])
                        initial = trajectory[0]
                        np.testing.assert_array_equal(initial.positions, atoms.positions)
                        for key in ("energy", "forces", "stress"):
                            np.testing.assert_allclose(initial.calc.results[key], reference.results[key], atol=1e-6, rtol=1e-5)
                self.assertEqual(len(trajectories[0]), len(trajectories[1]))
                for eager, captured in zip(*trajectories):
                    np.testing.assert_allclose(captured.positions, eager.positions, atol=2e-7, rtol=1e-7)
                    np.testing.assert_allclose(captured.get_momenta(), eager.get_momenta(), atol=2e-6, rtol=1e-5)
        finally:
            torch.set_default_dtype(previous_dtype)

    def _check_empty_cueq_capture(self, adapter, reference_model):
        atoms = Atoms("OHC", positions=[[1., 1., 1.], [8., 8., 8.], [16., 16., 16.]],
                      cell=np.eye(3) * 24, pbc=True)
        cpu = graph_for(atoms, adapter)
        batch = dataclasses.replace(cpu, **{field.name: getattr(cpu, field.name).cuda()
            for field in dataclasses.fields(cpu) if isinstance(getattr(cpu, field.name), torch.Tensor)})
        self.assertEqual(batch.edge_index.shape[0], 0)
        reference = NequIPAdapter(reference_model)
        expected = EnergyDerivative(reference)(batch, reference.prepare(batch), task="efs")
        prepared = adapter.prepare(batch)
        derivative = EnergyDerivative(adapter)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                derivative(batch, prepared, task="efs")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = derivative(batch, prepared, task="efs")
        graph.replay()
        torch.cuda.synchronize()
        for key in ("e", "f", "s"):
            torch.testing.assert_close(captured[key][0], expected[key][0], rtol=0, atol=0)

    def test_float32_weights_keep_double_geometry_in_calculator_and_md(self):
        import copy
        import tempfile
        from pathlib import Path
        from ase.io import Trajectory
        from nequip.model import NequIPGNNModel
        from nequip.utils.global_state import set_global_state
        from fastmd import MLIPCalculator, GPUResidentMolecularDynamics, InferenceConfig
        from fastmd.engine.energy_derivative import edges_only_builder_options

        old_dtype = torch.get_default_dtype()
        try:
            set_global_state(allow_tf32=False)
            torch.set_default_dtype(torch.float64)
            model = NequIPGNNModel(seed=17, model_dtype="float32", r_max=3.5,
                type_names=["O", "H", "C"], num_layers=2, num_features=4, l_max=1,
                radial_mlp_depth=1, radial_mlp_width=8, num_bessels=4, avg_num_neighbors=3.,
                per_type_energy_shifts={"O": -2., "H": -.5, "C": -1.},
                per_type_energy_scales={"O": 1.2, "H": .8, "C": .9}).eval().cuda()
            reference = upstream_calculator(copy.deepcopy(model), device="cuda")
            adapter = NequIPAdapter(model)
            atoms = Atoms("HCO", positions=[[.4123456789, 1., 1.], [7.123456789, 1.25, 1.], [1.8, 2.1, 1.7]],
                          cell=[[8.123456789, 0, 0], [1., 7., 0], [.5, 1., 9.]], pbc=True)
            config = InferenceConfig(execution="eager", optimization_profile="generic", matmul_precision="highest")
            with MLIPCalculator(model=adapter, device="cuda", task="efs", config=config) as calc:
                reference.calculate(atoms)
                probe = atoms.copy()
                probe.calc = calc
                probe.get_potential_energy()
                batch = calc._per_call_graph(None, atoms)
                torch.testing.assert_close(batch.positions.cpu(), torch.from_numpy(atoms.positions), rtol=0, atol=0)
                self.assertEqual(edges_only_builder_options(adapter)["positions_dtype"], torch.float64)
                for key in ("energy", "forces", "stress"):
                    actual = probe.get_stress() if key == "stress" else calc.results[key]
                    np.testing.assert_allclose(actual, reference.results[key], rtol=1e-5, atol=1e-6)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "md.traj"
                dyn = GPUResidentMolecularDynamics(atoms, model=adapter, config=config, task="efs",
                    ensemble="nvt_nhc", temperature=300, timestep=.25, tdamp=25, fix_com=False,
                    trajectory=str(path), loginterval=10, log_stress=True, steps_per_launch=1)
                try:
                    dyn.run(10)
                finally:
                    dyn.close()
                with Trajectory(str(path)) as trajectory:
                    self.assertEqual(len(trajectory), 2)
                    np.testing.assert_array_equal(trajectory[0].cell.array, atoms.cell.array)
                    np.testing.assert_array_equal(trajectory[0].positions, atoms.positions)
                    np.testing.assert_allclose(trajectory[0].get_forces(), reference.results["forces"], rtol=1e-5, atol=1e-6)
        finally:
            torch.set_default_dtype(old_dtype)


@unittest.skipUnless(NEQUIP and os.environ.get("FASTMD_NEQUIP_CHECKPOINT"),
                     "set FASTMD_NEQUIP_CHECKPOINT to test the official local package")
class NequIPCheckpointTest(unittest.TestCase):
    @unittest.skipUnless(CUEQ, "requires optional cuEquivariance")
    def test_official_cueq_loader_retains_state_and_validates_cpu_zero_padding(self):
        previous_dtype = torch.get_default_dtype()
        with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU loading must not initialize CUDA")):
            adapter = load_nequip(model_path=os.environ["FASTMD_NEQUIP_CHECKPOINT"], device="cpu",
                                  kernel="cueq", fixed_shape=True)
        self.assertEqual(torch.get_default_dtype(), previous_dtype)
        self.assertEqual(adapter.geometry_dtype, torch.float64)
        self.assertEqual(adapter.kernel, "cueq_naive")
        self.assertTrue(adapter.kernel_metadata["original_state_tensors_unchanged"])
        self.assertEqual(adapter.kernel_metadata["tensor_products_converted"], adapter._layers)
        self.assertTrue(adapter.padding_validation["validated"])
        self.assertEqual(adapter.padding_validation["pair_potential"], "ZBL")
        self.assertEqual(adapter.padding_validation["species_checked"], len(adapter.type_names))

    def test_official_zbl_package_fixed_shape_matches_dynamic_with_padding(self):
        torch.set_num_threads(2)
        with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU loading must not initialize CUDA")):
            adapter = load_nequip(model_path=os.environ["FASTMD_NEQUIP_CHECKPOINT"], device="cpu", fixed_shape=True)
        self.assertTrue(adapter.padding_validation["validated"])
        self.assertEqual(adapter.padding_validation["pair_potential"], "ZBL")
        atoms = Atoms("Si2", positions=[[.2, 1., 1.], [10.1, 1.3, 1.2]], cell=np.eye(3) * 12, pbc=True)
        graph = graph_for(atoms, adapter)
        padded = dataclasses.replace(graph,
            edge_index=torch.cat([graph.edge_index, torch.zeros((8, 2), dtype=torch.int32)]),
            edge_image=torch.cat([graph.edge_image, torch.tensor([[2., -1., 1.]]).repeat(8, 1)]),
            padding=adapter.padding_rule())
        actual = EnergyDerivative(adapter)(padded, adapter.prepare(padded), task="efs")
        dynamic = copy.copy(adapter)
        dynamic.fixed_shape = False
        expected = EnergyDerivative(dynamic)(padded, dynamic.prepare(padded), task="efs")
        for key in ("e", "f", "s"):
            torch.testing.assert_close(actual[key][0], expected[key][0], rtol=2e-5, atol=2e-6 if key != "s" else 1e-8)

    def test_official_packaged_module_loads_through_registry(self):
        from fastmd.models import load_adapter

        torch.set_num_threads(2)
        previous_dtype = torch.get_default_dtype()
        original_storage_loader = torch.storage._load_from_bytes
        with patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU loading must not initialize CUDA")):
            adapter = load_adapter("NequIP-OAM-L:0.1", model_path=os.environ["FASTMD_NEQUIP_CHECKPOINT"],
                                   device="cpu", optimization_profile="generic")
        self.assertEqual(torch.get_default_dtype(), previous_dtype)
        self.assertIs(torch.storage._load_from_bytes, original_storage_loader)
        self.assertTrue(all(p.device.type == "cpu" for p in adapter.module.parameters()))
        atoms = Atoms("Si2", positions=[[1., 1., 1.], [3.1, 1.3, 1.2]], cell=np.eye(3) * 15, pbc=True)
        batch = graph_for(atoms, adapter)
        result = EnergyDerivative(adapter)(batch, adapter.prepare(batch), task="efs")
        for key in ("e", "f", "s"):
            self.assertTrue(bool(torch.isfinite(result[key][0]).all()))


if __name__ == "__main__":
    unittest.main()
