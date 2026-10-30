"""MD's GPU relaxation regressions migrated to FastMD (no downloaded model required).

Run on an idle CUDA device with ``python -m unittest discover -s tests
-p test_md_relaxation_gpu.py -v``. Tests skip when CUDA builders are unavailable.
"""
from __future__ import annotations

from contextlib import nullcontext
import dataclasses
import functools
import gc
import os
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from fastmd.config import CapacityConfig, InferenceConfig
from fastmd.execution.whole_step_cuda_graph import FixedCapacityOverflow
from fastmd.execution.system_batch import SlotLayout
from fastmd.execution.system_relax import SystemFIRE, relaxation_builder
from fastmd.graph.fixed_capacity_builder import _fixed_builder_available
from fastmd.graph.gpu_graph_builder import op_available
from fastmd.interfaces import multi_system_md, multi_system_relax, replica_md
from fastmd.interfaces.calculator import MLIPCalculator
from fastmd.interfaces.multi_system_relax import MultiSystemRelaxation
from fastmd.interfaces.relax import StructOptimizer
from fastmd.models.matris.model import MatRIS
from pair_potential_adapter import SmoothPairPotential, UnionPairAdapter
from test_runtime_config import deterministic_algorithms


class RelaxationLifetimeTest(unittest.TestCase):
    ENGINES = (replica_md.ReplicaMolecularDynamics, multi_system_md.MultiSystemMolecularDynamics,
               multi_system_relax._SlotRelaxationEngine)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_recapture_survives_last_graph_release(self):
        for kind in self.ENGINES:
            with self.subTest(engine=kind.__name__):
                engine = kind.__new__(kind)
                engine.device = torch.device('cuda')
                engine._log = None
                for _ in range(2):  # a new pool after a startup OOM must get a new keeper
                    previous = weakref.ref(engine._pool_keeper[1]) if engine._pool_keeper is not None else None
                    engine._graph_pool = torch.cuda.graph_pool_handle()
                    constructor = torch.cuda.CUDAGraph

                    def allocate():
                        if previous is not None:
                            self.assertIsNone(previous(), 'old pool must be releasable before the new allocation')
                        return constructor()

                    with mock.patch.object(torch.cuda, 'CUDAGraph', side_effect=allocate):
                        engine._keep_graph_pool_alive()
                    pool = engine._graph_pool
                    first = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(first, pool=pool):
                        state = torch.ones(32, device='cuda')
                    first.replay()
                    torch.cuda.synchronize()
                    del first
                    gc.collect()
                    torch.cuda.empty_cache()
                    engine._keep_graph_pool_alive()
                    second = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(second, pool=pool):
                        result = state + 2
                    second.replay()
                    torch.testing.assert_close(result, torch.full_like(result, 3))
                    del second, result, state
                keeper = weakref.ref(engine._pool_keeper[1])
                engine.close()
                self.assertIsNone(keeper())

    def test_each_capture_keeps_its_pool_alive_before_allocating_a_runner(self):
        class StopBeforeRunner(Exception):
            pass

        class Resource:
            def zero_(self):
                return self

        builders = ((replica_md.ReplicaMolecularDynamics, replica_md, 'union_builder'),
                    (multi_system_md.MultiSystemMolecularDynamics, multi_system_md, 'slot_union_builder'),
                    (multi_system_relax._SlotRelaxationEngine, multi_system_relax, 'relaxation_builder'))
        for kind, module, builder in builders:
            with self.subTest(engine=kind.__name__):
                engine = kind.__new__(kind)
                engine.device, engine._graph_pool = 'cuda', (1, 2)
                engine._adapter = engine.cell = engine.numbers = engine.pbc = None
                engine._admit_cells, engine.candidate_skin = False, 1.0
                union = SimpleNamespace(replicas=1, layout=None, cells=None, numbers=None, atoms=[1],
                                        u_capacity=1, t_capacity=0,
                                        integrator=SimpleNamespace(cell_filter=False))

                def check_pool(*args, **kwargs):
                    self.assertIsNotNone(engine._pool_keeper)
                    self.assertEqual(engine._pool_keeper[0], engine._graph_pool)
                    raise StopBeforeRunner()

                # Only CUDA resources are simulated: each production _capture and the
                # inherited keeper method execute, before a builder can allocate a runner.
                with (mock.patch.object(torch.cuda, 'synchronize'),
                      mock.patch.object(torch.cuda, 'device', side_effect=lambda *a: nullcontext()),
                      mock.patch.object(torch.cuda, 'graph', side_effect=lambda *a, **k: nullcontext()),
                      mock.patch.object(torch.cuda, 'CUDAGraph', side_effect=Resource),
                      mock.patch.object(torch, 'empty', side_effect=lambda *a, **k: Resource()),
                      mock.patch.object(module, builder, side_effect=check_pool)):
                    with self.assertRaises(StopBeforeRunner):
                        engine._capture(union, None, None, forces=None)
                self.assertIsNotNone(engine._pool_keeper)

    def test_close_releases_pool_keeper_even_when_log_or_adapter_close_fails(self):
        class Resource:
            pass

        for kind in self.ENGINES:
            for failure in (None, 'log', 'adapter'):
                with self.subTest(engine=kind.__name__, failure=failure):
                    engine = kind.__new__(kind)
                    graph, held = Resource(), Resource()
                    references = weakref.ref(graph), weakref.ref(held)
                    engine._pool_keeper = ((1, 2), graph, held)
                    del graph, held
                    engine._log = mock.Mock()
                    if failure == 'log':
                        engine._log.close.side_effect = RuntimeError('log close failed')
                    with mock.patch.object(replica_md, 'release_adapter') as release:
                        if failure == 'adapter':
                            release.side_effect = RuntimeError('adapter close failed')
                        if failure:
                            with self.assertRaisesRegex(RuntimeError, 'close failed'):
                                engine.close()
                        else:
                            engine.close()
                        self.assertIsNone(engine._pool_keeper)
                        self.assertTrue(all(reference() is None for reference in references))
                        release.side_effect = None
                        engine.close()
                        self.assertEqual(release.call_count, 2)

    def test_closed_relaxation_releases_graphs_without_cyclic_gc(self):
        class Resource:
            pass

        def make(fail):
            relaxation = object.__new__(MultiSystemRelaxation)
            runner, graph, integrator = Resource(), Resource(), Resource()
            runner.graph = graph
            union = SimpleNamespace(runner=runner, integrator=integrator)
            close = mock.Mock(side_effect=RuntimeError("close failed") if fail else None)
            relaxation._engine = SimpleNamespace(close=close, _unions=[union], moved=relaxation._moved,
                                                  _planner=lambda owner=relaxation: owner)
            return relaxation, [weakref.ref(value) for value in (relaxation, runner, graph, integrator)]

        enabled = gc.isenabled()
        gc.disable()
        try:
            for fail in (False, True):
                with self.subTest(close_fails=fail):
                    relaxation, references = make(fail)
                    for _ in range(2):  # close is safe repeatedly, including when releasing the adapter fails.
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "close failed"):
                                relaxation.close()
                        else:
                            relaxation.close()
                    self.assertIsNone(relaxation._engine.moved)
                    self.assertIsNone(relaxation._engine._planner)
                    self.assertEqual(relaxation._engine.close.call_count, 2)
                    self.assertIs(relaxation._engine._unions[0].runner, references[1]())
                    del relaxation
                    self.assertTrue(all(reference() is None for reference in references))
        finally:
            if enabled:
                gc.enable()

    def test_ase_optimizer_closes_its_log_on_success_and_failure(self):
        for fail in (False, True):
            closed = []

            class TrackedFIRE(FIRE):
                def run(self, *args, **kwargs):
                    if fail:
                        raise RuntimeError("relaxation failed")
                    return super().run(*args, **kwargs)

                def __exit__(self, exc_type, exc, traceback):
                    result = super().__exit__(exc_type, exc, traceback)
                    closed.append(self.logfile.closed)
                    return result

            optimizer = object.__new__(StructOptimizer)
            optimizer.calculator = _HarmonicCalculator()
            optimizer.optimizer, optimizer.engine = TrackedFIRE, "ase"
            atoms = Atoms("H", positions=[[0.5, 0.1, 0]], cell=[5, 5, 5], pbc=True)
            with self.subTest(run_fails=fail), tempfile.TemporaryDirectory() as directory:
                options = dict(steps=1, relax_cell=False, verbose=False, assign_magmoms=False,
                               logfile=os.path.join(directory, "relax.log"))
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "relaxation failed"):
                        optimizer.relax(atoms, **options)
                else:
                    optimizer.relax(atoms, **options)
                self.assertEqual(closed, [True])


class _HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        q = atoms.positions
        self.results = {
            "energy": float(0.5 * np.square(q).sum()),
            "forces": -q.copy(),
            "stress": np.eye(3) * float(np.square(q).mean()) * 1e-4,
        }


class SystemRecordsVersusASETest(unittest.TestCase):
    """Device-recording operations run on CPU too; compare them with ASE's actual observer cadence."""

    def test_records_match_ase_and_survive_slot_compaction(self):
        structures = [Atoms("H", positions=[[0, 0, 0]], cell=[4, 5, 6], pbc=True),
                      Atoms("H2", positions=[[0.8, 0.2, 0.1], [-1.5, 0.5, 0.3]],
                            cell=[4, 5, 6], pbc=True)]
        for cell_filter in (False, True):
            for interval in (1, 3, 0, -2, -20):
                with self.subTest(cell_filter=cell_filter, interval=interval):
                    expected = []
                    for atoms in structures:
                        copy = atoms.copy()
                        copy.calc = _HarmonicCalculator()
                        target = FrechetCellFilter(copy) if cell_filter else copy
                        records = []
                        with FIRE(target, logfile=None) as optimizer:
                            def observe():
                                records.append((optimizer.nsteps, copy.get_potential_energy(),
                                                copy.get_positions(), copy.get_forces(),
                                                copy.get_stress(voigt=False), copy.cell.array.copy()))
                            optimizer.attach(observe, interval=interval)
                            optimizer.run(fmax=1e-12, steps=7)
                            expected.append((optimizer.nsteps, records))

                    options = dict(fmax=1e-12, max_steps=7, cell_filter=cell_filter, device="cpu",
                                   record_interval=interval, record_stress=True)
                    fire = SystemFIRE(SlotLayout((3, 4)), **options)
                    for slot, atoms in enumerate(structures):
                        fire.write_system(slot, positions=atoms.positions, cell=atoms.cell.array)
                    fire.sync()
                    for evaluation in range(8):
                        fire.step_pre_force()
                        forces = torch.zeros_like(fire.positions)
                        energies = torch.zeros(fire.systems, dtype=torch.float64)
                        stresses = torch.zeros((fire.systems, 3, 3), dtype=torch.float64)
                        source = structures if fire.systems == 2 else structures[1:]
                        for slot, atoms in enumerate(source):
                            start, n = fire.layout.starts[slot], len(atoms)
                            copy = atoms.copy()
                            copy.positions = fire.positions[start:start + n].numpy()
                            if cell_filter:
                                copy.cell = fire.cells[slot].numpy()
                            copy.calc = _HarmonicCalculator()
                            forces[start:start + n] = torch.as_tensor(copy.get_forces())
                            energies[slot] = copy.get_potential_energy()
                            stresses[slot] = torch.as_tensor(copy.get_stress(voigt=False))
                        fire.step_post_force(forces, energy=energies, stress=stresses)
                        if evaluation == 3:
                            first = fire.records.read(0, 0, 1, int(fire.state.steps[0]))
                            compact = SystemFIRE(SlotLayout((2,)), **options)
                            compact.take_systems(fire, [1])
                            compact.sync()
                            fire = compact
                    second = fire.records.read(0, 0, 2, int(fire.state.steps[0]))
                    for actual, (steps, reference) in zip((first, second), expected):
                        self.assertEqual(actual["steps"], [row[0] for row in reference])
                        for column, name in enumerate(("energies", "positions", "forces", "stresses", "cells"), 1):
                            if name == "cells" and not cell_filter:
                                self.assertIsNone(actual[name])
                                continue
                            for row, value in zip(reference, actual[name]):
                                np.testing.assert_allclose(value, row[column], rtol=0, atol=1e-11,
                                                           err_msg=f"{name} at step {row[0]}")
                    self.assertEqual(int(fire.state.steps[0]), expected[1][0])


class _RelaxationFixtures:
    @staticmethod
    def adapter() -> UnionPairAdapter:
        return UnionPairAdapter(SmoothPairPotential(strength=0.5).cuda())

    @staticmethod
    def config(candidate_skin: float = 1.0) -> InferenceConfig:
        return InferenceConfig(execution="whole_step", report=False,
                               capacity=CapacityConfig(warmup=1, candidate_skin=candidate_skin))

    @staticmethod
    def matris(dtype: torch.dtype = torch.float32) -> MatRIS:
        torch.manual_seed(0)
        model = MatRIS(num_layers=2, node_feat_dim=16, edge_feat_dim=16, three_body_feat_dim=16,
                       mlp_hidden_dims=(16,), num_radial=4, num_angular=4, max_n=2, max_l=2, pairwise_cutoff=5.0,
                       three_body_cutoff=3.0, reference_energy="demo", enable_compile=False, enable_checkpoint=False)
        return model.to(device="cuda", dtype=dtype).eval().requires_grad_(False)



@unittest.skipUnless(
    torch.cuda.is_available() and op_available and _fixed_builder_available,
    "CUDA graph builders are unavailable",
)
class MultiSystemRelaxationCudaTest(unittest.TestCase):
    """MultiSystemRelaxation (structures relaxed by FIRE in the slots of a union, in whole-step CUDA graphs) with the
    test pair potential behind UnionPairAdapter at ten times its default strength: six structures of 4 to 32 atoms
    (fcc Cu and Al, diamond Si; cubic, primitive and orthorhombic cells), each moved by up to 0.15 A, windows of 5
    steps, deterministic scatters. Each structure is compared with ASE's FIRE on it alone (fmax 0.02 eV/A) with
    MLIPCalculator of the same adapter, with a fixed cell and with ASE's FrechetCellFilter."""

    FMAX, MAX_STEPS, WINDOW = 0.02, 300, 5

    def test_captured_cell_update_preserves_single_system_geometry(self):
        """A replay after changing cells must use each single-system FP32 inverse.

        These nonorthogonal cells came from a real-checkpoint relaxation whose
        inverse64-then-cast path changed fractional coordinates by one FP32 ulp.
        No checkpoint or force tolerance is needed to test the geometry contract.
        """
        target = torch.tensor([
            [[-0.001182793213312505, 2.731857185610092, 2.7323548594795923],
             [2.732873175516406, -0.0018889664997513934, 2.7330606037883514],
             [2.731937163068748, 2.7316272936769557, -0.0009527608726891491]],
            [[3.6188309733993633, -8.879738678063493e-05, -0.00011436160505377404],
             [-8.879685931963344e-05, 3.6187463897219465, -2.6277240493296836e-05],
             [-0.0001143617660871848, -2.6277551458050766e-05, 3.618846356657117]],
        ], dtype=torch.float64, device="cuda")
        cells = target * 1.002
        layout = SlotLayout((2, 4))
        builder = relaxation_builder(
            _RelaxationFixtures.adapter(), cell_filter=True, layout=layout, cells=cells,
            atomic_numbers=torch.tensor([14, 14, 29, 29, 29, 29], device="cuda"), atoms=[2, 4],
            u_capacity=128, candidate_skin=1.0, device="cuda")
        initial_inverse = builder.inv_cells.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                builder.set_cells(cells)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                builder.set_cells(cells)
        torch.cuda.current_stream().wait_stream(stream)
        # Change the same input storage after capture: replay must read the new
        # cells, without a host calculation or a stale inverse from construction.
        cells.copy_(target)
        graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(torch.equal(builder.inv_cells, initial_inverse))
        for slot, start in enumerate(layout.starts):
            reference = torch.linalg.inv(cells[slot].float())
            self.assertTrue(torch.equal(builder.inv_cells[slot], reference))
            for row in range(start, start + layout.ceilings[slot]):
                self.assertTrue(torch.equal(builder.row_inv_cells[row], reference))
        self.assertTrue(torch.equal(builder.current_cells, target))
        self.assertTrue(torch.equal(builder.cell, target.float()))
        self.assertTrue(torch.equal(builder.systems.cells, target.float()))
        self.assertTrue(torch.equal(builder.row_cells, target.float().index_select(0, builder.systems.system_of_row)))
        self.assertTrue(torch.equal(builder.batch.cell, target[0].float()))

    @staticmethod
    def structures() -> list[Atoms]:
        structures = [bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2), bulk("Cu", "fcc", a=3.6).repeat(3),
                      bulk("Al", "fcc", a=4.05, orthorhombic=True).repeat((2, 2, 1)),
                      bulk("Si", "diamond", a=5.43).repeat((2, 1, 1)), bulk("Cu", "fcc", a=3.5).repeat(2),
                      bulk("Al", "fcc", a=4.0).repeat((3, 2, 2))]
        for seed, atoms in enumerate(structures):
            atoms.positions += np.random.default_rng(seed).uniform(-0.15, 0.15, atoms.positions.shape)
        return structures

    def relax(self, adapter, structures, **options) -> MultiSystemRelaxation:
        options = {"config": _RelaxationFixtures.config(), "self_check": False,
                   "steps_per_launch": self.WINDOW, "fmax": self.FMAX, "max_steps": self.MAX_STEPS, **options}
        return MultiSystemRelaxation(adapter, structures, **options)

    def alone(self, adapter, cell_filter: bool, execution: str = "eager") -> list[tuple]:
        """ASE's FIRE on each structure alone: its steps, positions, cell, energy, forces and stress (eV/A^3)."""
        out = []
        calculator = MLIPCalculator(model=adapter, task="efs" if cell_filter else "ef", device="cuda",
                                      config=InferenceConfig(execution=execution, report=False))
        for atoms in self.structures():
            atoms.calc = calculator
            optimizer = FIRE(FrechetCellFilter(atoms) if cell_filter else atoms, logfile=None)
            self.assertTrue(optimizer.run(fmax=self.FMAX, steps=self.MAX_STEPS))
            out.append((optimizer.nsteps, atoms.positions.copy(), atoms.cell.array.copy(), atoms.get_potential_energy(),
                        atoms.get_forces(), atoms.get_stress(voigt=False) if cell_filter else None))
        calculator.close()
        return out

    def test_structures_relax_as_ase_fire_alone(self):
        """A fixed cell and FrechetCellFilter, the six structures in six slots, in two (four structures written
        into freed slots, no recapture), drained in six or in two (a union captured again with only its relaxing
        systems once they hold at most half its rows: the drain's payback patched to 0),
        with a capture again that fails (the union runs on as it was), or in three unions of at most 40 rows, in two
        slots, drained, for the potential with a three-body term over a line graph, and with a growth that runs out
        of CUDA memory: in six slots, the first window overflowing, above 48 rows (the union of 91 rows splits into
        unions of 32, 27, 8 and 24 rows), and in two slots, the third window overflowing with structures queued, above
        32 rows (64 rows into two of 32, each taking the next structures): each structure's FIRE steps equal ASE's (13
        to 96; 12 to 137 with
        the three-body term), its positions and cell within 1e-5 A, its energy and stress (cell filter) within 1e-5
        of their largest value and its forces within 1e-5 eV/A (with the three-body term: positions within 2e-4 A,
        cell within 1e-4 A and forces within 1e-4 eV/A). FIRE amplifies FP32 rounding over a long trajectory:
        on RTX 3090 the original MD source and this migration have bitwise-identical ASE and GPU final arrays,
        including a 1.0047e-4 A ASE/GPU position difference after 137 steps (the same convergence step), recorded
        in validation/md_relax_migration/three-body-source-comparison.json. The position tolerance covers that
        source behavior; energy, force, cell and step-count checks remain unchanged. Every result is converged
        and comes back in input order with a SinglePointCalculator of the relaxed structure."""
        adapter = _RelaxationFixtures.adapter()
        three_body = UnionPairAdapter(adapter.module, triplet_cutoff=3.3, angle=0.5)
        cases = [(adapter, *case) for case in ((None, False, None), (2, False, None), (None, True, None),
                                               (2, True, None), (None, "fails", None), (6, False, 40))]
        cases.append((three_body, 2, True, None))
        # A window of the first union overflows its pair capacity (all in slots: the first; in two slots: the third,
        # mid-run with structures queued), and a growth whose capture runs out of CUDA memory (above 48 and 32 rows)
        # splits the union.
        splitting = UnionPairAdapter(adapter.module)
        cases += [(splitting, None, False, None), (splitting, 2, False, None)]
        runner = multi_system_relax.SlotRelaxationRunner
        overflow = {None: (1, 48), 2: (3, 32)}  # slots: (the window that overflows, the rows no capture fits above)

        class Overflowing(runner):
            def replay(self, *args, **kwargs):
                self.windows = getattr(self, "windows", 0) + 1
                if not self.recapture and self.windows == Overflowing.at:
                    raise FixedCapacityOverflow(2 * self.builder.u_capacity + 2, 0)
                return super().replay(*args, **kwargs)

        def start(*args, **kwargs):
            if kwargs["recapture"] and kwargs["builder"].layout.rows > Overflowing.rows:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
            return Overflowing(*args, **kwargs)

        with deterministic_algorithms():
            for cell_filter in (False, True):
                references = {}
                for model, slots, drain, union_rows in cases:
                    if model is splitting:  # the same potential
                        references[model] = references[adapter]
                    if model not in references:  # a line graph: the fixed-capacity builder's positions, as ours
                        references[model] = self.alone(model, cell_filter,
                                                       "model_graph" if model is three_body else "eager")
                    expected, line_graph = references[model], model is three_body
                    with self.subTest(cell_filter=cell_filter, slots=slots, drain=drain, union_rows=union_rows,
                                      line_graph=line_graph):
                        structures = self.structures()
                        draining = (mock.patch.multiple(multi_system_relax, DRAIN_PAYBACK=0.0) if drain
                                    else nullcontext())
                        Overflowing.at, Overflowing.rows = overflow.get(slots, (0, 0))
                        failing = (mock.patch.object(multi_system_relax._SlotRelaxationEngine, "compact",
                                                     side_effect=torch.OutOfMemoryError("simulated"))
                                   if drain == "fails" else
                                   mock.patch.object(multi_system_relax, "SlotRelaxationRunner", start)
                                   if model is splitting else nullcontext())
                        with draining, failing, self.relax(model, structures, slots=slots, union_rows=union_rows,
                                                           cell_filter="frechet" if cell_filter else None) as relax:
                            results = relax.run()
                            stats = relax.stats()
                        if line_graph:
                            self.assertGreater(max(stats["max_triplet_count"]), 0)
                        self.assertEqual(stats["structure_writes"], 0 if slots in (None, 6) else len(structures) - 2)
                        self.assertEqual(stats["compactions"] > 0, drain is True)
                        self.assertEqual(len(stats["drain_failures"]), 1 if drain == "fails" else 0)
                        if model is splitting:  # all in slots: halves of 32 and 59 rows, the second runs out too
                            self.assertEqual(stats["splits"][0][:2], [91, [32, 27, 8, 24]] if slots is None
                                             else [64, [32, 32]])
                        else:
                            self.assertEqual(stats["recaptures"], stats["compactions"])
                            self.assertEqual(len(stats["unions"]), 1 if union_rows is None else 3)
                        self.assertEqual(stats["finished"], {"converged": 6, "max_steps": 0, "failed": 0})
                        for index, (result, (steps, positions, cell, energy, forces, stress)) in enumerate(
                                zip(results, expected)):
                            what = f"structure {index}"
                            self.assertEqual(result.index, index)
                            self.assertEqual((result.steps, result.status), (steps, "converged"), what)
                            self.assertLess(result.fmax, self.FMAX)
                            atol = 1e-4 if line_graph else 1e-5
                            position_atol = 2e-4 if line_graph else atol
                            np.testing.assert_allclose(result.atoms.positions, positions, rtol=0, atol=position_atol,
                                                       err_msg=what)
                            np.testing.assert_allclose(result.atoms.cell.array, cell, rtol=0, atol=atol, err_msg=what)
                            self.assertAlmostEqual(result.atoms.get_potential_energy(), energy,
                                                   delta=1e-5 * abs(energy))
                            np.testing.assert_allclose(result.atoms.get_forces(), forces, rtol=0, atol=atol,
                                                       err_msg=what)
                            if cell_filter:
                                np.testing.assert_allclose(result.stress, stress, rtol=0,
                                                           atol=1e-5 * np.abs(stress).max(), err_msg=what)
                                np.testing.assert_allclose(result.atoms.get_stress(voigt=False), stress, rtol=0,
                                                           atol=1e-5 * np.abs(stress).max(), err_msg=what)
                            else:
                                self.assertIsNone(result.stress)

    def test_matris_structures_relax_as_ase_fire_alone(self):
        """MatRIS in slots relaxes as ASE's FIRE with MLIPCalculator on each structure alone: with a fixed cell to
        convergence at fmax 0.005 eV/A (this small random MatRIS's forces are a few meV/A): the same FIRE steps (22
        to 40), positions within 1e-4 A, forces within 1e-4 eV/A, energy within 2e-5 relative (measured: up to 4.8e-5
        A, 1.5e-5 eV/A and 7.2e-6); and with FrechetCellFilter for its first 15 steps: positions and cell within 1e-4
        A, stress within 2e-3 of the largest (measured: up to 2.7e-5 A). Longer, this random MatRIS stretches the
        cells by up to 0.8 A in 40 steps, along which FIRE amplifies float32 rounding. The fixed cell also in two
        slots, drained (a union captured again with only its relaxing systems: MatRIS's own call with gradients on
        under the drain's no_grad), with the same results. A MatRIS with direct forces and stress
        (is_conservation=False) gives one stress per graph: a variable-cell relaxation of it raises ValueError."""
        model = _RelaxationFixtures.matris()
        draining = mock.patch.multiple(multi_system_relax, DRAIN_PAYBACK=0.0)
        with deterministic_algorithms():
            for cell_filter, fmax, steps, slots in ((False, 0.005, 300, None), (False, 0.005, 300, 2),
                                                    (True, 1e-9, 15, None)):
                calculator = MLIPCalculator(model=model, task="efs" if cell_filter else "ef", device="cuda",
                                              config=InferenceConfig(execution="eager", report=False))
                expected = []
                for atoms in self.structures():
                    atoms.calc = calculator
                    optimizer = FIRE(FrechetCellFilter(atoms) if cell_filter else atoms, logfile=None)
                    self.assertEqual(optimizer.run(fmax=fmax, steps=steps), not cell_filter)
                    expected.append((optimizer.nsteps, atoms.positions.copy(), atoms.cell.array.copy(),
                                     atoms.get_potential_energy(), atoms.get_forces(),
                                     atoms.get_stress(voigt=False) if cell_filter else None))
                calculator.close()
                with self.subTest(cell_filter=cell_filter, slots=slots):
                    with (draining if slots else nullcontext()), self.relax(
                            model, self.structures(), fmax=fmax, max_steps=steps, self_check=True, slots=slots,
                            cell_filter="frechet" if cell_filter else None) as relax:
                        results = relax.run()
                        plan, stats = relax.plan_record(), relax.stats()
                    self.assertEqual((plan.model, plan.decision("system_batching").chosen), ("matris", "union"))
                    if slots:
                        self.assertGreater(stats["compactions"], 0)
                        self.assertEqual(stats["drain_failures"], [])
                    largest = max(float(np.abs(stress).max()) for *_, stress in expected) if cell_filter else 0.0
                    for index, (result, (count, positions, cell, energy, forces, stress)) in enumerate(
                            zip(results, expected)):
                        what = f"structure {index}"
                        self.assertEqual((result.steps, result.status),
                                         (count, "max_steps" if cell_filter else "converged"), what)
                        np.testing.assert_allclose(result.atoms.positions, positions, rtol=0, atol=1e-4, err_msg=what)
                        np.testing.assert_allclose(result.atoms.cell.array, cell, rtol=0, atol=1e-4, err_msg=what)
                        if cell_filter:
                            np.testing.assert_allclose(result.stress, stress, rtol=0, atol=2e-3 * largest,
                                                       err_msg=what)
                        else:
                            self.assertAlmostEqual(result.atoms.get_potential_energy(), energy,
                                                   delta=2e-5 * abs(energy))
                            np.testing.assert_allclose(result.atoms.get_forces(), forces, rtol=0, atol=1e-4,
                                                       err_msg=what)
        torch.manual_seed(0)
        direct = MatRIS(num_layers=2, node_feat_dim=16, edge_feat_dim=16, three_body_feat_dim=16, mlp_hidden_dims=(16,),
                        num_radial=4, num_angular=4, max_n=2, max_l=2, pairwise_cutoff=5.0, three_body_cutoff=3.0,
                        reference_energy="demo", is_conservation=False, enable_compile=False,
                        enable_checkpoint=False).cuda().eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "direct forces and stress"):
            self.relax(direct, self.structures(), fmax=0.005, max_steps=5, cell_filter="frechet")

    def test_rejections_and_the_step_limit(self):
        """Invalid arguments and structures (a slab, constraints, a queued structure with an isolated atom where the
        configuration rejects them) are rejected at construction; with max_steps 0 every structure stops at its start
        (status "max_steps", 0 steps) with the forces there, and a structure whose energy is not finite fails without
        stopping the others."""
        adapter = _RelaxationFixtures.adapter()
        structures = self.structures()
        slab = structures[0].copy()
        slab.pbc = (True, True, False)
        for options, pattern in (({"cell_filter": "exp"}, "cell_filter"), ({"fmax": 0.0}, "fmax"),
                                 ({"fmax": float("nan")}, "fmax"), ({"max_steps": -1}, "max_steps"),
                                 ({"steps_per_launch": 0}, "steps_per_launch"), ({"slots": 0}, "slots"),
                                 ({"fire": {"dt": 0.1}}, "FIREParameters")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, pattern):
                self.relax(adapter, structures, **options)
        fixed = structures[0].copy()
        fixed.set_constraint(FixAtoms(indices=[0]))
        for bad, pattern in (([], "one or more"), ([slab] + structures[1:], "periodic"),
                             (structures[1:] + [fixed], "constraints")):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                self.relax(adapter, bad)
        lone = structures[0].copy()  # queued behind structure 0 in its slot class: an atom far from the others
        lone.set_cell(lone.cell.array * [1.0, 1.0, 10.0], scale_atoms=False)
        lone.positions[0, 2] = 0.5 * lone.cell.array[2, 2]
        config = dataclasses.replace(_RelaxationFixtures.config(), isolated_atoms="error")
        with self.subTest(pattern="neighbour"), self.assertRaisesRegex(ValueError, r"structures \[2\] have an atom"):
            self.relax(adapter, structures[:2] + [lone], slots=2, config=config)
        helium = structures[3].copy()
        helium.numbers[1] = 2

        class HeliumFails(UnionPairAdapter):  # a non-finite energy on helium rows
            def node_energy(self, inputs, positions, cell, edge_shifts=None):
                energies = super().node_energy(inputs, positions, cell, edge_shifts=edge_shifts)
                return torch.where(inputs.species == 2, torch.full_like(energies, float("nan")), energies)

        with self.relax(HeliumFails(adapter.module), structures + [helium], max_steps=0, slots=3) as relax:
            results = relax.run()
        self.assertEqual(results[-1].status, "failed")
        calculator = MLIPCalculator(model=adapter, task="ef", device="cuda",
                                      config=InferenceConfig(execution="eager", report=False))
        for atoms, result in zip(structures, results):
            self.assertEqual((result.status, result.steps), ("max_steps", 0))
            np.testing.assert_allclose(result.atoms.positions, atoms.positions, rtol=0, atol=0)
            atoms.calc = calculator
            np.testing.assert_allclose(result.forces, atoms.get_forces(), rtol=0,
                                       atol=1e-5 * np.abs(atoms.get_forces()).max())
        calculator.close()

    def test_struct_optimizer_relaxes_on_the_gpu_path_as_with_ase(self):
        """StructOptimizer.relax with FIRE on a CUDA calculator takes the GPU path and returns what ASE's optimizer
        does (the class's structures, fmax 0.02 eV/A): with a fixed cell (loginterval 1 and 0) and FrechetCellFilter
        (loginterval 3), one structure and a list, each trajectory records the same steps (and the final state) with
        energies, forces, stresses, positions and cells within 1e-5 (of the largest value), the input atoms relaxed in
        place to the final structure; save_path writes the same file. Another optimizer, FIRE's other arguments and
        constraints take ASE's optimizer, named with the reason in config_summary; engine="gpu" raises there, and a
        list with a constrained structure relaxes it with ASE's optimizer, the others on the GPU path, in order."""
        adapter = _RelaxationFixtures.adapter()
        config = InferenceConfig(report=False, capacity=CapacityConfig(warmup=1, candidate_skin=1.0))

        def relax(engine, structures, **options):
            with StructOptimizer(model=adapter, device="cuda", config=config, **options.pop("init", {})) as optimizer:
                results = optimizer.relax(structures, fmax=self.FMAX, steps=self.MAX_STEPS, verbose=False,
                                          engine=engine, **options)
                return results, optimizer.config_summary()["relaxation"]["last"]

        def assert_same(result, expected, atoms, what):
            ours, theirs = result["trajectory"], expected["trajectory"]
            self.assertEqual(len(ours), len(theirs), what)
            for name in ("energies", "forces", "stresses", "atom_positions", "cells"):
                for k, (value, reference) in enumerate(zip(getattr(ours, name), getattr(theirs, name))):
                    reference = np.asarray(reference)
                    np.testing.assert_allclose(value, reference, rtol=0, err_msg=f"{what}: {name}[{k}]",
                                               atol=1e-5 * max(1.0, float(np.abs(reference).max())))
            np.testing.assert_allclose(result["final_structure"].cart_coords, expected["final_structure"].cart_coords,
                                       rtol=0, atol=1e-5, err_msg=what)
            np.testing.assert_allclose(atoms.positions, result["final_structure"].cart_coords, rtol=0, atol=1e-12)

        with deterministic_algorithms():
            for relax_cell, loginterval in ((False, 1), (True, 3), (False, 0)):
                with self.subTest(relax_cell=relax_cell, loginterval=loginterval):
                    expected, last = relax("ase", self.structures()[2], relax_cell=relax_cell,
                                           loginterval=loginterval)
                    self.assertEqual(last["path"], "ase")
                    atoms = self.structures()[2]
                    result, last = relax("auto", atoms, relax_cell=relax_cell, loginterval=loginterval)
                    self.assertEqual((last["path"], last["structures"]), ("gpu", 1))
                    assert_same(result, expected, atoms, f"relax_cell={relax_cell}")
            with tempfile.TemporaryDirectory() as directory:
                structures = self.structures()[:3]
                structures[1].set_constraint(FixAtoms(indices=[0]))
                results, last = relax("auto", structures, relax_cell=True,
                                      save_path=os.path.join(directory, "relax_{index}.pkl"))
                self.assertEqual((last["path"], last["gpu"]), ("both", 2))
                self.assertIn("constraints (FixAtoms)", last["reason"])
                for index, (atoms, result) in enumerate(zip(structures, results)):
                    reference = self.structures()[index]
                    if index == 1:
                        reference.set_constraint(FixAtoms(indices=[0]))
                    expected, _ = relax("ase", reference, relax_cell=True)
                    assert_same(result, expected, atoms, f"list structure {index}")
                    with open(os.path.join(directory, f"relax_{index}.pkl"), "rb") as file:
                        saved = pickle.load(file)
                    self.assertEqual(len(saved["energy"]), len(result["trajectory"]))
                    np.testing.assert_array_equal(saved["atomic_number"], atoms.numbers)
        for options, reason in (({"init": {"optimizer": "BFGS"}}, "optimizer BFGS"),
                                ({"downhill_check": True}, "FIRE arguments ['downhill_check']"),
                                ({"dt": -1.0}, "FIRE arguments outside"),
                                ({"relax_cell": True, "ase_filter": functools.partial(FrechetCellFilter)},
                                 "cell filter functools.partial")):
            with self.subTest(reason=reason):
                options = {"relax_cell": False, **options}
                _, last = relax("auto", self.structures()[3], **dict(options))
                self.assertEqual(last["path"], "ase")
                self.assertIn(reason, last["reason"])
                with self.assertRaisesRegex(ValueError, "engine='gpu' cannot relax"):
                    relax("gpu", self.structures()[3], **dict(options))

    def test_struct_optimizer_gpu_path_edges_and_fallbacks(self):
        """StructOptimizer's GPU path at the edges: MatRIS (the class's small random MatRIS, enable_compile on: its
        blocks run uncompiled there and keep the flag for ASE's path) with task "efsm", fmax 0.005 eV/A: the same
        steps as ASE's optimizer, records within 2e-4 A, 1e-4 eV/A and 2e-5 of the energy (measured: up to 1.1e-4 A;
        this MatRIS's forces are a few meV/A, along which FIRE amplifies float32 rounding), the final record's and the
        final structure's magnetic moments from one calculator call on it (the other records hold None); loginterval
        -2 and steps 0 record what ASE's observer does; logfile=None stays on the GPU path; and a structure whose
        energy is not finite (its forces are) fails on the GPU path and relaxes with ASE's optimizer instead, alone
        and in a list (the others on the GPU path), as ASE's optimizer alone relaxes it; engine="gpu" raises."""
        model, compiled_model = _RelaxationFixtures.matris(), _RelaxationFixtures.matris()
        for block in compiled_model.interaction_block:  # as MatRIS.load(enable_compile=True) sets it
            block.enable_compile = True
        config = InferenceConfig(report=False, capacity=CapacityConfig(warmup=1, candidate_skin=1.0))
        with deterministic_algorithms():
            with StructOptimizer(model=compiled_model, task="efsm", device="cuda", config=config) as compiled, \
                    StructOptimizer(model=model, task="efsm", device="cuda", config=config) as eager:
                for index in (0, 3):
                    atoms = self.structures()[index]
                    result = compiled.relax(atoms, fmax=0.005, steps=300, relax_cell=False, verbose=False,
                                            logfile=None)
                    self.assertEqual(compiled.last_relaxation["path"], "gpu")
                    self.assertTrue(compiled.calculator.adapter.compiled)
                    expected = eager.relax(self.structures()[index], fmax=0.005, steps=300, relax_cell=False,
                                           verbose=False, engine="ase")
                    ours, theirs = result["trajectory"], expected["trajectory"]
                    self.assertEqual(len(ours), len(theirs))
                    np.testing.assert_allclose(np.array(ours.atom_positions), np.array(theirs.atom_positions),
                                               rtol=0, atol=2e-4)
                    np.testing.assert_allclose(np.array(ours.forces), np.array(theirs.forces), rtol=0, atol=1e-4)
                    np.testing.assert_allclose(ours.energies, theirs.energies, rtol=2e-5)
                    self.assertTrue(all(magmoms is None for magmoms in ours.magmoms[:-1]))
                    np.testing.assert_allclose(ours.magmoms[-1], theirs.magmoms[-1], rtol=0, atol=1e-4)
                    np.testing.assert_allclose(result["final_structure"].site_properties["magmom"],
                                               expected["final_structure"].site_properties["magmom"], rtol=0,
                                               atol=1e-4)
        adapter = _RelaxationFixtures.adapter()
        pair_config = InferenceConfig(report=False, capacity=CapacityConfig(warmup=1, candidate_skin=1.0))
        with StructOptimizer(model=adapter, device="cuda", config=pair_config) as optimizer, deterministic_algorithms():
            for loginterval, steps in ((-2, 300), (1, 0)):
                with self.subTest(loginterval=loginterval, steps=steps):
                    runs = [optimizer.relax(self.structures()[1], fmax=self.FMAX, steps=steps, relax_cell=True,
                                            loginterval=loginterval, verbose=False, engine=engine)["trajectory"]
                            for engine in ("gpu", "ase")]
                    self.assertEqual(len(runs[0]), len(runs[1]))
                    np.testing.assert_allclose(np.array(runs[0].cells), np.array(runs[1].cells), rtol=0, atol=1e-5)
                    np.testing.assert_allclose(runs[0].energies, runs[1].energies, rtol=1e-5)

        class HeliumFails(UnionPairAdapter):  # a non-finite energy on helium rows (finite forces)
            def node_energy(self, inputs, positions, cell, edge_shifts=None):
                energies = super().node_energy(inputs, positions, cell, edge_shifts=edge_shifts)
                return torch.where(inputs.species == 2, torch.full_like(energies, float("nan")), energies)

        def with_helium():
            atoms = self.structures()[2]
            atoms.numbers[1] = 2
            return atoms

        with StructOptimizer(model=HeliumFails(adapter.module), device="cuda", config=pair_config) as optimizer:
            results = optimizer.relax(self.structures()[:2] + [with_helium()], fmax=self.FMAX, steps=self.MAX_STEPS,
                                      relax_cell=False, verbose=False)
            last = optimizer.last_relaxation
            self.assertEqual((last["path"], last["gpu"]), ("both", 2))
            self.assertIn("failed at step", last["reason"])
            self.assertEqual(len(results), 3)
            alone = optimizer.relax(with_helium(), fmax=self.FMAX, steps=self.MAX_STEPS, relax_cell=False,
                                    verbose=False)
            self.assertEqual(optimizer.last_relaxation["path"], "ase")  # one structure: the same fallback
            self.assertIn("failed at step", optimizer.last_relaxation["reason"])
            expected = optimizer.relax(with_helium(), fmax=self.FMAX, steps=self.MAX_STEPS, relax_cell=False,
                                       verbose=False, engine="ase")
            for result in (results[2], alone):  # ASE's optimizer's relaxation (its forces are finite)
                self.assertEqual(len(result["trajectory"]), len(expected["trajectory"]))
                np.testing.assert_allclose(result["final_structure"].cart_coords,
                                           expected["final_structure"].cart_coords, rtol=0, atol=1e-5)
            with self.assertRaisesRegex(ValueError, "engine='gpu' cannot relax"):
                optimizer.relax(with_helium(), fmax=self.FMAX, steps=self.MAX_STEPS, relax_cell=False,
                                verbose=False, engine="gpu")


if __name__ == "__main__":
    unittest.main()
