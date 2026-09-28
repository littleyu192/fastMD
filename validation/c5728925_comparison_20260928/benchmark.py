"""Benchmark an unmodified exported revision using ordinary ASE MD.

The subclass only counts captures/tasks after calculate(); it changes no model,
cache, property request, graph lifetime, or integrator behavior.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import weakref

os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
import ase
from ase import units
from ase.build import bulk
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
import torch
import fastmd
from fastmd import FastMDCalculator

COMMIT = 'c5728925f0fe13503ad5aa081d2e4db3733040d5'

class CountedCalculator(FastMDCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_captures = 0
        self.tasks = Counter()

    def calculate(self, *args, **kwargs):
        previous = weakref.ref(self.backend.runner) if self.backend.runner is not None else None
        captures = self.backend.runner.captures if self.backend.runner is not None else 0
        super().calculate(*args, **kwargs)
        runner = self.backend.runner
        if runner is not None:
            self.total_captures += runner.captures - (captures if previous is not None and runner is previous() else 0)
            self.tasks[runner.task] += 1
        else:
            self.tasks[self.backend.calculator.task] += 1


def initial_atoms(repeat):
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((repeat,)*3)
    atoms.positions += np.random.default_rng(2026).normal(0, .01, atoms.positions.shape)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))
    Stationary(atoms)
    return atoms


def trajectory(calc, initial, ensemble, steps, save=False):
    atoms = initial.copy()
    atoms.calc = calc
    calc.reset()
    captures_before = calc.total_captures
    calls_before = calc.stats()['calls']
    invalidations_before = calc.stats()['invalidations']
    tasks_before = calc.tasks.copy()
    torch.cuda.synchronize()
    start = time.perf_counter()
    if ensemble == 'npt':
        dyn = NPTBerendsen(atoms, timestep=.5*units.fs, temperature_K=300,
                          pressure_au=units.GPa, taut=100*units.fs,
                          taup=1000*units.fs, compressibility_au=1/(100*units.GPa))
    else:
        dyn = VelocityVerlet(atoms, timestep=.5*units.fs)
    frames = []
    with dyn:
        if save:
            for _ in range(steps):
                dyn.run(1)
                frames.append(dict(positions=atoms.positions.copy(), cell=atoms.cell.array.copy(),
                                   momenta=atoms.get_momenta().copy()))
        else:
            dyn.run(steps)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    assert np.isfinite(atoms.positions).all() and np.isfinite(atoms.cell.array).all()
    if ensemble == 'npt':
        assert np.max(np.abs(atoms.cell.array-initial.cell.array)) > 1e-8
    else:
        np.testing.assert_array_equal(atoms.cell.array, initial.cell.array)
    return dict(seconds=seconds, ms_per_step=seconds*1000/steps,
                model_calls=calc.stats()['calls']-calls_before,
                captures_during_run=calc.total_captures-captures_before,
                invalidations_during_run=calc.stats()['invalidations']-invalidations_before,
                task_calls=dict(calc.tasks-tasks_before), final_volume=atoms.get_volume(),
                final_temperature=atoms.get_temperature()), frames


def frame_error(actual, reference):
    out = {}
    for key, tol in [('positions', 2e-4), ('cell', 2e-5), ('momenta', 2e-3)]:
        a=np.asarray([x[key] for x in actual]);b=np.asarray([x[key] for x in reference])
        np.testing.assert_allclose(a,b,atol=tol,rtol=1e-5,err_msg=key)
        out[key] = float(np.max(np.abs(a-b)))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--ensemble',choices=['npt','nve','both'],default='npt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    assert Path(fastmd.__file__).resolve().is_relative_to(args.source.resolve())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    checkpoint=Path(args.checkpoint)
    report=dict(status='RUNNING',commit=COMMIT,source=str(args.source),fastmd_file=fastmd.__file__,
                started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), steps=args.steps,repeats=args.repeats,
                checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                environment=dict(node=platform.node(),job=os.getenv('SLURM_JOB_ID'),step=os.getenv('SLURM_STEP_ID'),
                                 gpu=torch.cuda.get_device_name(),torch=torch.__version__,ase=ase.__version__,
                                 cuda=torch.version.cuda,python=platform.python_version(),threads=torch.get_num_threads(),
                                 tf32=torch.backends.cuda.matmul.allow_tf32,
                                 gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip()),
                results={})
    def save():
        tmp=args.output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output)
    save()
    try:
        ensembles=['npt','nve'] if args.ensemble=='both' else [args.ensemble]
        for ensemble in ensembles:
            report['results'][ensemble]={}
            for repeat in (2,3):
                atoms=initial_atoms(repeat);size=str(len(atoms))
                report['results'][ensemble][size]={}
                reference=None
                for mode in ('eager_default','graph_default'):
                    enabled=mode.startswith('graph')
                    print(time.strftime('%T'), 'START',ensemble,size,mode,flush=True)
                    start=time.perf_counter()
                    calc=CountedCalculator('matris',checkpoint=str(checkpoint),device='cuda',cuda_graph=enabled)
                    load=time.perf_counter()-start
                    cold,_=trajectory(calc,atoms,ensemble,args.steps)
                    print(time.strftime('%T'),'WARMUP',ensemble,size,mode,cold,flush=True)
                    trials=[]
                    for trial in range(args.repeats):
                        row,_=trajectory(calc,atoms,ensemble,args.steps)
                        trials.append(row)
                        print(time.strftime('%T'),'TRIAL',ensemble,size,mode,trial+1,row,flush=True)
                        report['results'][ensemble][size][mode]=dict(load_seconds=load,warmup=cold,trials=trials,
                            median_ms_per_step=statistics.median(t['ms_per_step'] for t in trials))
                        save()
                    _,frames=trajectory(calc,atoms,ensemble,20,save=True)
                    errors=frame_error(frames,reference) if reference is not None else {}
                    if reference is None:reference=frames
                    if enabled:
                        assert calc.stats()['eager_calls']==0 and calc.stats()['mode']=='cuda_graph'
                        assert calc.total_captures > 0
                    report['results'][ensemble][size][mode].update(trajectory_errors=errors,
                        total_captures=calc.total_captures,stats={k:v for k,v in calc.stats().items() if k!='kernel_options'})
                    save()
                    print(time.strftime('%T'),'DONE',ensemble,size,mode,flush=True)
                    calc.clear_cache();del calc
                    import gc
                    gc.collect();torch.cuda.empty_cache()
        report['status']='PASS'
    except BaseException:
        import traceback
        report['status']='FAIL';report['error']=traceback.format_exc()
        raise
    finally:
        report['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()

if __name__=='__main__':main()
