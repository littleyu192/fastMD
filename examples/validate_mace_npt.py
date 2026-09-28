"""Validate and benchmark MACE ASE NPT against the official calculator.

Run on an otherwise idle CUDA GPU. Each timed trajectory starts from the same
seeded state; capture/neighbor maintenance and ASE integration are included.
"""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
from ase import units
from ase.build import bulk
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator

from fastmd import FastMDCalculator
from fastmd._vendor.mace_opt.tensor_batch import default_dtype


class CountedMACECalculator(MACECalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def calculate(self, *args, **kwargs):
        self.calls += 1
        return super().calculate(*args, **kwargs)


def counters(calc):
    if isinstance(calc, FastMDCalculator):
        stats = calc.stats()
        return dict(calls=stats['calls'], captures=stats['cache'].get('captures', 0),
                    invalidations=stats['invalidations'])
    return dict(calls=calc.calls, captures=0, invalidations=0)


def initial_atoms(repeat):
    atoms = bulk('Si', 'diamond', a=5.43, cubic=True).repeat((repeat,)*3)
    atoms.positions += np.random.default_rng(2026).normal(0, .01, atoms.positions.shape)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))
    Stationary(atoms)
    return atoms


def trajectory(calc, initial, steps, record=False):
    atoms = initial.copy()
    atoms.calc = calc
    calc.reset()
    before = counters(calc)
    frames = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    with NPTBerendsen(atoms, timestep=.5*units.fs, temperature_K=300,
                     pressure_au=units.GPa, compressibility_au=1/(100*units.GPa),
                     taut=100*units.fs, taup=1000*units.fs) as dyn:
        if record:
            for _ in range(steps):
                dyn.run(1)
                frames.append(dict(positions=atoms.positions.copy(), cell=atoms.cell.array.copy(),
                    momenta=atoms.get_momenta().copy(), energy=atoms.get_potential_energy(),
                    forces=atoms.get_forces().copy(), stress=atoms.get_stress().copy()))
        else:
            dyn.run(steps)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    assert np.isfinite(atoms.positions).all() and np.isfinite(atoms.get_momenta()).all()
    assert np.max(np.abs(atoms.cell.array-initial.cell.array)) > 1e-7
    row = {k: v-before[k] for k,v in counters(calc).items()}
    if not record:
        assert row['calls'] == 2*steps+1, row
    row.update(seconds=seconds, ms_per_step=seconds*1000/steps,
               final_volume=atoms.get_volume(), final_temperature=atoms.get_temperature())
    return row, frames


def compare(actual, reference):
    errors = {}
    tolerances = dict(positions=2e-6, cell=2e-7, momenta=2e-5,
                      energy=2e-5, forces=2e-5, stress=2e-6)
    for key, atol in tolerances.items():
        a = np.asarray([r[key] for r in actual])
        b = np.asarray([r[key] for r in reference])
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=atol,err_msg=key)
        errors[key] = float(np.max(np.abs(a-b)))
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mp0', required=True, type=Path)
    p.add_argument('--mpa0', required=True, type=Path)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, default=Path('mace_npt_results.json'))
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    report = dict(status='RUNNING', started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        steps=args.steps,repeats=args.repeats,dtype='float64',
        environment=dict(job=os.getenv('SLURM_JOB_ID'),step=os.getenv('SLURM_STEP_ID'),
            node=platform.node(),python=platform.python_version(),torch=torch.__version__,
            cuda=torch.version.cuda,ase=version('ase'),mace=version('mace-torch'),e3nn=version('e3nn'),
            gpu=torch.cuda.get_device_name(),threads=torch.get_num_threads(),
            gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip(),
            tf32=torch.backends.cuda.matmul.allow_tf32),results={})
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n')
        tmp.replace(args.output)
    save()
    try:
        with default_dtype(torch.float64):
            for kind,path in [('mp0',args.mp0),('mpa0',args.mpa0)]:
                entry = dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sizes={})
                report['results'][kind] = entry
                for repeat in (2,3):
                    initial=initial_atoms(repeat)
                    results=entry['sizes'][str(len(initial))]={}
                    reference = None
                    for mode in ('official','eager','graph_plain','graph_fast'):
                        print(time.strftime('%T'),'START',kind,len(initial),mode,flush=True)
                        load=time.perf_counter()
                        if mode=='official':
                            calc=CountedMACECalculator(model_paths=str(path),device='cuda',default_dtype='float64')
                        else:
                            calc=FastMDCalculator('mace',checkpoint=str(path),device='cuda',
                                cuda_graph=mode.startswith('graph'),
                                model_kwargs=dict(compute_stress=True,default_dtype='float64',
                                                  variant='plain' if mode=='graph_plain' else 'fast'))
                        result=results[mode]=dict(load_seconds=time.perf_counter()-load,trials=[])
                        result['warmup'],_=trajectory(calc,initial,args.steps)
                        print(time.strftime('%T'),'WARMUP',result['warmup'],flush=True)
                        for trial in range(args.repeats):
                            row,_=trajectory(calc,initial,args.steps)
                            result['trials'].append(row)
                            print(time.strftime('%T'),'TRIAL',trial+1,row,flush=True)
                            if mode.startswith('graph'):
                                assert row['captures']==row['invalidations']==0,row
                            save()
                        result['median_ms_per_step']=statistics.median(r['ms_per_step'] for r in result['trials'])
                        _,frames=trajectory(calc,initial,20,record=True)
                        result['trajectory_max_abs_error']=compare(frames,reference) if reference is not None else {}
                        if reference is None:
                            reference=frames
                        if isinstance(calc,FastMDCalculator):
                            result['stats']=calc.stats()
                            if mode.startswith('graph'):
                                assert calc.stats()['mode']=='cuda_graph' and calc.stats()['eager_calls']==0
                            calc.clear_cache()
                        save()
                        print(time.strftime('%T'),'DONE',kind,len(initial),mode,flush=True)
                        del calc
                        gc.collect()
                        torch.cuda.empty_cache()
        report['status']='PASS'
    except BaseException:
        import traceback
        report['status']='FAIL'
        report['error']=traceback.format_exc()
        raise
    finally:
        report['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        save()


if __name__=='__main__':
    main()
