import json
import sys
from pathlib import Path
import numpy as np
import torch
from ase import units

sys.path.insert(0, str(Path.cwd() / 'examples'))
from validate_matris_npt import initial_atoms, make_calculator
from fastmd._vendor.matris.graph.gpu_graph_builder import atoms_to_graph_gpu

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
checkpoint = '../checkpoint/MatRIS_10M_OAM.pth.tar'
out = {}
for variant in ('eager_generic', 'graph_default'):
    calc = make_calculator(checkpoint, variant)
    atoms = initial_atoms(2)
    atoms.calc = calc
    ref = atoms.get_stress()
    cell, frac, volume = atoms.cell.array.copy(), atoms.get_scaled_positions(wrap=False), atoms.get_volume()
    result = {'reference': ref.tolist(), 'differences': {}}
    for h in (5e-4, 2e-4, 1e-4, 5e-5):
        fd = []
        counts = []
        for i,j in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1)):
            d=np.zeros((3,3)); d[i,j]=d[j,i]=1 if i==j else .5
            es=[]
            for sign in (1,-1):
                atoms.set_cell(cell@(np.eye(3)+sign*h*d)); atoms.set_scaled_positions(frac)
                es.append(atoms.get_potential_energy())
                graph=atoms_to_graph_gpu(atoms, atom_graph_cutoff=calc.backend.model.graph_converter.atom_graph_cutoff,
                                         line_graph_cutoff=calc.backend.model.graph_converter.line_graph_cutoff,
                                         device='cuda')
                counts.append([len(graph.undirected2directed),len(graph.line_graph)])
            fd.append((es[0]-es[1])/(2*h*volume))
        result['differences'][str(h)]={'finite':fd,'max_error':float(np.max(np.abs(fd-ref))), 'counts':counts}
        print(variant,h,fd,'ref',ref.tolist(),'counts',sorted(set(map(tuple,counts))),flush=True)
    out[variant]=result
    Path('validation/matris_npt_20260928/stress_diagnosis.json').write_text(json.dumps(out,indent=2))
    calc.clear_cache()
    del calc
    torch.cuda.empty_cache()
