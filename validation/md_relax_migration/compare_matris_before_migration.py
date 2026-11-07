"""Reproduce the 26 tensor comparisons already observed during the migration review.

Run from FastMD with PYTHONPATH=. and the records2610 CHGNet Python environment.
This script was assembled from the two successful review commands; it has not
been executed again after assembly. The baseline commit is pinned below.
"""
import copy
import subprocess
import types
import torch
from tests.test_isolated_atom_handling import make_graph, make_model
from fastmd.execution.cuda_graph import pad_radius_graph

BASELINE = '4961fc018b37e58dc2edc4982614d92f555ab626'

def load(path, name):
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'], text=True)
    module = types.ModuleType(name)
    module.__file__ = path
    exec(compile(source, path, 'exec'), module.__dict__)
    return module

old = load('fastmd/models/matris/model.py', 'old_model')
old.process_graphs = load('fastmd/models/matris/processgraph.py', 'old_processgraph').process_graphs
checks = 0
for intensive in (True, False):
    torch.manual_seed(922)
    model = make_model(is_intensive=intensive)
    pair, lone = make_graph(2, True), make_graph(1, False)
    padded, counts = [], []
    for connected in (True, False):
        graph, count = pad_radius_graph(make_graph(2, connected), u_max=4, t_max=8, n_dummy=4)
        padded.append(graph)
        counts.append(count)
    cases = [('single', [pair], None), ('list_batch', [pair, lone], None),
             ('padded_list_batch', padded, counts)]
    for name, graphs, real in cases:
        actual = model(copy.deepcopy(graphs), task='efs', is_training=False, n_real=real)
        model._graph_sum = types.MethodType(old.MatRIS._graph_sum, model)
        try:
            expected = old.MatRIS._forward(model, copy.deepcopy(graphs), 'efs', False, real, True)
        finally:
            del model._graph_sum
        for key in ('e', 'f', 's'):
            xs = [actual[key]] if isinstance(actual[key], torch.Tensor) else actual[key]
            ys = [expected[key]] if isinstance(expected[key], torch.Tensor) else expected[key]
            assert len(xs) == len(ys)
            for x, y in zip(xs, ys):
                if not torch.equal(x, y):
                    raise AssertionError((intensive, name, key, float((x - y).abs().max())))
                checks += 1
        print(f'bitwise same: intensive={intensive}, {name}')
assert checks == 26, checks
print(f'tensor comparisons: {checks}; baseline: {BASELINE}')
