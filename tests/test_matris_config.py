"""Regression coverage for the optimized MatRIS migration."""
import os

import numpy as np
import pytest
from ase.build import bulk

from fastmd import CUDAGraphConfig, FastMDCalculator


def test_topology_profile_binds_private_modules():
    pytest.importorskip("pymatgen")
    from fastmd._vendor.matris.config import InferenceConfig, resolve_config
    from fastmd._vendor.matris.model import functions
    before = functions._INDEXED_CAT_LINEAR_FUSE
    topology = resolve_config(InferenceConfig(optimization_profile="topology", report=False))
    generic = resolve_config(InferenceConfig(optimization_profile="generic", report=False))
    with topology.scope():
        assert functions._INDEXED_CAT_LINEAR_FUSE is True
        with generic.scope():
            assert functions._INDEXED_CAT_LINEAR_FUSE is False
        assert functions._INDEXED_CAT_LINEAR_FUSE is True
    assert functions._INDEXED_CAT_LINEAR_FUSE == before


@pytest.mark.integration
def test_matris_profiles_match_on_cpu():
    path = os.environ.get("FASTMD_MATRIS_CHECKPOINT")
    if not path:
        pytest.skip("Set FASTMD_MATRIS_CHECKPOINT")
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.positions[0] += [.02, -.01, .015]
    results = []
    for fused in (False, True):
        calc = FastMDCalculator("matris", checkpoint=path, device="cpu",
                                 cuda_graph=CUDAGraphConfig(enabled=False, enable_fusions=fused))
        calc.calculate(atoms, ["energy", "forces", "stress", "magmoms"])
        results.append(calc.results)
        assert calc.stats()["optimization_profile"] == ("topology" if fused else "generic")
    for key in ("energy", "forces", "stress", "magmoms"):
        np.testing.assert_allclose(results[0][key], results[1][key], rtol=1e-5, atol=1e-5)
