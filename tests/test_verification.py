import numpy as np
import pytest

from src import fem, verification as V
from src.config import load

CFG = load()


@pytest.mark.parametrize("incompatible", [True, False])
def test_patch(incompatible):
    for r in V.patch_test(CFG, incompatible):
        assert r["rel_err"] < 1e-10, r


def test_cantilever_matches_euler_bernoulli():
    (r,) = V.cantilever_test(CFG)
    assert r["rel_err"] < 0.01, r


def test_plain_hex8_locks():
    """Documents why incompatible modes are the default: plain hex8 is much too stiff."""
    (good,) = V.cantilever_test(CFG)
    (plain,) = V.cantilever_test(CFG, incompatible=False)
    assert plain["value"] < 0.9 * good["value"]


def test_stiffness_symmetric_with_six_rigid_modes():
    nodes, conn = fem.box_mesh((1.0, 0.5, 0.3), (1, 1, 1))
    K = np.asarray(fem.hex8_ke(nodes[conn[0]] * [1, 1.2, 0.9], 1.0, 0.3))
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.sum(np.abs(np.linalg.eigvalsh(K)) < 1e-10 * np.abs(K).max()) == 6
