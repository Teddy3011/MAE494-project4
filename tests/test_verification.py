"""Fast suite (< 2 min): every verification row, gradient checks on a 40 mm mesh, optimizer
pieces, conventions and input validation. Full-size runs live in run_all.py."""
import numpy as np
import pytest

from src import fem, verification as V, gradcheck
from src.bracket import Bracket, ks, CONSTRAINTS
from src.config import load
from src.mma import MMA
from src.optimize import kkt

CFG = load()


def _all_pass(rows):
    bad = [r for r in rows if not r["passed"]]
    assert not bad, bad


@pytest.mark.parametrize("incompatible", [True, False])
def test_patch(incompatible):
    _all_pass(V.patch_test(CFG, incompatible))


def test_cantilever_and_locking():
    (good,) = V.cantilever_test(CFG)
    (plain,) = V.cantilever_test(CFG, incompatible=False)
    _all_pass([good])
    assert plain["value"] < 0.9 * good["value"]  # plain hex8 shear-locks


def test_modal_and_buckling_coarse():
    rows, _ = V.modal_test(CFG, meshes=((10, 1, 1), (20, 2, 2)))
    rows += V.buckling_test(CFG, meshes=((20, 1, 1), (40, 2, 2)))[0]
    _all_pass(rows)


def test_global_and_bracket_checks():
    _all_pass(V.global_checks(CFG) + V.bracket_checks(CFG))


def test_gradients_directional():
    rows, _ = gradcheck.directional(CFG, h=0.04, steps=np.array([1e-4, 1e-5]))
    assert {r["function"] for r in rows} == set(gradcheck.FUNCS)
    _all_pass(rows)


def test_gradients_method_i():
    _all_pass(gradcheck.method_i_check(CFG))


def test_buckling_sign_convention():
    """Compression gives mu > 0 (alpha = 1/mu); the same load in tension gives no positive mu."""
    a_c = V.column_buckling(CFG, (20, 1, 1), P=1.0)
    assert a_c[0] > 0 and np.all(np.diff(1 / a_c) <= 1e-12)  # mu sorted descending
    a_d = V.column_buckling(CFG, (20, 1, 1), P=1.0, dense=True)
    assert abs(a_d[0] / a_c[0] - 1) < 1e-8  # sparse ARPACK agrees with dense eigh
    mu_t = 1 / V.column_buckling(CFG, (20, 1, 1), P=-1.0, dense=True)
    assert mu_t.max() <= 1e-9 * abs(mu_t).max()  # tension: no positive mu, no buckling


def test_modal_sorted_ascending():
    f, _, _ = V.beam_modal(CFG, (10, 1, 1), n=4)
    assert np.all(np.diff(f) >= 0)


def test_constraint_conventions():
    m = Bracket(CFG, 0.04)
    solid = m.analyze(np.ones(m.n), 3.0, 1.0, grad=False)
    assert np.all(solid["g"] < 0), dict(zip(CONSTRAINTS, solid["g"]))  # solid domain is feasible
    void = m.analyze(np.full(m.n, 1e-3), 3.0, 1.0, grad=False)
    assert void["g"][CONSTRAINTS.index("freq")] > 0  # almost no material: frequency violated
    v = np.array([0.3, 1.0, 0.99])
    k, w = ks(v, 50.0)
    assert k >= v.max() and abs(w.sum() - 1) < 1e-14


def test_invalid_input():
    with pytest.raises(ValueError):
        fem.check_material(1.0, 0.5)
    with pytest.raises(ValueError):
        Bracket(CFG, 0.03)  # does not divide the domain
    m = Bracket(CFG, 0.04)
    with pytest.raises(ValueError):
        m.analyze(np.full(m.n, np.nan), 3.0, 1.0)


def test_mma_and_kkt_on_known_problem():
    """min sum(x) s.t. sum(1/x) - n <= 0: optimum x = 1, multiplier 1."""
    n = 5
    mma = MMA(np.full(n, 0.1), np.full(n, 10.0), move=0.5)
    x = np.linspace(2.0, 6.0, n)
    for _ in range(100):
        x, lam, _ = mma.step(x, np.ones(n), np.array([np.sum(1 / x) - n]), -(1 / x ** 2)[None])
    assert np.allclose(x, 1.0, atol=1e-6) and abs(lam[0] - 1) < 1e-4
    r = dict(df=np.ones(n), g=np.array([np.sum(1 / x) - n]), dg=-(1 / x ** 2)[None])
    k = kkt(r, x, 0.1, 10.0, [0])
    assert abs(k["multipliers"]["stress_LC1"] - 1) < 1e-4 and k["stationarity"] < 1e-5
