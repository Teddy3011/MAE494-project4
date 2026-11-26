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


def test_mesh_transfer_preserves_model():
    """Splitting every 20 mm voxel into eight 10 mm voxels keeps the physical model."""
    from src.report import _upsample
    c, f = Bracket(CFG, 0.02), Bracket(CFG, 0.01)
    rho = np.random.default_rng(0).uniform(0.01, 1.0, c.ne)
    up = _upsample(rho, c.nel)
    assert np.isclose(rho.sum() * c.h ** 3, up.sum() * f.h ** 3, rtol=1e-12)  # volume (and mass)
    assert np.array_equal(_upsample(c.passive.astype(float), c.nel) > 0, f.passive)  # passive regions
    for m in (c, f):
        xs, xi = m.nodes[m.support_nodes], m.nodes[m.iface_nodes]
        assert np.isclose(len(m.support_nodes) and (xs[:, 0].max() - xs[:, 0].min()), 0.2)
        assert np.allclose([np.ptp(xi[:, 0]), np.ptp(xi[:, 1])], [0.08, 0.04])  # interface patch
        F = m.F.reshape(-1, 3, 3)
        assert np.allclose(F.sum(0)[2], -m.load_magnitude)  # total vertical force, every LC
        dx = m.nodes[:, 0] - m.L[0] / 2
        assert np.isclose(-(dx * (F[:, 2, 1] - F[:, 2, 0])).sum(), m.torque)  # torque about y
        assert np.isclose(m.M_engine.diagonal().sum(), 3 * CFG["loads"]["engine_mass"])  # lumped engine mass
        q = (xs[:, 0] > m.L[0] / 2) * 2 + (xs[:, 1] > m.L[1] / 2)  # quadrant of each support node
        assert sorted(set(q)) == [0, 1, 2, 3]  # four bolt pads
        for k in range(4):
            assert np.allclose(np.ptp(xs[q == k][:, :2], axis=0), CFG["geometry"]["pad_size"])  # 40 x 40 mm each


def test_selection_rule_needs_refined_margin():
    """The lightest candidate is rejected when its refined-mesh frequency margin is below 1 Hz."""
    import copy
    import run_all
    cfg = copy.deepcopy(CFG)
    cfg["sweep"]["targets"] = [105.0, 106.0, 107.0]

    def cand(mass, f5, g=-1e-6):
        return dict(metrics=dict(mass=mass, g={"freq": g}, alpha_min=200.0, connectivity=dict(passive_connected=True, components=1)),
                    refined=dict(freq_Hz=[f5], pnorm_MPa=[100.0], disp_mm=[0.05]))
    runs = {"B_105": cand(1.30, 100.5), "B_106": cand(1.32, 101.2), "B_107": cand(1.35, 102.0)}
    target, rows = run_all.select(cfg, runs)
    assert target == 106.0 and not rows[0]["ok"]
    runs["B_106"] = cand(1.32, 101.2, g=2e-5)  # smooth constraint not met to 1e-5
    assert run_all.select(cfg, runs)[0] == 107.0
    runs = {k: cand(1.3, f5, g=2e-5) for k, f5 in [("B_105", 100.2), ("B_106", 100.6), ("B_107", 100.4)]}
    target, rows = run_all.select(cfg, runs)  # nothing passes: highest refined frequency, every row flagged
    assert target == 106.0 and not any(r["ok"] for r in rows)


def test_kkt_without_active_constraints():
    """No near-active constraint: no multipliers, and stationarity is just the objective gradient."""
    r = dict(df=np.ones(4), g=np.array([-0.5]), dg=np.ones((1, 4)))
    k = kkt(r, np.full(4, 0.5), 0.0, 1.0, [0])
    assert k["multipliers"] == {} and abs(k["stationarity"] - 1.0) < 1e-12 and k["dual_feasible"]
