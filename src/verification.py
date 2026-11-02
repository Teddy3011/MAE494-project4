"""Verification ladder. Each check returns rows {id, name, value, exact, rel_err, note}.

Exact values come from closed-form theory written here, never from the solver.
Run `python -m src.verification` to write results/verification.csv.
"""
import csv
import numpy as np

from . import fem
from .config import ROOT, load


def _row(id, name, value, exact, rel_err=None, note=""):
    if rel_err is None:
        rel_err = abs(value - exact) / abs(exact)
    return dict(id=id, name=name, value=float(value), exact=float(exact), rel_err=float(rel_err), note=note)


def patch_test(cfg, incompatible=True):
    """V1: distorted 3x3x3 mesh, linear displacement u = A x imposed on the boundary.

    Exact: interior nodes follow u = A x and every element has sigma = D sym(A).
    """
    E, nu = cfg["material"]["E"], cfg["material"]["nu"]
    rng = np.random.default_rng(cfg["seed"])
    ref, conn = fem.box_mesh((1.0, 1.0, 1.0), (3, 3, 3))
    on_bnd = np.any(np.isclose(ref, 0.0) | np.isclose(ref, 1.0), axis=1)
    h = 1.0 / 3
    nodes = ref + rng.uniform(-0.2 * h, 0.2 * h, ref.shape)  # distort every node
    A = rng.uniform(-1e-3, 1e-3, (3, 3))
    u_exact = nodes @ A.T

    fixed = (3 * np.flatnonzero(on_bnd)[:, None] + np.arange(3)).ravel()
    u, _, stress = fem.linear_static(nodes, conn, E, nu, np.zeros(3 * len(nodes)),
                                     fixed, u_exact.ravel()[fixed], incompatible)

    interior = (3 * np.flatnonzero(~on_bnd)[:, None] + np.arange(3)).ravel()
    ue, ui = u_exact.ravel()[interior], u[interior]
    eps = 0.5 * (A + A.T)
    eps_v = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])
    s_exact = np.asarray(fem.dmat(E, nu)) @ eps_v
    s_err = np.max(np.linalg.norm(stress - s_exact, axis=1)) / np.linalg.norm(s_exact)
    tag = "" if incompatible else " (plain hex8)"
    return [
        _row("V1a", "Patch test: interior displacement norm" + tag, np.linalg.norm(ui), np.linalg.norm(ue),
             np.linalg.norm(ui - ue) / np.linalg.norm(ue), "distorted 3x3x3 mesh; rel_err = ||u-u_ex||/||u_ex||"),
        _row("V1b", "Patch test: sigma_xx [Pa]" + tag, stress[0, 0], s_exact[0],
             s_err, "rel_err = max over elements of ||sigma-sigma_ex||/||sigma_ex||"),
    ]


def cantilever_test(cfg, nel=(20, 2, 2), incompatible=True):
    """V2: clamped cantilever, tip shear load. Exact (Euler-Bernoulli): d = P L^3 / (3 E I).

    nu = 0 so the clamp introduces no Poisson restraint EB theory does not have.
    Timoshenko shear adds P L / (kappa G A), about 0.15 % at L/h = 20; reported in the note.
    """
    E = cfg["material"]["E"]
    nu = 0.0
    L, b, h, P = 0.2, 0.01, 0.01, 100.0
    nodes, conn = fem.box_mesh((L, b, h), nel)
    root = np.flatnonzero(np.isclose(nodes[:, 0], 0.0))
    fixed = (3 * root[:, None] + np.arange(3)).ravel()

    # uniform shear traction on the tip face -> consistent bilinear nodal weights
    tip = np.flatnonzero(np.isclose(nodes[:, 0], L))
    wy = np.interp(nodes[tip, 1], *_trap(b, nel[1]))
    wz = np.interp(nodes[tip, 2], *_trap(h, nel[2]))
    f = np.zeros(3 * len(nodes))
    f[3 * tip + 2] = -P * wy * wz / (b * h)

    u, _, _ = fem.linear_static(nodes, conn, E, nu, f, fixed, 0.0, incompatible)
    d = -u[3 * tip + 2].mean()
    I = b * h ** 3 / 12
    d_eb = P * L ** 3 / (3 * E * I)
    d_timo = d_eb + P * L / (5 / 6 * E / 2 * b * h)
    tag = "" if incompatible else " (plain hex8, shear locking)"
    return [_row("V2" if incompatible else "V2b", f"Cantilever tip deflection [m], mesh {nel}" + tag,
                 d, d_eb, note=f"L/h=20, nu=0; Timoshenko {d_timo:.6e} m (rel diff {abs(d - d_timo) / d_timo:.2e})")]


def _trap(length, n):
    """Node positions and trapezoid weights for n equal segments."""
    x = np.linspace(0.0, length, n + 1)
    w = np.full(n + 1, length / n)
    w[[0, -1]] *= 0.5
    return x, w


def run_all(cfg):
    return (patch_test(cfg) + cantilever_test(cfg)
            + cantilever_test(cfg, incompatible=False))


def write_csv(rows, path=ROOT / "results" / "verification.csv"):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    rows = run_all(load())
    for r in rows:
        print(f"{r['id']:4s} {r['name']:60s} value={r['value']:.6e} exact={r['exact']:.6e} rel_err={r['rel_err']:.2e}")
    print("wrote", write_csv(rows))
