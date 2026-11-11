"""Verification ladder V1-V5 (V6, gradients, lives in src/gradcheck.py).

Each check returns rows {id, name, value, exact, error, tol, passed, note}. Exact values come
from closed-form theory written here (hand derivations noted in each docstring), never
from the solver. `error` is relative unless the note says otherwise.
"""
import csv
import numpy as np

from . import fem
from .config import ROOT, load

BETA1_L = 1.8751040687119611  # first root of cos(x) cosh(x) = -1 (clamped-free beam)


def _row(id, name, value, exact, tol, error=None, note=""):
    if error is None:
        error = abs(value - exact) / abs(exact)
    return dict(id=id, name=name, value=float(value), exact=float(exact), error=float(error),
                tol=float(tol), passed=bool(error <= tol), note=note)


def _trap(length, n):
    """Node positions and trapezoid weights for n equal segments."""
    x = np.linspace(0.0, length, n + 1)
    w = np.full(n + 1, length / n)
    w[[0, -1]] *= 0.5
    return x, w


def _end_face_load(nodes, x_end, b, h, nel, total, direction):
    """Uniform traction resultant `total` on the face x = x_end (consistent bilinear weights)."""
    face = np.flatnonzero(np.isclose(nodes[:, 0], x_end))
    w = np.interp(nodes[face, 1], *_trap(b, nel[1])) * np.interp(nodes[face, 2], *_trap(h, nel[2])) / (b * h)
    f = np.zeros(3 * len(nodes))
    f[3 * face + direction] = total * w
    return f, face


def _clamp_x0(nodes):
    return fem.node_dofs(np.flatnonzero(np.isclose(nodes[:, 0], 0.0)))


# ---------------------------------------------------------------- V1 patch test

def patch_test(cfg, incompatible=True):
    """V1: distorted 3x3x3 mesh, linear displacement u = A x imposed on the boundary.
    Exact: interior nodes follow u = A x; every Gauss point has sigma = D sym(A)."""
    E, nu = cfg["material"]["E"], cfg["material"]["nu"]
    rng = np.random.default_rng(cfg["seed"])
    ref, conn = fem.box_mesh((1.0, 1.0, 1.0), (3, 3, 3))
    on_bnd = np.any(np.isclose(ref, 0.0) | np.isclose(ref, 1.0), axis=1)
    h = 1.0 / 3
    nodes = ref + rng.uniform(-0.2 * h, 0.2 * h, ref.shape)  # distort every node
    A = rng.uniform(-1e-3, 1e-3, (3, 3))
    u_exact = nodes @ A.T
    fixed = fem.node_dofs(np.flatnonzero(on_bnd))
    u, _, stress = fem.linear_static(nodes, conn, E, nu, np.zeros(3 * len(nodes)),
                                     fixed, u_exact.ravel()[fixed], incompatible)
    interior = fem.node_dofs(np.flatnonzero(~on_bnd))
    ue, ui = u_exact.ravel()[interior], u[interior]
    eps = 0.5 * (A + A.T)
    eps_v = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])
    s_exact = np.asarray(fem.dmat(E, nu)) @ eps_v
    s = stress.reshape(-1, 6)
    s_err = np.max(np.linalg.norm(s - s_exact, axis=1)) / np.linalg.norm(s_exact)
    tag, sfx = ("incompatible", "a") if incompatible else ("plain hex8", "b")
    return [
        _row(f"V1.{sfx}1", f"Patch test, interior displacement ({tag})", np.linalg.norm(ui), np.linalg.norm(ue), 1e-10,
             np.linalg.norm(ui - ue) / np.linalg.norm(ue), "distorted 3x3x3 mesh, all nodes moved up to 20% of h; error = ||u-u_ex||/||u_ex||"),
        _row(f"V1.{sfx}2", f"Patch test, Gauss-point stress sigma_xx [Pa] ({tag})", s[0, 0], s_exact[0], 1e-10,
             s_err, "error = max over all 216 Gauss points of ||sigma-sigma_ex||/||sigma_ex||"),
    ]


# ---------------------------------------------------------------- V2 static cantilever

def cantilever_test(cfg, nel=(20, 2, 2), incompatible=True):
    """V2: clamped cantilever, tip shear P. Euler-Bernoulli: d = P L^3/(3 E I).
    nu = 0 so the clamp adds no Poisson restraint that beam theory lacks.
    Timoshenko adds P L/(kappa G A), kappa = 5/6: about 0.15 % at L/h = 20."""
    E, nu = cfg["material"]["E"], 0.0
    L, b, h, P = 0.2, 0.01, 0.01, 100.0
    nodes, conn = fem.box_mesh((L, b, h), nel)
    f, tip = _end_face_load(nodes, L, b, h, nel, -P, 2)
    u, _, _ = fem.linear_static(nodes, conn, E, nu, f, _clamp_x0(nodes), 0.0, incompatible)
    d = -u[3 * tip + 2].mean()
    I = b * h ** 3 / 12
    d_eb = P * L ** 3 / (3 * E * I)
    d_timo = d_eb + P * L / (5 / 6 * E / 2 * b * h)
    if incompatible:
        return [_row("V2.a", f"Cantilever tip deflection [m], mesh {nel}", d, d_eb, 5e-3,
                     note=f"vs Euler-Bernoulli; Timoshenko {d_timo:.6e} m, difference {abs(d - d_timo) / d_timo:.1e}")]
    return [_row("V2.b", f"Cantilever tip deflection [m], mesh {nel}, plain hex8", d, d_eb, 0.5,
                 note="expected to FAIL a tight tolerance: shear locking (33 % too stiff). "
                      "Tolerance 0.5 only documents the size of the locking error; this is why incompatible modes are used.")]


# ---------------------------------------------------------------- V3 modal

def beam_modal(cfg, nel, n=4):
    """Clamped-free beam along x, square section. Returns (f [Hz], phi, M, free)."""
    E, nu, rho = cfg["material"]["E"], 0.0, cfg["material"]["rho"]
    L, b, h = 0.2, 0.01, 0.01
    nodes, conn = fem.box_mesh((L, b, h), nel)
    Xe = nodes[conn]
    K = fem.assemble(conn, fem.ke_batch(Xe, np.full(len(conn), E), nu, True), 3 * len(nodes))
    M = fem.assemble(conn, fem.me_batch(Xe, np.full(len(conn), rho)), 3 * len(nodes))
    free = fem.free_dofs(3 * len(nodes), _clamp_x0(nodes))
    Kff, Mff = K[free][:, free], M[free][:, free]
    lam, phi = fem.modal(Kff, Mff, n, fem.SPDSolver(Kff))
    return np.sqrt(lam) / (2 * np.pi), phi, Mff


def modal_test(cfg, meshes=((10, 1, 1), (20, 2, 2), (40, 4, 4))):
    """V3: first bending frequency of a clamped-free beam,
    f1 = (beta1 L)^2/(2 pi L^2) sqrt(E I/(rho A)), beta1 L = 1.87510 (L = 0.2, b = h = 0.01 m: 204.1 Hz).
    Square section: the y- and z-bending modes are a repeated pair."""
    E, rho = cfg["material"]["E"], cfg["material"]["rho"]
    L, b, h = 0.2, 0.01, 0.01
    I, A = b * h ** 3 / 12, b * h
    f_exact = BETA1_L ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * I / (rho * A))
    rows, conv = [], []
    for nel in meshes:
        f, phi, Mff = beam_modal(cfg, nel)
        conv.append((L / nel[0], f[0]))
        rows.append(_row(f"V3.{len(rows) + 1}", f"Cantilever f1 [Hz], mesh {nel}", f[0], f_exact,
                         0.01 if nel == meshes[-1] else 1.0,
                         note="consistent mass; finest mesh must be within 1 %, coarser rows show convergence"))
    rows.append(_row("V3.p", "Repeated pair: f2/f1 (square section)", f[1] / f[0], 1.0, 1e-6,
                     note="y and z bending are degenerate by symmetry"))
    norm = np.einsum("ij,ij->j", phi, Mff @ phi)
    rows.append(_row("V3.m", "Mass normalization max |phi^T M phi - 1|", 1.0 + np.max(np.abs(norm - 1)), 1.0, 1e-12,
                     np.max(np.abs(norm - 1)), "absolute error"))
    off = phi.T @ Mff @ phi - np.eye(phi.shape[1])
    rows.append(_row("V3.o", "M-orthogonality max |phi_i^T M phi_j|, i != j", np.max(np.abs(off)), 0.0, 1e-10,
                     np.max(np.abs(off)), "absolute error; holds inside the repeated pair too"))
    return rows, conv


# ---------------------------------------------------------------- V4 buckling

def column_buckling(cfg, nel, L=0.4, b=0.01, h=0.01, P=1.0, dense=False):
    """Fixed-free column along x, compressive dead load P on the free end. Returns alpha (load factors).
    dense=True uses scipy.linalg.eigh on the full spectrum (small meshes; works for tension too)."""
    E, nu = cfg["material"]["E"], 0.0
    nodes, conn = fem.box_mesh((L, b, h), nel)
    Xe = nodes[conn]
    Ee = np.full(len(conn), E)
    f, _ = _end_face_load(nodes, L, b, h, nel, -P, 0)
    fixed = _clamp_x0(nodes)
    u, K, _ = fem.linear_static(nodes, conn, E, nu, f, fixed)
    KG = fem.assemble(conn, fem.kg_batch(Xe, u[fem.element_dofs(conn)], Ee, nu, True), 3 * len(nodes))
    free = fem.free_dofs(3 * len(nodes), fixed)
    Kff = K[free][:, free]
    if dense:
        import scipy.linalg as sl
        return 1.0 / sl.eigh(-KG[free][:, free].toarray(), Kff.toarray(), eigvals_only=True)[::-1]
    mu, _ = fem.buckling(Kff, KG[free][:, free], 4, fem.SPDSolver(Kff))
    return 1.0 / mu


def buckling_test(cfg, meshes=((20, 1, 1), (40, 2, 2), (80, 4, 4))):
    """V4: Euler column, clamped base (all DOFs on x = 0) and free loaded tip, the K = 2 case:
    P_cr = pi^2 E I / (2 L)^2 (L = 0.4, b = h = 0.01 m: 886.7 N). The FE restraint is a full
    clamp at x = 0 and nothing at x = L, which is exactly the fixed-free condition.
    Engesser shear correction is 1/(1 + P_cr/(kappa G A)) = 0.9997, negligible."""
    E = cfg["material"]["E"]
    L, b, h = 0.4, 0.01, 0.01
    I = b * h ** 3 / 12
    P_cr = np.pi ** 2 * E * I / (2 * L) ** 2
    rows, conv = [], []
    for nel in meshes:
        a = column_buckling(cfg, nel)
        conv.append((L / nel[0], a[0]))
        rows.append(_row(f"V4.{len(rows) + 1}", f"Euler column P_cr [N], mesh {nel}", a[0], P_cr,
                         0.01 if nel == meshes[-1] else 1.0, note="fixed-free; finest mesh within 1 %"))
    rows.append(_row("V4.p", "Repeated pair: alpha2/alpha1 (square section)", a[1] / a[0], 1.0, 1e-6,
                     note="buckling about y and z is degenerate"))
    rows.append(_row("V4.2", "Second mode ratio alpha3/alpha1", a[2] / a[0], 9.0, 0.02,
                     note="fixed-free modes scale as (2n-1)^2: second distinct mode = 9 P_cr"))
    return rows, conv


# ---------------------------------------------------------------- V5 global and invariance

def global_checks(cfg):
    E, nu, rho = cfg["material"]["E"], cfg["material"]["nu"], cfg["material"]["rho"]
    rows = []

    # rigid-body modes of an unsupported block: exactly six zero eigenvalues
    nodes, conn = fem.box_mesh((0.03, 0.02, 0.01), (3, 2, 1))
    Xe = nodes[conn]
    K = fem.assemble(conn, fem.ke_batch(Xe, np.full(len(conn), E), nu, True), 3 * len(nodes)).toarray()
    ev = np.linalg.eigvalsh(K)
    nzero = int(np.sum(np.abs(ev) < 1e-9 * ev.max()))
    rows.append(_row("V5.1", "Free-free block: number of zero eigenvalues", nzero, 6, 0.0,
                     note=f"|lambda| < 1e-9 lambda_max; 7th/max eigenvalue ratio {ev[6] / ev.max():.2e}"))

    # total consistent mass of a box equals rho V (x translation)
    M = fem.assemble(conn, fem.me_batch(Xe, np.full(len(conn), rho)), 3 * len(nodes))
    ex = np.zeros(3 * len(nodes))
    ex[0::3] = 1.0
    rows.append(_row("V5.2", "Consistent mass: 1x^T M 1x [kg]", ex @ M @ ex, rho * 0.03 * 0.02 * 0.01, 1e-12,
                     note="equals rho V for a rigid translation"))

    # matrix properties on one distorted element
    rng = np.random.default_rng(cfg["seed"])
    X = fem.XI * 0.005 + rng.uniform(-0.001, 0.001, (8, 3))
    ue = rng.uniform(-1e-5, 1e-5, 24)
    for name, Mt in [("K_e", fem.hex8_ke(X, E, nu)), ("M_e", fem.hex8_me(X, rho)), ("K_G,e", fem.hex8_kg(X, ue, E, nu))]:
        Mt = np.asarray(Mt)
        asym = np.linalg.norm(Mt - Mt.T) / np.linalg.norm(Mt)
        rows.append(_row(f"V5.s{name}", f"Symmetry ||A - A^T||/||A||, distorted {name}", asym, 0.0, 1e-13, asym, "absolute"))
    me_min = np.linalg.eigvalsh(np.asarray(fem.hex8_me(X, rho))).min()
    rows.append(_row("V5.pd", "Element mass positive definite: min eigenvalue > 0", me_min, 0.0, 0.0,
                     0.0 if me_min > 0 else 1.0, f"min eigenvalue {me_min:.3e} kg"))

    # rotation invariance: rotated cantilever gives the same tip deflection magnitude
    L, b, h, nel = 0.2, 0.01, 0.01, (20, 2, 2)
    nodes, conn = fem.box_mesh((L, b, h), nel)
    f, tip = _end_face_load(nodes, L, b, h, nel, -100.0, 2)
    fixed = _clamp_x0(nodes)
    u0, _, _ = fem.linear_static(nodes, conn, E, 0.0, f, fixed)
    a = rng.standard_normal(3)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.linalg.matrix_power(np.eye(3) + Kx / 64, 64)  # approx exp(Kx)
    R, _ = np.linalg.qr(R)  # exactly orthogonal
    u1, _, _ = fem.linear_static(nodes @ R.T, conn, E, 0.0, (f.reshape(-1, 3) @ R.T).ravel(), fixed)
    d0 = np.linalg.norm(u0.reshape(-1, 3)[tip], axis=1).mean()
    d1 = np.linalg.norm(u1.reshape(-1, 3)[tip], axis=1).mean()
    rows.append(_row("V5.r", "Rotation invariance: tip |u| after a random rotation [m]", d1, d0, 1e-9,
                     note="whole mesh and load rotated; magnitudes must not change. Tolerance 1e-9, not 1e-10: "
                          "the rotated solve loses digits to cond(K) ~ 1e6 (first set at 1e-10, observed 1.2e-10)"))

    # invalid input is rejected
    bad = nodes[conn][:1].copy()
    bad[0, [0, 1]] = bad[0, [1, 0]]  # swap two nodes -> inverted element
    try:
        fem.check_mesh(bad[0], np.arange(8)[None])
        inverted = 0.0
    except ValueError:
        inverted = 1.0
    rows.append(_row("V5.inv", "Inverted element is rejected", inverted, 1.0, 0.0, note="check_mesh raises ValueError"))
    return rows


def bracket_checks(cfg, h=0.02):
    """Equilibrium of the bracket model and an independent check of its K_G assembly."""
    from .bracket import Bracket
    m = Bracket(cfg, h)
    x = np.ones(m.n)
    r = m.analyze(x, 3.0, 1.0, grad=False)
    rows = []
    K = fem.assemble(m.conn, fem.ke_batch(m.nodes[m.conn], np.full(m.ne, m.E0), m.nu, True), m.ndof)
    for k, lc in enumerate(["LC1", "LC2", "LC3"]):
        u = r["U"][:, k]
        reac = (K @ u - m.F[:, k]).reshape(-1, 3)  # nonzero only at supports
        Fn, Rn = m.F[:, k].reshape(-1, 3), reac
        force = np.linalg.norm(Fn.sum(0) + Rn.sum(0)) / m.load_magnitude
        mom = np.linalg.norm(np.cross(m.nodes, Fn).sum(0) + np.cross(m.nodes, Rn).sum(0)) / (m.load_magnitude * m.L.max())
        rows.append(_row(f"V5.eq{k + 1}", f"Bracket {lc}: force + reaction balance", force, 0.0, 1e-9, force,
                         "||sum F + sum R|| / |W|; reactions from the global K u at the supports"))
        rows.append(_row(f"V5.em{k + 1}", f"Bracket {lc}: moment balance about the origin", mom, 0.0, 1e-9, mom,
                         "normalized by W x domain length"))
    # applied torque is exactly T about the patch centre
    c = np.array([m.L[0] / 2, m.L[1] / 2, m.L[2]])
    Ty = np.cross(m.nodes - c, (m.F[:, 1] - m.F[:, 0]).reshape(-1, 3)).sum(0)[1]
    rows.append(_row("V5.T", "Applied torque about y [N m] (LC2 - LC1)", Ty, m.torque, 1e-12))
    # bracket K_G (reference element map Q) vs generic per-element K_G assembly
    from . import fem as F
    KGg = F.assemble(m.conn, F.kg_batch(m.nodes[m.conn], r["U"][m.edofs, 1], np.full(m.ne, m.E0), m.nu, True), m.ndof)
    KGg = KGg[m.free][:, m.free]
    KGm = m.pat.matrix(m.pat.P @ (np.einsum("ijk,ek->eij", m.Q, r["U"][m.edofs, 1])).ravel())
    err = abs(KGm - KGg).max() / abs(KGg).max()
    rows.append(_row("V5.kg", "Bracket K_G assembly vs generic element routine", err, 0.0, 1e-12, err,
                     "max entry difference / max entry; this check caught a double factor E0 in development"))
    mass = m.rho0 * m.Ve * m.ne
    rows.append(_row("V5.mb", "Bracket solid-domain mass [kg]", r["mass"], 2700 * 0.2 * 0.12 * 0.16, 1e-12))
    return rows


def bracket_convergence(cfg):
    """Solid design domain at several voxel sizes: interface displacement, f1, alpha, max stress."""
    from .bracket import Bracket
    out = []
    for h in cfg["mesh"]["h_convergence"]:
        m = Bracket(cfg, h)
        r = m.analyze(np.ones(m.n), 3.0, 1.0, grad=False)
        out.append(dict(h=h, ne=m.ne, ndof=m.ndof, disp_LC1=float(np.linalg.norm(r["disp"][0])),
                        f1=float(r["freq"][0]), alpha_LC2=float(1 / r["mu"][1][0]),
                        max_vm=float(r["vm_max"].max()), pnorm_LC2=float(r["pnorm"][1]),
                        seconds=float(sum(r["timings"].values()))))
    return out


def run_all(cfg):
    """All V1-V5 rows plus convergence data for V3/V4."""
    v3, c3 = modal_test(cfg)
    v4, c4 = buckling_test(cfg)
    rows = (patch_test(cfg) + patch_test(cfg, incompatible=False) + cantilever_test(cfg)
            + cantilever_test(cfg, incompatible=False) + v3 + v4 + global_checks(cfg) + bracket_checks(cfg))
    return rows, dict(modal=c3, buckling=c4)


def write_csv(rows, path=ROOT / "results" / "verification.csv"):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    rows, _ = run_all(load())
    for r in rows:
        print(f"{r['id']:8s} {'PASS' if r['passed'] else 'FAIL'} {r['name'][:62]:62s} value={r['value']:.6e} exact={r['exact']:.6e} err={r['error']:.2e}")
