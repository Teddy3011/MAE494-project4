"""V6 gradient verification.

1. Directional central differences of the adjoint gradient for the objective and all nine
   constraints, along a random unit direction (step sweep / Taylor test) and along single
   variables next to a support pad, next to the engine interface and in the middle.
2. Course method (i) cross-check: the complete pipeline (filter, projection, dense assembly,
   dense solve, eigen-decompositions) written in JAX on a 40 mm mesh and differentiated by
   reverse-mode AD, compared with the hand-assembled adjoint (method ii) for every variable.
3. Cost: adjoint gradient vs finite differences at two mesh sizes.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

from .bracket import Bracket, CONSTRAINTS, BUCKLING_LC, _stress_vg, project
from . import fem

FUNCS = ["mass"] + CONSTRAINTS
P_CHECK, BETA_CHECK = 3.0, 2.0  # an interior continuation state: penalized, partly projected


def _values(m, x):
    r = m.analyze(x, P_CHECK, BETA_CHECK, grad=False)
    return np.r_[r["f"], r["g"]]


def _nearest_design(m, point):
    d = np.linalg.norm(m.centers[m.design_idx] - point, axis=1)
    return int(np.argmin(d))


def directional(cfg, h=0.02, steps=np.logspace(-1, -9, 17), tol=1e-5):
    m = Bracket(cfg, h)
    rng = np.random.default_rng(cfg["seed"])
    x = rng.uniform(0.25, 0.85, m.n)
    r = m.analyze(x, P_CHECK, BETA_CHECK)
    grad = np.vstack([r["df"], r["dg"]])
    f0 = np.r_[r["f"], r["g"]]
    d = rng.standard_normal(m.n)
    dirs = [("random unit", d / np.linalg.norm(d), steps)]
    pad = np.array([m.cfg["geometry"]["pad_size"] / 2, m.cfg["geometry"]["pad_size"] / 2, m.cfg["geometry"]["pad_thickness"] + h / 2])
    iface = np.array([m.L[0] / 2, m.L[1] / 2, m.L[2] - m.cfg["geometry"]["interface_thickness"] - h / 2])
    for name, pt in [("voxel above support pad", pad), ("voxel below engine interface", iface), ("voxel at domain centre", m.L / 2)]:
        e = np.zeros(m.n)
        e[_nearest_design(m, pt)] = 1.0
        dirs.append((name, e, np.array([1e-3, 1e-4, 1e-5])))
    sweep, rows = [], []
    for name, d, eps_list in dirs:
        ad = grad @ d
        floor = 1e-8 * np.linalg.norm(grad, axis=1)  # derivative scale below which relative error is meaningless
        best = np.full(len(FUNCS), np.inf)
        fd_best = np.zeros(len(FUNCS))
        for eps in eps_list:
            fp, fm = _values(m, x + eps * d), _values(m, x - eps * d)
            fd = (fp - fm) / (2 * eps)
            err = np.abs(fd - ad) / np.maximum.reduce([np.abs(ad), np.abs(fd), floor])
            taylor = np.abs(fp - f0 - eps * ad)
            for i, fn in enumerate(FUNCS):
                sweep.append(dict(direction=name, function=fn, eps=eps, adjoint=ad[i], fd=fd[i], rel_error=err[i], taylor_remainder=taylor[i]))
            better = err < best
            best[better], fd_best[better] = err[better], fd[better]
        for i, fn in enumerate(FUNCS):
            rows.append(dict(direction=name, function=fn, adjoint=float(ad[i]), central_difference=float(fd_best[i]),
                             rel_error=float(best[i]), tol=tol, passed=bool(best[i] <= tol)))
    return rows, sweep


# ------------------------------------------------------------------ method (i): dense JAX

def dense_pipeline(m, p, beta):
    """All ten outputs (mass, 9 constraints) of the bracket as one differentiable JAX function."""
    H = jnp.asarray(m.H.toarray())
    free = jnp.asarray(m.free)
    edofs = jnp.asarray(m.edofs)
    rows = jnp.repeat(edofs, 24, axis=1).ravel()
    cols = jnp.tile(edofs, (1, 24)).ravel()
    Ke1, Me1, Q, SG = map(jnp.asarray, (m.Ke1, m.Me1, m.Q, m.SG))
    Ff = jnp.asarray(m.Ff)
    Wavg = jnp.asarray(m.Wavg.toarray())
    Meng = jnp.asarray(m.M_engine.toarray())
    passive = jnp.asarray(m.passive)
    eps, c = m.eps, m.knee ** (p - 1)
    nd = m.ndof

    def assemble(Ke):
        return jnp.zeros((nd, nd)).at[rows, cols].add(Ke.ravel())[free][:, free]

    def ksmax(v, s):
        return jnp.max(v) + jnp.log(jnp.sum(jnp.exp(s * (v - jnp.max(v))))) / s

    def outputs(x):
        y = jnp.ones(m.ne).at[m.design_idx].set(x)
        a, b = np.tanh(beta * m.eta), np.tanh(beta * (1 - m.eta))
        rho = jnp.where(passive, 1.0, (a + jnp.tanh(beta * (H @ y - m.eta))) / (a + b))
        EK = m.E0 * (eps + (1 - eps) * rho ** p)
        K = assemble(EK[:, None, None] * Ke1)
        Uf = jnp.linalg.solve(K, Ff)
        U = jnp.zeros((nd, 3)).at[free].set(Uf)
        Ue = U[edofs]
        g = []
        for k in range(3):
            val, _ = _stress_vg(Ue[:, :, k], rho, SG, m.q, m.P)
            g.append(jnp.sum(val) ** (1 / m.P) - 1)
        for k in range(3):
            g.append(jnp.linalg.norm(Wavg.T @ U[:, k]) / m.d_allow - 1)
        mm = rho ** p * (1 + c) / (rho ** (p - 1) + c)
        M = assemble((m.rho0 * mm)[:, None, None] * Me1) + Meng  # M_engine is already on free DOFs
        L = jnp.linalg.cholesky(M)
        A = jax.scipy.linalg.solve_triangular(L, jax.scipy.linalg.solve_triangular(L, K, lower=True).T, lower=True)
        lam = jnp.linalg.eigh(0.5 * (A + A.T))[0][:m.n_modes]
        g.append(1 + ksmax(-lam / m.lam_req, m.s_f))
        C = jnp.linalg.cholesky(K)
        for k in BUCKLING_LC:
            KG = assemble((rho ** p)[:, None, None] * jnp.einsum("ijk,ek->eij", Q, Ue[:, :, k]))
            B = jax.scipy.linalg.solve_triangular(C, jax.scipy.linalg.solve_triangular(C, -KG, lower=True).T, lower=True)
            mu = jnp.linalg.eigh(0.5 * (B + B.T))[0][::-1][:m.n_buck]
            g.append(ksmax(m.alpha_req * mu, m.s_b) - 1)
        return jnp.concatenate([jnp.atleast_1d(jnp.sum(rho) / m.ne), jnp.stack(g)])

    return outputs


def method_i_check(cfg, h=0.04, tol=1e-6):
    m = Bracket(cfg, h)
    rng = np.random.default_rng(cfg["seed"] + 1)
    x = rng.uniform(0.25, 0.85, m.n)
    r = m.analyze(x, P_CHECK, BETA_CHECK)
    f = dense_pipeline(m, P_CHECK, BETA_CHECK)
    vals = np.asarray(f(jnp.asarray(x)))
    J = np.asarray(jax.jacrev(f)(jnp.asarray(x)))
    adj = np.vstack([r["df"], r["dg"]])
    out = []
    for i, fn in enumerate(FUNCS):
        err = np.linalg.norm(adj[i] - J[i]) / np.linalg.norm(J[i])
        cos = adj[i] @ J[i] / (np.linalg.norm(adj[i]) * np.linalg.norm(J[i]))
        out.append(dict(function=fn, value_sparse=float(np.r_[r["f"], r["g"]][i]), value_dense=float(vals[i]),
                        rel_error=float(err), cosine=float(cos), tol=tol, passed=bool(err <= tol), n=m.n))
    return out


# ------------------------------------------------------------------ cost

def cost(cfg, hs=(0.02, 0.01)):
    out = []
    rng = np.random.default_rng(cfg["seed"])
    for h in hs:
        m = Bracket(cfg, h)
        x = rng.uniform(0.25, 0.85, m.n)
        m.analyze(x, P_CHECK, BETA_CHECK)  # warm-up (JIT compile)
        t = time.perf_counter()
        m.analyze(x, P_CHECK, BETA_CHECK, grad=False)
        t_val = time.perf_counter() - t
        t = time.perf_counter()
        m.analyze(x, P_CHECK, BETA_CHECK)
        t_grad = time.perf_counter() - t
        out.append(dict(h=h, n=m.n, ndof=m.ndof, t_analysis=t_val, t_analysis_plus_adjoint=t_grad,
                        adjoint_overhead=t_grad - t_val, t_fd_central_estimate=2 * m.n * t_val,
                        speedup=2 * m.n * t_val / t_grad))
    return out
