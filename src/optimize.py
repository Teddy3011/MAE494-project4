"""Continuation + MMA driver, per-iteration history and an independent KKT check."""
import time
import numpy as np
from scipy.optimize import nnls

from .bracket import Bracket, CONSTRAINTS
from .mma import MMA


def active_set(use_buckling, use_freq=True):
    return list(range(6)) + ([6] if use_freq else []) + ([7, 8] if use_buckling else [])


def kkt(r, x, xmin, xmax, active, tol_bound=1e-6, tol_active=1e-2):
    """KKT residuals at x estimated independently of the optimizer.

    Multipliers of the near-active constraints (g > -tol_active; the others get zero by
    complementarity) come from non-negative least squares on stationarity over the
    variables strictly inside their bounds; bound multipliers are the remaining gradient
    components, and must have the right sign on the bounds where they act.
    """
    active = [i for i in active if r["g"][i] > -tol_active]
    df, gA, dgA = r["df"], r["g"][active], r["dg"][active]
    lo, hi = x <= xmin + tol_bound, x >= xmax - tol_bound
    inner = ~(lo | hi)
    lam, _ = nnls(-dgA[:, inner].T, df[inner])
    s = df + dgA.T @ lam  # = bound multipliers: +z_lo on lower bounds, -z_hi on upper bounds
    scale = np.linalg.norm(df)
    return dict(
        multipliers=dict(zip([CONSTRAINTS[i] for i in active], lam.tolist())),
        stationarity=float(np.linalg.norm(s[inner]) / scale),
        bound_sign_violation=float((np.linalg.norm(np.minimum(s[lo], 0)) + np.linalg.norm(np.maximum(s[hi], 0))) / scale),
        primal_violation=float(max(0.0, gA.max())),
        complementarity=float(np.max(np.abs(lam * gA))),
        n_lower=int(lo.sum()), n_upper=int(hi.sum()), n_inner=int(inner.sum()),
    )


def run(cfg, h, x0, use_buckling=True, use_freq=True, label="", log=print):
    """Optimize from the uniform design x = x0. Returns a dict with the final design,
    a re-evaluation on a fresh model instance, KKT residuals and the full history.

    The reported design is the last iterate if it is feasible (max g <= tol_feas);
    otherwise the lowest-mass feasible iterate of the final stage (flagged in the result).
    """
    t_start = time.perf_counter()
    model = Bracket(cfg, h)
    opt, simp = cfg["optimizer"], cfg["simp"]
    active = active_set(use_buckling, use_freq)
    stages = simp["continuation"]
    xmin = simp["x_min"]
    x = np.full(model.n, float(x0))
    mma = MMA(np.full(model.n, xmin), np.ones(model.n), move=opt["move"])
    hist, it, lam, best = [], 0, None, None
    for stage, (p, beta, maxit) in enumerate(stages):
        mma.reset()
        last = stage == len(stages) - 1
        mma.move = opt["move_final"] if last else opt["move"]
        for _ in range(int(maxit)):
            t = time.perf_counter()
            r = model.analyze(x, p, beta, buckling=True, buckling_grad=use_buckling)
            if not (np.isfinite(r["f"]) and np.all(np.isfinite(r["g"])) and np.all(np.isfinite(r["dg"][active]))):
                raise FloatingPointError(f"{label}: non-finite response at iteration {it}")
            xnew, lam, y = mma.step(x, r["df"], r["g"][active], r["dg"][active])
            change = float(np.max(np.abs(xnew - x)))
            gray = float(np.mean(4 * r["rho"] * (1 - r["rho"])))
            row = dict(iter=it, stage=stage, p=p, beta=beta, mass=r["mass"], f=r["f"],
                       **{c: float(v) for c, v in zip(CONSTRAINTS, r["g"])},
                       f1=float(r["freq"][0]), alpha_LC2=float(1 / r["mu"][1][0]), alpha_LC3=float(1 / r["mu"][2][0]),
                       max_vm=float(r["vm_max"].max()), change=change, max_elastic=float(y.max()),
                       grayness=gray, seconds=time.perf_counter() - t)
            hist.append(row)
            if it % 10 == 0:
                log(f"[{label}] it {it:3d} st {stage} p={p} b={beta} mass={r['mass']:.3f} kg "
                    f"gmax={np.max(r['g'][active]):+.4f} f1={row['f1']:.1f} Hz a={min(row['alpha_LC2'], row['alpha_LC3']):.2f} "
                    f"dx={change:.3f} {row['seconds']:.1f}s")
            feasible = np.max(r["g"][active]) <= opt["tol_feas"]
            if last and feasible and (best is None or r["f"] < best[0]):
                best = (r["f"], x.copy(), it, lam.copy())
            it += 1
            if change < opt["tol_change"] and feasible:
                break
            x = xnew
    p, beta = stages[-1][:2]
    t_opt = time.perf_counter() - t_start

    # independent re-evaluation on a fresh model instance
    fresh = Bracket(cfg, h)
    fin = fresh.analyze(x, p, beta)
    chosen = "last iterate"
    if np.max(fin["g"][active]) > opt["tol_feas"] and best is not None:
        _, x, best_it, lam = best
        fin = fresh.analyze(x, p, beta)
        chosen = f"best feasible iterate of the final stage (iteration {best_it})"
    log(f"[{label}] done: {it} iterations, {t_opt:.0f} s, mass {fin['mass']:.4f} kg, max g {np.max(fin['g'][active]):+.2e} ({chosen})")
    return dict(label=label, h=h, x0=x0, use_buckling=use_buckling, use_freq=use_freq, active=active, x=x,
                final=fin, history=hist, chosen=chosen,
                kkt=kkt(fin, x, xmin, 1.0, active), mma_multipliers=dict(zip([CONSTRAINTS[i] for i in active], lam.tolist())),
                seconds=t_opt, iterations=it, model=fresh)
