"""Method of Moving Asymptotes (Svanberg 1987; asymptote rules and regularization of
Svanberg 2007, "MMA and GCMMA - two methods for nonlinear optimization").

Problem: min f0(x)  s.t.  g_i(x) <= 0,  xmin <= x <= xmax.
Each step solves the convex separable subproblem through its concave dual, maximized with
L-BFGS-B. Elastic variables y_i >= 0 with penalty c*y + y^2/2 keep the subproblem feasible;
y > 0 at the solution means the linearized constraint could not be met.

Why MMA rather than SLSQP: SLSQP keeps a dense n x n quasi-Newton matrix and solves a dense
least-squares QP every iteration; with n ~ 10^3 to 10^4 voxel densities that dominates the run.
MMA needs O(n m) work per step and returns dual multipliers for the KKT report.
"""
import numpy as np
from scipy.optimize import minimize


class MMA:
    def __init__(self, xmin, xmax, move=0.2, c=1000.0, asyinit=0.5, asyincr=1.2, asydecr=0.7):
        self.xmin, self.xmax = np.asarray(xmin, float), np.asarray(xmax, float)
        self.move, self.c = move, c
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.reset()

    def reset(self):
        """Forget asymptote history (call when the problem changes, e.g. a continuation step)."""
        self.k, self.xold1, self.xold2 = 0, None, None

    def step(self, x, df0, g, dg):
        """One MMA iteration. Returns (x_new, lam, y): new design, subproblem multipliers
        (one per constraint) and elastic variables."""
        x, g, dg = np.asarray(x, float), np.atleast_1d(g), np.atleast_2d(dg)
        xr = self.xmax - self.xmin
        if self.k < 2:
            self.low = x - self.asyinit * xr
            self.upp = x + self.asyinit * xr
        else:
            s = (x - self.xold1) * (self.xold1 - self.xold2)
            gam = np.where(s > 0, self.asyincr, np.where(s < 0, self.asydecr, 1.0))
            self.low = np.clip(x - gam * (self.xold1 - self.low), x - 10 * xr, x - 0.01 * xr)
            self.upp = np.clip(x + gam * (self.upp - self.xold1), x + 0.01 * xr, x + 10 * xr)
        low, upp = self.low, self.upp
        alpha = np.maximum.reduce([self.xmin, low + 0.1 * (x - low), x - self.move * xr])
        beta = np.minimum.reduce([self.xmax, upp - 0.1 * (upp - x), x + self.move * xr])

        ux2, xl2 = (upp - x) ** 2, (x - low) ** 2
        reg = 1e-5 / np.maximum(xr, 1e-5)
        p0 = (1.001 * np.maximum(df0, 0) + 0.001 * np.maximum(-df0, 0) + reg) * ux2
        q0 = (0.001 * np.maximum(df0, 0) + 1.001 * np.maximum(-df0, 0) + reg) * xl2
        P = (1.001 * np.maximum(dg, 0) + 0.001 * np.maximum(-dg, 0)) * ux2
        Q = (0.001 * np.maximum(dg, 0) + 1.001 * np.maximum(-dg, 0)) * xl2
        b = P @ (1 / (upp - x)) + Q @ (1 / (x - low)) - g  # subproblem: P/(U-x) + Q/(x-L) <= b

        def primal(lam):
            pj, qj = p0 + lam @ P, q0 + lam @ Q
            sp, sq = np.sqrt(pj), np.sqrt(qj)
            xs = np.clip((low * sp + upp * sq) / (sp + sq), alpha, beta)
            y = np.maximum(0.0, lam - self.c)
            return xs, y, pj, qj

        def neg_dual(lam):
            xs, y, pj, qj = primal(lam)
            ui, li = 1 / (upp - xs), 1 / (xs - low)
            W = pj @ ui + qj @ li + np.sum(self.c * y + 0.5 * y ** 2 - lam * y) - lam @ b
            grad = P @ ui + Q @ li - y - b
            return -W, -grad

        m = len(g)
        res = minimize(neg_dual, np.ones(m), jac=True, method="L-BFGS-B",
                       bounds=[(0, None)] * m, options=dict(maxiter=2000, ftol=1e-15, gtol=1e-12))
        lam = res.x
        xnew, y, _, _ = primal(lam)
        self.xold2, self.xold1 = self.xold1, x.copy()
        self.k += 1
        return xnew, lam, y
