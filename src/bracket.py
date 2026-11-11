"""Engine-mount bracket: voxel SIMP model, responses and adjoint sensitivities.

Design pipeline for x (one variable per non-passive voxel, x_min <= x <= 1):
  y       = x on design voxels, 1 on passive voxels
  rho_t   = H y                                  linear density filter (Bruns & Tortorelli 2001)
  rho     = proj(rho_t), rho = 1 on passive      tanh projection (Wang, Lazarov & Sigmund 2011)
  E_K     = E0 (eps + (1-eps) rho^p)             SIMP stiffness (Bendsoe 1989)
  E_G     = E0 rho^p                             stress used in K_G (no floor: voids carry no prestress)
  mass    = rho0 V sum(rho)                      objective, linear in rho
  m_modal = rho^p (1+c)/(rho^(p-1)+c), c = knee^(p-1)
                                                 modal mass: 0 and 1 at the ends, ~rho^p near 0, so
                                                 soft voids cannot create spurious low modes
  sigma   = rho^q D0 B u                         relaxed stress (Le et al. 2010), Gauss points

Every voxel is the same cube, so each element matrix is a scalar multiple (or, for K_G, a
linear map of u_e) of one reference element from src/fem.py.

Sensitivities (course method ii): element-level partials come from JAX autodiff of the
element functions (_stress_elem, _kg_elem); global adjoints are hand-assembled and solved
with the same sparse factorization of K used for the state.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import jax
import jax.numpy as jnp

from . import fem

CONSTRAINTS = ["stress_LC1", "stress_LC2", "stress_LC3", "disp_LC1", "disp_LC2", "disp_LC3",
               "freq", "buckling_LC2", "buckling_LC3"]
N_STATIC = 6  # first six constraints are static functionals of one load case each
BUCKLING_LC = (1, 2)  # LC2 and LC3 (0-based); LC1 = mean of both, so alpha_LC1 >= min(alpha_LC2, alpha_LC3)


def _stress_elem(ue, rho, SG, q, P):
    """sum over Gauss points of (rho^q sigma_vm / sigma_allow)^P; SG is pre-divided by sigma_allow."""
    s = SG @ ue
    vm2 = (0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
           + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    return rho ** (q * P) * jnp.sum(vm2 ** (P / 2))  # no sqrt: smooth at zero stress


def _kg_elem(ue, psie, Q):
    """psi_e^T K_G,e(u_e) psi_e for unit E_G (linear in u_e)."""
    return psie @ (Q @ ue) @ psie


_stress_vg = jax.jit(jax.vmap(jax.value_and_grad(_stress_elem, argnums=(0, 1)),
                              in_axes=(0, 0, None, None, None)))
_kg_vg = jax.jit(jax.vmap(jax.value_and_grad(_kg_elem), in_axes=(0, 0, None)))


def project(rt, beta, eta):
    a, b = np.tanh(beta * eta), np.tanh(beta * (1 - eta))
    t = np.tanh(beta * (rt - eta))
    return (a + t) / (a + b), beta * (1 - t ** 2) / (a + b)


def ks(v, s):
    """Kreisselmeier-Steinhauser max and its weights (sum to 1). KS >= max(v)."""
    vmax = np.max(v)
    e = np.exp(s * (v - vmax))
    return vmax + np.log(e.sum()) / s, e / e.sum()


class Pattern:
    """Maps element matrices (ne,24,24) straight into the free-DOF CSR data array."""

    def __init__(self, edofs, free, ndof):
        fmap = np.full(ndof, -1)
        fmap[free] = np.arange(len(free))
        n = edofs.shape[1]
        r = np.repeat(fmap[edofs], n, axis=1).ravel()
        c = np.tile(fmap[edofs], (1, n)).ravel()
        keep = (r >= 0) & (c >= 0)
        nf = len(free)
        uniq, inv = np.unique(r[keep].astype(np.int64) * nf + c[keep], return_inverse=True)
        self.P = sp.csr_matrix((np.ones(len(inv)), (inv, np.flatnonzero(keep))), shape=(len(uniq), r.size))
        self.rows, self.cols, self.nf = uniq // nf, uniq % nf, nf

    def scalar(self, Ke1, ne):
        """Precompute A with K.data = A @ s for K = sum_e s_e Ke1."""
        return (self.P @ sp.kron(sp.eye(ne), sp.csr_matrix(np.asarray(Ke1).reshape(-1, 1)))).tocsr()

    def matrix(self, data):
        return sp.csr_matrix((data, (self.rows, self.cols)), shape=(self.nf, self.nf))


class Bracket:
    def __init__(self, cfg, h):
        self.cfg = cfg
        mat, geo, simp, agg, lim, lo = (cfg[k] for k in ("material", "geometry", "simp", "aggregation", "limits", "loads"))
        self.E0, self.nu, self.rho0 = mat["E"], mat["nu"], mat["rho"]
        fem.check_material(self.E0, self.nu)
        self.sig_allow = mat["yield"] / lim["safety_factor"]
        self.d_allow, self.alpha_req = lim["disp_max"], lim["buckling_min"]
        self.lam_req = (2 * np.pi * lim["freq_min"]) ** 2
        self.eps, self.q, self.knee, self.eta = simp["E_min_ratio"], simp["stress_q"], simp["mass_knee"], simp["eta"]
        self.x_min = simp["x_min"]
        self.P, self.s_f, self.s_b = agg["stress_P"], agg["ks_freq"], agg["ks_buckling"]
        self.n_modes, self.n_buck = agg["n_modes"], agg["n_buckling_modes"]
        inc = cfg["solver"]["incompatible_modes"]

        L = np.array(geo["domain"], float)
        nel = np.rint(L / h).astype(int)
        if not np.allclose(nel * h, L, rtol=1e-9):
            raise ValueError(f"h={h} does not divide the domain {L}")
        self.h, self.L, self.nel = h, L, tuple(int(n) for n in nel)
        self.nodes, self.conn = fem.box_mesh(L, self.nel)
        self.ne, self.ndof = len(self.conn), 3 * len(self.nodes)
        self.edofs = fem.element_dofs(self.conn)
        c = self.nodes[self.conn].mean(1)
        self.centers = c
        tol = 1e-9
        pad = geo["pad_size"]
        ax, ay = geo["interface_size"]

        def in_pad(p):
            return (((p[:, 0] <= pad + tol) | (p[:, 0] >= L[0] - pad - tol))
                    & ((p[:, 1] <= pad + tol) | (p[:, 1] >= L[1] - pad - tol)))

        def in_iface(p):
            return (np.abs(p[:, 0] - L[0] / 2) <= ax / 2 + tol) & (np.abs(p[:, 1] - L[1] / 2) <= ay / 2 + tol)

        self.passive = ((in_pad(c) & (c[:, 2] <= geo["pad_thickness"] + tol))
                        | (in_iface(c) & (c[:, 2] >= L[2] - geo["interface_thickness"] - tol)))
        self.design_idx = np.flatnonzero(~self.passive)
        self.n = len(self.design_idx)

        # supports: bottom-face nodes inside the pads, all DOFs clamped (bolted pads, assumption)
        self.support_nodes = np.flatnonzero(np.isclose(self.nodes[:, 2], 0.0) & in_pad(self.nodes))
        self.fixed = fem.node_dofs(self.support_nodes)
        self.free = fem.free_dofs(self.ndof, self.fixed)

        # engine interface: top-face nodes in the patch, consistent (trapezoid) area weights
        top = np.isclose(self.nodes[:, 2], L[2]) & in_iface(self.nodes)
        self.iface_nodes = np.flatnonzero(top)
        xi = self.nodes[self.iface_nodes]
        wx = np.where(np.isclose(np.abs(xi[:, 0] - L[0] / 2), ax / 2), h / 2, h)
        wy = np.where(np.isclose(np.abs(xi[:, 1] - L[1] / 2), ay / 2), h / 2, h)
        w = wx * wy / (wx * wy).sum()
        dx = xi[:, 0] - L[0] / 2
        W = lo["engine_mass"] * lo["g"] * lo["g_factor"]
        fz_v = -W * w
        fz_t = -lo["torque"] * dx * w / np.sum(w * dx ** 2)  # moment about y through the patch centre = +T
        F = np.zeros((self.ndof, 3))
        F[3 * self.iface_nodes + 2] = np.column_stack([fz_v, fz_v + fz_t, fz_v - fz_t])
        self.F, self.Ff = F, F[self.free]
        self.load_magnitude, self.torque = W, lo["torque"]
        self.Wavg = sp.csr_matrix((np.tile(w, 3), (fem.node_dofs(self.iface_nodes).reshape(-1, 3).T.ravel(),
                                                    np.repeat(np.arange(3), len(w)))), shape=(self.ndof, 3))
        lumped = np.zeros(self.ndof)
        lumped[fem.node_dofs(self.iface_nodes)] = np.repeat(lo["engine_mass"] * w, 3)
        self.M_engine = sp.diags(lumped[self.free])

        # reference element (cube of side h)
        X0 = self.nodes[self.conn[0]]
        self.Ke1 = np.asarray(fem.hex8_ke(X0, 1.0, self.nu, inc))
        self.Me1 = np.asarray(fem.hex8_me(X0, 1.0))
        B, _, _ = fem.hex8_B(X0, self.nu, inc)
        self.SG = np.asarray(jnp.einsum("ab,gbj->gaj", fem.dmat(self.E0, self.nu), B)) / self.sig_allow
        self.Q = np.asarray(jax.jacfwd(lambda u: fem.hex8_kg(X0, u, self.E0, self.nu, inc))(jnp.zeros(24)))
        self.Ve = h ** 3
        self.mass_solid = self.rho0 * self.Ve * self.ne

        self.pat = Pattern(self.edofs, self.free, self.ndof)
        self.A_K = self.pat.scalar(self.Ke1, self.ne)
        self.A_M = self.pat.scalar(self.Me1, self.ne)

        # density filter, radius in metres
        r = simp["filter_radius"]
        D = cKDTree(c).sparse_distance_matrix(cKDTree(c), r, output_type="coo_matrix")
        Hm = sp.coo_matrix((r - D.data, (D.row, D.col)), shape=(self.ne,) * 2) + r * sp.eye(self.ne)
        self.H = sp.diags(1.0 / np.asarray(Hm.sum(1)).ravel()) @ Hm.tocsr()

    # ------------------------------------------------------------------ helpers

    def densities(self, x, beta):
        y = np.ones(self.ne)
        y[self.design_idx] = x
        rt = self.H @ y
        rho, drho = project(rt, beta, self.eta)
        rho[self.passive], drho[self.passive] = 1.0, 0.0
        return rho, drho

    def chain(self, drho, G):
        """d/d rho (ne, k) -> d/dx (k, n) through projection and filter."""
        return (self.H.T @ (drho[:, None] * G))[self.design_idx].T

    def full(self, vf):
        v = np.zeros((self.ndof,) + vf.shape[1:])
        v[self.free] = vf
        return v

    def scatter(self, ge):
        """Assemble element vectors (ne,24) into a free-DOF vector."""
        v = np.zeros(self.ndof)
        np.add.at(v, self.edofs, ge)
        return v[self.free]

    def k_contract(self, a_e, b_e):
        """a_e^T Ke1 b_e per element."""
        return np.einsum("ei,ij,ej->e", a_e, self.Ke1, b_e)

    # ------------------------------------------------------------------ analysis

    def analyze(self, x, p, beta, grad=True, buckling=True, buckling_grad=True, rho=None):
        """Mass objective and constraints g <= 0 (order: CONSTRAINTS) with adjoint gradients.

        Returns a dict: f, g (9,), df (n,), dg (9, n) when grad, plus physical results.
        buckling=False skips the two buckling eigenproblems (their g entries are NaN).
        rho, if given, is a physical density field (ne,) analysed as is (x is ignored and
        no gradients are returned); used for mesh-refinement and thresholded-design checks.
        """
        T = {}
        t = time.perf_counter()
        if rho is not None:
            rho, drho, grad = np.asarray(rho, float), np.zeros(self.ne), False
            if rho.shape != (self.ne,) or not np.all((rho > 0) & (rho <= 1)):
                raise ValueError("rho must have one value in (0, 1] per element")
        else:
            x = np.asarray(x, float)
            if x.shape != (self.n,) or not np.all(np.isfinite(x)):
                raise ValueError("design vector has wrong shape or non-finite entries")
            rho, drho = self.densities(x, beta)
        eps = self.eps
        EK = self.E0 * (eps + (1 - eps) * rho ** p)
        dEK = self.E0 * (1 - eps) * p * rho ** (p - 1)
        EG = rho ** p  # self.Q already carries E0
        dEG = p * rho ** (p - 1)
        c = self.knee ** (p - 1)
        mm = rho ** p * (1 + c) / (rho ** (p - 1) + c)
        dmm = (1 + c) * (p * rho ** (p - 1) * (rho ** (p - 1) + c) - rho ** p * (p - 1) * rho ** (p - 2)) / (rho ** (p - 1) + c) ** 2
        K = self.pat.matrix(self.A_K @ EK)
        S = fem.SPDSolver(K)
        U = self.full(S(self.Ff))  # (ndof, 3)
        Ue = U[self.edofs]         # (ne, 24, 3)
        T["static"] = time.perf_counter() - t

        out = dict(rho=rho, U=U, p=p, beta=beta)
        g = np.full(len(CONSTRAINTS), np.nan)
        G = np.zeros((self.ne, len(CONSTRAINTS)))  # d g / d rho
        f = rho.sum() / self.ne
        G0 = np.full(self.ne, 1.0 / self.ne)

        # static functionals: stress p-norm and interface displacement, per load case
        t = time.perf_counter()
        R = np.zeros((len(self.free), N_STATIC))  # adjoint right-hand sides  -(d g / d u)
        vm_max = np.zeros((self.ne, 3))
        pn, disp = np.zeros(3), np.zeros((3, 3))
        for k in range(3):
            val, (gu, gr) = _stress_vg(Ue[:, :, k], rho, self.SG, self.q, self.P)
            Ssum = float(np.sum(val))
            pn[k] = Ssum ** (1 / self.P)
            dpn = pn[k] / (self.P * Ssum)
            g[k] = pn[k] - 1.0
            G[:, k] = dpn * np.asarray(gr)
            R[:, k] = -dpn * self.scatter(np.asarray(gu))
            s = np.einsum("gaj,ej->ega", self.SG, Ue[:, :, k]) * self.sig_allow
            vm_max[:, k] = rho ** self.q * np.max(np.asarray(fem.von_mises(s)), axis=1)
            ub = self.Wavg.T @ U[:, k]
            d = np.linalg.norm(ub)
            disp[k] = ub
            g[3 + k] = d / self.d_allow - 1.0
            R[:, 3 + k] = -(self.Wavg @ (ub / (d * self.d_allow)))[self.free]
        if grad:
            Lam = self.full(S(R))[self.edofs]  # (ne, 24, 6)
            for i in range(N_STATIC):
                G[:, i] += dEK * self.k_contract(Lam[:, :, i], Ue[:, :, i % 3])
        out.update(pnorm=pn * self.sig_allow, vm_max=vm_max, disp=disp)
        T["stress_disp"] = time.perf_counter() - t

        # natural frequencies (engine as lumped mass on the interface)
        t = time.perf_counter()
        M = self.pat.matrix(self.A_M @ (self.rho0 * mm)) + self.M_engine
        lam, phi = fem.modal(K, M, self.n_modes, S)
        # g = 1 - KSmin(lam_i/lam_req), KSmin = -KS(-v) <= min: nearly linear in stiffness and
        # bounded above by 1, unlike lam_req/lam - 1, which explodes as the design thins.
        gf, wf = ks(-lam / self.lam_req, self.s_f)
        g[6] = 1.0 + gf
        if grad:
            Ph = self.full(phi)[self.edofs]
            for i in range(self.n_modes):
                dlam = dEK * self.k_contract(Ph[:, :, i], Ph[:, :, i]) \
                    - lam[i] * self.rho0 * dmm * np.einsum("ei,ij,ej->e", Ph[:, :, i], self.Me1, Ph[:, :, i])
                G[:, 6] -= wf[i] / self.lam_req * dlam
        out.update(lam=lam, freq=np.sqrt(np.maximum(lam, 0)) / (2 * np.pi), phi=phi)
        T["modal"] = time.perf_counter() - t

        # linear buckling with prestress from LC2 and LC3
        t = time.perf_counter()
        out["mu"], out["psi"] = {}, {}
        if buckling:
            for j, k in enumerate(BUCKLING_LC):
                KGe = EG[:, None, None] * np.einsum("ijk,ek->eij", self.Q, Ue[:, :, k])
                KG = self.pat.matrix(self.pat.P @ KGe.ravel())
                mu, psi = fem.buckling(K, KG, self.n_buck, S)
                gb, wb = ks(self.alpha_req * mu, self.s_b)
                g[7 + j] = gb - 1.0
                out["mu"][k], out["psi"][k] = mu, psi
                if grad and buckling_grad:
                    Ps = self.full(psi)[self.edofs]
                    cvec = np.zeros(len(self.free))
                    for i in range(self.n_buck):
                        hval, hgrad = _kg_vg(Ue[:, :, k], Ps[:, :, i], self.Q)
                        wi = wb[i] * self.alpha_req
                        # explicit: d mu = -psi^T dK_G psi - mu psi^T dK psi
                        G[:, 7 + j] += wi * (-dEG * np.asarray(hval) - mu[i] * dEK * self.k_contract(Ps[:, :, i], Ps[:, :, i]))
                        cvec += wi * self.scatter(EG[:, None] * np.asarray(hgrad))
                    # through the state: K lambda = c, then + lambda^T dK u
                    lb = self.full(S(cvec))[self.edofs]
                    G[:, 7 + j] += dEK * self.k_contract(lb, Ue[:, :, k])
        T["buckling"] = time.perf_counter() - t

        out.update(f=f, g=g, mass=f * self.mass_solid, timings=T)
        if grad:
            out["df"] = self.chain(drho, G0[:, None])[0]
            out["dg"] = self.chain(drho, G)
        return out
