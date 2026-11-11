"""Hex8 linear elasticity: element routines in JAX, global sparse algebra in SciPy.

Element routines are pure JAX so Stage-4 sensitivities can differentiate them at the
element level (course method ii). Global matrices are always SciPy sparse.
DOF ordering is node-major: (ux, uy, uz) per node. Voigt order: xx, yy, zz, yz, xz, xy
with engineering shear strains. Stress is tension-positive.

Sign convention for buckling: (K + alpha K_G) psi = 0, with K_G built from the stress of
the reference load. We solve the equivalent symmetric-definite problem
(-K_G) psi = mu K psi, mu = 1/alpha, and report the largest mu (smallest positive alpha).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Reference nodes in standard hex8 order (bottom face counter-clockwise, then top).
XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
GP = XI / np.sqrt(3.0)  # 2x2x2 Gauss points, all weights 1


def _dN_ref(p):
    """(8,3) derivatives of trilinear shape functions w.r.t. (xi, eta, zeta) at p."""
    t = 1.0 + XI * p
    return 0.125 * XI * np.stack([t[:, 1] * t[:, 2], t[:, 0] * t[:, 2], t[:, 0] * t[:, 1]], 1)


N_GP = np.prod(1.0 + XI[None] * GP[:, None], axis=2) / 8.0  # (gp, node) shape functions
DN = np.stack([_dN_ref(p) for p in GP])                     # (gp, node, 3)
DN0 = _dN_ref(np.zeros(3))                                  # at the centroid
# Wilson incompatible modes P_k = 1 - xi_k^2  ->  dP_k/dxi_j = -2 xi_k delta_kj
DP = np.stack([np.diag(-2.0 * p) for p in GP])              # (gp, mode, 3)
# Unit symmetric tensors for each Voigt component (used to build K_G)
VOIGT_T = np.zeros((6, 3, 3))
for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
    VOIGT_T[k, i, j] = VOIGT_T[k, j, i] = 1.0


def check_material(E, nu):
    if not (np.all(np.asarray(E) > 0) and -1.0 < nu < 0.5):
        raise ValueError(f"need E > 0 and -1 < nu < 0.5, got E={E}, nu={nu}")


def dmat(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    normal = jnp.zeros((6, 6)).at[:3, :3].set(lam)
    return normal + jnp.diag(jnp.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0]) * mu)


def bmat(dNdx):
    """(n,3) spatial shape-function derivatives -> (6, 3n) strain-displacement matrix."""
    x, y, z = dNdx.T
    o = jnp.zeros_like(x)
    B = jnp.stack([jnp.stack([x, o, o], 1), jnp.stack([o, y, o], 1), jnp.stack([o, o, z], 1),
                   jnp.stack([o, z, y], 1), jnp.stack([z, o, x], 1), jnp.stack([y, x, o], 1)])
    return B.reshape(6, -1)


def _spatial(J, dref):
    """dN/dx = J^-1 dN/dxi, with J_ij = dx_j/dxi_i."""
    return jnp.linalg.solve(J, dref.T).T


def hex8_B(X, nu, incompatible=True):
    """Effective Gauss-point strain-displacement matrices for node coordinates X (8,3).

    Returns B (gp,6,24), detJ (gp,), dNdx (gp,8,3). With incompatible=True the 9 Wilson
    bubble modes (Taylor's correction: centroid Jacobian scaled by detJ0/detJ, so the
    patch test passes) are condensed out: alpha = -Kaa^-1 Kau u, B_eff = Bu + Ba A.
    Then K_e = sum_g B_eff^T D B_eff detJ equals the statically condensed stiffness.
    A does not depend on E, only on nu.
    """
    J = jnp.einsum("gni,nj->gij", DN, X)
    detJ = jnp.linalg.det(J)
    dNdx = jax.vmap(_spatial)(J, DN)
    Bu = jax.vmap(bmat)(dNdx)
    if not incompatible:
        return Bu, detJ, dNdx
    J0 = DN0.T @ X
    Ba = jax.vmap(lambda d, s: s * bmat(_spatial(J0, d)))(DP, jnp.linalg.det(J0) / detJ)
    D = dmat(1.0, nu)
    Kua = jnp.einsum("gai,ab,gbj,g->ij", Bu, D, Ba, detJ)
    Kaa = jnp.einsum("gai,ab,gbj,g->ij", Ba, D, Ba, detJ)
    A = -jnp.linalg.solve(Kaa, Kua.T)
    return Bu + jnp.einsum("gai,ij->gaj", Ba, A), detJ, dNdx


def hex8_ke(X, E, nu, incompatible=True):
    B, detJ, _ = hex8_B(X, nu, incompatible)
    return jnp.einsum("gai,ab,gbj,g->ij", B, dmat(E, nu), B, detJ)


def hex8_me(X, rho):
    """Consistent mass matrix (24,24); the bubble modes carry no mass."""
    detJ = jnp.linalg.det(jnp.einsum("gni,nj->gij", DN, X))
    m8 = rho * jnp.einsum("ga,gb,g->ab", N_GP, N_GP, detJ)
    return jnp.kron(m8, jnp.eye(3))


def hex8_stress(X, ue, E, nu, incompatible=True):
    """Gauss-point stresses (gp, 6), recovered with the condensed bubble modes."""
    B, _, _ = hex8_B(X, nu, incompatible)
    return jnp.einsum("ab,gbj,j->ga", dmat(E, nu), B, ue)


def hex8_kg(X, ue, E, nu, incompatible=True):
    """Initial-stress (geometric) stiffness (24,24) from the Gauss-point stress of ue.

    K_G = sum_g G_g^T S_g G_g detJ_g, assembled as kron(kg8, I3) where
    kg8_ab = sum_g dN_a/dx . sigma_g . dN_b/dx detJ_g.
    """
    B, detJ, dNdx = hex8_B(X, nu, incompatible)
    sig = jnp.einsum("ab,gbj,j->ga", dmat(E, nu), B, ue)
    S = jnp.einsum("gk,kij->gij", sig, VOIGT_T)
    kg8 = jnp.einsum("gai,gij,gbj,g->ab", dNdx, S, dNdx, detJ)
    return jnp.kron(kg8, jnp.eye(3))


def von_mises(s):
    sxx, syy, szz, syz, sxz, sxy = (s[..., i] for i in range(6))
    return jnp.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                    + 3.0 * (syz ** 2 + sxz ** 2 + sxy ** 2))


def min_detJ(X):
    return jnp.min(jnp.linalg.det(jnp.einsum("gni,nj->gij", DN, X)))


ke_batch = jax.jit(jax.vmap(hex8_ke, in_axes=(0, 0, None, None)), static_argnums=3)
me_batch = jax.jit(jax.vmap(hex8_me, in_axes=(0, 0)))
stress_batch = jax.jit(jax.vmap(hex8_stress, in_axes=(0, 0, 0, None, None)), static_argnums=4)
kg_batch = jax.jit(jax.vmap(hex8_kg, in_axes=(0, 0, 0, None, None)), static_argnums=4)
min_detJ_batch = jax.jit(jax.vmap(min_detJ))


# ---------------------------------------------------------------- global level

def box_mesh(L, nel):
    """Structured hex mesh of the box [0,Lx]x[0,Ly]x[0,Lz] with nel=(nx,ny,nz) elements.
    Element e = i + nx*(j + ny*k); node n = i + (nx+1)*(j + (ny+1)*k)."""
    axes = [np.linspace(0.0, l, n + 1) for l, n in zip(L, nel)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    nx, ny, nz = nel
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    n0 = (i + (nx + 1) * (j + (ny + 1) * k)).transpose(2, 1, 0).ravel()
    sx, sy = 1, nx + 1
    sz = (nx + 1) * (ny + 1)
    conn = np.column_stack([n0, n0 + sx, n0 + sx + sy, n0 + sy,
                            n0 + sz, n0 + sz + sx, n0 + sz + sx + sy, n0 + sz + sy])
    return nodes, conn


def node_dofs(nodes_idx):
    return (3 * np.asarray(nodes_idx)[:, None] + np.arange(3)).ravel()


def element_dofs(conn):
    return (3 * conn[:, :, None] + np.arange(3)).reshape(len(conn), -1)


def check_mesh(nodes, conn):
    if np.min(min_detJ_batch(nodes[conn])) <= 0:
        raise ValueError("mesh has an inverted or degenerate element (detJ <= 0)")


def assemble(conn, Ke, ndof):
    """Sum element matrices (ne, 24, 24) into a sparse CSR global matrix."""
    edofs = element_dofs(conn)
    n = edofs.shape[1]
    rows = np.repeat(edofs, n, axis=1).ravel()
    cols = np.tile(edofs, (1, n)).ravel()
    return sp.coo_matrix((np.asarray(Ke).ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()


class SPDSolver:
    """Sparse LU of a symmetric positive-definite matrix, reused for many right-hand sides
    (load cases, adjoints, eigen-solver iterations)."""

    def __init__(self, A):
        self.lu = spla.splu(sp.csc_matrix(A), permc_spec="MMD_AT_PLUS_A",
                            diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        self.n = A.shape[0]

    def __call__(self, b):
        return self.lu.solve(np.asarray(b, float))

    def op(self):
        return spla.LinearOperator((self.n, self.n), matvec=self.__call__, dtype=float)


def free_dofs(ndof, fixed):
    return np.setdiff1d(np.arange(ndof), fixed)


def solve_dirichlet(K, f, fixed, u_fixed=0.0):
    """Solve K u = f with u[fixed] = u_fixed by eliminating the fixed DOFs."""
    u = np.zeros(K.shape[0])
    u[fixed] = u_fixed
    free = free_dofs(K.shape[0], fixed)
    rhs = f[free] - K[free][:, fixed] @ u[fixed]
    u[free] = SPDSolver(K[free][:, free])(rhs)
    return u


def modal(Kff, Mff, n, solver=None, sigma=0.0):
    """Lowest n eigenpairs of K phi = lam M phi (lam = omega^2), ascending, M-normalized.

    Shift-invert about sigma; fixed start vector so repeated calls are bit-identical
    (finite-difference checks need that). Nothing is filtered: negative or rigid-body eigenvalues are
    returned as computed so callers can see them.
    """
    kw = {}
    if solver is not None and sigma == 0.0:
        kw["OPinv"] = solver.op()
    lam, phi = spla.eigsh(Kff, k=n, M=Mff, sigma=sigma, which="LM", v0=np.ones(Kff.shape[0]), **kw)
    order = np.argsort(lam)
    lam, phi = lam[order], phi[:, order]
    phi /= np.sqrt(np.einsum("ij,ij->j", phi, Mff @ phi))
    return lam, phi


def buckling(Kff, KGff, n, solver):
    """Largest n eigenpairs of (-K_G) psi = mu K psi, mu = 1/alpha, descending, psi^T K psi = 1.

    alpha = 1/mu is the load factor of (K + alpha K_G) psi = 0. mu <= 0 means no buckling
    under this load direction; such values are returned, not hidden.
    Limitation: if the prestress has no compression at all, every mu <= 0 and they cluster at
    0, so ARPACK cannot isolate the largest and raises ArpackNoConvergence (never silent).
    """
    mu, psi = spla.eigsh(-KGff, k=n, M=Kff, Minv=solver.op(), which="LA", v0=np.ones(Kff.shape[0]))
    order = np.argsort(-mu)
    mu, psi = mu[order], psi[:, order]
    psi /= np.sqrt(np.einsum("ij,ij->j", psi, Kff @ psi))
    return mu, psi


def linear_static(nodes, conn, E, nu, f, fixed, u_fixed=0.0, incompatible=True):
    """Assemble and solve. E may be a scalar or per-element array.
    Returns (u, K, stress) with Gauss-point stresses of shape (ne, 8, 6)."""
    check_material(E, nu)
    check_mesh(nodes, conn)
    Xe = nodes[conn]
    Ee = np.broadcast_to(np.asarray(E, float), len(conn))
    K = assemble(conn, ke_batch(Xe, Ee, nu, incompatible), 3 * len(nodes))
    u = solve_dirichlet(K, f, fixed, u_fixed)
    stress = np.asarray(stress_batch(Xe, u[element_dofs(conn)], Ee, nu, incompatible))
    return u, K, stress
