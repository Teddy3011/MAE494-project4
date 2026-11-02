"""Hex8 linear elasticity.

Element routines are written in JAX so Stage 4 can differentiate them (method ii:
element-level AD, hand-assembled adjoint). Global assembly and solves use SciPy sparse.
DOF ordering is node-major: (ux, uy, uz) per node. Voigt order: xx, yy, zz, yz, xz, xy.
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


DN = np.stack([_dN_ref(p) for p in GP])  # (gp, node, 3)
DN0 = _dN_ref(np.zeros(3))               # at the centroid
# Incompatible modes P_k = 1 - xi_k^2  ->  dP_k/dxi_j = -2 xi_k delta_kj
DP = np.stack([np.diag(-2.0 * p) for p in GP])  # (gp, mode, 3)


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


def hex8_ke(X, E, nu, incompatible=True):
    """24x24 element stiffness for node coordinates X (8,3).

    incompatible=True adds Wilson's 9 bubble modes with Taylor's correction
    (derivatives taken with the centroid Jacobian, scaled by detJ0/detJ) so the
    patch test still passes, then condenses them out statically.
    """
    D = dmat(E, nu)
    J = jnp.einsum("gni,nj->gij", DN, X)
    detJ = jnp.linalg.det(J)
    Bu = jax.vmap(lambda j, d: bmat(_spatial(j, d)))(J, DN)  # (gp, 6, 24)
    Kuu = jnp.einsum("gai,ab,gbj,g->ij", Bu, D, Bu, detJ)
    if not incompatible:
        return Kuu
    J0 = DN0.T @ X
    detJ0 = jnp.linalg.det(J0)
    Ba = jax.vmap(lambda d, s: s * bmat(_spatial(J0, d)))(DP, detJ0 / detJ)  # (gp, 6, 9)
    Kua = jnp.einsum("gai,ab,gbj,g->ij", Bu, D, Ba, detJ)
    Kaa = jnp.einsum("gai,ab,gbj,g->ij", Ba, D, Ba, detJ)
    return Kuu - Kua @ jnp.linalg.solve(Kaa, Kua.T)


def hex8_centroid_stress(X, ue, E, nu):
    """Stress at the element centroid. The incompatible modes have zero strain there."""
    B0 = bmat(_spatial(DN0.T @ X, DN0))
    return dmat(E, nu) @ (B0 @ ue)


def von_mises(s):
    sxx, syy, szz, syz, sxz, sxy = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4], s[..., 5]
    return jnp.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                    + 3.0 * (syz ** 2 + sxz ** 2 + sxy ** 2))


ke_batch = jax.jit(jax.vmap(hex8_ke, in_axes=(0, 0, None, None)), static_argnums=3)
stress_batch = jax.jit(jax.vmap(hex8_centroid_stress, in_axes=(0, 0, 0, None)))


# ---------------------------------------------------------------- global level

def box_mesh(L, nel):
    """Structured hex mesh of the box [0,Lx]x[0,Ly]x[0,Lz] with nel=(nx,ny,nz) elements."""
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


def element_dofs(conn):
    return (3 * conn[:, :, None] + np.arange(3)).reshape(len(conn), -1)


def assemble(conn, Ke, ndof):
    """Sum element matrices (ne, 24, 24) into a sparse CSR global matrix."""
    edofs = element_dofs(conn)
    n = edofs.shape[1]
    rows = np.repeat(edofs, n, axis=1).ravel()
    cols = np.tile(edofs, (1, n)).ravel()
    return sp.coo_matrix((np.asarray(Ke).ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()


def solve_dirichlet(K, f, fixed, u_fixed=0.0):
    """Solve K u = f with u[fixed] = u_fixed by eliminating the fixed DOFs."""
    u = np.zeros(K.shape[0])
    u[fixed] = u_fixed
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    rhs = f[free] - K[free][:, fixed] @ u[fixed]
    # ponytail: direct sparse LU per call; keep a factorization once adjoints reuse it (Stage 4)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    return u


def linear_static(nodes, conn, E, nu, f, fixed, u_fixed=0.0, incompatible=True):
    """Assemble and solve. E may be a scalar or per-element array. Returns (u, K, stress)."""
    Xe = nodes[conn]
    Ee = np.broadcast_to(np.asarray(E, float), len(conn))
    K = assemble(conn, ke_batch(Xe, Ee, nu, incompatible), 3 * len(nodes))
    u = solve_dirichlet(K, f, fixed, u_fixed)
    stress = np.asarray(stress_batch(Xe, u[element_dofs(conn)], Ee, nu))
    return u, K, stress
