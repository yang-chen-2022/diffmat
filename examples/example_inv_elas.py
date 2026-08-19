"""
Inverse identification (Newton-Raphson) example for linear elasticity
on a particle-reinforced composite RVE.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from diffmat.fracture.rvegen import (
    generate_particles_periodic,
    voxelise_particles_periodic,
    init_material,
)
from diffmat.commons.utilities import eng2lame

from jaxmaterials.common import get_grid_spec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger


import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')


# ---------- Reparameterization utils ----------
nu_min = -0.49
nu_max = 0.49

def s_to_nu(s):
    """Map unconstrained scalar s -> nu in (nu_min, nu_max)."""
    return nu_min + (nu_max - nu_min) * jax.nn.sigmoid(s)

def nu_to_s(nu):
    """Inverse map (nu in (nu_min, nu_max)) -> s (real)."""
    # clip to avoid exact 0/1
    eps = 1e-12
    frac = (nu - nu_min) / (nu_max - nu_min)
    frac = jnp.clip(frac, eps, 1.0 - eps)
    return jnp.log(frac / (1.0 - frac))

# --------------------------------------------


def build_rve(box_size, spacing, n_particles, radius_range, seed=42):
    """Generate periodic particle RVE and voxelize it."""
    np.random.seed(seed)
    grid = get_grid_spec(
        box_size[0],
        box_size[1],
        box_size[2],
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[2],
    )

    positions, radii = generate_particles_periodic(n_particles, box_size, radius_range)
    matID = voxelise_particles_periodic(grid, positions, radii)

    return grid, matID


def forward_sigma_vector(u, grid, matID, eps_probes, dtype=jnp.float64):
    """
    Forward model: build lambda/mu fields from params and compute macroscopic sigma
    for a list of macroscopic strain probes eps_probes (list/array of 6-vectors).
    Returns concatenated sigma vectors: (n_probes*6,).
    u: [logE_matrix, logE_particle, s_nu_matrix, s_nu_particle]
    """

    # Map to physical parameters
    logE_matrix, logE_particle, s_nu_matrix, s_nu_particle = u
    E_matrix = jnp.exp(logE_matrix)
    E_particle = jnp.exp(logE_particle)
    nu_matrix = s_to_nu(s_nu_matrix)
    nu_particle = s_to_nu(s_nu_particle)

    # Convert to Lamé params for each phase (scalar per phase)
    lmbda_matrix, mu_matrix = eng2lame(E_matrix, nu_matrix)
    lmbda_particle, mu_particle = eng2lame(E_particle, nu_particle)

    # Create grids
    lmbda_grid, mu_grid, _, _ = init_material(
        matID,
        lmbda_list=[lmbda_matrix, lmbda_particle],
        mu_list=[mu_matrix, mu_particle],
        gc_list=[0.0, 0.0],
        lc_list=[1.0, 1.0],
        dtype=dtype,
    )

    # reference (homogeneous) parameters for solver (stop_gradient so not part of AD)
    lmbda0 = jax.lax.stop_gradient(0.5 * (jnp.max(lmbda_grid) + jnp.min(lmbda_grid)))
    mu0 = jax.lax.stop_gradient(0.5 * (jnp.max(mu_grid) + jnp.min(mu_grid)))

    sigma_list = []
    for eps in eps_probes:
        eps = jnp.asarray(eps, dtype=dtype)
        epsilon, sigma = lippmann_schwinger(
            compute_sigma_iso,
            (lmbda_grid, mu_grid),
            eps,
            ref_params={"lambda": lmbda0, "mu": mu0},
            grid_spec=grid,
            tol=1.0e-4,
            maxits=2000,
            verbose=0,
            depth=4,
        )

        sigma_macro = jnp.mean(sigma, axis=[1, 2, 3])  # shape (6,)
        sigma_list.append(sigma_macro)

    sigma_concat = jnp.concatenate(sigma_list, axis=0)  # (n_probes*6,)
    return sigma_concat


def compute_sigma_iso(epsilon, params):
    # params = (lmbda_field, mu_field) both are (nx,ny,nz) arrays broadcastable to epsilon
    lmbda, mu = params
    tr = epsilon[0] + epsilon[1] + epsilon[2]
    sigma = jnp.zeros_like(epsilon)
    sigma = sigma.at[:3].set((lmbda * tr)[None, ...] + 2.0 * mu * epsilon[:3])
    sigma = sigma.at[3:].set(2.0 * mu * epsilon[3:])
    return sigma


def newton_raphson_inverse(
    sigma_target,
    u0,
    grid,
    matID,
    eps_probes,
    maxiter=20,
    tol=1e-6,
    reg=1e-8,
    damp_init=1.0,
):
    """
    Newton-Raphson loop to solve r(u) = 0 where r = forward(u) - sigma_target.

    Uses Jacobian J = dr/du computed with JAX (finite dimension: m x p).
    Solves normal equations (J^T J + reg I) delta = -J^T r for stability (p small).
    """

    dtype = jnp.float64

    u = jnp.asarray(u0, dtype=dtype)
    sigma_target = jnp.asarray(sigma_target, dtype=dtype)

    # jitted forward/residual/jacobian for speed
    forward_fn = lambda p: forward_sigma_vector(p, grid, matID, eps_probes)
    residual_fn = lambda p: forward_fn(p) - sigma_target

    jitted_residual = jax.jit(residual_fn)
    jitted_jac = jax.jit(jax.jacobian(residual_fn))

    history = {"res_norm": [], "u": [], "svd": []}

    damp = damp_init

    for k in range(maxiter):
        r = jitted_residual(u)  # shape (m,)
        r_norm = jnp.linalg.norm(r)
        history["res_norm"].append(float(r_norm))
        history["u"].append(np.array(u))
        print(f"[Iter {k}] residual norm = {r_norm:.6e}")

        if r_norm < tol:
            print("Converged.")
            break

        J = jitted_jac(u)  # shape (m, p)

        # diagnostics: singular values
        try:
            sv = jnp.linalg.svd(J, compute_uv=False)
            cond = float(sv[0] / (sv[-1] + 1e-30))
            history["svd"].append(np.array(sv))
            print(f"  singular values (J): {sv}")
            print(f"  cond(J) ~ {cond:.3e}")
        except Exception:
            sv = None

        # Normal equations: (J^T J + reg I) delta = -J^T r
        JTJ = J.T @ J
        rhs = -J.T @ r

        # Regularize
        JTJ_reg = JTJ + reg * jnp.eye(JTJ.shape[0], dtype=JTJ.dtype)

        # Solve for delta_u in parameter space u
        try:
            delta_u = jnp.linalg.solve(JTJ_reg, -JT_r)
        except Exception as e:
            # fallback to lstsq if solve fails
            delta_u, *_ = jnp.linalg.lstsq(J, -r, rcond=None)

        # Backtracking line search / damping
        alpha = damp
        success = False
        r_norm_current = r_norm
        for trial in range(10):
            u_candidate = u + alpha * delta_u
            r_new = jitted_residual(u_candidate)
            r_new_norm = jnp.linalg.norm(r_new)
            if r_new_norm < r_norm_current:
                success = True
                print(f"  Accept step with alpha={alpha:.3f}, new residual {r_new_norm:.6e}")
                u = u_candidate
                # slightly reduce reg when successful to speed up convergence
                reg = max(reg * 0.9, 1e-12)
                damp = min(1.0, damp * 1.2)
                break
            else:
                alpha *= 0.5

        if not success:
            # If we couldn't find an improving alpha, increase regularization and try tiny step
            print("  Line search failed to reduce residual; increasing regularization and taking small step.")
            reg = reg * 10.0 + 1e-12
            u = u + 1e-2 * delta_u  # small guarded step
            damp = max(1e-3, damp * 0.5)

    return u, history


# helper
def pack_true_to_u(E_matrix, E_particle, nu_matrix, nu_particle):
    logE_matrix = jnp.log(E_matrix)
    logE_particle = jnp.log(E_particle)
    s_nu_matrix = nu_to_s(nu_matrix)
    s_nu_particle = nu_to_s(nu_particle)
    return jnp.asarray([logE_matrix, logE_particle, s_nu_matrix, s_nu_particle], dtype=jnp.float64)



if __name__ == "__main__":
    # Example usage: create synthetic target from known parameters, then try to recover them.

    # RVE geometry
    box_size = [0.4, 0.4, 0.4]
    spacing = [0.01, 0.01, 0.01]
    n_particles = 20
    radius_range = [0.05, 0.08]

    print("Building RVE...")
    grid, matID = build_rve(box_size, spacing, n_particles, radius_range)

    # True parameters (generate synthetic data)
    E_true_matrix = 22e3
    E_true_particle = 175e3
    nu_true_matrix = 0.37
    nu_true_particle = 0.28

    u_true = pack_true_to_u(E_true_matrix, E_true_particle, nu_true_matrix, nu_true_particle)

    # macroscopic strain probes (Voigt: [e11,e22,e33,e12,e13,e23])
    eps_probe1 = jnp.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)  # e11
    eps_probe2 = jnp.array([0.0, 1e-3, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)  # e22
    eps_probe3 = jnp.array([0.0, 0.0, 1e-3, 0.0, 0.0, 0.0], dtype=jnp.float64)  # e33
    eps_probe4 = jnp.array([0.0, 0.0, 0.0, 1e-3, 0.0, 0.0], dtype=jnp.float64)  # e12
    eps_probes = [eps_probe1, eps_probe2, eps_probe3, eps_probe4]

    print("Computing synthetic target sigma (forward solve with true params)...")
    sigma_target = forward_sigma_vector(u_true, grid, matID, eps_probes)

    # Add small measurement noise if desired:
    #sigma_target = sigma_target + 1e-2 * jax.random.normal(jax.random.PRNGKey(0), sigma_target.shape)

    # Initial guess
    E_init_matrix = 10e3
    E_init_particle = 50e3
    nu_init_matrix = 0.30
    nu_init_particle = 0.30
    u0 = pack_true_to_u(E_init_matrix, E_init_particle, nu_init_matrix, nu_init_particle)

    # Run Newton-Raphson
    print("Starting inverse solve (reparameterized variables)...")
    t0 = time.time()
    u_opt, history = newton_raphson_inverse(
        sigma_target,
        u0,
        grid,
        matID,
        eps_probes,
        maxiter=20,
        tol=1e-6,
        reg=1e-8,
        damp_init=1.0,
    )


    # Unpack optimized parameters
    logE_m_opt, logE_p_opt, s_nu_m_opt, s_nu_p_opt = u_opt
    E_m_opt = jnp.exp(logE_m_opt)
    E_p_opt = jnp.exp(logE_p_opt)
    nu_m_opt = s_to_nu(s_nu_m_opt)
    nu_p_opt = s_to_nu(s_nu_p_opt)

    print("True params:")
    print("  E_matrix =", float(E_true_matrix), "E_particle =", float(E_true_particle))
    print("  nu_matrix =", float(nu_true_matrix), "nu_particle =", float(nu_true_particle))
    print("Recovered params:")
    print("  E_matrix =", float(E_m_opt), "E_particle =", float(E_p_opt))
    print("  nu_matrix =", float(nu_m_opt), "nu_particle =", float(nu_p_opt))

    # Optionally, show residual history
    print("Residual norm history:", history["res_norm"])


