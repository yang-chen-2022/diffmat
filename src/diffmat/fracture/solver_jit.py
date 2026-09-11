import jax
from jax import numpy as jnp

from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger
from diffmat.fracture.utilities import voigt_to_tensor, tensor_to_voigt
from diffmat.fracture.lippmann_schwinger import solve
from diffmat.fracture.solve_one_step import solve_one_load_step
from diffmat.commons.io import save_arrays_to_vti

import os


def compute_sigma_damaged(epsilon, params):
    """
    Compute stress with asymmetric degradation.

     :arg epsilon: strain in Voigt notation (6, Nx, Ny, Nz)
     :arg params:
          lmbda - spatially varying Lame parameter lambda
           mu - spatially varying Lame parameter mu
           d_phase - damage variable (Nx, Ny, Nz).
           k_stab -  Stabilisation parameter for the damage
    """
    lmbda, mu, d_phase, k_stab = params

    eps_tensor = voigt_to_tensor(epsilon)
    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    sigma = {}
    for sign, op in (("+", jnp.maximum), ("-", jnp.minimum)):
        tr_eps_signed = op(tr_eps, 0.0)

        eigvals, eigvecs = jnp.linalg.eigh(eps_tensor)
        eigvals_signed = op(eigvals, 0.0)

        eps_signed_tensor = jnp.einsum(
            "...ia,...a,...ja->...ij", eigvecs, eigvals_signed, eigvecs
        )

        eps_signed_v = tensor_to_voigt(eps_signed_tensor)

        sigma[sign] = 2.0 * mu * eps_signed_v
        sigma[sign] = sigma[sign].at[:3].add(lmbda * tr_eps_signed)

    return ((1.0 - d_phase[None, ...]) ** 2 + k_stab) * sigma["+"] + sigma["-"]


def compute_strain_energy(lmbda, mu, epsilon):
    """Compute the ONLY the positive/ tensile elastic strain energy to drive the fracture.

    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter mu
    :arg epsilon: strain in Voigt notation [11,22,33,12,13,23], shape (6, Nx, Ny, Nz)
    """
    
    eps_tensor = voigt_to_tensor(epsilon)

    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)

    eigvals = jnp.linalg.eigvalsh(eps_tensor)

    eigvals_plus = jnp.maximum(eigvals, 0.0)
    eps_sq_plus = jnp.sum(eigvals_plus**2, axis=-1)

    psi_plus = 0.5 * lmbda * (tr_eps_plus**2) + mu * eps_sq_plus

    return psi_plus


# @jax.jit(static_argnames=["grid", "tolerance", "maxiter"])
def phase_field_solve(HH, d_old, gc, lc, grid, tolerance=1e-6, maxiter=1000, verbose=0):
    """Fixed-point iteration solver for phase-field problem (fracture)

    :arg HH: history strain energy (field), (1, Nx, Ny, Nz)
    :arg d_old: damage variable at previous time step (field), (1, Nx, Ny, Nz)
    :arg gc: fracture toughness (field), (1, Nx, Ny, Nz)
    :arg lc: regularisation length (field)), (1, Nx, Ny, Nz)
    :arg grid: grid specs
    :arg tolerance: tolerance for convergence check
    :arg maxiter: maximal number of iterations
    """

    # Coefficients A^t_n and B^t_n
    A_n = 1.0 / (lc**2) + 2.0 * HH / (gc * lc)
    B_n = 2.0 * HH / (gc * lc)

    d_final = solve(
        B_n, 
        A_n, 
        grid, 
        #u_in=jax.lax.stop_gradient(d_old), 
        u_in=None, 
        tol=tolerance, 
        maxits=maxiter, 
        verbose=verbose
    )

    return d_final





# Incremental solution with staggered scheme
@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "maxiter_PF",
        "maxiter_Elas",
        "maxiter_inner",
    ),
)
def solve_loading_history(
    Emean_steps,
    lmbda,
    mu,
    gc,
    lc,
    grid,
    k_stab,
    maxiter_PF,
    maxiter_Elas,
    maxiter_inner,
    tolerance_inner,
):

    dtype = lmbda.dtype

    d0 = jnp.zeros(
        (grid.nx, grid.ny, grid.nz),
        dtype=dtype,
    )

    HH0 = jnp.zeros_like(d0)

    epsilon0 = jnp.zeros(
        (6, grid.nx, grid.ny, grid.nz),
        dtype=dtype,
    ) + E_mean[:, None, None, None]

    lmbda0 = jax.lax.stop_gradient(0.5 * (lmbda.max() + lmbda.min()))
    mu0 = jax.lax.stop_gradient(0.5 * (mu.max() + mu.min()))

    def step_fn(carry, Emean):

        d, epsilon, HH = carry

        (
            d,
            epsilon,
        ) = solve_one_load_step(
            (
                d,
                epsilon,
            ),
            (
                lmbda,
                mu,
                gc,
                lc,
                Emean,
                HH,
                grid,
                lmbda0,
                mu0,
                k_stab,
                maxiter_PF,
                maxiter_Elas,
            ), 
            maxiter_inner,
            tolerance_inner,
        )

        sigma = compute_sigma_damage(
                epsilon,
                (lmbda, mu, d, k_stab),
            )

        psi = compute_strain_energy(
                lmbda,
                mu,
                epsilon,
            )
        HH = jnp.maximum(HH, jax.lax.stop_gradient(psi))

        eps_macro = jnp.mean(epsilon, axis=(1, 2, 3))
        sig_macro = jnp.mean(sigma, axis=(1, 2, 3))

        output = (
            eps_macro,
            sig_macro,
            epsilon,
            sigma,
            d,
        )

        return (d, epsilon, HH), output

    _, outputs = jax.lax.scan(
        step_fn,
        (d0, epsilon0, HH0),
        Emean_steps,
    )

    return outputs


