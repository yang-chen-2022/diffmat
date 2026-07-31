import jax
from jax import numpy as jnp

from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger
from diffmat.utilities import voigt_to_tensor, tensor_to_voigt
from diffmat.lippmann_schwinger import solve


def compute_sigma_damaged(epsilon, params):
    """
    Compute stress with asymmetric degradation.

     :arg epsilon: strain in Voigt notation (6, Nx, Ny, Nz)
     :arg params:
          lmbda - spatially varying Lame parameter lambda
           mu - spatially varying Lame parameter mu
           d - damage variable (Nx, Ny, Nz).
           k -  Stabilisation parameter for the damage
    """

    lmbda, mu, d, k = params

    eps_tensor = voigt_to_tensor(epsilon)

    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)
    tr_eps_minus = jnp.minimum(tr_eps, 0.0)

    # Get eigenvalues n eigenvectors
    eigvals, eigvecs = jnp.linalg.eigh(eps_tensor)

    eigvals_plus = jnp.maximum(eigvals, 0.0)
    eigvals_minus = jnp.minimum(eigvals, 0.0)

    # Reconstruct the positive and negative strain tensors (eps_plus / eps_minus)
    # This uses einsum to do: V * Lambda_plus * V^T across the entire 3D grid instantly
    eps_plus_tensor = jnp.einsum(
        "...ia,...a,...ja->...ij", eigvecs, eigvals_plus, eigvecs
    )
    eps_minus_tensor = jnp.einsum(
        "...ia,...a,...ja->...ij", eigvecs, eigvals_minus, eigvecs
    )

    # Convert back to Voigt notation for the stress equation
    eps_plus_v = tensor_to_voigt(eps_plus_tensor)
    eps_minus_v = tensor_to_voigt(eps_minus_tensor)

    # Calculate pure tension stress and pure compression stress
    sigma_plus = 2.0 * mu * eps_plus_v
    sigma_minus = 2.0 * mu * eps_minus_v
    vol = vol = lmbda * tr_eps_plus
    sigma_plus = sigma_plus.at[0].add(vol)
    sigma_plus = sigma_plus.at[1].add(vol)
    sigma_plus = sigma_plus.at[2].add(vol)

    vol = lmbda * tr_eps_minus
    sigma_minus = 2.0 * mu * eps_minus_v
    sigma_minus = sigma_minus.at[0].add(vol)
    sigma_minus = sigma_minus.at[1].add(vol)
    sigma_minus = sigma_minus.at[2].add(vol)

    # Apply damage degradation (g_d) ONLY to the tension (positive) stress
    return ((1.0 - d[None, ...]) ** 2 + k) * sigma_plus + sigma_minus


def compute_strain_energy(lmbda, mu, epsilon):
    """Compute the ONLY the positive/ tensile elastic strain energy to drive the fracture.

    :arg lmbda: spatially varying Lame parameter lambda
    :arg mu: spatially varying Lame parameter mu
    :arg epsilon: strain in Voigt notation [11,22,33,12,13,23], shape (6, Nx, Ny, Nz)
    """
    # 1. Convert to 3x3 tensor
    eps_tensor = voigt_to_tensor(epsilon)

    # 2. Get trace and split into positive part
    tr_eps = jnp.trace(eps_tensor, axis1=-2, axis2=-1)
    tr_eps_plus = jnp.maximum(tr_eps, 0.0)

    # 3. Calculate eigenvalues using JAX's eigh function
    eigvals = jnp.linalg.eigvalsh(eps_tensor)

    # 4. Filter only the positive eigenvalues
    eigvals_plus = jnp.maximum(eigvals, 0.0)
    eps_sq_plus = jnp.sum(eigvals_plus**2, axis=-1)

    # 5. Compute only the tensile energy (psi_plus)
    psi_plus = 0.5 * lmbda * (tr_eps_plus**2) + mu * eps_sq_plus

    return psi_plus


# @jax.jit(static_argnames=["grid", "tolerance", "maxiter"])
def phase_field_solve(HH, d_old, gc, lc, grid, tolerance=1e-6, maxiter=1000):
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

    d_final = solve(B_n, A_n, grid, d_old, tolerance, maxiter)

    # temporary fix: do we really need to know the number of Lippmann-Schwinger iterations?
    iter_count = 0
    return d_final, iter_count


# Staggered scheme for solving elasticity + phase-field equations
def elastodamage_phasefield_solve(
    grid,
    lmbda,
    mu,
    gc,
    lc,
    Emean_steps,
    save_steps,
    k_stab=1e-6,
    maxiter_PF=10000,
    maxiter_Elas=10000,
):

    dtype = lmbda.dtype

    lmbda0 = 0.5 * (lmbda.max() + lmbda.min())
    mu0 = 0.5 * (mu.max() + mu.min())

    # initialize damage field & history field
    d = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype)
    HH = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype)

    # variable placeholder
    dfield = {}
    epsfield = {}
    sigfield = {}

    sig_steps = []
    eps_steps = []

    for step, E_mean in enumerate(Emean_steps):
        print(f"======== Time Step {step}  ========")

        # solve phase-field
        d, iter_pf = phase_field_solve(
            HH,
            d,
            gc,
            lc,
            grid,
            tolerance=1e-6,
            maxiter=maxiter_PF,
        )
        jax.block_until_ready(iter_pf)
        print(f"    PF solve: {iter_pf} iterations.")

        # solve elasticity
        epsilon, sigma = lippmann_schwinger(
            compute_sigma_damaged,
            (lmbda, mu, d, k_stab),
            E_mean,
            ref_params={"lambda": lmbda0, "mu": mu0},
            grid_spec=grid,
            tol=1.0e-5,
            maxits=maxiter_Elas,
            verbose=1,
            depth=4,
        )

        jax.block_until_ready(epsilon)

        #  Save & display
        sigAV = jnp.array([jnp.mean(sigma[i]) for i in range(6)])
        sig_steps.append(sigAV)

        epsAV = jnp.array([jnp.mean(epsilon[i]) for i in range(6)])
        eps_steps.append(epsAV)

        # update the history field
        psi = compute_strain_energy(lmbda, mu, epsilon)
        HH = jnp.maximum(HH, psi)
        jax.block_until_ready(HH)

        if step in save_steps:
            dfield[step] = d
            epsfield[step] = epsilon
            sigfield[step] = sigma

    return jnp.array(eps_steps), jnp.array(sig_steps), epsfield, sigfield, dfield
