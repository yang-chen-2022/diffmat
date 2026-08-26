import jax
from jax import numpy as jnp

from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger
from diffmat.fracture.utilities import voigt_to_tensor, tensor_to_voigt
from diffmat.fracture.lippmann_schwinger import solve
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
        B_n, A_n, grid, jax.lax.stop_gradient(d_old), tolerance, maxiter, verbose
    )

    return d_final


# Staggered scheme for solving elasticity + phase-field equations
def solve_fracture_staggered(
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
    out_dir="",
    earlystop=None,
    maxiter_inner=1,
    tolerance_inner=1e-5,
):

    if maxiter_inner < 1:
        raise ValueError("maxiter_inner must be at least 1")

    dtype = lmbda.dtype

    lmbda0 = 0.5 * (lmbda.max() + lmbda.min())
    mu0 = 0.5 * (mu.max() + mu.min())

    # initialize damage field & history field
    d = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype)
    HH = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype)

    # variable placeholder
    sig_steps = []
    eps_steps = []

    # output file for macroscopic stresses & strains
    file_path = os.path.join(out_dir, "macro_curve.txt")
    with open(file_path, "w") as f:
        header = (
            f"{'step':>8}"
            f"{'e11':>15}{'e22':>15}{'e33':>15}"
            f"{'e12':>15}{'e13':>15}{'e23':>15}"
            f"{'s11':>15}{'s22':>15}{'s33':>15}"
            f"{'s12':>15}{'s13':>15}{'s23':>15}"
            f"{'vtk':>15}\n"
        )
        f.write(header)

    # variables for early stopping
    peak_stress = 0.0  # Track the peak stress
    prev_sig_norm = 0.0  # Track previous stress norm
    decreasing_steps = 0  # Count consecutive steps of decreasing stress
    min_decreasing_steps = (
        3  # Require at least this many consecutive decreasing steps to confirm trend
    )

    # Solution loop
    for step, E_mean in enumerate(Emean_steps):
        print(f"======== Time Step {step}  ========")

        d_previous = d
        epsilon_previous = None
        for inner_iteration in range(maxiter_inner):
            # Solve both fields at the same load level until they stop changing.
            d = phase_field_solve(
                HH,
                d,
                gc,
                lc,
                grid,
                tolerance=1e-5,
                maxiter=maxiter_PF,
                verbose=1,
            )
            jax.block_until_ready(d)

            epsilon, sigma = lippmann_schwinger(
                compute_sigma_damaged,
                (lmbda, mu, d, k_stab),
                E_mean,
                ref_params={"lambda": lmbda0, "mu": mu0},
                grid_spec=grid,
                tol=1.0e-2,
                maxits=maxiter_Elas,
                verbose=1,
                depth=4,
            )   #NOTE: initialise with previous step's solution to speed up convergence
            jax.block_until_ready(epsilon)

            psi = compute_strain_energy(lmbda, mu, epsilon)
            HH = jnp.maximum(HH, psi)
            jax.block_until_ready(HH)

            if epsilon_previous is not None:
                strain_delta = voigt_to_tensor(epsilon - epsilon_previous)
                strain_delta_norms = jnp.linalg.norm(strain_delta, axis=(-2, -1))
                strain_norms = jnp.linalg.norm(
                    voigt_to_tensor(epsilon), axis=(-2, -1)
                )
                previous_strain_norms = jnp.linalg.norm(
                    voigt_to_tensor(epsilon_previous), axis=(-2, -1)
                )
                strain_scale = jnp.maximum(
                    jnp.maximum(strain_norms, previous_strain_norms),
                    jnp.finfo(epsilon.dtype).tiny,
                )
                strain_change = jnp.max(strain_delta_norms / strain_scale)
                damage_change = jnp.max(jnp.abs(d - d_previous))
                if bool((strain_change < tolerance_inner) & (damage_change < tolerance_inner)):
                    break

            epsilon_previous = epsilon
            d_previous = d

        #  Save & display
        sigAV = jnp.array([jnp.mean(sigma[i]) for i in range(6)])
        sig_steps.append(sigAV)

        epsAV = jnp.array([jnp.mean(epsilon[i]) for i in range(6)])
        eps_steps.append(epsAV)

        # Save vtk fields
        vtk_saved = False
        if step in save_steps:
            save_arrays_to_vti(
                filename=f"{out_dir}/fields_{step:04d}.vtk",
                arrays=[epsilon, sigma, d[None, ...]],
                names=["Strain", "Stress", "Damage"],
                spacing=grid.grid_spacings,
                origin=(0, 0, 0),
                stack_components=True,
            )
            vtk_saved = True

        # Early stopping condition: stop after peak stress when consistently decreasing
        break_flag = False
        if earlystop is not None:
            sig_norm = jnp.linalg.norm(sigAV)

            # Update peak stress
            if sig_norm > peak_stress:
                peak_stress = sig_norm
                decreasing_steps = 0  # Reset counter when peak increases
            elif step > 0 and sig_norm < prev_sig_norm:
                # Stress is decreasing
                decreasing_steps += 1
            else:
                # Stress increased or stayed same after peak, reset counter
                decreasing_steps = 0

            # Check stopping condition: consistently decreasing AND below threshold
            if decreasing_steps >= min_decreasing_steps:
                threshold_value = peak_stress * earlystop
                if sig_norm < threshold_value:
                    save_arrays_to_vti(
                        filename=f"{out_dir}/fields_{step:04d}.vtk",
                        arrays=[epsilon, sigma, d[None, ...]],
                        names=["Strain", "Stress", "Damage"],
                        spacing=grid.grid_spacings,
                        origin=(0, 0, 0),
                        stack_components=True,
                    )

                    print(
                        f"Early stopping at step {step}: stress norm {sig_norm:.6f} < threshold {threshold_value:.6f} "
                        f"({earlystop * 100}% of peak {peak_stress:.6f}) after {decreasing_steps} consecutive decreasing steps"
                    )
                    vtk_saved = True
                    break_flag = True

            prev_sig_norm = sig_norm

        # write macro stress, strain
        with open(os.path.join(out_dir, "macro_curve.txt"), "a") as f:
            line = (
                f"{step:8d}"
                + "".join(f"{x:15.6e}" for x in epsAV)
                + "".join(f"{x:15.6e}" for x in sigAV)
                + f"{int(vtk_saved):15d}"
                + "\n"
            )
            f.write(line)

        if break_flag:
            break

    return jnp.array(eps_steps), jnp.array(sig_steps)

