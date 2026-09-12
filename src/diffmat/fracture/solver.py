import jax
from jax import numpy as jnp

from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger
from diffmat.fracture.utilities import voigt_to_tensor, tensor_to_voigt
from diffmat.fracture.lippmann_schwinger import solve
from diffmat.fracture.constitutive import compute_sigma_damaged, compute_strain_energy
from diffmat.commons.io import save_arrays_to_vti

import os


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
    load_reduction_factor=None,
    output_fields=False,
):
    """
    Staggered scheme for solving elasticity + phase-field fracture equations.

    Parameters:
    -----------
    grid : GridSpec
        Computational grid specification
    lmbda : array
        Lamé parameter lambda field (nx, ny, nz)
    mu : array
        Lamé parameter mu (shear modulus) field (nx, ny, nz)
    gc : array
        Fracture toughness field (nx, ny, nz)
    lc : array
        Characteristic length field (nx, ny, nz)
    Emean_steps : list
        List of macroscopic strain vectors for each load step
    save_steps : array
        Indices of steps at which to save VTK output
    k_stab : float
        Stabilization parameter for damage (default: 1e-6)
    maxiter_PF : int
        Max iterations for phase-field solver (default: 10000)
    maxiter_Elas : int
        Max iterations for elasticity solver (default: 10000)
    out_dir : str
        Output directory for VTK files and macro_curve.txt (default: "")
    earlystop : float, optional
        Early stopping parameter: stop when stress < earlystop * peak_stress
        after consecutive decreasing steps. If None, run full loading.
    maxiter_inner : int
        Max inner loop iterations for staggered scheme (default: 1)
    tolerance_inner : float
        Tolerance for inner loop convergence (default: 1e-5)
    load_reduction_factor : float, optional
        Factor for load step subdivision if inner loop doesn't converge
    output_fields : bool
        If True, returns stress/strain/damage fields at each saved step. If False, returns None.
        (default: False)

    Returns:
    --------
    eps_steps : array
        Macroscopic strain at each step, shape (n_steps, 6)
    sig_steps : array
        Macroscopic stress at each step, shape (n_steps, 6)
    stress_field : array or None
        Local stress field (n_steps, 6, nx, ny, nz)
    strain_field : array or None
        Local strain field (n_steps, 6, nx, ny, nz)
    damage_field : array or None
        Local damage field (n_steps, 1, nx, ny, nz)
    """

    if maxiter_inner < 1:
        raise ValueError("maxiter_inner must be at least 1")

    if load_reduction_factor is not None and not 0.0 < load_reduction_factor < 1.0:
        raise ValueError("load_reduction_factor must be between 0 and 1")

    if load_reduction_factor is not None and maxiter_inner < 2:
        raise ValueError("maxiter_inner must be >= 2 when load step subdivision is active")


    dtype = lmbda.dtype

    lmbda0 = 0.5 * (lmbda.max() + lmbda.min())
    mu0 = 0.5 * (mu.max() + mu.min())

    # initialize damage field & history field & local strain pertubation
    d = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype)
    HH = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype)
    depsilon = jnp.zeros((6, grid.nx, grid.ny, grid.nz), dtype)
    depsilon = jax.lax.stop_gradient(depsilon)

    # variable placeholder
    sig_steps = []
    eps_steps = []
    stress_field = [] if output_fields else None  # Only allocate if needed
    strain_field = [] if output_fields else None  # Only allocate if needed
    damage_field = [] if output_fields else None  # Only allocate if needed

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
    load_steps = list(Emean_steps)
    previous_load = jnp.zeros_like(load_steps[0]) if load_steps else None
    step = 0
    while step < len(load_steps):
        E_mean = load_steps[step]
        print(f"======== Time Step {step}  ========")

        d_start = d
        HH_start = HH
        d_previous = d
        epsilon_previous = None
        converged = False

        for inner_iteration in range(maxiter_inner):
            d = phase_field_solve(
                HH,
                d,
                gc,
                lc,
                grid,
                tolerance=1e-5,
                maxiter=maxiter_PF,
                verbose=0,
            )
            jax.block_until_ready(d)

            epsilon, sigma = lippmann_schwinger(
                compute_sigma_damaged,
                (lmbda, mu, d, k_stab),
                E_mean,
                delta_epsilon_initial=depsilon,
                ref_params={"lambda": lmbda0, "mu": mu0},
                grid_spec=grid,
                tol=1.0e-2,
                maxits=maxiter_Elas,
                verbose=0,
                depth=4,
            )   
            jax.block_until_ready(epsilon)

            depsilon = epsilon - E_mean[:, None, None, None]
            depsilon = jax.lax.stop_gradient(depsilon)

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
                    converged = True
                    break

            epsilon_previous = epsilon
            d_previous = d

        print(
            f"Time Step {step}: completed {inner_iteration + 1} inner iteration(s)"
        )

        if not converged and load_reduction_factor is not None:
            load_increment = E_mean - previous_load
            if bool(jnp.linalg.norm(load_increment) <= 1e-12): #TODO: parameterise this: min_load_increment=1e-12
                raise RuntimeError(
                    "Fracture statggered solve reached maxiter_inner at the "
                    "minimum load increment withouth converging"
                )

            reduced_load = previous_load + load_reduction_factor * load_increment
            load_steps[step : step + 1] = [reduced_load, E_mean]
            d = d_start
            HH = HH_start
            print(
                f"Inner loop did not converge at step [step]; "
                f"retrying with load increment factor {load_reduction_factor}"
            )
            continue

        previous_load = E_mean

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
            # Store final stress/strain/damage fields if requested
            if output_fields:
                stress_field.append(sigma)
                strain_field.append(epsilon)
                damage_field.append(d[None,...])

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
                    # Store final stress/strain/damage fields if requested
                    if output_fields:
                        stress_field.append(sigma)
                        strain_field.append(epsilon)
                        damage_field.append(d[None,...])

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

        step +=1

    if output_fields and damage_field:
        stress_array = jnp.array(stress_field, dtype=dtype)
        strain_array = jnp.array(strain_field, dtype=dtype)
        damage_array = jnp.array(damage_field, dtype=dtype)
    else:
        stress_array = None
        strain_array = None
        damage_array = None

    return jnp.array(eps_steps), jnp.array(sig_steps), stress_array, strain_array, damage_array


