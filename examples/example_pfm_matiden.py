"""
Inverse identification (Newton-Raphson) example for phase-field fracture model.

Goal: Identify the fracture toughness (gc) and characteristic length (lc)
for particles and matrix from:
  - Known microstructure (particle positions and radii)
  - Known elastic moduli (E, nu for particles and matrix)
  - Known macroscopic strain loading history
  - Reference data (from prior simulation):
    * One measured stress component (e.g., tensile stress σ11 in uniaxial tension)
    * Full-field strain measurement (e.g., from digital image correlation)
    * Crack geometry / damage field (after image processing)

The inverse solver uses a weighted combination of:
1. Measured stress component: σ_measured (scalar per step)
2. Full-field strain: ε_measured (6 components, spatial field)
3. Damage field: d_measured (spatial field, thresholded from image)

Feature: Experimental measurements have DIFFERENT resolution than FFT simulation.
Coarsening is applied to match resolutions for comparison.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from diffmat.fracture.rvegen import (
    generate_particles_periodic,
    voxelise_particles_periodic,
    init_material,
)
from diffmat.commons.utilities import eng2lame, newton_raphson
from diffmat.commons.io import save_arrays_to_vti
from diffmat.fracture.solver import solve_fracture_staggered

from jaxmaterials.common import get_grid_spec

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')


# ---------- Reparameterization utils for gc and lc ----------
gc_min = 1e-4  # N/mm
gc_max = 1e-2  # N/mm
lc_min = 0.001  # mm
lc_max = 0.05   # mm


def s_to_gc(s):
    """Map unconstrained scalar s -> gc in (gc_min, gc_max)."""
    return gc_min + (gc_max - gc_min) * jax.nn.sigmoid(s)


def gc_to_s(gc):
    """Inverse map (gc in (gc_min, gc_max)) -> s (real)."""
    eps = 1e-12
    frac = (gc - gc_min) / (gc_max - gc_min)
    frac = jnp.clip(frac, eps, 1.0 - eps)
    return jnp.log(frac / (1.0 - frac))


def s_to_lc(s):
    """Map unconstrained scalar s -> lc in (lc_min, lc_max)."""
    return lc_min + (lc_max - lc_min) * jax.nn.sigmoid(s)


def lc_to_s(lc):
    """Inverse map (lc in (lc_min, lc_max)) -> s (real)."""
    eps = 1e-12
    frac = (lc - lc_min) / (lc_max - lc_min)
    frac = jnp.clip(frac, eps, 1.0 - eps)
    return jnp.log(frac / (1.0 - frac))


# ---------------------------------------------------------------
# Coarsening utilities to handle resolution mismatch
# ---------------------------------------------------------------

def coarsen_field_3d(field, coarse_shape):
    """
    Coarsen a 3D field to match experimental resolution.
    
    Parameters:
    -----------
    field : array
        Input field, shape (nx, ny, nz) or (6, nx, ny, nz) for strain
    coarse_shape : tuple
        Target shape (nx_coarse, ny_coarse, nz_coarse)

    Returns:
    --------
    coarse_field : array
        Coarsened field, shape matching coarse_shape or (6, *coarse_shape)
    """
    if field.ndim == 3:
        fine_shape = field.shape
        
        stride = tuple(
            max(1, fine_dim // coarse_dim)
            for fine_dim, coarse_dim in zip(fine_shape, coarse_shape)
        )
        
        pad_x = (stride[0] * coarse_shape[0]) - fine_shape[0]
        pad_y = (stride[1] * coarse_shape[1]) - fine_shape[1]
        pad_z = (stride[2] * coarse_shape[2]) - fine_shape[2]
        
        field_padded = jnp.pad(
            field, 
            ((0, max(0, pad_x)), (0, max(0, pad_y)), (0, max(0, pad_z))),
            mode='edge'
        )
        
        coarse = jnp.mean(
            field_padded[:stride[0]*coarse_shape[0], :stride[1]*coarse_shape[1], :stride[2]*coarse_shape[2]]
            .reshape(coarse_shape[0], stride[0], coarse_shape[1], stride[1], coarse_shape[2], stride[2]),
            axis=(1, 3, 5)
        )
        return coarse

    elif field.ndim == 4:
        n_components = field.shape[0]
        coarse_list = []
        for comp in range(n_components):
            coarse_list.append(coarsen_field_3d(field[comp], coarse_shape))
        return jnp.stack(coarse_list, axis=0)

    else:
        raise ValueError(f"Expected 3D or 4D field, got {field.ndim}D")


def refine_field_3d_to_shape(field, fine_shape):
    """
    Refine (upsample) a field to match fine resolution using nearest-neighbor interpolation.
    
    Parameters:
    -----------
    field : array
        Input coarse field, shape (nx_coarse, ny_coarse, nz_coarse) or (6, nx_coarse, ny_coarse, nz_coarse)
    fine_shape : tuple
        Target shape (nx_fine, ny_fine, nz_fine)
    
    Returns:
    --------
    fine_field : array
        Refined field at fine resolution
    """
    if field.ndim == 3:
        # Single component
        coarse_shape = field.shape
        scale_x = fine_shape[0] / coarse_shape[0]
        scale_y = fine_shape[1] / coarse_shape[1]
        scale_z = fine_shape[2] / coarse_shape[2]
        
        fine = np.zeros(fine_shape, dtype=field.dtype)
        for i in range(fine_shape[0]):
            for j in range(fine_shape[1]):
                for k in range(fine_shape[2]):
                    i_coarse = int(i / scale_x)
                    j_coarse = int(j / scale_y)
                    k_coarse = int(k / scale_z)
                    i_coarse = min(i_coarse, coarse_shape[0] - 1)
                    j_coarse = min(j_coarse, coarse_shape[1] - 1)
                    k_coarse = min(k_coarse, coarse_shape[2] - 1)
                    fine[i, j, k] = field[i_coarse, j_coarse, k_coarse]
        return fine
    
    elif field.ndim == 4:
        n_components = field.shape[0]
        fine = np.zeros((n_components, *fine_shape), dtype=field.dtype)
        for comp in range(n_components):
            fine[comp] = refine_field_3d_to_shape(field[comp], fine_shape)
        return fine
    
    else:
        raise ValueError(f"Expected 3D or 4D field, got {field.ndim}D")


# ---------------------------------------------------------------




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


def forward_full_response(
    u,
    grid,
    matID,
    lmbda_grid,
    mu_grid,
    strain_loading,
    stress_component_idx=0,
    dvc_resolution=None,
    dvc_steps=None,
    dtype=jnp.float64,
):
    """
    Forward model: simulate fracture and return:
    - One measured stress component (e.g., σ11 for tensile test)
    - Full-field strain (6 components at all spatial points)
    - Damage field d (spatial field)

    u: [s_gc_matrix, s_gc_particle, s_lc_matrix, s_lc_particle]
    strain_loading: list of macroscopic strain vectors
    stress_component_idx: which stress component to extract (0-5 for Voigt notation)
    dvc_resolution: tuple (nx_exp, ny_exp, nz_exp) for experimental field resolution
                    If None, use full FFT resolution
    dvc_steps: step indices at which full-field scan / DVC were performed (if None, use all)

    Returns:
    --------
    dict with keys:
        'sigma_macro': macroscopic stress component at each step, shape (n_steps,)
        'strain_field': full strain field at experimental resolution, shape (n_steps_dic, 6, nx_exp, ny_exp, nz_exp)
        'damage_field': damage field at experimental resolution, shape (n_steps_dic, nx_exp, ny_exp, nz_exp)
    """

    s_gc_matrix, s_gc_particle, s_lc_matrix, s_lc_particle = u
    gc_matrix = s_to_gc(s_gc_matrix)
    gc_particle = s_to_gc(s_gc_particle)
    lc_matrix = s_to_lc(s_lc_matrix)
    lc_particle = s_to_lc(s_lc_particle)

    _, _, gc_grid, lc_grid = init_material(
        matID,
        lmbda_list=[0., 0.],
        mu_list=[0., 0.],   #dummy values for lambda and mu
        gc_list=[gc_matrix, gc_particle],
        lc_list=[lc_matrix, lc_particle],
        dtype=dtype,
    )

    strain_dvc = [strain_loading[i] for i in dvc_steps]

    n_steps = len(strain_loading)
    n_steps_dvc = len(dvc_steps)

    tmp_out_dir = "/tmp/pfm_inv_sim"
    os.makedirs(tmp_out_dir, exist_ok=True)

    _, sigAV, _, efield, dfield = solve_fracture_staggered(
        grid,
        lmbda_grid,
        mu_grid,
        gc_grid,
        lc_grid,
        strain_loading,
        dvc_steps,
        k_stab=1e-6,
        maxiter_PF=2000,
        maxiter_Elas=2000,
        out_dir=tmp_out_dir,
        earlystop=None,
        maxiter_inner=20,
        tolerance_inner=1e-2,
        output_fields=True,
    )

    sigma_macro = sigAV[:, stress_component_idx]  # shape (n_steps,)

#    # Coarsen strain and damage fields to experimental resolution if specified
#    fft_shape = (grid.nx, grid.ny, grid.nz)
#    if dvc_resolution is not None:
#        efield_coarse_list = []
#        dfield_coarse_list = []
#
#        for step in range(n_steps_dvc):
#            eps_coarse = coarsen_field_3d(efield[step], dvc_resolution)
#            efield_coarse_list.append(eps_coarse)
#            d_coarse = coarsen_field_3d(dfield[step], dvc_resolution)
#            dfield_coarse_list.append(d_coarse)
#        efield_exp = jnp.stack(efield_coarse_list, axis=0) 
#        dfield_exp = jnp.stack(dfield_coarse_list, axis=0)
#    else:
#        efield_exp = efield
#        dfield_exp = dfield
    efield_exp = efield
    dfield_exp = dfield

    return {
        "sigma_macro": sigma_macro,
        "strain_field": efield_exp,
        "damage_field": dfield_exp, 
    }



def newton_raphson_inverse(
    sigma_target,
    strain_target,
    u0,
    grid,
    matID,
    lmbda_grid,
    mu_grid,
    strain_loading,
    stress_component_idx=0,
    dvc_resolution=None,
    dvc_steps=None,
    weight_sigma=1.0,
    weight_strain=0.5,
    maxiter=15,
    tol=1e-4,
    reg=1e-6,
    damp_init=1.0,
):
    """
    Newton-Raphson loop to solve r(u) = 0 where r combines:
    - Measured stress residual
    - Full-field strain residual

    Parameters:
    -----------
    sigma_target : array
        Target stress component at each step, shape (n_steps,)
    strain_target : array
        Target full strain, shape (n_steps, 6) or flattened
    u0 : array
        Initial guess for [s_gc_matrix, s_gc_particle, s_lc_matrix, s_lc_particle]
    stress_component_idx : int
        Which stress component to use (0-5 in Voigt notation)
    dvc_resolution : tuple, optional
        Target experimental resolution (nx_exp, ny_exp, nz_exp)
    dvc_steps : int, optional
        Strain loading step at which full-field strain/damage were acquired
    weight_sigma : float
        Weight for stress in combined residual
    weight_strain : float
        Weight for strain in combined residual
    maxiter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    reg : float
        Regularization parameter
    damp_init : float
        Initial damping for line search
    """

    dtype = jnp.float64

    u0 = jnp.asarray(u0, dtype=dtype)
    sigma_target = jnp.asarray(sigma_target, dtype=dtype)
    strain_target = jnp.asarray(strain_target, dtype=dtype)
    strain_target_flat = strain_target.reshape(strain_target.shape[0], -1).flatten()

    sigma_target_norm = jnp.linalg.norm(sigma_target)
    strain_target_norm = jnp.linalg.norm(strain_target_flat)
    sigma_scale = jnp.maximum(sigma_target_norm, 1e-10)
    strain_scale = jnp.maximum(strain_target_norm, 1e-10)

    # Weighted combined residual function
    def residual_fn(p):
        response = forward_full_response(
            p,
            grid,
            matID,
            lmbda_grid,
            mu_grid,
            strain_loading,
            stress_component_idx=stress_component_idx,
            dvc_resolution=dvc_resolution,
            dvc_steps=dvc_steps,
            dtype=dtype,
        )

        sigma_macro = response["sigma_macro"]
        efield = response["strain_field"]

        # Stress residual
        r_sigma = (sigma_macro - sigma_target) / sigma_scale

        # Strain residual (flatten all components and spatial dimensions)
        r_strain = (efield.reshape(efield.shape[0], -1).flatten() - strain_target_flat) / strain_scale

        return jnp.concatenate([
                   r_sigma * weight_sigma,
                   r_strain * weight_strain
               ])

    u, history = newton_raphson(
        residual_fn,
        u0=u0,
        maxiter=20,
        tol=1e-4,
        reg_init=1e-8,
        damp_init=1.0,
        verbose=1,
    )
    return u, history


def pack_true_to_u(gc_matrix, gc_particle, lc_matrix, lc_particle):
    """Pack true fracture parameters into optimization vector."""
    s_gc_matrix = gc_to_s(gc_matrix)
    s_gc_particle = gc_to_s(gc_particle)
    s_lc_matrix = lc_to_s(lc_matrix)
    s_lc_particle = lc_to_s(lc_particle)
    return jnp.asarray([s_gc_matrix, s_gc_particle, s_lc_matrix, s_lc_particle], dtype=jnp.float64)


def unpack_u_to_physical(u):
    """Unpack optimization vector to physical fracture parameters."""
    s_gc_matrix, s_gc_particle, s_lc_matrix, s_lc_particle = u
    gc_matrix = s_to_gc(s_gc_matrix)
    gc_particle = s_to_gc(s_gc_particle)
    lc_matrix = s_to_lc(s_lc_matrix)
    lc_particle = s_to_lc(s_lc_particle)
    return gc_matrix, gc_particle, lc_matrix, lc_particle


########################################
########################################
########################################
########################################
########################################
########################################

if __name__ == "__main__":
    print("=" * 70)
    print("Inverse Identification for Phase-Field Fracture Model")
    print("Identifying: gc (fracture toughness) and lc (characteristic length)")
    print("Data: Measured macroscopic stress component + Full-field strain (DVC-like)")
    print("=" * 70)

    # RVE geometry
    box_size = [0.4, 0.4, 0.01]
    spacing = [0.01, 0.01, 0.01]
    n_particles = 8
    radius_range = [0.05, 0.08]
    dvc_resolution = [int(L/h/5) for L,h in zip(box_size, spacing)]
    dvc_resolution = [max(1, dim) for dim in dvc_resolution]

    print("\n1. Building RVE...")
    grid, matID = build_rve(box_size, spacing, n_particles, radius_range, seed=42)
    print(f"   Grid: {grid.nx} x {grid.ny} x {grid.nz} voxels")

    # Known elastic parameters (NMC particle + LPSC matrix)
    E_particle = 175e3  # MPa
    nu_particle = 0.28
    lmbda_particle, mu_particle = eng2lame(E_particle, nu_particle)

    E_matrix = 22e3  # MPa
    nu_matrix = 0.37
    lmbda_matrix, mu_matrix = eng2lame(E_matrix, nu_matrix)

    # Create elastic grids
    lmbda_grid, mu_grid, _, _ = init_material(
        matID,
        lmbda_list=[lmbda_matrix, lmbda_particle],
        mu_list=[mu_matrix, mu_particle],
        gc_list=[0.0, 0.0],
        lc_list=[1.0, 1.0],
        dtype=jnp.float64,
    )

    print(f"   E_particle = {E_particle} MPa, nu_particle = {nu_particle}")
    print(f"   E_matrix = {E_matrix} MPa, nu_matrix = {nu_matrix}")

    # True fracture parameters
    gc_true_matrix = 2.8e-3  # N/mm
    gc_true_particle = 2.5e-3  # N/mm
    lc_true_matrix = 0.01  # mm
    lc_true_particle = 0.01  # mm

    u_true = pack_true_to_u(
        gc_true_matrix, gc_true_particle, lc_true_matrix, lc_true_particle
    )

    print(f"\n2. True fracture parameters:")
    print(f"   gc_matrix = {float(gc_true_matrix):.6e} N/mm")
    print(f"   gc_particle = {float(gc_true_particle):.6e} N/mm")
    print(f"   lc_matrix = {float(lc_true_matrix):.6f} mm")
    print(f"   lc_particle = {float(lc_true_particle):.6f} mm")

    # Uniaxial tension: strain in x-direction (e11)
    Emean, deps = 6.2e-4, 5e-5
    eps_steps = np.arange(0.0, Emean, deps) + deps
    nsteps = len(eps_steps)
    dvc_steps = np.arange(10, nsteps-1, 20)
    nsteps_dvc = len(dvc_steps)

    strain_loading = [
        jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        for eps_xx in eps_steps
    ]

    print(f"\n3. Generating synthetic reference data (uniaxial tension)...")
    print(f"   Loading: {nsteps} strain steps from {deps} to {Emean}")
    print(f"   Running forward solve with true parameters...")

    t0 = time.time()
    response_true = forward_full_response(
        u_true,
        grid,
        matID,
        lmbda_grid,
        mu_grid,
        strain_loading,
        stress_component_idx=0,  # σ11 (tensile stress)
        dvc_resolution=dvc_resolution,
        dvc_steps=dvc_steps,
    )
    t_fwd = time.time() - t0
    print(f"   Forward solve took {t_fwd:.2f} s")

    sigma_target = jax.lax.stop_gradient(response_true["sigma_macro"])  # shape (nsteps,)
    strain_target = jax.lax.stop_gradient(response_true["strain_field"])  # shape (nsteps_dvc, 6)

    print(f"   Target stress shape: {sigma_target.shape}")
    print(f"   Target strain shape: {strain_target.shape}")
    print(f"   Stress range: [{sigma_target.min():.3e}, {sigma_target.max():.3e}] MPa")
    print(f"   Strain range: [{strain_target.min():.3e}, {strain_target.max():.3e}]")

    plt.figure()
    plt.plot(eps_steps, sigma_target, "-*", label="Stress-strain curve")
    plt.plot(
        eps_steps[dvc_steps],
        sigma_target[dvc_steps],
        "o",
        label="DVC steps",
    )
    plt.xlabel(r"E11")
    plt.ylabel(r"S11")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Initial guess (perturbed from true values)
    gc_init_matrix = 5.0e-3
    gc_init_particle = 5.0e-3
    lc_init_matrix = 0.015
    lc_init_particle = 0.015

    u0 = pack_true_to_u(
        gc_init_matrix, gc_init_particle, lc_init_matrix, lc_init_particle
    )

    print(f"\n4. Initial guess for parameters:")
    print(f"   gc_matrix = {float(gc_init_matrix):.6e} N/mm")
    print(f"   gc_particle = {float(gc_init_particle):.6e} N/mm")
    print(f"   lc_matrix = {float(lc_init_matrix):.6f} mm")
    print(f"   lc_particle = {float(lc_init_particle):.6f} mm")

    # Run inverse solve
    print(f"\n5. Starting Newton-Raphson inverse solve...")
    print(f"   Data: Measured stress component (σ11) + Full-field strain")
    print(f"   Weights: stress={1.0}, strain={0.5}")
    print(f"   Evaluating on {nsteps_dvc} strain fields from DVC")

    t0 = time.time()
    u_opt, history = newton_raphson_inverse(
        sigma_target,
        strain_target,
        u0,
        grid,
        matID,
        lmbda_grid,
        mu_grid,
        strain_loading,
        stress_component_idx=0,
        dvc_resolution=dvc_resolution,
        dvc_steps=dvc_steps,
        weight_sigma=1.0,
        weight_strain=0.5,
        maxiter=10,
        tol=1e-4,
        reg=1e-6,
        damp_init=1.0,
    )
    t_inv = time.time() - t0

    print(f"\nInverse solve completed in {t_inv:.2f} s")

    # Unpack results
    gc_m_opt, gc_p_opt, lc_m_opt, lc_p_opt = unpack_u_to_physical(u_opt)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("\nTrue parameters:")
    print(f"  gc_matrix = {float(gc_true_matrix):.6e} N/mm")
    print(f"  gc_particle = {float(gc_true_particle):.6e} N/mm")
    print(f"  lc_matrix = {float(lc_true_matrix):.6f} mm")
    print(f"  lc_particle = {float(lc_true_particle):.6f} mm")

    print("\nRecovered parameters:")
    print(f"  gc_matrix = {float(gc_m_opt):.6e} N/mm")
    print(f"  gc_particle = {float(gc_p_opt):.6e} N/mm")
    print(f"  lc_matrix = {float(lc_m_opt):.6f} mm")
    print(f"  lc_particle = {float(lc_p_opt):.6f} mm")

    # Relative errors
    err_gc_m = abs(float(gc_m_opt) - float(gc_true_matrix)) / float(gc_true_matrix) * 100
    err_gc_p = abs(float(gc_p_opt) - float(gc_true_particle)) / float(gc_true_particle) * 100
    err_lc_m = abs(float(lc_m_opt) - float(lc_true_matrix)) / float(lc_true_matrix) * 100
    err_lc_p = abs(float(lc_p_opt) - float(lc_true_particle)) / float(lc_true_particle) * 100

    print("\nRelative errors:")
    print(f"  gc_matrix:     {err_gc_m:.2f}%")
    print(f"  gc_particle:   {err_gc_p:.2f}%")
    print(f"  lc_matrix:     {err_lc_m:.2f}%")
    print(f"  lc_particle:   {err_lc_p:.2f}%")

    print(f"\nResidual norm history: {history['res_norm']}")

    # Visualization
    print("\n6. Generating convergence plots...")

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
    })

    if len(history["u"]) > 0:
        u_hist = np.asarray(history["u"])
        iters = np.arange(u_hist.shape[0])

        gc_m_hist = np.array([s_to_gc(jnp.asarray(s)).item() for s in u_hist[:, 0]])
        gc_p_hist = np.array([s_to_gc(jnp.asarray(s)).item() for s in u_hist[:, 1]])
        lc_m_hist = np.array([s_to_lc(jnp.asarray(s)).item() for s in u_hist[:, 2]])
        lc_p_hist = np.array([s_to_lc(jnp.asarray(s)).item() for s in u_hist[:, 3]])

        gc_m_true = float(gc_true_matrix)
        gc_p_true = float(gc_true_particle)
        lc_m_true = float(lc_true_matrix)
        lc_p_true = float(lc_true_particle)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # gc_matrix
        ax = axs[0, 0]
        ax.plot(iters, gc_m_hist * 1e3, "o-", linewidth=2, markersize=6, label="Iteration")
        ax.hlines(gc_m_true * 1e3, iters[0], iters[-1], colors="red", linestyles="--", linewidth=2, label="True")
        ax.set_xlabel("Newton iteration")
        ax.set_ylabel(r"$g_c$ matrix (N/mm)")
        ax.set_title(r"Fracture toughness: $g_c$ (matrix)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # gc_particle
        ax = axs[0, 1]
        ax.plot(iters, gc_p_hist * 1e3, "s-", linewidth=2, markersize=6, label="Iteration")
        ax.hlines(gc_p_true * 1e3, iters[0], iters[-1], colors="red", linestyles="--", linewidth=2, label="True")
        ax.set_xlabel("Newton iteration")
        ax.set_ylabel(r"$g_c$ particle (N/mm)")
        ax.set_title(r"Fracture toughness: $g_c$ (particle)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # lc_matrix
        ax = axs[1, 0]
        ax.plot(iters, lc_m_hist, "o-", linewidth=2, markersize=6, label="Iteration")
        ax.hlines(lc_m_true, iters[0], iters[-1], colors="red", linestyles="--", linewidth=2, label="True")
        ax.set_xlabel("Newton iteration")
        ax.set_ylabel(r"$l_c$ matrix (mm)")
        ax.set_title(r"Characteristic length: $l_c$ (matrix)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # lc_particle
        ax = axs[1, 1]
        ax.plot(iters, lc_p_hist, "s-", linewidth=2, markersize=6, label="Iteration")
        ax.hlines(lc_p_true, iters[0], iters[-1], colors="red", linestyles="--", linewidth=2, label="True")
        ax.set_xlabel("Newton iteration")
        ax.set_ylabel(r"$l_c$ particle (mm)")
        ax.set_title(r"Characteristic length: $l_c$ (particle)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("results/matiden", exist_ok=True)
        plt.savefig("results/matiden/inv_pfm_convergence.png", dpi=300, bbox_inches="tight")
        print("   Saved: results/matiden/inv_pfm_convergence.png")
        plt.show()

        # Residual norm history
        plt.figure(figsize=(8, 5))
        plt.semilogy(np.arange(len(history["res_norm"])), history["res_norm"], "o-", linewidth=2, markersize=8)
        plt.xlabel("Newton iteration")
        plt.ylabel("Residual norm")
        plt.title("Convergence of inverse solve (Phase-Field Fracture)\nData: Measured stress + Full-field strain")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/matiden/inv_pfm_residual.png", dpi=300, bbox_inches="tight")
        print("   Saved: results/matiden/inv_pfm_residual.png")
        plt.show()
    else:
        print("   No history recorded for visualization.")

    print("\n" + "=" * 70)
    print("Inverse identification complete!")
    print("=" * 70)
    print("\nNotes:")
    print("------")
    print("• This example uses measured stress (σ11) + full-field strain as data.")
    print("• Full-field strain can be obtained from DIC (Digital Image Correlation).")
    print("• Damage field comparison requires:")
    print("  1. Image processing to threshold crack geometry from experiments")
    print("  2. Comparison with simulated damage field (d > d_threshold)")
    print("  3. Additional regularization to prefer sharp cracks (high gradient)")
    print("• To include damage field, modify forward_full_response() to:")
    print("  1. Save and read damage field from VTK outputs")
    print("  2. Downsample to match image resolution")
    print("  3. Add damage residual to residual_fn with appropriate weight")

