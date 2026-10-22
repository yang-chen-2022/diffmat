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
    generate_voids_periodic,
    init_material,
)
from diffmat.commons.utilities import eng2lame, newton_raphson
from diffmat.commons.io import save_arrays_to_vti
from diffmat.fracture.solver import solve_fracture_staggered
from diffmat.fracture.solve_one_step import MaterialParams, SolverConfig
from diffmat.fracture.solver_jit import solve_loading_history

from jaxmaterials.common import get_grid_spec

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)

jax.config.update("jax_platform_name", "gpu")


# ---------- Reparameterization utils for gc and lc ----------
gc_min = 1e-4  # N/mm
gc_max = 1e-2  # N/mm
lc_min = 0.001  # mm
lc_max = 0.05   # mm

VOID_ID = 2
VOID_STIFFNESS_RATIO = 1e-9  # void stiffness relative to the matrix


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

        if any(coarse_dim <= 0 or coarse_dim > fine_dim
               for fine_dim, coarse_dim in zip(fine_shape, coarse_shape)):
            raise ValueError(
                f"coarse_shape must be positive and no larger than field.shape; "
                f"got {coarse_shape} for {fine_shape}"
            )
        
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




def plot_microstructure(matID, box_size, save_path=None):
    """Plot a mid-plane slice of the voxelised microstructure (matID)."""
    mid_k = matID.shape[2] // 2
    slice_2d = np.asarray(matID[:, :, mid_k]).T

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        slice_2d,
        origin="lower",
        extent=[0, box_size[0], 0, box_size[1]],
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Initial microstructure (matID, mid-plane slice)")
    fig.colorbar(im, ax=ax, label="Material ID (0=matrix, 1=particle)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   Saved: {save_path}")
    plt.show()


def plot_strain_maps(strain_field, dvc_steps, eps_steps, box_size, save_path=None, step_indices=None):
    """
    Plot mid-plane slices of the full-field strain data at selected DVC steps.

    strain_field: array, shape (n_steps_dvc, 6, nx, ny, nz)
    dvc_steps: indices (into eps_steps) at which strain_field was recorded
    step_indices: which entries of strain_field (0-based, into dvc_steps) to plot;
                  defaults to first, middle, and last available step
    """
    strain_field = np.asarray(strain_field)
    n_steps_dvc = strain_field.shape[0]
    if step_indices is None:
        step_indices = sorted(set([0, n_steps_dvc // 2, n_steps_dvc - 1]))

    component_labels = [r"$\epsilon_{11}$", r"$\epsilon_{22}$", r"$\epsilon_{33}$",
                         r"$\epsilon_{23}$", r"$\epsilon_{13}$", r"$\epsilon_{12}$"]
    mid_k = strain_field.shape[-1] // 2

    fig, axs = plt.subplots(len(step_indices), 6, figsize=(3 * 6, 3 * len(step_indices)), squeeze=False)
    for row, s_idx in enumerate(step_indices):
        e_load = eps_steps[dvc_steps[s_idx]]
        for comp in range(6):
            ax = axs[row, comp]
            slice_2d = strain_field[s_idx, comp, :, :, mid_k].T
            im = ax.imshow(
                slice_2d,
                origin="lower",
                extent=[0, box_size[0], 0, box_size[1]],
                cmap="RdBu_r",
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(component_labels[comp])
            if comp == 0:
                ax.set_ylabel(f"E11={e_load:.2e}\ny (mm)")
            ax.set_xlabel("x (mm)")

    fig.suptitle("Full-field strain data (DVC-like) from forward simulation")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   Saved: {save_path}")
    plt.show()


def build_rve(
    box_size,
    spacing,
    n_particles,
    radius_range,
    seed=42,
    void_fraction=0.0,
    void_diameter_mean=None,
    void_diameter_std=0.0,
):
    """Generate periodic particle RVE (matID: 0 matrix, 1 particle, 2 void in matrix) and voxelize it."""
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

    if void_fraction > 0.0:
        void_mask = generate_voids_periodic(
            grid, matID == 1, void_fraction, void_diameter_mean, void_diameter_std
        )
        matID[void_mask] = VOID_ID

    return grid, matID


def build_solver_config(grid, lmbda_grid, mu_grid):
    """Build the static solver configuration from fixed elastic grids."""
    return SolverConfig(
        grid=grid,
        lmbda0=float(0.5 * (lmbda_grid.max() + lmbda_grid.min())),
        mu0=float(0.5 * (mu_grid.max() + mu_grid.min())),
        k_stab=1e-6,
        maxiter_PF=2000,
        maxiter_Elas=2000,
        maxiter_staggered=30,
        tol_PF=1e-5,
        tol_Elas=1e-3,
        tol_staggered=1e-2,
        AA_depth=4,
        verbose=1,
    )


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
    solver_cfg=None,
):
    """
    Forward model: simulate fracture and return:
    - One measured stress component (e.g., σ11 for tensile test)
    - Full-field strain (6 components at all spatial points)

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
        lmbda_list=[0., 0., 0.],
        mu_list=[0., 0., 0.],   #dummy values for lambda and mu
        gc_list=[gc_matrix, gc_particle, gc_matrix],
        lc_list=[lc_matrix, lc_particle, lc_matrix],
        dtype=dtype,
    )

    n_steps = len(strain_loading)
    if dvc_steps is None:
        dvc_steps = np.arange(n_steps)
    else:
        dvc_steps = np.asarray(dvc_steps, dtype=int)
    if np.any(dvc_steps < 0) or np.any(dvc_steps >= n_steps):
        raise ValueError("dvc_steps must contain valid strain-loading indices")

    material_params = MaterialParams(
        lmbda=lmbda_grid,
        mu=mu_grid,
        gc=gc_grid,
        lc=lc_grid,
    )
    if solver_cfg is None:
        solver_cfg = build_solver_config(grid, lmbda_grid, mu_grid)

    _, sigAV, _, efield, _ = solve_loading_history(
        strain_loading,
        material_params,
        solver_cfg,
    )

    sigma_macro = sigAV[:, stress_component_idx]  # shape (n_steps,)

    efield_exp = efield[dvc_steps]
    if dvc_resolution is not None:
        efield_exp = jnp.stack(
            [coarsen_field_3d(field, dvc_resolution) for field in efield_exp]
        )

    return {
        "sigma_macro": sigma_macro,
        "strain_field": efield_exp,
    }


def forward_macro_response(
    u,
    grid,
    matID,
    lmbda_grid,
    mu_grid,
    strain_loading,
    stress_component_idx=0,
    dtype=jnp.float64,
    solver_cfg=None,
):
    """Simulate only the macroscopic stress response for a given parameter vector u."""
    s_gc_matrix, s_gc_particle, s_lc_matrix, s_lc_particle = u
    gc_matrix = s_to_gc(s_gc_matrix)
    gc_particle = s_to_gc(s_gc_particle)
    lc_matrix = s_to_lc(s_lc_matrix)
    lc_particle = s_to_lc(s_lc_particle)

    _, _, gc_grid, lc_grid = init_material(
        matID,
        lmbda_list=[0., 0., 0.],
        mu_list=[0., 0., 0.],
        gc_list=[gc_matrix, gc_particle, gc_matrix],
        lc_list=[lc_matrix, lc_particle, lc_matrix],
        dtype=dtype,
    )

    material_params = MaterialParams(
        lmbda=lmbda_grid,
        mu=mu_grid,
        gc=gc_grid,
        lc=lc_grid,
    )
    if solver_cfg is None:
        solver_cfg = build_solver_config(grid, lmbda_grid, mu_grid)

    _, sigAV, _, _, _ = solve_loading_history(
        strain_loading,
        material_params,
        solver_cfg,
    )

    return sigAV[:, stress_component_idx]

def newton_raphson_inverse(
    sigma_target,
    strain_target=None,
    u0=None,
    grid=None,
    matID=None,
    lmbda_grid=None,
    mu_grid=None,
    strain_loading=None,
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
    Newton-Raphson loop to solve r(u) = 0.

    If strain_target is None, the residual uses only the macroscopic stress
    component and skips all DVC/full-field strain computations.
    """

    dtype = jnp.float64

    u0 = jnp.asarray(u0, dtype=dtype)
    sigma_target = jnp.asarray(sigma_target, dtype=dtype)
    sigma_target_norm = jnp.linalg.norm(sigma_target)
    sigma_scale = jnp.maximum(sigma_target_norm, 1e-10)
    solver_cfg = build_solver_config(grid, lmbda_grid, mu_grid)

    if strain_target is not None:
        strain_target = jnp.asarray(strain_target, dtype=dtype)
        strain_target_flat = strain_target.reshape(strain_target.shape[0], -1).flatten()
        strain_target_norm = jnp.linalg.norm(strain_target_flat)
        strain_scale = jnp.maximum(strain_target_norm, 1e-10)
    else:
        strain_target_flat = None
        strain_scale = None

    def residual_fn(p):
        if strain_target is None or weight_strain == 0.0:
            sigma_macro = forward_macro_response(
                p,
                grid,
                matID,
                lmbda_grid,
                mu_grid,
                strain_loading,
                stress_component_idx=stress_component_idx,
                dtype=dtype,
                solver_cfg=solver_cfg,
            )
            r_sigma = (sigma_macro - sigma_target) / sigma_scale
            return r_sigma * weight_sigma

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
            solver_cfg=solver_cfg,
        )

        sigma_macro = response["sigma_macro"]
        efield = response["strain_field"]

        r_sigma = (sigma_macro - sigma_target) / sigma_scale
        r_strain = (efield.reshape(efield.shape[0], -1).flatten() - strain_target_flat) / strain_scale

        return jnp.concatenate([
                   r_sigma * weight_sigma,
                   r_strain * weight_strain
               ])

    u, history = newton_raphson(
        residual_fn,
        u0=u0,
        maxiter=maxiter,
        tol=tol,
        reg_init=reg,
        damp_init=damp_init,
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
    box_size = [0.4, 0.4, 0.003]
    spacing = [0.003, 0.003, 0.003]
    n_particles = 8
    radius_range = [0.05, 0.08]
    # Spherical voids in the matrix (void_fraction = 0 disables them); lengths in mm
    void_fraction = 0.02
    void_diameter_mean = 0.04
    void_diameter_std = 0.01
    dvc_resolution = [int(L/h/5) for L,h in zip(box_size, spacing)]
    dvc_resolution = [max(1, dim) for dim in dvc_resolution]

    print("\n1. Building RVE...")
    grid, matID = build_rve(
        box_size, spacing, n_particles, radius_range, seed=42,
        void_fraction=void_fraction,
        void_diameter_mean=void_diameter_mean,
        void_diameter_std=void_diameter_std,
    )
    print(f"   Grid: {grid.nx} x {grid.ny} x {grid.nz} voxels")
    print(f"   Void volume fraction: {float(np.mean(matID == VOID_ID)):.4f}")

    os.makedirs("results/matiden", exist_ok=True)
    plot_microstructure(matID, box_size, save_path="results/matiden/microstructure.png")

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
        lmbda_list=[lmbda_matrix, lmbda_particle, VOID_STIFFNESS_RATIO * lmbda_matrix],
        mu_list=[mu_matrix, mu_particle, VOID_STIFFNESS_RATIO * mu_matrix],
        gc_list=[0.0, 0.0, 0.0],
        lc_list=[1.0, 1.0, 1.0],
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
    Emean, deps = 0.65e-3, 5e-6
    eps_steps = np.arange(0.0, Emean, deps) + deps
    nsteps = len(eps_steps)
    dvc_steps = np.arange(10, nsteps-1, 20)
    nsteps_dvc = len(dvc_steps)

    strain_loading = jnp.stack(
        [
            jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
            for eps_xx in eps_steps
        ],
        axis=0,
    )

    print(f"\n3. Generating synthetic reference data (uniaxial tension)...")
    print(f"   Loading: {nsteps} strain steps from {deps} to {Emean}")
    print(f"   Running forward solve with true parameters...")

    t0 = time.time()
    sigma_target = forward_macro_response(
        u_true,
        grid,
        matID,
        lmbda_grid,
        mu_grid,
        strain_loading,
        stress_component_idx=0,
    )
    t_fwd = time.time() - t0
    print(f"   Forward solve took {t_fwd:.2f} s")

    sigma_target = jax.lax.stop_gradient(sigma_target)  # shape (nsteps,)

    print(f"   Target stress shape: {sigma_target.shape}")
    print(f"   Stress range: [{sigma_target.min():.3e}, {sigma_target.max():.3e}] MPa")

    plt.figure()
    plt.plot(eps_steps, sigma_target, "-*", label="Stress-strain curve")
    plt.xlabel(r"E11")
    plt.ylabel(r"S11")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/matiden/stress_strain_curve.png", dpi=300, bbox_inches="tight")
    print("   Saved: results/matiden/stress_strain_curve.png")
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
    print(f"   Data: Measured stress component (σ11) only")
    print(f"   Weight: stress={1.0}")

    t0 = time.time()
    u_opt, history = newton_raphson_inverse(
        sigma_target=sigma_target,
        u0=u0,
        grid=grid,
        matID=matID,
        lmbda_grid=lmbda_grid,
        mu_grid=mu_grid,
        strain_loading=strain_loading,
        stress_component_idx=0,
        weight_sigma=1.0,
        weight_strain=0.0,
        maxiter=30,
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
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/inv_pfm_convergence.png", dpi=300, bbox_inches="tight")
        print(f"   Saved: {out_dir}/inv_pfm_convergence.png")
        plt.show()

        # Residual norm history
        plt.figure(figsize=(8, 5))
        plt.semilogy(np.arange(len(history["res_norm"])), history["res_norm"], "o-", linewidth=2, markersize=8)
        plt.xlabel("Newton iteration")
        plt.ylabel("Residual norm")
        plt.title("Convergence of inverse solve (Phase-Field Fracture)\nData: Measured stress + Full-field strain")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/inv_pfm_residual.png", dpi=300, bbox_inches="tight")
        print(f"   Saved: {out_dir}/inv_pfm_residual.png")
        plt.show()
    else:
        print("   No history recorded for visualization.")

    print("\n" + "=" * 70)
    print("Inverse identification complete!")
    print("=" * 70)
    print("\nNotes:")
    print("------")
