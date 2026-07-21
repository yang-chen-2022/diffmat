"""
Test case on topology optimisation with differentiable FFT-based solvers.

This test demonstrates a gradient-based topology optimization workflow using
JAX-based FFT solvers (Lippmann-Schwinger). It combines density-based topology 
optimization with automatic differentiation to optimize material layouts for
desired mechanical properties.

Inspired by:
  Mohit Pundir & David S. Kammer, 2025, Computer Methods in Applied Mechanics 
  and Engineering, 435, 117572

Key features:
- Density parameterization of material layout
- Sensitivity filtering and optimality criteria (OC) updates
- Automatic differentiation through the FFT solver
- Real-time visualization of convergence and final topology
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import convolve
import matplotlib.pyplot as plt
import time

jax.config.update("jax_enable_x64", True)

from jaxmaterials.common import GridSpec
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ============================================================================
# Constitutive Relations and Helper Functions
# ============================================================================

def compute_sigma_iso(epsilon, params):
    """
    Compute isotropic stress from strain using Lamé parameters.
    
    Parameters
    ----------
    epsilon : ndarray (6, nx, ny, nz)
        Strain field in Voigt notation
    params : tuple
        (lambda, mu) - Lamé parameters
        
    Returns
    -------
    sigma : ndarray (6, nx, ny, nz)
        Stress field in Voigt notation
    """
    lmbda, mu = params
    tr_epsilon = epsilon[0, ...] + epsilon[1, ...] + epsilon[2, ...]
    sigma = jnp.zeros_like(epsilon)
    sigma = sigma.at[:3].set((lmbda * tr_epsilon)[None, ...] + 2. * mu * epsilon[:3])
    sigma = sigma.at[3:].set(2. * mu * epsilon[3:])
    return sigma


# ============================================================================
# Filtering and Optimization Utilities
# ============================================================================

def periodic_convolve(x, kernel):
    """
    Apply periodic convolution (wrap boundary conditions).
    
    Parameters
    ----------
    x : ndarray
        Field to convolve
    kernel : ndarray
        Convolution kernel
        
    Returns
    -------
    result : ndarray
        Convolved field
    """
    pad_width = [(k // 2, k // 2) for k in kernel.shape]
    x_pad = jnp.pad(x, pad_width, mode="wrap")
    return convolve(x_pad, kernel, mode="valid")


def build_filter_kernel(radii):
    """
    Build a 3D filter kernel for sensitivity filtering.
    
    Parameters
    ----------
    radii : tuple
        (rx, ry, rz) - Filter radii in each direction
        
    Returns
    -------
    kernel : ndarray (2*rx+1, 2*ry+1, 2*rz+1)
        Normalized filter kernel
    """
    rx, ry, rz = radii
    x = np.arange(-rx, rx + 1)
    y = np.arange(-ry, ry + 1)
    z = np.arange(-rz, rz + 1)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    
    kernel = jnp.maximum(0.0, max(radii) - dist)
    return kernel / kernel.sum()


@jax.jit(static_argnames=["ft_type"])
def apply_sensitivity_filter(ft_type, x, dc, dv, kernel):
    """
    Apply sensitivity filter to objective and volume gradients.
    
    Parameters
    ----------
    ft_type : int
        Filter type: 1=sensitivity filter, 2=density-style averaging
    x : ndarray (nx, ny, nz)
        Density field
    dc : ndarray (nx, ny, nz)
        Objective sensitivity (gradient)
    dv : ndarray (nx, ny, nz)
        Volume sensitivity (typically all ones)
    kernel : ndarray
        Precomputed filter kernel
        
    Returns
    -------
    dc_filtered : ndarray
        Filtered objective sensitivity
    dv_filtered : ndarray (optional)
        Filtered volume sensitivity (only if ft_type==2)
    """
    Hs = kernel.sum()
    
    if ft_type == 1:
        # Sensitivity filter: weight by current density
        numerator = periodic_convolve(x * dc, kernel)
        dc_filtered = numerator / Hs / jnp.maximum(1e-3, x)
        return dc_filtered, dv
        
    elif ft_type == 2:
        # Density filter: standard convolution
        dc_filtered = periodic_convolve(dc, kernel) / Hs
        dv_filtered = periodic_convolve(dv, kernel) / Hs
        return dc_filtered, dv_filtered
    
    else:
        raise ValueError("ft_type must be 1 or 2")


def get_initial_density(vf, shape):
    """
    Generate initial density field with circular hole pattern.
    
    Parameters
    ----------
    vf : float
        Target volume fraction
    shape : tuple
        (nx, ny, nz) grid dimensions
        
    Returns
    -------
    rho : ndarray
        Initial density field
    """
    nx, ny, nz = shape
    rho = np.full(shape, vf, dtype=float)
    
    d = nx / 3.0
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    r = np.sqrt((x - nx / 2.0 - 0.5) ** 2 + (y - ny / 2.0 - 0.5) ** 2)
    
    rho[r < d] = vf / 2.0
    return jnp.array(rho)


def oc(rho, dc, dv, ft_type, vf, kernel=None, move=0.2, tol=1e-6):
    """
    Optimality Criteria (OC) update for topology optimization.
    
    Implements the standard OC algorithm with move limits and volume constraint.
    
    Parameters
    ----------
    rho : ndarray
        Current density field
    dc : ndarray
        Objective sensitivity field
    dv : ndarray
        Volume sensitivity field
    ft_type : int
        Filter type (1 or 2)
    vf : float
        Target volume fraction
    kernel : ndarray, optional
        Filter kernel (required if ft_type==2)
    move : float
        Maximum per-iteration density change
    tol : float
        Binary-search tolerance for Lagrange multiplier
        
    Returns
    -------
    rho_new : ndarray
        Updated density field
    change : float
        Maximum absolute design-variable change
    """
    x = rho.copy()
    l1, l2 = 0.0, 1e9
    Hs = kernel.sum() if ft_type == 2 else 0.0
    
    while (l2 - l1) > tol:
        lmid = 0.5 * (l1 + l2)
        
        # OC update: rho_new = rho * sqrt(-dc / (dv * lambda))
        dr = np.abs(-dc / (dv * lmid))
        x_trial = x * np.sqrt(dr)
        
        # Apply move limits
        xnew = np.clip(x_trial, x - move, x + move)
        xnew = np.clip(xnew, 0.0, 1.0)
        
        # Apply filter
        if ft_type == 1:
            rho_candidate = xnew
        elif ft_type == 2:
            rho_candidate = periodic_convolve(xnew, kernel) / Hs
        
        # Check volume constraint
        if rho_candidate.mean() > vf:
            l1 = lmid
        else:
            l2 = lmid
    
    change = np.max(np.abs(xnew - x))
    return jnp.array(rho_candidate), change


# ============================================================================
# Material Parameter Functions
# ============================================================================

def eng2lame(E, nu):
    """Convert Young's modulus and Poisson's ratio to Lamé parameters."""
    lmbda = E * nu / (1. + nu) / (1. - 2. * nu)
    mu = E / 2. / (1. + nu)
    return lmbda, mu


# ============================================================================
# Setup: Domain and Material Properties
# ============================================================================

# Domain dimensions [mm]
Lx, Ly, Lz = 0.5, 0.5, 0.5
nx, ny, nz = 99, 99, 1

grid_spec = GridSpec(nx, ny, nz, Lx, Ly, Lz)
shape = (nx, ny, nz)
dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

# Material parameters [MPa, N/mm^2]
E0 = 1e-1  # Void/soft material
E1 = 1.0   # Solid material
nu = 0.3   # Poisson's ratio
kk = 0.0   # Regularization parameter (optional)

# Topology optimization parameters
vf = 0.3         # Target volume fraction
penalty = 5.0    # Penalization exponent (SIMP)
ft_type = 1      # Filter type: 1=sensitivity, 2=density

# Build filter kernel
kernel = build_filter_kernel((2, 2, 0))


# ============================================================================
# Objective Function and Sensitivity
# ============================================================================

def param(rho):
    """
    Material property interpolation via SIMP model.
    
    E(rho) = E0 + (E1 - E0) * (rho + kk)^penalty
    """
    return E0 + (E1 - E0) * (rho + kk) ** penalty


def fft_solve(lmbda, mu, epsilon_bar):
    """
    Solve linear elastic problem via Lippmann-Schwinger FFT solver.
    
    Parameters
    ----------
    lmbda : ndarray (nx, ny, nz)
        Lamé parameter field
    mu : ndarray (nx, ny, nz)
        Shear modulus field
    epsilon_bar : ndarray (6,)
        Macroscopic strain (Voigt notation)
        
    Returns
    -------
    sigma_macro : ndarray (6,)
        Macroscopic stress (Voigt notation)
    """
    lmbda0 = jax.lax.stop_gradient(0.5 * (jnp.max(lmbda) + jnp.min(lmbda)))
    mu0 = jax.lax.stop_gradient(0.5 * (jnp.max(mu) + jnp.min(mu)))
    
    epsilon, sigma = lippmann_schwinger(
        compute_sigma_iso,
        (lmbda, mu),
        epsilon_bar,
        ref_params={"lambda": lmbda0, "mu": mu0},
        grid_spec=grid_spec,
        verbose=0,
        depth=4,
    )
    
    return jnp.mean(sigma, axis=[1, 2, 3])


def compute_c(lmbda, mu):
    """
    Compute macroscopic compliance (inverse stiffness).
    """
    epsilon_bars = jnp.eye(6)[:2]
    batched_fft_solve = jax.vmap(lambda eps: fft_solve(lmbda, mu, eps))
    Cmacro = batched_fft_solve(epsilon_bars)
    
    # Bulk modulus related: -C_00 - C_11 - C_01 - C_10
    return -(Cmacro[0, 0] + Cmacro[1, 1] + Cmacro[0, 1] + Cmacro[1, 0])


def compute_c_and_dc(rho):
    """
    Compute compliance and sensitivity via automatic differentiation.
    
    Parameters
    ----------
    rho : ndarray
        Density field
        
    Returns
    -------
    c : float
        Compliance value
    dc : ndarray
        Compliance gradient w.r.t. density
    """
    lmbda, mu = eng2lame(param(rho), nu)
    value_grad_fn = jax.value_and_grad(compute_c, argnums=1, has_aux=False)
    c, dc_dmu = value_grad_fn(lmbda, mu)
    
    # Chain rule: dc/drho = dc/dmu * dmu/dparam * dparam/drho
    dc = (
        dc_dmu 
        * (E1 - E0) / 2.0 / nu 
        * penalty 
        * (rho + kk) ** (penalty - 1) 
        * (nx * ny * nz)
    )
    
    return c, dc


# ============================================================================
# Topology Optimization Loop
# ============================================================================

def optimize(max_iter=100):
    """
    Run topology optimization using OC algorithm.
    
    Parameters
    ----------
    max_iter : int
        Maximum number of iterations
        
    Returns
    -------
    rho : ndarray
        Optimized density field
    objective_values : list
        Compliance history
    """
    rho = jnp.array(get_initial_density(vf, shape))
    change, loop = 10.0, 0
    objective_values = []
    
    while change > 0.01 and loop < max_iter:
        loop += 1
        
        # Compute objective and sensitivity
        t0 = time.time()
        c, dc = compute_c_and_dc(rho)
        c.block_until_ready()
        time_compute = time.time() - t0
        
        # Apply filter
        t0 = time.time()
        dv = jnp.ones(shape)
        dc, dv = apply_sensitivity_filter(ft_type, rho, dc, dv, kernel)
        dc.block_until_ready()
        time_filter = time.time() - t0
        
        # Update design variables
        t0 = time.time()
        rho, change = oc(rho, dc, dv, ft_type, vf, kernel=kernel, move=0.2, tol=1e-6)
        rho.block_until_ready()
        time_oc = time.time() - t0
        
        # Logging
        vol_frac = jnp.mean(rho)
        objective_values.append(c)
        status = f"iter {loop:3d} | obj {c:8.4f} | vol {vol_frac:6.3f} | Δρ {change:6.3f}"
        
        print(f"{status}")
        print(f"  └─ compute: {time_compute:.3f}s | filter: {time_filter:.3f}s | OC: {time_oc:.3f}s")
        
        # Visualization every 20 iterations
        if loop % 20 == 0:
            plt.figure(figsize=(5, 4))
            plt.imshow(-np.array(rho[..., nz // 2]), cmap="gray")
            plt.title(status)
            plt.colorbar()
            plt.show()
    
    # Final visualization
    plt.figure(figsize=(5, 4))
    plt.imshow(-np.array(rho[..., nz // 2]), cmap="gray")
    plt.title(f"Final Design | obj {objective_values[-1]:.4f} | vol {jnp.mean(rho):.3f}")
    plt.colorbar()
    plt.show()
    
    return rho, objective_values


# ============================================================================
# Run Optimization
# ============================================================================

print("\n" + "=" * 70)
print("TOPOLOGY OPTIMIZATION WITH DIFFERENTIABLE FFT SOLVER")
print("=" * 70 + "\n")

rho_opt, obj_hist = optimize(max_iter=30)

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(-np.array(obj_hist), "*-", linewidth=2, markersize=6)
plt.xlabel("Iteration")
plt.ylabel("Compliance (negative)")
plt.title("Topology Optimization Convergence")
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================================
# Post-Processing: Visualize Final Solution
# ============================================================================

print("\nPost-processing final solution...")

# Macroscopic loading
epsilon_bar = jnp.array([1., 0., 0., 0., 0., 0.])
lmbda_opt, mu_opt = eng2lame(param(rho_opt), nu)

# Compute strain and stress fields
lmbda0 = 0.5 * (lmbda_opt.max() + lmbda_opt.min())
mu0 = 0.5 * (mu_opt.max() + mu_opt.min())

epsilon, sigma = lippmann_schwinger(
    compute_sigma_iso,
    (lmbda_opt, mu_opt),
    epsilon_bar,
    ref_params={"lambda": lmbda0, "mu": mu0},
    grid_spec=grid_spec,
    verbose=0,
    depth=4,
)

# Visualize topology, strain, and stress
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

ax[0].imshow(-np.array(rho_opt[..., nz // 2]), cmap="gray")
ax[0].set_title("Optimized Topology")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

im1 = ax[1].imshow(np.array(epsilon[0][..., nz // 2]))
ax[1].set_title("Strain (xx)")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
plt.colorbar(im1, ax=ax[1])

im2 = ax[2].imshow(np.array(sigma[0][..., nz // 2]))
ax[2].set_title("Stress (xx)")
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()

print("\nOptimization complete!")
