"""
Test case on topology optimisation with differentiable FFT-based solvers.

This test demonstrates a gradient-based topology optimization workflow using
JAX-based FFT solvers (Lippmann-Schwinger). It combines density-based topology 
optimization with automatic differentiation to optimize material layouts for
desired mechanical properties.

Inspired by:
  Mohit Pundir & David S. Kammer, 2025, Computer Methods in Applied Mechanics 
  and Engineering, 435, 117572

"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from jaxmaterials.common import GridSpec

from diffmat.commons.io import save_arrays_to_vti
from diffmat.topology.filtering import build_filter_kernel, apply_sensitivity_filter  
from diffmat.topology.oc import oc
from diffmat.topology.solver import compute_c
from diffmat.topology.initialiser import get_initial_density

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')


# ============================================================================
# Topology Optimization Loop
# ============================================================================

def optimize(
        mat,
        grid_spec,
        max_iter=100, 
        prefix="to_",
        ):

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

    # Output directories
    out_dir = f"results/topology/to_vf{vf}"
    os.makedirs(out_dir, exist_ok=True)
    vtk_dir = os.path.join(out_dir, "vtk_seq")
    os.makedirs(vtk_dir, exist_ok=True)

    # Initial density field
    rho = get_initial_density(vf, shape)

    save_arrays_to_vti(
            filename=f"{vtk_dir}/{prefix}000.vtk",
            arrays=[rho[None, ...]],
            names=["density"],
            spacing=shape,
            origin=(0,0,0),
            )
    open(os.path.join(out_dir, "convergence.txt"), "w").close()

    change, loop = 10.0, 0
    objective_values = []
    
    # Pre-create and JIT the value_and_grad function bound to mat and grid_spec.
    # Doing this once avoids repeated retracing and compilation inside the loop,
    # which was causing increasing host memory usage.
    #
    # We bind mat and grid_spec so the jitted function only takes rho as input.
    value_and_grad_fn = jax.jit(
        jax.value_and_grad(lambda rho: compute_c(rho, mat, grid_spec), argnums=0, has_aux=False)
    )

    while change > 0.01 and loop < max_iter:
        loop += 1

        # Ensure rho is a JAX array before calling the jitted function
        rho = jnp.asarray(rho)    

        # Compute objective and sensitivity (single pre-jitted call)
        t0 = time.time()
        c, dc = value_and_grad_fn(rho)
        c.block_until_ready()
        time_compute = time.time() - t0

        # Move computed arrays to host in one explicit call to avoid many small host copies
        c_host, dc_host = jax.device_get((c, dc))
        c = np.asarray(c_host)
        dc = np.asarray(dc_host)

        # Convert rho to numpy for downstream (filter / OC / IO)
        rho = np.asarray(rho)

        # Apply filter
        t0 = time.time()
        dv = np.full(shape, 1.0 / rho.size, dtype=rho.dtype)
        dc, dv = apply_sensitivity_filter(
                ft_type, 
                rho, 
                dc, 
                dv, 
                kernel,
                )
        time_filter = time.time() - t0
        
        # Update design variables
        t0 = time.time()
        rho, change = oc( 
                rho, 
                dc, 
                dv, 
                ft_type, 
                vf, 
                kernel=kernel, 
                move=0.5, 
                tol=1e-6, 
                )
        time_oc = time.time() - t0
        
        # Logging
        vol_frac = float(np.mean(rho))
        objective_values.append(float(c))
        status = f"iter {loop:3d} | obj {c:8.4f} | vol {vol_frac:6.3f} | Δρ {change:6.3f}"
        
        with open(f"results/topology/to_vf{vf}/convergence.txt", "a") as f:
            f.write(f"{-float(c)}\n")

        print(f"{status}")
        print(f"  └─ compute: {time_compute:.3f}s | filter: {time_filter:.3f}s | OC: {time_oc:.3f}s")
        
        # Visualization every 20 iterations
        if (loop % 10 == 0):
            #plt.figure(figsize=(5, 4))
            #plt.imshow(-np.array(rho[..., nz // 2]), cmap="gray")
            #plt.title(status)
            #plt.colorbar()
            #plt.show()
            #plt.close()
            # Save
            save_arrays_to_vti(
                filename=f"{vtk_dir}/{prefix}{loop:03d}.vtk",
                arrays=[rho[None, ...]],
                names=["density"],
                spacing=shape,
                origin=(0,0,0),
            )
    
    ## Final visualization
    #plt.figure(figsize=(5, 4))
    #plt.imshow(-rho[..., nz // 2], cmap="gray")
    #plt.title(f"Final Design | obj {objective_values[-1]:.4f} | vol {vol_frac:.3f}")
    #plt.colorbar()
    #plt.show()

    # Save
    save_arrays_to_vti(
        filename=f"{vtk_dir}/{prefix}{loop:03d}.vtk",
        arrays=[rho[None, ...]],
        names=["density"],
            spacing=shape,
        origin=(0,0,0),
    )
    
    return rho, objective_values


# ============================================================================
# Setup: Domain and Material Properties
# ============================================================================

# Domain dimensions [mm]
Lx, Ly, Lz = 0.5, 0.5, 0.5
nx, ny, nz = 99, 99, 99

grid_spec = GridSpec(Lx, Ly, Lz, nx, ny, nz)
shape = (nx, ny, nz)
dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

# Material parameters [MPa, N/mm^2]
mat = {
    'E0': 1e-6,  # Void/soft material
    'E1': 1.0 ,  # Solid material
    'nu': 0.3 ,  # Poisson's ratio
    'kk': 0.0 ,  # Regularization parameter (optional)
    'penalty': 5.0,    # Penalization exponent (SIMP)
    }

# Topology optimization parameters
vf = 0.4        # Target volume fraction
ft_type = 1      # Filter type: 1=sensitivity, 2=density

# Build filter kernel
kernel = build_filter_kernel((2, 2, 2))


# ============================================================================
# Run Optimization
# ============================================================================

print("\n" + "=" * 70)
print("TOPOLOGY OPTIMIZATION WITH DIFFERENTIABLE FFT SOLVER")
print("=" * 70 + "\n")

t_start = time.time()

rho_opt, obj_hist = optimize(
        mat=mat,
        grid_spec=grid_spec,
        max_iter=100,
        prefix=f"to_vf{vf}_",
        )

# save obj history
np.savetxt(f"results/topology/to_vf{vf}/to_vf{vf}_convergence.txt", -np.array(obj_hist))

# Plot convergence
ftsize = 14
plt.figure(figsize=(8, 5))
plt.plot(-np.array(obj_hist), "*-", linewidth=2, markersize=6)
plt.xlabel("Iteration", fontsize=ftsize)
plt.ylabel("Builk modulus", fontsize=ftsize)
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
#plt.title("Topology Optimization Convergence")
plt.grid(True, alpha=0.3)
#plt.savefig(
#    f"to_vf{vf}_convergence.png",
#    dpi=300,
#    bbox_inches="tight"
#)
plt.show()

print("\nOptimization complete!")

print(f"\n---------- TOTAL TIME CONSUMED: {time.time()-t_start} s")

