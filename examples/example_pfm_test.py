"""
Test case for phase-field fracture simulation.
"""

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp

from diffmat.fracture.solver import solve_fracture_staggered
from diffmat.fracture.rvegen import generate_particles_periodic, voxelise_particles_periodic, init_material
from diffmat.commons.io import save_arrays_to_vti
from diffmat.commons.utilities import eng2lame

from jaxmaterials.common import get_grid_spec

import time
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')



load = "tension"

# Output directories
if load=="tension":
    out_dir = f"results/fracture/test/SENT"
elif load=="shear":
    out_dir = f"results/fracture/test/SENS"

os.makedirs(out_dir, exist_ok=True)

# ============================================================================
# Setup: Grid and RVE Geometry
# ============================================================================

# Create a 2D computational grid
spacing = [0.01, 0.01, 0.01]
if load=="tension":
    box_size = [1.05, 1.0, spacing[2]] #physical length, mm
elif load=="shear":
    box_size = [1.0, 1.0, spacing[2]] #physical length, mm
grid = get_grid_spec(
        box_size[0], 
        box_size[1], 
        box_size[2], 
        dx=spacing[0], 
        dy=spacing[1], 
        dz=spacing[2],
        )

# Notch geometry
thickness = 1
length = grid.ny * 0.5

x = np.arange(0, grid.nx)
y = np.arange(0, grid.ny)
z = np.arange(0, grid.nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

matID = np.ones((grid.nx, grid.ny, grid.nz), dtype=np.uint8)
matID[(X <= length) & (np.abs(Y-grid.ny/2)<=thickness/2)] = 0

# Void margin boder
if load=="tension":
    matID[(X>grid.ny)] = 0

# save
save_arrays_to_vti(
    filename=f"{out_dir}/matID.vtk",
    arrays=[matID[None, ...]],
    names=["matID"],
    spacing=spacing,
    origin=(0, 0, 0),
    stack_components=True,
)

# Define material properties
lmbda_list = [0.0, 121.15]    # Lame parameter, GPa, kN/mm2
mu_list = [0.0, 80.77]        # Shear modulus, GPa, kN/mm2
gc_list = [2.7e-3, 2.7e-3]     # Critical energy release rate, kN/mm
lc_list = [0.015, 0.015]       # Characteristic length, mm

lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(
        matID,
        lmbda_list, 
        mu_list, 
        gc_list, 
        lc_list, 
        jnp.float64,
        )

# Define monotonic uniaxial loading (strain in x-direction)

if load=="tension":
    eps1, deps1 = 0.005, 5e-5
    eps2, deps2 = 0.008, 5e-5
    idx = 1
elif load=="shear":
    eps1, deps1 = 0.008, 5e-5
    eps2, deps2 = 0.02, 5e-5
    idx = 3

eps_steps = np.concatenate((np.arange(0, eps1, deps1)+deps1, 
                                np.arange(eps1, eps2, deps2)+deps2),
                            )
Emean_steps = [
    jnp.array([0.0] * idx + [eps] + [0.0] * (5 - idx))
    for eps in eps_steps
]

# Steps at which to save output fields
nsteps = eps_steps.size
save_steps = np.arange(0, nsteps-1, 100)
if nsteps-1 not in save_steps:
    save_steps = np.append(save_steps, nsteps-1)


# Solve the phase-field problem
t_start = time.time()
epsMacro, sigMacro = solve_fracture_staggered(
    grid,
    lmbda_grid,
    mu_grid,
    gc_grid,
    lc_grid,
    Emean_steps,
    save_steps,
    k_stab=1e-6,
    maxiter_PF=2000,
    maxiter_Elas=2000,
    out_dir=out_dir,
    earlystop=0.2,
    maxiter_inner=30,
    tolerance_inner=1e-1,
    load_reduction_factor=None,
)
print(f"TOTAL TIME FOR PFM SOLVE: {(time.time()-t_start):.3f} s")

# ============================================================================
# Results Visualization and Output
# ============================================================================

# Plot stress-strain curve
filename = f"{out_dir}/macro_curve.txt"
data = np.genfromtxt(filename, names=True)
data = {name: data[name] for name in data.dtype.names}

save_steps = (data["step"][data["vtk"]==1]).astype(int)

if load=="tension":
    ij = "22"
elif load=="shear":
    ij = "12"

plt.figure()
plt.plot(data[f"e{ij}"], data[f"s{ij}"], "-*", label="Stress-strain curve")
plt.plot(
    data[f"e{ij}"][save_steps],
    data[f"s{ij}"][save_steps],
    "o",
    label="Saved steps",
)
plt.xlabel(r"Strain ($\varepsilon_{ij}$)")
plt.ylabel(r"Stress ($\sigma_{ij}$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



