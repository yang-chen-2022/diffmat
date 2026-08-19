"""
Test case for elastodamage phase-field fracture simulation.
"""

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp

from diffmat.fracture.solver import elastodamage_phasefield_solve
from diffmat.fracture.rvegen import generate_particles_periodic, voxelise_particles_periodic
from diffmat.commons.io import save_arrays_to_vti

from jaxmaterials.common import get_grid_spec

import time
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')


# Output directories
out_dir = f"results/fracture/"
os.makedirs(out_dir, exist_ok=True)


# ============================================================================
# Setup: Grid and RVE Geometry
# ============================================================================

# Create a 3D computational grid
box_size = [2., 2., 2.] #physical length, mm
spacing = [0.08, 0.08, 0.08]
grid = get_grid_spec(
        box_size[0], 
        box_size[1], 
        box_size[2], 
        dx=spacing[0], 
        dy=spacing[1], 
        dz=spacing[2],
        )

# Random particle distribution
np.random.seed(42)
n_particles = 20
radius_range = [0.1, 0.3]
t0 = time.time()
positions, radii = generate_particles_periodic(
    n_particles,
    box_size,
    radius_range,
)
print(f'  gerenate_random_particles took {time.time()-t0} s')

# Material Association: Assign voxels to particle or matrix
t0 = time.time()
matID = voxelise_particles_periodic(
    grid,
    positions,
    radii,
)
print(f'  creating matID took {time.time()-t0} s')

save_arrays_to_vti(
    filename=f"{out_dir}/matID.vtk",
    arrays=[matID[None, ...]],
    names=["matID"],
    spacing=spacing,
    origin=(0, 0, 0),
    stack_components=True,
)


# ============================================================================
# Material Properties for Phase-Field Fracture Model
# ============================================================================

def init_material(lmbda_list, mu_list, gc_list, lc_list,dtype=jnp.float64):
    lmbda_grid = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    mu_grid = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    gc_grid = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    lc_grid = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    
    num_mats = len(lmbda_list)
    matids = np.unique(matID)
    
    for i in range(num_mats):
        lmbda_grid = lmbda_grid.at[matID == matids[i]].set(lmbda_list[i])
        mu_grid = mu_grid.at[matID == matids[i]].set(mu_list[i])
        gc_grid = gc_grid.at[matID == matids[i]].set(gc_list[i])
        lc_grid = lc_grid.at[matID == matids[i]].set(lc_list[i])
    
    return lmbda_grid, mu_grid, gc_grid, lc_grid


# Define material properties: [matrix, inclusion]
lmbda_list = [10., 100.]    # Lame parameter
mu_list = [8., 80.]         # Shear modulus
gc_list = [2.e-3, 2.e-3]    # Critical energy release rate
lc_list = [0.08, 0.08]      # Characteristic length

lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(lmbda_list, mu_list, gc_list, lc_list, jnp.float64)

# ============================================================================
# Loading and Solver Setup
# ============================================================================

# Define monotonic uniaxial loading (strain in x-direction)
Emean = 0.02
nsteps = 100

# Create strain history: ramp from >0 to Emean_xx over nsteps
exx0 = Emean/nsteps/2
Emean_steps = [
    jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0])
    for eps_xx in np.linspace(exx0, Emean, nsteps)
]

# Steps at which to save output fields
save_steps = np.arange(0, nsteps-1, 10)
if nsteps-1 not in save_steps:
    save_steps = np.append(save_steps, nsteps-1)


# Solve the elastodamage phase-field problem
t_start = time.time()
epsMacro, sigMacro = elastodamage_phasefield_solve(
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
)
print(f"TOTAL TIME FOR PFM SOLVE: {(time.time()-t_start):.3f} s")

# ============================================================================
# Results Visualization and Output
# ============================================================================

# Plot stress-strain curve
icomp = 0  # x-component
plt.figure()
plt.plot(epsMacro[:, icomp], sigMacro[:, icomp], "-*", label="Full history")
plt.plot(
    epsMacro[save_steps, icomp],
    sigMacro[save_steps, icomp],
    "o",
    label="Saved steps",
)
plt.xlabel("Strain (xx-component)")
plt.ylabel("Stress (xx-component)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#
filename = f"{out_dir}/macro_curve.txt"
data = np.genfromtxt(filename, names=True)
data = {name: data[name] for name in data.dtype.names}


plt.figure()
plt.plot(data["e11"], data["s11"], "-*", label="Stress-strain curve")
plt.plot(
    data["e11"][save_steps],
    data["s11"][save_steps],
    "o",
    label="Saved steps",
)
plt.xlabel(r"Strain ($\varepsilon_{11}$)")
plt.ylabel(r"Stress ($\sigma_{11}$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


