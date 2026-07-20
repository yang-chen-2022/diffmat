"""
Test case for elastodamage phase-field fracture simulation.

This test demonstrates a phase-field fracture model applied to a composite 
material with reinforcing fibres. It:
1. Sets up a 3D computational grid with embedded fibres
2. Initializes material properties for matrix and fibre materials
3. Performs incremental loading with phase-field damage evolution
4. Visualizes and saves the stress, strain, and damage fields
"""

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp
import pyvista as pv

from diffmat.utilities import grid_spec, voxelise_particles_periodic
from diffmat.solver import elastodamage_phasefield_solve
from diffmat.rvegen import generate_particles_periodic
from diffmat.helper import save_arrays_to_vti

import time

# ============================================================================
# Setup: Grid and RVE Geometry
# ============================================================================

# Create a 3D computational grid
box_size = [2., 2., 2.]
spacing = [0.05, 0.05, 0.05]
grid = grid_spec(box_size[0], box_size[1], box_size[1], spacing[0], spacing[1], spacing[2])

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
print(np.unique(matID))
#matID = matID > 1
print(f'  create matID took {time.time()-t0} s')

# Visualize the material distribution
mesh = pv.ImageData()
mesh.dimensions = np.array(matID.shape) + 1
mesh.spacing = spacing
mesh.origin = (0, 0, 0)
mesh.cell_data["value"] = matID.flatten(order="F")
mesh.plot(volume=True, opacity=0.1)


# ============================================================================
# Material Properties for Phase-Field Fracture Model
# ============================================================================

def init_material(lmbda_list, mu_list, gc_list, lc_list):
    dtype = jnp.float32
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


# Define material properties: [matrix, fibre]
lmbda_list = [10., 100.]      # Lame parameter
mu_list = [8., 80.]          # Shear modulus
gc_list = [2.e-3, 1.e-3]        # Critical energy release rate

lc_particle = jnp.array(0.08)
lc_list = [0.08, lc_particle]        # Characteristic length

lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(lmbda_list, mu_list, gc_list, lc_list)

# ============================================================================
# Loading and Solver Setup
# ============================================================================

# Define monotonic uniaxial loading (strain in x-direction)
Emean = 0.03
nsteps = 10

# Create strain history: ramp from >0 to Emean_xx over nsteps
exx0 = Emean/nsteps/2
Emean_steps = [
    jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0])
    for eps_xx in np.linspace(exx0, Emean, nsteps)
]

# Steps at which to save output fields
save_steps = [0, nsteps - 1]

# Solve the elastodamage phase-field problem
epsMacro, sigMacro, epsfield, sigfield, dfield = elastodamage_phasefield_solve(
    grid,
    lmbda_grid,
    mu_grid,
    gc_grid,
    lc_grid,
    Emean_steps,
    save_steps,
    k_stab=1e-6,
    maxiter_PF=1000,
    maxiter_Elas=500,
)

print(epsMacro[:,0])
print(sigMacro[:,0])

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

# Save results to VTK format for visualization in ParaView or similar tools
for step in save_steps:
    save_arrays_to_vti(
        filename=f"stress_strain_damage_field_{step}.vtk",
        arrays=[epsfield[step], sigfield[step], dfield[step][None, ...]],
        names=["Strain", "Stress", "Damage"],
        spacing=spacing,
        origin=(0, 0, 0),
        stack_components=True,
    )




# ================== #
# Differentiation    #
# ================== #

def loss_fn(lc_particle):
    
    lmbda_list = [10., 100.]      # Lame parameter
    mu_list = [8., 80.]          # Shear modulus
    gc_list = [2.e-3, 1.e-3]        # Critical energy release rate

    lc_list = [0.08, lc_particle]        # Characteristic length
    
    lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(lmbda_list, mu_list, gc_list, lc_list)
    
    epsMacro, sigMacro, epsfield, sigfield, dfield = elastodamage_phasefield_solve(
        grid,
        lmbda_grid,
        mu_grid,
        gc_grid,
        lc_grid,
        Emean_steps,
        save_steps,
        k_stab=1e-6,
        maxiter_PF=1000,
        maxiter_Elas=500,
    )
    
    epsMacro0 = jnp.array([0.0015, 0.00466667, 0.00783334, 0.01100001, 0.01416667, 0.01733333,
                           0.02050005, 0.02366663, 0.02683346, 0.03000061])
    sigMacro0 = jnp.array([0.04443982, 0.13747472, 0.21981922, 0.28068838, 0.31564325, 0.32615548,
                           0.31734097, 0.29530105, 0.26511085, 0.22972433])

    loss = jnp.mean( (((epsMacro0 - epsMacro[:,icomp])/max(epsMacro0))**2 + 
                    ((sigMacro0 - sigMacro[:,icomp])/max(sigMacro0))**2)**0.5 )
    return loss

grad_fn = jax.jacrev(loss_fn, argnums=0, has_aux=False)    
dl_dlc = grad_fn(0.08)
print(dl_dlc)

value_grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=False)    
l, dl_dlc = value_grad_fn(0.08)
print(dl_dlc)
print(l)



