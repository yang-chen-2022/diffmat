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
from jax import numpy as jnp
import pyvista as pv

from diffmat.solver import elastodamage_phasefield_solve
from diffmat.utils_compvoxel import fibre, grid_spec, fibre_association
from diffmat.helper import save_arrays_to_vti


# ============================================================================
# Setup: Grid and Fibre Geometry
# ============================================================================

# Create a 3D computational grid
# Parameters: domain_size, nx, ny, nz, dx, dy, dz
grid = grid_spec(1.0, 2, 3, 0.03, 0.03, 0.03)

# Define a single reinforcing fibre
# Parameters: reference points, fibre direction vectors, fibre radii
fibres = fibre(
    [[0.2, 0.5, 0.5]],      # Reference point on fibre
    [[0.0, 0.0, 1.0]],      # Fibre direction (z-axis)
    [0.3]                   # Fibre radius
)

# ============================================================================
# Material Association: Assign voxels to fibre or matrix
# ============================================================================

centers = np.array(grid.build_centers())
x0 = np.array(fibres.x0)
v = np.array(fibres.v)
r = np.array(fibres.r)
h_vec = np.array([grid.dx, grid.dy, grid.dz])

# Associate each voxel with the nearest fibre (or matrix if outside all fibres)
all_in, all_out, composite, idx_closest = fibre_association(
    centers, h_vec, x0, v, r
)
matID = np.where(all_out, 0, idx_closest + 1)
matID = matID.reshape((grid.nx, grid.ny, grid.nz))

# Visualize the material distribution
mesh = pv.ImageData()
mesh.dimensions = np.array(matID.shape) + 1
mesh.spacing = h_vec
mesh.origin = (0, 0, 0)
mesh.cell_data["value"] = matID.flatten(order="F")
mesh.plot(volume=True, opacity=0.1)


# ============================================================================
# Material Properties for Phase-Field Fracture Model
# ============================================================================

def initialise_materials_phasefield(
    grid, matID, lmbda, mu, gc, lc, dtype=np.float32
):
    """
    Initialize material property grids for phase-field fracture simulation.
    
    Args:
        grid: Computational grid specification
        matID: Material ID map for each voxel
        lmbda: Lame parameter (list, one per material)
        mu: Shear modulus (list, one per material)
        gc: Critical energy release rate (list, one per material)
        lc: Characteristic length scale (list, one per material)
        dtype: Data type for arrays
        
    Returns:
        Tuple of (lmbda_grid, mu_grid, gc_grid, lc_grid)
    """
    lmbda_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    mu_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    gc_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    lc_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    
    num_mats = len(lmbda)
    matids = np.unique(matID)
    
    if num_mats != len(matids):
        raise ValueError(
            "Number of material property entries must match the number "
            "of unique material IDs in the material map."
        )
    
    for i in range(num_mats):
        lmbda_grid[matID == matids[i]] = lmbda[i]
        mu_grid[matID == matids[i]] = mu[i]
        gc_grid[matID == matids[i]] = gc[i]
        lc_grid[matID == matids[i]] = lc[i]
    
    return jnp.array(lmbda_grid), jnp.array(mu_grid), gc_grid, lc_grid


# Define material properties: [matrix, fibre]
lmbda_list = [121.15, 0.0]      # Lame parameter
mu_list = [80.77, 0.0]          # Shear modulus
gc_list = [2.7e-3, 5e-3]        # Critical energy release rate
lc_list = [0.015, 0.015]        # Characteristic length

lmbda, mu, gc, lc = initialise_materials_phasefield(
    grid, matID, lmbda_list, mu_list, gc_list, lc_list, np.float32
)


# ============================================================================
# Loading and Solver Setup
# ============================================================================

# Define monotonic uniaxial loading (strain in x-direction)
Emean = 0.05
nsteps = 10

# Create strain history: ramp from 0 to Emean_xx over nsteps
Emean_steps = [
    jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0])
    for eps_xx in np.linspace(0, Emean, nsteps)
]

# Steps at which to save output fields
save_steps = [0, nsteps - 1]

# Solve the elastodamage phase-field problem
epsMacro, sigMacro, epsfield, sigfield, dfield = elastodamage_phasefield_solve(
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
)


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
        spacing=h_vec,
        origin=(0, 0, 0),
        stack_components=True,
    )
