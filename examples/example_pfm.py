"""
Test case for elastodamage phase-field fracture simulation.
"""

import numpy as np
from matplotlib import pyplot as plt
import jax
from jax import numpy as jnp

from diffmat.fracture.solver import elastodamage_phasefield_solve
from diffmat.fracture.rvegen import generate_particles_periodic, voxelise_particles_periodic, init_material
from diffmat.commons.io import save_arrays_to_vti
from diffmat.commons.utilities import eng2lame

from jaxmaterials.common import get_grid_spec

import time
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')


# Output directories
out_dir = f"results/fracture/forward_2"
os.makedirs(out_dir, exist_ok=True)


# ============================================================================
# Setup: Grid and RVE Geometry
# ============================================================================

# Create a 3D computational grid
box_size = [0.4, 0.4, 0.4] #physical length, mm
spacing = [0.005, 0.005, 0.005]
grid = get_grid_spec(
        box_size[0], 
        box_size[1], 
        box_size[2], 
        dx=spacing[0], 
        dy=spacing[1], 
        dz=spacing[2],
        )

# Random particle distribution
np.random.seed(123)
n_particles = 20
radius_range = [0.05, 0.1]
t0 = time.time()
positions, radii = generate_particles_periodic(
    n_particles,
    box_size,
    radius_range,
    min_gap=max(spacing),
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

# save for AMITEX simulation
from diffmat.perf2amitex.io import saveMesh2VTK_amitex
saveMesh2VTK_amitex('results/fracture/amitex/micr/matID.vtk', matID.astype(np.uint8)+1, 'matID', origin=[0,0,0], spacing=spacing)

# Define material properties
# NMC particle + LPSC matrix [Taghikhani et al. JMPS 2025]
E_particle = 175e3 #MPa
nu_particle = 0.28 
lmbda_particle, mu_particle = eng2lame(E_particle, nu_particle)
lc_particle = 0.01 #mm
gc_particle = 2.5e-3 #N/mm

E_matrix = 22e3 #MPa
nu_matrix = 0.37
lmbda_matrix, mu_matrix = eng2lame(E_matrix, nu_matrix)
lc_matrix = 0.01 #mm
gc_matrix = 2.8e-3 #N/mm

lmbda_list = [lmbda_matrix, lmbda_particle]    # Lame parameter
mu_list = [mu_matrix, mu_particle]         # Shear modulus
gc_list = [gc_matrix, gc_particle]    # Critical energy release rate
lc_list = [lc_matrix, lc_particle]      # Characteristic length

print(f'lambda_particle={lmbda_particle}, mu_particle={mu_particle}')
print(f'lc_particle={lc_particle}, gc_particle={gc_particle}')
print(f'lambda_matrix={lmbda_matrix}, mu_matrix={mu_matrix}')
print(f'lc_matrix={lc_matrix}, gc_matrix={gc_matrix}')
print(f'lambda0={(lmbda_particle+lmbda_matrix)/2.}, mu0={(mu_matrix+mu_particle)/2.}')

lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(
        matID,
        lmbda_list, 
        mu_list, 
        gc_list, 
        lc_list, 
        jnp.float64,
        )

# Define monotonic uniaxial loading (strain in x-direction)
Emean = 0.002
nsteps = 1000

exx0 = Emean/nsteps/10
Emean_steps = [
    jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0])
    for eps_xx in np.linspace(exx0, Emean, nsteps)
]

# Steps at which to save output fields
save_steps = np.arange(0, nsteps-1, 100)
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
    earlystop=0.2,
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



