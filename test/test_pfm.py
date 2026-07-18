import numpy as np
from contextlib import contextmanager
import time
from collections import namedtuple
from matplotlib import pyplot as plt
import jax

from jax import numpy as jnp


from src.solver import elastodamage_phasefield_solve
from src.utils_compvoxel import (
    fibre,
    grid_spec,
    fibre_association,
    reconstruct_plane,
    get_level_set_at_points,
    voxel_clipping,
)

from src.helper import save_arrays_to_vti




# initialise grid
grid = grid_spec(1.0, 2, 3, 0.03, 0.03, 0.03)

# generate fibres
fibres = fibre([[0.2, 0.5, 0.5]], # ref point
               [[.0, 0.0, 1.0]], # fibre vector
               [0.3])                         # fibre radi

#
centers = np.array(grid.build_centers())
x0 = np.array(fibres.x0)
v = np.array(fibres.v)
r = np.array(fibres.r)
h_vec = np.array([grid.dx, grid.dy, grid.dz])

###########################################
## associate voxels to the closest fibre ##
###########################################

all_in, all_out, composite, idx_closest = fibre_association(centers, h_vec, x0, v, r)
matID = np.where(all_out, 0, idx_closest + 1)
matID = matID.reshape((grid.nx, grid.ny, grid.nz))

#visu
import pyvista as pv
mesh = pv.ImageData()
mesh.dimensions = np.array(matID.shape) + 1
mesh.spacing = h_vec
mesh.origin = (0, 0, 0)
mesh.cell_data["value"] = matID.flatten(order="F")

opacity = [0, 0.1, 0.3, 0.6, 1.0]

mesh.plot(volume=True, opacity=0.1)




##################################################
## phase-field fracture without composite voxels##
##################################################

def initialise_materials_phasefield(grid, matID, lmbda, mu, gc, lc, dtype=np.float32):
    '''
    initialise material properties on a grid,
    with a phase-field damage model
    '''
    
    lmbda_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    mu_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    gc_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    lc_grid = np.zeros((grid.nx, grid.ny, grid.nz), dtype=dtype)
    
    num_mats = len(lmbda)
    matids = np.unique(matID)
    
    if num_mats != len(matids):
        raise ValueError("length of lambda/mu list must match the matID map!")
    
    for i in range(num_mats):
        lmbda_grid[matID==matids[i]] = lmbda[i]
        mu_grid[matID==matids[i]] = mu[i]
        gc_grid[matID==matids[i]] = gc[i]
        lc_grid[matID==matids[i]] = lc[i]
    
    return jnp.array(lmbda_grid), jnp.array(mu_grid), gc_grid, lc_grid

# initialise material properties 
lmbda_list = [121.15, 0.] #should match the values in matID
mu_list    = [80.77, 0.] 
gc_list    = [2.7e-3, 5e-3, ]
lc_list    = [0.015, 0.015]
lmbda, mu, gc, lc = initialise_materials_phasefield(grid, matID, lmbda_list, mu_list, gc_list, lc_list, np.float32)

# loading condition
Emean = 0.05
nsteps = 10

Emean_steps = [jnp.array([eps_xx, 0.0, 0.0, 0.0, 0.0, 0.0]) 
                  for eps_xx in np.linspace(0, Emean, nsteps)]

save_steps = [0, nsteps-1]

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


icomp = 0
plt.figure()
plt.plot(epsMacro[:,icomp], sigMacro[:,icomp], '-*')
plt.plot(epsMacro[save_steps,icomp], sigMacro[save_steps,icomp], 'o')
plt.show()


# save into vtk
for step in save_steps:
    save_arrays_to_vti(
        filename=f"stress_strain_damage_field_{step}.vtk",
        arrays=[epsfield[step], sigfield[step], dfield[step][None,...]],
        names=["Strain", "Stress", "Damage"],
        spacing = h_vec,
        origin = (0,0,0),
        stack_components=True,
    )



