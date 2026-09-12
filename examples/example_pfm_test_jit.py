"""
Test case for phase-field fracture simulation using `solve_loading_history`.
"""

import os
import time

import jax
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt

from diffmat.commons.io import save_arrays_to_vti
from diffmat.fracture.rvegen import init_material
from diffmat.fracture.solver_jit import solve_loading_history
from jaxmaterials.common import get_grid_spec

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

load = "tension"

# Output directories
if load == "tension":
    out_dir = "results/fracture/test_jit/SENT"
elif load == "shear":
    out_dir = "results/fracture/test_jit/SENS"
else:
    raise ValueError(f"Unsupported load case: {load}")

os.makedirs(out_dir, exist_ok=True)

# ============================================================================
# Setup: Grid and RVE Geometry
# ============================================================================

spacing = [0.01, 0.01, 0.01]
if load == "tension":
    box_size = [1.05, 1.0, spacing[2]]  # physical length, mm
elif load == "shear":
    box_size = [1.0, 1.0, spacing[2]]  # physical length, mm

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
matID[(X <= length) & (np.abs(Y - grid.ny / 2) <= thickness / 2)] = 0

# Void margin border
if load == "tension":
    matID[(X > grid.ny)] = 0

save_arrays_to_vti(
    filename=f"{out_dir}/matID.vtk",
    arrays=[matID[None, ...]],
    names=["matID"],
    spacing=spacing,
    origin=(0, 0, 0),
    stack_components=True,
)

# Define material properties
lmbda_list = [0.0, 121.15]   # Lame parameter, GPa, kN/mm2
mu_list = [0.0, 80.77]       # Shear modulus, GPa, kN/mm2
gc_list = [2.7e-3, 2.7e-3]   # Critical energy release rate, kN/mm
lc_list = [0.015, 0.015]     # Characteristic length, mm

lmbda_grid, mu_grid, gc_grid, lc_grid = init_material(
    matID,
    lmbda_list,
    mu_list,
    gc_list,
    lc_list,
    jnp.float64,
)

# Define monotonic uniaxial loading
if load == "tension":
    eps1, deps1 = 0.005, 5e-5
    eps2, deps2 = 0.008, 5e-5
    idx = 1
elif load == "shear":
    eps1, deps1 = 0.008, 5e-5
    eps2, deps2 = 0.02, 5e-5
    idx = 3

eps_steps = np.concatenate(
    (
        np.arange(0, eps1, deps1) + deps1,
        np.arange(eps1, eps2, deps2) + deps2,
    )
)

Emean_steps = jnp.stack(
    [
        jnp.array([0.0] * idx + [eps] + [0.0] * (5 - idx), dtype=jnp.float64)
        for eps in eps_steps
    ],
    axis=0,
)

# Solve the phase-field problem using solve_loading_history
t_start = time.time()
result = solve_loading_history(
    Emean_steps,
    lmbda_grid,
    mu_grid,
    gc_grid,
    lc_grid,
    grid,
    k_stab=1e-6,
    maxiter_PF=2000,
    maxiter_Elas=2000,
    maxiter_inner=30,
    tolerance_inner=1e-1,
    AA_depth=4,
)

# Handle either tuple or dict-style return safely
if isinstance(result, dict):
    epsMacro = result.get("epsMacro")
    sigMacro = result.get("sigMacro")
    sfield = result.get("sfield")
    efield = result.get("efield")
    dfield = result.get("dfield")
else:
    epsMacro, sigMacro, sfield, efield, dfield = result

print(f"TOTAL TIME FOR PFM SOLVE: {(time.time() - t_start):.3f} s")

# ============================================================================
# Results Visualization and Output
# ============================================================================

plt.figure()
plt.plot(epsMacro[:,1], sigMacro[:,1], "-*", label="Stress-strain curve")
plt.xlabel(r"Strain")
plt.ylabel(r"Stress")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
