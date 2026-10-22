
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from diffmat.fracture.rvegen import generate_brick_mortar_rve, init_material
from diffmat.commons.utilities import eng2lame, newton_raphson
from diffmat.commons.io import save_arrays_to_vti
from diffmat.fracture.solver import solve_fracture_staggered
from diffmat.fracture.solve_one_step import MaterialParams, SolverConfig
from diffmat.fracture.solver_jit import solve_loading_history

from jaxmaterials.common import get_grid_spec

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

jax.config.update("jax_enable_x64", True)

jax.config.update("jax_platform_name", "gpu")


# Geometry in µm (nacre-like: L/W = 10, exaggerated mortar)
L, W, t = 5.0, 0.5, 0.08
n_rep = (1, 1)
vxsiz = 0.01
matID, (Lx, Ly) = generate_brick_mortar_rve(L, W, t, n_rep=n_rep, voxel_size=vxsiz)
print("brick fraction:", matID.mean())

# Plot
plt.imshow(matID.T, origin="lower", extent=[0, Lx, 0, Ly], cmap="Oranges")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Brick (1) / mortar (0)")
os.makedirs("results/mortarbrick", exist_ok=True)
plt.savefig("results/mortarbrick/microstructure.png", dpi=150)
plt.close()

# Properties in MPa, µm; index 0 = mortar, 1 = brick
lam_m, mu_m = eng2lame(3.0e3, 0.35)       # biopolymer (or PMMA), 	Munch et al. 2008 Science
lam_b, mu_b = eng2lame(370.0e3, 0.22)     # alumina, Munch et al. 2008 Science
gc_b, gc_m = 40.0, 4.0  # J/m² == MPa·µm, 	Munch et al. 2008 Science
lc = 3 * vxsiz

lmbda, mu, gc, lc_grid = init_material(
    matID[..., None],
    lmbda_list=[lam_m, lam_b],
    mu_list   =[mu_m,  mu_b],
    gc_list   =[gc_m, gc_b], # J/m² == MPa·µm
    lc_list   =[lc, lc],
)

# ---------------------------------------------------------------------------
# Phase-field fracture simulations: uniaxial tension (e_xx) and simple shear (e_xy)
# Voigt order: [xx, yy, zz, xy, xz, yz] with tensor shear components (xy = gamma/2).
# Plane strain: nz = 1 and all out-of-plane strains are prescribed to zero.
# ---------------------------------------------------------------------------
nx, ny = matID.shape
grid = get_grid_spec(Lx, Ly, vxsiz, nx=nx, ny=ny, nz=1)

material_params = MaterialParams(lmbda=lmbda, mu=mu, gc=gc, lc=lc_grid)
solver_cfg = SolverConfig(
    grid=grid,
    lmbda0=float(0.5 * (lmbda.max() + lmbda.min())),
    mu0=float(0.5 * (mu.max() + mu.min())),
    k_stab=1e-6,
    maxiter_PF=1000,
    maxiter_Elas=1000,
    maxiter_staggered=100,
    tol_PF=1e-5,
    tol_Elas=1e-3,
    tol_staggered=1e-2,
    AA_depth=4,
    verbose=1,
)

nsteps = int(os.environ.get("NSTEPS", 200))
print(f'Running phase-field fracture simulations with {nsteps} steps')
loadings = {
    # name: (Voigt index of the loaded component, final strain, label)
    "tension": (0, 5e-2, r"$\varepsilon_{xx}$", r"$\sigma_{xx}$ (MPa)"),
    "shear":   (3, 8e-2, r"$\varepsilon_{xy}$", r"$\sigma_{xy}$ (MPa)"),
}

results = {}
for name, (idx, emax, xlabel, ylabel) in loadings.items():
    eps_steps = np.linspace(0.0, emax, nsteps + 1)[1:]
    strain_loading = np.zeros((nsteps, 6))
    strain_loading[:, idx] = eps_steps
    strain_loading = jnp.asarray(strain_loading, dtype=jnp.float64)

    t0 = time.time()
    eps_macro, sig_macro, _, _, damage = solve_loading_history(
        strain_loading,
        material_params,
        solver_cfg,
        out_dir=f"results/mortarbrick/{name}",
        save_steps=np.linspace(0, nsteps - 1, 10, dtype=int),
        save_unconverged=True,
    )
    jax.block_until_ready(sig_macro)
    print(f"{name}: {nsteps} steps in {time.time() - t0:.1f} s, "
          f"peak stress = {float(sig_macro[:, idx].max()):.1f} MPa")

    results[name] = dict(
        idx=idx, xlabel=xlabel, ylabel=ylabel,
        eps=np.asarray(eps_macro[:, idx]),
        sig=np.asarray(sig_macro[:, idx]),
        damage=np.asarray(damage),
    )
    np.savez(f"results/mortarbrick/{name}.npz", eps=results[name]["eps"],
             sig=results[name]["sig"], damage=results[name]["damage"][-1, :, :, 0])

    # Macroscopic stress-strain curves
    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.plot(eps_macro[:, idx], sig_macro[:, idx], "-", lw=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(name.capitalize()); ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Final damage fields, with the brick outlines overlaid
    fig, ax = plt.subplots(1, figsize=(9, 8))
    im = ax.imshow(damage[-1][:, :, 0].T, origin="lower", extent=[0, Lx, 0, Ly],
                   cmap="inferno", vmin=0, vmax=1)
    ax.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), matID.T,
               levels=[0.5], colors="c", linewidths=0.5)
    ax.set_title(f"Final damage ({name})")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="d")
    fig.tight_layout()
    
    plt.show()
