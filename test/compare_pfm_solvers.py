"""Numerically compare the Python and JIT phase-field fracture solvers.

The JIT solver uses a previous strain perturbation as its elasticity initial guess; 
the legacy solver currently does not, so this measures their practical agreement 
rather than asserting identical iteration trajectories.
"""

from pathlib import Path
import tempfile
import time

import jax
import numpy as np
from jax import numpy as jnp

from diffmat.fracture.rvegen import init_material
from diffmat.fracture.solver import solve_fracture_staggered
from diffmat.fracture.solver_jit import solve_loading_history
from diffmat.fracture.solve_one_step import MaterialParams, SolverConfig
from jaxmaterials.common import get_grid_spec

N_STEPS = 160
MAXITER = 1000
MAXITER_STAGGERED = 30
STAGGERED_TOLERANCE = 1e-3
RTOL = 1e-3
ATOL = 1e-8
PLOT_FILE = "pfm_solver_comparison.png"


def build_problem():
    spacing = [0.05, 0.05, 0.05]
    grid = get_grid_spec(1.05, 1.0, spacing[2], dx=spacing[0], dy=spacing[1], dz=spacing[2])

    x, y, z = np.meshgrid(
        np.arange(grid.nx), np.arange(grid.ny), np.arange(grid.nz), indexing="ij"
    )
    mat_id = np.ones((grid.nx, grid.ny, grid.nz), dtype=np.uint8)
    mat_id[(x <= grid.ny * 0.5) & (np.abs(y - grid.ny / 2) <= spacing[1])] = 0
    mat_id[x > grid.ny] = 0

    material = init_material(
        mat_id,
        [0.0, 121.15],
        [0.0, 80.77],
        [2.7e-3, 2.7e-3],
        [0.015, 0.015],
        jnp.float64,
    )

    eps_steps = np.concatenate(
        (
            np.arange(0.0, 0.005, 5e-5) + 5e-5,
            np.arange(0.005, 0.008, 5e-5) + 5e-5,
        )
    )
    mean_strain = jnp.zeros((len(eps_steps), 6), dtype=jnp.float64)
    mean_strain = mean_strain.at[:, 1].set(jnp.asarray(eps_steps))
    return grid, material, mean_strain[:N_STEPS]


def discrepancy(reference, candidate):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    difference = reference - candidate
    reference_norm = np.linalg.norm(reference.ravel())
    relative_l2 = np.linalg.norm(difference.ravel()) / max(
        reference_norm, np.finfo(np.float64).tiny
    )
    return float(np.max(np.abs(difference))), float(relative_l2)


def save_comparison_plot(path, legacy, compiled):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    legacy_strain = np.asarray(legacy[0])[:, 1]
    legacy_stress = np.asarray(legacy[1])[:, 1]
    jit_strain = np.asarray(compiled[0])[:, 1]
    jit_stress = np.asarray(compiled[1])[:, 1]
    strain_error = np.max(np.abs(np.asarray(legacy[0]) - np.asarray(compiled[0])), axis=1)
    stress_error = np.max(np.abs(np.asarray(legacy[1]) - np.asarray(compiled[1])), axis=1)

    legacy_damage_field = np.asarray(legacy[4])
    jit_damage_field = np.asarray(compiled[4])[:, None, ...]
    legacy_damage = legacy_damage_field[:, 0, :, :, 0]
    jit_damage = jit_damage_field[:, 0, :, :, 0]
    damage_error = np.max(
        np.abs(legacy_damage_field - jit_damage_field).reshape(len(legacy_strain), -1),
        axis=1,
    )
    final_damage_diff = np.abs(legacy_damage[-1] - jit_damage[-1])

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].plot(legacy_strain, legacy_stress, label="Legacy", linewidth=2)
    axes[0, 0].plot(jit_strain, jit_stress, "--", label="JIT", linewidth=2)
    axes[0, 0].set(title="Stress-strain response", xlabel="Mean strain $\\varepsilon_{yy}$", ylabel="Mean stress $\\sigma_{yy}$")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    steps = np.arange(len(legacy_strain))
    axes[0, 1].semilogy(steps, np.maximum(stress_error, np.finfo(float).tiny))
    axes[0, 1].set(title="Macro stress error", xlabel="Load step", ylabel="Maximum absolute error")
    axes[0, 1].grid(alpha=0.3)

    image_options = {"origin": "lower", "extent": (0, 1.05, 0, 1.0), "aspect": "equal"}
    damage_image = axes[0, 2].imshow(legacy_damage[-1].T, vmin=0, vmax=1, cmap="viridis", **image_options)
    axes[0, 2].set(title="Legacy final damage", xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(damage_image, ax=axes[0, 2], label="Damage")

    damage_image = axes[1, 0].imshow(jit_damage[-1].T, vmin=0, vmax=1, cmap="viridis", **image_options)
    axes[1, 0].set(title="JIT final damage", xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(damage_image, ax=axes[1, 0], label="Damage")

    difference_image = axes[1, 1].imshow(final_damage_diff.T, cmap="magma", **image_options)
    axes[1, 1].set(title="Final |damage difference|", xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(difference_image, ax=axes[1, 1], label="Absolute difference")

    axes[1, 2].semilogy(steps, np.maximum(strain_error, np.finfo(float).tiny), label="Strain")
    axes[1, 2].semilogy(steps, np.maximum(damage_error, np.finfo(float).tiny), label="Damage")
    axes[1, 2].set(title="Field and macro strain errors", xlabel="Load step", ylabel="Maximum absolute error")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    jax.config.update("jax_enable_x64", True)
    grid, (lmbda, mu, gc, lc), mean_strain = build_problem()
    material_params = MaterialParams(lmbda=lmbda, mu=mu, gc=gc, lc=lc)
    lmbda0 = float(0.5 * (lmbda.max() + lmbda.min()))
    mu0 = float(0.5 * (mu.max() + mu.min()))

    solver_cfg = SolverConfig(
        grid=grid,
        lmbda0=lmbda0,
        mu0=mu0,
        k_stab=1e-6,
        maxiter_PF=MAXITER,
        maxiter_Elas=MAXITER,
        maxiter_staggered=MAXITER_STAGGERED,
        tol_PF=1e-5,
        tol_Elas=1e-2,
        tol_staggered=STAGGERED_TOLERANCE,
        AA_depth=4,
        verbose=0,
    )

    with tempfile.TemporaryDirectory(prefix="compare_pfm_solvers_") as output_dir:
        start = time.perf_counter()
        compiled = solve_loading_history(mean_strain, material_params, solver_cfg)
        compiled = jax.tree.map(lambda value: value.block_until_ready(), compiled)
        jit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        legacy = solve_fracture_staggered(
            grid,
            lmbda,
            mu,
            gc,
            lc,
            list(mean_strain),
            save_steps=np.arange(len(mean_strain)),
            k_stab=solver_cfg.k_stab,
            maxiter_PF=solver_cfg.maxiter_PF,
            maxiter_Elas=solver_cfg.maxiter_Elas,
            out_dir=output_dir,
            earlystop=None,
            maxiter_staggered=solver_cfg.maxiter_staggered,
            tol_staggered=solver_cfg.tol_staggered,
            output_fields=True,
        )
        legacy = jax.tree.map(
            lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
            legacy,
        )
        legacy_seconds = time.perf_counter() - start

    compiled_damage = np.asarray(compiled[4])[:, None, ...]

    # Both APIs return (mean strain, mean stress, stress field, strain field,
    # damage field), though the JIT API puts fields in a different order.
    pairs = {
        "macro strain": (legacy[0], compiled[0]),
        "macro stress": (legacy[1], compiled[1]),
        "strain field": (legacy[3], compiled[2]),
        "stress field": (legacy[2], compiled[3]),
        "damage field": (legacy[4], compiled_damage),
    }

    print(f"Compared {len(mean_strain)} tension load steps on grid {grid.nx}x{grid.ny}x{grid.nz}.")
    print(f"Legacy elapsed: {legacy_seconds:.2f} s")
    print(f"JIT elapsed (includes compilation): {jit_seconds:.2f} s")
    print(f"Acceptance: np.allclose(rtol={RTOL:g}, atol={ATOL:g})")

    for label, (reference, candidate) in pairs.items():
        reference = np.asarray(reference)
        candidate = np.asarray(candidate)
        max_abs, relative_l2 = discrepancy(reference, candidate)
        matches = np.allclose(reference, candidate, rtol=RTOL, atol=ATOL)
        print(
            f"{label:13s}: shape={reference.shape}, max_abs={max_abs:.6e}, "
            f"relative_L2={relative_l2:.6e}, allclose={matches}"
        )

    print("\nPer-step macro maximum absolute errors:")
    print(" step       strain       stress       damage")
    legacy_damage = np.asarray(legacy[4])
    jit_damage = compiled_damage
    for step in range(len(mean_strain)):
        strain_error = discrepancy(legacy[0][step], compiled[0][step])[0]
        stress_error = discrepancy(legacy[1][step], compiled[1][step])[0]
        damage_error = discrepancy(legacy_damage[step], jit_damage[step])[0]
        print(f"{step:5d}  {strain_error:12.4e} {stress_error:12.4e} {damage_error:12.4e}")

    plot_path = save_comparison_plot(PLOT_FILE, legacy, compiled)
    print(f"Saved comparison plot: {plot_path}")


if __name__ == "__main__":
    main()