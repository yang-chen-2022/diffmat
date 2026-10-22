"""Integration test: JIT-compiled phase-field solver on a single-edge-notched
tension (SENT) specimen.

Runs a small 2D-like problem (one voxel thick) on the CPU and checks the physics
of the response (elastic loading, stress peak, brittle softening, crack
localisation), the file output, and gradient compatibility.
"""

import glob
import os

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from diffmat.fracture.rvegen import init_material
from diffmat.fracture.solve_one_step import MaterialParams, SolverConfig
from diffmat.fracture.solver_jit import solve_loading_history
from jaxmaterials.common import get_grid_spec

SPACING = 0.05
LAMBDA, MU = 121.15, 80.77
GC, LC = 2.7e-3, 0.1
# Uniaxial strain E22 = e; the plane-strain stress is sigma22 = (lambda + 2 mu) e
# in the undamaged ligament, scaled by the solid fraction of the section.
EPS_STEPS = np.arange(1, 17) * 4e-4


@pytest.fixture(scope="module", autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def build_problem():
    grid = get_grid_spec(1.05, 1.0, SPACING, dx=SPACING, dy=SPACING, dz=SPACING)
    X, Y, _ = np.meshgrid(
        *[np.arange(n) for n in (grid.nx, grid.ny, grid.nz)], indexing="ij"
    )
    mat_id = np.ones((grid.nx, grid.ny, grid.nz), dtype=np.uint8)
    mat_id[(X <= grid.ny * 0.5) & (np.abs(Y - grid.ny / 2) <= 0.5)] = 0  # notch
    mat_id[X > grid.ny] = 0  # void margin

    lmbda, mu, gc, lc = init_material(
        mat_id, [0.0, LAMBDA], [0.0, MU], [GC] * 2, [LC] * 2, jnp.float64
    )
    material_params = MaterialParams(lmbda=lmbda, mu=mu, gc=gc, lc=lc)
    solver_cfg = SolverConfig(
        grid=grid,
        lmbda0=float(0.5 * (lmbda.max() + lmbda.min())),
        mu0=float(0.5 * (mu.max() + mu.min())),
        k_stab=1e-6,
        maxiter_PF=1000,
        maxiter_Elas=1000,
        maxiter_staggered=30,
        tol_PF=1e-5,
        tol_Elas=1e-3,
        tol_staggered=1e-2,
        AA_depth=4,
    )
    Emean_steps = jnp.stack(
        [jnp.array([0.0, e, 0.0, 0.0, 0.0, 0.0]) for e in EPS_STEPS]
    )
    return grid, mat_id, material_params, solver_cfg, Emean_steps


@pytest.fixture(scope="module")
def sent_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("sent")
    grid, mat_id, material_params, solver_cfg, Emean_steps = build_problem()
    outputs = solve_loading_history(
        Emean_steps,
        material_params,
        solver_cfg,
        out_dir=str(out_dir),
        save_steps=[0, len(EPS_STEPS) - 1],
        save_unconverged=False,
    )
    outputs = jax.tree.map(np.asarray, outputs)
    return out_dir, mat_id, outputs, (material_params, solver_cfg, Emean_steps)


def test_macroscopic_response_is_physical(sent_run):
    _, mat_id, (eps_macro, sig_macro, _, _, damage), _ = sent_run
    sig_yy = sig_macro[:, 1]

    assert all(np.isfinite(x).all() for x in (eps_macro, sig_macro, damage))
    np.testing.assert_allclose(eps_macro[:, 1], EPS_STEPS, rtol=1e-6)

    # Linear elastic start: slope of the first step is the solid-fraction-scaled
    # stiffness (the notch and void margin reduce the effective modulus).
    solid_fraction = mat_id.mean()
    assert 0.0 < sig_yy[0] < (LAMBDA + 2 * MU) * EPS_STEPS[0]
    assert sig_yy[0] > 0.1 * solid_fraction * (LAMBDA + 2 * MU) * EPS_STEPS[0]
    np.testing.assert_allclose(sig_yy[1] / sig_yy[0], 2.0, rtol=0.05)

    # Stress peaks, then softens abruptly after the crack propagates.
    peak = int(np.argmax(sig_yy))
    assert 0 < peak < len(sig_yy) - 1
    assert sig_yy[-1] < 0.2 * sig_yy[peak]
    assert np.all(np.diff(sig_yy[: peak + 1]) > 0)


def test_damage_localises_and_is_bounded(sent_run):
    _, mat_id, (_, _, _, _, damage), _ = sent_run

    # Small tolerance: the solver is not strictly bounded to [0, 1].
    assert damage.min() > -1e-3 and damage.max() < 1.0 + 1e-3

    # Damage never decreases (irreversibility through the history field).
    max_damage = damage.max(axis=(1, 2, 3))
    assert np.all(np.diff(max_damage) > -1e-3)

    # Fully developed crack at the end, far from the loading-free initial state.
    assert max_damage[0] < 0.1
    assert max_damage[-1] > 0.9

    # The crack runs through the ligament ahead of the notch tip, at mid-height.
    final = damage[-1, :, :, 0]
    ny = final.shape[1]
    row = ny // 2
    ligament = slice(int(ny * 0.5) + 2, ny - 1)
    assert final[ligament, row - 1 : row + 2].max() > 0.9
    assert final[ligament, :3].max() < 0.5  # far from the crack path


def test_results_written_to_disk(sent_run):
    out_dir, _, (eps_macro, sig_macro, _, _, _), _ = sent_run

    table = np.loadtxt(os.path.join(out_dir, "macro_curve.txt"), skiprows=1)
    assert table.shape == (len(EPS_STEPS), 14)
    np.testing.assert_allclose(table[:, 0], np.arange(len(EPS_STEPS)))
    np.testing.assert_allclose(table[:, 1:7], eps_macro, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(table[:, 7:13], sig_macro, rtol=1e-4, atol=1e-10)

    saved = np.flatnonzero(table[:, 13])
    np.testing.assert_array_equal(saved, [0, len(EPS_STEPS) - 1])
    names = sorted(os.path.basename(p) for p in glob.glob(f"{out_dir}/fields_*.vtk"))
    assert names == [f"fields_{s:04d}.vtk" for s in saved]


def test_gradient_is_finite_and_unaffected_by_saving(sent_run, tmp_path):
    _, _, _, (material_params, solver_cfg, Emean_steps) = sent_run
    short_loading = Emean_steps[:3]

    def loss(mu, **kwargs):
        params = MaterialParams(
            lmbda=material_params.lmbda,
            mu=mu,
            gc=material_params.gc,
            lc=material_params.lc,
        )
        return solve_loading_history(
            short_loading, params, solver_cfg, **kwargs
        )[1][:, 1].sum()

    grad_plain = jax.grad(loss)(material_params.mu)
    grad_saving = jax.grad(
        lambda mu: loss(
            mu, out_dir=str(tmp_path), save_steps=[0], save_unconverged=True
        )
    )(material_params.mu)

    assert np.isfinite(grad_plain).all()
    assert np.abs(grad_plain).max() > 0
    np.testing.assert_array_equal(grad_plain, grad_saving)
    assert os.path.exists(os.path.join(tmp_path, "fields_0000.vtk"))
