import numpy as np
import jax
from jax import numpy as jnp

import diffmat.fracture.solve_one_step as solve_one_step_module
from diffmat.fracture.solve_one_step import (
    LoadConditions,
    MaterialParams,
    SolverConfig,
    StateVariables,
    solve_one_load_step,
)


def test_solver_input_dataclasses_are_pytrees():
    material_params = MaterialParams(
        lmbda=jnp.arange(2.0).reshape((2, 1, 1)),
        mu=jnp.ones((2, 1, 1)),
        gc=2.0 * jnp.ones((2, 1, 1)),
        lc=3.0 * jnp.ones((2, 1, 1)),
        lmbda0=jnp.array(4.0),
        mu0=jnp.array(5.0),
        k_stab=jnp.array(6.0),
    )
    load_conditions = LoadConditions(Emean=jnp.arange(6.0))
    state_variables = StateVariables(HH=7.0 * jnp.ones((2, 1, 1)))

    flat, unravel = jax.flatten_util.ravel_pytree(
        (material_params, load_conditions, state_variables)
    )
    restored_material, restored_load, restored_state = unravel(flat)

    assert isinstance(restored_material, MaterialParams)
    assert isinstance(restored_load, LoadConditions)
    assert isinstance(restored_state, StateVariables)
    np.testing.assert_allclose(restored_material.lmbda, material_params.lmbda)
    np.testing.assert_allclose(restored_load.Emean, load_conditions.Emean)
    np.testing.assert_allclose(restored_state.HH, state_variables.HH)


def test_solve_one_load_step_gradients_with_structured_inputs(monkeypatch):
    def fake_compute_strain_energy(lmbda, mu, epsilon):
        del lmbda, mu, epsilon
        return jnp.zeros((2, 1, 1))

    def fake_phase_field_solve(
        HH,
        d_old,
        gc,
        lc,
        grid,
        tolerance=1e-6,
        maxiter=1000,
        verbose=0,
    ):
        del d_old, grid, tolerance, maxiter, verbose
        return HH + gc + 2.0 * lc

    def fake_lippmann_schwinger(
        constitutive_model,
        params,
        Emean,
        delta_epsilon_initial,
        ref_params,
        grid_spec,
        tol,
        maxits,
        verbose,
        depth,
    ):
        del constitutive_model, delta_epsilon_initial, grid_spec, tol, maxits, verbose, depth
        lmbda, mu, d_new, k_stab = params
        epsilon = (
            Emean[:, None, None, None]
            + d_new[None, ...]
            + lmbda[None, ...]
            + 2.0 * mu[None, ...]
            + ref_params["lambda"]
            + 3.0 * ref_params["mu"]
            + k_stab
        )
        return epsilon, jnp.zeros_like(epsilon)

    monkeypatch.setattr(
        solve_one_step_module,
        "compute_strain_energy",
        fake_compute_strain_energy,
    )
    monkeypatch.setattr(
        solve_one_step_module,
        "phase_field_solve",
        fake_phase_field_solve,
    )
    monkeypatch.setattr(
        solve_one_step_module,
        "lippmann_schwinger",
        fake_lippmann_schwinger,
    )

    x0 = (
        jnp.zeros((2, 1, 1)),
        jnp.zeros((6, 2, 1, 1)),
    )
    material_params = MaterialParams(
        lmbda=jnp.ones((2, 1, 1)),
        mu=2.0 * jnp.ones((2, 1, 1)),
        gc=3.0 * jnp.ones((2, 1, 1)),
        lc=4.0 * jnp.ones((2, 1, 1)),
        lmbda0=jnp.array(5.0),
        mu0=jnp.array(6.0),
        k_stab=jnp.array(7.0),
    )
    load_conditions = LoadConditions(Emean=jnp.arange(6.0))
    state_variables = StateVariables(HH=jnp.ones((2, 1, 1)))
    solver_cfg = SolverConfig(
        grid=object(),
        maxiter_PF=1,
        maxiter_Elas=1,
        maxiter_inner=1,
        tolerance_inner=1e-6,
    )

    def loss_fn(material_params, load_conditions, state_variables):
        d_new, epsilon_new = solve_one_load_step(
            x0,
            material_params,
            load_conditions,
            state_variables,
            solver_cfg,
        )
        return jnp.sum(d_new) + jnp.sum(epsilon_new)

    material_bar, load_bar, state_bar = jax.grad(
        loss_fn,
        argnums=(0, 1, 2),
    )(material_params, load_conditions, state_variables)

    num_cells = 2.0
    np.testing.assert_allclose(material_bar.lmbda, 6.0 * jnp.ones((2, 1, 1)))
    np.testing.assert_allclose(material_bar.mu, 12.0 * jnp.ones((2, 1, 1)))
    np.testing.assert_allclose(material_bar.gc, 7.0 * jnp.ones((2, 1, 1)))
    np.testing.assert_allclose(material_bar.lc, 14.0 * jnp.ones((2, 1, 1)))
    np.testing.assert_allclose(material_bar.lmbda0, 6.0 * num_cells)
    np.testing.assert_allclose(material_bar.mu0, 18.0 * num_cells)
    np.testing.assert_allclose(material_bar.k_stab, 6.0 * num_cells)
    np.testing.assert_allclose(load_bar.Emean, num_cells * jnp.ones((6,)))
    np.testing.assert_allclose(state_bar.HH, 7.0 * jnp.ones((2, 1, 1)))
