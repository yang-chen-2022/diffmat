import jax
from jax import numpy as jnp

from diffmat.fracture.solve_one_step import (
    LoadConditions,
    MaterialParams,
    SolverConfig,
    StateVariables,
    solve_one_load_step,
)
from diffmat.fracture.constitutive import compute_sigma_damaged, compute_strain_energy

from functools import partial


# Incremental solution with staggered scheme
@partial(
    jax.jit,
    static_argnames=(
        "solver_cfg",
    ),
)
def solve_loading_history(
    Emean_steps,
    material_params: MaterialParams,
    solver_cfg: SolverConfig,
):

    dtype = material_params.lmbda.dtype

    d0 = jnp.zeros(
        (solver_cfg.grid.nx, solver_cfg.grid.ny, solver_cfg.grid.nz),
        dtype=dtype,
    )

    HH0 = jnp.zeros_like(d0)

    epsilon0 = jnp.zeros(
        (6, solver_cfg.grid.nx, solver_cfg.grid.ny, solver_cfg.grid.nz),
        dtype=dtype,
    ) + Emean_steps[0][:, None, None, None]

    depsilon0 = jnp.zeros_like(epsilon0)

    def step_fn(carry, Emean):

        d, epsilon, HH, depsilon = carry

        load_conditions = LoadConditions(Emean=Emean)
        state_variables = StateVariables(
                HH=HH,
                depsilon=depsilon,
            )

        (
            d,
            epsilon,
        ) = solve_one_load_step(
            (
                d,
                epsilon,
            ),
            material_params,
            load_conditions,
            state_variables,
            solver_cfg,
        )

        sigma = compute_sigma_damaged(
                epsilon,
                (material_params.lmbda, material_params.mu, d, solver_cfg.k_stab),
            )

        psi = compute_strain_energy(
                material_params.lmbda,
                material_params.mu,
                epsilon,
            )
        HH = jax.lax.stop_gradient(jnp.maximum(HH, psi))

        depsilon = epsilon - Emean[:, None, None, None]
        depsilon = jax.lax.stop_gradient(depsilon)

        eps_macro = jnp.mean(epsilon, axis=(1, 2, 3))
        sig_macro = jnp.mean(sigma, axis=(1, 2, 3))

        output = (
            eps_macro,
            sig_macro,
            epsilon,
            sigma,
            d,
        )

        return (d, epsilon, HH, depsilon), output

    _, outputs = jax.lax.scan(
        step_fn,
        (d0, epsilon0, HH0, depsilon0),
        Emean_steps,
    )

    return outputs
