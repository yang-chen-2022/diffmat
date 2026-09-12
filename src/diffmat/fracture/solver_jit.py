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
        "grid",
        "maxiter_PF",
        "maxiter_Elas",
        "maxiter_inner",
    ),
)
def solve_loading_history(
    Emean_steps,
    lmbda,
    mu,
    gc,
    lc,
    grid,
    k_stab,
    maxiter_PF,
    maxiter_Elas,
    maxiter_inner,
    tolerance_inner,
):

    dtype = lmbda.dtype

    d0 = jnp.zeros(
        (grid.nx, grid.ny, grid.nz),
        dtype=dtype,
    )

    HH0 = jnp.zeros_like(d0)

    epsilon0 = jnp.zeros(
        (6, grid.nx, grid.ny, grid.nz),
        dtype=dtype,
    ) + Emean_steps[0][:, None, None, None]

    lmbda0 = jax.lax.stop_gradient(0.5 * (lmbda.max() + lmbda.min()))
    mu0 = jax.lax.stop_gradient(0.5 * (mu.max() + mu.min()))

    material_params = MaterialParams(
        lmbda=lmbda,
        mu=mu,
        gc=gc,
        lc=lc,
        lmbda0=lmbda0,
        mu0=mu0,
        k_stab=k_stab,
    )
    solver_cfg = SolverConfig(
        grid=grid,
        maxiter_PF=maxiter_PF,
        maxiter_Elas=maxiter_Elas,
        maxiter_inner=maxiter_inner,
        tolerance_inner=tolerance_inner,
    )

    def step_fn(carry, Emean):

        d, epsilon, HH = carry

        load_conditions = LoadConditions(Emean=Emean)
        state_variables = StateVariables(HH=HH)

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
                (lmbda, mu, d, k_stab),
            )

        psi = compute_strain_energy(
                lmbda,
                mu,
                epsilon,
            )
        HH = jnp.maximum(HH, jax.lax.stop_gradient(psi))

        eps_macro = jnp.mean(epsilon, axis=(1, 2, 3))
        sig_macro = jnp.mean(sigma, axis=(1, 2, 3))

        output = (
            eps_macro,
            sig_macro,
            epsilon,
            sigma,
            d,
        )

        return (d, epsilon, HH), output

    _, outputs = jax.lax.scan(
        step_fn,
        (d0, epsilon0, HH0),
        Emean_steps,
    )

    return outputs

