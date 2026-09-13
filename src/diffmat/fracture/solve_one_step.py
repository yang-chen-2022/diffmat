
import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import gmres
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

from diffmat.fracture.constitutive import compute_sigma_damaged, compute_strain_energy
from diffmat.fracture.lippmann_schwinger import solve
from diffmat.fracture.utilities import voigt_to_tensor, tensor_to_voigt


@jax.tree_util.register_pytree_node_class
@dataclass
class MaterialParams:
    lmbda: jnp.ndarray
    mu: jnp.ndarray
    gc: jnp.ndarray
    lc: jnp.ndarray

    def tree_flatten(self):
        return (
            (
                self.lmbda,
                self.mu,
                self.gc,
                self.lc,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class LoadConditions:
    Emean: jnp.ndarray

    def tree_flatten(self):
        return ((self.Emean,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class StateVariables:
    HH: jnp.ndarray
    depsilon: jnp.ndarray

    def tree_flatten(self):
        return (
            (
                self.HH,
                self.depsilon,
            ), 
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class SolverConfig(NamedTuple):
    grid: Any
    lmbda0: float
    mu0: float
    k_stab: float = 1e-6
    maxiter_PF: int = 1000
    maxiter_Elas: int = 1000
    maxiter_inner: int = 1
    tol_PF: float = 1e-5
    tol_Elas: float = 1e-2
    tol_inner: float = 1e-3
    AA_depth: int = 4
    verbose: int = 0


# @jax.jit(static_argnames=["grid", "tolerance", "maxiter"])
def phase_field_solve(HH, d_old, gc, lc, grid, tolerance=1e-6, maxiter=1000, verbose=0):
    """Fixed-point iteration solver for phase-field problem (fracture)

    :arg HH: history strain energy (field), (1, Nx, Ny, Nz)
    :arg d_old: damage variable at previous time step (field), (1, Nx, Ny, Nz)
    :arg gc: fracture toughness (field), (1, Nx, Ny, Nz)
    :arg lc: regularisation length (field)), (1, Nx, Ny, Nz)
    :arg grid: grid specs
    :arg tolerance: tolerance for convergence check
    :arg maxiter: maximal number of iterations
    """

    # Coefficients A^t_n and B^t_n
    A_n = 1.0 / (lc**2) + 2.0 * HH / (gc * lc)
    B_n = 2.0 * HH / (gc * lc)

    d_final = solve(
        B_n, 
        A_n, 
        grid, 
        u_in=jax.lax.stop_gradient(d_old), 
        tol=tolerance, 
        maxits=maxiter, 
        verbose=verbose
    )

    return d_final



def staggered_step(
    x,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
    solver_cfg: SolverConfig,
):

    (
        grid,
        lmbda0,
        mu0,
        k_stab,
        maxiter_PF,
        maxiter_Elas,
        maxiter_inner,
        tol_PF,
        tol_Elas,
        tol_inner,
        AA_depth,
        verbose,
    ) = solver_cfg

    d, epsilon = x

    psi = compute_strain_energy(
        material_params.lmbda,
        material_params.mu,
        epsilon,
    )
    HH_inner = jnp.maximum(state_variables.HH, jax.lax.stop_gradient(psi))

    d_new = phase_field_solve(
        HH_inner,
        d,
        material_params.gc,
        material_params.lc,
        grid,
        tolerance=tol_PF,
        maxiter=maxiter_PF,
        verbose=verbose,
    )

    epsilon_new, _ = lippmann_schwinger(
        compute_sigma_damaged,
        (
            material_params.lmbda,
            material_params.mu,
            d_new,
            k_stab,
        ),
        load_conditions.Emean,
        delta_epsilon_initial=state_variables.depsilon, 
        ref_params={
            "lambda": lmbda0,
            "mu": mu0,
        },
        grid_spec=grid,
        tol=tol_Elas,
        maxits=maxiter_Elas,
        verbose=verbose,
        depth=AA_depth,
    )

    return (
        d_new,
        epsilon_new,
    )


def inner_fixed_point(
    x0,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
    solver_cfg: SolverConfig,
):
    (
        grid,
        lmbda0,
        mu0,
        k_stab,
        maxiter_PF,
        maxiter_Elas,
        maxiter_inner,
        tolerance_inner,
        phase_field_tolerance,
        elasticity_tolerance,
        AA_depth,
        verbose,
    ) = solver_cfg

    def cond_fn(state):

        i, x, converged = state

        return (i < maxiter_inner) & (~converged)

    def body_fn(state):

        i, x, _ = state

        x_new = staggered_step(
            x,
            material_params,
            load_conditions,
            state_variables,
            solver_cfg,
        )

        d, epsilon = x
        d_new, epsilon_new = x_new

        damage_change = (
            jnp.max(
                jnp.abs(
                    d_new - d
                )
            )
        )

        strain_delta = voigt_to_tensor(
            epsilon_new - epsilon
        )

        strain_delta_norm = jnp.linalg.norm(
            strain_delta,
            axis=(-2, -1),
        )

        strain_norm = jnp.linalg.norm(
            voigt_to_tensor(epsilon_new),
            axis=(-2, -1),
        )

        prev_strain_norm = jnp.linalg.norm(
            voigt_to_tensor(epsilon),
            axis=(-2, -1),
        )

        strain_scale = jnp.maximum(
            jnp.maximum(
                strain_norm,
                prev_strain_norm,
            ),
            jnp.finfo(
                epsilon.dtype
            ).tiny,
        )

        strain_change = jnp.max(
            strain_delta_norm / strain_scale
        )

        converged = (
            strain_change < tol_inner
        ) & (
            damage_change < tol_inner
        )

        return (
            i + 1,
            x_new,
            converged,
        )

    state0 = (
        0,
        x0,
        False,
    )

    _, x_star, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        state0,
    )

    return x_star

@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def solve_one_load_step(
    x0,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
    solver_cfg: SolverConfig,
):
    return inner_fixed_point(
        x0,
        material_params,
        load_conditions,
        state_variables,
        solver_cfg,
    )

def solve_fwd(
    solver_cfg: SolverConfig,
    x0,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
):

    x_star = inner_fixed_point(
        x0,
        material_params,
        load_conditions,
        state_variables,
        solver_cfg,
    )

    return x_star, (
        x_star,
        material_params,
        load_conditions,
        state_variables,
        solver_cfg,
    )

def solve_bwd(
    solver_cfg,
    residuals,
    g,
):

    x_star, material_params, load_conditions, state_variables, solver_cfg = residuals

    def JT_lambda(v):
        _, pullback = jax.vjp(
            lambda x: staggered_step(
                x,
                material_params,
                load_conditions,
                state_variables,
                solver_cfg,
            ),
            x_star,
        )
        JTv = pullback(v)[0]
        return jax.tree.map(
            lambda v_, j_:
            v_ - j_,
            v,
            JTv,
        )

    rhs, unravel = ravel_pytree(g)

    def linear_operator(vec):
        tree = unravel(vec)
        out = JT_lambda(tree)
        flat, _ = ravel_pytree(out)
        return flat

    lam_flat, _ = gmres(
        linear_operator,
        rhs,
    )

    lam = unravel(lam_flat)

    _, pullback = jax.vjp(
        lambda mp, lc, sv: staggered_step(
            x_star,
            mp,
            lc,
            sv,
            solver_cfg,
        ),
        material_params,
        load_conditions,
        state_variables,
    )

    material_bar, load_bar, state_bar = pullback(lam)
    x0_bar = jax.tree_util.tree_map(jnp.zeros_like, x_star)

    return (
        x0_bar,
        material_bar,
        load_bar,
        state_bar,
    )

solve_one_load_step.defvjp(
    solve_fwd,
    solve_bwd,
)
