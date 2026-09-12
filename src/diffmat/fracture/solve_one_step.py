
import functools
from dataclasses import dataclass
from typing import Any

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

    def tree_flatten(self):
        return ((self.HH,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class SolverConfig:
    """Static fracture-solver settings carried outside differentiated state."""

    grid: Any
    lmbda0: jnp.ndarray
    mu0: jnp.ndarray
    k_stab: float
    maxiter_PF: int
    maxiter_Elas: int
    maxiter_inner: int
    tolerance_inner: float
    phase_field_tolerance: float = 1e-5
    elasticity_tolerance: float = 1e-2
    AA_depth: int = 4
    verbose: int = 0

    def tree_flatten(self):
        return (
            (
                self.lmbda0,
                self.mu0,
                self.k_stab,
            ),
            (
            self.grid,
                self.maxiter_PF,
                self.maxiter_Elas,
                self.maxiter_inner,
                self.tolerance_inner,
                self.phase_field_tolerance,
                self.elasticity_tolerance,
                self.AA_depth,
                self.verbose,
            ),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data[0], *children, *aux_data[1:])



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
        solver_cfg.grid,
        tolerance=solver_cfg.phase_field_tolerance,
        maxiter=solver_cfg.maxiter_PF,
        verbose=solver_cfg.verbose,
    )

    depsilon = epsilon - load_conditions.Emean[:, None, None, None]

    epsilon_new, _ = lippmann_schwinger(
        compute_sigma_damaged,
        (
            material_params.lmbda,
            material_params.mu,
            d_new,
            solver_cfg.k_stab,
        ),
        load_conditions.Emean,
        #delta_epsilon_initial=depsilon, #TODO:JaxMaterials has an if condition, which causes Tracer issue
        delta_epsilon_initial=None,
        ref_params={
            "lambda": solver_cfg.lmbda0,
            "mu": solver_cfg.mu0,
        },
        grid_spec=solver_cfg.grid,
        tol=solver_cfg.elasticity_tolerance,
        maxits=solver_cfg.maxiter_Elas,
        verbose=solver_cfg.verbose,
        depth=solver_cfg.AA_depth,
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

    def cond_fn(state):

        i, x, converged = state

        return (i < solver_cfg.maxiter_inner) & (~converged)

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
            strain_change < solver_cfg.tolerance_inner
        ) & (
            damage_change < solver_cfg.tolerance_inner
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

@jax.custom_vjp
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
    x0,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
    solver_cfg: SolverConfig,
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

    return (
        jax.tree.map(jnp.zeros_like, x_star),
        material_bar,
        load_bar,
        state_bar,
        None,
    )

solve_one_load_step.defvjp(
    solve_fwd,
    solve_bwd,
)
