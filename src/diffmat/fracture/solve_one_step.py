
import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import gmres
from jaxmaterials.solver import _backend as _lsbackend
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger

from diffmat.fracture.constitutive import compute_sigma_damaged, compute_strain_energy
from diffmat.fracture.lippmann_schwinger import solve
from diffmat.fracture.utilities import voigt_to_tensor, tensor_to_voigt


def _warn_if_not_converged(label, its, maxits):
    if its >= maxits:
        print(f"WARNING: JAX {label} LS solver reached maxits={int(maxits)} without converging")


# jaxmaterials raises on non-convergence; resolved at trace time, so patch before any jit.
_lsbackend._raise_if_not_converged = _warn_if_not_converged


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
    maxiter_staggered: int = 1
    tol_PF: float = 1e-5
    tol_Elas: float = 1e-2
    tol_staggered: float = 1e-3
    AA_depth: int = 4
    verbose: int = 0

class StaggeredLoopConfig(NamedTuple):
    maxiter_staggered: int
    tol_staggered: float

class StaggeredConfig(NamedTuple):
    grid: object
    lmbda0: float
    mu0: float
    k_stab: float
    maxiter_PF: int
    maxiter_Elas: int
    tol_PF: float
    tol_Elas: float
    AA_depth: int
    verbose: int

def split_solver_cfg(solver_cfg):
    staggered_loop_cfg = StaggeredLoopConfig(
        maxiter_staggered=int(solver_cfg.maxiter_staggered),
        tol_staggered=float(solver_cfg.tol_staggered),
    )
    staggered_cfg = StaggeredConfig(
        grid=solver_cfg.grid,
        lmbda0=float(solver_cfg.lmbda0),
        mu0=float(solver_cfg.mu0),
        k_stab=float(solver_cfg.k_stab),
        maxiter_PF=int(solver_cfg.maxiter_PF),
        maxiter_Elas=int(solver_cfg.maxiter_Elas),
        tol_PF=float(solver_cfg.tol_PF),
        tol_Elas=float(solver_cfg.tol_Elas),
        AA_depth=int(solver_cfg.AA_depth),
        verbose=int(solver_cfg.verbose),
    )
    return staggered_loop_cfg, staggered_cfg


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
    staggered_cfg: StaggeredConfig,
):

    (
        grid,
        lmbda0,
        mu0,
        k_stab,
        maxiter_PF,
        maxiter_Elas,
        tol_PF,
        tol_Elas,
        AA_depth,
        verbose,
    ) = staggered_cfg

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


def staggered_fixed_point(
    x0,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
    staggered_cfg: StaggeredConfig,
    staggered_loop_cfg: StaggeredLoopConfig,
):
    maxiter = staggered_loop_cfg.maxiter_staggered
    tol = staggered_loop_cfg.tol_staggered

    def cond_fn(state):
        i, x, converged = state
        return (i < maxiter) & (~converged)

    def body_fn(state):

        i, x, _ = state

        x_new = staggered_step(
            x,
            material_params,
            load_conditions,
            state_variables,
            staggered_cfg,
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
            strain_change < tol
        ) & (
            damage_change < tol
        )

        if staggered_cfg.verbose > 0:
            jax.debug.print(
                "\033[4m  staggered iter {i:4d}: strain_change={s:.3e} damage_change={d:.3e} converged={c}\033[0m",
                i=i + 1,
                s=strain_change,
                d=damage_change,
                c=converged,
                ordered=True,
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

    n_iter, x_star, converged = jax.lax.while_loop(
        cond_fn,
        body_fn,
        state0,
    )

    return x_star, n_iter, converged

@functools.partial(
    jax.custom_vjp,
    nondiff_argnames=("staggered_cfg", "staggered_loop_cfg"),
)
def solve_one_load_step_with_info(
    x0,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
    staggered_cfg: StaggeredConfig,
    staggered_loop_cfg: StaggeredLoopConfig,
):
    x_star, n_iter, converged = staggered_fixed_point(
        x0,
        material_params,
        load_conditions,
        state_variables,
        staggered_cfg,
        staggered_loop_cfg,
    )
    # Convergence info is integer/boolean, hence non-differentiable.
    return x_star, (n_iter, converged)


def solve_one_load_step(*args, **kwargs):
    """Differentiable load-step solve returning only the solution (d, epsilon)."""
    return solve_one_load_step_with_info(*args, **kwargs)[0]

def solve_fwd(
    x0,
    material_params: MaterialParams,
    load_conditions: LoadConditions,
    state_variables: StateVariables,
    staggered_cfg: StaggeredConfig,
    staggered_loop_cfg: StaggeredLoopConfig,
):

    x_star, n_iter, converged = staggered_fixed_point(
        x0,
        material_params,
        load_conditions,
        state_variables,
        staggered_cfg,
        staggered_loop_cfg,
    )

    return (x_star, (n_iter, converged)), (
        x_star,
        material_params,
        load_conditions,
        state_variables,
    )

def solve_bwd(
    staggered_cfg: StaggeredConfig,
    staggered_loop_cfg: StaggeredLoopConfig,
    residuals,
    g,
):
    x_star, material_params, load_conditions, state_variables = residuals
    g = g[0]  # cotangents of the convergence info are discarded

    def JT_lambda(v):
        _, pullback = jax.vjp(
            lambda x: staggered_step(
                x,
                material_params,
                load_conditions,
                state_variables,
                staggered_cfg,
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
            staggered_cfg,
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

solve_one_load_step_with_info.defvjp(
    solve_fwd,
    solve_bwd,
)
