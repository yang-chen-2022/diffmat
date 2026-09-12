
import jax
from jax import numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jaxmaterials.solver.lippmann_schwinger import lippmann_schwinger
from diffmat.fracture.constitutive import compute_sigma_damaged, compute_strain_energy
from diffmat.fracture.utilities import voigt_to_tensor, tensor_to_voigt
from diffmat.fracture.lippmann_schwinger import solve



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
    params,
):

    d, epsilon = x

    (
        lmbda,
        mu,
        gc,
        lc,
        Emean,
        HH,
        grid,
        lmbda0,
        mu0,
        k_stab,
        maxiter_PF,
        maxiter_Elas,
    ) = params

    psi = compute_strain_energy(
        lmbda,
        mu,
        epsilon,
    )
    HH_inner = jnp.maximum(HH, jax.lax.stop_gradient(psi))

    d_new = phase_field_solve(
        HH_inner,
        d,
        gc,
        lc,
        grid,
        tolerance=1e-5,
        maxiter=maxiter_PF,
        verbose=0,
    )

    depsilon = epsilon - E_mean[:, None, None, None]

    epsilon_new, _ = lippmann_schwinger(
        compute_sigma_damaged,
        (lmbda, mu, d_new, k_stab),
        E_mean,
        delta_epsilon_initial=depsilon,
        ref_params={
            "lambda": lmbda0,
            "mu": mu0,
        },
        grid_spec=grid,
        tol=1e-2,
        maxits=maxiter_Elas,
        verbose=0,
        depth=4,
    )

    return (
        d_new,
        epsilon_new,
    )



def inner_fixed_point(
    x0,
    params,
    maxiter_inner,
    tolerance_inner,
):

    def cond_fn(state):

        i, x, converged = state

        return (i < maxiter_inner) & (~converged)

    def body_fn(state):

        i, x, _ = state

        x_new = staggered_step(
            x,
            params,
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
            strain_change < tolerance_inner
        ) & (
            damage_change < tolerance_inner
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



def residual(
    x,
    params,
):
    Fx = staggered_step(
        x,
        params,
    )

    return jax.tree.map(
        lambda a, b: a - b,
        Fx,
        x,
    )


@jax.custom_vjp
def solve_one_load_step(
    x0,
    params,
    maxiter_inner,
    tolerance_inner,
):
    return inner_fixed_point(
        x0,
        params,
        maxiter_inner,
        tolerance_inner,
    )

def solve_fwd(
    x0,
    params,
    maxiter_inner,
    tolerance_inner,
):

    x_star = inner_fixed_point(
        x0,
        params,
        maxiter_inner,
        tolerance_inner,
    )

    return x_star, (
        x_star,
        params,
    )

def solve_bwd(
    residuals,
    g,
):

    x_star, params = residuals

    def JT_lambda(v):

        _, pullback = jax.vjp(
            lambda x:
            staggered_step(
                x,
                params,
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

    rhs, unravel = jax.flatten_util.ravel_pytree(g)

    def linear_operator(vec):

        tree = unravel(vec)

        out = JT_lambda(tree)

        flat, _ = jax.flatten_util.ravel_pytree(out)

        return flat

    lam_flat, _ = gmres(
        linear_operator,
        rhs,
    )

    lam = unravel(lam_flat)

    _, pullback = jax.vjp(
            lambda p:
                staggered_step(
                    x_star,
                    p
                ),
        params,
    )

    param_bar = pullback(lam)[0]

    return (
        jax.tree.map(jnp.zeros_like, x_star),
        param_bar,
        None,
        None,
    )

solve_one_load_step.defvjp(
    solve_fwd,
    solve_bwd,
)

