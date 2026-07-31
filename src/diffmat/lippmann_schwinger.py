import jax
from jax import numpy as jnp
import numpy as np
import functools


@functools.partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "grid_spec",
        "u_in",
        "tolerance",
        "maxiter",
    ),
)
def solve(b_rhs, a, grid_spec, u_in=None, tolerance=1e-6, maxiter=1000):
    """Lippmann-Schwinger iteration for the scalar second order problem

        -Laplace u(x) + a(x)*u(x) = b_rhs(x)

    :arg grid: specification of computational grid
    :arg b_rhs: right hand side (field) (1, Nx, Ny, Nz)
    :arg a: ceofficient of zero order term (field) (1, Nx, Ny, Nz)
    :arg u_in: initial value of solution u
    :arg tolerance: tolerance for convergence check
    :arg maxiter: maximal number of iterations
    """
    dtype = b_rhs.dtype

    # Reference parameter A_0
    a_ref = 0.5 * (jnp.min(a) + jnp.max(a))

    # Laplacian operator in Fourier space
    n_vec = (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    L_vec = (grid_spec.Lx, grid_spec.Ly, grid_spec.Lz)
    hsq = (np.asarray(L_vec) / np.asarray(n_vec)) ** 2
    # Normalised momentum vectors in all three spatial directions
    # Grid with normalised momentum vectors
    xi = np.meshgrid(*[2 * np.pi * np.arange(n) / n for n in n_vec], indexing="ij")
    # Grid with xi*xi (laplacian in Fourier space)
    xixi = np.sum(
        2.0 * (np.cos(xi) - 1.0) / np.expand_dims(hsq, axis=(1, 2, 3)), axis=0
    )
    laplacian = xixi.astype(dtype)

    def exit_condition(state):
        """Check exit condition

        Check whether the relative difference
           ||chi^{k+1} - chi^k||_2 / ||chi^{k+1}||_2 > tolerance or iter > maxiter
        """
        iter, rel_difference = state[2:]
        return (rel_difference > tolerance) & (iter < maxiter)

    def loop_body(state):
        """Update phase-field, polarisation field, and compute residual"""
        chi_k, iter, rel_difference = state[1:]

        # Compute new phase field with polarisation field
        u_new = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(chi_k) / (a_ref - laplacian)))

        # Update the polarisation field
        chi_new = b_rhs - (a - a_ref) * u_new

        # Convergence test based on the L-2 norm over the unit cell
        norm_diff = jnp.linalg.norm(chi_new - chi_k)
        norm_chi_new = jnp.linalg.norm(chi_new)

        rel_difference = jnp.where(norm_chi_new > 1e-12, norm_diff / norm_chi_new, 0.0)

        return (u_new, chi_new, iter + 1, rel_difference)

    # Initialising variables for the first iteration (at k=0)

    # Set initial residual to 1
    residual = jnp.array(1.0, dtype=dtype)

    if u_in is None:
        u_0 = jnp.zeros_like(b_rhs)
    else:
        u_0 = u_in

    # Execute the fixed-point iteration loop
    u_final, _, iter, __ = jax.lax.while_loop(
        exit_condition,
        loop_body,
        init_val=(u_0, b_rhs - (a - a_ref) * u_0, 0, residual),
    )
    jax.lax.cond(
        iter >= maxiter,
        lambda x: jax.debug.print(
            "JAX forward solver failed to converge after {:6d} iterations",
            x,
            ordered=True,
        ),
        lambda x: None,
        maxiter,
    )
    return u_final


def solve_fwd(b_rhs, a, grid_spec, u_in, tolerance, maxiter):
    """Forward solve

    :arg b_rhs: right hand side (field) (1, Nx, Ny, Nz)
    :arg a: ceofficient of zero order term (field) (1, Nx, Ny, Nz)
    :arg grid_spec: specification of computational grid
    :arg u_in: initial value of solution u
    :arg tolerance: tolerance for convergence check
    :arg maxiter: maximal number of iterations
    """
    out = solve(b_rhs, a, grid_spec, u_in=u_in, tolerance=tolerance, maxiter=maxiter)
    return out, (a, out)


def solve_bwd(grid_spec, u_in, tolerance, maxiter, res, gradients):
    """Forward solve

    :arg b_rhs: right hand side (field) (1, Nx, Ny, Nz)
    :arg a: ceofficient of zero order term (field) (1, Nx, Ny, Nz)
    :arg grid_spec: specification of computational grid
    :arg u_in: initial value of solution u
    :arg tolerance: tolerance for convergence check
    :arg maxiter: maximal number of iterations
    :arg res: result of forward solve
    :arg gradients: gradient with respect to solution u
    """
    (a, u) = res
    g_u = gradients
    b_rhs_ad = -g_u
    Theta = solve(
        b_rhs_ad, a, grid_spec, u_in=None, tolerance=tolerance, maxiter=maxiter
    )
    g_b_rhs = -Theta
    g_a = Theta * u
    return g_b_rhs, g_a


solve.defvjp(solve_fwd, solve_bwd)
