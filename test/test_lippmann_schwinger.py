import numpy as np
import jax
import functools
import pytest
from jax import numpy as jnp
from jax.test_util import check_vjp
from jaxmaterials.common import GridSpec
from diffmat.fracture.lippmann_schwinger import solve

jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[[32, 16, 8], [33, 17, 9]], ids=["even", "odd"])
def grid_spec(request):
    """Return grid specification"""
    # Domain size in all three spatial direction
    Lx = 2.1
    Ly = 0.95
    Lz = 0.6
    # Number of grid cells in all three spatial directions
    nx, ny, nz = request.param

    return GridSpec(nx, ny, nz, Lx, Ly, Lz)


def compute_rhs(grid_spec, a, u):
    """Compute exact RHS as

        b_{rhs}^{exact} = -Laplace u(x) + a*u(x)

    :arg grid_spec: specification of computational grid
    :arg a: reaction coefficient field a(x)
    :arg u: exact solution field u(x)
    """
    b_rhs = a * u
    n_vec = (grid_spec.nx, grid_spec.ny, grid_spec.nz)
    L_vec = (grid_spec.Lx, grid_spec.Ly, grid_spec.Lz)
    hsq = (np.asarray(L_vec) / np.asarray(n_vec)) ** 2
    for dim in range(3):
        b_rhs += (2 * u - jnp.roll(u, +1, axis=dim) - jnp.roll(u, -1, axis=dim)) / hsq[
            dim
        ]
    return b_rhs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_solve(grid_spec, dtype):
    """Verify the Lippmann-Schwinger solves works as expected"""
    rng = np.random.default_rng(seed=241745)
    a = np.exp(
        -0.1 * rng.normal(size=(grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    )
    u_exact = rng.normal(size=(grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    b_rhs = compute_rhs(grid_spec, a, u_exact)
    u = solve(
        b_rhs,
        a,
        grid_spec,
        tol=1.0e-14,
        verbose=1,
    )
    rtol = 1.0e-10 if dtype == np.float64 else 1.0e-6
    diff = np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)
    assert diff < rtol


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adjoint(grid_spec, dtype):
    """Verify that derivative is computed correctly with adjoint method
    For this, compare to corresponding finite difference."""
    rng = np.random.default_rng(seed=241745)
    b_rhs = rng.normal(size=(grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    a = np.exp(
        -rng.normal(size=(grid_spec.nx, grid_spec.ny, grid_spec.nz)).astype(dtype)
    )

    def loss_fn(b_rhs, a):
        u_sol = solve(
            b_rhs,
            a,
            grid_spec,
            tol=1.0e-12 if dtype == np.float64 else 1.0e-6,
            verbose=1,
        )
        return jnp.sum(u_sol**2)

    rtol = 1.0e-9 if dtype == np.float64 else 1.0e-4
    check_vjp(
        loss_fn,
        functools.partial(jax.vjp, loss_fn),
        args=(b_rhs, a),
        rtol=rtol,
    )
