import numpy as np
import jax
import functools
import pytest
from jax import numpy as jnp
from jax.test_util import check_vjp
from jaxmaterials.common import GridSpec
from diffmat.lippmann_schwinger import solve

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
