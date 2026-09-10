
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Dict, List, Optional

def eng2lame(E, nu):
    lmbda = E*nu / (1.+nu) / (1. - 2.*nu)
    mu = E / 2 / (1. + nu)
    return lmbda, mu


def newton_raphson(
    residual_fn: Callable,
    u0: np.ndarray,
    maxiter: int = 20,
    tol: float = 1e-4,
    reg_init: float = 1e-6,
    reg_min: float = 1e-12,
    reg_max: float = 1.0,
    reg_increase_factor: float = 10.0,
    reg_decrease_factor: float = 0.9,
    damp_init: float = 1.0,
    damp_min: float = 1e-3,
    damp_max: float = 1.0,
    damp_increase_factor: float = 1.2,
    damp_decrease_factor: float = 0.5,
    line_search_trials: int = 10,
    line_search_step_size_factor: float = 0.5,
    line_search_armijo_factor: float = 1e-4,
    guarded_step_size: float = 1e-2,
    verbose: int = 1,
    dtype: type = jnp.float64,
) -> tuple:
    """
    Newton-Raphson solver.

    Solves: minimize ||r(u)|| where r(u) is the residual vector.

    Uses normal equations: (J^T J + reg*I) delta = -J^T r
    with backtracking line search and adaptive regularization/damping.

    Parameters:
    -----------
    residual_fn : Callable
        Function that computes residual vector: r = residual_fn(u)

    u0 : array
        Initial parameter guess, shape (n_params,)

    maxiter : int
        Maximum number of iterations (default: 20)

    tol : float
        Convergence tolerance on residual norm (default: 1e-4)

    reg_init : float
        Initial regularization parameter (default: 1e-6)

    reg_min : float
        Minimum regularization value (default: 1e-12)

    reg_max : float
        Maximum regularization value (default: 1.0)

    reg_increase_factor : float
        Factor to multiply reg when line search fails (default: 10.0)

    reg_decrease_factor : float
        Factor to multiply reg when step succeeds (default: 0.9)

    damp_init : float
        Initial damping factor (step size scaling) (default: 1.0)

    damp_min : float
        Minimum damping factor (default: 1e-3)

    damp_max : float
        Maximum damping factor (default: 1.0)

    damp_increase_factor : float
        Factor to multiply damp when step succeeds (default: 1.2)

    damp_decrease_factor : float
        Factor to multiply damp when step fails (default: 0.5)

    line_search_trials : int
        Number of backtracking trials in line search (default: 10)

    line_search_step_size_factor : float
        Factor to reduce step size in backtracking (default: 0.5)

    line_search_armijo_factor : float
        Armijo condition factor (currently not used, for future extension) (default: 1e-4)

    guarded_step_size : float
        Step size when line search fails (default: 1e-2)

    verbose : int
        Verbosity level:
        - 0: silent
        - 1: iteration summary (residual, condition number)
        - 2: detailed (singular values, step acceptance)
        (default: 1)

    dtype : type
        JAX dtype for computations (default: jnp.float64)

    Returns:
    --------
    u_opt : array
        Optimized parameter vector, shape (n_params,)

    history : dict
        Convergence history with keys:
        - 'res_norm': list of residual norms at each iteration
        - 'u': list of parameter vectors at each iteration
        - 'svd': list of singular value arrays (when computed)
        - 'alpha': list of damping factors used at each iteration
        - 'converged': bool, whether solver converged
        - 'n_iterations': number of iterations completed
        - 'final_residual_norm': final residual norm
        - 'initial_residual_norm': initial residual norm
        - 'reduction_factor': ratio of initial to final residual norm

    Examples:
    ---------
    # Simple example: fit two parameters
    def residual_fn(u):
        pred = forward_model(u, grid, matID)
        return (pred - target) / target_norm

    u_opt, history = newton_raphson(
        residual_fn,
        u0=initial_guess,
        maxiter=20,
        tol=1e-4,
        verbose=1,
    )

    # Example with custom line search and regularization
    u_opt, history = newton_raphson_inverse(
        residual_fn,
        u0=initial_guess,
        maxiter=15,
        tol=1e-5,
        reg_init=1e-8,
        damp_init=0.5,
        line_search_trials=20,
        verbose=2,
    )
    """

    dtype = jnp.asarray(u0, dtype=dtype).dtype
    u = jnp.asarray(u0, dtype=dtype)

    jitted_residual = jax.jit(residual_fn)
    jitted_jacobian = jax.jit(jax.jacobian(residual_fn))

    history = {
        "res_norm": [],
        "u": [],
        "svd": [],
        "alpha": [],
    }

    reg = reg_init
    damp = damp_init

    if verbose >= 1:
        print("=" * 80)
        print("Newton-Raphson Inverse Solver")
        print("=" * 80)
        print(f"Max iterations:           {maxiter}")
        print(f"Convergence tolerance:    {tol:.6e}")
        print(f"Initial regularization:   {reg_init:.6e}")
        print(f"Initial damping:          {damp_init:.6e}")
        print(f"Verbosity level:          {verbose}")
        print("=" * 80)
        print()

    converged = False
    for k in range(maxiter):
        if verbose >= 1:
            print(f"[Iter {k}] Computing residual...")

        r = jitted_residual(u)
        r_norm = jnp.linalg.norm(r)
        history["res_norm"].append(float(r_norm))
        history["u"].append(np.array(u))
        if verbose >= 1:
            print(f"[Iter {k}] residual norm = {r_norm:.6e}")

        if r_norm < tol:
            converged = True
            if verbose >= 1:
                print(f"✓ Converged at iteration {k}!")
            break

        if verbose >= 1:
            print(f"[Iter {k}] Computing Jacobian...")
        J = jitted_jacobian(u)  # shape (m, p)

        
        try:    # Compute singular values for diagnostics
            sv = jnp.linalg.svd(J, compute_uv=False)
            cond = float(sv[0] / (sv[-1] + 1e-30))
            history["svd"].append(np.array(sv))

            if verbose >= 2:
                print(f"  Singular values: min={sv[-1]:.3e}, max={sv[0]:.3e}")
            if verbose >= 1:
                print(f"  Condition number: {cond:.3e}")
        except Exception as e:
            if verbose >= 1:
                print(f"  SVD computation failed: {e}")

        if verbose >= 2:
            print(f"  Solving normal equations with reg={reg:.3e}...")
        JTJ = J.T @ J
        rhs = -J.T @ r
        JTJ_reg = JTJ + reg * jnp.eye(JTJ.shape[0], dtype=dtype)
        try:
            delta_u = jnp.linalg.solve(JTJ_reg, rhs)
        except Exception as e:
            if verbose >= 1:
                print(f"  Direct solve failed: {e}, falling back to lstsq")
            delta_u, *_ = jnp.linalg.lstsq(J, -r, rcond=None)

        if verbose >= 2:
            print(f"  Starting line search with initial step size: {damp}")
        alpha = damp
        success = False
        r_norm_current = r_norm
        for trial in range(line_search_trials):
            u_candidate = u + alpha * delta_u
            r_new = jitted_residual(u_candidate)
            r_new_norm = jnp.linalg.norm(r_new)

            if r_new_norm < r_norm_current:
                success = True
                if verbose >= 1:
                    print(f"  ✓ Step accepted: alpha={alpha:.6e}, new residual {r_new_norm:.6e}")

                u = u_candidate
                history["alpha"].append(float(alpha))

                # Adapt regularization (decrease) and damping (increase)
                reg = jnp.clip(reg * reg_decrease_factor, reg_min, reg_max)
                damp = jnp.clip(damp * damp_increase_factor, damp_min, damp_max)

                if verbose >= 2:
                    print(f"    Adapted: reg={reg:.3e}, damp={damp:.3e}")

                break
            else:
                if verbose >= 2:
                    print(f"    Trial {trial+1}: alpha={alpha:.6e} rejected (residual {r_new_norm:.6e})")
                alpha *= line_search_step_size_factor

        if not success:
            if verbose >= 1:
                print(f"  ✗ Line search exhausted; increasing regularization and taking guarded step")

            history["alpha"].append(0.0)

            # Adapt regularization (increase) and damping (decrease)
            reg = jnp.clip(reg * reg_increase_factor + 1e-12, reg_min, reg_max)
            damp = jnp.clip(damp * damp_decrease_factor, damp_min, damp_max)

            # Take small guarded step
            u = u + guarded_step_size * delta_u

            if verbose >= 2:
                print(f"    Adapted: reg={reg:.3e}, damp={damp:.3e}, step_size={guarded_step_size:.3e}")

        if verbose >= 1:
            print()

    final_residual_norm = history["res_norm"][-1]
    initial_residual_norm = history["res_norm"][0]
    reduction_factor = (
        initial_residual_norm / (final_residual_norm + 1e-30)
        if final_residual_norm > 0
        else float('inf')
    )

    history["converged"] = converged
    history["n_iterations"] = len(history["res_norm"])
    history["final_residual_norm"] = final_residual_norm
    history["initial_residual_norm"] = initial_residual_norm
    history["reduction_factor"] = reduction_factor

    if verbose >= 1:
        print("=" * 80)
        print("CONVERGENCE SUMMARY")
        print("=" * 80)
        print(f"Iterations completed:       {history['n_iterations']}")
        print(f"Converged:                  {converged}")
        print(f"Initial residual norm:      {initial_residual_norm:.6e}")
        print(f"Final residual norm:        {final_residual_norm:.6e}")
        print(f"Reduction factor:           {reduction_factor:.6e}")
        print("=" * 80)

    return u, history

