import numpy as np

def oc(rho, dc, dv, ft_type, vf, kernel=None, move=0.2, tol=1e-6):
    """
    Optimality Criteria (OC) update for topology optimization.
    
    Implements the standard OC algorithm with move limits and volume constraint.
    
    Parameters
    ----------
    rho : ndarray
        Current density field
    dc : ndarray
        Objective sensitivity field
    dv : ndarray
        Volume sensitivity field
    ft_type : int
        Filter type (1 or 2)
    vf : float
        Target volume fraction
    kernel : ndarray, optional
        Filter kernel (required if ft_type==2)
    move : float
        Maximum per-iteration density change
    tol : float
        Binary-search tolerance for Lagrange multiplier
        
    Returns
    -------
    rho_new : ndarray
        Updated density field
    change : float
        Maximum absolute design-variable change
    """
    x = rho.copy()
    l1, l2 = 0.0, 1e9
    Hs = kernel.sum() if ft_type == 2 else 0.0
    
    while (l2 - l1) > tol:
        lmid = 0.5 * (l1 + l2)
        
        # OC update: rho_new = rho * sqrt(-dc / (dv * lambda))
        dr = np.abs(-dc / (dv * lmid))
        x_trial = x * np.sqrt(dr)
        
        # Apply move limits
        xnew = np.clip(x_trial, x - move, x + move)
        xnew = np.clip(xnew, 0.0, 1.0)
        
        # Apply filter
        if ft_type == 1:
            rho_candidate = xnew
        elif ft_type == 2:
            rho_candidate = periodic_convolve(xnew, kernel) / Hs
        
        # Check volume constraint
        if rho_candidate.mean() > vf:
            l1 = lmid
        else:
            l2 = lmid
    
    change = np.max(np.abs(xnew - x))
    return rho_candidate, change



