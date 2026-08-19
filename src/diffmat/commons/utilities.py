
def eng2lame(E, nu):
    lmbda = E*nu / (1.+nu) / (1. - 2.*nu)
    mu = E / 2 / (1. + nu)
    return lmbda, mu
