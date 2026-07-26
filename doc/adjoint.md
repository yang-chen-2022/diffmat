# Adjoint solve

## Primary equation
Consider the equation
$$
-\nabla\cdot(K(x)\nabla u(x)) + a(x)u(x) = b(x).
$$
which can be rewritten as
$$
\mathcal{D}_0 u(x) = \chi(x)
$$
where
$$
\chi(x) = b(x) - (a(x)-a_0) u(x) + \nabla\cdot((K(x)-K_0)\nabla u(x))
$$
and the differential operator $\mathcal{D}_0$ is defined by
$$
\mathcal{D}_0 u(x) = a_0 u(x) - K_0 \Delta u(x)
$$
Let $-\Gamma_0$ be the Greens function of $\mathcal{D}_0$. In Fourier space this is given by $a_0+K_0 \xi\cdot\xi$. Then the Lippmann-Schwinger iteration is given by
$$
u(x) = -\int_\Omega \Gamma^0(x-y) \chi(y)\;dy
$$
or short $u = -\Gamma^0* \chi$ where $\chi$ depends on $u$ as above.

## Adjoint equation
Let $\mathcal{J}=\mathcal{J}(u(K,a,b))$ be the objective function. We want to compute the functional derivatives
$$
\frac{\delta J}{\delta K(x)},\quad
\frac{\delta J}{\delta a(x)},\quad
\frac{\delta J}{\delta u(x)}
$$
under the assumption that $u$ satisfies the primary equation above.

Define
$$
\mathcal{A}(u;K,a,b) = u + \Gamma^0 * \left( b - (a-a_0) u + \nabla\cdot((K-K_0)\nabla u)\right)
$$
and introduce the Lagrange multiplier $\Lambda$. Then
$$
\mathcal{L}(u,\Lambda;K,a,b) := \mathcal{J}(u) + \int_\Omega \Lambda(z)\mathcal{A}(u;K,a,b)(z)\; dz
$$
We require
$$
\frac{\delta \mathcal{L}}{\delta u(x)}=0,
$$
which guarantees that $u$ satifies the primary equation.

Compute
$$
\begin{aligned}
\frac{\delta}{\delta u(x)} \int_\Omega \Lambda(z) \mathcal{A}(u;K,a,b)(z)\;dz
&= \frac{\delta}{\delta u(x)}\left\{
    \int_\Omega \Lambda(z)u(z)\;dz
    +\int_\Omega \Lambda(z)\Gamma^0(z-y)b(y)\;dz
    -\int_\Omega \int_\Omega \Lambda(z) \Gamma^0(z-y) (a(y)-a_0)u(y)\;dy\;dz
    + \int_\Omega \int_\Omega \Lambda(z) \Gamma^0(z-y)\nabla_y\left((K(y)-K_0) u(y)\right)\;dy\;dz\right\}\\
    &= \Lambda(x) - (a(x) - a_0) \int_\Omega \Lambda(z)\Gamma^0(z-x)\;dz+ \frac{\delta}{\delta u(x)}\int_\Omega \int_\Omega u(y) \nabla_y \left((K(y)-K_0)\nabla_y\Lambda(z)\Gamma^0(z-y)\right)\;dy\;dz \\
    &= \Lambda(x) - (a(x) - a_0)\int_\Omega \Lambda(z)\Gamma^0(z-x) \;dz + \int_\Omega \nabla_x\left((K(x)-K_0)\nabla_x \Lambda(z) \Gamma^0 (z-x)\right)\;dz\\
    &= \Lambda(x) - (a(x)-a_0) (\Gamma^0 * \Lambda)(x) + \nabla \left((K(x)-K_0) \nabla (\Gamma^0 * \Lambda)(x)\right)
\end{aligned}
$$
which leads to the adjoint equation for $\Lambda$
$$
\Lambda - (a-a_0) \Gamma^0 * \Lambda + \nabla\cdot \left((K-K_0) \nabla (\Gamma^0*\Lambda)\right) = -\mathcal{E} := -\frac{\delta\mathcal{J}}{\delta u(x)}
$$

## Rewriting in primary form
Defining $\Theta := -\Gamma^0 * \Lambda$, this can be rewritten in the same form as the primary equation, namely
$$
\mathcal{D}_0 \Theta = \chi^{\text{ad}}
$$
with
$$
\chi^{\text{ad}} := \mathcal{E} - (a-a_0)\Theta + \nabla\left((K-K_0)\nabla\Theta \right)
$$
Hence, we can use the same Lippmann-Schwinger iteration as for $u$, provided we replace $b$ with $\mathcal{E}$. Finally, $\Lambda$ can be computed from $\Theta$ as
$$
\Lambda = \mathcal{D}_0 \Theta = a_0 \Theta - K_0 \Delta \Theta.
$$

## Derivatives with respect to input parameters
We find for the derivatives of the objective function with respect to $a(x)$, $b(x)$ and $K(x)$:
$$
\begin{aligned}
\frac{\delta \mathcal{J}}{\delta a(x)} &= -\Theta(x)u(x)\\[2ex]
\frac{\delta \mathcal{J}}{\delta b(x)} &= \Theta(x)\\[2ex]
\frac{\delta \mathcal{J}}{\delta K(x)} &= - (\nabla \Theta(x))\cdot \nabla u(x)
\end{aligned}
$$
