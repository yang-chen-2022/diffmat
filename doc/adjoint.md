# Adjoint solve

## Primary equation
Consider the equation
$$
\mathcal{D}u(x) = -\nabla\cdot(K(x)\nabla u(x)) + a(x)u(x) = b(x).
$$
which is solved in the domain $[0,L]^d$ with periodic boundary conditions in all dimensions.

This can be rewritten as
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
Let $\Gamma_0$ be the Greens function of $\mathcal{D}_0$, i.e.
$$
\mathcal{D}_0\mathcal{G}_0(x-y) = \delta(x-y).
$$
Note that $\mathcal{G}_0$ only depends on the difference $x-y$ (and in fact on $\|x-y\|_2$), and that $\mathcal{G}_0(x-y) = \mathcal{G}_0(y-x)$.
In Fourier space the Greens function is given by
$$
\widehat{\mathcal{G}}_0(\xi) =\widehat{\mathcal{G}}_0(\|\xi\|_2) = \frac{1}{a_0+K_0 \|\xi\|_2^2}.
$$
With this the Lippmann-Schwinger iteration is given by
$$
u(x) = \int_\Omega \mathcal{G}_0(x-y) \chi(y)\;dy
$$
or short $u = \mathcal{G}_0* \chi$ where $\chi$ depends on $u$ as above.

## Adjoint equation
Let $\mathcal{J}=\mathcal{J}(u(K,a,b))$ be the objective function. We want to compute the functional derivatives
$$
\frac{\delta \mathcal{J}}{\delta K(x)},\quad
\frac{\delta \mathcal{J}}{\delta a(x)},\quad
\frac{\delta \mathcal{J}}{\delta u(x)}
$$
under the assumption that $u$ satisfies the primary equation above.

Define
$$
\mathcal{A}(u;K,a,b) = u - \mathcal{G}_0 * \left( b - (a-a_0) u + \nabla\cdot((K-K_0)\nabla u)\right)
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
    -\int_\Omega\int_\Omega \Lambda(z)\mathcal{G}_0(z-y)b(y)\;dy\;dz
    +\int_\Omega \int_\Omega \Lambda(z) \mathcal{G}_0(z-y) (a(y)-a_0)u(y)\;dy\;dz
    - \int_\Omega \int_\Omega \Lambda(z) \mathcal{G}_0(z-y)\nabla_y\left((K(y)-K_0) \nabla_y u(y)\right)\;dy\;dz\right\}\\
    &= \Lambda(x) + (a(x) - a_0) \int_\Omega \Lambda(z)\mathcal{G}_0(z-x)\;dz- \frac{\delta}{\delta u(x)}\int_\Omega \int_\Omega u(y) \nabla_y \left((K(y)-K_0)\nabla_y\Lambda(z)\mathcal{G}_0(z-y)\right)\;dy\;dz \\
    &= \Lambda(x) + (a(x) - a_0)\int_\Omega \Lambda(z)\mathcal{G}_0(z-x) \;dz - \int_\Omega \nabla_x\left((K(x)-K_0)\nabla_x \Lambda(z) \mathcal{G}_0 (z-x)\right)\;dz\\
    &= \Lambda(x) + (a(x)-a_0) (\mathcal{G}_0 * \Lambda)(x) - \nabla \left((K(x)-K_0) \nabla (\mathcal{G}_0 * \Lambda)(x)\right)
\end{aligned}
$$
which leads to the adjoint equation for $\Lambda$
$$
\Lambda + (a-a_0) \mathcal{G}_0 * \Lambda - \nabla\cdot \left((K-K_0) \nabla (\mathcal{G}_0*\Lambda)\right) = \mathcal{E} := -\frac{\delta\mathcal{J}}{\delta u(x)}
$$

## Rewriting in primary form
Defining $\Theta := \mathcal{G}_0 * \Lambda$, this can be rewritten in the same form as the primary equation, namely
$$
\mathcal{D}_0 \Theta = \chi^{\text{ad}}
$$
with
$$
\chi^{\text{ad}} := \mathcal{E} - (a-a_0)\Theta + \nabla\cdot\left((K-K_0)\nabla\Theta \right)
$$
Hence, we can use the same Lippmann-Schwinger iteration as for $u$, provided we replace $b$ with $\mathcal{E}$. Finally, $\Lambda$ can be computed from $\Theta$ as
$$
\Lambda = \mathcal{D}_0 \Theta = a_0 \Theta - K_0 \Delta \Theta.
$$

## Derivatives with respect to input parameters
We find for the derivatives of the objective function with respect to $a(x)$, $b(x)$ and $K(x)$:
$$
\begin{aligned}
\frac{\delta \mathcal{J}}{\delta a(x)} &= \Theta(x)u(x)\\[2ex]
\frac{\delta \mathcal{J}}{\delta b(x)} &= -\Theta(x)\\[2ex]
\frac{\delta \mathcal{J}}{\delta K(x)} &= (\nabla \Theta(x))\cdot (\nabla u(x))
\end{aligned}
$$
