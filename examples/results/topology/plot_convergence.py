import numpy as np
import matplotlib.pyplot as plt

vfs = ['0.1', '0.2', '0.3']


# theoritical upper limit
E = 1.
nu = 0.3
Ks = E / (3. - 6.*nu)
Gs = E / 2. / (1.+nu)


#
ftsize = 16
plt.figure(figsize=(8, 5))

for vf in vfs:
    filename = f"to_vf{vf}/to_vf{vf}_convergence.txt"
    data = np.loadtxt(filename)
    plt.plot(data, "*-", linewidth=2, markersize=6, label=f"$\phi$={vf}")

    #theoretical value
    K = 4. * Gs * Ks * float(vf) / (4*Gs + 3*Ks*(1-float(vf)))
    plt.axhline(y=K, linestyle="--",linewidth=0.6, color="black")
    print(K)


plt.legend(fontsize=ftsize)
plt.xlabel("Iteration", fontsize=ftsize)
plt.ylabel("Builk modulus", fontsize=ftsize)
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
#plt.title("Topology Optimization Convergence")
plt.grid(True, alpha=0.3)
plt.savefig(
    f"to_convergence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
