#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#%%
def run_M(adj, s0, noise, K, G):
    s = s0.copy()
    T = noise.shape[0]
    m = np.empty(T, dtype=float)
    for t in range(T):
        x = K*(adj @ s) + noise[t] + G
        s = np.where(x >= 0, 1, -1)
        m[t] = s.mean()
    return m.mean()

def susc(K, std=1.0, N=100, T=200, burn=50, eps=1e-2, R=50, seed=1):
    rng = np.random.default_rng(seed)
    vals = []
    for r in range(R):
        Gx = nx.barabasi_albert_graph(N, 1, seed=int(rng.integers(1<<32)))
        adj = nx.to_numpy_array(Gx)

        s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
        noise = std * rng.standard_normal((T, N))

        Mp = run_M(adj, s0, noise, K, +eps)
        Mm = run_M(adj, s0, noise, K, -eps)

        vals.append((Mp - Mm)/(2*eps))

    return float(np.mean(vals))

#%%
Ks = np.linspace(0, 4, 1000)
chi = np.array([susc(K) for K in Ks])

# %%
fig, ax = plt.subplots(dpi=300)
ax.plot(Ks, chi, color='black')
ax.set_ylabel('Suscetibility ($\chi$)')
ax.set_xlabel('K')

# %%
