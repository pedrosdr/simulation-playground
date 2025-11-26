#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it

#%%

def hierarchical_diamond(levels: int, b: int = 2, s: int = 2, 
                         cross_edges: bool = False, 
                         multigraph: bool = False):
    G = nx.MultiGraph() if multigraph else nx.Graph()

    source, target = ("s0", "t0")
    G.add_edge(source, target)

    counter = 0
    def new_node():
        nonlocal counter
        counter += 1
        return f"n{counter}"

    for _ in range(levels):
        # Para MultiGraph, se quiser granularidade por aresta, use edges(keys=True)
        edges_to_replace = list(G.edges())
        G.remove_edges_from(edges_to_replace)

        for (u, v) in edges_to_replace:
            branch_internal_nodes = []
            for _branch in range(b):
                if s == 1:
                    G.add_edge(u, v)
                    continue

                internal = [new_node() for _ in range(s - 1)]
                path_nodes = [u] + internal + [v]
                nx.add_path(G, path_nodes)  # <-- corrigido

                if s == 2:
                    branch_internal_nodes.append(internal[0])

            if cross_edges and s == 2 and len(branch_internal_nodes) >= 2:
                for a, bnode in zip(branch_internal_nodes, branch_internal_nodes[1:]):
                    G.add_edge(a, bnode)

    return G


#%%
g = hierarchical_diamond(5)
nx.draw(g)

#%%
plt.hist(nx.to_numpy_array(g).sum(axis=1))

#%%
M = nx.to_numpy_array(g)
s = np.random.choice([-1.0, 1.0], size=len(g))

#%%
K = 0.00
std = 0.1

#%%
T = 200
ms = np.zeros(T)

for i in range(T):
    u = K*(M@s) + std*np.random.normal(size=s.shape)
    s = np.where(u > 0, 1.0, -1.0)
    ms[i] = s.mean()
    K += 0.001

#%%
plt.plot(np.arange(len(ms)), ms+1.0)
plt.loglog()

#%%