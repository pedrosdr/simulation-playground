#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
def dist(n, mu_0, P):
    if n <= 0:
        return mu_0
    
    mu_i = mu_0@P
    for _ in range(n-1):
        mu_i = mu_i@P

    return mu_i

# %%
P = np.array([
    [1/3, 1/3, 1/3],
    [1/2, 0, 1/2],
    [3/10, 2/10, 1/2]
], dtype=float)

mu_0 = np.array([1, 0, 0], dtype=float)

#%%
P.T@P.T@mu_0

# %%
P = np.array([
    [0, 1/2, 0, 1/2],
    [1/2, 0, 1/2, 0],
    [0, 1/2, 0, 1/2],
    [1/2, 0, 1/2, 0]
], dtype=float)

mu_0 = np.array([1, 0, 0, 0], dtype=float)

# %%
P.T@mu_0

# %%
# Gothenburg Weather:
P = np.array([
    [0.75, 0.25],
    [0.25, 0.75]
], dtype=float)
mu_0 = np.array([1, 0], dtype=float)

# %%
# Los Angeles Weather
P = np.array([
    [0.5, 0.5],
    [0.1, 0.9]
], dtype=float)
mu_0 = np.array([1/6, 5/6], dtype=float)

P.T@P.T@P.T@mu_0
dist(60, mu_0, P)

#%%
vals, vecs = np.linalg.eig(P)

#%%
vecs@np.diag(vals)@np.linalg.inv(vecs)

# %%
# Gothenburg weather (seasonal)
seasons = np.arange(start=1, stop=13)

ind_summer = ((seasons >= 5) & (seasons <= 9))
ind_summer

# %%
P_summer = np.array([
    [0.75, 0.25, 0],
    [0.25, 0.75, 0],
    [0.5, 0.5, 0]
], dtype=float)

P_winter = np.array([
    [0.5, 0.3, 0.2],
    [0.15, 0.7, 0.15],
    [0.2, 0.3, 0.5]
], dtype=float)

mu_0 = np.array([1, 0, 0], dtype=float)

# %%
mu_0@P_summer@P_winter

#%%
P_winter.T@P_summer.T@mu_0

#%%
mu_0[None, :]@P_summer@P_winter

#%%
import networkx as nx

# %%
g = nx.from_numpy_array(np.array([
    [1,2,0,0],
    [3,4,0,0],
    [0,0,3,4],
    [0,0,1,3],
]), create_using=nx.MultiDiGraph)
nx.draw(g)

# %%
