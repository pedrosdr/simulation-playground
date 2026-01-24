#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
