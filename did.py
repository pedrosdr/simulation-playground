#%%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant

# %%
N = 1000
T_0 = np.random.normal(235, 3, N)
T_1 = np.random.normal(248, 5, N)
C_0 = np.random.normal(226, 3, N)
C_1 = np.random.normal(229, 3, N)
y = np.concatenate([T_0, T_1, C_0, C_1])
x = np.concatenate([
    np.random.normal(20, 3, 2*N),
    np.random.normal(12, 2, 2*N)
])
y = y+10*x

# %%
t = np.concatenate([
    np.zeros(N), np.ones(N),
    np.zeros(N), np.ones(N)
])

g = np.concatenate([
    np.ones(2*N), np.zeros(2*N)
])

gt = t*g

# %%
X = add_constant(np.stack([g, t, gt, x], axis=1))

# %%
OLS(y, X).fit().summary()
# %%
