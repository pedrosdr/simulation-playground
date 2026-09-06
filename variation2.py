#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.linspace(-10, 10, 100000)
y = np.sin(x)
plt.plot(x, y)

# %%
V = np.sum(np.abs(y[1:]-y[:-1]))
Q = np.sum(np.square(y[1:]-y[:-1]))
V, Q

# %%
T = 100_000
dt = 1/T
W = np.cumsum(
    np.random.normal(
        loc=0, scale=np.sqrt(dt), size=T
    )
)
plt.plot(W)

#%%
V = np.sum(np.abs(W[1:]-W[:-1]))
Q = np.sum(np.square(W[1:]-W[:-1]))
V,Q
# %%
