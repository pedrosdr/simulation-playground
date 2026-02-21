#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from math import factorial

# %%
T = 1000
lbd = 20
t = np.linspace(0, 1, T)
dt = t[1]-t[0]
incr = poisson.ppf(
    np.random.rand(T),
    mu = lbd*dt 
)
n = incr.cumsum()
plt.plot(t, n)
# %%
