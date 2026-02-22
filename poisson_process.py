#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, bernoulli

# %%
T = 1000
lbd = 20
t = np.linspace(0, 1, T)
dt = t[1]-t[0]
incr = poisson.ppf(
    np.random.rand(T),
    mu = lbd*dt 
)
N = incr.cumsum()
plt.plot(t, N)

# %%
mu = 10
n = 1_000_000
p = mu/n
x = np.arange(n)
incr = np.random.choice(
    [1, 0], size=n, p=[p, 1-p]
)
N = np.cumsum(incr)
plt.plot(x,N)

# %%
