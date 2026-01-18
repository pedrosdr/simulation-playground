#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats.distributions import norm

#%%
def add_constant(x):
    return np.concatenate([
        np.ones([x.shape[0], 1]),
        x
    ], axis=1)

# %%
n = 1000
X = add_constant(
    np.random.normal((5, 10), (0.2, 1), size=(n, 2))
)
beta_r = np.array([4.0, 2.0, -0.5])
theta_r = np.array([0.5, 1.2, -0.8])
mu_r = X@beta_r
sigma_r = np.sqrt(np.exp(X@theta_r))
e_r = np.random.randn(n)*sigma_r
y = mu_r + e_r

# %%
fig, axs = plt.subplots(1, 2)
for i, ax in enumerate(axs.ravel(), start=1):
    ax.scatter(X[:,i], e_r)

# %%
fig, axs = plt.subplots(1, 2)
for i, ax in enumerate(axs.ravel(), start=1):
    ax.scatter(X[:,i], y)

# %%
def loglikelihood(params, X, y):
    k = X.shape[1]
    beta = params[:k]
    theta = params[k:]
    mu = X@beta
    sigma = np.sqrt(np.exp(X@theta))
    return -norm.logpdf(
        y, loc=mu, scale=sigma
    ).sum()

#%%
params = np.zeros(X.shape[1]*2)

# %%
minimize(
    loglikelihood,
    x0 = params,
    args=(X, y)
)

# %%
