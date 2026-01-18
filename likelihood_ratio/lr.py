#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, chi2

#%%
def add_constant(x):
    return np.concatenate([
        np.ones([x.shape[0], 1]),
        x
    ], axis=1)

# %%
n = 300
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
def loglikelihood(params, X, y, restrict=(), h0=()):
    k = X.shape[1]
    full_params = np.zeros(2*k)
    full_params[list(restrict)] = list(h0)
    mask = np.ones(2*k, dtype=bool)
    mask[list(restrict)] = False
    full_params[mask] = params
    beta = full_params[:k]
    theta = full_params[k:]
    mu = X@beta
    sigma = np.sqrt(np.exp(X@theta))
    return norm.logpdf(
        y, loc=mu, scale=sigma
    ).sum()

# %%
lkl_free = -minimize(
    lambda params: -loglikelihood(params, X, y),
    x0 = np.zeros(X.shape[1]*2),
    method='BFGS'
).fun

lkl_res = -minimize(
    lambda params: -loglikelihood(params, X, y, restrict=(1, 5), h0=(2, -0.8)),
    x0 = np.zeros(X.shape[1]*2 - 2),
    method='BFGS'
).fun

lkl_ratio = -2*(lkl_res-lkl_free)

p_val = chi2.sf(lkl_ratio, df=2)
print(f'p-valor: {p_val:.4f}')
print(f'Likelihood ratio = {lkl_ratio}')

# %%
