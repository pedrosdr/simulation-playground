#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from statsmodels.api import OLS

# %%
N, K = 1000, 3
sigma = 0.5
beta = np.array([5.0, 0.08, 5.0, 0.2])
X = np.random.normal([3, 0.2, 20], [0.5, 0.04, 5], size=[N, K])
X = np.concatenate([
    np.ones((N,1)), X
], axis=1)
y = X@beta + np.random.normal(0, sigma, N)

#%%
fig, axs = plt.subplots(3, 1, figsize=(7, 7))
for i, ax in enumerate(axs.ravel()):
    ax.scatter(y, X[:,i+1], color='black')

# %%
X_h1 = X
X_h0 = X[:,[0,2,3]]

#%%
beta_hat_h1 = np.linalg.inv(X_h1.T@X_h1)@X_h1.T@y
beta_hat_h0 = np.linalg.inv(X_h0.T@X_h0)@X_h0.T@y

# %%
u_hat_h1 = y - X_h1@beta_hat_h1
u_hat_h0 = y - X_h0@beta_hat_h0

#%%
sigma_sq_hat_h1 = u_hat_h1.T@u_hat_h1/N
sigma_sq_hat_h0 = u_hat_h0.T@u_hat_h0/N

#%%
def likelihood(n, sigma_sq, u):
    return -(n/2)*np.log(sigma_sq) - (u.T@u)/(2*sigma_sq)

#%%
ll_h1 = likelihood(N, sigma_sq_hat_h1, u_hat_h1)
ll_h0 = likelihood(N, sigma_sq_hat_h0, u_hat_h0)
lbd = -2*(ll_h0 - ll_h1)

#%%
pvalue = 1-chi2.cdf(lbd, df=1)
print(f'LR={lbd}, pval={pvalue}')

# %%
sm_model = OLS(y, X).fit()
sm_model.summary()

# %%
