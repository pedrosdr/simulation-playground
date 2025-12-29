#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# %%
@njit
def garch11_arma11(omega, alpha, beta, mu, phi, theta, n):
    n += 50
    z = np.random.normal(0, 1, n)
    sigma_sqr = np.empty_like(z)
    sigma_sqr[0] = omega/(1-alpha-beta)
    sigma_sqr[0] = np.maximum(sigma_sqr[0], 1e-8)
    epsilon = np.empty_like(sigma_sqr)
    epsilon[0]=np.sqrt(sigma_sqr[0])*z[0]
    s = np.empty_like(epsilon)
    s[0] = mu/(1 - phi)
    for i in range(1, n):
        sigma_sqr[i] = omega + alpha*epsilon[i-1]**2 + beta*sigma_sqr[i-1]
        sigma_sqr[i] = np.maximum(sigma_sqr[i], 1e-8)
        epsilon[i] = np.sqrt(sigma_sqr[i])*z[i]
        s[i] = mu + phi*s[i-1] + epsilon[i] + theta*epsilon[i-1]
    
    return np.sqrt(sigma_sqr)[50:], s[50:]
    
garch11_arma11(0.1, 0.05, 0.94, 0, 0.5, 0.2 ,10)

# %%
sigma, s = garch11_arma11(
    omega = 0.1,
    alpha = 0.05,
    beta = 0.94,
    mu = 0.0,
    phi = 0.5,
    theta = 0.2,
    n=1000
)
plt.plot(s)
plt.plot(sigma)

# %%
plt.plot(np.cumsum(s))

# %%
