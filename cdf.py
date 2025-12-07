#%%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF
from scipy.integrate import cumulative_trapezoid

# %%
def cdf_norm(x, loc=0.0, scale=1.0):
    A = 1/(scale*np.sqrt(2*np.pi))
    B = np.exp(-0.5*((x-loc)/scale)**2)
    return A*cumulative_trapezoid(B, x, initial=0)

# %%
x = np.linspace(6, 14, 1000)
plt.plot(x, cdf_norm(x, loc=10, scale=1))

#%%
X = np.random.normal(10, 1, 1000)
plt.plot(x, ECDF(X)(x))
plt.plot(x, cdf_norm(x, 10, 1))

#%%
Y = 2*X - 4
y = np.linspace(np.min(Y), np.max(Y))
plt.plot(y, ECDF(Y)(y))
plt.plot(y, cdf_norm((y+4)/2, 10, 1))

# %%
pdf_y = np.gradient(cdf_norm((y+4)/2, 10, 1), y)
plt.plot(y, pdf_y)