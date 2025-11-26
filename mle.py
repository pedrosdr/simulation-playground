#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.distributions import ECDF
from scipy.stats import gamma

# %%
x = np.random.gamma(5, 2, 10000)
plt.hist(x, bins=50, ec='black', color="#d1d1d1")

#%%
ecdf = ECDF(x)
plt.plot(
    np.linspace(0, 40), ecdf(np.linspace(0, 40)), color='black'
)
plt.grid()

#%%
def nll(args, x):
    a, scale = args

    if a <= 0 or scale <= 0:
        return np.inf

    return -np.sum(
        gamma.logpdf(x, a=a, scale=scale)
    )

#%%
minimize(
    nll,
    x0=(1, 1),
    args=(x,),
    method='Nelder-Mead'
)

# %%
