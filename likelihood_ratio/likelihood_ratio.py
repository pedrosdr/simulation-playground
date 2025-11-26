#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from statsmodels.distributions import ECDF

#%%
def ll(params, x):
    loc, scale = params
    return np.sum(norm.logpdf(x, loc=loc, scale=scale))

def nll(params, x):
    return -ll(params, x)

def ll_r(params, x):
    scale = params
    return np.sum(norm.logpdf(x, loc=0, scale=scale)) if scale > 0 else -np.inf

def nll_r(params, x):
    return -ll_r(params, x)

def mle(x, params0):
    return minimize(
        nll,
        x0 = params0,
        args=(x,),
        method='Nelder-Mead'
    ).x

def mle_r(x, params0):
    return minimize(
        nll_r,
        x0 = params0,
        args=(x,),
        method='Nelder-Mead'
    ).x

# %%
n = 200
x = np.random.normal(0.05, 2, size=n)
max_h0 = mle_r(x, (1, ))
logl_r = ll_r(max_h0, x)
logl = ll(mle(x, (0, 1)), x)
lbd = -2*(logl_r - logl)

# %%
N = 1000
sample_h0 = np.random.normal(0, max_h0[0], size=[N, n])

# %%
logl_r_boot = []
logl_boot = []
for i in range(N):
    logl_r_boot.append(ll_r(mle_r(sample_h0[i], (1,)), sample_h0[i]))
    logl_boot.append(ll(mle(sample_h0[i], (0,1)), sample_h0[i]))

#%%
logl_r_boot = np.array(logl_r_boot)
logl_boot = np.array(logl_boot)
lbd_boot = -2*(logl_r_boot - logl_boot)

# %%
plt.hist(lbd_boot, bins=20)

# %%
ecdf = ECDF(lbd_boot)
pval = 1-ecdf(lbd)
pval_theo = 1-chi2.cdf(lbd, df=1)
print(f'LR={lbd}, pval={pval}, pval(theory)={pval_theo}')

# %%
#----------------------------------------------------------
# Analytical Likelihood Ratio
#----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from statsmodels.distributions import ECDF

#%%
n = 500
x = np.random.normal(0.2, 2, size=n)

# %%
def log_likelihood(loc, scale, x):
    return np.sum(norm.logpdf(x, loc=loc, scale=scale))

# %%
max_ll_h0 = log_likelihood(0       , x.std(), x)
max_ll_h1 = log_likelihood(x.mean(), x.std(), x)
lbd = -(max_ll_h0 - max_ll_h1)
theoretical_pval = 1-chi2.cdf(lbd, df=1)
print(f'Likelihood Ratio={lbd}, p-value (theoretical)={theoretical_pval}')

# %%
N = 3000
x_boot = np.random.normal(0, x.std(), size=[N, n])
max_ll_h0_boot = np.empty(N)
max_ll_h1_boot = np.empty(N)
for i in range(N):
    max_ll_h0_boot[i] = log_likelihood(0               , x_boot[i].std(), x_boot[i])
    max_ll_h1_boot[i] = log_likelihood(x_boot[i].mean(), x_boot[i].std(), x_boot[i])
lbd_boot = -2*(max_ll_h0_boot - max_ll_h1_boot)
bootstrap_pval = 1-ECDF(lbd_boot)(lbd)
plt.hist(lbd_boot)

# %%
print(f'Likelihood Ratio       :  {lbd}')
print(f'p-value (theoretical)  :  {theoretical_pval}')
print(f'p-value  (Bootstrap)   :  {bootstrap_pval}')

# %%
