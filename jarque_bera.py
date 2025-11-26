#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.distributions.empirical_distribution import ECDF

# %%
def jb_stat(x):
    m3 = skew(x)
    m4 = kurtosis(x)
    return len(x)*(m3**2/6 + m4**2/24)
    
def jbtest(x):
    n = len(x)
    xsim = np.random.normal(0, 1, [100_000, n])
    m3 = skew(xsim, axis=1)
    m4 = kurtosis(xsim, axis=1)
    jb = n*(m3**2/6 + m4**2/24)
    ecdf = ECDF(jb)
    stat = jb_stat(x)
    return {
        'jb': stat,
        'p': 1-ecdf(stat)
    }


# %%
# x = np.random.normal(20, 3, 195)
# x = np.random.standard_t(5, 180)
x = np.random.chisquare(500, 30)
jbtest(x)

#%%
jarque_bera(x)
# %%
