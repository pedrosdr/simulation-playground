#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import cauchy

# %%
mu = 1.2
sigma = 0.5
t = np.linspace(0, 1.5, 1000)
f = cauchy.pdf(t, loc=mu, scale=sigma)
F = cauchy.cdf(t, loc=mu, scale=sigma)
h = f/(1-F)

#%%
plt.plot(t, h)
# %%
