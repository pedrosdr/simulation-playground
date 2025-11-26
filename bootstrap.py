#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
n = 100
lbd = 2
x = np.random.exponential(1/lbd, n)
med = np.log(2)/lbd
plt.hist(x, bins=10)

# %%
lbd_ = 1/x.mean()
med_ = np.log(2)/lbd_

# %%
N = 100000
boot = np.random.exponential(1/lbd_, size=[N, n])

# %%
b_lbd = 1/np.mean(boot, axis=1)
b_med = np.log(2)/b_lbd

# %%
plt.hist(b_med, bins=30)
b_med.mean()

# %%
print('Actual median: ', med)
print('Estimated median: ', med_)
print('Bootstrap median: ', b_med.mean())
print('Median standard errors: ', b_med.std())
print('Actual lambda: ', lbd)
print('Estimated lambda: ', lbd_)
print('Bootstrap lambda: ', b_lbd.mean())
print('Lambda standard errors: ', b_lbd.std())

# %%
