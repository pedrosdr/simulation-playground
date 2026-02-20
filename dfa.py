#%%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# %%
N = 100
x = np.random.randn(N+10)
x = np.cumsum(x)[10:]
m = np.mean(x)

#%%
plt.plot(x)
print(m)

#%% (1) Dispersion profile
y = np.cumsum(x-m)
plt.plot(y)
# %%
