#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

# %%
a, b = 0, 1
def f(x):
    return 4*np.sqrt(1-x**2)

#%%
N = 10**4
x = np.random.uniform(a, b, [20000, N])
y = f(x)

plt.scatter(x[0], y[0])
plt.axhline(y[0].mean(), linestyle='--')
plt.axvline(a)
plt.axvline(b)

# %%
area = (b-a)*y.mean(axis=1)
plt.hist(area, 30)
jarque_bera(area)

# %%
area.std()
# %%
