#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def solution1(x, c, r):
    x = np.asarray(x).reshape(-1, 1)
    c = np.asarray(c).reshape(1, -1)
    r = np.asarray(r).reshape(1, -1)
    return np.sum(c*x**r, axis=1)

#%%
x = np.linspace(0, 10, 100)

#%%
y = solution1(x, [2, 3, 4], [-0.2, 0.3, 0.4])
plt.plot(x, y)


# %%
