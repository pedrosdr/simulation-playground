#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 15, 1000)
y = np.where(
    ((x >= 0) & (x < 3)) | ((x >= 11) & (x <= 15)), 1, np.where(
    (x >= 3) & (x < 5),                             2, np.where(
    (x >= 5) & (x < 11),                          1.5, 0
)))
plt.plot(x, y)

# %%
x = np.linspace(0, 10, 1000)
# y = 2.5-0.1*(x-5)**2
y = 1+np.sin(x)
plt.plot(x, y)

# %%
sets = pd.DataFrame({
    'x': x,
    'y': y,
    'set': pd.cut(y, bins=4, right=False)
})
sets['s'] = sets.groupby('set')['y'].transform('mean')
plt.fill_between(sets['x'], sets['s'], alpha=0.2)
plt.plot(sets['x'], sets['y'])
# %%
