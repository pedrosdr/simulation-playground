#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
def pdf(x, loc, scale):
    return (1/scale) * np.exp(-(x-loc)**2/scale)

# %%
x = np.linspace(-10, 10)
p = pdf(x, 0, 1.0)

#%%
plt.plot(x, p)