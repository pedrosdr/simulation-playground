#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
x = np.linspace(1, 100, 10000)
N = 2**np.floor(np.log(x)/np.log(3))
D = np.log(2)/np.log(3)

#%%
def fourier(x, n):
    k = np.arange(-(n), n+1)[:, None]
    x = np.asarray(x)[None, :]
    c_k = 1.0/(2*(np.log(2) + 2j*np.pi*k))
    expo = 2j*np.pi*k*(np.log(x)/np.log(3))
    f = (x**D).ravel()*np.sum(c_k*np.exp(expo), axis=0)
    return f.real

#%%
plt.plot(x, N)
plt.plot(x, fourier(x, 100))

# %%
