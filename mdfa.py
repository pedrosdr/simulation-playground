#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def gln(x, lbd):
    if lbd==0:
        return np.log(x)
    return (x**lbd-1)/lbd

def yeo_johnson(x, lbd):
    y = np.zeros_like(x)
    mask = x >= 0
    y[mask] = gln(1+x[mask], lbd)
    y[~mask] = -gln(1-x[~mask], 2-lbd)
    return y

# %%
x = np.random.chisquare(3, 1000)
plt.hist(x)
# %%
y = yeo_johnson(x, -0.4)
plt.hist(y)

#%%
x = np.linspace(0, 20, 1000)
for lbd in np.linspace(-2, 2, 5):
    plt.plot(x, yeo_johnson(x, lbd), label=f'$\lambda={lbd}$')
plt.legend()

#%%
x = np.linspace(0, 20, 1000)
for lbd in np.linspace(-2, 2, 5):
    plt.plot(x, gln(x, lbd), label=f'$\lambda={lbd}$')
plt.legend()

#%%
x = np.linspace(-5, 5, 1000)
y = yeo_johnson(np.exp(x), -6)
plt.plot(x, y)

#%%
x = np.linspace(-5, 5, 1000)
y = 0.5*x+2
plt.plot(x, y)

#%%
gmean = (np.mean(y)-2)*2
gmean
# %%
