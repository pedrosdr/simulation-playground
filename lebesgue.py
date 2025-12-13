#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
x = np.linspace(0, 1, 10000)
y = np.sin((2*np.pi/0.5)*x) + 1.5

# %%
fig, ax = plt.subplots()
ax.plot(x, y, color='black')
ax.axhline(0, color='black', linestyle='dashed')
ax.axvline(0, color='black', linestyle='dashed')
ax.axvline(1, color='black', linestyle='dashed')
ax.fill_between(x, y, color='none', edgecolor="#9f9f9f", hatch='....')

#%%
n = 40
binwidth = np.random.uniform(0.1, 1, n)
binwidth = binwidth/np.sum(binwidth)
bins = np.empty(n+1)
bins[0] = 0.0
for i in range(len(binwidth)):
    bins[i+1] = bins[i] + binwidth[i]
bins

# %%
ind = np.empty([len(x), len(bins)-1])
c = np.empty(len(bins)-1)
mu = np.empty(len(bins)-1)
for i in range(len(bins)-1):
    mask = (bins[i] <= x) & (x < bins[i+1])
    c[i] = np.min(y[mask])
    ind[:,i] = mask
    mu[i] = bins[i+1]-bins[i]
ind[-1,-1] = 1
s = ind@c

#%%
print('Real AUC: ', np.trapz(y, x))
print('Lebesgue: ', c@mu)

# %%
fig, ax = plt.subplots()
for ci, bi in zip(c, bins[1:]):
    ax.plot([bi, bi], [0, ci], color='#9f9f9f')
ax.axhline(0, color='black', linestyle='dashed')
ax.axvline(0, color='black', linestyle='dashed')
ax.axvline(1, color='black', linestyle='dashed')
ax.fill_between(x, s, color="#e8e8e8", edgecolor="#9f9f9f")
ax.plot(x, y, color='black')

# %%
