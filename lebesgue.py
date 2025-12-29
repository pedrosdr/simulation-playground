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
bins = np.linspace(0, 3, 6)
bins

# %%
ind = np.empty([len(x), len(bins)-1])
c = np.empty(len(bins)-1)
mu = np.empty(len(bins)-1)
dx = x[1] - x[0]
for i in range(len(bins)-1):
    print(bins[i+1])
    mask = (bins[i] < y) & (y < bins[i+1])
    c[i] = bins[i]
    ind[:,i] = mask
    mu[i] = mask.sum() * dx
s = ind@c
#%%
print('Real AUC: ', np.trapz(y, x))
print('Lebesgue: ', c@mu)

# %%
fig, ax = plt.subplots()
ax.axhline(0, color='black', linestyle='dashed')
ax.axvline(0, color='black', linestyle='dashed')
ax.axvline(1, color='black', linestyle='dashed')
ax.fill_between(x, s, color="#e8e8e8", edgecolor="#9f9f9f")
ax.plot(x, y, color='black')

# %%
