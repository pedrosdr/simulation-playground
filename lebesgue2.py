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

fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=300)
ax.plot(x, y, color='black')
ax.set_ylabel(r'$f\,(\omega)$')
ax.set_xlabel(r'$\omega$')

# %%
ns = [3, 5, 10, 100]

fig, axs = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True, dpi=300)
for i, (n, ax) in enumerate(zip(ns, axs.ravel())):
    sets = pd.DataFrame({
        'x': x,
        'y': y,
        'set': pd.cut(y, bins=n, right=False)
    })
    sets['s'] = sets.groupby('set')['y'].transform('max')

    ax.set_title(f'$n={n}$', pad=12)
    ax.fill_between(sets['x'], sets['s'], alpha=0.08, color='black', edgecolor='none')
    ax.plot(sets['x'], sets['s'], alpha=0.4, color='black', label=r'$\varphi_{' + str(n) + r'}\,(\omega)$')
    ax.plot(sets['x'], sets['y'], color='black', label=r'$f\,(\omega)$')
    ax.set_xlabel('$\omega$')
    ax.legend()
    ax.set_ylabel('Valor da função')

# %%
n=100
fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)
sets = pd.DataFrame({
    'x': x,
    'y': y,
    'set': pd.cut(y, bins=n, right=False)
})
sets['s'] = sets.groupby('set')['y'].transform('max')

dx = sets["x"].iloc[1] - sets["x"].iloc[0]
L = float((sets["s"] * dx).sum())

ax.plot(sets['x'], sets['s'], alpha=0.4, color='black', label=r'$\varphi_{' + str(n) + r'}\,(\omega)$')
ax.plot(sets['x'], sets['y'], color='black', label=r'$f\,(\omega)$')
ax.set_xlabel('$\omega$')
ax.set_ylabel('Valor da função')
print(f"AUC={L}")