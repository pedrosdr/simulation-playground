#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
part = np.linspace(0, 1.0, 3)
part

#%%
abs_part = np.max(part[1:] - part[:-1])
abs_part

# %%
# Considerando que com n=10.000, |part|->0
for size in [10_000, 100_000, 1_000_000]:
    part = np.linspace(0, 1, size)
    f = np.zeros_like(part)
    mask = part > 0
    f[mask] = part[mask]*np.cos(1/part[mask])
    V = np.sum(np.abs(np.diff(f)))
    QV = np.sum(np.abs(np.diff(f))**2)
    print(V)
    print(QV)
plt.plot(part, f)

# %%
