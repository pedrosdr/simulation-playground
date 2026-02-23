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
# Quadratic Variation
x = np.linspace(0, 1, 100_000)
y = 0.5*x
plt.plot(x, y)

V = np.sum(np.abs(np.diff(y)))
print('Variation =', V)
QV = np.sum(np.square(np.diff(y)))
print('Quadratic variation = ', QV)

# %%
# Brownian Motion
# The quadratic variation of a Brownian
# motion in the interval [a, b] is b-a
t = np.linspace(0, 2, 100_000)
dt = t[1]-t[0]
B = np.cumsum(np.random.randn(len(t))*np.sqrt(dt))
plt.plot(t, B)

V = np.sum(np.abs(np.diff(B)))
print('Variation =', V)
QV = np.sum(np.square(np.diff(B)))
print('Quadratic variation = ', QV)

#%%
