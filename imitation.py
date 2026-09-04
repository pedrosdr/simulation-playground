#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
sigma = 0.2
k = 0.5
T = 2000
dt = 1/T

def h(t):
    A = 1.0
    B = 0.1
    tc = 1.0
    omega = 9.0
    phi = 3.0
    m = -0.3

    x = tc-t
    P1 = A*x**m
    P2 = B*x**m * np.cos(omega*np.log(x) + phi)
    return P1 + P2

p = np.ones(T)
plt.plot(h(np.linspace(0.001, 0.99, T)))

# %%
crashed = False
for i in range(1, T):
    t = dt*(i-1)
    dw = np.random.normal(0, np.sqrt(dt))

    if not crashed:
        dj = np.array(np.random.rand()<h(t)*dt, dtype=float)
    else:
        dj = 0.0

    if dj > 0:
        crashed = True

    dp = (k*h(t)*dt + sigma*dw - k*dj)*p[i-1]
    p[i] = p[i-1]+dp 

#%%
plt.plot(np.log(p))

#%%
