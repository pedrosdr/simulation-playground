#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
def alpha(x, lbd, mu, sigma):
    A = mu
    B = (sigma**2*(1-lbd))/2
    if lbd == 0:
        return (mu+(sigma**2)/2)*x
    return A*x**(1-lbd) + B*x**(1-2*lbd)

def beta(x, lbd, sigma):
    if lbd == 0:
        return sigma*x
    return sigma*x**(1-lbd)


# %%
lbd = -3
mu = 0.1
sigma = 0.1

T = 1
N = 10_000
dt = T/N
x = np.ones(N)
for i in range(1, N):
    Alpha = alpha(x[i-1], lbd, mu, sigma)
    Beta = beta(x[i-1], lbd, sigma)
    dw = np.random.normal(0, np.sqrt(dt))
    dx = Alpha*dt + Beta*dw
    x[i] = x[i-1] + dx
plt.plot(np.linspace(0, T, N), x)

# %%
plt.plot(np.linspace(0, T, N-1), np.diff(x))

#%%
plt.hist(np.diff(x), bins=40)
# %%
