#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
T = 10000
N = 5
# r = np.zeros([N, T+10])
coef = np.random.uniform(0, 0.5, N)
sigma = np.random.uniform(0, 1.0, N)
icpt = np.random.uniform(1, 5, N)
# for i in range(1,T+10):
#     r[:,i] = coef*r[:,i-1] + np.random.normal(0, 1, N)*sigma + icpt
# r = r[:,10:]

r = np.random.uniform(0, 1, [N, T]) * sigma.reshape(-1, 1) + icpt.reshape(-1,1)

#%%
plt.plot(np.arange(T), r[0])

# %%
S = 10000
props = np.random.normal(0, 1, [S, N])
props = props / props.sum(axis=1, keepdims=True)

#%%
pfs = props@r
std_pfs = np.std(pfs, axis=1)
mean_pfs = np.mean(pfs, axis=1)
mask = (mean_pfs < 10) & (std_pfs < 0.2)

mean_pfs = mean_pfs[mask]
std_pfs = std_pfs[mask]

#%%
sharpe = (mean_pfs-2)/std_pfs
index = np.argmax(sharpe)

mean_max = mean_pfs[index]
std_max = std_pfs[index]


#%%
rfx = [0, std_max]
rfy = [2.0, mean_max]

# %%
plt.scatter(std_pfs, mean_pfs)
plt.scatter(std_max, mean_max, color='red')
plt.plot(rfx, rfy, color='red')

# %%
x = np.random.normal(2, np.sqrt(2), 10000)
y = x*0.3
y = y*(np.sqrt(3)/y.std())

# %%
x.var()
# %%
y.var()
# %%
np.var(x+y)

# %%
np.cov(x, y)
# %%
