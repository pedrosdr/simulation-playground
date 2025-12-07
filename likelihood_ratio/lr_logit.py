#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def Sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid(x):
    return Sigmoid(x)*(1-Sigmoid(x))

# %%
N = 100
X = np.concatenate([
    np.ones([N, 1]),
    np.random.normal(
        loc  =[0.3, 4, 10],
        scale=[0.1, 1, 4],
        size=[N, 3]
    )
], axis=1)
beta = np.array([5, 0.4, -9, 3])

p = Sigmoid(X@beta)
y = np.where(p > 0.5, 1.0, 0.0)

# %%
z = np.linspace(-10, 10, 1000)
plt.plot(z, Sigmoid(z))
# %%
