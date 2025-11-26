#%%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

# %%
x = np.random.normal(0, 1, [1000, 130, 2])
x = np.concatenate([np.ones([1000, 130, 1]), x], axis=-1)

# %%
y = np.random.normal(0, 1, [1000, 130, 1])

# %%
xTx = x.transpose(0,2,1)@x
xTy = x.transpose(0,2,1)@y
beta = np.linalg.inv(xTx)@xTy
beta = beta.reshape(3, -1)

# %%
plt.hist(beta[2], 20)

# %%
# sample data
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")  # 3D axes
n = 200
x = np.random.randn(n)
y = np.random.randn(n)
z = x**2 + y + np.random.randn(n)*0.3

ax.scatter(x, y, z, s=20)
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("3D scatter")
plt.show()