#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(1)

#%%
N = 2
fig, ax = plt.subplots()
p = 0.2 
data = np.random.choice([1,0], [N, N], p=[p, 1-p])
grid = ax.imshow(data)

#%%
N = 20
fig, ax = plt.subplots()
p = np.linspace(0, 1, N)
data = np.random.choice([1,0], [20, 20], p=[p[0], 1-p[0]])
grid = ax.imshow(data)

def update(p):
    data = np.random.choice([1,0], [20, 20], p=[p, 1-p])
    ax.imshow(data)
    any_x = np.any(np.sum(data, axis=1)==N)
    any_y = np.any(np.sum(data, axis=0)==N)

    ax.set_title(f'p={p:.2f},  spanning={any_x or any_y}')
    return grid

ani = FuncAnimation(fig, update, frames=p)
plt.show()
    