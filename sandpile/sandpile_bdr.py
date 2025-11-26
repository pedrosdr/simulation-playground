#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colormaps
from numba import njit

#%%
N = 100
Kc = 4
n_frame_iter=20
z = np.zeros([N, N])
fig, ax = plt.subplots()

surf = ax.imshow(z, cmap='PuOr', vmin=0, vmax=Kc)

#%%
def update(frame):
    n_frame_iter = 1500 if frame<120 else 7
    count = 0
    while count < n_frame_iter:
        count += 1
        if np.any(z >= Kc):
            coords = np.argwhere(z >= Kc)
            ii, jj = coords[:,0], coords[:,1]
            drop = np.random.uniform(0, Kc)

            nbrs = np.vstack([
                np.column_stack([ii+1, jj]),
                np.column_stack([ii-1, jj]),
                np.column_stack([ii, jj+1]),
                np.column_stack([ii, jj-1]),
                np.column_stack([ii-1, jj-1]),
                np.column_stack([ii+1, jj+1]),
                np.column_stack([ii+1, jj-1]),
                np.column_stack([ii-1, jj+1])
            ])
            nbrs[nbrs[:,0] == N, 0] = 0
            nbrs[nbrs[:,0] == -1, 0] = N-1
            nbrs[nbrs[:,1] == N, 1] = 0
            nbrs[nbrs[:,1] == -1, 1] = N-1

            np.add.at(z, (nbrs[:,0], nbrs[:,1]), drop/8)
            z[coords[:,0], coords[:,1]] -= drop
        else:
            # i, j = np.random.randint(0, N, size=2)
            i = j = int(N/2)
            z[i,j] += 1.0

    surf.set_data(z)
    print(frame)
    return (surf,)

ani = animation.FuncAnimation(fig, update, interval=1)
plt.show()
# %%
