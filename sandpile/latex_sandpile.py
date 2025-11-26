#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
N = 100
zc = 4
T = 50000
z = np.zeros([N, N])
u = np.zeros([N, N])

#%%
def plot_avalanches(u, epoch):
    _, ax = plt.subplots()
    ax.imshow(u, cmap='gray_r', vmin=0, vmax=1)
    legend = ax.legend()
    legend.remove()
    ax.set_title(f'Época {epoch}')
    plt.show()
    plt.close()

#%%
sizes = np.zeros(T)
for epoch in range(T):
    i, j = np.random.randint(0, N, size=2)
    z[i,j] += 1.0
    while np.any(z >= zc):
        coords = np.argwhere(z >= zc)
        ii, jj = coords[:,0], coords[:,1]

        nbrs = np.vstack([
            np.column_stack([ii+1, jj]),
            np.column_stack([ii-1, jj]),
            np.column_stack([ii, jj+1]),
            np.column_stack([ii, jj-1])
        ])
        on = (
            (nbrs[:,0]< N) & (nbrs[:,0]>=0) &
            (nbrs[:,1]< N) & (nbrs[:,1]>=0)
        )
        nbrs = nbrs[on]
        np.add.at(z, (nbrs[:,0], nbrs[:,1]), zc/4)
        z[ii, jj] -= zc
        np.add.at(u, (ii, jj), 1.0)
    u = np.where(u>0, 1, 0)
    sizes[epoch] = u.sum()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}")
        plot_avalanches(u, epoch)

    u = np.zeros([N, N])
# %%
