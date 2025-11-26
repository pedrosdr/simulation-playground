import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

N = 500
Kc = 4
z = np.zeros([N, N])
fig, ax = plt.subplots()
surf = ax.imshow(z, cmap='PuOr', vmin=0, vmax=Kc)

@njit
def update_array(z, Kc, N, n_frame_iter):
    for _ in range(n_frame_iter):
        found = False
        for i in range(N):
            for j in range(N):
                if z[i, j] >= Kc:
                    drop = np.random.uniform(0, Kc)
                    neighbors = []
                    # 8 vizinhos
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (1, -1), (-1, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < N:
                            neighbors.append((ni, nj))
                    n_nbrs = len(neighbors)
                    if n_nbrs > 0:
                        drop_per = drop / n_nbrs
                        for ni, nj in neighbors:
                            z[ni, nj] += drop_per
                    z[i, j] -= drop
                    found = True
        if not found:
            i, j = np.random.randint(0, N, size=2)
            z[i, j] += 1.0

def update(frame):
    n_frame_iter = 2000 if frame < 300 else 7
    update_array(z, Kc, N, n_frame_iter)
    surf.set_data(z)
    return (surf,)

ani = animation.FuncAnimation(fig, update, interval=1)
plt.show()
