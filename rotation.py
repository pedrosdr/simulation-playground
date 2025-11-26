#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %%
fig, ax = plt.subplots()
ax.set_ylim(-2, 2)
ax.set_xlim(-2, 2)
ax.axhline(y=0, linestyle='--', color='#444444')
ax.axvline(x=0, linestyle='--', color='#444444')
x = np.random.uniform(0, 0.7, [10, 2])
dots = ax.scatter(x[:,0], x[:,1])
ang = np.pi/100
M1 = np.array([
    [np.cos(ang), -np.sin(ang)],
    [np.sin(ang), np.cos(ang)]
])
M2 = np.array([
    [1, 0.02],
    [0, 1]
])

#%%
def update(frame):
    global x, dots
    x = x@M1.T@M2.T + np.random.normal(0, 0.01, x.shape)
    dots.set_offsets(x)
    return (dots,)

#%%
ani = animation.FuncAnimation(fig, update, frames=60, interval=60)
plt.show()