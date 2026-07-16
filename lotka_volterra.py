import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = [0.5]
y = [0.5]
t = [0]
alpha = 0.05
beta = 0.05
gamma = 0.05
delta = 0.05
T = 1000

fig, ax = plt.subplots()
ax.set_ylim(0, 4)
ax.set_xlim(0, T)
line1, = ax.plot(t, x, label='prey')
line2, = ax.plot(t, y, label='predator')
ax.legend()

def update(frame):
    t.append(frame+1)
    dx = alpha*x[frame] - beta*x[frame]*y[frame]
    dy = -gamma*y[frame] + delta*x[frame]*y[frame]
    x.append(x[frame]+dx)
    y.append(y[frame]+dy)
    line1.set_data(t, x)
    line2.set_data(t, y)

ani = FuncAnimation(fig, update, interval=10)
plt.show()
    