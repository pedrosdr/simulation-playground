#%%
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#%%
np.random.seed(0)

x = np.linspace(0, 1, 1000)

def func(x, tc, m):
    x = np.asarray(x, dtype=float)
    return (tc - x) ** m

#%%
tc_real = 1.2
m_real = -0.5

y_real = func(x, tc_real, m_real)
y = y_real + np.random.normal(0, 0.2, len(x))

#%%
plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=5, label="dados com ruído")
plt.plot(x, y_real, color="black", label="função real")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#%%
tc_values = np.linspace(1.01, 1.5, 200)
m_values = np.linspace(-1, 1, 200)

TC, M = np.meshgrid(tc_values, m_values)

#%%
loss = np.empty_like(TC)

for i in range(TC.shape[0]):
    for j in range(TC.shape[1]):
        tc = TC[i, j]
        m = M[i, j]

        y_pred = func(x, tc, m)
        loss[i, j] = np.mean((y - y_pred) ** 2)

#%%
best_idx = np.unravel_index(np.argmin(loss), loss.shape)

best_tc = TC[best_idx]
best_m = M[best_idx]
best_loss = loss[best_idx]

print("tc real:", tc_real)
print("m real:", m_real)
print()
print("tc estimado:", best_tc)
print("m estimado:", best_m)
print("MSE mínimo:", best_loss)

#%%
log_loss = np.log10(loss)

#%%
fig = go.Figure(
    data=[
        go.Surface(
            x=TC,
            y=M,
            z=log_loss
        )
    ]
)

fig.update_layout(
    title="Superfície de erro",
    scene=dict(
        xaxis_title="tc",
        yaxis_title="m",
        zaxis_title="log10(MSE)"
    )
)

fig.show()

#%%
fig = go.Figure(
    data=[
        go.Heatmap(
            x=tc_values,
            y=m_values,
            z=log_loss
        )
    ]
)

fig.update_layout(
    title="Mapa de calor do erro",
    xaxis_title="tc",
    yaxis_title="m"
)

fig.show()

#%%
y_best = func(x, best_tc, best_m)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=5, label="dados com ruído")
plt.plot(x, y_real, color="black", label="função real")
plt.plot(x, y_best, color="red", linestyle="--", label="melhor ajuste")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#%%