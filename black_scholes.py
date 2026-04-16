#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
r = 0.05
sigma = 0.2
Es = [5, 10, 20]
linestyles = ['dotted', '-.', 'solid']
fig, ax = plt.subplots(dpi=300, figsize=(6,4))
for i, E in enumerate(Es):
    u = (2*r)/(sigma**2)
    C = E*u/(u+1)
    S = np.linspace(C, 60, 1000)
    G = (E/(u+1))*(((u+1)*S)/(E*u))**(-u)
    ax.plot(
        S, G, label=f'$K={E}$',
        linestyle=linestyles[i],
        color='black'
    )
    ax.set_ylabel('Put price $(V)$')
    ax.set_xlabel('Stock price $(S)$')
    plt.legend()

# %%
# Legenda em português
r = 0.05
sigma = 0.2
Es = [5, 10, 20]
linestyles = ['dotted', '-.', 'solid']
fig, ax = plt.subplots(dpi=500, figsize=(6,4))
for i, E in enumerate(Es):
    u = (2*r)/(sigma**2)
    C = E*u/(u+1)
    S = np.linspace(C, 60, 1000)
    G = (E/(u+1))*(((u+1)*S)/(E*u))**(-u)
    ax.plot(
        S, G, label=f'$K={E}$',
        linestyle=linestyles[i],
        color='black'
    )
    ax.set_ylabel('Preço da opção $put$ $(V)$')
    ax.set_xlabel('Preço da ação $(S)$')
    plt.legend()
