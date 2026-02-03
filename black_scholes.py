#%%
import torch as th
import matplotlib.pyplot as plt

# %%
for E in range(1, 20, 4):
    r = 1.1**(1/252)-1
    sigma = 0.02
    u = (2*r)/(sigma**2)
    C = E*u/(u+1)
    S = th.linspace(C, 50, 1000)
    G = (E/(u+1))*(((u+1)*S)/(E*u))**(-u)
    plt.plot(S, G, label=f'E={E}')
    plt.legend()

# %%
