#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
p = 0.5
M = 10_000
N = 100
C = -5
x = np.random.choice(
    (1, -1), p=(p, 1-p), size=(M, N)
)
s = np.cumsum(x, axis=1)

# %%
fig, ax = plt.subplots()
for y in s[:10]:
    ax.plot(np.arange(N)+1, y, lw=1)
ax.axhline(C, label=rf'$C={C}$', lw=2, color='black', linestyle='--')
ax.legend()
ax.set_ylabel(r'Processo ($S_n$)')
ax.set_xlabel('Iteração ($n$)')

# %%
hit = s == C
hit

# %%
teve_hit = hit.any(axis=1)
tau = hit.argmax(axis=1) + 1
tau = tau[teve_hit]
tau

#%%
valores, contagens = np.unique(tau, return_counts=True)

fig, ax = plt.subplots()
ax.bar(valores, contagens, width=0.8, color='black')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Tempo de ruína ($\tau$)')
ax.set_ylabel('Número de ocorrências')
plt.show()

# %%
taus = np.arange(1, N + 1)
num = np.bincount(tau[tau != -1], minlength=N + 1)[1:]
den = (s == C).sum(axis=0)
p_hat = np.divide(num, den, out=np.full(N, np.nan), where=(den > 0))

# teórico
p_theo = -C / taus
mask = ((taus + C) % 2 == 0) & (den > 0)

# %%
fig, ax = plt.subplots()

ax.bar(taus[mask], p_hat[mask], width=0.8, alpha=0.6, label='Empírico', color='black')
ax.plot(taus[mask], p_theo[mask], 'o-', lw=2, label=r'Teórico: $-C/\tau$', color='black')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'Tempo $\tau$')
ax.set_ylabel(r'$\mathrm{P}(S_{-C},\dots,S_{\tau-2}=\tau\mid S_\tau=C)$')
ax.legend()
plt.show()

# %%
from math import comb

taus = np.arange(1, N + 1)

# estimativa empírica de P(tau_C = tau)
num = np.bincount(tau, minlength=N + 1)[1:]
p_hat = num / M

# probabilidade teórica
p_theo = np.zeros_like(taus, dtype=float)

mask = ((taus + C) % 2 == 0) & (taus >= abs(C))
ks = ((taus[mask] + C) // 2).astype(int)

p_theo[mask] = [
    (-C / t) * comb(t, k) * (p ** k) * ((1 - p) ** (t - k))
    for t, k in zip(taus[mask], ks)
]

# %%
fig, ax = plt.subplots()

ax.bar(taus[mask], p_hat[mask], width=0.8, alpha=0.6, color='black', label='Estimado')
ax.plot(taus[mask], p_theo[mask], 'o-', color='black', lw=2, label='Teórico')

ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\mathrm{P}(T=\tau)$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
plt.show()
# %%
