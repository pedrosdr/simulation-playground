#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm

# %%
def shannon_entropy(p):
    p = np.column_stack([p, 1-p])
    mask = p>0
    return -np.sum(np.where(mask, p*np.log(p), 0.0), axis=1)

def gini(p):
    p = np.column_stack([p, 1-p])
    return 1-np.sum(p**2, axis=1)

def q_log(x, q):
    if q == 1:
        return np.log(x)
    
    lbd = 1-q
    return (x**lbd-1)/lbd

def tsallis_entropy(p, q):
    p = np.column_stack([p, 1-p])
    mask = p>0
    return np.sum(np.where(mask, p*q_log(1/p, q), 0.0), axis=1)

def tsallis_diff_entropy(scales, q):
    I = []
    for s in np.atleast_1d(scales):
        x = np.linspace(-8*s, 8*s, 5000)
        pdf = norm.pdf(x, loc=0, scale=s)
        integrand = pdf * q_log(1/pdf, q)
        I.append(np.trapz(integrand, x))
    return np.array(I)

#%%
p = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(p, shannon_entropy(p), color='black', label='Entropia de Shannon')
ax.plot(p, shannon_entropy(p)/np.log(2), color='black', linestyle='--', label='Entropia de Shannon normalizada')
ax.set_ylabel('$H(X;p)$')
ax.set_xlabel('$p$')
ax.legend()

# %%
size=1000
scale = np.linspace(0.001, 10, size)
I = []
for s in scale:
    x = np.linspace(-8*s, 8*s, 5000)
    pdf = norm.pdf(x, loc=0, scale=s)
    integrand = -pdf * np.log(pdf) 
    I.append(np.trapz(integrand, x)) 

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(scale, I, color='black')
ax.set_ylabel('$H(X;\\sigma)$')
ax.set_xlabel('$\\sigma$')

#%%
size=1000
loc = np.linspace(-10, 10, size)
I = []
for m in loc:
    x = np.linspace(m-8, m+8, 5000)
    pdf = norm.pdf(x, loc=m, scale=1)
    integrand = -pdf * np.log(pdf)
    I.append(np.trapz(integrand, x))

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(loc, I, color='black')
ax.set_ylabel('$H(X;\\mu)$')
ax.set_xlabel('$\\mu$')

# %%
p = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(5.5,4.5))
ax.plot(p, shannon_entropy(p), color='black', label='Entropia de Shannon')
ax.plot(p, gini(p), color='black', linestyle='-.', label='Índice de Gini-Simpson')
ax.set_ylabel('$H(X;p)$ e $G(X;p)$')
ax.set_xlabel('$p$')
ax.legend()

# %%
p = np.linspace(0, 1, 1000)
max_shannon = np.log(2)
max_gini = (2-1)/2
fig, ax = plt.subplots(figsize=(5.5,4.5))
ax.plot(p, shannon_entropy(p)/max_shannon, color='black', label='Entropia de Shannon normalizada')
ax.plot(p, gini(p)/max_gini, color='black', linestyle='-.', label='Índice de Gini-Simpson normalizado')
ax.set_ylabel('$H_{norm}(X;p)$ e $G_{norm}(X;p)$')
ax.set_xlabel('$p$')
ax.legend()

# %%
x = np.linspace(0.005, 3, 1000)
fig, ax = plt.subplots(figsize=(7,5))
qs = [-2, -0.5, 0, 0.5, 1, 2]
linestyles = ['-.', (0, (1,1,1,1,5,1)), 'dotted', 'dashed', '-', (0, (7,1))]
labels = ['$q=-2$', '$q=-0.5$', '$q=0$', '$q=0.5$', '$q=1,\,\, (\ln x)$', '$q=2$']
for i,q in enumerate(qs):
    ax.plot(x, q_log(x, q), color='black', linestyle=linestyles[i], label=labels[i], lw=1.5)
ax.set_ylabel('q-logarítmo ($\ln_q x$)')
ax.set_xlabel('$x$')
ax.legend()
ax.set_ylim(-4, 6)

# %%
p = p[p!=1]
p = p[p!=0]
x = np.linspace(0.005, 3, 1000)
fig, ax = plt.subplots(figsize=(7,5))
qs = [-0.1, -0.05, 0, 0.5, 1, 2]
linestyles = ['-.', (0, (1,1,1,1,5,1)), 'dotted', 'dashed', '-', (0, (7,1))]
labels = ['$q=-2$', '$q=-0.5$', '$q=0$', '$q=0.5$', '$q=1,\,\,$ (Shannon)', '$q=2,\,\,$ (Gini-Simpson)']
for i,q in enumerate(qs):
    plt.plot(p, tsallis_entropy(p, q), label=labels[i], linestyle = linestyles[i], color='black')
ax.set_ylabel('$H_q(X)$')
ax.set_xlabel('$p$')
ax.legend()
ax.set_ylim(0, 2)

#%%
p = np.linspace(0, 1, 1000)
x = np.linspace(0.005, 3, 1000)
fig, ax = plt.subplots(figsize=(7,5))
qs = [0.02, 0.2, 1, 2, 20]
linestyles = ['-.', (0, (1,1,1,1,5,1)), '-', 'dashed', 'dotted']
labels = [
    f'$q={qs[0]}$',
    f'$q={qs[1]}$',
    f'$q={qs[2]},\,\,$ (Shannon)',
    f'$q={qs[3]},\,\,$ (Gini-Simpson)',
    f'$q={qs[4]}$'
]
for i,q in enumerate(qs):
    H_max = np.log(2) if q==1 else (1-2**(1-q))/(q-1)
    plt.plot(p, tsallis_entropy(p, q)/H_max, label=labels[i], linestyle = linestyles[i], color='black')
ax.set_ylabel('$H_{q,norm}(X)$')
ax.set_xlabel('$p$')
ax.legend()

#%%
size = 1000
scales = np.linspace(0.001, 10, size)
fig, ax = plt.subplots(figsize=(7,5))

qs = [0.02, 0.2, 1, 2, 20]
linestyles = ['-.', (0, (1,1,1,1,5,1)), '-', 'dashed', 'dotted']
labels = [
    f'$q={qs[0]}$',
    f'$q={qs[1]}$',
    f'$q={qs[2]}$',
    f'$q={qs[3]}$',
    f'$q={qs[4]}$'
]

for i, q in enumerate(qs):
    H = tsallis_diff_entropy(scales, q)
    ax.plot(scales, H, label=labels[i],
            linestyle=linestyles[i], color='black')

ax.set_ylabel(r'$H_q(X)$')
ax.set_xlabel(r'$\sigma$')
ax.legend()
ax.set_ylim(-4, 4.3)

# %%
