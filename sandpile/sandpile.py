#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# %%
N = 100
zc = 4
T = 200000
z = np.zeros([N, N])
u = np.zeros([N, N])

#%%
def plot_grid(u):
    fig, ax = plt.subplots()
    ax.imshow(u, cmap='gray_r', vmin=0, vmax=1)
    legend = ax.legend()
    legend.remove()
    plt.show()
    plt.close()

#%%
# Open
sizes = np.zeros(T)
means = np.zeros(T)
freqs = np.zeros([T, 4])
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
    means[epoch] = z.mean()
    freqs[epoch, 0] = np.sum(z==0)
    freqs[epoch, 1] = np.sum(z==1)
    freqs[epoch, 2] = np.sum(z==2)
    freqs[epoch, 3] = np.sum(z==3)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}")
        plot_grid(u)

    u = np.zeros([N, N])
#%%
# Fast Driven
p = 0.05
sizes = np.zeros(T)
means = np.zeros(T)
freqs = np.zeros([T, 4])
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

        if np.random.uniform(0, 1) > p:
            i, j = np.random.randint(0, N, size=2)
            z[i,j] += 1.0

    u = np.where(u>0, 1, 0)
    sizes[epoch] = u.sum()
    means[epoch] = z.mean()
    freqs[epoch, 0] = np.sum(z==0)
    freqs[epoch, 1] = np.sum(z==1)
    freqs[epoch, 2] = np.sum(z==2)
    freqs[epoch, 3] = np.sum(z==3)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}")
        plot_grid(u)

    u = np.zeros([N, N])

#%%
# Close
sizes = np.zeros(T)
means = np.zeros(T)
freqs = np.zeros([T, 4])
for epoch in range(T):
    i, j = np.random.randint(0, N, size=2)
    z[i,j] += 1.0
    counter = 0
    while np.any(z >= zc):
        counter += 1
        coords = np.argwhere(z >= zc)
        ii, jj = coords[:,0], coords[:,1]

        nbrs = np.vstack([
            np.column_stack([ii+1, jj]),
            np.column_stack([ii-1, jj]),
            np.column_stack([ii, jj+1]),
            np.column_stack([ii, jj-1])
        ]) 
        nbrs[nbrs[:,0]>=N, 0] = 0
        nbrs[nbrs[:,1]>=N, 1] = 0
        nbrs[nbrs[:,0]<0, 0] = N-1
        nbrs[nbrs[:,1]<0, 1] = N-1

        np.add.at(z, (nbrs[:,0], nbrs[:,1]), zc/4)
        z[ii, jj] -= zc
        np.add.at(u, (ii, jj), 1.0)
        u = np.where(u>0, 1, 0)
        if counter % 500 == 0:
            print(f"Epoch {epoch}, counter={counter}")
            plot_grid(u)
    counter = 0
    u = np.where(u>0, 1, 0)
    sizes[epoch] = u.sum()
    means[epoch] = z.mean()
    freqs[epoch, 0] = np.sum(z==0)
    freqs[epoch, 1] = np.sum(z==1)
    freqs[epoch, 2] = np.sum(z==2)
    freqs[epoch, 3] = np.sum(z==3)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}")
        plot_grid(u)

    u = np.zeros([N, N])
#%%
freqs = freqs/N**2
stats = [z, sizes, means, freqs]

#%%
import pickle as pkl
pkl.dump(stats, open('stats_100.pkl', 'wb'))

#%%
import pickle as pkl
z, sizes, means, freqs = pkl.load(open('stats_100.pkl', 'rb'))

#%%
sizes = sizes[sizes>0]
min_val = sizes.min()
max_val = sizes.max()
bins = np.logspace(np.log10(min_val), np.log10(max_val), num=20)

counts, edges = np.histogram(sizes, bins=bins, density=True)

bin_centers = (edges[:-1] + edges[1:]) / 2

plt.figure(figsize=(4.5, 4.5))
plt.plot(bin_centers, counts, color='black', label='Distribuição empírica')
ref_x = np.linspace(min_val, max_val, 100)
ref_y = ref_x**(-1.0) * (np.median(counts) * np.median(bin_centers) / (ref_x[0]**(-1.0)))
plt.plot(ref_x, ref_y, 'r--', label='Inclinação $\\tau=-1$')
plt.loglog()

plt.xlabel("Tamanho da avalanche")
plt.ylabel("Densidade")
plt.legend()
plt.show()

log_x = np.log10(bin_centers)
log_y = np.log10(counts)

slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
tau = -slope

print(f"=== RESULTADOS DO FIT ===")
print(f"Inclinação teórica (-1): {slope:.4f}")
print(f"Expoente Crítico (tau): {tau:.4f}")
print(f"R² (Qualidade do ajuste): {r_value**2:.4f}")

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
x = np.arange(N)
x, y = np.meshgrid(x, x)
surf = ax.plot_surface(x, y, z, color="#FFE5A7", shade=True)
fig.savefig('sandpile.png', dpi=200)

#%%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(means, color='black')
ax.set_ylabel('Altura média')
ax.set_xlabel('Iteração')
ax.grid()

#%%
plt.imshow(z, cmap='PuOr', vmin=0, vmax=4)
plt.colorbar(label='Altura da pilha')

# %%
plt.plot(freqs[:,0], label='Nenhum grão')
plt.plot(freqs[:,1], label='1 grão')
plt.plot(freqs[:,2], label='2 grãos')
plt.plot(freqs[:,3], label='3 grãos')
plt.ylabel('Frequência')
plt.xlabel('Iteração')
plt.legend()

# %%
plt.imshow(z)