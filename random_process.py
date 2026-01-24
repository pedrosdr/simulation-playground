#%%
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
T = 1000
sigma_sq = np.ones(T)
z = np.random.randn(T)
e = np.empty_like(z)
s = np.empty_like(z)

# Inicialização
e[0] = z[0] * np.sqrt(sigma_sq[0])
s[0] = e[0]

# Loop principal
for i in range(1, T):
    # Equação da variância condicional
    sigma_sq[i] = 0.05 + 0.9 * sigma_sq[i-1] + 0.05 * e[i-1]**2
    e[i] = z[i] * np.sqrt(sigma_sq[i])
    s[i] = np.sin(s[i-1]) + 0.9*s[i-1] + e[i]

# Plotagem: Retornos vs. Variância
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Gráfico dos Retornos (e)
ax1.plot(s, color='blue', alpha=0.7)
ax1.set_title(r'Processo simulado ($e_t$)')
ax1.set_ylabel('Processo')
ax1.grid(True, alpha=0.3)

# Gráfico da Variância (sigma^2)
ax2.plot(sigma_sq, color='red', alpha=0.7)
ax2.set_title(r'Variância Condicional ($\sigma^2_t$)')
ax2.set_ylabel('Volatilidade')
ax2.set_xlabel('Tempo')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%
