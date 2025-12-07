import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from time import perf_counter

N = 1_000_000   # Número de iterações
mu = 0.05       # Drift
sigma = 0.1     # Desvio padrão

# Função para plotar a série
def plot_chart(t, X):
    plt.plot(t, X, color='#000000', lw=0.6)
    plt.ylabel('Preço do ativo')
    plt.xlabel('Tempo')
    plt.show()
    plt.close()

# Função para simular um movimento browniano geométrico
def generate_simulation(N, mu, sigma):
    X = np.empty(N)
    dt = 1.0 / N
    X[0] = np.random.normal(50, sigma)

    for i in range(1, N):
        dW = np.random.normal(0.0, np.sqrt(dt))
        dX = mu * X[i-1] * dt + sigma * X[i-1] * dW
        X[i] = X[i-1] + dX
    return X

# Decorando a função com o njit
generate_simulation_numba = nb.njit(generate_simulation)

# Chamando a função a primeira vez para que ela seja compilada
generate_simulation_numba(N, mu, sigma)

# Medindo o tempo da função em Python puro
time_start = perf_counter()
generate_simulation(N, mu, sigma)
time_end = perf_counter()
time_python = time_end - time_start
print(f'Tempo sem o Numba: {time_python} segundos')

# Medindo o tempo da função compilada
time_start = perf_counter()
generate_simulation_numba(N, mu, sigma)
time_end = perf_counter()
time_numba = time_end - time_start
print(f'Tempo com o Numba: {time_numba} segundos')

# Calcula a razão entre as durações (quanto o Numba é mais rápido)
ratio = time_python / time_numba
print(f'O Numba foi {ratio:.2f} vezes mais rápido que o Python puro')

# Plotando a série para visualização
X = generate_simulation_numba(N, mu, sigma)
t = np.linspace(0, 1, N)
plot_chart(t, X)
