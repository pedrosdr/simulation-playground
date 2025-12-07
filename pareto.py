#%%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

# %%
def check_pareto(x0, alpha):
    if x0 <= 0:
        raise ValueError(
            'x0 must be greater than 0 (x0 > 0).'
        )
    if alpha <=0:
        raise ValueError(
            'alpha must be greater than 0 (alpha > 0).'
        )

def pareto_pdf(x, x0, alpha):
    check_pareto(x0, alpha)
    x = np.asarray(x)
    a = alpha*(x0**alpha)
    b = x**(-(alpha+1))
    return np.where(x < x0, 0.0, a*b)

def pareto_cdf(x, x0, alpha):
    check_pareto(x0, alpha)
    x = np.asarray(x)
    return np.where(
        x < x0, 0.0,
        1-(x0/x)**alpha
    )

def rand_pareto(x0, alpha, size):
    check_pareto(x0, alpha)
    u = np.random.uniform(0, 1, size)
    return x0*(1-u)**(-(1/alpha))

def mle_pareto(x):
    x = np.asarray(x)
    if np.any(x<=0):
        raise ValueError('x must be positive (x > 0)')
    
    n = len(x)
    x0 = np.min(x)
    alpha = n/np.sum(np.log(x/x0))

    x_ = rand_pareto(x0, alpha, size=[5000, n])
    x0s = np.min(x_, axis=1, keepdims=True)
    alphas = n/np.sum(np.log(x_/x0s), axis=1)
    x0s = x0s.ravel()

    return (
        (x0, alpha), 
        {
            'conf_int': (
                (np.quantile(x0s, 0.025), np.quantile(x0s, 0.975)),
                (np.quantile(alphas, 0.025), np.quantile(alphas, 0.975))
            ),
            'std_err': (x0s.std(), alphas.std())
        }
    )

# %%
x = np.linspace(-3, 10, 1000)
plt.plot(x, pareto_cdf(x, 1, 0.2))

# %%
x0 = 1.0
alpha = 0.17

u = rand_pareto(x0, alpha, 10_000)

# Exemplo de plot em log-log com corte de quantil
x_max = np.quantile(u, 0.99)
bins = np.logspace(np.log10(x0), np.log10(x_max), 50)
plt.hist(u, bins=bins, density=True, alpha=0.5)

xs = np.logspace(np.log10(x0), np.log10(x_max), 500)
plt.plot(xs, pareto_pdf(xs, x0, alpha))

plt.xscale('log')
plt.yscale('log')
plt.xlim(x0, x_max)
plt.show()

# %%
stats = mle_pareto(u)
stats

# %%
