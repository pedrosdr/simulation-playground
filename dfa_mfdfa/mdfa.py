#%%
import torch as th
import matplotlib.pyplot as plt
import math

# %%
# Definindo os incrementos White Noise
N=10000
t = th.arange(N, dtype=th.float64)
incr = th.randn(t.shape[0], dtype=t.dtype)
plt.plot(t, incr)

# %%
# Definindo o Signal Profile
x = th.cumsum(incr-incr.mean(), dim=0)
plt.plot(t, incr)
plt.plot(t, x)

# %%
# Criando uma função para regressão polinomial
def polyfit(x, y, deg=2):
    degs = th.arange(deg+1)
    degs = degs.view(1,-1) if x.dim() == 1 else degs.view(1,1,-1)
    X = x.unsqueeze(-1)**degs
    betas = th.linalg.lstsq(X, y).solution.unsqueeze(-1)
    res = X@betas
    return res.squeeze(-1), betas

#%%
plt.plot(t, polyfit(t, x, deg=2)[0])
plt.plot(t, x)

#%%
# Calculando as flutuações para apenas uma escala
scale = 10
n_segments = math.floor(x.shape[0]/scale)
segments = x.unfold(0, scale, scale)
intervals = t.unfold(0, scale, scale)
trends, _ = polyfit(intervals, segments, deg=2)
dtr_segments = segments-trends
F = th.sqrt(th.mean(dtr_segments**2))

for itv, fit, seg in list(zip(intervals, trends, segments))[1:10]:
    plt.plot(itv, seg)
    plt.plot(itv, fit, color='gray')
plt.show()
plt.close()
for itv, seg in zip(intervals, dtr_segments):
    plt.plot(itv, seg)

# %%
# Definindo uma função para o cálculo das flutuações
def fluctuation(scale, x, q=2):
    scale = int(scale)
    t = th.arange(x.shape[0], dtype=x.dtype)
    segments = x.unfold(0, scale, scale)
    intervals = t.unfold(0, scale, scale)
    trends, _ = polyfit(intervals, segments, deg=2)
    dtr_segments = segments-trends
    var_dtr_segs = th.mean(dtr_segments**2, axis=1)
    return th.mean(var_dtr_segs**(q/2))**(1/q)

fluctuation(1000, x, 2)

# %%
scales=th.unique(
    th.logspace(
        th.log10(th.tensor(10.0)), 
        th.log10(th.tensor(x.shape[0]//4).to(x.dtype)),
        200
    ).round()
).to(int)

F = th.stack([fluctuation(scale, x) for scale in scales])
plt.scatter(scales, F)
plt.loglog()

# %%
log_scales_const = th.stack([
    th.ones_like(scales.to(x.dtype)),
    th.log10(scales.to(x.dtype))
], dim=-1)
log_F = th.log10(F)
betas = th.linalg.lstsq(log_scales_const, log_F).solution
alpha = betas[1]
print(fr"alpha estimado = {alpha:.2f}")

#%%
# Definindo uma função para DFA
def dfa(signal, q=2):
    x = th.cumsum(signal - signal.mean(), dim=0)
    scales=th.unique(
        th.logspace(
            th.log10(th.tensor(10.0)), 
            th.log10(th.tensor(x.shape[0]//4).to(x.dtype)),
            200
        ).round()
    ).to(int)
    F = th.stack([fluctuation(scale, x, q=q) for scale in scales])
    log_scales = th.log10(scales.to(x.dtype))
    log_F = th.log10(F.to(x.dtype))
    _, betas = polyfit(log_scales, log_F, deg=1)
    plt.plot(log_scales, log_F)
    return betas[1].item()

# %%
# testando com AR(1,1)
x = th.randn(N+20)
for i in range(1, x.shape[0]):
    x[i] = x[i-1]*(-0.9)+x[i]*0.3
x = x[20:]
plt.plot(x)

# %%
qs = [-3, -2, -1, 1, 2, 3]
res = [dfa(x, q=q) for q in qs]
# %%
