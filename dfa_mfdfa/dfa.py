#%%
import numpy as np
import matplotlib.pyplot as plt
import torch as tch

# %%
# Fractional Gaussian noise (fGn)
T = 100_000
t = np.linspace(0, 1, T)
dt = t[1]-t[0]
G = np.random.randn(T)*np.sqrt(dt) + 2
plt.plot(G)

#%%
# Fractional Brownian motion (fBm)
B = np.cumsum(G-G.mean())
plt.plot(t, B)

# %%
# DFA
N = 100_000
time = tch.linspace(0, 100, N)
dt = time[1]-time[0]
G = tch.randn(N) * tch.sqrt(dt)
B = tch.cumsum(G-G.mean(), 0)

scales = tch.unique(
    tch.logspace(tch.log10(tch.tensor(10)), tch.log10(tch.tensor(len(B)//4)), 200).to(tch.int)
)

F = tch.empty(scales.shape[0])
for i, scale in enumerate(scales):
    step = scale // 2
    starts = tch.arange(0, len(B) - scale + 1, step)
    ends = starts + scale
    segments = tch.stack(
        [B[s:e] for s, e in zip(starts, ends)],
        dim=0
    )
    times = tch.stack(
        [time[s:e] for s, e in zip(starts, ends)],
        dim=0
    )
    times = tch.stack([
        tch.ones(times.shape),
        times
    ], dim=-1)

    betas = tch.linalg.lstsq(times, segments).solution
    betas = betas.unsqueeze(-1)
    trends = times@betas
    dtr_segments = segments-trends.squeeze(-1)
    std = tch.std(dtr_segments, dim=1)
    F[i] = tch.mean(std)
    if i == scales.shape[0]-1:
        for j, (seg, tm, trd) in enumerate(zip(segments, times[...,1], trends)):
            plt.plot(tm, seg+j*5)
            plt.plot(tm, trd+j*5, color='black')
            plt.ylabel('Ignore este eixo (:')
            plt.xlabel('Tempo (t)')

log_scales_const = tch.stack([
    tch.ones_like(scales),
    tch.log10(scales)
], dim=-1).to(tch.float)
log_F = tch.log10(F)
betas = tch.linalg.lstsq(log_scales_const, log_F).solution
alpha = betas[1]
print(fr"alpha estimado = {alpha:.2f}")

# %%
plt.scatter(scales, F)
plt.loglog()

#%%