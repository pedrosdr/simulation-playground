#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
diffs = np.random.normal(0, 1, [1000, 250])

# %%
R_pos = np.empty(diffs.shape[0])
R_neg = np.empty(diffs.shape[0])
for i, diff in enumerate(diffs):
    diff = diff[diff != 0]
    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff)
    rank = np.empty_like(order, dtype='float')
    rank[order] = np.arange(1, len(abs_diff)+1)
    mask = diff > 0
    r_pos = np.sum(rank[mask])
    r_neg = np.sum(rank[~mask])
    R_pos[i] = r_pos
    R_neg[i] = r_neg


# %%
T = R_pos-R_neg
plt.hist(T, bins=30)

# %%
plt.hist(np.minimum(R_pos, R_neg))

# %%
