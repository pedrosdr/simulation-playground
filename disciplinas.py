#%%
import pandas as pd
import numpy as np

#%%
fine = pd.read_csv('fine.csv')
eco = pd.read_csv('eco.csv')

# %%
left_eco = pd.merge(eco, fine, how='left', on='code')

# %%
aprov = left_eco['name_y'].dropna()