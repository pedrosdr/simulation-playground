#%%
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# %%
class LinearRWF(nn.Module):
    def __init__(
            self, in_features, out_features,
            mu=0.5, sigma=0.1
        ):
        super().__init__()
        W = nn.init.xavier_normal_(th.empty([out_features, in_features]))
        self.s = nn.Parameter(mu + th.randn(out_features)*sigma)
        self.V = nn.Parameter(th.exp(-self.s).unsqueeze(1)*W)
        self.bias = nn.Parameter(th.zeros(out_features))
    
    def forward(self, x):
        x = x @ self.V.t()
        return x * th.exp(self.s).unsqueeze(0) + self.bias.unsqueeze(0)

#%%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            LinearRWF(2, 10),
            nn.Tanh(),
            LinearRWF(10, 1)
        )
    
    def forward(self, x):
        return self.nn(x)

#%%
x = np.random.randn(200)
y = 1 + 0.1*x + 0.1*x**2 + np.random.randn(200)*0.09
plt.scatter(x, y)

x_t = th.tensor(x, dtype=th.float32)
y_t = th.tensor(y, dtype=th.float32).unsqueeze(1)
X = th.stack([x_t, x_t**2], dim=1)

#%%
net = Net().to(device)
optim = th.optim.Adam(net.parameters(), 0.001)
criterion = nn.MSELoss()
dataset = th.utils.data.TensorDataset(X, y_t)
loader = th.utils.data.DataLoader(dataset, batch_size=10)

for i in range(1000):
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optim.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()
    print('epoch:', i)

#%%
with th.no_grad():
    ynew = net(X.to(device)).detach().cpu().squeeze(1).numpy()
    plt.scatter(x, ynew)
    plt.scatter(x, y)
    
#%%
