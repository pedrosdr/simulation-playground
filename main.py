# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression

torch.manual_seed(0)
np.random.seed(0)

#%%
device = torch.device("cuda")
cpu = torch.device("cpu")

#%%
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.h = nn.Linear(3, 2)
        self.y = nn.Linear(2, 1)

    def forward(self, x):
        u = self.h(x)
        h = f.relu(u)
        y = self.y(h)
        return y
    
# %%
x = torch.randn((1000, 3))
y = torch.randn((1000, 1))*0.02 + 0.2*x[:,0:1] + 0.5*x[:,1:2] + 0.7*x[:,2:3]

# %%
sns.lineplot(x=x[:,0].detach().numpy(), y=y[:,0].detach().numpy())

# %%
model = Model().to(device)
optim = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

#%%
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, 100)

# %%
model.train()
for i in range(1000):
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()
    print(f"epoch: {i}")

#%%
model.eval()

xnew = x.detach().to(device).requires_grad_()
ynew = model(xnew.to(device))

fxs = torch.autograd.grad(
    ynew, xnew,
    grad_outputs=torch.ones_like(ynew),  # 1s para cada amostra
    retain_graph=False,
    create_graph=False
)
fxs = fxs[0].to(cpu).detach().numpy()

# %%
lr = LinearRegression()
xsk, ysk = x.detach().to(cpu).numpy(), y.detach().to(cpu).flatten().numpy()
lr.fit(xsk, ysk)

# %%
print(lr.coef_)
print(fxs.mean(axis=0))

# %%
