# %% [markdown]
# **Chapter 10 – Building Neural Networks with PyTorch**
# %% [markdown]
# _This notebook contains all the sample code and solutions to the exercises in chapter 10._
# %% [markdown]
# <table align="left">
#   <td>
#     <a href="https://colab.research.google.com/github/ageron/handson-mlp/blob/main/10_neural_nets_with_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#   </td>
#   <td>
#     <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-mlp/blob/main/10_neural_nets_with_pytorch.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
#   </td>
# </table>
# %% [markdown]
# # Setup
# %% [markdown]
# This project requires Python 3.10 or above:
# %%
import sys

assert sys.version_info >= (3, 10)
# %% [markdown]
# It also requires Scikit-Learn ≥ 1.6.1:
# %%
from packaging.version import Version
import sklearn

assert Version(sklearn.__version__) >= Version("1.6.1")
# %% [markdown]
# Are we using Colab or Kaggle?
# %%
IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules
# %% [markdown]
# If using Colab, a couple libraries are not pre-installed so we must install them manually:
# %%
if IS_COLAB:
    %pip install -q optuna torchmetrics
# %% [markdown]
# And of course we need PyTorch, specifically PyTorch ≥ 2.6.0:
# %%
import torch

assert Version(torch.__version__) >= Version("2.6.0")
# %% [markdown]
# As we did in earlier chapters, let's define the default font sizes to make the figures prettier:
# %%
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
# %% [markdown]
# # PyTorch Fundamentals
# ## PyTorch Tensors
# %%
import torch

X = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
X
# %%
X.shape
# %%
X.dtype
# %%
X[0, 1]
# %%
X[:, 1]
# %%
10 * (X + 1.0)  # item-wise addition and multiplication
# %%
X.exp()
# %%
X.mean()
# %%
X.max(dim=0)
# %%
X @ X.T
# %%
import numpy as np

X.numpy()
# %%
torch.tensor(np.array([[1., 4., 7.], [2., 3., 6.]]))
# %%
torch.tensor(np.array([[1., 4., 7.], [2., 3., 6.]]), dtype=torch.float32)
# %%
torch.FloatTensor(np.array([[1., 4., 7.], [2., 3., 6]]))
# %%
# extra code: demonstrate torch.from_numpy()
X2_np = np.array([[1., 4., 7.], [2., 3., 6]])
X2 = torch.from_numpy(X2_np)  # X2_np and X2 share the same data in memory
X2_np[0, 1] = 88
X2
# %%
X[:, 1] = -99
X
# %%
X.relu_()
X
# %% [markdown]
# PyTorch tensors really resemble NumPy arrays. In fact, they have over 200 common functions!
# %%
# extra code: list functions that appear both in NumPy and PyTorch
functions = lambda mod: set(f for f in dir(mod) if callable(getattr(mod, f)))
", ".join(sorted(functions(torch) & functions(np)))
# %% [markdown]
# ## Hardware Acceleration
# %%
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

device
# %%
M = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
M = M.to(device)
M.device
# %%
M = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device)
# %%
R = M @ M.T  # run some operations on the GPU
R
# %%
M = torch.rand((1000, 1000))  # on the CPU
M @ M.T  # warmup
# %timeit M @ M.T

M = M.to(device)
M @ M.T  # warmup
# %timeit M @ M.T
# %% [markdown]
# ## Autograd
# %% [markdown]
# Consider a simple function, $f(x) = x^2$.
# Calculus tells us that the derivative of this function is $f'(x)=2x$. Let's evaluate $f(5)$ and the derivative $f'(5)$ using autograd. We expect to find $f(5)=5^2=25$ and $f'(5)=2*5=10$. Let's see!
# %%
x = torch.tensor(5.0, requires_grad=True)
f = x ** 2
f
# %%
f.backward()
x.grad
# %%
learning_rate = 0.1
with torch.no_grad():
    x -= learning_rate * x.grad  # gradient descent step
# %%
x
# %% [markdown]
# Alternatively, we could have used this code for the gradient descent step (but using `no_grad()` is more common for this):
# %%
x_detached = x.detach()
x_detached -= learning_rate * x.grad
# %%
x.grad.zero_()
# %% [markdown]
# Let's put everything together to get our training loop:
# %%
learning_rate = 0.1
x = torch.tensor(5.0, requires_grad=True)
for iteration in range(100):
    f = x ** 2  # forward pass
    f.backward()  # backward pass
    with torch.no_grad():
        x -= learning_rate * x.grad  # gradient descent step
    x.grad.zero_()  # reset the gradients
# %% [markdown]
# The variable `x` gets pushed towards 0, since that's the value that minimizes $f(x) = x^2$:
# %%
x
# %% [markdown]
# # Implementing Linear Regression
# ## Linear Regression Using Tensors & Autograd
# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
# %%
X_train = torch.FloatTensor(X_train)
X_valid = torch.FloatTensor(X_valid)
X_test = torch.FloatTensor(X_test)
means = X_train.mean(dim=0, keepdims=True)
stds = X_train.std(dim=0, keepdims=True)
X_train = (X_train - means) / stds
X_valid = (X_valid - means) / stds
X_test = (X_test - means) / stds
# %% [markdown]
# PyTorch expects the targets to have one row per sample, so let's reshape the targets to be column vectors:
# %%
y_train = torch.FloatTensor(y_train).view(-1, 1)
y_valid = torch.FloatTensor(y_valid).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)
# %%
torch.manual_seed(42)
n_features = X_train.shape[1]  # there are 8 input features
w = torch.randn((n_features, 1), requires_grad=True)
b = torch.tensor(0., requires_grad=True)
# %% [markdown]
# **Note**: in the next section, we will build an almost identical model using PyTorch's high-level API. Its results will be slightly different because it will use a different parameter initialization method: it will use a uniform random distribution from $-\frac{1}{2\sqrt 2}$ to $+\frac{1}{2\sqrt 2}$ to initialize both the weights and the bias term. If you want to get exactly the same result here as in the next section, you can uncomment and run the initialization code in the following cell, instead of the code in the previous cell:
# %%
# torch.manual_seed(42)
# n_features = X_train.shape[1]  # there are 8 input features
# r = 2 ** -1.5  # this is equal to 1 / 2√2
# w = torch.empty(n_features, 1).uniform_(-r, r)
# b = torch.empty(1).uniform_(-r, r)
# w.requires_grad_(True)
# b.requires_grad_(True)
# %%
learning_rate = 0.4
n_epochs = 20
for epoch in range(n_epochs):
    y_pred = X_train @ w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        b -= learning_rate * b.grad
        w -= learning_rate * w.grad
        b.grad.zero_()
        w.grad.zero_()
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")
# %%
X_new = X_test[:3]  # pretend these are new instances
with torch.no_grad():
    y_pred = X_new @ w + b  # use the trained parameters to make predictions
# %%
y_pred
# %% [markdown]
# ## Linear Regression Using PyTorch's High-Level API
# %%
import torch.nn as nn

torch.manual_seed(42)  # to get reproducible results
model = nn.Linear(in_features=n_features, out_features=1)
# %%
model.bias
# %%
model.weight
# %%
for param in model.parameters():
    print(param)
# %%
model(X_train[:2])
# %%
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
# %%
def train_bgd(model, optimizer, criterion, X_train, y_train, n_epochs):
    for epoch in range(n_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")
# %%
train_bgd(model, optimizer, mse, X_train, y_train, n_epochs)
# %%
X_new = X_test[:3]  # pretend these are new instances
with torch.no_grad():
    y_pred = model(X_new)  # use the trained model to make predictions

y_pred
# %% [markdown]
# # Implementing a Regression MLP
# %%
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(n_features, 50),
    nn.ReLU(),
    nn.Linear(50, 40),
    nn.ReLU(),
    nn.Linear(40, 1)
)
# %%
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
train_bgd(model, optimizer, mse, X_train, y_train, n_epochs)
# %% [markdown]
# # Implementing Mini-Batch Gradient Descent using DataLoaders
# %%
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# %%
# extra code – build the model just like earlier
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(n_features, 50), nn.ReLU(),
    nn.Linear(50, 40), nn.ReLU(),
    nn.Linear(40, 1)
)

model = model.to(device)

# extra code – build the optimizer and loss function, as earlier
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
# %%
def train(model, optimizer, criterion, train_loader, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.4f}")
# %%
train(model, optimizer, mse, train_loader, n_epochs)
# %% [markdown]
# # Model Evaluation
# %%
def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean):
    model.eval()
    metrics = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric = metric_fn(y_pred, y_batch)
            metrics.append(metric)
    return aggregate_fn(torch.stack(metrics))
# %%
valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=32)
valid_mse = evaluate(model, valid_loader, mse)
valid_mse
# %%
def rmse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().sqrt()

evaluate(model, valid_loader, rmse)
# %%
valid_mse.sqrt()
# %%
evaluate(model, valid_loader, mse,
         aggregate_fn=lambda metrics: torch.sqrt(torch.mean(metrics)))
# %%
import torchmetrics

def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # update it at each iteration
    return metric.compute()  # compute the final result at the end
# %%
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
evaluate_tm(model, valid_loader, rmse)
# %%
def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
               n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

torch.manual_seed(42)
learning_rate = 0.01
model = nn.Sequential(
    nn.Linear(n_features, 50), nn.ReLU(),
    nn.Linear(50, 40), nn.ReLU(),
    nn.Linear(40, 30), nn.ReLU(),
    nn.Linear(30, 1)
)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train2(model, optimizer, mse, rmse, train_loader, valid_loader,
                 n_epochs)

# Since we compute the training metric
plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], ".--",
         label="Training")
plt.plot(np.arange(n_epochs) + 1.0, history["valid_metrics"], ".-",
         label="Validation")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid()
plt.title("Learning curves")
plt.axis([0.5, 20, 0.4, 1.0])
plt.legend()
plt.show()
# %% [markdown]
# # Building Nonsequential Models Using Custom Modules
# %%
class WideAndDeep(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
            nn.Linear(40, 30), nn.ReLU(),
        )
        self.output_layer = nn.Linear(30 + n_features, 1)

    def forward(self, X):
        deep_output = self.deep_stack(X)
        wide_and_deep = torch.concat([X, deep_output], dim=1)
        return self.output_layer(wide_and_deep)
# %%
torch.manual_seed(42)
model = WideAndDeep(n_features).to(device)
learning_rate = 0.002  # the model changed, so did the optimal learning rate
# %%
# extra code: train the model, exactly our previous models
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train2(model, optimizer, mse, rmse, train_loader, valid_loader,
                 n_epochs)
# %%
class WideAndDeepV2(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features - 2, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
            nn.Linear(40, 30), nn.ReLU(),
        )
        self.output_layer = nn.Linear(30 + 5, 1)

    def forward(self, X):
        X_wide = X[:, :5]
        X_deep = X[:, 2:]
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)
# %%
torch.manual_seed(42)
model = WideAndDeepV2(n_features).to(device)
# %%
# extra code: train the model, exactly our previous models
learning_rate = 0.002
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train2(model, optimizer, mse, rmse, train_loader, valid_loader,
                 n_epochs)
# %% [markdown]
# ## Building Models with Multiple Inputs
# %%
class WideAndDeepV3(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features - 2, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
            nn.Linear(40, 30), nn.ReLU(),
        )
        self.output_layer = nn.Linear(30 + 5, 1)

    def forward(self, X_wide, X_deep):
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)
# %%
torch.manual_seed(42)
train_data_wd = TensorDataset(X_train[:, :5], X_train[:, 2:], y_train)
train_loader_wd = DataLoader(train_data_wd, batch_size=32, shuffle=True)
valid_data_wd = TensorDataset(X_valid[:, :5], X_valid[:, 2:], y_valid)
valid_loader_wd = DataLoader(valid_data_wd, batch_size=32)
test_data_wd = TensorDataset(X_test[:, :5], X_test[:, 2:], y_test)
test_loader_wd = DataLoader(test_data_wd, batch_size=32)
# %%
def evaluate_multi_in(model, data_loader, metric):
    model.eval()
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for X_batch_wide, X_batch_deep, y_batch in data_loader:
            X_batch_wide = X_batch_wide.to(device)
            X_batch_deep = X_batch_deep.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch_wide, X_batch_deep)
            metric.update(y_pred, y_batch)  # update it at each iteration
    return metric.compute()  # compute the final result at the end

def train_multi_in(model, optimizer, criterion, metric, train_loader,
                   valid_loader, n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for *X_batch_inputs, y_batch in train_loader:
            model.train()
            X_batch_inputs = [X.to(device) for X in X_batch_inputs]
            y_batch = y_batch.to(device)
            y_pred = model(*X_batch_inputs)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_multi_in(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

torch.manual_seed(42)
learning_rate = 0.01
model = WideAndDeepV3(n_features).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train_multi_in(model, optimizer, mse, rmse, train_loader_wd,
                         valid_loader_wd, n_epochs)
# %%
class WideAndDeepDataset(torch.utils.data.Dataset):
    def __init__(self, X_wide, X_deep, y):
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        input_dict = {"X_wide": self.X_wide[idx], "X_deep": self.X_deep[idx]}
        return input_dict, self.y[idx]
# %%
torch.manual_seed(42)
train_data_named = WideAndDeepDataset(
    X_wide=X_train[:, :5], X_deep=X_train[:, 2:], y=y_train)
train_loader_named = DataLoader(train_data_named, batch_size=32, shuffle=True)
valid_data_named = WideAndDeepDataset(
    X_wide=X_valid[:, :5], X_deep=X_valid[:, 2:], y=y_valid)
valid_loader_named = DataLoader(valid_data_named, batch_size=32)
test_data_named = WideAndDeepDataset(
    X_wide=X_test[:, :5], X_deep=X_test[:, 2:], y=y_test)
test_loader_named = DataLoader(test_data_named, batch_size=32)
# %%
def evaluate_named(model, data_loader, metric):
    model.eval()
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for inputs, y_batch in data_loader:
            inputs = {name: X.to(device) for name, X in inputs.items()}
            y_batch = y_batch.to(device)
            y_pred = model(X_wide=inputs["X_wide"], X_deep=inputs["X_deep"])
            metric.update(y_pred, y_batch)
    return metric.compute()  # compute the final result at the end

def train_named(model, optimizer, criterion, metric, train_loader,
                   valid_loader, n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for inputs, y_batch in train_loader:
            model.train()
            inputs = {name: X.to(device) for name, X in inputs.items()}
            y_batch = y_batch.to(device)
            y_pred = model(**inputs)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_named(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

torch.manual_seed(42)
learning_rate = 0.01
model = WideAndDeepV3(n_features).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train_named(model, optimizer, mse, rmse, train_loader_named,
                      valid_loader_named, n_epochs)
# %% [markdown]
# ## Building Models with Multiple Outputs
# %%
class WideAndDeepV4(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features - 2, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
            nn.Linear(40, 30), nn.ReLU(),
        )
        self.output_layer = nn.Linear(30 + 5, 1)
        self.aux_output_layer = nn.Linear(30, 1)

    def forward(self, X_wide, X_deep):
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        main_output = self.output_layer(wide_and_deep)
        aux_output = self.aux_output_layer(deep_output)
        return main_output, aux_output
# %%
import torchmetrics

def evaluate_multi_out(model, data_loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for inputs, y_batch in data_loader:
            inputs = {name: X.to(device) for name, X in inputs.items()}
            y_batch = y_batch.to(device)
            y_pred, _ = model(**inputs)
            metric.update(y_pred, y_batch)
    return metric.compute()

def train_multi_out(model, optimizer, criterion, metric, train_loader,
                   valid_loader, n_epochs):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for inputs, y_batch in train_loader:
            model.train()
            inputs = {name: X.to(device) for name, X in inputs.items()}
            y_batch = y_batch.to(device)
            y_pred, y_pred_aux = model(**inputs)
            main_loss = criterion(y_pred, y_batch)
            aux_loss = criterion(y_pred_aux, y_batch)
            loss = 0.8 * main_loss + 0.2 * aux_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_multi_out(model, valid_loader, metric).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

torch.manual_seed(42)
learning_rate = 0.01
model = WideAndDeepV4(n_features).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
mse = nn.MSELoss()
rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
history = train_multi_out(model, optimizer, mse, rmse, train_loader_named,
                          valid_loader_named, n_epochs)
# %% [markdown]
# # Building an Image Classifier with PyTorch
# %% [markdown]
# ## Using TorchVision to Load the Dataset
# %%
import torchvision
import torchvision.transforms.v2 as T

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])
# %%
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)
# %% [markdown]
# Each entry is a tuple (image, target):
# %%
X_sample, y_sample = train_data[0]
# %% [markdown]
# Each image has a shape \[channels, rows, columns\]. Grayscale images like in Fashion MNIST have a single channel (while RGB images have 3, and other types of images, such as satellite images, may have many more). Fashion images are grayscale and 28x28 pixels:
# %%
X_sample.shape
# %%
X_sample.dtype
# %%
train_and_valid_data.classes[y_sample]
# %% [markdown]
# ## Building the Classifier
# %%
class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)

torch.manual_seed(42)
model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)
xentropy = nn.CrossEntropyLoss()
# %%
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
_ = train2(model, optimizer, xentropy, accuracy, train_loader, valid_loader,
           n_epochs)
# %%
model.eval()
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)  # index of the largest logit
y_pred
# %%
[train_and_valid_data.classes[index] for index in y_pred]
# %% [markdown]
# Let's check whether the model made the correct predictions:
# %%
y_new[:3]
# %% [markdown]
# All correct! 😃
# %%
import torch.nn.functional as F
y_proba = F.softmax(y_pred_logits, dim=1)
if device == "mps":
    y_proba = y_proba.cpu()
y_proba.round(decimals=3)
# %%
y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1)
if device == "mps":
    y_top4_probas = y_top4_probas.cpu()
y_top4_probas.round(decimals=3)
# %%
y_top4_indices
# %%
sum([param.numel() for param in model.parameters()])
# %% [markdown]
# # Hyperparameter Tuning using Optuna
# %%
import optuna

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=n_hidden,
                            n_hidden2=n_hidden, n_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    accuracy = accuracy.to(device)
    history = train2(model, optimizer, xentropy, accuracy, train_loader,
                     valid_loader, n_epochs=10)
    validation_accuracy = max(history["valid_metrics"])
    return validation_accuracy
# %%
torch.manual_seed(42)
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=5)
# %%
study.best_params
# %%
study.best_value
# %%
def objective(trial, train_loader, valid_loader):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=n_hidden,
                            n_hidden2=n_hidden, n_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    accuracy = accuracy.to(device)
    best_validation_accuracy = 0.0
    for epoch in range(n_epochs):
        history = train2(model, optimizer, xentropy, accuracy, train_loader,
                         valid_loader, n_epochs=1)
        validation_accuracy = max(history["valid_metrics"])
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
        trial.report(validation_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_validation_accuracy
# %%
objective_with_data = lambda trial: objective(
    trial, train_loader=train_loader, valid_loader=valid_loader)
# %%
from functools import partial

objective_with_data = partial(objective, train_loader=train_loader,
                              valid_loader=valid_loader)
# %%
torch.manual_seed(42)
sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction="maximize", sampler=sampler,
                            pruner=pruner)
study.optimize(objective_with_data, n_trials=20)
# %%
study.best_value
# %%
study.best_params
# %% [markdown]
# # Saving and Loading a PyTorch Model
# %%
torch.save(model, "my_fashion_mnist.pt")
# %%
loaded_model = torch.load("my_fashion_mnist.pt", weights_only=False)
# %%
loaded_model.eval()
y_pred_logits = loaded_model(X_new)
# %%
torch.save(model.state_dict(), "my_fashion_mnist_weights.pt")
# %%
type(model.state_dict())
# %%
new_model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                            n_classes=10)
loaded_weights = torch.load("my_fashion_mnist_weights.pt", weights_only=True)
new_model.load_state_dict(loaded_weights)
new_model.eval()
# %%
model_data = {
    "model_state_dict": model.state_dict(),
    "model_hyperparameters": {
        "n_inputs": 1 * 28 * 28,
        "n_hidden1": 300,
        "n_hidden2": 100,
        "n_classes": 10,
    }
}
torch.save(model_data, "my_fashion_mnist_model.pt")
# %%
loaded_data = torch.load("my_fashion_mnist_model.pt", weights_only=True)
new_model = ImageClassifier(**loaded_data["model_hyperparameters"])
new_model.load_state_dict(loaded_data["model_state_dict"])
new_model.eval()
# %% [markdown]
# # Compiling and Optimizing a PyTorch Model
# %%
torchscript_model = torch.jit.trace(model, X_new)
# %%
torchscript_model = torch.jit.script(model)
# %%
optimized_model = torch.jit.optimize_for_inference(torchscript_model)
# %%
optimized_model.save("my_fashion_mnist_torchscript.pt")
# %%
loaded_torchscript_model = torch.jit.load("my_fashion_mnist_torchscript.pt")
# %%
y_pred_logits = loaded_torchscript_model(X_new)
y_pred_logits
# %%
compiled_model = torch.compile(model)
# %%
if device == "cuda":
    y_pred_logits = compiled_model(X_new)
# %% [markdown]
# # Exercise Solutions
# %% [markdown]
# ## Exercises 1. to 12.
# %% [markdown]
# 1. PyTorch is similar to NumPy is many ways, but it offers some extra features. The main ones are:
#     * Auto-differentiation
#     * Support for hardware accelerators
#     * Includes optimizers and ready-to-use neural net components
# 
# 2. `torch.exp()` returns a copy of the input tensor while `torch.exp_()` modifies it in place. Similarly, `torch.relu()` returns a copy while `torch.relu_()` modifies in place.
# 
# 3. To create a new tensor on the GPU, you can use one of the following methods:
#      * Set the `device` argument when calling `torch.tensor()`, `torch.rand()`, or other functions that create new tensors. For example, `torch.randn(10, device="cuda")` creates a new tensor on the CUDA GPU, with 10 random elements.
#      * Create a new tensor on the CPU, then transfer it to the GPU by using calling its `to()` method, for example `torch.randn(10).to("cuda")`. However, it's more efficient to create the tensor directly on the GPU.
#      * The `*_like()` functions such as `ones_like()` and `zeros_like()` create a new tensor on the same device as another tensor. They also use the same data type.
#     * Lastly, if you execute an operation on a tensor that lives on the GPU, the result will generally be a new tensor on the GPU (unless you use an in-place operation such as `torch.exp_()`.
# 
# 4. Here are three ways to perform tensor computations without using autograd:
#     * Manipulate tensors created with `requires_grad=False` (which is the default).
#     * Run the computations inside a `with torch.no_grad():` block.
#     * Call the `detach()` method on the tensor you want to manipulate without autograd.
# 
# 5. Here's what happens in each case:
#     * The first code sample will work fine, it will _not_ cause a `RuntimeError`: indeed, the `cos()` method creates a new (non-leaf) tensor, then `exp_()` modifies it in place. During the backward pass, PyTorch is able to backpropagate through the `exp_()` operation because the derivative of exp(x) is exp(x), so PyTorch doesn't need to know what the input x was, it can just use the output of the forward pass (i.e., exp(x)) during the backward pass.
#     * If you replace `z = t.cos().exp_()` with `z = t.cos_().exp()` then you will get a `RuntimeError` on that line ("_a leaf Variable that requires grad is being used in an in-place operation_"). Indeed, `t` is a leaf tensor (since it was created directly by the user, not the result of any computation) and you cannot apply an in-place operation on a leaf tensor with `requires_grad=True`.
#     * If you replace `z = t.cos().exp_()` with `z = t.exp().cos_()`, then you will get a `RuntimeError` ("_one of the variables needed for gradient computation has been modified by an inplace operation_") during the backward pass. Indeed, the `exp()` operation relies on the fact that the derivative of exp(_t_) is exp(_t_), so it doesn't need to store the tensor `t` for the backward pass, instead it relies on the fact that it can just use the tensor returned by `t.exp()` (let's call it `e`). So far so good. But when we call the `cos_()` operation, it knows that it will need its input during the backward pass (since the derivative of cos(_e_) is –sin(_e_)), so it keeps a copy of its input tensor `e`. Next, it tries to modify the original `e` in-place, and in doing so it notices that this tensor is needed by another operation (`exp()`) for the backward pass, so it knows that something is fishy and it raises a `RuntimeError`.
#     * The second code example will fail during backpropogation, with a `RuntimeError`. It's a very similar error to the previous one: the `cos()` operation stores a reference to its input tensor `v`, since `v` will be needed during the backward pass. Indeed, the derivative of cos(_v_) is –sin(_v_), so we need to save _v_. Next, the `sin_()` operation creates a copy of its input `v` (since it's an in-place operation, it knows that it must create a copy, not just preserve a reference) then it proceeds to modify the original `v`, but this tensor is needed to compute the gradient of `cos(_v_)`, so PyTorch raises a `RuntimeError`.
#     * If you replace `w = v.cos() * v.sin_()` with `w = v.cos_() * v.sin()`, then there is no longer any `RuntimeError`. Indeed, the `cos_()` operation creates a copy of its input `v` so it can compute –sin(_v_) during the backward pass. The `sin()` operation is not in-place so in keeps a reference to its input `v`, not a copy. Backprop then runs just fine. However, there's a catch: by the time the `sin()` operation runs, its input `v` is no longer equal to 3.0, but instead it's equal to cos(3.0) since the `cos_()` modified `v` in place. As a result, `w = v.cos() * v.sin_()` does _not_ give the same result as `w = v.cos_() * v.sin()`. The former computes cos(3) * sin(3) while the latter computes cos(3) * sin(cos(3)). And of course the gradients change as well. That's why you should be very careful with in-place operations: they can make your code faster, sure, but they can also make it silently wrong.
# 
# 6. A `Linear(100, 200)` module has 200 neurons: one per output. Its `weight` tensor has a shape of [200, 100] and its `bias` parameter has a shape of [200]. It expects its inputs to have a shape of [..., 100], for example [32, 100], or [32, 64, 100]. It treats all dimensions independently, except for the last one. The output shape is identical to the input shape, except that the last dimension is replaced with 200. For example, if the input shape is [32, 64, 100], then the output shape is [32, 64, 200].
# 
# 7. The main steps of a PyTorch training loop are:
#     * Prepare a batch of samples from the training set. You can use a `DataLoader` for this.
#     * Optionally transfer these samples to the GPU (typically using `X_batch.to(device)` and `y_batch.to(device)`).
#     * Run the inputs through the model, for example `y_pred = model(X_batch)`.
#     * Compute the loss, for example `loss = criterion(y_pred, y_true)`.
#     * Backpropagate through the loss using `loss.backward()`.
#     * Perform an optimizer step: `optimizer.step()`.
#     * Zero out the gradients: `optimizer.zero_grad()` (alternatively, you can do this before the backward pass, which may be safer if the gradients are non-zero before the training loop starts).
#     * The whole training loop is often split into epochs, but this is optional.
# 
# 8. It is recommended to create the optimizer _after_ the model is moved to the GPU because most optimizers have some internal state, and this state is usually allocated on the same device as the model parameters.
# 
# 9. To speed up training when using a GPU, you should generally set the following `DataLoader` options:
#     * Set the data loader's `num_workers` argument to the number of processes you want to use for data loading and preprocessing. This will often speed up training by pre-fetching the next batches on the CPU while the GPU is still working on the current batch. The optimal number depends on your platform, hardware, and workload, so you should experiment with different values.
#     * Set the data loader's `prefetch_factor` argument to control the number of batches that each worker pre-fetches.
#     * If spawning and synchronizing workers causes too much overhead (especially on Windows), you can try setting `persistent_workers=True` to reuse the same workers across epochs.
# 
# 10. The main classification losses provided by PyTorch are:
#     * `nn.CrossEntropyLoss`: this is generally the loss you want to use for multiclass classification, as it's efficient and numerically stable. This loss works directly on logits, not probabilities, so your model must not include the softmax activation function on the output layer. As a result, whenever you need to estimate probabilities, you must call the `F.softmax()` function on the logits output by the model.
#     * `nn.BCEWithLogitsLoss` (BCE stands for _binary cross-entropy_): this is usually the loss you want to use for binary classification, for the same reason as `nn.CrossEntropyLoss`. Just like the previous loss, it works directly with logits, so your model must not include the sigmoid activation function on the output layer. Whenever you need to estimate probabilities, you must call the `F.sigmoid()` function on the logits output by the model. Note that some people prefer to use `nn.CrossEntropyLoss` even for binary classification, as it makes the code more consistent regardless of the number of classes, at a tiny computational and memory cost.
#     * `nn.NLLLoss` (NLL stands for _negative log-likelihood_): this is an alternative to `nn.CrossEntropyLoss` for multiclass classification. To use it, your model must output log probabilities rather than logits. This can be done using `nn.LogSoftmax()` or `F.log_softmax()`. This approach is a bit slower than using `CrossEntropyLoss`, but it can be useful if you want your model to output log probabilities rather than logits, or when you wish to tweak the probability distribution before computing the final loss (indeed, it is sometimes easier to modify log probabilities rather than logits). Whenever you need to estimate probabilities, you must call `torch.exp()` on the model's outputs.
#     * `nn.BCELoss`: this loss is an alternative to `nn.BCEWithLogitsLoss` for binary classification. It assumes that your model outputs probabilities rather than logits, so your model's output layer must use the sigmoid activation function. This can be convenient if you want your model to output probabilities directly, but it's a bit slower and less numerically stable than using `BCEWithLogitsLoss`.
# 
# 11. Calling `model.train()` before training and `model.eval()` before evaluation is important because some layers (such as `nn.Dropout`, `nn.BatchNorm1d` or `nn.BatchNorm2d`) don't behave in the same way during training and evaluation, therefore we must tell the model in which mode it should run.
# 
# 12. Both `torch.jit.trace()` and `torch.jit.script()` attempt to capture your model's computation graph and turn it into TorchScript code that can be optimized, saved, and deployed to various platforms. However, these functions work very differently:
#     * The `torch.jit.trace()` function runs your model with a tracing tensor that captures which operations are executed. It's quite simple and works well for simple models, but it cannot capture conditionals (e.g., `if`, `elif`, `else`, `match`): it only captures the branch of the conditional that is actually executed during tracing. Similarly, if your model contains a loop (e.g., `for` or `while`) then tracing will not capture the loop itself, it will only capture the repeated operations.
#     * The`torch.jit.script()` function actually parses your Python code to generate TorchScript code. This allows it to detect conditionals (as long as the conditions are tensors), and also capture loops. However, it only works with a subset of Python: you cannot use global variables, Python generators (`yield`), complex list comprehensions, variable length function arguments (`*args` or `**kwargs`), or `match` statements. Moreover, types must be fixed (a function cannot return an integer in some cases and a float in others), and you can only call other functions if they also respect these rules, so no standard library, no third-party libraries, etc.
# %% [markdown]
# ## Exercise 13.
# %% [markdown]
# Exercise: _Use autograd to find the gradient vector of f(_x_, _y_) = sin(_x_<sup>2</sup> _y_) at the point (_x_, _y_) = (1.2, 3.4)._
# %%
def f(x, y):
    return torch.sin(x ** 2 * y)

x = torch.tensor(1.2, requires_grad=True)
y = torch.tensor(3.4, requires_grad=True)
result = f(x, y)
result.backward()

x.grad.item(), y.grad.item()
# %% [markdown]
# Alternatively, we could use a vectorized implementation:
# %%
def f_vec(v):
    return torch.sin(v[0] ** 2 * v[1])

v = torch.tensor([1.2, 3.4], requires_grad=True)
result = f_vec(v)
result.backward()

v.grad
# %% [markdown]
# We get the same partial derivatives, but this time they are wrapped in a gradient tensor.
# %% [markdown]
# Let's check this result by computing an approximation of the partial derivatives, using the fact that the partial derivative of a function $f(x, y)$ with regard to $x$, at a point $(x_0, y_0)$ is the limit of $\dfrac{f(x_0 + \epsilon, y_0) - f(x_0, y0)}{\epsilon}$ as $\epsilon$ approaches zero. Similarly, the partial derivative of that function with regard to $y$, at the same point $(x_0, y_0)$, is the limit of $\dfrac{f(x_0, y_0 + \epsilon) - f(x_0, y0)}{\epsilon}$ as $\epsilon$ approaches zero.
# %%
eps = 0.00005
df_dx = (f(x + eps, y) - f(x, y)) / eps
df_dy = (f(x, y + eps) - f(x, y)) / eps

df_dx.item(), df_dy.item()
# %% [markdown]
# That's close enough.
# %% [markdown]
# We can also derive the equations for the partial derivatives mathematically (see the [calculus tutorial](math_differential_calculus.ipynb) to learn how to do this):
# * $\dfrac{\partial f}{\partial x} = 2xy \cos(x^2 y)$
# * $\dfrac{\partial f}{\partial y} = x^2 \cos(x^2 y)$
# %%
df_dx = 2 * x * y * torch.cos(x**2 * y)
df_dy = x ** 2 * torch.cos(x**2 * y)

df_dx.item(), df_dy.item()
# %% [markdown]
# Perfect!
# %% [markdown]
# ## Exercise 14.
# %% [markdown]
# Exercise: _Create a custom `Dense` module that replicates the functionality of an `nn.Linear` module followed by an `nn.ReLU` module. Try implementing it first using the `nn.Linear` and `nn.ReLU` modules, and then reimplement it using `nn.Parameter` and the `relu()` function._
# %%
class Dense(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(self.linear(X))
# %% [markdown]
# Let's try it out:
# %%
torch.manual_seed(42)
dense = Dense(3, 5)
X = torch.randn(2, 3)
y_pred = dense(X)
y_pred.shape
# %% [markdown]
# Looks fine. We can double-check that it gives the right result:
# %%
y_pred_check = dense.relu(X @ dense.linear.weight.T + dense.linear.bias)
torch.allclose(y_pred, y_pred_check)
# %% [markdown]
# Nice. Now let's reimplement the `Dense` class using `nn.Parameter` and `relu()` instead of `nn.Linear` and `nn.ReLU`:
# %%
class Dense2(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        z = X @ self.weight.T + self.bias
        return F.relu(z)
# %% [markdown]
# Let's try it out:
# %%
torch.manual_seed(42)
dense2 = Dense2(3, 5)
X = torch.randn(2, 3)
y_pred2 = dense2(X)
y_pred2.shape
# %% [markdown]
# Looks fine. Again, we can double-check that it gives the right result:
# %%
y_pred2_check = F.relu(X @ dense2.weight.T + dense2.bias)
torch.allclose(y_pred2, y_pred2_check)
# %% [markdown]
# In Chapter 11, we will see that it's preferable to initialize the weights using Kaiming initialization, when the activation function is ReLU, so let's do that:
# %%
class Dense3(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        z = X @ self.weight.T + self.bias
        return F.relu(z)
# %%
torch.manual_seed(42)
dense3 = Dense3(3, 5)
X = torch.randn(2, 3)
y_pred3 = dense3(X)
y_pred3.shape
# %% [markdown]
# Still works fine. Again, we can double-check that this module gives the right result:
# %%
y_pred3_check = F.relu(X @ dense3.weight.T + dense3.bias)
torch.allclose(y_pred3, y_pred3_check)
# %% [markdown]
# ## Exercise 15.
# %% [markdown]
# Exercise: _Build and train a classification MLP on the CoverType dataset._
# 
# Step 1: _Load the dataset using `sklearn.datasets.fetch_covtype()` and create a custom PyTorch `Dataset` for this data._
# %% [markdown]
# **Warning**: do not forget to scale the inputs, as gradient descent might not work well otherwise (as we saw in Chapter 4).
# %%
from sklearn.datasets import fetch_covtype
from torch.utils.data import TensorDataset

covtype = fetch_covtype()

X_covtype = torch.tensor(covtype.data, dtype=torch.float32)
means = X_covtype.mean(dim=0, keepdim=True)
stds = X_covtype.std(dim=0, keepdim=True)
X_standardized_covtype = (X_covtype - means) / stds

y_covtype = torch.tensor(covtype.target - 1, dtype=torch.long)

covtype_dataset = TensorDataset(X_standardized_covtype, y_covtype)
# %% [markdown]
# Note that the targets range from 1 to 7, but the `nn.CrossEntropyLoss` expects it to start at 0, which is why we subtract 1 from the targets. We also convert the inputs from 64-bit floats to 32-bit floats, and the targets to 64-bit integers. These are the default types in PyTorch for floats and integers.
# %% [markdown]
# Let's look at the first instance:
# %%
sample0, target0 = covtype_dataset[0]
sample0.shape, target0.shape
# %% [markdown]
# Looks good!
# %% [markdown]
# Step 2: _Create data loaders for training, validation, and testing._
# %% [markdown]
# The dataset is not shuffled or split by default, so we shuffle and split it:
# %%
from torch.utils.data import random_split

torch.manual_seed(42)

train_size = len(covtype_dataset) * 80 // 100
valid_size = len(covtype_dataset) * 10 // 100
test_size = len(covtype_dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(
    covtype_dataset,
    [train_size, valid_size, test_size])
# %% [markdown]
# Now let's create the data loaders:
# %%
from torch.utils.data import DataLoader

batch_size = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# %% [markdown]
# Step 3: _Build a custom MLP module to tackle this classification task. You can optionally use the custom `Dense` module from the previous exercise._
# %% [markdown]
# There are several ways to do this. Let's look at a few alternatives.
# %% [markdown]
# Since we're building an MLP, which is just a stack of layers, we can just use a `nn.Sequential` module:
# %%
n_inputs = len(covtype.feature_names)  # == 54
n_classes = len(set(covtype.target))  # == 7

torch.manual_seed(42)
covtype_model = nn.Sequential(
    nn.Linear(n_inputs, 200), nn.ReLU(),
    nn.Linear(200, 100), nn.ReLU(),
    nn.Linear(100, 50), nn.ReLU(),
    nn.Linear(50, n_classes)
).to(device)
# %% [markdown]
# Note that the output layer must not use any activation function since we will use the `nn.CrossEntropyLoss`.
# %% [markdown]
# Alternatively, we can use the `Dense3` class we defined earlier:
# %%
torch.manual_seed(42)
covtype_model = nn.Sequential(
    Dense3(n_inputs, 200),
    Dense3(200, 100),
    Dense3(100, 50),
    nn.Linear(50, n_classes)
).to(device)
# %% [markdown]
# Or we can create a custom module. Let's make it flexible so it accepts a list containing the number of neurons in each hidden layer:
# %%
class CoverTypeModel(nn.Module):
    def __init__(self, n_neurons, n_inputs=n_inputs, n_classes=n_classes):
        super().__init__()
        layers = [
            Dense3(n_in, n_out)
            for n_in, n_out in zip([n_inputs] + n_neurons, n_neurons)
        ] + [nn.Linear(n_neurons[-1], n_classes)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, X):
        return self.mlp(X)

torch.manual_seed(42)
covtype_model = CoverTypeModel([200, 100, 50]).to(device)
# %% [markdown]
# Lastly, we could use a `nn.ModuleList` instead of a `nn.Sequential` module, and modify the `forward()` method to explicitly run each layer:
# %%
class CoverTypeModel(nn.Module):
    def __init__(self, n_neurons, n_inputs=n_inputs, n_classes=n_classes):
        super().__init__()
        layers = [
            Dense3(n_in, n_out)
            for n_in, n_out in zip([n_inputs] + n_neurons, n_neurons)
        ] + [nn.Linear(n_neurons[-1], n_classes)]
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
# %% [markdown]
# Step 4: _Train this model on the GPU, and try to reach 93% accuracy on the test set. For this, you will likely have to perform hyperparameter search to find the right number of layers and neurons per layer, a good learning rate and batch size, and so on. You can optionally use Optuna for this._
# 
# %% [markdown]
# Let's train the model on the GPU. As we saw in Chapter 4, it's often helpful to reduce the learning rate at the end of training, so let's do that. We will train for 6 times 15 epochs, reducing the learning rate each time. The results might vary depending on the platform, but you should get over 94% accuracy on the test set.
# %%
torch.manual_seed(42)
covtype_model = CoverTypeModel([200, 100, 50]).to(device)

for learning_rate in [0.16, 0.08, 0.04, 0.02, 0.01, 0.005]:
    n_epochs = 15
    optimizer = torch.optim.SGD(covtype_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(task="multiclass",
                                   num_classes=n_classes).to(device)
    history = train2(covtype_model, optimizer, criterion, metric, train_loader,
                     valid_loader, n_epochs)
# %%
evaluate_tm(covtype_model, test_loader, metric)
# %% [markdown]
# I manually tuned the hyperparameters above through trial and error. It took about 30 minutes of search to achieve over 94% accuracy, so I didn’t need to run a full hyperparameter search. However, you can definitely use Optuna for this task if you want to get even better results. The code below tunes the learning rate, the number of hidden layers, and the number of neurons per hidden layer.
# %%
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 1.0, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_hidden = trial.suggest_int("n_hidden", 30, 150)
    covtype_model = CoverTypeModel([n_hidden] * n_layers).to(device)
    optimizer = torch.optim.SGD(covtype_model.parameters(), lr=learning_rate)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass",
                                     num_classes=n_classes).to(device)
    best_validation_accuracy = 0.0
    for epoch in range(n_epochs):
        history = train2(covtype_model, optimizer, xentropy, accuracy,
                         train_loader, valid_loader, n_epochs=1)
        validation_accuracy = max(history["valid_metrics"])
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
        trial.report(validation_accuracy, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_validation_accuracy
# %% [markdown]
# **Note**: I’ve set `n_trials=2` so you can run a quick test, but if you want to actually tune the hyperparameters, you should increase this to at least 50—and ideally to several hundred, but it will take several hours.
# %%
torch.manual_seed(42)
sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction="maximize", sampler=sampler,
                            pruner=pruner)
study.optimize(objective, n_trials=2)
# %%
study.best_value
# %%
study.best_params
# %% [markdown]
# Hope you enjoyed this chapter!