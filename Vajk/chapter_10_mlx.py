# %%
import mlx.core as mlx
import mlx.nn as nn
import mlx.optimizers as opt

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
# %%
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42, train_size=0.6
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %%

y_train = mlx.array(y_train).reshape(-1, 1)
y_test = mlx.array(y_test).reshape(-1, 1)


# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# %%
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = mlx.array(
    X_train
)  # .reshape(-1,1) No reshaping needed** — just ensure the NumPy array from `scaler.fit_transform()` has shape `(N, 8)`.
X_test = mlx.array(X_test)  # .reshape(-1,1)
# %%
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %%
n_feature = X_train.shape[1]
model = nn.Linear(input_dims=n_feature, output_dims=1)
# %%
print(model(X_train), model(X_train).shape)


# %%
def loss_fn(model, X, y):
    return nn.losses.mse_loss(model(X), y)


# %%
import numpy as np

# %%
print(model.parameters())
# %%

np.random.seed(53)
mlx.eval(  # Set the model to evaluation mode.
    model.parameters()
)
num_epochs = 20
# %%
optimizer = opt.optimizers.SGD(learning_rate=0.1)
lag = nn.value_and_grad(model, loss_fn)


# @partial()
def step(X, y):
    loss, grads = lag(model, X, y)
    optimizer.update(model, grads)
    return loss


# @partial()
def eval_fn(X, y):
    predictions = model(X)
    return mlx.mean(mlx.abs(predictions - y))  # MAE


# %%
import time

# %%
print(
    X_train.shape,  # ok
    model(X_train).shape,
    y_train.shape,  # ok
)
# %%
for epoch in range(num_epochs):
    tic = time.perf_counter()
    # loss = loss_fn(model, X_train, y_train)
    # optimizer.update(model)
    step(X_train, y_train)
    mlx.eval(model.state)
    accu = eval_fn(X_test, y_test)
    toc = time.perf_counter()
    print(f"Epoch {epoch}: Test accuracy {accu.item():.3f}, Time {toc - tic:.3f} (s)")
