# %%
from time import time

import torch

X = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
X
# %%
X.shape
X.dtype
X[0, 1]
X.mean()
# %%
import numpy as np

X.numpy()
torch.tensor(np.array([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]]))
torch.backends.mps.is_available()
X = X.to("mps")
X.device
# X.T #TypeError
# %% Same with MLX
import mlx.core as mlx
X_mlx = mlx.array([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
X_mlx
# %%
X_mlx.shape
X_mlx.dtype
X_mlx[0, 1]
X_mlx.mean()
# %%
X_np=np.array(X_mlx)
X_np
# %%
mlx.default_device()

# %%
"""
Why the Error Occurs:**
1. **MPS Backend Restrictions**:
   - MPS (Metal Performance Shaders) is a GPU acceleration backend for PyTorch on Apple devices. MPS tensors are designed to run efficiently on Apple silicon (M1, M2, etc.) and cannot be freely converted to NumPy arrays **without moving the data to CPU first**.

2. **`X.numpy()` on an MPS Tensor**:
   - The error message explicitly states: *"Use `Tensor.cpu()` to copy the tensor to host memory first."*
   - PyTorch's default assumption is that `X.numpy()` should work on CPU tensors. For MPS tensors, you **must explicitly move the tensor to the CPU** (`X.cpu()` or `.to('cpu')`) before converting it to NumPy.
"""
X.T.cpu().numpy()
X.device
# %%
X.T
# %%
M=torch.rand((1000,1000))
M.device
%timeit M @ M.T #2.78 ms ± 187 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
M_mps = M.to("mps")
%timeit M_mps @ M_mps.T #71.3 μs ± 45.4 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)
# %%

x=torch.tensor(5.0,requires_grad=True)
f=x**2
f
# %%
f.backward()
x.grad
# %%

learning_rate = 0.1
with torch.no_grad():
    x -= learning_rate * x.grad
x
# %% training loop

learning_rate=0.1
x=torch.tensor(5.0, requires_grad=True)
for i in range(100):
    f=x**2
    f.backward()
    with torch.no_grad():
        x -= learning_rate * x.grad
    x.grad.zero_() #reset gradients before backward()

# %% same with built-in torch fn
t=torch.tensor(5.0, requires_grad=True)
z=t.exp()
z
# %%
# z+=1
# z.backward() RuntimeError:
z=z+1
z.backward()
z
# %% linear regression
# pytorch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
housing = fetch_california_housing()
# %%
housing.data.shape  #n_feature=8
# %%
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42, train_size=.6)
X_train, X_valid, y_train, y_valid=train_test_split(X_train, y_train, train_size=.6, random_state=45)
# %% converting arrays to tensors
X_train=torch.FloatTensor(X_train)
X_valid=torch.FloatTensor(X_valid)
X_test=torch.FloatTensor(X_test)

y_train=torch.FloatTensor(y_train).reshape(-1,1) #reshape the tensors to column vectors
y_valid=torch.FloatTensor(y_valid).reshape(-1,1)
y_test=torch.FloatTensor(y_test).reshape(-1,1)
# %% StandardScaler with pure torch
means=X_train.mean(dim=0, keepdim=True)
stds=X_train.std(dim=0,keepdim=True)
"""
dim` parameter:**
This specifies which dimension of the tensor to split or slice. By default, PyTorch splits dimensions by rows (`dim=0`), meaning it slices along the "rows" axis.
- In your case: `mean(dim=0)` splits the tensor across the **dimension 0** (which corresponds to samples). This operation effectively groups your data into chunks of the same shape as `dtype=torch.float`, with one extra dimension to represent batch size.

**`keepdim=True`:**
This parameter tells PyTorch not to collapse the original dimension. If `keepdim=True`, the resulting tensor's shape will have a new second dimension equal to the original number of dimensions minus 1, helping maintain proper batch sizes and tensor dimensions during splits or operations.

dim=0 keeps the number of features when aggregating
"""
X_train=(X_train-means)/stds
X_valid=(X_valid-means)/stds
X_test=(X_test-means)/stds
# %%
X_train.shape #n_feature=8
# %%
torch.manual_seed(53)
n_feature=X_train.shape[1]
w=torch.randn((n_feature, 1), requires_grad=True)
b=torch.tensor(0., requires_grad=True)
# The weights are initialized randomly, while the bias is initialized to zero.
# The bias is initialized to zero to avoid bias in the model's predictions.
# %%
learning_rate=.4
n_epochs=20
for epoch in range(n_epochs):
    y_pred=X_train@w+b
    loss=((y_pred-y_train)**2).mean()
    loss.backward()
    with torch.no_grad():
        b -= learning_rate*b.grad # gradient descent step
        w -= learning_rate*w.grad
        b.grad.zero_()
        w.grad.zero_()
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
"""
It’s best to use a with torch.no_grad() context during inference:
PyTorch will consume less RAM and run faster since it won’t have
to keep track of the computation graph.
"""
# %%
X_new=X_test[:3]
with torch.no_grad():
    y_pred=X_new@w+b
print(y_pred)
# %%
y_test[:3]
# %% pytorch API
