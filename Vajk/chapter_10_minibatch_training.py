# Corrected Mini-Batch Gradient Descent with MLX
import mlx.core as mlx
import mlx.nn as nn
import mlx.optimizers as opt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import time

# Load data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=42, train_size=0.6
)

# Convert to MLX arrays
y_train = mlx.array(y_train).reshape(-1, 1)
# Apply standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X_train)
X_train_standardized = scaler.transform(X_test)  # Correctly apply to test data after fitting
X_train = mlx.array(X_train_standardized.reshape(-1, 8))
X_test = mlx.array(scaler.transform(X_test).reshape(-1, 8))

# Shuffle data for proper mini-batching
shuffled_indices = mlx.arange(X_train.shape[0])
mlx.random.shuffle(shuffled_indices)
X_shuffled = X_train[shuffled_indices]
y_shuffled = y_train[shuffled_indices]

# Model Definition for Linear Regression in MLX
class LinearModel(nn.Module):
    def __init__(self, input_dims=8, output_dims=1):
        super().__init__()
        self.linear = nn.Linear(input_dims=input_dims, output_dims=output_dims)
    
    def __call__(self, x):
        return self.linear(x)

# Create loss function for Mean Squared Error with vmap for batching
@mlx.vmap
def mse_loss_fn(preds, targets):
    return nn.losses.mse_loss(preds, targets)

# Define training function with mini-batches
def train_mini_batch(model, X_shuffled, y_shuffled, optimizer, batch_size=32):
    n_samples = X_shuffled.shape[0]

    # Training loop for mini-batches
    model_params = mlx.eval(model.parameters)
    
    for i in range(0, n_samples, batch_size):
        # Get current mini-batch
        end_idx = min(i + batch_size, n_samples)
        X_batch = X_shuffled[i:end_idx]
        y_batch = y_shuffled[i:end_idx]

        # Forward pass
        with mlx.jit:
            preds = model(X_batch)
            loss = mse_loss_fn(preds, y_batch)

        # Backward pass (gradient computation)
        with mlx.jit:
            grads = nn.value_and_grad(model, loss_fn=lambda m, x, y: mse_loss_fn(m(x), y))

        # Update optimizer
        optimizer.update(model, grads)
    
# Training function with evaluation
def train_with_minibatches(model, X_shuffled, y_shuffled, X_test, y_test,
                            optimizer_class=opt.optimizers.SGD, 
                            learning_rate=0.1, num_epochs=20):
    optimizer = optimizer_class(learning_rate)

    # Training loop with mini-batches
    for epoch in range(num_epochs):
        tic = time.perf_counter()

        # Mini-batch training
        train_mini_batch(model, X_shuffled, y_shuffled, optimizer)

        # Evaluate on test set
        with mlx.jit:
            y_pred = model(X_test)
            mse = nn.losses.mse_loss(y_pred, y_test).item()
            mae = mlx.mean(mlx.abs(y_pred - y_test)).item()

        toc = time.perf_counter()
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"- Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
        print(f"- Time: {(toc - tic):.3f} seconds")

    return model

# Initialize model and train
model = LinearModel()
trained_model = train_with_minibatches(model, X_shuffled, y_shuffled, X_test, y_test)