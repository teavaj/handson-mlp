# Complete Linear Regression Pipeline using MLX
# Based on chapter_10.py data

import mlx.core as mlx
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

# --- Load and Prepare Data ---
# Using a small subset of the California housing data for demonstration
housing = fetch_california_housing(as_frame=False)
X, y = housing.data[:10], housing.target[:10]  # Using first 10 samples

# Normalize features (mean=0, std=1)
means = X.mean(axis=0)                        # Shape: (8,) 
stds = X.std(axis=0)                          # Shape: (8,)
X_normalized = mlx.array((X - means) / stds)   # Normalize features using proper MLX broadcasting

y_normalized = y.reshape(-1, 1)            # Reshape targets to column vector

print("Data shapes:")
print(f"Original X: {mlx.array(X).shape}, Original y: {mlx.array(y).reshape(-1, 1).shape}")
print(f"X_normalized (MLX array): {mlx.array(X_normalized).reshape(-1, 8).shape}")
print(f"y_mlx_normalized: {mlx.array(y).reshape(-1, 1).shape}")

# --- Define Linear Model and Optimizer ---
import mlx.nn as nn
from mlx.optimizers import optimizers as mlxopt

# Define linear model
def create_linear_model(input_dims):
    return nn.Linear(input_dims=input_dims, output_dims=1)

model = create_linear_model(X_normalized.shape[1])
print("Model created:", model)

# Define optimizer with learning rate
def create_optimizer(learning_rate):
    return mlxopt.SGD(learning_rate=learning_rate)

optimizer = create_optimizer(learning_rate=0.4)
print("Optimizer created:", optimizer)

# --- Training Function ---
def mlx_train_bgd(model, optimizer, loss_fn, X, y, n_epochs):
    for epoch in range(n_epochs):
        # Forward pass: compute predictions
        with mlx.eval():
            y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)

        print(f"Loss before gradient computation: {loss.item():.4f}")

        # Define gradient computation function
        def compute_grad(X):
            mse = loss_fn(model(X), y)
            # Compute gradient of the loss with respect to model parameters
            grad = mlx.nn.value_and_grad(mse)(model(X))
            return mlx.item(mlx.grad(model), 0) if mlx.is_leaf(mlx.grad(model)) else grad
            
        # Compute gradients and update parameters
        current_loss, grads = compute_grad(X)
        
        # Update model parameters
        params_to_update = mlx.nn.params(params=model)
        for param in params_to_update:
            optimizer.update(param, grads.item() if mlx.is_leaf(mlx.grad(model)) else mlx.item(mlx.grad(param), 0))

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {current_loss.item():.4f}")
    
    return model

# --- Test with Simple Data for Debugging ---
print("\n--- Testing with simple data ---")
simple_X = mlx.array([[1.0, 4.0], [2.0, 3.0]])
simple_y = mlx.array([[5.0], [6.0]])

simple_model = create_linear_model(2)
simple_optimizer = create_optimizer(learning_rate=0.4)

print("Training simple linear regression model:")
simple_model = mlx_train_bgd(simple_model, simple_optimizer, nn.losses.mse_loss, simple_X, simple_y, 20)

# --- Use the Simple Model for Predictions ---
simple_X_test = mlx.array([[1.0, 4.0], [3.0, 6.0]])

with mlx.eval():
    simple_predictions = simple_model(simple_X_test)
print("\nSimple model predictions:")
for i, pred in enumerate(simple_predictions):
    print(f"Sample {i+1}: {mlx.item(pred, 0):.4f}")

# --- Final Test with California Housing Data ---
print("\n--- Testing with normalized data ---")
x_test = np.array([[3.478, 5.172, 26.0], [4.032, 5.017, 28.6]])  # Simplified test data
x_test_mlx = mlx.array(x_test).reshape(-1, 3)  # Reshape to match normalized features

# Normalize test data using same means and stds
x_test_normalized = mlx.array((x_test - means) / stds)
print(f"Test X shape: {x_test_normalized.shape}")

with mlx.eval():
    predictions = model(x_test_normalized)
print("\nPredictions using trained model:")
for i, pred in enumerate(predictions):
    print(f"Test sample {i+1}: {mlx.item(pred, 0):.4f}")