"""
Phase 1.4 - Batch Processing & Gradient Descent Variants
Covers: full-batch, mini-batch, stochastic gradient descent + convergence comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

os.makedirs('docs/visualizations', exist_ok=True)

from src.year1.neural_network.network import NeuralNetwork
from src.year1.neural_network.loss import mse, mse_derivative


def get_batches(X, y, batch_size):
    """Split X and y into mini-batches"""
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


def train_with_batch_size(X, y, batch_size, epochs=200, lr=0.05):
    """Train network with a given batch size, return loss history and time"""
    np.random.seed(42)
    nn = NeuralNetwork([X.shape[1], 8, 1], ['relu', 'sigmoid'])
    loss_history = []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        epoch_losses = []
        for X_batch, y_batch in get_batches(X, y, batch_size):
            y_pred = nn.forward(X_batch)
            loss = mse(y_pred, y_batch)
            epoch_losses.append(loss)
            grad = mse_derivative(y_pred, y_batch)
            nn.backward(grad)
            nn.update(lr)
        loss_history.append(np.mean(epoch_losses))

    elapsed = time.perf_counter() - t0
    return loss_history, elapsed, nn


if __name__ == "__main__":
    np.random.seed(42)

    # Generate a larger synthetic dataset
    N = 500
    X = np.random.randn(N, 2)
    # Label: 1 if x1^2 + x2^2 < 1 (inside unit circle), else 0
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(float).reshape(-1, 1)
    print(f"Dataset: {N} samples, {int(y.sum())} positive, {int((1-y).sum())} negative")

    print("\n=== BATCH SIZE COMPARISON ===")
    batch_sizes = {
        "SGD (batch=1)":        1,
        "Mini-batch (batch=32)": 32,
        "Full-batch (batch=500)": 500,
    }

    results = {}
    for name, bs in batch_sizes.items():
        history, elapsed, nn = train_with_batch_size(X, y, batch_size=bs)
        final_loss = history[-1]
        preds = np.round(nn.predict(X))
        acc = np.mean(preds == y)
        results[name] = history
        print(f"{name:30s} | loss={final_loss:.4f} | acc={acc:.2%} | time={elapsed:.3f}s")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 5))
    for name, history in results.items():
        plt.plot(history, label=name)
    plt.title("Loss Convergence: Different Batch Sizes")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('docs/visualizations/batch_convergence.png', dpi=100)
    plt.close()
    print("\nSaved: docs/visualizations/batch_convergence.png")

    # --- MEMORY USAGE COMPARISON ---
    print("\n=== MEMORY EFFICIENCY ===")
    for bs in [1, 32, 128, 500]:
        batches = list(get_batches(X, y, bs))
        print(f"batch_size={bs:4d}: {len(batches):4d} batches per epoch, "
              f"each batch shape: {batches[0][0].shape}")

    print("\nDone: Batch training complete.")
