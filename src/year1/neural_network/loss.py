"""
Phase 1.3 - Loss Functions
Covers: MSE (regression), Cross-Entropy (classification) + derivatives
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('docs/visualizations', exist_ok=True)


# --- MEAN SQUARED ERROR ---
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    """dL/dy_pred"""
    return 2 * (y_pred - y_true) / y_true.size


# --- CROSS-ENTROPY LOSS ---
def cross_entropy(y_pred, y_true):
    """
    y_pred: softmax probabilities (batch_size, num_classes)
    y_true: one-hot labels (batch_size, num_classes)
    """
    eps = 1e-9  # prevent log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

def cross_entropy_derivative(y_pred, y_true):
    """Gradient of cross-entropy w.r.t. softmax input (combined)"""
    return (y_pred - y_true) / y_true.shape[0]


# --- BINARY CROSS-ENTROPY ---
def binary_cross_entropy(y_pred, y_true):
    eps = 1e-9
    return -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )

def binary_cross_entropy_derivative(y_pred, y_true):
    eps = 1e-9
    return (-(y_true / (y_pred + eps)) + (1 - y_true) / (1 - y_pred + eps)) / y_true.size


if __name__ == "__main__":
    print("=== MSE LOSS ===")
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"MSE:    {mse(y_pred, y_true):.6f}")
    print(f"dMSE:   {mse_derivative(y_pred, y_true).round(4)}")

    print("\n=== CROSS-ENTROPY LOSS ===")
    # 3 samples, 3 classes
    y_true_ce = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=float)
    y_pred_ce = np.array([[0.7,0.2,0.1], [0.1,0.8,0.1], [0.1,0.2,0.7]], dtype=float)
    print(f"CE Loss: {cross_entropy(y_pred_ce, y_true_ce):.6f}")
    print(f"dCE:\n{cross_entropy_derivative(y_pred_ce, y_true_ce).round(4)}")

    print("\n=== BINARY CROSS-ENTROPY ===")
    y_true_b = np.array([1.0, 0.0, 1.0, 1.0])
    y_pred_b = np.array([0.9, 0.1, 0.8, 0.6])
    print(f"BCE Loss: {binary_cross_entropy(y_pred_b, y_true_b):.6f}")

    # --- VISUALIZE LOSS SURFACE ---
    w_vals = np.linspace(-2, 4, 100)
    # L(w) = MSE of w*x vs y, x=2, y=4 => minimum at w=2
    losses = [(w * 2 - 4) ** 2 for w in w_vals]

    plt.figure(figsize=(8, 4))
    plt.plot(w_vals, losses)
    plt.axvline(x=2.0, color='red', linestyle='--', label='minimum (w=2)')
    plt.title("MSE Loss Surface: L(w) = (2w - 4)^2")
    plt.xlabel("w")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('docs/visualizations/loss_surface.png', dpi=100)
    plt.close()
    print("\nSaved: docs/visualizations/loss_surface.png")
    print("\nDone: Loss functions complete.")
