"""
Phase 1.3 - Activation Functions
Covers: ReLU, Sigmoid, Tanh, Softmax + their derivatives for backprop
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('docs/visualizations', exist_ok=True)


# --- RELU ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# --- SIGMOID ---
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# --- TANH ---
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# --- SOFTMAX ---
def softmax(x):
    # subtract max for numerical stability
    e = np.exp(x - np.max(x))
    return e / e.sum()

def softmax_batch(x):
    """Softmax for a batch (2D array, each row is one sample)"""
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    x = np.linspace(-5, 5, 11)
    print("=== ACTIVATION FUNCTIONS ===")
    print(f"x       = {x}")
    print(f"relu    = {relu(x)}")
    print(f"sigmoid = {sigmoid(x).round(4)}")
    print(f"tanh    = {tanh(x).round(4)}")

    print("\n=== DERIVATIVES ===")
    print(f"relu'    = {relu_derivative(x)}")
    print(f"sigmoid' = {sigmoid_derivative(x).round(4)}")
    print(f"tanh'    = {tanh_derivative(x).round(4)}")

    print("\n=== SOFTMAX ===")
    logits = np.array([2.0, 1.0, 0.5, -1.0])
    probs = softmax(logits)
    print(f"logits = {logits}")
    print(f"probs  = {probs.round(4)}")
    print(f"sum    = {probs.sum():.4f} (should be 1.0)")

    # Batch softmax
    batch = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.5]])
    print(f"\nBatch softmax:\n{softmax_batch(batch).round(4)}")

    # --- VISUALIZATION ---
    x_plot = np.linspace(-5, 5, 200)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    pairs = [
        (relu,    relu_derivative,    "ReLU"),
        (sigmoid, sigmoid_derivative, "Sigmoid"),
        (tanh,    tanh_derivative,    "Tanh"),
    ]

    for i, (f, df, name) in enumerate(pairs):
        axes[0][i].plot(x_plot, f(x_plot), color='blue')
        axes[0][i].set_title(f"{name}")
        axes[0][i].axhline(0, color='black', linewidth=0.5)
        axes[0][i].grid(True)

        axes[1][i].plot(x_plot, df(x_plot), color='orange')
        axes[1][i].set_title(f"{name} derivative")
        axes[1][i].axhline(0, color='black', linewidth=0.5)
        axes[1][i].grid(True)

    # Softmax bar chart
    axes[0][3].bar(range(len(logits)), softmax(logits), color='green')
    axes[0][3].set_title("Softmax output")
    axes[0][3].set_xlabel("Class")
    axes[1][3].axis('off')

    plt.tight_layout()
    plt.savefig('docs/visualizations/activations.png', dpi=100)
    plt.close()
    print("\nSaved: docs/visualizations/activations.png")
    print("\nDone: Activations complete.")
