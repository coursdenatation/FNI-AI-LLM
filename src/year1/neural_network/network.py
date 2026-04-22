"""
Phase 1.3 - Complete Neural Network
Covers: NeuralNetwork class, training loop, XOR problem, decision boundary
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('docs/visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

from src.year1.neural_network.layers import DenseLayer
from src.year1.neural_network.loss import mse, mse_derivative, binary_cross_entropy


class NeuralNetwork:
    """
    A fully connected neural network built from DenseLayer components.
    Supports variable depth, training with gradient descent, and prediction.
    """

    def __init__(self, layer_sizes, activations):
        """
        layer_sizes: list of ints e.g. [2, 4, 1]
        activations: list of strings e.g. ['relu', 'sigmoid']
        """
        assert len(activations) == len(layer_sizes) - 1
        self.layers = [
            DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            for i in range(len(activations))
        ]
        self.loss_history = []

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        self.loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = mse(y_pred, y)
            self.loss_history.append(loss)

            grad = mse_derivative(y_pred, y)
            self.backward(grad)
            self.update(learning_rate)

            if verbose and (epoch + 1) % (epochs // 5) == 0:
                print(f"Epoch {epoch+1:5d}/{epochs}: loss={loss:.6f}")

    def predict(self, X):
        return self.forward(X)

    def summary(self):
        print("NeuralNetwork Summary:")
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = layer.W.size + layer.b.size
            total_params += params
            print(f"  Layer {i+1}: {layer} | params={params}")
        print(f"  Total parameters: {total_params}")


if __name__ == "__main__":
    np.random.seed(42)

    # --- XOR PROBLEM ---
    # XOR cannot be solved by a single layer (not linearly separable)
    # Requires at least one hidden layer
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    print("=== XOR PROBLEM ===")
    print("Input -> Expected Output")
    for xi, yi in zip(X, y):
        print(f"  {xi} -> {yi[0]}")

    # Build network: 2 inputs -> 4 hidden (relu) -> 1 output (sigmoid)
    nn = NeuralNetwork(
        layer_sizes=[2, 4, 1],
        activations=['relu', 'sigmoid']
    )
    nn.summary()

    print("\nTraining...")
    nn.train(X, y, epochs=5000, learning_rate=0.1, verbose=True)

    print("\nPredictions after training:")
    preds = nn.predict(X)
    for xi, yi, pi in zip(X, y, preds):
        print(f"  {xi} -> expected={yi[0]}, predicted={pi[0]:.4f}, "
              f"rounded={round(pi[0])}")

    # Verify XOR learned
    rounded = np.round(preds).flatten()
    correct = np.sum(rounded == y.flatten())
    print(f"\nAccuracy: {correct}/4 correct")

    # --- LOSS CURVE ---
    plt.figure(figsize=(8, 4))
    plt.plot(nn.loss_history)
    plt.title("Training Loss - XOR Problem")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('docs/visualizations/xor_loss.png', dpi=100)
    plt.close()
    print("\nSaved: docs/visualizations/xor_loss.png")

    # --- DECISION BOUNDARY ---
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                         np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = nn.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, zz, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='Output')
    colors = ['blue' if yi == 0 else 'red' for yi in y.flatten()]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, zorder=5, edgecolors='black')
    plt.title("XOR Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig('docs/visualizations/xor_decision_boundary.png', dpi=100)
    plt.close()
    print("Saved: docs/visualizations/xor_decision_boundary.png")

    print("\nDone: Neural network complete.")
