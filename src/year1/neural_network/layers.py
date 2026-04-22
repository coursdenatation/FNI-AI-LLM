"""
Phase 1.3 - Layers
Covers: DenseLayer with weight matrix, bias vector, forward pass
"""

import numpy as np
from src.year1.neural_network.activations import relu, sigmoid, tanh


class DenseLayer:
    """
    Fully connected layer.
    forward: output = activation(X @ W + b)
    """

    def __init__(self, input_size, output_size, activation='relu'):
        # He initialization for relu, Xavier for others
        if activation == 'relu':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros(output_size)
        self.activation_name = activation

        # Cache for backprop
        self.last_input = None
        self.last_z = None
        self.last_output = None

        # Gradients
        self.dW = None
        self.db = None

    def _activate(self, z):
        if self.activation_name == 'relu':
            return relu(z)
        elif self.activation_name == 'sigmoid':
            return sigmoid(z)
        elif self.activation_name == 'tanh':
            return tanh(z)
        elif self.activation_name == 'none':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def _activate_derivative(self, z):
        from src.year1.neural_network.activations import (
            relu_derivative, sigmoid_derivative, tanh_derivative
        )
        if self.activation_name == 'relu':
            return relu_derivative(z)
        elif self.activation_name == 'sigmoid':
            return sigmoid_derivative(z)
        elif self.activation_name == 'tanh':
            return tanh_derivative(z)
        elif self.activation_name == 'none':
            return np.ones_like(z)

    def forward(self, x):
        self.last_input = x
        self.last_z = x @ self.W + self.b
        self.last_output = self._activate(self.last_z)
        return self.last_output

    def backward(self, grad_output):
        """
        grad_output: gradient from next layer (dL/d_output)
        returns: gradient to pass to previous layer (dL/d_input)
        """
        act_grad = self._activate_derivative(self.last_z)
        delta = grad_output * act_grad

        self.dW = self.last_input.T @ delta
        self.db = delta.sum(axis=0)

        return delta @ self.W.T

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def __repr__(self):
        return (f"DenseLayer({self.W.shape[0]} -> {self.W.shape[1]}, "
                f"activation={self.activation_name})")


if __name__ == "__main__":
    np.random.seed(42)

    # Single layer test
    layer = DenseLayer(input_size=3, output_size=4, activation='relu')
    print(layer)

    # Batch of 2 samples, 3 features each
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    out = layer.forward(X)
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output:\n{out.round(4)}")

    # Stack 2 layers and verify dimensions propagate
    print("\n=== LAYER STACKING ===")
    l1 = DenseLayer(4, 8, activation='relu')
    l2 = DenseLayer(8, 4, activation='relu')
    l3 = DenseLayer(4, 2, activation='sigmoid')

    x = np.random.randn(5, 4)   # batch of 5
    out = l3.forward(l2.forward(l1.forward(x)))
    print(f"Input:  {x.shape}")
    print(f"After l1: {l1.last_output.shape}")
    print(f"After l2: {l2.last_output.shape}")
    print(f"After l3: {out.shape}")
    print(f"Output values (should be 0-1 from sigmoid):\n{out.round(4)}")

    print("\nDone: Layers complete.")
