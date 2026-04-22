"""
Phase 1.3 - Single Neuron
Formula: output = sum(input[i] * weight[i]) + bias
"""

import numpy as np


class Neuron:
    """A single artificial neuron"""

    def __init__(self, input_size):
        np.random.seed(42)
        self.weights = np.random.randn(input_size) * 0.1
        self.bias = 0.0
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        self.last_input = np.array(x)
        self.last_output = np.dot(self.weights, self.last_input) + self.bias
        return self.last_output

    def update(self, grad_output, learning_rate):
        grad_weights = grad_output * self.last_input
        grad_bias = grad_output
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

    def __repr__(self):
        return f"Neuron(inputs={len(self.weights)}, bias={self.bias:.4f})"


if __name__ == "__main__":
    neuron = Neuron(input_size=3)
    print(neuron)

    x = np.array([1.0, 2.0, 3.0])
    out = neuron.forward(x)
    print(f"Input:  {x}")
    print(f"Output: {out:.4f}")

    # Simulate one weight update
    neuron.update(grad_output=0.5, learning_rate=0.01)
    print(f"After update: {neuron}")
    print("\nDone: Neuron complete.")
