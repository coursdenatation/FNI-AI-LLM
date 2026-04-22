"""
Phase 1.3 - Backpropagation
Covers: chain rule, gradient flow, numerical gradient checking
"""

import numpy as np
from src.year1.neural_network.activations import sigmoid, sigmoid_derivative
from src.year1.neural_network.loss import mse, mse_derivative


def numerical_gradient(f, params, h=1e-5):
    """
    Compute numerical gradient of f w.r.t. params using finite differences.
    Used to verify analytical gradients are correct.
    """
    grad = np.zeros_like(params)
    it = np.nditer(params, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original = params[idx]

        params[idx] = original + h
        f_plus = f(params)

        params[idx] = original - h
        f_minus = f(params)

        grad[idx] = (f_plus - f_minus) / (2 * h)
        params[idx] = original
        it.iternext()
    return grad


def gradient_check(layer, X, y_true, loss_fn, loss_grad_fn, threshold=1e-5):
    """
    Compare analytical gradients from backprop vs numerical gradients.
    Returns True if they match within threshold.
    """
    # Forward + backward pass
    y_pred = layer.forward(X)
    loss_grad = loss_grad_fn(y_pred, y_true)
    layer.backward(loss_grad)
    analytical_dW = layer.dW.copy()
    analytical_db = layer.db.copy()

    # Numerical gradient for W
    def loss_from_W(W):
        layer.W = W
        pred = layer.forward(X)
        return loss_fn(pred, y_true)

    numerical_dW = numerical_gradient(loss_from_W, layer.W.copy())

    # Numerical gradient for b
    def loss_from_b(b):
        layer.b = b
        pred = layer.forward(X)
        return loss_fn(pred, y_true)

    numerical_db = numerical_gradient(loss_from_b, layer.b.copy())

    # Compare
    dW_diff = np.max(np.abs(analytical_dW - numerical_dW))
    db_diff = np.max(np.abs(analytical_db - numerical_db))

    print(f"Max dW difference: {dW_diff:.2e} (threshold: {threshold:.2e})")
    print(f"Max db difference: {db_diff:.2e} (threshold: {threshold:.2e})")
    passed = dW_diff < threshold and db_diff < threshold
    print(f"Gradient check: {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    from src.year1.neural_network.layers import DenseLayer

    np.random.seed(0)

    print("=== GRADIENT CHECK: Single Layer ===")
    layer = DenseLayer(3, 2, activation='sigmoid')
    X = np.random.randn(4, 3)
    y = np.random.randn(4, 2)
    gradient_check(layer, X, y, mse, mse_derivative)

    print("\n=== MANUAL BACKPROP WALKTHROUGH ===")
    # Simple 1-layer network: y = sigmoid(X @ W + b)
    # Loss = MSE(y_pred, y_true)
    np.random.seed(1)
    W = np.random.randn(2, 1) * 0.1
    b = np.zeros(1)
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_true = np.array([[0.0], [1.0], [1.0], [0.0]])  # XOR

    for step in range(5):
        # Forward
        z = X @ W + b
        y_pred = sigmoid(z)
        loss = mse(y_pred, y_true)

        # Backward (chain rule)
        dL_dy   = mse_derivative(y_pred, y_true)     # dL/dy_pred
        dy_dz   = sigmoid_derivative(z)               # dy/dz
        delta   = dL_dy * dy_dz                       # dL/dz
        dW      = X.T @ delta                         # dL/dW
        db      = delta.sum(axis=0)                   # dL/db

        # Update
        W -= 0.1 * dW
        b -= 0.1 * db

        print(f"Step {step+1}: loss={loss:.6f}")

    print("\nDone: Backprop complete.")
