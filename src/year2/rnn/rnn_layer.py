"""
Phase 2.3 - RNN Layer
Formula: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
"""

import numpy as np


class RNNCell:
    def __init__(self, input_size, hidden_size):
        scale = np.sqrt(1.0 / hidden_size)
        self.W_x = np.random.randn(input_size, hidden_size) * scale
        self.W_h = np.random.randn(hidden_size, hidden_size) * scale
        self.b   = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h_prev):
        self.x      = x
        self.h_prev = h_prev
        self.z      = x @ self.W_x + h_prev @ self.W_h + self.b
        self.h      = np.tanh(self.z)
        return self.h

    def backward(self, dh, learning_rate=0.01):
        dz     = dh * (1 - self.h ** 2)          # tanh derivative
        dW_x   = self.x.T @ dz
        dW_h   = self.h_prev.T @ dz
        db     = dz.sum(axis=0)
        dx     = dz @ self.W_x.T
        dh_prev = dz @ self.W_h.T

        self.W_x -= learning_rate * dW_x
        self.W_h -= learning_rate * dW_h
        self.b   -= learning_rate * db
        return dx, dh_prev


class RNNLayer:
    """Processes a full sequence through an RNN cell"""

    def __init__(self, input_size, hidden_size):
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, X, h0=None):
        """
        X: (batch, seq_len, input_size)
        Returns: all hidden states (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = X.shape
        if h0 is None:
            h0 = np.zeros((batch, self.hidden_size))

        h = h0
        self.hidden_states = [h]
        self.inputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            self.inputs.append(x_t)
            h = self.cell.forward(x_t, h)
            self.hidden_states.append(h)

        return np.stack(self.hidden_states[1:], axis=1)

    def __repr__(self):
        return (f"RNNLayer(input={self.cell.W_x.shape[0]}, "
                f"hidden={self.hidden_size})")


if __name__ == "__main__":
    np.random.seed(42)

    print("=== RNN CELL ===")
    cell = RNNCell(input_size=4, hidden_size=8)
    x = np.random.randn(2, 4)       # batch=2, input=4
    h = np.zeros((2, 8))
    h_new = cell.forward(x, h)
    print(f"Input shape:  {x.shape}")
    print(f"Hidden shape: {h_new.shape}")
    print(f"Hidden values (first sample): {h_new[0].round(4)}")

    print("\n=== RNN LAYER (sequence) ===")
    layer = RNNLayer(input_size=4, hidden_size=8)
    print(layer)
    X = np.random.randn(2, 5, 4)    # batch=2, seq_len=5, input=4
    out = layer.forward(X)
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Final hidden state: {out[:, -1, :].round(4)}")

    print("\n=== HIDDEN STATE EVOLUTION ===")
    layer2 = RNNLayer(input_size=2, hidden_size=3)
    X2 = np.random.randn(1, 6, 2)
    out2 = layer2.forward(X2)
    for t in range(6):
        print(f"  t={t}: {out2[0, t].round(4)}")

    print("\nDone: RNN layer complete.")
