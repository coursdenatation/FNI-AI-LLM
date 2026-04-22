"""
Phase 2.3 - GRU (Gated Recurrent Unit)
Simpler than LSTM: reset gate + update gate, no separate cell state
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class GRUCell:
    def __init__(self, input_size, hidden_size):
        scale = np.sqrt(1.0 / hidden_size)
        # Reset and update gates
        self.W_r = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.W_z = np.random.randn(input_size + hidden_size, hidden_size) * scale
        # Candidate hidden state
        self.W_n = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_r = np.zeros(hidden_size)
        self.b_z = np.zeros(hidden_size)
        self.b_n = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h_prev):
        combined = np.concatenate([x, h_prev], axis=1)

        r = sigmoid(combined @ self.W_r + self.b_r)   # reset gate
        z = sigmoid(combined @ self.W_z + self.b_z)   # update gate

        combined_r = np.concatenate([x, r * h_prev], axis=1)
        n = np.tanh(combined_r @ self.W_n + self.b_n) # candidate

        h = (1 - z) * n + z * h_prev                  # new hidden state

        self.r, self.z, self.n, self.h = r, z, n, h
        return h

    def __repr__(self):
        return f"GRUCell(hidden_size={self.hidden_size})"


class GRULayer:
    def __init__(self, input_size, hidden_size):
        self.cell = GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, X, h0=None):
        batch, seq_len, _ = X.shape
        if h0 is None:
            h0 = np.zeros((batch, self.hidden_size))

        h = h0
        hidden_states = []
        for t in range(seq_len):
            h = self.cell.forward(X[:, t, :], h)
            hidden_states.append(h)

        return np.stack(hidden_states, axis=1)

    def __repr__(self):
        return f"GRULayer(hidden_size={self.hidden_size})"


if __name__ == "__main__":
    np.random.seed(42)

    print("=== GRU CELL ===")
    cell = GRUCell(input_size=4, hidden_size=8)
    print(cell)
    x = np.random.randn(2, 4)
    h = np.zeros((2, 8))
    h_new = cell.forward(x, h)
    print(f"Reset gate:  {cell.r[0].round(4)}")
    print(f"Update gate: {cell.z[0].round(4)}")
    print(f"Output h:    {h_new[0].round(4)}")

    print("\n=== GRU vs LSTM PARAMETER COUNT ===")
    lstm_params = (4 + 8) * (4 * 8) + (4 * 8)
    gru_params  = (4 + 8) * (3 * 8) + (3 * 8)
    print(f"LSTM params: {lstm_params}")
    print(f"GRU params:  {gru_params}")
    print(f"GRU is {lstm_params/gru_params:.2f}x smaller")

    print("\n=== GRU LAYER ===")
    layer = GRULayer(input_size=4, hidden_size=8)
    print(layer)
    X = np.random.randn(2, 6, 4)
    out = layer.forward(X)
    print(f"Input:  {X.shape}")
    print(f"Output: {out.shape}")

    print("\nDone: GRU complete.")
