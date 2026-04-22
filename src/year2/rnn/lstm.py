"""
Phase 2.3 - LSTM (Long Short-Term Memory)
Gates: forget, input, output + cell state
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        scale = np.sqrt(1.0 / hidden_size)
        # Combined weight matrix for all 4 gates [f, i, g, o]
        self.W = np.random.randn(input_size + hidden_size,
                                  4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h_prev, c_prev):
        self.x      = x
        self.h_prev = h_prev
        self.c_prev = c_prev

        combined = np.concatenate([x, h_prev], axis=1)
        gates = combined @ self.W + self.b

        H = self.hidden_size
        f = sigmoid(gates[:, :H])           # forget gate
        i = sigmoid(gates[:, H:2*H])        # input gate
        g = np.tanh(gates[:, 2*H:3*H])      # cell gate
        o = sigmoid(gates[:, 3*H:])         # output gate

        self.f, self.i, self.g, self.o = f, i, g, o

        c = f * c_prev + i * g              # new cell state
        h = o * np.tanh(c)                  # new hidden state

        self.c = c
        self.h = h
        return h, c

    def __repr__(self):
        return f"LSTMCell(hidden_size={self.hidden_size})"


class LSTMLayer:
    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, X, h0=None, c0=None):
        batch, seq_len, _ = X.shape
        if h0 is None:
            h0 = np.zeros((batch, self.hidden_size))
        if c0 is None:
            c0 = np.zeros((batch, self.hidden_size))

        h, c = h0, c0
        hidden_states = []

        for t in range(seq_len):
            h, c = self.cell.forward(X[:, t, :], h, c)
            hidden_states.append(h)

        return np.stack(hidden_states, axis=1), (h, c)

    def __repr__(self):
        return f"LSTMLayer(hidden_size={self.hidden_size})"


if __name__ == "__main__":
    np.random.seed(42)

    print("=== LSTM CELL ===")
    cell = LSTMCell(input_size=4, hidden_size=8)
    print(cell)
    x = np.random.randn(2, 4)
    h = np.zeros((2, 8))
    c = np.zeros((2, 8))
    h_new, c_new = cell.forward(x, h, c)
    print(f"h shape: {h_new.shape}, c shape: {c_new.shape}")
    print(f"Forget gate (sample 0): {cell.f[0].round(4)}")
    print(f"Input gate (sample 0):  {cell.i[0].round(4)}")
    print(f"Output gate (sample 0): {cell.o[0].round(4)}")

    print("\n=== LSTM LAYER (sequence) ===")
    layer = LSTMLayer(input_size=4, hidden_size=8)
    print(layer)
    X = np.random.randn(2, 6, 4)
    out, (h_final, c_final) = layer.forward(X)
    print(f"Input:  {X.shape}")
    print(f"Output: {out.shape}")
    print(f"Final h: {h_final.shape}, Final c: {c_final.shape}")

    print("\n=== SIMPLE SEQUENCE LEARNING ===")
    # Train LSTM to predict next value in sequence [0,1,0,1,...]
    layer2 = LSTMLayer(input_size=1, hidden_size=4)
    X_seq = np.array([[[0.0],[1.0],[0.0],[1.0],[0.0]]])
    out2, _ = layer2.forward(X_seq)
    print(f"Sequence output shape: {out2.shape}")
    print(f"Hidden states:\n{out2[0].round(4)}")

    print("\nDone: LSTM complete.")
