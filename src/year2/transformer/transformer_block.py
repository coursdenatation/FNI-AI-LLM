"""
Phase 2.4 - Transformer Block
Multi-head attention + Add & Norm + Feed-forward + Add & Norm
"""

import numpy as np
from src.year2.transformer.attention import scaled_dot_product_attention, softmax


def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model  = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        scale = np.sqrt(1.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x, batch):
        # (batch, seq, d_model) -> (batch, heads, seq, d_k)
        x = x.reshape(batch, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        batch = Q.shape[0]

        Q = Q @ self.W_Q
        K = K @ self.W_K
        V = V @ self.W_V

        Q = self.split_heads(Q, batch)
        K = self.split_heads(K, batch)
        V = self.split_heads(V, batch)

        out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(batch, -1, self.d_model)
        return out @ self.W_O

    def __repr__(self):
        return f"MultiHeadAttention(d_model={self.d_model}, heads={self.num_heads})"


class FeedForward:
    def __init__(self, d_model, d_ff):
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff   = FeedForward(d_model, d_ff)
        self.d_model = d_model

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out = self.attn.forward(x, x, x, mask)
        x = layer_norm(x + attn_out)

        # Feed-forward + residual + norm
        ff_out = self.ff.forward(x)
        x = layer_norm(x + ff_out)
        return x

    def count_params(self):
        mha = (self.d_model ** 2) * 4
        ff  = self.d_model * self.ff.W1.shape[1] * 2
        return mha + ff

    def __repr__(self):
        return (f"TransformerBlock(d_model={self.d_model}, "
                f"heads={self.attn.num_heads}, params={self.count_params():,})")


if __name__ == "__main__":
    np.random.seed(42)

    print("=== MULTI-HEAD ATTENTION ===")
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    print(mha)
    x = np.random.randn(2, 10, 64)   # batch=2, seq=10, d_model=64
    out = mha.forward(x, x, x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Attn weights shape: {mha.attn_weights.shape}")

    print("\n=== TRANSFORMER BLOCK ===")
    block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
    print(block)
    out = block.forward(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    print("\n=== LAYER NORM ===")
    sample = np.array([[1.0, 2.0, 3.0, 4.0]])
    normed = layer_norm(sample)
    print(f"Input:  {sample}")
    print(f"Normed: {normed.round(4)}")
    print(f"Mean: {normed.mean():.6f}, Std: {normed.std():.4f}")

    print("\nDone: Transformer block complete.")
