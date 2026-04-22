"""
Phase 2.4 - Scaled Dot-Product Attention
Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V
"""

import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq, d_k)
    K: (batch, heads, seq, d_k)
    V: (batch, heads, seq, d_v)
    Returns: output (batch, heads, seq, d_v), weights (batch, heads, seq, seq)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    weights = softmax(scores, axis=-1)
    output  = weights @ V
    return output, weights


if __name__ == "__main__":
    np.random.seed(42)
    batch, heads, seq, d_k = 2, 4, 5, 16

    Q = np.random.randn(batch, heads, seq, d_k)
    K = np.random.randn(batch, heads, seq, d_k)
    V = np.random.randn(batch, heads, seq, d_k)

    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Q shape:       {Q.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum (should be 1.0): {weights[0,0,0].sum():.4f}")
    print(f"Sample weights row 0:\n{weights[0,0,0].round(4)}")
    print("\nDone: Attention complete.")
