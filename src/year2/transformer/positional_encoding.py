"""
Phase 2.4 - Positional Encoding
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('docs/visualizations', exist_ok=True)


def positional_encoding(max_seq_len, d_model):
    PE = np.zeros((max_seq_len, d_model))
    pos = np.arange(max_seq_len)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE


if __name__ == "__main__":
    PE = positional_encoding(max_seq_len=50, d_model=64)
    print(f"PE shape: {PE.shape}")
    print(f"PE[0,:8] = {PE[0,:8].round(4)}")
    print(f"PE[1,:8] = {PE[1,:8].round(4)}")

    # Visualize
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(PE, aspect='auto', cmap='RdBu')
    plt.colorbar()
    plt.title("Positional Encoding Matrix")
    plt.xlabel("Dimension")
    plt.ylabel("Position")

    plt.subplot(1, 2, 2)
    for i in [0, 1, 2, 4]:
        plt.plot(PE[:, i], label=f"dim {i}")
    plt.title("PE values across positions")
    plt.xlabel("Position")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('docs/visualizations/positional_encoding.png', dpi=100)
    plt.close()
    print("Saved: docs/visualizations/positional_encoding.png")
    print("\nDone: Positional encoding complete.")
