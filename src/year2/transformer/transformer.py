"""
Phase 2.4 - Full Transformer Model
Embedding + Positional Encoding + N Transformer Blocks + Output Layer
"""

import numpy as np
from src.year2.transformer.positional_encoding import positional_encoding
from src.year2.transformer.transformer_block import TransformerBlock


CONFIGS = {
    "small": dict(vocab_size=1000, d_model=64,  num_heads=4, d_ff=256,  num_layers=2),
    "base":  dict(vocab_size=5000, d_model=128, num_heads=8, d_ff=512,  num_layers=4),
    "large": dict(vocab_size=10000,d_model=256, num_heads=8, d_ff=1024, num_layers=6),
}


class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=512):
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding: vocab_size -> d_model (Xavier init)
        scale = np.sqrt(1.0 / d_model)
        self.embedding = np.random.randn(vocab_size, d_model) * scale

        # Positional encoding
        self.PE = positional_encoding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Output projection: d_model -> vocab_size
        self.W_out = np.random.randn(d_model, vocab_size) * scale

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len) integer array
        Returns: logits (batch, seq_len, vocab_size)
        """
        seq_len = token_ids.shape[1]

        # Embed tokens
        x = self.embedding[token_ids]           # (batch, seq, d_model)
        x = x + self.PE[:seq_len]               # add positional encoding

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Project to vocab
        logits = x @ self.W_out                 # (batch, seq, vocab_size)
        return logits

    def count_params(self):
        emb    = self.vocab_size * self.d_model
        blocks = sum(b.count_params() for b in self.blocks)
        out    = self.d_model * self.vocab_size
        return emb + blocks + out

    def __repr__(self):
        return (f"Transformer(vocab={self.vocab_size}, d_model={self.d_model}, "
                f"layers={len(self.blocks)}, params={self.count_params():,})")


if __name__ == "__main__":
    np.random.seed(42)

    print("=== TRANSFORMER CONFIGS ===")
    for name, cfg in CONFIGS.items():
        t = Transformer(**cfg)
        print(f"  {name:6s}: {t}")

    print("\n=== FORWARD PASS ===")
    model = Transformer(**CONFIGS["small"])
    batch_size, seq_len = 2, 10
    token_ids = np.random.randint(0, CONFIGS["small"]["vocab_size"],
                                  (batch_size, seq_len))
    logits = model.forward(token_ids)
    print(f"Input tokens: {token_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Sample logits (first token): {logits[0,0,:5].round(4)}")

    # Greedy next-token prediction
    next_tokens = logits.argmax(axis=-1)
    print(f"Predicted next tokens: {next_tokens}")

    print("\nDone: Transformer complete.")
