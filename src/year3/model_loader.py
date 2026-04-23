"""
Trained Model Loader
Loads the PyTorch checkpoint saved from Colab training
"""

import os
import json
import re
import numpy as np


def load_tokenizer(path):
    """Load tokenizer from saved JSON"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    w2i = data['w2i']
    i2w = {int(i): w for w, i in w2i.items()}
    return w2i, i2w, data['vocab_size']


def generate_text(model, w2i, i2w, prompt,
                  max_new=25, temp=0.8, top_k=40,
                  seq_len=64, device='cpu'):
    """Generate text from a trained PyTorch model"""
    import torch

    def tokenize(text):
        return re.findall(r"[a-z']+|[.,!?]", text.lower())

    ids  = [w2i.get(w, 1) for w in tokenize(prompt)]
    seen = {}

    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            x      = torch.tensor(
                [ids[-seq_len:]], dtype=torch.long).to(device)
            logits = model(x)[0, -1].cpu().numpy().astype(float)
            logits[0] = logits[1] = -1e9
            for tid, cnt in seen.items():
                logits[tid] -= 2.0 * cnt
            logits   = logits / temp
            top_idx  = np.argsort(logits)[-top_k:]
            probs    = np.exp(logits[top_idx] - np.max(logits[top_idx]))
            probs   /= probs.sum()
            nxt      = int(np.random.choice(top_idx, p=probs))
            seen[nxt] = seen.get(nxt, 0) + 1
            ids.append(nxt)

    prompt_len = len([w2i.get(w, 1) for w in tokenize(prompt)])
    return ' '.join(
        i2w.get(i, '') for i in ids[prompt_len:]
        if i > 1 and i2w.get(i, ''))


def load_trained_model(language='english', device='cpu'):
    """
    Load a trained model checkpoint.
    Returns (model, w2i, i2w) or None if not found.
    """
    checkpoint_path = f'models/checkpoints/{language}_best.pt'
    tokenizer_path  = f'models/{language}/tokenizer.json'

    if not os.path.exists(checkpoint_path):
        return None, None, None, None

    try:
        import torch
        import torch.nn as nn

        ckpt   = torch.load(checkpoint_path, map_location=device)
        config = ckpt['config']

        # Rebuild model architecture
        class FNITransformer(nn.Module):
            def __init__(self, vocab_size, d_model=256, num_heads=8,
                         d_ff=1024, num_layers=4,
                         max_seq_len=64, dropout=0.1):
                super().__init__()
                self.vocab_size = vocab_size
                self.embedding  = nn.Embedding(
                    vocab_size, d_model, padding_idx=0)
                self.pos_emb    = nn.Embedding(max_seq_len, d_model)
                self.drop       = nn.Dropout(dropout)
                enc = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=num_heads,
                    dim_feedforward=d_ff, dropout=dropout,
                    batch_first=True, norm_first=True)
                self.transformer = nn.TransformerEncoder(
                    enc, num_layers=num_layers)
                self.norm        = nn.LayerNorm(d_model)
                self.out         = nn.Linear(d_model, vocab_size, bias=False)
                self.out.weight  = self.embedding.weight

            def forward(self, x):
                T   = x.shape[1]
                pos = torch.arange(T, device=x.device).unsqueeze(0)
                x   = self.drop(self.embedding(x) + self.pos_emb(pos))
                x   = self.transformer(x)
                return self.out(self.norm(x))

        model = FNITransformer(
            vocab_size  = config['vocab_size'],
            d_model     = config['d_model'],
            num_heads   = config['num_heads'],
            d_ff        = config['d_ff'],
            num_layers  = config['num_layers'],
            max_seq_len = config['max_seq_len']
        ).to(device)

        model.load_state_dict(ckpt['model'])
        model.eval()

        # Load tokenizer
        w2i = ckpt.get('vocab', {})
        i2w = {int(i): w for w, i in w2i.items()}

        # Try loading from separate tokenizer file if available
        if os.path.exists(tokenizer_path):
            w2i, i2w, _ = load_tokenizer(tokenizer_path)

        return model, w2i, i2w, config

    except Exception as e:
        print(f'Error loading model: {e}')
        return None, None, None, None
