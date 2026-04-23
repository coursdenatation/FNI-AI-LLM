"""
Phase 3.4 - Text Dataset
Loads tokenized text and creates (input, target) pairs for language modeling
Target: predict the next token given the previous tokens
"""

import numpy as np
import os
from src.year3.tokenization.african_tokenizers import CameroonTokenizer


class TextDataset:
    """
    Language modeling dataset.
    For each sequence of length seq_len:
      input  = tokens[0:seq_len]
      target = tokens[1:seq_len+1]  (next token prediction)
    """

    def __init__(self, language, seq_len=32, max_vocab=500):
        self.language = language
        self.seq_len  = seq_len
        self.tokenizer = CameroonTokenizer(language=language)
        self.data = []
        self.vocab_size = 0

    def load(self, corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            corpus = f.read()

        self.tokenizer.build_vocab(corpus, max_vocab=500)
        self.vocab_size = self.tokenizer.vocab_size

        # Encode entire corpus as one long sequence
        all_ids = self.tokenizer.encode(corpus)

        # Create overlapping windows of seq_len+1
        self.data = []
        for i in range(0, len(all_ids) - self.seq_len, self.seq_len // 2):
            chunk = all_ids[i: i + self.seq_len + 1]
            if len(chunk) == self.seq_len + 1:
                self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = np.array(chunk[:-1])   # input tokens
        y = np.array(chunk[1:])    # target tokens (shifted by 1)
        return x, y

    def __repr__(self):
        return (f"TextDataset(language={self.language}, "
                f"samples={len(self.data)}, "
                f"seq_len={self.seq_len}, "
                f"vocab={self.vocab_size})")


if __name__ == "__main__":
    DATA_ROOT = "data/cameroon_languages"
    LANGUAGES = ["english", "french", "bayangi", "douala"]

    print("=== TEXT DATASETS ===\n")
    for lang in LANGUAGES:
        path = os.path.join(DATA_ROOT, lang, "processed", f"{lang}_clean.txt")
        if not os.path.exists(path):
            print(f"{lang}: file not found")
            continue

        ds = TextDataset(language=lang, seq_len=32)
        ds.load(path)
        print(ds)

        if len(ds) > 0:
            x, y = ds[0]
            print(f"  Sample input:  {x[:8]}...")
            print(f"  Sample target: {y[:8]}...")
            decoded_x = ds.tokenizer.decode(x[:8].tolist())
            decoded_y = ds.tokenizer.decode(y[:8].tolist())
            print(f"  Input text:    '{decoded_x}'")
            print(f"  Target text:   '{decoded_y}'")
        print()

    print("Done: Dataset complete.")
