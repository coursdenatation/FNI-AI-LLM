"""
Phase 2.2 - Vocabulary Management
Builds, saves and loads vocabulary with special tokens and frequency tracking
"""

import json
import re
from collections import Counter


class Vocabulary:
    PAD   = "<PAD>"
    UNK   = "<UNK>"
    START = "<START>"
    END   = "<END>"
    SPECIAL_TOKENS = [PAD, UNK, START, END]

    def __init__(self, max_vocab=None):
        self.max_vocab = max_vocab
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_freq = {}
        self.vocab_size = 0

    def build(self, corpus):
        tokens = re.findall(r"\b\w+\b", corpus.lower())
        self.word_freq = dict(Counter(tokens))

        words = sorted(self.word_freq, key=self.word_freq.get, reverse=True)
        if self.max_vocab:
            words = words[:self.max_vocab - len(self.SPECIAL_TOKENS)]

        vocab = self.SPECIAL_TOKENS + words
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        unk = self.word_to_id[self.UNK]
        return [self.word_to_id.get(w, unk)
                for w in re.findall(r"\b\w+\b", text.lower())]

    def decode(self, ids):
        return ' '.join(self.id_to_word.get(i, self.UNK) for i in ids)

    def save(self, path):
        data = {
            "word_to_id": self.word_to_id,
            "id_to_word": {str(k): v for k, v in self.id_to_word.items()},
            "word_freq":  self.word_freq,
            "vocab_size": self.vocab_size,
            "max_vocab":  self.max_vocab,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.word_to_id = data["word_to_id"]
        self.id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
        self.word_freq  = data["word_freq"]
        self.vocab_size = data["vocab_size"]
        self.max_vocab  = data["max_vocab"]

    def __repr__(self):
        return f"Vocabulary(vocab_size={self.vocab_size}, max_vocab={self.max_vocab})"


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    corpus = """
    building african language models from scratch requires understanding
    of mathematics linear algebra calculus and programming in python.
    swahili yoruba hausa amharic are major african languages.
    language models learn patterns from large amounts of text data.
    african languages have unique features that make them challenging.
    """

    vocab = Vocabulary(max_vocab=30)
    vocab.build(corpus)
    print(vocab)
    print(f"Top 10 words: {list(vocab.word_to_id.items())[:10]}")

    # Encode / decode
    text = "african language models"
    ids = vocab.encode(text)
    print(f"\nEncode '{text}': {ids}")
    print(f"Decode {ids}: '{vocab.decode(ids)}'")

    # OOV test
    ids_oov = vocab.encode("unknown xyz word")
    print(f"\nOOV test: {ids_oov} -> '{vocab.decode(ids_oov)}'")

    # Save and reload
    vocab.save("data/processed/vocab.json")
    vocab2 = Vocabulary()
    vocab2.load("data/processed/vocab.json")
    assert vocab2.vocab_size == vocab.vocab_size
    print(f"\nSave/load: PASSED (vocab_size={vocab2.vocab_size})")
    print("\nDone: Vocabulary complete.")
