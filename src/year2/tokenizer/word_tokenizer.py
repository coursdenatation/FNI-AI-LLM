"""
Phase 2.1 - Word-Level Tokenizer
Splits text into words, builds vocabulary, handles unknown tokens
"""

import re


class WordTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    START = "<START>"
    END = "<END>"

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def tokenize(self, text):
        # Split on whitespace and punctuation, lowercase
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, corpus, max_vocab=None):
        # Count frequencies
        freq = {}
        for word in self.tokenize(corpus):
            freq[word] = freq.get(word, 0) + 1

        # Sort by frequency
        words = sorted(freq, key=freq.get, reverse=True)
        if max_vocab:
            words = words[:max_vocab - 4]

        # Special tokens first
        specials = [self.PAD, self.UNK, self.START, self.END]
        vocab = specials + words
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        unk_id = self.word_to_id[self.UNK]
        return [self.word_to_id.get(w, unk_id) for w in self.tokenize(text)]

    def decode(self, ids):
        return ' '.join(self.id_to_word.get(i, self.UNK) for i in ids)

    def __repr__(self):
        return f"WordTokenizer(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    corpus = """
    building an african language model from scratch is a great challenge.
    language models learn from text data. african languages are beautiful.
    swahili yoruba hausa amharic are african languages.
    """

    tok = WordTokenizer()
    tok.build_vocab(corpus, max_vocab=50)
    print(tok)
    print(f"Sample vocab: {list(tok.word_to_id.items())[:10]}")

    text = "african languages are beautiful"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"\nText:    '{text}'")
    print(f"Encoded: {ids}")
    print(f"Decoded: '{decoded}'")

    # UNK handling
    ids_unk = tok.encode("unknown word xyz")
    print(f"\nWith UNK: {ids_unk}")
    print(f"Decoded:  '{tok.decode(ids_unk)}'")
    print("\nDone: WordTokenizer complete.")
