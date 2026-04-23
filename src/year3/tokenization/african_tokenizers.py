"""
Phase 3.3 - Cameroon Language Tokenizers
Language-specific tokenizers handling punctuation and special characters
"""

import re
from src.year2.tokenizer.word_tokenizer import WordTokenizer


class CameroonTokenizer(WordTokenizer):
    """
    Extended tokenizer for Cameroon languages.
    Handles French accents, Bayangi/Douala special characters.
    """

    LANGUAGE_PATTERNS = {
        "english": r"\b\w+\b",
        "french":  r"\b[\w\u00C0-\u024F]+\b",   # includes accented chars
        "bayangi": r"\b[\w\u00C0-\u024F]+\b",
        "douala":  r"\b[\w\u00C0-\u024F]+\b",
    }

    def __init__(self, language="english"):
        super().__init__()
        self.language = language
        self.pattern = self.LANGUAGE_PATTERNS.get(language, r"\b\w+\b")

    def tokenize(self, text):
        return re.findall(self.pattern, text.lower())

    def build_vocab(self, corpus, max_vocab=None):
        freq = {}
        for word in self.tokenize(corpus):
            freq[word] = freq.get(word, 0) + 1
        words = sorted(freq, key=freq.get, reverse=True)
        if max_vocab:
            words = words[:max_vocab - 4]
        specials = [self.PAD, self.UNK, self.START, self.END]
        vocab = specials + words
        self.word_to_id = {w: i for i, w in enumerate(vocab)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(vocab)

    def __repr__(self):
        return f"CameroonTokenizer(language={self.language}, vocab={self.vocab_size})"


if __name__ == "__main__":
    import os

    DATA_ROOT = "data/cameroon_languages"
    LANGUAGES = ["english", "french", "bayangi", "douala"]

    print("=== CAMEROON LANGUAGE TOKENIZERS ===\n")
    for lang in LANGUAGES:
        path = os.path.join(DATA_ROOT, lang, "processed", f"{lang}_clean.txt")
        if not os.path.exists(path):
            print(f"{lang}: processed file not found - run pipeline first")
            continue

        with open(path, encoding="utf-8") as f:
            corpus = f.read()

        tok = CameroonTokenizer(language=lang)
        tok.build_vocab(corpus, max_vocab=500)
        print(f"{tok}")
        print(f"  Sample vocab: {list(tok.word_to_id.items())[4:12]}")

        sample = corpus.split('\n')[0]
        ids = tok.encode(sample)
        decoded = tok.decode(ids)
        print(f"  Sample: '{sample[:50]}...'")
        print(f"  Encoded: {ids[:8]}...")
        print(f"  Decoded: '{decoded[:50]}...'")
        print()

    print("Done: Cameroon tokenizers complete.")
