"""
Phase 2.1 - Character-Level Tokenizer
Converts text to/from integer sequences at the character level
"""


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '') for i in ids)

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    corpus = "hello world. building an african llm from scratch!"
    tok = CharTokenizer()
    tok.build_vocab(corpus)
    print(tok)

    text = "hello"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Text:    '{text}'")
    print(f"Encoded: {ids}")
    print(f"Decoded: '{decoded}'")
    assert decoded == text, "Round-trip failed!"
    print("Round-trip: PASSED")
    print("\nDone: CharTokenizer complete.")
