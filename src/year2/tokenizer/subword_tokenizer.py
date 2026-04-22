"""
Phase 2.1 - Subword Tokenizer (Byte Pair Encoding)
BPE merges the most frequent character pairs iteratively to build a vocabulary
"""

from collections import Counter


class BPETokenizer:
    def __init__(self, num_merges=50):
        self.num_merges = num_merges
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, vocab):
        """Count all adjacent symbol pairs across all words"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair, vocab):
        """Merge all occurrences of pair in vocab"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    def train(self, corpus):
        # Start: each word is split into characters + end-of-word marker
        word_freq = Counter(corpus.lower().split())
        vocab = {' '.join(list(word)) + ' </w>': freq
                 for word, freq in word_freq.items()}

        for i in range(self.num_merges):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_pair(best, vocab)
            self.merges[best] = i
            self.vocab = vocab

    def tokenize(self, text):
        tokens = []
        for word in text.lower().split():
            word_chars = ' '.join(list(word)) + ' </w>'
            # Apply merges in order
            for (a, b) in sorted(self.merges, key=self.merges.get):
                word_chars = word_chars.replace(f'{a} {b}', f'{a}{b}')
            tokens.extend(word_chars.split())
        return tokens

    def __repr__(self):
        return f"BPETokenizer(num_merges={self.num_merges}, merges_learned={len(self.merges)})"


if __name__ == "__main__":
    corpus = """
    building language models for african languages swahili yoruba hausa
    language model training requires large amounts of text data
    african languages have unique linguistic features and structures
    building building building language language
    """

    tok = BPETokenizer(num_merges=20)
    tok.train(corpus)
    print(tok)

    # Show learned merges
    print(f"\nTop 5 merges learned:")
    for pair, idx in sorted(tok.merges.items(), key=lambda x: x[1])[:5]:
        print(f"  {pair[0]} + {pair[1]} -> {''.join(pair)}")

    # Tokenize new text
    text = "language building"
    tokens = tok.tokenize(text)
    print(f"\nText:   '{text}'")
    print(f"Tokens: {tokens}")

    # Compare with word tokenizer on rare word
    rare = "linguistic"
    print(f"\nRare word '{rare}' -> {tok.tokenize(rare)}")
    print("\nDone: BPE Tokenizer complete.")
