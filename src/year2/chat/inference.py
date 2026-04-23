"""
Phase 2.5 - Inference System
Loads tokenizer, vocabulary and transformer model for text generation
"""

import numpy as np
from src.year2.tokenizer.word_tokenizer import WordTokenizer
from src.year2.vocabulary.vocab import Vocabulary
from src.year2.transformer.transformer import Transformer


class InferenceEngine:
    def __init__(self, vocab_size=200, d_model=64, num_heads=4,
                 d_ff=256, num_layers=2, max_seq_len=64):
        np.random.seed(42)
        self.tokenizer = WordTokenizer()
        self.vocab = Vocabulary(max_vocab=vocab_size)
        self.model = None
        self.trained = False
        self.config = dict(vocab_size=vocab_size, d_model=d_model,
                           num_heads=num_heads, d_ff=d_ff,
                           num_layers=num_layers, max_seq_len=max_seq_len)

    def build(self, corpus):
        """Build vocab and model from corpus"""
        self.vocab.build(corpus)
        actual_vocab = self.vocab.vocab_size
        self.config['vocab_size'] = actual_vocab
        self.model = Transformer(**self.config)
        self.trained = False

    def encode(self, text):
        return self.vocab.encode(text)

    def decode(self, ids):
        return self.vocab.decode(ids)

    def forward(self, text):
        """Run one forward pass, return logits"""
        ids = self.encode(text)
        token_ids = np.array([ids])
        return self.model.forward(token_ids)

    def __repr__(self):
        status = "trained" if self.trained else "untrained"
        return f"InferenceEngine({status}, vocab={self.vocab.vocab_size})"
