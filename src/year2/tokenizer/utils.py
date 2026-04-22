"""
Phase 2.1 - Tokenizer Utilities
Padding, truncation, batch tokenization
"""

import numpy as np


def pad_sequence(ids, max_len, pad_id=0):
    """Pad or truncate a sequence to max_len"""
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def truncate_sequence(ids, max_len):
    return ids[:max_len]


def batch_tokenize(tokenizer, texts, max_len=None):
    """
    Tokenize a list of texts.
    If max_len given, pad/truncate all to same length.
    Returns list of id sequences and attention masks.
    """
    encoded = [tokenizer.encode(t) for t in texts]

    if max_len is None:
        max_len = max(len(e) for e in encoded)

    padded = [pad_sequence(e, max_len) for e in encoded]
    masks  = [[1 if i < len(e) else 0 for i in range(max_len)]
              for e in encoded]

    return np.array(padded), np.array(masks)


if __name__ == "__main__":
    from src.year2.tokenizer.word_tokenizer import WordTokenizer

    corpus = "building african language models from scratch is challenging"
    tok = WordTokenizer()
    tok.build_vocab(corpus)

    texts = [
        "building language models",
        "african languages",
        "from scratch",
    ]

    print("=== BATCH TOKENIZATION ===")
    ids, masks = batch_tokenize(tok, texts, max_len=8)
    print(f"Input texts: {texts}")
    print(f"Token IDs:\n{ids}")
    print(f"Attention masks:\n{masks}")

    print("\n=== PADDING TEST ===")
    seq = [1, 2, 3]
    print(f"Original: {seq}")
    print(f"Padded to 6: {pad_sequence(seq, 6)}")
    print(f"Truncated to 2: {truncate_sequence(seq, 2)}")

    print("\nDone: Tokenizer utils complete.")
