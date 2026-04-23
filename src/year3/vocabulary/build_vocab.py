"""
Phase 3.3 - Vocabulary Builder for Cameroon Languages
Builds, analyzes and saves vocabulary for each language
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.year3.tokenization.african_tokenizers import CameroonTokenizer

DATA_ROOT  = "data/cameroon_languages"
LANGUAGES  = ["english", "french", "bayangi", "douala"]
MAX_VOCAB  = 500


def build_and_save(language):
    path = os.path.join(DATA_ROOT, language, "processed", f"{language}_clean.txt")
    if not os.path.exists(path):
        print(f"  Skipping {language} - processed file not found")
        return None

    with open(path, encoding="utf-8") as f:
        corpus = f.read()

    tok = CameroonTokenizer(language=language)
    tok.build_vocab(corpus, max_vocab=MAX_VOCAB)

    # Save vocab JSON
    vocab_path = os.path.join(DATA_ROOT, language, "vocab.json")
    vocab_data = {
        "language":    language,
        "vocab_size":  tok.vocab_size,
        "word_to_id":  tok.word_to_id,
        "id_to_word":  {str(k): v for k, v in tok.id_to_word.items()},
        "special_tokens": {
            "PAD": tok.PAD, "UNK": tok.UNK,
            "START": tok.START, "END": tok.END
        }
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    return tok


def plot_vocab_analysis(tokenizers):
    os.makedirs("docs/visualizations", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (lang, tok) in enumerate(tokenizers.items()):
        if tok is None:
            continue
        # Get top 15 words (skip special tokens)
        words = [(w, tok.word_to_id[w]) for w in tok.word_to_id
                 if not w.startswith('<')]
        words = sorted(words, key=lambda x: x[1])[:15]
        labels = [w for w, _ in words]
        ids    = [i for _, i in words]

        axes[i].barh(labels, ids, color='steelblue')
        axes[i].set_title(f"{lang.capitalize()} — vocab size: {tok.vocab_size}")
        axes[i].set_xlabel("Token ID")
        axes[i].invert_yaxis()

    plt.suptitle("Cameroon Language Vocabularies", fontsize=14)
    plt.tight_layout()
    plt.savefig("docs/visualizations/cameroon_vocab.png", dpi=100)
    plt.close()
    print("Saved: docs/visualizations/cameroon_vocab.png")


if __name__ == "__main__":
    print("=== BUILDING CAMEROON VOCABULARIES ===\n")
    tokenizers = {}
    for lang in LANGUAGES:
        print(f"Building {lang}...")
        tok = build_and_save(lang)
        tokenizers[lang] = tok
        if tok:
            print(f"  Vocab size: {tok.vocab_size}")
            print(f"  Saved to: {DATA_ROOT}/{lang}/vocab.json")

    plot_vocab_analysis(tokenizers)
    print("\nDone: Vocabulary building complete.")
