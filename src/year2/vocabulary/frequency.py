"""
Phase 2.2 - Frequency Analysis
Word frequency distribution, stop words, visualization
"""

import re
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('docs/visualizations', exist_ok=True)


def word_frequencies(corpus):
    tokens = re.findall(r"\b\w+\b", corpus.lower())
    return Counter(tokens)


def top_n(freq, n=20):
    return freq.most_common(n)


def identify_stop_words(freq, threshold=0.01):
    """Words appearing in more than threshold fraction of total tokens"""
    total = sum(freq.values())
    return [w for w, c in freq.items() if c / total > threshold]


def plot_frequency(freq, title="Word Frequency", top_n=20, save_path=None):
    words, counts = zip(*freq.most_common(top_n))
    plt.figure(figsize=(12, 5))
    plt.bar(words, counts, color='steelblue')
    plt.title(title)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    corpus = """
    building african language models from scratch requires understanding
    of mathematics linear algebra calculus and programming in python.
    swahili yoruba hausa amharic are major african languages.
    language models learn patterns from large amounts of text data.
    african languages have unique features that make them challenging.
    building language models for african languages is the goal of this project.
    we are building building building language language models models models.
    """

    freq = word_frequencies(corpus)
    print("=== WORD FREQUENCIES ===")
    print(f"Total unique words: {len(freq)}")
    print(f"Total tokens: {sum(freq.values())}")
    print(f"\nTop 15 words:")
    for word, count in top_n(freq, 15):
        bar = '#' * count
        print(f"  {word:15s} {count:3d} {bar}")

    stops = identify_stop_words(freq)
    print(f"\nStop words (>1% of corpus): {stops}")

    plot_frequency(
        freq,
        title="Word Frequency Distribution",
        top_n=15,
        save_path="docs/visualizations/word_frequency.png"
    )
    print("\nDone: Frequency analysis complete.")
