"""
Phase 3.2 - Text Cleaning Pipeline
Removes noise, normalizes text, handles Unicode for Cameroon languages
"""

import re
import unicodedata


def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)


def remove_html(text):
    return re.sub(r'<[^>]+>', '', text)


def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)


def remove_special_chars(text, keep_punctuation=True):
    if keep_punctuation:
        return re.sub(r'[^\w\s\.\,\!\?\'\-]', '', text)
    return re.sub(r'[^\w\s]', '', text)


def clean_line(line, lowercase=True, keep_punctuation=True):
    line = normalize_unicode(line)
    line = remove_urls(line)
    line = remove_html(line)
    line = remove_special_chars(line, keep_punctuation)
    line = normalize_whitespace(line)
    if lowercase:
        line = line.lower()
    return line


def clean_corpus(text, lowercase=True, keep_punctuation=True,
                 min_length=3, max_length=500):
    lines = text.strip().split('\n')
    cleaned = []
    for line in lines:
        line = clean_line(line, lowercase, keep_punctuation)
        if min_length <= len(line) <= max_length:
            cleaned.append(line)
    return cleaned


if __name__ == "__main__":
    sample = """
    Hello! This is a TEST sentence with <b>HTML</b> tags.
    Visit https://example.com for more info.
    Cameroon   has   extra   spaces   here.
    Short
    Le Cameroun est un beau pays en Afrique centrale.
    Douala est la capitale economique du Cameroun.
    """

    cleaned = clean_corpus(sample)
    print("=== CLEANED CORPUS ===")
    for line in cleaned:
        print(f"  {line}")
    print(f"\nTotal lines: {len(cleaned)}")
    print("\nDone: Cleaner complete.")
