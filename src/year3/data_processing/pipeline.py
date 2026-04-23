"""
Phase 3.2 - Complete Data Pipeline
clean -> deduplicate -> validate -> save for all Cameroon languages
"""

import os
import json
from datetime import datetime
from src.year3.data_processing.cleaner import clean_corpus

LANGUAGES = ["english", "french", "bayangi", "douala"]
DATA_ROOT = "data/cameroon_languages"


def load_raw(language):
    path = os.path.join(DATA_ROOT, language, "raw", f"{language}_corpus.txt")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def deduplicate(lines):
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique


def validate(lines, min_words=3):
    return [l for l in lines if len(l.split()) >= min_words]


def save_processed(language, lines):
    out_dir = os.path.join(DATA_ROOT, language, "processed")
    os.makedirs(out_dir, exist_ok=True)

    # Save clean text
    text_path = os.path.join(out_dir, f"{language}_clean.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save metadata
    meta = {
        "language": language,
        "sentence_count": len(lines),
        "date_processed": str(datetime.now()),
        "steps": ["clean", "deduplicate", "validate"],
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return text_path, len(lines)


def run_pipeline(language):
    print(f"\n--- Processing: {language.upper()} ---")

    raw = load_raw(language)
    if not raw:
        return 0

    # Step 1: Clean
    cleaned = clean_corpus(raw, lowercase=True, keep_punctuation=True)
    print(f"  After cleaning:       {len(cleaned)} sentences")

    # Step 2: Deduplicate
    deduped = deduplicate(cleaned)
    removed_dupes = len(cleaned) - len(deduped)
    print(f"  After deduplication:  {len(deduped)} sentences "
          f"({removed_dupes} duplicates removed)")

    # Step 3: Validate
    validated = validate(deduped, min_words=3)
    removed_short = len(deduped) - len(validated)
    print(f"  After validation:     {len(validated)} sentences "
          f"({removed_short} too short removed)")

    # Step 4: Save
    path, count = save_processed(language, validated)
    print(f"  Saved to: {path}")
    return count


if __name__ == "__main__":
    print("=== CAMEROON LANGUAGE DATA PIPELINE ===")
    totals = {}
    for lang in LANGUAGES:
        count = run_pipeline(lang)
        totals[lang] = count

    print("\n=== SUMMARY ===")
    total = 0
    for lang, count in totals.items():
        print(f"  {lang:10s}: {count:5d} sentences")
        total += count
    print(f"  {'TOTAL':10s}: {total:5d} sentences")
    print("\nDone: Pipeline complete.")
