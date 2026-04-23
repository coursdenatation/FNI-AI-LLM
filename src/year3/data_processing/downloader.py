"""
Phase 3.1 - Data Downloader
Downloads English and French Wikipedia articles about Cameroon topics
"""

import re
import os
import requests
import time

DATA_ROOT = "data/cameroon_languages"

# Cameroon-related Wikipedia topics for English
ENGLISH_TOPICS = [
    "Cameroon", "Douala", "Yaounde", "Bamileke_people",
    "Fulani_people", "Bayangi_people", "Duala_people",
    "Cameroonian_cuisine", "Music_of_Cameroon",
    "History_of_Cameroon", "Economy_of_Cameroon",
    "Languages_of_Cameroon", "Mount_Cameroon",
    "Wouri_River", "Lake_Chad", "Ngondo_festival",
    "Indomitable_Lions", "Cameroonian_literature",
    "Southwest_Region_Cameroon", "Littoral_Region_Cameroon",
]

# Same topics in French Wikipedia
FRENCH_TOPICS = [
    "Cameroun", "Douala", "Yaounde", "Bamileke",
    "Peuls", "Langue_douala", "Cuisine_camerounaise",
    "Musique_du_Cameroun", "Histoire_du_Cameroun",
    "Economie_du_Cameroun", "Langues_du_Cameroun",
    "Mont_Cameroun", "Fleuve_Wouri", "Ngondo",
    "Lions_Indomptables", "Region_du_Littoral",
    "Region_du_Sud-Ouest_Cameroun",
]


def fetch_wikipedia(title, lang="en"):
    """Fetch plain text from Wikipedia API"""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "FNI-LLM/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            return data.get("extract", "")
        return ""
    except Exception as e:
        print(f"    Error fetching {title}: {e}")
        return ""


def text_to_sentences(text):
    """Split text into sentences"""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def download_corpus(topics, lang, output_path, existing_corpus=""):
    """Download Wikipedia articles and extract sentences"""
    all_sentences = set()

    # Keep existing seed corpus sentences
    if existing_corpus:
        for line in existing_corpus.strip().split('\n'):
            if line.strip():
                all_sentences.add(line.strip())

    print(f"  Downloading {len(topics)} articles from {lang}.wikipedia.org...")
    for i, topic in enumerate(topics):
        text = fetch_wikipedia(topic, lang)
        if text:
            sentences = text_to_sentences(text)
            all_sentences.update(sentences)
            print(f"  [{i+1:2d}/{len(topics)}] {topic}: +{len(sentences)} sentences "
                  f"(total: {len(all_sentences)})")
        else:
            print(f"  [{i+1:2d}/{len(topics)}] {topic}: not found")
        time.sleep(0.3)  # be polite to Wikipedia API

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sentences_list = sorted(all_sentences)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences_list))

    return len(sentences_list)


if __name__ == "__main__":
    print("=== DOWNLOADING CAMEROON LANGUAGE DATA ===\n")

    # Load existing seed corpora
    en_seed_path = f"{DATA_ROOT}/english/raw/english_corpus.txt"
    fr_seed_path = f"{DATA_ROOT}/french/raw/french_corpus.txt"

    en_seed = open(en_seed_path, encoding="utf-8").read() \
        if os.path.exists(en_seed_path) else ""
    fr_seed = open(fr_seed_path, encoding="utf-8").read() \
        if os.path.exists(fr_seed_path) else ""

    # Download English
    print("--- ENGLISH ---")
    en_count = download_corpus(
        ENGLISH_TOPICS, "en",
        f"{DATA_ROOT}/english/raw/english_corpus.txt",
        existing_corpus=en_seed
    )
    print(f"  Total English sentences: {en_count}\n")

    # Download French
    print("--- FRENCH ---")
    fr_count = download_corpus(
        FRENCH_TOPICS, "fr",
        f"{DATA_ROOT}/french/raw/french_corpus.txt",
        existing_corpus=fr_seed
    )
    print(f"  Total French sentences: {fr_count}\n")

    print("=== DOWNLOAD COMPLETE ===")
    print(f"English: {en_count} sentences")
    print(f"French:  {fr_count} sentences")
    print(f"Total:   {en_count + fr_count} sentences")
