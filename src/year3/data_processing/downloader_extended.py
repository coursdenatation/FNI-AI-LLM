"""
Phase 3.1 - Extended Data Downloader
Fetches full Wikipedia article sections for more sentences
"""

import re
import os
import requests
import time

DATA_ROOT = "data/cameroon_languages"

ENGLISH_TOPICS = [
    "Cameroon", "Douala", "Yaounde", "History_of_Cameroon",
    "Economy_of_Cameroon", "Languages_of_Cameroon",
    "Culture_of_Cameroon", "Mount_Cameroon", "Wouri_River",
    "Lake_Chad", "Bamileke_people", "Fulani_people",
    "Duala_people", "Ewondo_people", "Bassa_people",
    "Indomitable_Lions", "Cameroonian_cuisine",
    "Music_of_Cameroon", "Cameroonian_literature",
    "Paul_Biya", "Roger_Milla", "Samuel_Eto'o",
    "Anglophone_Crisis", "Cameroon_national_football_team",
    "University_of_Yaounde", "Adamawa_Plateau",
    "Sanaga_River", "Mungo_River", "Manyu_Division",
    "Southwest_Region,_Cameroon", "Littoral_Region,_Cameroon",
    "Centre_Region,_Cameroon", "West_Region,_Cameroon",
    "Bantu_languages", "Niger-Congo_languages",
    "African_literature", "Central_Africa",
]

FRENCH_TOPICS = [
    "Cameroun", "Douala", "Yaounde", "Histoire_du_Cameroun",
    "Bamileke", "Peuls", "Cuisine_camerounaise",
    "Langues_du_Cameroun", "Mont_Cameroun",
    "Ngondo", "Economie_du_Cameroun",
    "Paul_Biya", "Roger_Milla", "Samuel_Eto'o",
    "Universite_de_Yaounde", "Plateau_de_l'Adamaoua",
    "Fleuve_Sanaga", "Afrique_centrale",
    "Langues_bantoues", "Litterature_africaine",
    "Makossa", "Bikutsi", "Ndole_(plat)",
]


def fetch_full_article(title, lang="en"):
    """Fetch full article text using Wikipedia API"""
    url = (f"https://{lang}.wikipedia.org/w/api.php"
           f"?action=query&titles={title}&prop=extracts"
           f"&explaintext=true&format=json")
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "FNI-LLM/1.0"})
        if resp.status_code == 200:
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                return page.get("extract", "")
        return ""
    except Exception as e:
        print(f"    Error: {e}")
        return ""


def text_to_sentences(text, min_len=30, max_len=300):
    """Clean and split text into quality sentences"""
    # Remove section headers and references
    text = re.sub(r'==+[^=]+=+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = []
    for s in sentences:
        s = s.strip()
        # Keep sentences with good length and no special chars
        if (min_len <= len(s) <= max_len
                and not s.startswith('(')
                and s[0].isupper() if s else False):
            clean.append(s)
    return clean


def download_full_corpus(topics, lang, output_path, existing_corpus=""):
    all_sentences = set()

    if existing_corpus:
        for line in existing_corpus.strip().split('\n'):
            if line.strip():
                all_sentences.add(line.strip())

    print(f"  Fetching {len(topics)} full articles from {lang}.wikipedia.org...")
    for i, topic in enumerate(topics):
        text = fetch_full_article(topic, lang)
        if text:
            sentences = text_to_sentences(text)
            all_sentences.update(sentences)
            print(f"  [{i+1:2d}/{len(topics)}] {topic}: "
                  f"+{len(sentences)} (total: {len(all_sentences)})")
        else:
            print(f"  [{i+1:2d}/{len(topics)}] {topic}: not found")
        time.sleep(0.5)

    sentences_list = sorted(all_sentences)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences_list))

    return len(sentences_list)


if __name__ == "__main__":
    print("=== EXTENDED DOWNLOAD ===\n")

    en_path = f"{DATA_ROOT}/english/raw/english_corpus.txt"
    fr_path = f"{DATA_ROOT}/french/raw/french_corpus.txt"

    en_seed = open(en_path, encoding="utf-8").read() \
        if os.path.exists(en_path) else ""
    fr_seed = open(fr_path, encoding="utf-8").read() \
        if os.path.exists(fr_path) else ""

    print("--- ENGLISH (full articles) ---")
    en_count = download_full_corpus(
        ENGLISH_TOPICS, "en", en_path, en_seed)
    print(f"  Final English: {en_count} sentences\n")

    print("--- FRENCH (full articles) ---")
    fr_count = download_full_corpus(
        FRENCH_TOPICS, "fr", fr_path, fr_seed)
    print(f"  Final French: {fr_count} sentences\n")

    print(f"TOTAL: {en_count + fr_count} sentences")
