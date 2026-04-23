"""
Standalone data downloader for Colab
Run this before training to get more English data
"""

import requests
import re
import time
import os

DATA_PATH = 'data/cameroon_languages/english/processed/english_clean.txt'

TOPICS = [
    # Cameroon
    "Cameroon", "Douala", "Yaounde", "History_of_Cameroon",
    "Economy_of_Cameroon", "Languages_of_Cameroon",
    "Culture_of_Cameroon", "Mount_Cameroon", "Wouri_River",
    "Bamileke_people", "Duala_people", "Fulani_people",
    "Anglophone_Crisis", "Cameroon_national_football_team",
    "Paul_Biya", "Ahmadou_Ahidjo", "Roger_Milla",
    "Cameroonian_cuisine", "Music_of_Cameroon",
    "Kribi", "Limbe,_Cameroon", "Bamenda",
    "Garoua", "Bafoussam", "Lake_Chad",
    "Sanaga_River", "Adamawa_Plateau",
    "Waza_National_Park", "Dja_Faunal_Reserve",
    "Cameroonian_French", "Camfranglais",
    # General English grammar and structure
    "English_language", "English_grammar",
    "Africa", "Sub-Saharan_Africa", "Central_Africa",
    "West_Africa", "East_Africa", "Southern_Africa",
    "Bantu_languages", "Niger-Congo_languages",
    "African_literature", "French_language",
    "Bilingualism", "Multilingualism",
    "Language_acquisition", "Linguistics",
    # Science and knowledge
    "Science", "Mathematics", "Physics",
    "Biology", "Geography", "History",
    "Education", "Technology", "Computer_science",
    "Agriculture", "Economy", "Politics",
    "Government", "Democracy", "Health",
    "Medicine", "Food", "Water",
    "Climate", "Environment", "Nature",
    # Society and culture
    "Family", "Community", "Culture",
    "Religion", "Tradition", "Music",
    "Art", "Sport", "Football",
    "City", "Village", "Market",
    "School", "University", "Work",
    "Business", "Trade", "Transport",
    # More African countries for context
    "Nigeria", "Ghana", "Senegal",
    "Ethiopia", "Kenya", "Tanzania",
    "South_Africa", "Egypt", "Morocco",
]


def fetch_article(title, lang="en"):
    url = (f"https://{lang}.wikipedia.org/w/api.php"
           f"?action=query&titles={title}&prop=extracts"
           f"&explaintext=true&format=json")
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "FNI-LLM/1.0"})
        pages = r.json().get("query", {}).get("pages", {})
        for p in pages.values():
            return p.get("extract", "")
    except:
        return ""


def extract_sentences(text, min_words=8, max_words=40):
    # Clean text
    text = re.sub(r'==+[^=]+=+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\([^)]{0,50}\)', '', text)
    text = re.sub(r'\s+', ' ', text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if (min_words <= len(words) <= max_words
                and s and s[0].isupper()
                and not s.startswith('(')
                and s.endswith('.')):
            clean.append(s)
    return clean


# Load existing sentences
existing = set()
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, encoding='utf-8') as f:
        existing = set(line.strip() for line in f
                       if line.strip())
    print(f'Existing sentences: {len(existing):,}')

all_sentences = set(existing)

print(f'Downloading {len(TOPICS)} articles...\n')
for i, topic in enumerate(TOPICS):
    text  = fetch_article(topic)
    sents = extract_sentences(text)
    before = len(all_sentences)
    all_sentences.update(sents)
    added = len(all_sentences) - before
    print(f'[{i+1:3d}/{len(TOPICS)}] {topic:40s} '
          f'+{added:4d} | total: {len(all_sentences):,}')
    time.sleep(0.3)

# Save
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
with open(DATA_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(sorted(all_sentences)))

print(f'\nFinal corpus: {len(all_sentences):,} sentences')
print(f'Saved to: {DATA_PATH}')
