# Cameroon Languages Data

## Language Implementation Order

| Priority | Language | Type | Status |
|----------|----------|------|--------|
| 1 | English (Cameroon) | Official | Planned |
| 2 | French (Cameroon) | Official | Planned |
| 3 | Bayangi | Indigenous | Planned |
| 4 | Douala (Duala) | Indigenous | Planned |
| 5 | Bamileke | Indigenous | Future |
| 6 | Fulfulde | Indigenous | Future |
| 7 | Ewondo | Indigenous | Future |
| 8 | Bassa | Indigenous | Future |

## Why This Order

- English and French are official languages with large available datasets
- They will be used to train and validate the model architecture first
- Bayangi and Douala are next as indigenous Cameroon languages
- Other languages will follow as data becomes available

## Data Sources to Explore

### English & French
- Wikipedia dumps (en.wikipedia.org, fr.wikipedia.org)
- OSCAR corpus (multilingual)
- Common Crawl
- News articles (Cameroon Tribune, Journal du Cameroun)

### Bayangi, Douala & Others
- SIL International linguistic resources
- SOAS University language archives
- Local recordings from native speakers
- Bible translations (widely available for many African languages)
- University of Yaounde linguistics department resources

## Folder Structure
```
cameroon_languages/
├── english/
│   ├── raw/
│   └── processed/
├── french/
│   ├── raw/
│   └── processed/
├── bayangi/
│   ├── raw/
│   └── processed/
├── douala/
│   ├── raw/
│   └── processed/
└── other/
```

## Notes
- All data must have open/permissive licenses
- Indigenous language data may be scarce — local collection may be needed
- Respect cultural sensitivity when collecting indigenous language data
