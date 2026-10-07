# MedLens 💊

**Scan a medicine strip or prescription → know what it is, find a cheaper generic, and catch dangerous combinations.**

> Status: 🚧 In development

## The problem
- Brand-name medicines often cost 50–90% more than generics with the *same salt composition*.
- Prescriptions and strips are hard to read, especially for elderly and low-literacy users.
- People on multiple medicines rarely know about harmful drug–drug interactions.

## What MedLens does
1. **Scan** – photo of a medicine strip / box / prescription → OCR extracts text.
2. **Identify** – fuzzy + semantic matching maps noisy OCR text to a known drug and its salt composition.
3. **Save** – lists cheaper alternatives with identical composition and strength (incl. Jan Aushadhi generics).
4. **Protect** – checks interactions across all medicines the user takes.
5. **Explain** – plain-language usage, side effects and warnings (RAG-grounded LLM, multilingual).

## Architecture
```
 image ──► OCR ──► text cleanup ──► drug matcher ──► salt composition
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
             generic finder     interaction checker    explainer (RAG + LLM)
                     └──────────────────┴──────────────────┘
                                        ▼
                                  API / Web app
```

## Project layout
```
medlens/
├── app/                 # Web UI / API
├── data/
│   ├── raw/             # Downloaded datasets (not committed)
│   ├── processed/       # Cleaned tables, indexes
│   └── eval/            # Hand-labelled evaluation set
├── docs/                # Design notes, roadmap
├── notebooks/           # EDA and experiments
├── src/medlens/
│   ├── ocr/             # Image → text
│   ├── matching/        # Noisy text → drug entity
│   ├── generics/        # Same-composition alternatives + pricing
│   ├── interactions/    # Drug–drug interaction graph
│   └── explain/         # RAG + LLM explanations
└── tests/
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Disclaimer
MedLens is an informational tool, not medical advice. Always consult a doctor or pharmacist before switching or combining medicines.
