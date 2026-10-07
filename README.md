# MedLens 💊

**Scan a medicine strip → know exactly what it is, find the same medicine for less, and catch dangerous combinations.**

India has ~250,000 branded medicines, and the *same* salt at the *same* strength routinely sells at a
**median 2.9× price spread** between brands (up to 10×+). People can't read strips, don't know generics
exist, and rarely know which of their medicines interact. MedLens fixes that from a single photo.

## What it does
1. **Scan** – photo of a strip or box → OCR (PaddleOCR models on ONNX Runtime, CPU-only).
2. **Identify** – a trained character-level neural retriever + fuzzy/composition reranker maps noisy OCR text to one of 254k medicines.
3. **Save** – cheaper brands with *identical* salts, strengths **and** dosage form (SR ≠ plain tablet), with data-error prices filtered out.
4. **Protect** – checks every pair of salts across everything you take against 160k documented interactions (DDInter), with severity.
5. **Be honest** – salts without interaction data are shown as *unchecked*, never as "safe".

## Results
Synthetic benchmark, 300 random active medicines × 3 input types (`python -m medlens.evaluation.matching_benchmark`).
*equiv@1* = top result has the same composition and form, which is what savings and safety checks depend on.

| Matcher | Input | Top-1 | Top-5 | Equiv@1 | Latency |
|---|---|---|---|---|---|
| Fuzzy baseline (hand-tuned) | typed | 93.0% | 98.7% | 94.7% | 350 ms |
| | OCR brand line | 77.7% | 89.3% | 82.0% | 354 ms |
| | full strip | 84.0% | 95.7% | 91.7% | 924 ms |
| Neural retriever (0.3M params, 5-min CPU smoke run) | typed | 95.7% | 99.7% | 96.3% | **18 ms** |
| | OCR brand line | 78.3% | 90.0% | 82.0% | **16 ms** |
| | full strip | 88.7% | 96.0% | **95.7%** | **96 ms** |
| Hybrid (neural ∪ fuzzy candidates) | typed | 95.0% | 99.7% | 95.7% | 461 ms |
| | OCR brand line | 80.7% | 91.3% | 84.0% | 538 ms |
| | full strip | 88.7% | 97.3% | 94.3% | 1302 ms |
| Neural retriever (full GPU run) | | *pending workstation run* | | | |

Hybrid is within ~1 point of neural-only at ~20× the latency, so the app defaults to neural when a trained
model is present (override with `MEDLENS_MATCH_MODE=fuzzy|neural|hybrid`).

Retriever recall on brands **never seen in training** (5% held out) matches seen brands, so it learned OCR
noise, not the catalog.

## Architecture
```
 photo ─► OCR (RapidOCR/ONNX) ─► line cleanup + OCR de-confusion (0→o, rn→m, …)
                                          │
                     ┌────────────────────┴────────────────────┐
           byte-level transformer retriever            rapidfuzz brand scoring
           (contrastive, ONNX, 242k-brand index)        (WRatio + core + strength-free)
                     └────────────────────┬────────────────────┘
                     rerank: salt / strength / form evidence from the strip text
                                          ▼
                         catalog row (salts, strengths, form, price)
                     ┌────────────────────┴────────────────────┐
           same composition + form → cheaper brands     salt names → DDInter (alias + suffix
           (outlier prices excluded)                    stripping + strict fuzzy) → pairwise severity
                                          ▼
                           FastAPI  +  mobile-first web UI
```

## Run it locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py      # needs a Kaggle token (~/.kaggle/access_token or kaggle.json)
python -m medlens.catalog            # builds data/processed/medicines.parquet
uvicorn app.main:app --reload        # http://127.0.0.1:8000
pytest -q
```
Without a trained model the app uses the fuzzy matcher; drop `models/retriever/` in to switch to the neural matcher.

## Train the retriever (GPU workstation)
```bash
git clone https://github.com/M-SaiCharan/medlens.git && cd medlens
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build
pip install -r requirements-train.txt && pip install -e .
python scripts/download_data.py && python -m medlens.catalog
python -m medlens.retrieval.train                      # ~30-60 min on an RTX 2000 Ada
python -m medlens.evaluation.matching_benchmark --mode neural --n 1000
python -m medlens.evaluation.matching_benchmark --mode hybrid --n 1000
```
Then copy `models/retriever/` (`model.onnx`, `index.npy`, `cores.json`, `metrics.json`) back to the serving machine.

## Deploy
```bash
docker build -t medlens . && docker run -p 7860:7860 medlens
```
The image is CPU-only (no torch) and runs on a free Hugging Face Docker Space (port 7860).

## Project layout
```
app/                    FastAPI backend + static web UI
scripts/download_data.py
src/medlens/
  catalog.py            raw CSV → parsed compositions, forms, price/unit, outlier flags
  matching/matcher.py   two-stage matcher (fuzzy / neural / hybrid)
  retrieval/            byte-level encoder, training, ONNX index
  generics/finder.py    same-composition cheaper alternatives
  interactions/         DDInter checker with salt-name resolution
  ocr/reader.py         RapidOCR wrapper
  evaluation/           synthetic OCR benchmark
  synth.py              OCR / typo noise models (shared by training and eval)
tests/
```

## Data & licences
- [A-Z Medicine Dataset of India](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india) (CC BY-SA 4.0), prices as of Nov 2022.
- [DDInter](https://ddinter.scbdd.com/) drug–drug interactions (free for academic / non-commercial use).

## Known limitations
- Prices are a 2022 snapshot; the source lists at most two salts per medicine.
- ~24% of salt mentions (mostly India-specific drugs, e.g. nimesulide, aceclofenac) have no DDInter entry; these are flagged as unchecked.
- The benchmark is synthetic; a hand-labelled set of real strip photos is the next evaluation step.

## Disclaimer
MedLens is an informational tool, not medical advice. Always consult a doctor or pharmacist before switching or combining medicines.
