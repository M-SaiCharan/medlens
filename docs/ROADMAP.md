# Roadmap

Built CPU-first (no local GPU); heavy training, if any, goes to Colab / Kaggle.

## Phase 1 — Data foundation
- [ ] Collect Indian medicine dataset (name, manufacturer, salt composition, price, pack size)
- [ ] Collect Jan Aushadhi generic product + MRP list
- [ ] Collect open drug–drug interaction data
- [ ] Clean + normalize salt names and strengths (e.g. "Paracetamol 500mg" == "Acetaminophen 500 mg")
- [ ] EDA notebook: price spread for same composition (this becomes the headline "why it matters" chart)

## Phase 2 — Core engine (text in, answers out)
- [ ] Drug matcher: exact → fuzzy (rapidfuzz) → semantic fallback
- [ ] Generic finder: group by normalized composition, rank by price per unit
- [ ] Interaction checker: salt-level graph lookup with severity
- [ ] Unit tests for each

## Phase 3 — Vision
- [ ] OCR baseline (EasyOCR / PaddleOCR) on strip & box photos
- [ ] Build eval set: ~150–300 real photos with ground-truth drug labels
- [ ] Measure image → correct drug accuracy; iterate on preprocessing & matching

## Phase 4 — Explain
- [ ] RAG over drug monographs / labels
- [ ] LLM explanations, grounded with citations, safety guardrails
- [ ] Hindi + one regional language

## Phase 5 — Ship
- [ ] FastAPI backend + web UI
- [ ] Deploy publicly (Hugging Face Spaces / Render)
- [ ] Demo GIF, metrics table, write-up / LinkedIn post

## Metrics to report
- OCR → drug top-1 / top-3 accuracy on the eval set
- Median % savings via generic alternative
- End-to-end latency
