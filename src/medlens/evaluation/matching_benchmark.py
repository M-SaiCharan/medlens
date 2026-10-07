"""Synthetic benchmark for the text -> medicine matcher.

    python -m medlens.evaluation.matching_benchmark [--n 1000] [--seed 0]

For a random sample of active medicines, simulates three kinds of input:
  typed  - what a user types into search ("augmentin 625"), with occasional typos
  ocr    - the brand line as OCR reads it off a strip (character confusions, drops)
  strip  - the full OCR text of a strip: noisy brand + composition + boilerplate

Reports top-1 / top-5 accuracy on the exact brand, and top-1 "clinically
equivalent" accuracy (same composition and form, which is what generics and
interaction checks actually depend on).
"""
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from medlens.catalog import CATALOG, ROOT
from medlens.matching.matcher import MedicineMatcher, brand_core

RESULTS = ROOT / "data" / "eval" / "matching_benchmark.json"

# Common OCR confusions on printed packaging.
OCR_CONFUSIONS = {
    "i": "l1", "l": "i1", "o": "0", "0": "o", "s": "5", "5": "s", "b": "8", "e": "c", "c": "e",
    "m": ["rn"], "n": "h", "g": "9", "1": "l", "t": "f", "u": "v",
}
BOILERPLATE = [
    "Mfd. by: {mfr}", "Batch No. {batch}", "Exp. {mm}/{yy}", "M.R.P. Rs. {price} incl. of all taxes",
    "Schedule H Prescription Drug", "Store below 25°C", "Keep out of reach of children",
]


def ocr_noise(text: str, rng: random.Random, rate: float) -> str:
    out = []
    for ch in text:
        r = rng.random()
        if r < rate and ch.lower() in OCR_CONFUSIONS:
            out.append(rng.choice(list(OCR_CONFUSIONS[ch.lower()])))
        elif r < rate * 1.3:
            continue  # dropped character
        else:
            out.append(ch)
    return "".join(out)


def typo(text: str, rng: random.Random) -> str:
    if len(text) < 4 or rng.random() < 0.5:
        return text
    i = rng.randrange(1, len(text) - 1)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    if op == "drop":
        return text[:i] + text[i + 1 :]
    return text[:i] + text[i] + text[i:]


def make_queries(row: pd.Series, rng: random.Random) -> dict[str, str]:
    core = brand_core(row["name_norm"])
    form = row["form"].split()[0].title()
    salts = " and ".join(row["salts"])
    strengths = " ".join(f"{s} {st}" for s, st in zip(row["salts"], row["strengths"]))
    lines = [
        ocr_noise(core.upper(), rng, 0.08),
        ocr_noise(f"{salts} {form}s IP", rng, 0.05),
        ocr_noise(f"Each {form.lower()} contains: {strengths}", rng, 0.05),
    ]
    for tmpl in rng.sample(BOILERPLATE, 3):
        lines.append(
            tmpl.format(
                mfr=row["manufacturer_name"], batch=rng.randint(1000, 99999), mm=rng.randint(1, 12),
                yy=rng.randint(25, 28), price=row["price"],
            )
        )
    rng.shuffle(lines)
    return {"typed": typo(core, rng), "ocr": ocr_noise(core.upper(), rng, 0.12), "strip": "\n".join(lines)}


def run(n: int = 1000, seed: int = 0) -> dict:
    catalog = pd.read_parquet(CATALOG).reset_index(drop=True)
    matcher = MedicineMatcher(catalog)
    rng = random.Random(seed)
    sample = catalog[~catalog["discontinued"]].sample(n, random_state=seed)

    names = catalog["name_norm"]
    equiv = catalog["composition_key"] + "#" + catalog["form"]
    stats = {k: {"top1": 0, "top5": 0, "equiv_top1": 0, "seconds": 0.0} for k in ("typed", "ocr", "strip")}
    failures = []
    for _, row in sample.iterrows():
        for kind, query in make_queries(row, rng).items():
            t = time.perf_counter()
            matches = matcher.match(query, limit=5)
            stats[kind]["seconds"] += time.perf_counter() - t
            hits = [names[m.index] == row["name_norm"] for m in matches]
            stats[kind]["top1"] += bool(hits[:1] and hits[0])
            stats[kind]["top5"] += any(hits)
            stats[kind]["equiv_top1"] += bool(matches) and equiv[matches[0].index] == f"{row['composition_key']}#{row['form']}"
            if not any(hits) and len(failures) < 30:
                failures.append({"kind": kind, "expected": row["name"], "query": query, "got": [m.name for m in matches[:3]]})

    report = {
        "n": n,
        "seed": seed,
        "results": {
            k: {
                "top1": round(v["top1"] / n, 4),
                "top5": round(v["top5"] / n, 4),
                "equiv_top1": round(v["equiv_top1"] / n, 4),
                "ms_per_query": round(1000 * v["seconds"] / n, 1),
            }
            for k, v in stats.items()
        },
        "failure_examples": failures,
    }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = run(args.n, args.seed)
    RESULTS.write_text(json.dumps(report, indent=2))
    print(f"{'input':8} {'top-1':>7} {'top-5':>7} {'equiv@1':>8} {'ms/q':>6}")
    for k, v in report["results"].items():
        print(f"{k:8} {v['top1']:7.1%} {v['top5']:7.1%} {v['equiv_top1']:8.1%} {v['ms_per_query']:6.0f}")
    print(f"Saved -> {RESULTS.relative_to(ROOT)}")
