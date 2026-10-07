"""Map noisy text (OCR output or user typing) to catalog medicines.

Two stages:
1. Retrieval: every text line is scored against all ~250k brand names
   (vectorised rapidfuzz, all cores). Blends a forgiving partial score with a
   strict score against the brand "core" (name minus dosage-form words), so
   "dolo 650" prefers "Dolo 650 Tablet" over "Loremdolo 650mg Tablet".
2. Rerank: medicine strips almost always print salt, strength and form, so
   candidates whose composition/form also appears in the text get a boost.
   This disambiguates look-alike brands that OCR alone can't.
"""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from medlens.text import norm_brand

# Words that appear on every strip and carry no brand signal.
_STOP = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps", "ip", "bp", "usp", "each",
    "film", "coated", "contains", "strip", "of", "for", "use", "mfd", "by", "mfg", "exp", "batch", "no",
    "price", "mrp", "rs", "incl", "all", "taxes", "schedule", "h", "drug", "and", "the", "per", "unit",
}
# Dosage-form words removed from a brand name to get its core ("augmentin 625 duo").
_FORM_WORDS = {
    "tablet", "tablets", "capsule", "capsules", "syrup", "injection", "suspension", "oral", "dry", "cream",
    "gel", "ointment", "drop", "drops", "eye", "ear", "nasal", "solution", "lotion", "spray", "infusion",
    "powder", "for", "sr", "er", "dt", "md", "cr", "pr", "xr", "xl", "soft", "gelatin", "softgel",
    "sachet", "granules", "kit", "inhaler", "rotacap", "respules", "shampoo", "soap", "jelly",
}
_FORM_HINTS = {
    "tablet": ("tablet", "tab"), "capsule": ("capsule", "cap"), "syrup": ("syrup",),
    "injection": ("injection", "inj", "vial"), "suspension": ("suspension",), "cream": ("cream",),
    "gel": ("gel",), "ointment": ("ointment",), "drop": ("drop", "drops"),
}
_TOKEN = re.compile(r"[a-z]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Digits OCR commonly reads in place of letters on packaging fonts.
_DIGIT_AS_LETTER = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "9": "g", "6": "g"})
_NUMERIC = re.compile(r"\b\d[\d.]*\s*(mg|mcg|ml|gm|g|iu)?\b")


@dataclass
class Match:
    index: int  # row position in the catalog
    name: str
    composition: str
    score: float  # 0-100


def brand_core(name_norm: str) -> str:
    return " ".join(w for w in name_norm.split() if w not in _FORM_WORDS)


def deconfuse(text: str) -> str:
    """Undo typical OCR confusions inside word-like tokens: 'D0M5I' -> 'domsi', 'ORSrnOX' -> 'orsmox'.

    Tokens that are mostly digits (strengths like '500mg') are left alone.
    """
    out = []
    for tok in text.split():
        letters = sum(c.isalpha() for c in tok)
        if letters and letters >= len(tok) / 2:
            tok = tok.translate(_DIGIT_AS_LETTER).replace("rn", "m")
        out.append(tok)
    return " ".join(out)


def strip_numbers(text: str) -> str:
    """'nise 100' -> 'nise': lets brands whose names omit the strength still match."""
    return " ".join(_NUMERIC.sub(" ", text).split())


class MedicineMatcher:
    """mode: "fuzzy" (brute-force rapidfuzz over the catalog), "neural" (trained
    retriever proposes candidates, rapidfuzz scores only those) or "hybrid" (union).
    Defaults to "hybrid" when a trained retriever is passed, else "fuzzy".
    """

    def __init__(self, catalog: pd.DataFrame, retriever=None, mode: str | None = None):
        self.catalog = catalog.reset_index(drop=True)
        self.retriever = retriever
        self.mode = mode or ("hybrid" if retriever is not None else "fuzzy")
        if self.mode != "fuzzy" and retriever is None:
            raise ValueError(f"mode={self.mode!r} needs a retriever")
        self._names = self.catalog["name_norm"].tolist()
        self._cores = [brand_core(n) for n in self._names]
        self._alpha_cores = [strip_numbers(c) for c in self._cores]
        self._name_to_idx = {n: i for i, n in reversed(list(enumerate(self._names)))}
        self._active = (~self.catalog["discontinued"]).to_numpy()
        self._salts = [[s.lower() for s in salts] for salts in self.catalog["salts"]]
        # Strength numbers as printed ("50mcg" -> "50"); strips print them the same way.
        self._strength_nums = [[m.group() for s in st if (m := _NUMBER.search(s))] for st in self.catalog["strengths"]]
        self._forms = self.catalog["form"].tolist()
        self._rows_by_core: dict[str, list[int]] = {}
        for i, c in enumerate(self._cores):
            self._rows_by_core.setdefault(c, []).append(i)

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(w for w in norm_brand(text).split() if w not in _STOP)

    def match(self, text: str, limit: int = 5, pool: int = 30) -> list[Match]:
        lines = [self._clean(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        # OCR often splits a brand across lines ("augmentin 625d" / "duo"): also try adjacent pairs.
        lines += [f"{a} {b}" for a, b in zip(lines, lines[1:]) if len(a) + len(b) <= 40]
        lines = [line for line in lines if len(line) >= 3]
        lines = list(dict.fromkeys(lines + [deconfuse(line) for line in lines]))
        if not lines:
            return []

        # Stage 1: candidate brands with a 0-100 brand score.
        brand: dict[int, float] = {}
        if self.mode in ("fuzzy", "hybrid"):
            brand.update(self._fuzzy_candidates(lines, pool))
        if self.mode in ("neural", "hybrid"):
            for idx, score in self._neural_candidates(lines, pool).items():
                brand[idx] = max(brand.get(idx, 0.0), score)

        # Stage 2: rerank with composition and form evidence from the full text.
        full = norm_brand(text)
        numbers = set(_NUMBER.findall(full))
        words = set(_TOKEN.findall(full))
        ranked = []
        for idx in brand:
            salts, strengths = self._salts[idx], self._strength_nums[idx]
            salt_hits = sum(fuzz.partial_ratio(s, full) >= 85 for s in salts) / max(len(salts), 1)
            strength_hits = sum(n in numbers for n in strengths) / max(len(strengths), 1)
            form_word = self._forms[idx].split()[0]
            form_hit = any(h in words for h in _FORM_HINTS.get(form_word, (form_word,)))
            score = (
                0.75 * brand[idx]
                + 12 * salt_hits
                + 6 * strength_hits
                + 5 * form_hit
                + 2 * self._active[idx]
            )
            ranked.append((min(float(score), 100.0), int(idx)))

        ranked.sort(reverse=True)
        # Same brand name can appear once per pack size; show each name once.
        out, seen = [], set()
        for s, idx in ranked:
            if self._names[idx] in seen:
                continue
            seen.add(self._names[idx])
            out.append(Match(idx, self.catalog.at[idx, "name"], self.catalog.at[idx, "composition"], round(s, 1)))
            if len(out) == limit:
                break
        return out

    def _fuzzy_candidates(self, lines: list[str], pool: int) -> dict[int, float]:
        """Brute force: (lines x catalog) score matrices, keep the top `pool` per line."""
        loose = process.cdist(lines, self._names, scorer=fuzz.WRatio, workers=-1, dtype=np.uint8)
        strict = process.cdist(lines, self._cores, scorer=fuzz.ratio, workers=-1, dtype=np.uint8)
        alpha = process.cdist(
            [strip_numbers(line) for line in lines], self._alpha_cores, scorer=fuzz.ratio, workers=-1, dtype=np.uint8
        )
        # Strength-free match is discounted; the strength evidence in stage 2 picks the variant.
        brand = np.maximum(0.5 * loose + 0.5 * strict, 0.9 * alpha).max(axis=0)
        k = min(pool * len(lines), len(brand) - 1)
        top = np.argpartition(-brand, k)[:k]
        return {int(i): float(brand[i]) for i in top}

    def _neural_candidates(self, lines: list[str], pool: int) -> dict[int, float]:
        """Trained retriever proposes brands; blend its similarity with pairwise fuzzy scores."""
        sims = self.retriever.search(lines, k=pool)
        alpha_lines = [strip_numbers(line) for line in lines]
        out = {}
        for core, sim in sims.items():
            for idx in self._rows_by_core.get(core, ()):
                fuzzy = max(
                    max(0.5 * fuzz.WRatio(line, self._names[idx]) + 0.5 * fuzz.ratio(line, core), 0.9 * fuzz.ratio(al, self._alpha_cores[idx]))
                    for line, al in zip(lines, alpha_lines)
                )
                out[idx] = 0.5 * fuzzy + 50 * max(sim, 0.0)
        return out

    def by_name(self, name: str) -> pd.Series | None:
        idx = self._name_to_idx.get(norm_brand(name))
        return None if idx is None else self.catalog.loc[idx]
