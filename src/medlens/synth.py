"""Synthetic noise models for typed and OCR'd medicine text.

Shared by the matching benchmark and the retriever's training data, so the
model is evaluated on the same noise families it was trained against --
but always with different random draws (and a held-out set of brands).
"""
import random

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
