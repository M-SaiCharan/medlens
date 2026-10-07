"""On-the-fly training pairs: (noisy query line, clean brand core).

Every step draws fresh noise, so the model never sees the same corrupted
string twice. ~5% of brands are held out entirely (by stable hash) to measure
generalisation to brands added to the catalog after training.
"""
import random
import zlib

import pandas as pd

from medlens.matching.matcher import MedicineMatcher, brand_core
from medlens.synth import ocr_noise, typo

FORM_SUFFIXES = ["", "", "", " tablet", " tablets", " tab", " capsule", " syrup", " injection", " cream", " drops"]
PREFIXES = ["", "", "", "", "tab ", "cap ", "syp ", "inj ", "rx "]


def unique_cores(catalog: pd.DataFrame) -> list[str]:
    cores = catalog["name_norm"].map(brand_core)
    return sorted(c for c in cores.unique() if len(c) >= 2)


def is_heldout(core: str, pct: int = 5) -> bool:
    return zlib.crc32(core.encode()) % 100 < pct


def noisy_query(core: str, rng: random.Random) -> str:
    """One corrupted rendering of a brand, preprocessed exactly like the matcher's input lines."""
    r = rng.random()
    if r < 0.45:
        q = ocr_noise(core.upper(), rng, rng.uniform(0.02, 0.15))
    elif r < 0.75:
        q = typo(core, rng)
    elif r < 0.9:
        q = rng.choice(PREFIXES) + core + rng.choice(FORM_SUFFIXES)
    else:
        q = core
    if rng.random() < 0.15:  # OCR merging words
        q = q.replace(" ", "", 1)
    return MedicineMatcher._clean(q) or core


class PairSampler:
    """Yields batches of (queries, targets).

    Half of each batch is built from runs of alphabetically adjacent brands
    ("augmentin", "augmentin duo", "augpen" ...): cheap, effective hard negatives.
    """

    def __init__(self, cores: list[str], batch_size: int, seed: int = 0, run: int = 8):
        self.cores, self.batch_size, self.run = cores, batch_size, run
        self.rng = random.Random(seed)

    def batch(self) -> tuple[list[str], list[str]]:
        n, half = len(self.cores), self.batch_size // 2
        picked: set[int] = set()
        while len(picked) < half:
            start = self.rng.randrange(n - self.run)
            picked.update(range(start, start + self.run))
        while len(picked) < self.batch_size:
            picked.add(self.rng.randrange(n))
        targets = [self.cores[i] for i in list(picked)[: self.batch_size]]
        return [noisy_query(t, self.rng) for t in targets], targets
