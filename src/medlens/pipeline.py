"""MedLens facade: text/image in, medicine report out."""
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from medlens.catalog import CATALOG
from medlens.generics.finder import GenericFinder
from medlens.interactions.checker import InteractionChecker
from medlens.matching.matcher import Match, MedicineMatcher

SEVERITY_ADVICE = {
    "Major": "Avoid this combination unless a doctor has specifically prescribed it together.",
    "Moderate": "Use together only under medical supervision; dose changes or monitoring may be needed.",
    "Minor": "Usually acceptable; be aware of possible mild effects.",
    "Unknown": "An interaction is documented but its severity is not classified. Ask a pharmacist.",
}


@dataclass
class MedicineReport:
    id: int
    name: str
    manufacturer: str
    composition: str
    salts: list[str]
    form: str
    pack: str
    price: float
    price_per_unit: float
    discontinued: bool
    market: dict | None
    alternatives: list[dict] = field(default_factory=list)


class MedLens:
    def __init__(self, catalog_path: Path = CATALOG):
        if not catalog_path.exists():
            raise FileNotFoundError(f"{catalog_path} missing; run `python -m medlens.catalog`")
        self.catalog = pd.read_parquet(catalog_path).reset_index(drop=True)
        self._by_id = pd.Series(self.catalog.index, index=self.catalog["id"])
        self.matcher = MedicineMatcher(self.catalog)
        self.finder = GenericFinder(self.catalog)
        self.checker = InteractionChecker()

    @cached_property
    def ocr(self):
        # Imported lazily: torch/easyocr are heavy and not needed for text search.
        from medlens.ocr.reader import OCRReader

        return OCRReader()

    def search(self, text: str, limit: int = 5) -> list[Match]:
        return self.matcher.match(text, limit=limit)

    def scan(self, image: bytes, limit: int = 5) -> tuple[str, list[Match]]:
        text = self.ocr.read(image)
        return text, self.search(text, limit=limit)

    def medicine(self, medicine_id: int, alternatives: int = 5) -> MedicineReport:
        row = self.catalog.loc[self._by_id[medicine_id]]
        return MedicineReport(
            id=int(row["id"]),
            name=row["name"],
            manufacturer=row["manufacturer_name"],
            composition=row["composition"],
            salts=list(row["salts"]),
            form=row["form"],
            pack=row["pack_size_label"],
            price=float(row["price"]),
            price_per_unit=float(row["price_per_unit"]),
            discontinued=bool(row["discontinued"]),
            market=self.finder.market_stats(row),
            alternatives=[asdict(a) for a in self.finder.alternatives(row, limit=alternatives)],
        )

    def report(self, medicine_ids: list[int]) -> dict:
        """Full report for everything a person takes: each medicine plus cross-interactions."""
        meds = [self.medicine(i) for i in medicine_ids]
        salt_owner = {}
        for m in meds:
            for s in m.salts:
                salt_owner.setdefault(s, m.name)
        found, unknown = self.checker.check(list(salt_owner))
        interactions = [
            {
                "salt_a": i.salt_a,
                "medicine_a": salt_owner[i.salt_a],
                "salt_b": i.salt_b,
                "medicine_b": salt_owner[i.salt_b],
                "severity": i.severity,
                "advice": SEVERITY_ADVICE[i.severity],
            }
            for i in found
        ]
        return {
            "medicines": [asdict(m) for m in meds],
            "interactions": interactions,
            "unchecked_salts": unknown,
            "potential_savings": _potential_savings(meds),
        }


def _potential_savings(meds: list[MedicineReport]) -> dict:
    """Per-pack savings if every medicine is swapped for its cheapest equivalent."""
    current = cheapest = 0.0
    for m in meds:
        current += m.price
        best = m.alternatives[0]["price_per_unit"] if m.alternatives else m.price_per_unit
        cheapest += best * (m.price / m.price_per_unit)
    return {
        "current": round(current, 2),
        "cheapest": round(cheapest, 2),
        "pct": round(100 * (1 - cheapest / current), 1) if current else 0.0,
    }
