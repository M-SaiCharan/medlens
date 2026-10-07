"""Same-composition, same-form alternatives ranked by price per unit."""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Alternative:
    id: int
    name: str
    manufacturer: str
    pack: str
    price: float
    price_per_unit: float
    savings_pct: float  # vs. the queried medicine, per unit


class GenericFinder:
    def __init__(self, catalog: pd.DataFrame):
        usable = catalog[~catalog["discontinued"] & ~catalog["price_suspect"]]
        self._groups = {k: g.sort_values("price_per_unit") for k, g in usable.groupby(["composition_key", "form"])}

    def alternatives(self, medicine: pd.Series, limit: int = 10) -> list[Alternative]:
        """Cheaper medicines with identical salts, strengths and dosage form.

        Form is part of the key on purpose: a sustained-release tablet is not a
        substitute for a plain tablet even with the same salt and strength.
        """
        group = self._groups.get((medicine["composition_key"], medicine["form"]))
        if group is None:
            return []
        ref = medicine["price_per_unit"]
        cheaper = group[(group["price_per_unit"] < ref) & (group["id"] != medicine["id"])]
        return [
            Alternative(
                id=int(r.id),
                name=r.name,
                manufacturer=r.manufacturer_name,
                pack=r.pack_size_label,
                price=float(r.price),
                price_per_unit=float(r.price_per_unit),
                savings_pct=round(100 * (1 - r.price_per_unit / ref), 1),
            )
            for r in cheaper.head(limit).itertuples()
        ]

    def market_stats(self, medicine: pd.Series) -> dict | None:
        group = self._groups.get((medicine["composition_key"], medicine["form"]))
        if group is None:
            return None
        ppu = group["price_per_unit"]
        return {
            "brands": int(len(group)),
            "min_per_unit": float(ppu.min()),
            "median_per_unit": float(ppu.median()),
            "max_per_unit": float(ppu.max()),
        }
