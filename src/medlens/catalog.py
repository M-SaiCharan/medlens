"""Build the processed medicine catalog from the raw Kaggle CSV.

    python -m medlens.catalog

Adds parsed salts, a canonical composition key, dosage form, pack quantity and
price per unit, so that "same medicine, different brand" becomes a group-by.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from medlens.text import collapse, norm_brand, norm_key, norm_strength

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "medicines" / "A_Z_medicines_dataset_of_India.csv"
CATALOG = ROOT / "data" / "processed" / "medicines.parquet"

_COMPONENT = re.compile(r"^(.*?)\s*\(([^()]*)\)$")
_PACK = re.compile(
    r"^(?P<container>.+?) of (?P<qty>[\d.]+)\s*(?P<unit>ml|gm|g|kg|ltr|l)?\b\s*(?P<form>.*)$",
    re.IGNORECASE,
)
_PLURAL = {"tablets": "tablet", "capsules": "capsule", "injections": "injection"}


def parse_component(text: str) -> tuple[str, str] | None:
    """'Amoxycillin  (500mg)' -> ('Amoxycillin', '500mg')."""
    if not isinstance(text, str) or not text.strip():
        return None
    m = _COMPONENT.match(text.strip())
    if not m:
        return collapse(text), ""
    return collapse(m.group(1)), collapse(m.group(2))


def composition_key(components: list[tuple[str, str]]) -> str:
    """Order-independent key: 'amoxycillin:500mg|clavulanic acid:125mg'."""
    return "|".join(sorted(f"{norm_key(s)}:{norm_strength(st)}" for s, st in components))


def parse_pack(label: str) -> dict:
    """'bottle of 100 ml Syrup' -> container=bottle, qty=100, unit=ml, form=syrup."""
    m = _PACK.match(collapse(label))
    if not m:
        return {"container": None, "qty": None, "unit": None, "form": norm_key(label)}
    words = [_PLURAL.get(w, w) for w in norm_key(m.group("form")).split()]
    return {
        "container": norm_key(m.group("container")),
        "qty": float(m.group("qty")),
        "unit": (m.group("unit") or "unit").lower(),
        "form": " ".join(words) or "unit",
    }


def flag_price_outliers(df: pd.DataFrame, factor: float = 5.0, min_group: int = 5) -> pd.Series:
    """Flag prices more than `factor`x away from their (composition, form) group median.

    The raw data has entry errors (e.g. a ₹0.70 Augmentin-equivalent when the
    median is ₹20); those must never be recommended as the "cheapest" option.
    """
    ppu = df["price_per_unit"].where(df["price_per_unit"] > 0)
    groups = ppu.groupby([df["composition_key"], df["form"]])
    ratio = ppu / groups.transform("median")
    big_group = groups.transform("size") >= min_group
    return ppu.isna() | (big_group & ((ratio > factor) | (ratio < 1 / factor)))


def build(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(raw_csv).rename(columns={"price(₹)": "price", "Is_discontinued": "discontinued"})

    components = [
        [c for c in (parse_component(a), parse_component(b)) if c]
        for a, b in zip(df["short_composition1"], df["short_composition2"])
    ]
    df["salts"] = [[s for s, _ in comp] for comp in components]
    df["strengths"] = [[st for _, st in comp] for comp in components]
    df["composition"] = [" + ".join(f"{s} ({st})" for s, st in comp) for comp in components]
    df["composition_key"] = [composition_key(comp) for comp in components]

    pack = pd.DataFrame([parse_pack(p) for p in df["pack_size_label"]], index=df.index)
    df = pd.concat([df, pack], axis=1)
    df["price_per_unit"] = df["price"] / df["qty"]
    df["price_suspect"] = flag_price_outliers(df)
    df["name_norm"] = df["name"].map(norm_brand)
    df["manufacturer_name"] = df["manufacturer_name"].map(collapse)

    cols = [
        "id", "name", "name_norm", "manufacturer_name", "price", "discontinued",
        "pack_size_label", "container", "qty", "unit", "form", "price_per_unit", "price_suspect",
        "salts", "strengths", "composition", "composition_key",
    ]
    return df[cols]


if __name__ == "__main__":
    catalog = build()
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(CATALOG, index=False)
    print(f"Wrote {len(catalog):,} medicines -> {CATALOG.relative_to(ROOT)}")
    print(f"  distinct compositions: {catalog['composition_key'].nunique():,}")
    print(f"  distinct forms:        {catalog['form'].nunique():,}")
    print(f"  suspect prices:        {catalog['price_suspect'].sum():,}")
