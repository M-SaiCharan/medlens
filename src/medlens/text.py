"""Text normalization shared across the catalog, matcher and interaction checker."""
import re

_SPACES = re.compile(r"\s+")
_MASS = re.compile(r"^([\d.]+)(mg|mcg|µg|gm|g)$")
_MASS_TO_MG = {"mg": 1.0, "mcg": 0.001, "µg": 0.001, "gm": 1000.0, "g": 1000.0}


def collapse(s: str) -> str:
    return _SPACES.sub(" ", s).strip()


def norm_key(s: str) -> str:
    """Lowercase, collapse whitespace. Used for salt names and lookups."""
    return collapse(s).lower()


def norm_strength(s: str) -> str:
    """Canonical strength string so '1gm', '1000mg' and '1000 mg' compare equal."""
    s = s.lower().replace(" ", "")
    m = _MASS.match(s)
    if m:
        mg = float(m.group(1)) * _MASS_TO_MG[m.group(2)]
        return f"{mg:g}mg"
    return s


def norm_brand(s: str) -> str:
    """Brand text for fuzzy matching: lowercase alphanumerics only."""
    return collapse(re.sub(r"[^a-z0-9]+", " ", s.lower()))
