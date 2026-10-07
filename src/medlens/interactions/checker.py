"""Drug-drug interaction checks over salts, backed by DDInter.

Indian catalog salt names ("Paracetamol", "Amoxycillin", "Metoprolol Succinate")
don't match DDInter's ("Acetaminophen", "Amoxicillin", "Metoprolol"), so every
salt goes through `resolve()`. A salt that can't be resolved is reported as
*unknown*, never as safe.
"""
import glob
import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from medlens.text import norm_key

ROOT = Path(__file__).resolve().parents[3]
DDINTER_DIR = ROOT / "data" / "raw" / "ddinter"

SEVERITY_RANK = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}

# Indian/British name -> DDInter (mostly US) name. Keys and values lowercase.
ALIASES = {
    "paracetamol": "acetaminophen",
    "paracetamol/acetaminophen": "acetaminophen",
    "aspirin": "acetylsalicylic acid",
    "amoxycillin": "amoxicillin",
    "thyroxine": "levothyroxine",
    "aluminium hydroxide": "aluminum hydroxide",
    "beclometasone": "beclomethasone dipropionate",
    "beclometasone dipropionate": "beclomethasone dipropionate",
    "cefalexin": "cephalexin",
    "tazobactum": "tazobactam",
    "sodium valproate": "valproic acid",
    "divalproex": "valproic acid",
    "vitamin d3": "cholecalciferol",
    "vitamin b6 (pyridoxine)": "pyridoxine",
    "vitamin b12": "cyanocobalamin",
    "methylcobalamin": "mecobalamin",
    "chlorpheniramine maleate": "chlorpheniramine",
    "frusemide": "furosemide",
    "glibenclamide": "glyburide",
    "lignocaine": "lidocaine",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "salbutamol": "salbutamol",
    "rifampicin": "rifampin",
    "cyclosporin": "cyclosporine",
}

# Salt-form suffixes that don't change the interacting moiety.
SALT_SUFFIXES = (
    "hydrochloride", "hcl", "succinate", "tartrate", "maleate", "oxalate", "hydrobromide", "besylate",
    "besilate", "mesylate", "mesilate", "sodium", "potassium", "calcium", "magnesium", "medoxomil",
    "proxetil", "axetil", "fumarate", "citrate", "sulphate", "sulfate", "phosphate", "acetate",
    "dipropionate", "propionate", "furoate", "bromide", "monohydrate", "trihydrate", "dihydrate",
    "disodium", "decanoate", "gluconate", "lactate", "valerate", "pivoxil",
)


@dataclass
class Interaction:
    salt_a: str
    salt_b: str
    severity: str  # Major / Moderate / Minor / Unknown


class InteractionChecker:
    def __init__(self, ddinter_dir: Path = DDINTER_DIR):
        files = sorted(glob.glob(str(ddinter_dir / "*.csv")))
        if not files:
            raise FileNotFoundError(f"No DDInter CSVs in {ddinter_dir}; run scripts/download_data.py ddinter")
        df = pd.concat(pd.read_csv(f) for f in files)
        self._pairs: dict[frozenset, str] = {}
        for a, b, level in zip(df["Drug_A"].str.lower(), df["Drug_B"].str.lower(), df["Level"]):
            key = frozenset((a, b))
            if SEVERITY_RANK[level] >= SEVERITY_RANK.get(self._pairs.get(key, "Unknown"), 0):
                self._pairs[key] = level
        self._drugs = sorted(set(df["Drug_A"].str.lower()) | set(df["Drug_B"].str.lower()))
        self._drug_set = set(self._drugs)
        self._cache: dict[str, str | None] = {}

    def resolve(self, salt: str) -> str | None:
        """Map a catalog salt name to a DDInter drug name, or None if unknown."""
        key = norm_key(salt)
        if key not in self._cache:
            self._cache[key] = self._resolve(key)
        return self._cache[key]

    def _resolve(self, key: str) -> str | None:
        if key in self._drug_set:
            return key
        if key in ALIASES:
            return ALIASES[key] if ALIASES[key] in self._drug_set else None

        words = key.replace("(", " ").replace(")", " ").split()
        # Strip salt-form suffixes: "metoprolol succinate" -> "metoprolol".
        while len(words) > 1 and words[-1] in SALT_SUFFIXES:
            words.pop()
            stem = " ".join(words)
            if stem in self._drug_set:
                return stem
            if stem in ALIASES:
                return ALIASES[stem]
        # High-threshold fuzzy match for spelling variants (levo-carnitine/levocarnitine).
        # Kept strict on purpose: a wrong match is worse than an honest "unknown" --
        # at 92, "arteether" matched "artemether", a different antimalarial.
        hit = process.extractOne(" ".join(words), self._drugs, scorer=fuzz.ratio, score_cutoff=96)
        return hit[0] if hit else None

    def check(self, salts: list[str]) -> tuple[list[Interaction], list[str]]:
        """Return (interactions found, salts with no interaction data)."""
        resolved = {s: self.resolve(s) for s in dict.fromkeys(salts)}
        unknown = [s for s, r in resolved.items() if r is None]
        known = [(s, r) for s, r in resolved.items() if r is not None]

        found = []
        for (sa, ra), (sb, rb) in itertools.combinations(known, 2):
            if ra == rb:
                continue
            level = self._pairs.get(frozenset((ra, rb)))
            if level:
                found.append(Interaction(sa, sb, level))
        found.sort(key=lambda i: SEVERITY_RANK[i.severity], reverse=True)
        return found, unknown
