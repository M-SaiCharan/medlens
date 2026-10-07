"""Download raw datasets into data/raw/.

Usage:
    python scripts/download_data.py            # everything
    python scripts/download_data.py ddinter    # one source

Sources:
    medicines - Kaggle "A-Z Medicine Dataset of India" (~250k brands, salts, prices).
                Needs a Kaggle API token at ~/.kaggle/kaggle.json.
    ddinter   - DDInter drug-drug interactions with severity (Major/Moderate/Minor).
                Public CSVs, split by ATC code.
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

KAGGLE_MEDICINES = "shudhanshusingh/az-medicine-dataset-of-india"
DDINTER_URL = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{code}.csv"
DDINTER_CODES = "ABDHLPRV"


def download_medicines() -> None:
    out = RAW / "medicines"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_MEDICINES, "-p", str(out), "--unzip"],
        check=True,
    )


def download_ddinter() -> None:
    out = RAW / "ddinter"
    out.mkdir(parents=True, exist_ok=True)
    for code in DDINTER_CODES:
        dest = out / f"ddinter_code_{code}.csv"
        if dest.exists():
            continue
        urllib.request.urlretrieve(DDINTER_URL.format(code=code), dest)
        print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")


SOURCES = {"medicines": download_medicines, "ddinter": download_ddinter}

if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        print(f"Downloading {name}...")
        SOURCES[name]()
