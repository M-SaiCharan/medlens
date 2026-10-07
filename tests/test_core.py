"""Unit tests for parsing and normalization (no datasets needed)."""
import pandas as pd

from medlens.catalog import composition_key, flag_price_outliers, parse_component, parse_pack
from medlens.matching.matcher import brand_core, deconfuse, strip_numbers
from medlens.text import norm_strength


def test_parse_component():
    assert parse_component("Amoxycillin  (500mg)") == ("Amoxycillin", "500mg")
    assert parse_component("Ambroxol (30mg/5ml)") == ("Ambroxol", "30mg/5ml")
    assert parse_component(None) is None


def test_composition_key_is_order_and_unit_independent():
    a = composition_key([("Ceftriaxone", "1000mg"), ("Sulbactam", "500mg")])
    b = composition_key([("Sulbactam", "0.5gm"), ("CEFTRIAXONE", "1 gm")])
    assert a == b


def test_norm_strength():
    assert norm_strength("1gm") == norm_strength("1000 mg") == "1000mg"
    assert norm_strength("100mcg") == "0.1mg"
    assert norm_strength("30mg/5ml") == "30mg/5ml"


def test_parse_pack():
    assert parse_pack("strip of 10 tablets") == {"container": "strip", "qty": 10.0, "unit": "unit", "form": "tablet"}
    assert parse_pack("bottle of 100 ml Syrup")["form"] == "syrup"
    assert parse_pack("bottle of 100 ml Syrup")["unit"] == "ml"
    # Release modifiers are kept: SR is not interchangeable with a plain tablet.
    assert parse_pack("strip of 10 tablet sr")["form"] == "tablet sr"


def test_flag_price_outliers():
    df = pd.DataFrame(
        {
            "composition_key": ["x"] * 6,
            "form": ["tablet"] * 6,
            "price_per_unit": [20.0, 21.0, 19.0, 22.0, 0.5, 200.0],
        }
    )
    assert flag_price_outliers(df).tolist() == [False, False, False, False, True, True]


def test_matcher_text_helpers():
    assert brand_core("augmentin 625 duo tablet") == "augmentin 625 duo"
    assert strip_numbers("nise 100 mg") == "nise"
    assert deconfuse("d0m5i 500mg") == "domsi 500mg"
