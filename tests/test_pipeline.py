"""End-to-end tests against the real catalog and DDInter data (skipped if not downloaded)."""
import pytest

from medlens.catalog import CATALOG

pytestmark = pytest.mark.skipif(not CATALOG.exists(), reason="run scripts/download_data.py + python -m medlens.catalog")


@pytest.fixture(scope="module")
def ml():
    from medlens.pipeline import MedLens

    return MedLens()


@pytest.mark.parametrize(
    "query, expected",
    [
        ("dolo 650", "Dolo 650 Tablet"),
        ("Thyronorm 50", "Thyronorm 50mcg Tablet"),
        ("Nise 100", "Nise Tablet"),
        ("AUGMENTlN 625 DUO\nAmoxycillin 500mg Clavulanic acid 125mg", "Augmentin 625 Duo Tablet"),
    ],
)
def test_search_top1(ml, query, expected):
    assert ml.search(query, 1)[0].name == expected


def test_alternatives_are_cheaper_and_equivalent(ml):
    med_id = int(ml.catalog.at[ml.search("Augmentin 625 Duo", 1)[0].index, "id"])
    report = ml.medicine(med_id)
    assert report.alternatives
    ref = ml.catalog.set_index("id")
    for alt in report.alternatives:
        assert alt["price_per_unit"] < report.price_per_unit
        assert ref.at[alt["id"], "composition_key"] == ref.at[med_id, "composition_key"]
        assert ref.at[alt["id"], "form"] == report.form
        assert not ref.at[alt["id"], "price_suspect"]


def test_warfarin_aspirin_is_major(ml):
    found, unknown = ml.checker.check(["Warfarin", "Aspirin"])
    assert [i.severity for i in found] == ["Major"]
    assert unknown == []


def test_unmapped_salt_is_reported_not_silently_safe(ml):
    found, unknown = ml.checker.check(["Nimesulide", "Warfarin"])
    assert "Nimesulide" in unknown


def test_salt_mapping_no_false_friends(ml):
    # Different drugs with similar names must not be merged.
    assert ml.checker.resolve("Arteether") != "artemether"
    assert ml.checker.resolve("Paracetamol") == "acetaminophen"
    assert ml.checker.resolve("Metoprolol Succinate") == "metoprolol"
