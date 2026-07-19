import pandas as pd
import pytest

from scripts.dataset_updates.update_ibibio_phonetics import (
    EXPECTED_UPDATES,
    apply_updates,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": name,
                "NameStrip": name,
                "Language": language,
                "Phonetic spelling": "",
                "pronunciation_by": "preserve me",
                "validated_by": "preserve me too",
            }
            for name, language in EXPECTED_UPDATES
        ]
    )


def test_applies_exact_updates_and_preserves_provenance_fields():
    before = _frame()
    updated, report = apply_updates(before, EXPECTED_UPDATES)

    assert {row["NameStrip"]: row["after"] for row in report} == {
        name: phonetic for (name, _language), phonetic in EXPECTED_UPDATES.items()
    }
    assert updated["pronunciation_by"].tolist() == before["pronunciation_by"].tolist()
    assert updated["validated_by"].tolist() == before["validated_by"].tolist()
    assert before["Phonetic spelling"].tolist() == ["", "", "", ""]


def test_rejects_duplicate_canonical_rows():
    duplicate = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one row"):
        apply_updates(duplicate, EXPECTED_UPDATES)
