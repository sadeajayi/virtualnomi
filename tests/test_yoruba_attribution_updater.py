import pandas as pd

from scripts.dataset_updates.update_yoruba_attributions import (
    NOMI_ATTRIBUTION,
    YORUBANAMES_ATTRIBUTION,
    apply_attributions,
)


def test_morenikeji_is_kept_as_nomi_sourced():
    rows = pd.DataFrame(
        [
            {
                "NameStrip": "Morenikeji",
                "Language": "Yoruba",
                "Attribution": YORUBANAMES_ATTRIBUTION,
            },
            {
                "NameStrip": "Adunni",
                "Language": "Yoruba",
                "Attribution": "",
            },
        ]
    )

    updated, report = apply_attributions(rows)

    attributions = dict(zip(updated["NameStrip"], updated["Attribution"]))
    assert attributions["Morenikeji"] == NOMI_ATTRIBUTION
    assert attributions["Adunni"] == YORUBANAMES_ATTRIBUTION
    assert {"NameStrip": "Morenikeji", "previous": YORUBANAMES_ATTRIBUTION} in report[
        "set_to_nomi"
    ]
    assert "Morenikeji" in report["counts"]["nomi_sourced_namestrips"]
