import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("datasets")
pytest.importorskip("unidecode")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dataset_updates"))

import add_igbo_names_from_semantic_analysis as script  # noqa: E402


def test_ifeedi_extraction_collapses_to_ifedi_candidate():
    store = {}

    script._add_candidate(store, "Ifeedi", "There is something", "p5_arrow")
    script._add_candidate(store, "Ifedi", "Something exists", "manual_structural")

    assert list(store) == ["Ifedi"]
    meaning, surface_name, _source = store["Ifedi"]
    assert meaning == "Something exists"
    assert surface_name == "Ifedi"
