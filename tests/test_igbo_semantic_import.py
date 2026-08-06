import importlib.util
import sys
import types
from pathlib import Path


def _load_script(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=object))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            HfFolder=types.SimpleNamespace(get_token=lambda: None),
            hf_hub_download=lambda **_kwargs: "",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "unidecode",
        types.SimpleNamespace(unidecode=lambda value: value),
    )

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dataset_updates"
        / "add_igbo_names_from_semantic_analysis.py"
    )
    spec = importlib.util.spec_from_file_location("_igbo_semantic_import", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manual_candidates_override_longer_ocr_extractions(monkeypatch):
    script = _load_script(monkeypatch)
    store = {}

    script._add_candidate(
        store,
        "Ekwutosi",
        "Don't talk evil/ don't malign/ blaspheme",
        "p9_injunction",
    )
    script._add_candidate(
        store,
        "Ekwutosi",
        "Don't talk evil",
        "manual_structural",
    )

    assert store["Ekwutosi"] == (
        "Don't talk evil",
        "Ekwutosi",
        "manual_structural",
    )

    script._add_candidate(
        store,
        "Ekwutosi",
        "Don't talk evil with a much longer OCR tail",
        "p9_injunction",
    )

    assert store["Ekwutosi"] == (
        "Don't talk evil",
        "Ekwutosi",
        "manual_structural",
    )


def test_same_priority_candidates_still_keep_longer_meaning(monkeypatch):
    script = _load_script(monkeypatch)
    store = {}

    script._add_candidate(store, "Chinyelu", "God gives", "p5_arrow")
    script._add_candidate(store, "Chinyelu", "God has given/God's gift", "p5_arrow")

    assert store["Chinyelu"] == (
        "God has given/God's gift",
        "Chinyelu",
        "p5_arrow",
    )
