import builtins
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "paraphrasing"))

import transform_yoruba_meanings as transform  # noqa: E402


def test_rephrase_all_rejects_single_name_mode():
    with pytest.raises(SystemExit):
        transform.parse_args(["--rephrase-all", "--name", "Folasade"])


def test_rephrase_all_processes_full_queue_without_prompt(monkeypatch):
    def fail_input(prompt):
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(builtins, "input", fail_input)

    assert transform.determine_processing_limit(
        total_names=25,
        rephrase_all=True,
        single_name=None,
    ) == 25


def test_invalid_existing_output_refuses_overwrite(tmp_path, monkeypatch):
    output_file = tmp_path / "yoruba_paraphrased_meanings.json"
    output_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(transform, "OUTPUT_FILE", str(output_file))

    with pytest.raises(transform.ExistingParaphrasesError, match="not valid JSON"):
        transform.load_existing_paraphrases()


def test_malformed_existing_results_refuses_overwrite(tmp_path, monkeypatch):
    output_file = tmp_path / "yoruba_paraphrased_meanings.json"
    output_file.write_text(json.dumps({"results": {"name": "Folasade"}}), encoding="utf-8")
    monkeypatch.setattr(transform, "OUTPUT_FILE", str(output_file))

    with pytest.raises(transform.ExistingParaphrasesError, match="malformed 'results'"):
        transform.load_existing_paraphrases()


def test_save_paraphrases_writes_valid_json_atomically(tmp_path):
    output_file = tmp_path / "yoruba_paraphrased_meanings.json"
    payload = {
        "summary": {"processed_this_run": 1},
        "results": [{"name": "Folasade", "variations": ["honor confers a crown"]}],
    }

    transform.save_paraphrases(str(output_file), payload)

    assert json.loads(output_file.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []
