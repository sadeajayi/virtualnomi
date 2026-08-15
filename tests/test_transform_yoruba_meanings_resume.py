import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "paraphrasing"))

import transform_yoruba_meanings as transform  # noqa: E402


def test_failed_existing_entries_do_not_block_resume(tmp_path, monkeypatch):
    output_file = tmp_path / "yoruba_paraphrased_meanings.json"
    output_file.write_text(
        json.dumps(
            {
                "results": [
                    {"name": "Folasade", "variations": ["honor brings a crown"]},
                    {"name": "Morenikeji", "variations": [], "error": "temporary"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(transform, "OUTPUT_FILE", str(output_file))

    _existing_results, existing_names = transform.load_existing_paraphrases()

    assert existing_names == {"folasade"}


def test_failed_rephrase_keeps_existing_success():
    existing = [
        {
            "name": "Morenikeji",
            "variations": ["I have found a companion"],
            "llm_model": "claude-sonnet-5",
        }
    ]
    failed_retry = [
        {
            "name": "Morenikeji",
            "variations": [],
            "error": "Failed to generate variations",
        }
    ]

    merged = transform.merge_paraphrase_results(existing, failed_retry)

    assert merged == existing


def test_successful_rephrase_replaces_prior_failed_entry():
    existing = [
        {
            "name": "Morenikeji",
            "variations": [],
            "error": "temporary provider failure",
        }
    ]
    successful_retry = [
        {
            "name": "Morenikeji",
            "variations": ["I have found a companion"],
            "llm_model": "claude-sonnet-5",
        }
    ]

    merged = transform.merge_paraphrase_results(existing, successful_retry)

    assert merged == successful_retry


def test_failed_retry_replaces_prior_failed_audit_entry():
    existing = [
        {
            "name": "Morenikeji",
            "variations": [],
            "error": "rate limit",
        }
    ]
    failed_retry = [
        {
            "name": "Morenikeji",
            "variations": [],
            "error": "json parse error",
        }
    ]

    merged = transform.merge_paraphrase_results(existing, failed_retry)

    assert merged == failed_retry
