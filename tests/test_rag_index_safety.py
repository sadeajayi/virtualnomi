import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

import index_language_papers  # noqa: E402
import rag_service  # noqa: E402


def test_streaming_index_write_preserves_existing_file_on_failure(tmp_path):
    index_file = tmp_path / "yoruba_papers_index.json"
    original = '{"metadata":{"ok":true},"chunks":[]}'
    index_file.write_text(original, encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        index_language_papers._write_index_streaming(
            index_file,
            {"language": "yoruba"},
            tmp_path / "missing_chunks.jsonl",
        )

    assert index_file.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".yoruba_papers_index.json.tmp").exists()


def test_streaming_index_write_replaces_with_valid_json(tmp_path):
    index_file = tmp_path / "yoruba_papers_index.json"
    jsonl_file = tmp_path / "chunks.jsonl"
    jsonl_file.write_text(
        json.dumps({"id": "chunk-1", "text": "A name excerpt."}) + "\n",
        encoding="utf-8",
    )

    index_language_papers._write_index_streaming(
        index_file,
        {"language": "yoruba", "total_chunks": 1},
        jsonl_file,
    )

    written = json.loads(index_file.read_text(encoding="utf-8"))
    assert written["metadata"]["language"] == "yoruba"
    assert written["chunks"] == [{"id": "chunk-1", "text": "A name excerpt."}]


def test_corrupt_rag_index_degrades_to_no_rag(tmp_path, monkeypatch):
    corrupt_index = tmp_path / "yoruba_papers_index.json"
    corrupt_index.write_text('{"metadata":', encoding="utf-8")
    rag_service._rag_instances.clear()

    def fake_config(rag_key):
        assert rag_key == "yoruba"
        return {
            "display_name": "Yoruba",
            "index_path": str(corrupt_index),
            "query_suffix": "Yoruba personal name",
            "morphemes": [],
        }

    monkeypatch.setattr(rag_service, "get_language_config", fake_config)

    service = rag_service.get_rag_service_for_dataset_language(
        "Yoruba", quiet=True, text_search_only=True
    )

    assert service is None
    assert "yoruba:text" not in rag_service._rag_instances
