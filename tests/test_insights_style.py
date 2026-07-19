import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nomi-name-search-api"))

import insights_service as service  # noqa: E402


class _FakeAnthropic:
    class Anthropic:
        def __init__(self, api_key):
            self.api_key = api_key


def _prepare(monkeypatch, outputs):
    calls = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
    monkeypatch.setattr(service, "anthropic", _FakeAnthropic)
    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "voice-v4")
    monkeypatch.setattr(
        service,
        "gather_rag_context",
        lambda *_: (
            "[Yoruba_Names_Gender_Markings.pdf]: Adunni means sweet to possess.",
            ["Yoruba_Names_Gender_Markings.pdf"],
            True,
            "yoruba",
        ),
    )

    def model_call(*_args, **kwargs):
        calls.append(kwargs.get("retry_note", ""))
        return outputs[min(len(calls) - 1, len(outputs) - 1)]

    monkeypatch.setattr(service, "_call_insights_model", model_call)
    service._clear_insight_cache()
    return calls


@pytest.mark.parametrize(
    "text",
    [
        "Adunni carries sweetness — and belonging.",
        "Adunni is not only sweetness; it is welcome.",
        "The name isn't merely praise. It gives a child a place.",
        "Parents were not naming a child but seating her among kin.",
        "This is praise rather than description.",
    ],
)
def test_style_validator_detects_banned_voice_patterns(text):
    assert service.insight_style_violations(text)


def test_style_violation_gets_one_corrective_retry(monkeypatch):
    calls = _prepare(
        monkeypatch,
        [
            "Adunni carries sweetness — she was not only named, but welcomed.",
            "Adunni carries a family's delight in having this child among them.",
        ],
    )
    result = service.generate_insight_paragraph(
        "Adunni", "Yoruba", "Sweet to have"
    )
    assert len(calls) == 2
    assert "Detected violations" in calls[1]
    assert service.insight_style_violations(result["insight"]) == []


def test_second_style_failure_is_rejected_and_not_cached(monkeypatch):
    calls = _prepare(
        monkeypatch,
        ["Adunni carries sweetness — a welcome.", "Adunni was not just named but welcomed."],
    )
    with pytest.raises(service.OffVoiceInsightError):
        service.generate_insight_paragraph("Adunni", "Yoruba", "Sweet to have")
    assert len(calls) == 2
    assert service._insight_result_cache == {}


def test_compliant_output_returns_without_retry(monkeypatch):
    calls = _prepare(
        monkeypatch,
        ["Adunni carries a family's delight in having this child among them."],
    )
    service.generate_insight_paragraph("Adunni", "Yoruba", "Sweet to have")
    assert len(calls) == 1
