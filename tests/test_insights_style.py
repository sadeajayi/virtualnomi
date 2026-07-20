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
    monkeypatch.setattr(service, "load_insights_system_prompt", lambda: "voice-v8-post-gate")
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
    "text,expected_label",
    [
        ("Adunni carries sweetness — and belonging.", "em dash"),
        ("Adunni carries sweetness – and belonging.", "en dash as pause"),
        ("Adunni carries sweetness -- and belonging.", "double-hyphen dash"),
        ("Adunni is not only sweetness; it is welcome.", "not only X"),
        ("The name isn't merely praise. It gives a child a place.", "negative corrective parallelism"),
        ("Parents were not naming a child but seating her among kin.", "not X but Y"),
        ("This is praise rather than description.", "rather than"),
        ("It is worth noting that Adunni marks delight.", "hollow intensifier / hedge phrase"),
        ("This naming is a testament to family joy.", "significance inflation"),
        ("Researchers delve into the ọlá root here.", "AI Tier-1 filler"),
        ("Moreover, Adunni marks belonging.", "brochure / transition AI-ism"),
        ("This name carries a rich cultural heritage of welcome.", "AI Tier-1 filler"),
    ],
)
def test_avoid_ai_writing_gate_detects_banned_patterns(text, expected_label):
    violations = service.avoid_ai_writing_gate(text)
    assert violations
    assert expected_label in violations
    assert service.insight_style_violations(text) == violations


def test_compliant_cultural_prose_passes_gate():
    text = (
        "Adunni carries a family's delight in having this child among them. "
        "Parents give this Yoruba name when the birth itself feels like a gift."
    )
    assert service.avoid_ai_writing_gate(text) == []


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
    assert "em dash" in calls[1]
    assert service.avoid_ai_writing_gate(result["insight"]) == []


def test_tier1_filler_triggers_rewrite_then_accept(monkeypatch):
    calls = _prepare(
        monkeypatch,
        [
            "Adunni lets us unpack a tapestry of belonging.",
            "Adunni carries a family's delight in having this child among them.",
        ],
    )
    result = service.generate_insight_paragraph(
        "Adunni", "Yoruba", "Sweet to have"
    )
    assert len(calls) == 2
    assert "AI Tier-1 filler" in calls[1]
    assert result["insight"].startswith("Adunni carries")


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
