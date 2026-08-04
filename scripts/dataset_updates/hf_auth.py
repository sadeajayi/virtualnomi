"""Compatibility helpers for Hugging Face authentication."""

from __future__ import annotations

try:
    from huggingface_hub import get_token as _get_token
except ImportError:  # pragma: no cover - for older huggingface_hub releases
    from huggingface_hub import HfFolder

    def _get_token() -> str | None:
        return HfFolder.get_token()


def get_hf_token() -> str | None:
    """Return the locally configured Hugging Face token, if available."""
    return _get_token()
