#!/usr/bin/env python3
"""Local Gradio lookup: search a Yoruba name → canonical meaning, variations, approval status."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA = _REPO_ROOT / "data" / "paraphrasing"
PARAPHRASED_FILE = _DATA / "yoruba_paraphrased_meanings.json"
APPROVED_FILE = _DATA / "approved_paraphrased_meanings.json"


def _normalize(name: str) -> str:
    return (name or "").strip().lower()


def load_paraphrases() -> Dict[str, dict]:
    if not PARAPHRASED_FILE.exists():
        return {}
    with open(PARAPHRASED_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {_normalize(item["name"]): item for item in data.get("results", []) if item.get("name")}


def load_approved() -> Dict[str, dict]:
    if not APPROVED_FILE.exists():
        return {}
    with open(APPROVED_FILE, encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        return {}
    return {_normalize(e["name"]): e for e in entries if e.get("name")}


def lookup_name(
    name: str,
    paraphrases: Dict[str, dict],
    approved: Dict[str, dict],
) -> Tuple[str, str, str, str]:
    key = _normalize(name)
    if not key:
        return "", "", "", "Enter a Yoruba name to search."

    item = paraphrases.get(key)
    if not item:
        matches = [p["name"] for k, p in paraphrases.items() if key in k][:10]
        hint = f"\n\nPartial matches: {', '.join(matches)}" if matches else ""
        return name, "", "", f"No paraphrase data for **{name.strip()}**.{hint}"

    original = item.get("original_meaning", "")
    variations = item.get("variations") or []
    variations_text = "\n\n".join(f"{i + 1}. {v}" for i, v in enumerate(variations))

    approval = approved.get(key)
    if approval:
        status = (
            f"**Approved** ({approval.get('timestamp', 'unknown date')})\n\n"
            f"{approval.get('approved_meaning', '')}"
        )
    else:
        status = "**Not yet approved** — use `review_paraphrased_variations.py` to approve."

    meta = []
    if item.get("priority"):
        meta.append(f"priority: {item['priority']}")
    if item.get("has_yorubanames"):
        meta.append("has yorubanames attribution")
    meta_line = f"\n\n_{' · '.join(meta)}_" if meta else ""

    return name.strip(), original, variations_text, status + meta_line


def create_interface() -> gr.Blocks:
    paraphrases = load_paraphrases()
    approved = load_approved()
    count = len(paraphrases)
    approved_count = len(approved)

    with gr.Blocks(title="Yoruba Paraphrase Lookup") as interface:
        gr.Markdown(
            f"# Yoruba Paraphrase Lookup\n"
            f"Search **{count:,}** paraphrased names "
            f"({approved_count:,} approved). Read-only — does not generate new paraphrases."
        )
        name_input = gr.Textbox(label="Yoruba name", placeholder="e.g. Afolabi")
        search_btn = gr.Button("Look up", variant="primary")
        name_out = gr.Textbox(label="Name", interactive=False)
        original_out = gr.Textbox(label="Canonical / original meaning", interactive=False, lines=2)
        variations_out = gr.Textbox(label="AI-generated variations", interactive=False, lines=10)
        status_out = gr.Markdown(label="Approval status")

        def do_lookup(name: str):
            return lookup_name(name, paraphrases, approved)

        search_btn.click(
            do_lookup,
            inputs=[name_input],
            outputs=[name_out, original_out, variations_out, status_out],
        )
        name_input.submit(
            do_lookup,
            inputs=[name_input],
            outputs=[name_out, original_out, variations_out, status_out],
        )

    return interface


if __name__ == "__main__":
    if not PARAPHRASED_FILE.exists():
        print(f"Missing {PARAPHRASED_FILE}. Run transform_yoruba_meanings.py first.")
        raise SystemExit(1)
    create_interface().launch(share=False)
