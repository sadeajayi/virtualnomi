import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

import gradio as gr

DEFAULT_API_BASE = "https://nomi-name-search-api.onrender.com"
API_BASE = os.environ.get("API_BASE", DEFAULT_API_BASE).rstrip("/")
INSIGHTS_TIMEOUT_SEC = float(os.environ.get("INSIGHTS_TIMEOUT_SEC", "20"))
SEARCH_TIMEOUT_SEC = float(os.environ.get("SEARCH_TIMEOUT_SEC", "60"))

SURPRISE_QUERIES = [
    "Amara",
    "Kwame",
    "Nneoma",
    "Folasade",
    "Adéọlá",
    "Imani",
    "Zuberi",
]


def _api_get(path: str, timeout: float) -> Optional[dict]:
    url = f"{API_BASE}{path}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


def fetch_search(query: str) -> List[Dict]:
    params = urllib.parse.urlencode({"q": query.strip(), "language": "All"})
    data = _api_get(f"/search?{params}", SEARCH_TIMEOUT_SEC)
    if not data:
        return []
    return (data.get("results") or [])[:3]


def fetch_insight(name: str, language: str) -> Optional[str]:
    params = urllib.parse.urlencode(
        {"name": name, "language": language or ""},
        quote_via=urllib.parse.quote,
    )
    data = _api_get(f"/insights?{params}", INSIGHTS_TIMEOUT_SEC)
    if not data:
        return None
    insight = (data.get("insight") or "").strip()
    return insight or None


def _story_text(result: Dict) -> str:
    story = result.get("story")
    if isinstance(story, dict):
        for key in ("story", "text", "body", "content"):
            val = (story.get(key) or "").strip()
            if val:
                return val
    return ""


def create_name_card(name_data: Dict, insight: Optional[str] = None) -> str:
    name = name_data.get("name") or name_data.get("name_strip") or ""
    phonetic = (name_data.get("phonetic_spelling") or "").strip() or "—"
    meaning = (name_data.get("meaning") or "").strip() or "—"
    language = (name_data.get("language") or "").strip()
    origin = language or "—"
    story = _story_text(name_data)
    pronunciation = phonetic

    insight_html = ""
    if insight:
        insight_html = f"""
        <div style="margin-bottom: 12px;">
            <strong style="color: #4a5568; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Cultural insight</strong>
            <p style="margin: 4px 0 0 0; color: #2d3748; font-size: 16px; line-height: 1.5;">
                {insight}
            </p>
        </div>
        """

    story_html = ""
    if story:
        story_html = f"""
        <div style="margin-bottom: 12px;">
            <strong style="color: #4a5568; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Story</strong>
            <p style="margin: 4px 0 0 0; color: #2d3748; font-size: 16px; line-height: 1.5;">
                {story}
            </p>
        </div>
        """

    audio_url = (name_data.get("audio_url") or name_data.get("pronunciation_url") or "").strip()
    play_btn = ""
    if audio_url:
        play_btn = f"""
            <a href="{audio_url}" target="_blank" rel="noopener" style="
                display: inline-block;
                background: #4299e1;
                color: white;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            ">🔊 Play pronunciation</a>
        """
    elif pronunciation and pronunciation != "—":
        play_btn = f"""
            <span style="color: #718096; font-size: 14px;">Pronunciation: {pronunciation}</span>
        """

    return f"""
    <div style="
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <h3 style="margin: 0 0 15px 0; color: #2d3748; font-size: 24px; font-weight: 600;">
            {name}
        </h3>

        <div style="margin-bottom: 12px;">
            <strong style="color: #4a5568; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Phonetic</strong>
            <p style="margin: 4px 0 0 0; color: #2d3748; font-size: 16px; font-weight: 500;">
                {phonetic}
            </p>
        </div>

        <div style="margin-bottom: 12px;">
            <strong style="color: #4a5568; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Meaning</strong>
            <p style="margin: 4px 0 0 0; color: #2d3748; font-size: 16px;">
                {meaning}
            </p>
        </div>

        <div style="margin-bottom: 12px;">
            <strong style="color: #4a5568; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">Origin</strong>
            <p style="margin: 4px 0 0 0; color: #2d3748; font-size: 16px;">
                {origin}
            </p>
        </div>

        {insight_html}
        {story_html}

        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e1e5e9;">
            {play_btn}
        </div>
    </div>
    """


def _empty_state(message: str) -> str:
    return f"""
    <div style="
        text-align: center;
        padding: 40px 20px;
        color: #718096;
        font-size: 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        {message}
    </div>
    """


def handle_search(query: str) -> str:
    if not query:
        return _empty_state("Try a name above or tap 'Surprise me' to explore.")

    matches = fetch_search(query)
    if not matches:
        return _empty_state(
            "No matches yet. Try a simpler spelling or another name."
        )

    cards_html = ""
    for name_data in matches:
        insight = fetch_insight(
            name_data.get("name") or name_data.get("name_strip") or "",
            name_data.get("language") or "",
        )
        cards_html += create_name_card(name_data, insight=insight)

    return cards_html


def handle_surprise() -> str:
    query = random.choice(SURPRISE_QUERIES)
    return handle_search(query)


# Create Gradio interface
with gr.Blocks(
    title="Name Explorer Demo",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    .search-container {
        background: #f7fafc;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
    }
    .title {
        text-align: center;
        color: #2d3748;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 18px;
        margin-bottom: 30px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .helper-text {
        text-align: center;
        color: #a0aec0;
        font-size: 14px;
        margin-top: 8px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .cta-section {
        text-align: center;
        padding: 30px;
        background: #edf2f7;
        border-radius: 12px;
        margin-top: 30px;
    }
    .cta-text {
        color: #4a5564;
        font-size: 16px;
        margin-bottom: 15px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .cta-link {
        color: #4299e1;
        text-decoration: none;
        font-weight: 600;
        font-size: 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .cta-link:hover {
        text-decoration: underline;
    }
    """
) as demo:

    gr.HTML("""
        <div class="title">Hear the meaning in every name</div>
        <div class="subtitle">Try a few names from our archive. Start with African names; each card carries pronunciation, meaning, origin—and a short story.</div>
    """)

    with gr.Group():
        gr.HTML('<div class="search-container">')

        with gr.Row():
            search_input = gr.Textbox(
                placeholder="Type a name… try Nneoma, Amara, Folasade, Kwame",
                label="",
                scale=4,
                elem_id="search-input",
            )
            surprise_btn = gr.Button("Surprise me", scale=1, variant="secondary")

        gr.HTML('<div class="helper-text">Accent-friendly search. We\'ll surface up to 3 matches.</div>')
        gr.HTML('</div>')

    results_html = gr.HTML(
        value=_empty_state("Try a name above or tap 'Surprise me' to explore."),
        elem_id="results",
    )

    gr.HTML("""
        <div class="cta-section">
            <div class="cta-text">This is a small demo of our larger archive. We're building the cultural layer of digital identity starting with African names.</div>
            <a href="https://nomistories.com" class="cta-link">Share your name story →</a>
        </div>
    """)

    search_input.submit(
        fn=handle_search,
        inputs=[search_input],
        outputs=[results_html],
    )

    surprise_btn.click(
        fn=handle_surprise,
        inputs=[],
        outputs=[results_html],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=9000,
        share=False,
        show_error=True,
    )
