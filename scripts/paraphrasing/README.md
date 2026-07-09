# Paraphrasing scripts

Yoruba name meaning paraphrasing pipeline.

1. **identify_yoruba_transformations.py** — Find names to transform; writes `data/paraphrasing/yoruba_names_ab_for_transformation.json`.
2. **transform_yoruba_meanings.py** — Generate paraphrases (optionally with RAG); writes `data/paraphrasing/yoruba_paraphrased_meanings.json`.
   - Batch: `python3 scripts/paraphrasing/transform_yoruba_meanings.py` (prompts for count)
   - Single name: `python3 scripts/paraphrasing/transform_yoruba_meanings.py --name Folasade`
   - Regenerate one: add `--rephrase` if already in output JSON
   - Default LLM: `claude-sonnet-5` (set `PARAPHRASE_CLAUDE_MODEL` to override)
   - View one result: CLI one-liner in `docs/RAG_PARAPHRASING_INTEGRATION.md` or `python3 scripts/paraphrasing/paraphrase_lookup.py`
3. **review_paraphrased_variations.py** — Gradio UI to review and approve; writes `data/paraphrasing/approved_paraphrased_meanings.json`.
4. **paraphrase_lookup.py** — Local read-only Gradio lookup by name (canonical meaning, 5 variations, approval status).

Then run **scripts/dataset_updates/batch_update_huggingface.py** to sync approvals to the Hugging Face dataset.

See **docs/PARAPHRASING_README.md** and **docs/RAG_PARAPHRASING_INTEGRATION.md**.  
Run from repo root. Requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for transform; `HF_TOKEN` for batch update.
