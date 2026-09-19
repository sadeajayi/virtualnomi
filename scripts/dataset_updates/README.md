# Dataset update scripts

Scripts that add or update columns/rows on **nomi-stories/nomi-names** (or related Hugging Face datasets).

- **Requirements:** `HF_TOKEN` (and often `datasets`, `huggingface_hub`).
- **Run from:** Repo root (`virtualnomi/`).

| Script | Purpose |
|--------|---------|
| `add_attribution_column.py` | Add attribution column |
| `add_cultural_context.py` | Add cultural context |
| `add_dataset_fields.py` | Add dataset fields |
| `add_edo_names.py` | Add Edo names |
| `add_missing_story_names.py` | Add missing story names |
| `add_pronunciation_by_from_commits.py` | Backfill pronunciation_by from commits |
| `add_remaining_attributions.py` | Add remaining attributions |
| `add_themes.py` | Add theme tags |
| `add_validation_status_column.py` | Add validation status column |
| `batch_update_huggingface.py` | Sync approved paraphrased meanings to HF dataset |
| `upload_audio_from_local_file.py` | Embed local audio (m4a/wav) into Audio Pronunciation for one name |
| `normalize_pronunciation_by.py` | Normalize pronunciation_by field |
| `backfill_pronunciation_by.py` | Backfill pronunciation_by |
| `update_*.py` | Various column/row updates (Hausa, Igbo, Yoruba, etc.) |
| `update_yoruba_attributions.py` | Set Yoruba `Attribution` to `YorubaNames.com` (Nomi exceptions for newer rows); review with `--from-cache`, push only from live HF |
