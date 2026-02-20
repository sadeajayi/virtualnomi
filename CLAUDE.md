# CLAUDE.md

This file gives Claude (and other AI tools) consistent context when working on the Nomi project. Use it for coding, planning, research, or converting this repo into a Claude project.

---

## Project context

**Nomi** is a language/culture product focused on **African names**: meanings, pronunciations, and the stories behind them. The goal is to preserve and make name data accessible, support correct pronunciation in digital tools, and reduce pressure to anglicize or shorten names when people move or work in the West.

**Stage:** Early product; core data and apps are live (e.g. Hugging Face Spaces). We iterate on meaning quality (paraphrasing from research papers), pronunciation tooling, and validate next angles (naming journey for parents, tools for educators) using customer research.

**Key surfaces:**
- **nomi-name-search** – Search/browse names (meanings, pronunciation, stories, optional RAG “research insights”).
- **nomi-meaning-review** – Review and approve paraphrased meanings.
- **nomi-meaning-inbox** – Contribute or suggest meanings.
- **nomi-pronunciation-inbox** – Contribute pronunciations.
- **Data:** Hugging Face dataset `nomi-stories/nomi-names`; local data in `data/` (e.g. paraphrasing, visitor log).

---

## Where key info lives

**Start here for “what’s the state of Nomi?” and “what are we planning?”**

| Need | Doc / location |
| --- | --- |
| Current phase, last updated, quick links | [PROJECT_STATUS.md](./docs/PROJECT_STATUS.md) |
| Product/eng plans by theme (paraphrasing, RAG, naming journey, schools) | [ROADMAP.md](./docs/ROADMAP.md) |
| Why we built RAG, API, customer research, paraphrasing | [DECISIONS_AND_HISTORY.md](./docs/DECISIONS_AND_HISTORY.md) |
| Pivot angles, monetization, positioning | [STRATEGY_AND_MONETIZATION.md](./docs/STRATEGY_AND_MONETIZATION.md) |
| Founder/product-lead context (persona, role, what they care about, current challenges) | [FOUNDER_CONTEXT.md](./docs/FOUNDER_CONTEXT.md) |

**Technical and feature docs:** [README.md](./docs/README.md) — paraphrasing pipeline, RAG usage, phonetic updater, dataset safety, etc.

**Customer research:** [README.md](./customer_research/README.md) — baby naming interviews, names-in-schools interviews, Reddit/discourse archives (mispronunciation, Chinese names in the West, cultural pride).

**Code layout:** `rag/`, `search/`, `scripts/` (paraphrasing, dataset_updates, pronunciation, utils), `data/`, `requirements/`, `tests/`. App folders (`nomi-name-search`, `nomi-meaning-review`, etc.) stay at repo root for Hugging Face Spaces.

---

## Your role when working on Nomi

When helping with this repo, you are expected to:

- **Respect existing structure** – Don’t move app folders or break Space paths; fix imports/paths when scripts or docs move.
- **Use the docs above** – Before proposing product or roadmap changes, read PROJECT_STATUS, ROADMAP, and (if relevant) DECISIONS_AND_HISTORY, STRATEGY_AND_MONETIZATION, and FOUNDER_CONTEXT (persona, priorities, pain points).
- **Ground suggestions in context** – Customer research lives in `customer_research/`; research-paper context flows through RAG and paraphrasing. Use that when suggesting features or copy.
- **Keep canonical data first** – Paraphrasing and display layers sit on top of a single source of truth for meanings; don’t propose changes that split or duplicate canonical meaning storage without good reason.
- **Prefer links over duplication** – Point to existing docs (e.g. `docs/ROADMAP.md`) rather than re-describing long decisions in new files.

For **subagents or a Claude project:** instruct them to read `docs/PROJECT_STATUS.md` first, then the specific doc (ROADMAP, DECISIONS_AND_HISTORY, STRATEGY_AND_MONETIZATION) by task. Use section names (e.g. “Paraphrasing”, “RAG”) in prompts so they can target the right part of the file.

---

## Conventions and terminology

- **Canonical meaning** – The single source-of-truth meaning we store; paraphrases are derived for display.
- **RAG** – Retrieval over indexed Yoruba research papers; used for cultural context in paraphrasing and (optionally) “research insights” in name search.
- **Spaces** – Hugging Face Spaces (nomi-name-search, etc.); paths and imports are set so they run from repo root with `rag/` and `data/` available.
- **NameStrip** – Normalized name key (e.g. lowercase, stripped) used across dataset and code.

---

## Project memory (and extra context)

*(Optional: paste in tips for a project memory file, or links to docs that give deeper context on Nomi—e.g. vision, user personas, or constraints that should stay stable across sessions.)*

**Tips for a project memory file:**
- (Add your preferred format: e.g. “Store decisions as dated bullets in DECISIONS_AND_HISTORY.md”; “Keep PROJECT_STATUS.md as the single ‘current phase’ entry”; “When building a new feature, always check ROADMAP and customer_research for relevance.”)
- (Add any rules like: “Don’t change X without updating Y”; “Always run Z from repo root.”)

**Extra context docs you provide:**
- **Founder/product-lead:** [FOUNDER_CONTEXT.md](./docs/FOUNDER_CONTEXT.md) — persona, role, what they care about, current pain points; read when reasoning about priorities, roadmap, or strategy.
- (Other docs as you add them: e.g. `docs/VISION.md`, `docs/USER_PERSONAS.md`.)

---

## Working with this repo

- **Run scripts from repo root** – Many scripts assume `Path(__file__).resolve().parent.parent` (or similar) as repo root and paths like `data/paraphrasing/`, `rag/`, `nomi_search_eval/`.
- **Don’t assume a single “main” app** – There are several Spaces and scripts; which one matters depends on the task.
- **When planning features** – Check ROADMAP for current themes (paraphrasing, RAG, naming journey, schools) and STRATEGY_AND_MONETIZATION for pivot/monetization so suggestions align with stated direction.
- **When explaining “why”** – Use DECISIONS_AND_HISTORY and customer_research so answers are consistent with past research and decisions.

Focus on **clarity, consistency with existing docs, and preserving what’s already working** (Spaces, data layout, canonical structure).

## Writing Style
**Tone:**
- Clear and outcome-focused
- Active voice (not passive)
- Concise (2-sentence max paragraphs for most content)
- Use "we" not "I" in documentation
- Avoid jargon unless it's standard PM terminology

**Formatting:**
- Always use Oxford commas
- Use bullet points for lists (not numbered unless sequence matters)
- Bold key terms on first use
- Include "Why this matters" sections in PRDs