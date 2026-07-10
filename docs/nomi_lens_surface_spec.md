# Nomi — Lens × Surface System Spec (v0)

Defines how the three-lens system (say-it / know-me / honor-me) renders across destination surfaces. Supersedes the assumption that lenses are a universal depth toggle applied uniformly everywhere.

---

## 1. Core finding

Say-it and know-me are content-invariant across surfaces. What changes is placement and register, not facts. Honor-me is not invariant. On some surfaces it is scholarship about the name. On others it is behavioral or practical guidance oriented to what someone in that role needs to do. That is a difference in content type, not depth, and it is the only lens that requires this spec.

**Consequence:** this document only builds a matrix for honor-me. Say-it and know-me get a short register note each, not a matrix.

---

## 2. Say-it — scope ruling

One content object. Pronunciation, phonetic spelling, audio. Identical across all surfaces and all sub-contexts. Only the rendering container changes (name bar, pill, chart header). No per-surface content pipeline. No further spec needed.

## 3. Know-me — scope ruling

Dataset fields (origin, short meaning, role, location) are also invariant. What legitimately varies is register and length, not content:

- **LinkedIn / video (keynote):** full sentence-length meaning, Fraunces italic, room to read.
- **Canvas roster row:** same fields, compressed to fit a row. No register change beyond brevity.
- **Epic patient strip:** same fields, compressed, sits next to clinical metadata rather than a bio. Slightly more clinical typographic treatment (already reflected in the sandbox's Know-me Epic panel), but no different facts.

This is a style-guide-level rule, not a content pipeline. No matrix required.

**Operational preferred form at know-me and above:** `preferred_form` (or `submitted_preferred_form` from intake) is operational metadata — how the person wants to be addressed. It renders at **know-me and above** on **all surfaces** (institutional and personal): roster rows, chart strips, professional cards, and the personal `/[slug]` page. It is not scholarship; it is a fact about how to address this person.

One exception surfaced in stress-testing: **education and healthcare have a practical-use need at honor-me** (confidence in saying the name unprompted, submitted preferred/short form) that doesn't fit know-me's "facts about the name" category. Rather than invent a fourth lens, pronunciation-confidence and submitted preferred-form lines are assigned to education and healthcare honor-me cells below, keeping know-me's definition clean everywhere. **These honor-me lines use `submitted_preferred_form` / `preferred_form` only — never `personal_note`.**

---

## 4. `personal_note` vs `preferred_form` (corrected rule)

Two distinct fields with different consent boundaries and surface eligibility.

| Field | Role | Where it renders | Lens gating |
|---|---|---|---|
| **`preferred_form`** | Operational — how to address this person | Everywhere at **know-me and above** (institutional + personal) | Follows surface lens rules |
| **`personal_note`** | Personal voice — warmth, family usage, nuance | **`/[slug]` personal card only**; also collected in `/share` prep flow | **Never lens-gated** on the personal card — always visible when present |

**`personal_note` hard exclusions:**

- **Never** on institutional surfaces (education, healthcare, professional mocks).
- **Never** in institutional API fields used for honor-me on edu/health.
- **Does not** feed honor-me on education or healthcare.

**Consent rationale:** Publishing a `personal_note` is an explicit choice to put personal voice on one's own page. Surfacing that same text in a chart note or class roster is **context collapse** — the reader treats it with the same authority as clinical or roster metadata. `preferred_form` is scoped operational data the person consents to share in those contexts; `personal_note` is not.

---

## 5. Honor-me — surface × sub-context matrix (v2)

Each cell: content type, tone, source, and a real example.

**Honor-me v2 sources (implemented):**

1. **RAG scholarship** — name as word + linguist attribution. **Professional and video only.**
2. **Individually submitted fields** — `submitted_preferred_form` / `preferred_form` + pronunciation-confidence flag. **Education and healthcare only.**

**Cut entirely:** Family B tradition-level conventions tables (formality norms per language/tradition, authored lookup tables). No fallback prose when honor-me sources are empty — the honor-me layer is **absent**, not padded.

**`personal_note` is excluded from the institutional honor-me path entirely.**

### Professional (LinkedIn)

| | |
|---|---|
| **Content type** | Two-box structure: **about_the_name** (headline + body from dataset/RAG) plus conditional **cultural_depth** (scholarly paragraph with attribution, shown only when it adds value beyond about_the_name) |
| **Tone** | Griot register, written for a colleague or recruiter reading a profile |
| **Source** | RAG over linguist-attributed research corpus, name-keyed |
| **Example (cultural_depth)** | "The name belongs to the oríkì tradition of praise-naming. Folá signals honour. Adé signals a crown. Together they mark a child whose life should carry both." — attributed to Ọláyẹmi Bámgbóṣé, Yoruba linguistics |
| **Honor-me empty state** | No honor-me block if `cultural_depth` and attribution are both missing |

**Two-box professional structure:**

- **Box 1 — `about_the_name`:** headline + body; always shown at know-me on professional surfaces.
- **Box 2 — `cultural_depth`:** conditional scholarly block at honor-me; rendered only when RAG returns attributed depth that is not redundant with box 1 (token-overlap / restatement check). If redundant or missing, box 2 is omitted — no fallback.

### Video call (keynote / panel introduction)

| | |
|---|---|
| **Content type** | Same scholarly content as professional, reshaped for spoken introduction |
| **Tone** | Third-person, brief, meant to be read aloud by a moderator, not read silently by a viewer |
| **Source** | Same RAG corpus as professional, reformatted by prompt template for spoken cadence, not regenerated from scratch |
| **Example** | "Ọbáfẹ́mi carries a name from a class of Yoruba names marking royal or divine attention. Ọbá names a king, or a deity." |
| **Note** | Rarely invoked. Routine video calls (team sync) should not surface honor-me by default; it belongs to formal introduction moments only. This is a policy decision about *when* to offer the lens, not just how to render it. |
| **Honor-me empty state** | Same as professional — absent when no attributed cultural_depth |

### Education (Canvas, day-one roster use)

| | |
|---|---|
| **Content type** | Practical-use / rapport note, not scholarship |
| **Tone** | Direct, practical, written for a professor or TA preparing to greet a student, not a bio |
| **Source** | Pronunciation-confidence flag (`pronunciation_verified`) + **`submitted_preferred_form` / `preferred_form` only** — not RAG, not tradition tables, **not `personal_note`** |
| **Example** | "Pronunciation verified from recording. Goes by Sade." |
| **Note** | Thin, rule-based lines. No generative fallback. If neither verified pronunciation nor submitted preferred form exists, honor-me is absent. |
| **Honor-me empty state** | No block when both sources are empty |

### Healthcare — v0 (conservative, ships now)

| | |
|---|---|
| **Content type** | Addressing conventions at the individual level: pronunciation confidence, preferred form |
| **Tone** | Clinical, terse, functional. No cultural scholarship, no behavioral inference |
| **Source** | Pronunciation-confidence flag + **`submitted_preferred_form` / `preferred_form` only** — not tradition-level lookup tables, **not `personal_note`** |
| **Example** | "Verified pronunciation. Preferred form: Fola." |
| **Note** | Deliberately does not include family decision-making, spiritual or ritual considerations, communication-style inference, or language-tradition formality tables. Deeper care-relevant cultural notes remain explicitly out of scope (see gap below). |
| **Honor-me empty state** | No block when both sources are empty |

### Healthcare — deeper layer (blocked, not built)

| | |
|---|---|
| **Content type** | Care-relevant cultural notes: communication style, family involvement in decisions, spiritual/ritual considerations, sub-context variants (acute care, intake, chaplaincy, palliative, behavioral health) |
| **Status** | **Explicitly out of scope for v0.** Named here as an open sourcing gap, not a design gap. |
| **Why blocked** | This content answers "how should a clinician behave toward this patient," inferred from a name's linguistic origin. That is inference about an individual based on ethnicity signalled by a name, not a fact about the name. Presented as a chart note with the same visual authority as MRN or allergy fields, it reads as clinical fact about the patient rather than a prompt to ask a question. No current Nomi contributor (linguists) is positioned to verify this; it requires a medical anthropologist, chaplain, or clinical cultural-competency specialist, none of whom Nomi currently has. |
| **If built later** | Content must be framed as prompts-to-ask ("consider asking whether family involvement in decisions is preferred") rather than assertions-about-the-patient ("this patient's family will be involved in decisions"). Sourced and attributed separately from linguistic RAG corpus. Requires a new sourcing chain — not resurrected Family B tradition tables. |

---

## 6. Schema and API implications

### Field routing (build contract)

| Field | Personal `/[slug]` | Institutional surfaces | Honor-me (edu/health) |
|---|---|---|---|
| `preferred_form` / `submitted_preferred_form` | know-me+ (always when set) | know-me+ (always when set) | **Yes** — honor-me lines when lens is honor-me |
| `personal_note` | Always visible when present (not lens-gated) | **Never** | **Never** |
| `cultural_depth` + attribution | Optional depth on personal card | Professional/video honor-me only | **Never** |
| Pronunciation-confidence flag | say-it / honor-me context | edu/health honor-me only | **Yes** |

### `/insights` routing

The endpoint should not treat "context" as a single parameter that reshapes one generation call.

- **Professional / video honor-me:** route to name-keyed RAG (cultural_depth + attribution), reshaped by a `surface` parameter for tone/length only.
- **Education / healthcare honor-me:** **no `/insights` generation** — honor-me is assembled from submitted operational fields on the client or a thin profile endpoint, not from RAG or tradition tables.
- **No Family B lookup path.** Tradition-level conventions tables are removed from v0.

---

## 7. Lens ceiling rule, for surfaces not yet built

As new surfaces are added (hotel check-in, restaurant reservation, customer service, and others), apply this rule before deciding what honor-me should look like there, rather than defaulting to "give it the deepest version available."

**Rule:** the ceiling on which lens a surface may offer is set by dwell time and relational context, not by what content happens to exist in the substrate.

- **High dwell time, ongoing relationship** (professional, education, healthcare, a returning hotel guest with a standing relationship) → honor-me may be offered, scoped to what that role's practical need actually is (see matrix above; it is never automatically the scholarly paragraph).
- **Low dwell time, transactional, single encounter** (hotel check-in, restaurant reservation, customer service call) → cap at know-me. A guest checking in for one night has not consented to a cultural-depth layer being generated about them at the front desk, and no one has the relational context to make that content land as care rather than performance. Do not build an honor-me variant for these surfaces. Leave it absent from the API's surface enum entirely, rather than shipping an unused or rarely-triggered option.
- **Formal but momentary** (video call keynote/panel introduction) → honor-me may be offered but only on explicit invocation for a formal-introduction moment, never as a default on a routine call. This is the video-call carve-out already noted in the matrix.

Write this rule into the surface-onboarding checklist now, so adding a fifth surface later is a lookup against this rule rather than a fresh argument each time.

---

## 8. Summary of what shipped v0 vs. what's flagged

**Ships now:** say-it and know-me unchanged. `preferred_form` at know-me+ everywhere. `personal_note` on `/[slug]` only, never lens-gated there, never on institutional surfaces. Professional and video honor-me as RAG scholarly paragraph (two-box: about_the_name + conditional cultural_depth). Education and healthcare honor-me as thin submitted-field lines (pronunciation confidence + preferred form). No fallback when honor-me sources are empty.

**Explicitly cut:** Family B tradition-level conventions tables for healthcare honor-me.

**Explicitly deferred, named as a gap:** healthcare honor-me's deeper layer (care-relevant cultural notes), blocked on finding a clinical or medical-anthropology collaborator. Sub-context differentiation (acute/intake/chaplaincy/palliative/behavioral health) deferred entirely, since it isn't in the individual-professional pilot's critical path.

**Written down now, not yet used:** the lens ceiling rule for future transactional surfaces, so hotel/restaurant/customer-service additions don't require re-deriving this reasoning.
