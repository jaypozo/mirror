# Mirror PRD — Voice & Style Mimicry

## Summary

Mirror is a private, self-hosted "learning model of the owner." It ingests the
owner's own message history, and when a new message arrives it drafts a reply in
the owner's voice for a human **Approve / Edit / Dismiss** decision. Nothing is
ever sent automatically.

This document is the product spec for the **voice/style-mimicry capability**: the
mechanisms by which Mirror's drafts come to read as if the owner wrote them, and
the feedback loop that keeps closing the gap.

## Goal

Draft replies **indistinguishable from how the owner writes**, and keep improving
from the owner's own edits. Success is when the owner approves a draft unchanged,
because it already sounds like them — length, cadence, punctuation, directness,
and all.

The owner's **full historical message corpus is the primary signal**. Their
edits are a refinement layer on top. The model is a stateless, closed-weight LLM
(GPT-5.5 via `codex exec`), so every technique here is prompt-time and data-time,
not weight-time.

## Non-goals

- No automatic sending. Every outbound reply is gated behind an explicit human
  Approve; Edit and Dismiss are equally first-class.
- No fine-tuning of the drafting model in this phase (the model is closed-weight;
  see [Roadmap](#roadmap--priorities)).
- No hosted UI; the approval card renders in the owner's own Telegram.

## What "style" decomposes into (the rubric)

Style is measurable. Stylometry — the quantitative study of writing style — shows
that authorship is captured by **content-independent** features: function-word
rates, punctuation, and sentence-length variance, not topic words
([StyleDistance, arXiv:2410.12757](https://arxiv.org/abs/2410.12757); classic
stylometry / function-word authorship attribution, e.g. Mosteller & Wallace on
the Federalist Papers). Mirror scores every candidate style sheet against this
rubric (`agent/style_sheet.py::compute_stylometry`):

- **Message length distribution** — mean/median/p10/p90 characters and words.
- **Sentence length + variance** — mean and standard deviation of words per
  sentence. Humans vary; uniform sentence length reads as machine.
- **Fragments vs. full sentences** — share of messages with no terminal
  punctuation.
- **Punctuation habits** — rate per 1k chars of em-dash, ellipsis, exclamation,
  question, comma. (Em-dash usage is called out explicitly; over-use of em-dashes
  is a common tell of machine text.)
- **Capitalization / casing** — lowercase-start share, all-lowercase share.
- **Contractions** — rate per 1k words.
- **Openings / closings / sign-offs** — most common first and last tokens.
- **Directness vs. hedging (BLUF)** — does the writer lead with the point?
- **Formatting** — bullets, line breaks, pasted logs.
- **Emoji usage** — emoji per message.
- **Signature phrases / tics** — recurrent bigrams and trigrams.
- **Function-word tendencies** — rate of closed-class words (the strongest,
  most content-independent authorship fingerprint).

## Architecture

```
Telegram (owner's USER account)
        │  Telethon USER session — sanctioned self-read
        ▼
ingest ──► Postgres + pgvector  = the PRIMARY signal (owner's FULL history)
        │       │
        │       ├── local sentence-transformers embeddings (384-dim, on-box)
        │       ▼
        │   message_embeddings (pgvector)
        │       │
incoming msg ───┤  retrieve top-K of the owner's own past replies (direction='out')
        │       ▼
        │   draft prompt  =  [living style sheet]
        │                  + [retrieved few-shot exemplars]
        │                  + [recent edit corrections]
        │       ▼
        │   stateless LLM draft (GPT-5.5 via codex exec)
        │       ▼
        │   Approve / Edit / Dismiss  ──► feedback table
        │                                     │
        └─────────────────────────────────────┘  edits refine the style sheet
```

Everything runs on-box with no third-party API key: local embeddings, and
drafting through `codex exec` (ChatGPT OAuth). The corpus never leaves the
machine for a hosted embedding or chat API.

### The corpus is the primary signal

The owner's full outbound history (`direction='out'`) is ground truth for their
voice. The living style sheet is built primarily by sampling and measuring that
corpus; the retrieval layer surfaces real past replies as few-shot examples;
edits are a *refinement* layer, valuable but secondary in volume.

### The living style sheet (`agent/style_sheet.py`)

A distilled, always-injected "how the owner writes" guide, persisted in Postgres
and regenerated periodically (not per-draft). Generation is **one LLM call** that
turns (a) the measured stylometric profile over a corpus sample, (b) a sample of
real messages, and (c) accumulated edits, into a short markdown style guide plus
candidate correction rules. The active guide is injected into every draft
alongside the exemplars and edit corrections; a missing or failed sheet degrades
gracefully (drafting still works).

This is the **Author Writing Sheet** technique from
[ACL 2024.personalize-1.6 (Learning to Generate Text in Arbitrary Writing
Styles)](https://aclanthology.org/2024.personalize-1.6/): a compact,
human-readable author descriptor that conditions generation and generalizes
better than raw exemplars alone.

#### Staged rule promotion (anti-overfit)

A single edit is noise; a *pattern* of edits is signal. To avoid over-fitting to
one-off corrections, rules mined from edits are staged in `style_rules`:

- A candidate rule graduates into the **active** guide only after it is supported
  by **≥ `STYLE_RULE_PROMOTE_THRESHOLD` (default 3) independent edits** — support
  is the distinct set of `feedback.id` values backing it.
- One-offs stay as `candidate` and never reach the drafter.
- Promoted rules that stop recurring across regenerations accrue `misses` and
  **decay** back out (`status='decayed'`) after `STYLE_RULE_DECAY_MISSES`.

Regenerate on demand with `python -m agent.style_sheet`; schedule a periodic
refresh with the units in `deploy/systemd/` (a daily timer is provided but not
installed by the repo).

### The edit-feedback loop (`agent/feedback.py`, `agent/draft.py`)

Every Approve / Edit / Dismiss decision is written to `feedback`. Edits are the
strongest single learning signal: the delta between what Mirror drafted and what
the owner actually sent is a direct correction. `fetch_recent_edits` injects
recent edits as few-shot corrections into the draft prompt, ranked by **both
recency and edit magnitude** (a bigger rewrite, measured by normalized edit
distance, outranks a one-word tweak) and **capped** so corrections never crowd
out the exemplars or the style sheet. This "learn from edits / fine-grained
feedback" approach follows
[arXiv:2512.23693 (fine-grained feedback / edit pairs)](https://arxiv.org/abs/2512.23693)
and the personalization-from-history framing of
[arXiv:2308.07968 (Teach LLMs to Personalize)](https://arxiv.org/abs/2308.07968).

### Dual learning: STYLE edits vs. INTENT edits (`agent/edit_classify.py`)

Not every edit is a voice correction. When the owner rewrites a draft they may
change **style** (phrasing/tone/length, same meaning), **intent/substance** (a
fact, decision, number, or commitment changed), **both**, or make a **trivial**
tweak. Training a single "prefer this" channel on all of them is wrong: a
decision change would teach the drafter a phrasing "rule" that was never about
phrasing.

At capture (the `/decide` edit path, after the send already succeeded, fully
guarded) a lightweight LLM classifies `original_draft` vs. `final_text` into
`style | intent | both | trivial` with a one-line what-changed note, persisted on
the `feedback` row (`edit_kind`, `edit_note`). Learning is then **routed**:

- **STYLE / BOTH → the voice channel.** `fetch_recent_edits` (the draft-time
  voice corrections) and the style-sheet rule miner both now filter to
  `edit_kind IN ('style','both')` (plus legacy unclassified rows) and **exclude
  pure `intent`** — a decision change never trains voice.
- **INTENT / BOTH → the intent/decision channel.** The what-changed note is stored
  as a durable **intent note** (`intent_notes`) tied to the feedback row and the
  matched thread (below), so future goal-summaries and drafts reflect what the
  owner actually decided, not a stale draft's guess.

A classify/DB failure degrades to a local heuristic and never blocks capture or
sending.

### Thread-aware goal state (`agent/threads.py`)

The owner runs **multiple threads interleaved in one conversation**. A flat
summary of the last N messages can't tell which thread a message belongs to, nor
track where in a task we are. Mirror now segments each incoming message to the
best-matching **active thread** (or opens a new one) and maintains per-thread
`goal`, `current_task`, and `stage` (`mid-step | awaiting-owner | done`):

1. Embed the incoming message locally (the same on-box encoder retrieval uses)
   and **pre-rank** candidate threads by cosine distance to their stored
   `anchor_embedding` (falling back to most-recently-updated).
2. **One LLM call** does segmentation + state update + summary together: given the
   message, recent thread, the candidate threads, and the matched thread's
   recent intent notes (locked-in decisions), it returns which thread this is (or
   a new one), the refreshed goal/current_task/stage, and a thread-aware
   goal/now/next/open. This call **replaces** the flat `summarize_thread` call, so
   drafting stays at the same ~one-exec-for-the-reply + one-for-state cost.
3. The matched thread's state + recent decisions are injected into the draft
   prompt so the reply understands the objective and where in the task we are, and
   the thread-aware summary becomes the card's Goal/Now/Next.

An agent-supplied Goal/Now/Next (passed in the request) is still honored. If
segmentation or state update fails, it falls back to the flat `summarize_thread`
and drafting still works.

## Techniques (ranked for a stateless, closed-weight setup)

Recent work shows LLMs can imitate a target style from **very few examples** when
those examples are well chosen — few-shot style imitation lifts style match by a
large margin ([arXiv:2509.24930, "few-shot 23×"](https://arxiv.org/abs/2509.24930)),
and style is detectable/attributable enough that content-independent
representations matter ([arXiv:2509.14543, "Catch Me If You Can"](https://arxiv.org/abs/2509.14543)).
Ranked by fit for Mirror's stateless `codex exec` drafter:

- **(A) Style-selected few-shot exemplars** — 2–5 of the owner's real past
  replies as in-context examples. Biggest lift for least cost. **Shipped**
  (retrieval → few-shot).
- **(B) Persistent distilled style card / "Author Writing Sheet"** — the living
  style sheet. **Shipped.**
- **(C) Learn-from-edits few-shot corrections** — recent edits as explicit
  drafted-vs-sent corrections. **Shipped** (recency + magnitude weighted).
- **(D) Style-embedding retrieval** — select exemplars by *style* similarity, not
  topic similarity. **Next.** (See the key insight below.)
- **(E) Retrieve → rank → summarize → synthesize** — a multi-stage pipeline that
  summarizes retrieved context before drafting ([arXiv:2308.07968](https://arxiv.org/abs/2308.07968)).
  Partially present (optional thread summary); expand later.
- **(F) Fine-tuning / LoRA** — poor fit now: the drafting model is closed-weight,
  so we cannot own or update its weights.
- **(G) DPO / ORPO from approve-vs-edit preference pairs** — future. Approve
  (chosen) vs. the draft that was edited (rejected) are natural preference pairs
  ([DPO, arXiv:2305.18290](https://arxiv.org/abs/2305.18290)). Needs owned
  weights, so we **bank G-ready preference rows now** (the `feedback` table
  already stores original draft vs. final text) and apply them once an
  open-weight drafting path exists.

**Recommendation implemented: A + B + C now; D next; bank G-ready pairs for
later.**

### Key insight: semantic ≠ stylistic similarity

The retrieval layer currently uses semantic embeddings (`all-MiniLM-L6-v2`),
which retrieve **topic-similar** exemplars — good for relevance, but not
guaranteed to be **style-similar**. Style similarity needs **content-independent**
representations such as StyleDistance / LUAR
([arXiv:2410.12757](https://arxiv.org/abs/2410.12757)). Technique (D) adds a
style-embedding index so exemplar selection optimizes for *how* the owner wrote,
independently of topic. The living style sheet (B) already compensates by
injecting content-independent style features directly, but style-based retrieval
is the principled next step.

## Eval / scoreboard

Style is measurable, so mimicry quality is measurable:

- **Approve-without-edit rate** — the primary product metric. Rising = drafts
  land as-is more often.
- **Edit distance draft→final** — normalized edit distance on the edits that do
  happen; should trend down as the style sheet and corrections improve.
- **Style-embedding cosine** vs. held-out real owner messages — content-independent
  style similarity of drafts to genuine messages.
- **Blind impersonation test** — periodic human check: can the owner tell their
  own message from a Mirror draft?

All four are computable from the `feedback` table plus a held-out slice of the
corpus.

## Roadmap / priorities

1. **Edit-feedback loop** — recency+magnitude-weighted, capped. **Shipped.**
2. **Living style sheet** with staged rule promotion + decay. **Shipped.**
3. **Dual learning (STYLE vs. INTENT edits)** — classify each edit; route style to
   the voice channel and intent to per-thread decision notes. **Shipped.**
4. **Thread-aware goal state** — segment interleaved threads; per-thread
   goal/current_task/stage feeds the summary + draft. **Shipped.**
5. **Style-based retrieval (D)** — add a style-embedding index; select exemplars
   by style, not topic.
6. **Scoreboard** — persist the four eval metrics and surface a trend.
7. **Preference-pair banking (G-ready)** — the `feedback` schema already captures
   chosen/rejected pairs; formalize export for a future DPO/ORPO run once an
   open-weight drafting path exists.

## Security & privacy

- Reading the owner's own history via a Telethon **USER** session is the
  sanctioned path for accessing one's own data. The corpus stays private and
  on-box.
- Nothing is sent without an explicit human Approve/Edit.
- No secrets in git: `.env`, the Telethon session, and DB dumps are gitignored.
  This document and the code refer only to "the owner"/"the user" and contain no
  personal identifiers or secret values.
- Fully local embeddings and on-box drafting mean the corpus never leaves the
  machine for a hosted API.

## References

- Catch Me If You Can — style detection/attribution: <https://arxiv.org/abs/2509.14543>
- LLMs imitate style few-shot (23×): <https://arxiv.org/abs/2509.24930>
- Author Writing Sheets (ACL 2024): <https://aclanthology.org/2024.personalize-1.6/>
- Teach LLMs to Personalize: <https://arxiv.org/abs/2308.07968>
- Fine-grained feedback / edit pairs: <https://arxiv.org/abs/2512.23693>
- Direct Preference Optimization (DPO): <https://arxiv.org/abs/2305.18290>
- StyleDistance — content-independent style embeddings: <https://arxiv.org/abs/2410.12757>
- Stylometry / function-word authorship attribution (Mosteller & Wallace, *Inference
  and Disputed Authorship: The Federalist*, 1964).
