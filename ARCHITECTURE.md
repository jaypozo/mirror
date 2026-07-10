# Mirror — Architecture

Mirror is a self-hosted "drafts replies in your voice, learns from your edits"
system. It ingests your own Telegram history into a local vector store, and when
a new message arrives it retrieves how you replied to similar things — matched on
both **topic and writing style** — and drafts a reply **in your voice** for you
to Approve / Edit / Dismiss. Nothing is ever sent automatically.

Everything runs **on-box with no third-party API key**:

- **Embeddings** are computed locally with sentence-transformers — a *topic*
  model (`all-MiniLM-L6-v2`, 384-dim) and a *style* model
  (`StyleDistance/styledistance`, 768-dim).
- **Drafting** uses GPT-5.5 through `codex exec` (ChatGPT OAuth), one fresh
  stateless call per draft. Swappable for an OpenAI key.

This document is a clean, secrets-free description of the full current pipeline so
anyone (or their agent) can replicate it. No API ids, hashes, tokens, or
credentials appear here.

---

## Pipeline overview

```
  Telegram (your USER account)
        │  Telethon user session — sanctioned self-read (Takeout, paced, resumable)
        ▼
  ┌───────────────────────── INGEST (ingest/) ──────────────────────────┐
  │  pull.py / pull_recent.py / backfill.py  →  messages  (+ sync_state) │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │
        ┌────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
  embed.py (TOPIC)                                    embed_style.py (STYLE)
  MiniLM 384-dim, context_tag                         StyleDistance 768-dim
        │                                                   │  (owner 'out' msgs only)
        ▼                                                   ▼
  message_embeddings  ◄──────── Postgres + pgvector ──────►  message_style_embeddings
        └───────────────────────────┬───────────────────────┘
                                    │
  ══════════════ on an INCOMING message (approve_service /draft) ═══════════════
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                        ▼
  threads.py                                             retrieve.py
  segment → matched thread                               embed query (MiniLM)
  refresh goal/current_task/stage                        → topic-nearest POOL (~40)
  → thread-aware Goal/Now/Next                           → style-score each (StyleDistance)
  + recent intent notes                                  → BLEND 0.6·topic + 0.4·style
        │                                                → MMR diversity → top-6 exemplars
        │                                                       │ (+ preceding context)
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                    draft.py — build prompt:
                      [ living style sheet ]           (style_sheet.py, staged rules)
                    + [ style/topic few-shot exemplars ]
                    + [ recent edit-corrections ]      (style/both only — intent excluded)
                    + [ thread state + intent notes ]
                                    ▼
                    stateless LLM draft  (codex exec, GPT-5.5)   llm.py
                                    ▼
        ┌──────────────── /draft → card → /decide loop ─────────────────┐
        │  bot/plugin renders  Approve / Edit / Dismiss  (outside repo)  │
        └──────────────────────────────┬────────────────────────────────┘
                                       ▼
              approve  → send draft AS YOU (Telethon)   ┐
              edit     → send edited text AS YOU        ├─► feedback row (always)
              dismiss  → send nothing                   ┘
                                       │
        ┌──────────────────────────────┴───────────────────────────────┐
        ▼                              ▼                                ▼
  corpus.py::add_owner_sample   edit_classify.py               threads.py
  FINAL text → messages +       style|intent|both|trivial      record_intent_note
  topic + style embeddings      → routes learning:             (intent/both edits)
  (real id; draft NEVER)          style→voice, intent→intent
```

The **draft is never a training sample** and never enters `messages` or either
embedding table (see [Real-samples-only](#real-samples-only-a-draft-is-never-a-sample)).

---

## Data store (Postgres + pgvector)

Full DDL and per-column detail is in [`db/schema.sql`](db/schema.sql) and
[DATA.md](DATA.md). The tables:

| table | what it holds |
| --- | --- |
| `messages` | one row per Telegram message; `direction` = `'out'` (yours) / `'in'` (received). PK `(chat_id, id)`. |
| `sync_state` | per-chat forward cursor (`last_message_id`) — incremental, resumable ingest. |
| `message_embeddings` | **TOPIC** vector `vector(384)` per message + `context_tag`. PK `(chat_id, message_id, context_tag)`. |
| `message_style_embeddings` | **STYLE** vector `vector(768)` — *how* an owner message is written, content-independent. Owner real messages only. PK `(chat_id, message_id)`. |
| `feedback` | every Approve / Edit / Dismiss decision + `original_draft`, `final_text`, `edit_kind`, `edit_note`. The learning signal. |
| `threads` | per-thread `goal` / `current_task` / `stage` + `anchors` + `anchor_embedding vector(384)`. |
| `intent_notes` | durable decisions captured from intent/both edits, tied to a thread + feedback row. |
| `style_sheet` | the living distilled "how the owner writes" guide (`guide_md` + measured `rubric`); newest `active` row is injected into every draft. |
| `style_rules` | candidate correction rules mined from edits; `candidate → active` only after ≥ `STYLE_RULE_PROMOTE_THRESHOLD` independent edits; decay out on misses. |

> **Dimension note:** `message_embeddings.embedding` is `vector(384)` to match
> `all-MiniLM-L6-v2`; `message_style_embeddings.embedding` is `vector(768)` to
> match StyleDistance. Swapping either model means changing that dimension and
> re-embedding — the table must be empty to `ALTER` an existing vector length.

---

## Phase 1 — Corpus ingestion (`ingest/`)

All pullers use a **Telethon USER session** (your own account — a bot cannot read
account history). The `.telethon/*.session` is a single-writer SQLite file, so
**only one puller may run at a time** (the run scripts enforce this).

- **`ingest/pull.py`** — forward/full history. Iterates all dialogs with
  `iter_messages(min_id=cursor, reverse=True)` (only messages newer than the
  cursor, oldest-first). First run performs the interactive login.
- **`ingest/pull_recent.py`** — windowed catch-up of the last `PULL_SINCE_HOURS`.
  Runs inside a **Telegram Takeout** session (official export mode, lower flood
  limits); `connect()`-only and **refuses if the session isn't already
  authorized**.
- **`ingest/backfill.py`** — older history. Walks strictly-older pages per dialog
  (`offset_id = min(id)` already stored) so it reaches history behind the forward
  cursor. Resumable with no extra schema — `messages` is the cursor.

Pacing: batches of `PULL_BATCH_SIZE`, sleeps between batches/chats, and
`FloodWaitError → sleep-and-resume`. Upserts are `ON CONFLICT DO UPDATE`
(idempotent).

---

## Local embeddings

Two independent indexes are built from the corpus:

**TOPIC — [`ingest/embed.py`](ingest/embed.py).** Walks every message with text
and no embedding yet, assigns a coarse `context_tag` (`code-review`, `planning`,
`cs`, `personal`, else `general`), and encodes via the provider from
`EMBEDDING_PROVIDER`:

- **`local`** (default) — `all-MiniLM-L6-v2`, 384-dim, L2-normalized, CPU, off the
  event loop. No network / key.
- `openai` — optional, needs `EMBEDDING_API_KEY`.
- `dry-run` — zero vectors for wiring tests.

**STYLE — [`ingest/embed_style.py`](ingest/embed_style.py) + [`agent/style_embed.py`](agent/style_embed.py).**
Embeds **only the owner's own messages** (`direction='out'`) into a
content-independent style space that captures *how* a message is written (length,
register, punctuation, casing, formality) rather than its subject. Two
interchangeable local embedders, chosen by `STYLE_RETRIEVAL_MODE`
(`auto` | `model` | `features` | `off`):

- **`StyleModelEmbedder`** (preferred) — `STYLE_EMBED_MODEL` (default
  `StyleDistance/styledistance`), 768-dim, normalized, on-box, no key. Fills
  `message_style_embeddings`.
- **`StyleFeatureEmbedder`** (fallback) — a normalized stylometric feature vector
  (the same primitives the style sheet measures), used on the fly if the model
  can't load offline. Never persisted; style retrieval never hard-fails.

Both embed passes are idempotent and resumable. Run: `make embed` and
`make embed-style`.

---

## Retrieval — [`agent/retrieve.py`](agent/retrieve.py) (TOPIC + STYLE + MMR)

At draft time Mirror retrieves the owner's own past replies as few-shot voice
examples, **always restricted to `direction='out'`** (genuine owner replies —
never a draft). `retrieve_examples(query, ...)`:

1. **Over-fetch** a topic-nearest candidate pool (`STYLE_CANDIDATE_POOL`, default
   40) from `message_embeddings JOIN messages LEFT JOIN message_style_embeddings`,
   ordered by pgvector cosine (`<=>`), optional `context_tag` filter.
2. **Style-score** each candidate: embed the incoming message's register with the
   style embedder; `style_sim` = style-cosine to it (candidates without a stored
   style vector are embedded on the fly).
3. **Blend**: `base = STYLE_BLEND_TOPIC_WEIGHT·(1−dist) +
   STYLE_BLEND_STYLE_WEIGHT·style_sim` (defaults 0.6 / 0.4).
4. **MMR** (`STYLE_MMR_LAMBDA`, default 0.7) over the candidates' style vectors —
   greedily pick high-`base` exemplars while penalizing ones stylistically
   near-identical to those already picked, so the returned `top_k`
   (`RETRIEVE_TOP_K`, default 6) spans different lengths/registers.
5. For each pick, pull the up-to-3 preceding messages so the drafter sees *what
   you were replying to*.

Fully guarded: if the style embedder is unavailable or errors, it degrades to the
original **topic-only** nearest-neighbour order (the pool is already topic-sorted),
so drafting never breaks.

---

## Drafting — [`agent/draft.py`](agent/draft.py) + [`agent/llm.py`](agent/llm.py)

`draft_reply()`:

1. Query = the incoming message (+ a tail of the last 3 thread messages as extra
   semantic anchor).
2. `retrieve_examples()` → top-K owner replies + their context.
3. **Thread-aware state** ([`agent/threads.py`](agent/threads.py)): the message is
   segmented to its best-matching thread; goal/current_task/stage are refreshed,
   and a thread-aware Goal/Now/Next is produced. This one LLM call **replaces** the
   flat `agent/summarize.py` call (it doesn't add to it). The matched thread's
   state + recent intent notes are injected into the draft prompt. Falls back to
   the flat summary if segmentation fails. An agent-supplied summary is honored.
4. Prompt = a **system** prompt (*"draft the owner's reply in their voice; match
   tone/length/directness; output ONLY the reply; don't reveal AI; don't invent
   facts; don't copy examples verbatim"*) + the **living style sheet** + a **user**
   prompt (the retrieved exemplars, optional thread transcript, then "Now draft the
   reply to this incoming message: …").
5. One LLM call; draft LLM errors, timeouts, and empty output raise to the
   caller. The inline approve service logs those failures and returns
   `204 No Content`, so the fleet bot deletes any placeholder and renders
   nothing.

**LLM backends** (`LLM_PROVIDER`): **`codex`** (default) shells out to
`codex exec --skip-git-repo-check -s read-only -m gpt-5.5 -c
model_reasoning_effort=high`, reads the final message from a temp file, 300s
client default timeout. The inline approve `/draft` path overrides this to 60s.
ChatGPT OAuth → **no OpenAI API key** (~13s typical). Also `openai` (needs a
key) and `dry-run`.

---

## The living style sheet — [`agent/style_sheet.py`](agent/style_sheet.py)

A distilled, always-injected "how the owner writes" guide, persisted in
`style_sheet` and regenerated periodically (not per-draft). One LLM call turns
(a) a measured stylometric profile over a corpus sample, (b) a sample of real
owner messages, and (c) accumulated edits into a short markdown guide plus
candidate correction rules.

**Staged rule promotion (anti-overfit):** a single edit is noise; a *pattern* is
signal. Mined rules stage in `style_rules` — a candidate graduates into the active
guide only after **≥ `STYLE_RULE_PROMOTE_THRESHOLD` (default 3) independent
edits** support it (support = distinct `feedback.id`); one-offs stay `candidate`;
promoted rules that stop recurring accrue `misses` and `decay` out. Regenerate on
demand with `python -m agent.style_sheet`, or install the daily
[`deploy/systemd/`](deploy/systemd/) timer.

---

## Edit learning — [`agent/edit_classify.py`](agent/edit_classify.py) + [`agent/feedback.py`](agent/feedback.py)

Every Approve / Edit / Dismiss decision is written to `feedback`. Edits are the
strongest single learning signal — the delta between what Mirror drafted and what
you actually sent is a direct correction.

**Dual learning.** Not every edit is a voice correction. At capture (the `/decide`
edit path, after the send already succeeded, fully guarded) a lightweight LLM
classifies `original_draft` vs `final_text` into `style | intent | both | trivial`
with a one-line note (`edit_kind`, `edit_note`). Learning is then **routed**:

- **STYLE / BOTH → the voice channel.** `fetch_recent_edits` (draft-time voice
  corrections, ranked by recency + edit magnitude, capped) and the style-sheet
  rule miner both filter to `edit_kind IN ('style','both')` ∪ NULL and **exclude
  pure `intent`** — a decision change never trains voice.
- **INTENT / BOTH → the intent channel.** The note is stored as a durable
  `intent_notes` row tied to the feedback row and the matched thread, so future
  summaries + drafts honor what you actually decided.

A classify/DB failure degrades to a local heuristic and never blocks capture or
sending.

---

## Thread-aware goal state — [`agent/threads.py`](agent/threads.py)

You run multiple threads interleaved in one conversation; a flat window of the
last N messages can't tell which thread a message belongs to. Mirror segments each
incoming message to its best-matching **active thread** (or opens a new one) and
maintains per-thread `goal`, `current_task`, and `stage`
(`mid-step | awaiting-owner | done`):

1. Embed the incoming message locally and **pre-rank** candidate threads by cosine
   distance to their stored `anchor_embedding` (falling back to most-recently
   updated).
2. **One LLM call** does segmentation + state update + summary together, given the
   message, recent thread, candidate threads, and the matched thread's recent
   intent notes. This call **replaces** the flat `summarize_thread` call, so the
   hot path stays ~one exec for the reply + one for state.
3. The matched thread's state + recent decisions are injected into the draft
   prompt, and the thread-aware summary becomes the card's Goal / Now / Next.

Guarded — returns `None` on failure so drafting falls back to the flat summary.

---

## The `/draft → card → /decide` loop — [`agent/approve_service.py`](agent/approve_service.py)

A headless aiohttp service on loopback (`MIRROR_APPROVE_HOST` default
`127.0.0.1`, `MIRROR_APPROVE_PORT` default `8791`). Your bot/plugin renders the
draft card; this service does the drafting and is the **only** thing that sends as
you. It opens the Telethon USER client with `.connect()` only, **refuses if not
already authorized** (never logs in), and verifies the session owner is
`MIRROR_OWNER_USER_ID`. Every request must carry header `X-Mirror-Secret` ==
`MIRROR_APPROVE_SECRET` (**fail closed** — no secret configured means refuse
everything).

Endpoints (full request/response shapes in [SETUP.md](SETUP.md#bot--plugin-integration-contract)):

- **POST `/draft`** — drafts, stores a pending keyed by a 12-hex `approval_id`
  (TTL 1h), returns `{ok, approval_id, draft, summary}`. Refuses the excluded
  group's topic threads and empty questions.
- **POST `/decide`** — `{approval_id, action: approve|edit|dismiss, edited_text?}`.
  `approve`/`edit` send **as the owner**; all three write a `feedback` row. On a
  successful send the FINAL text is added to the topic+style corpus
  (`corpus.py::add_owner_sample`); on `edit`, `classify_edit` sets the edit kind
  and an intent/both edit writes an `intent_notes` row. Classification / corpus /
  DB failures are swallowed — never fail the request.
- **GET `/health`** — `{ok, session_owner, pending}`.

**Warm-up gate.** On boot, before the HTTP site binds, `_warm_up()` eager-loads
the topic encoder and the style embedder (the same lazy heavy init the draft path
uses). `/draft` and `/decide` gate on a `READY` event, so a request that somehow
arrives cold **waits** instead of running against an unloaded pipeline. This fixes
the "first draft after a restart is slow, the caller times out, and its 0-byte
`200` looks like an empty reply" failure.

**Pending persistence.** Pending drafts are mirrored to disk
(`MIRROR_PENDING_STORE`, default `logs/pending.pkl`) so an undecided card survives
a restart (otherwise Approve would silently no-op). On `/decide` for a pending
that no longer exists, the service returns an unambiguous **HTTP 410** so the
plugin can tell a lost/expired draft from a normal decline and offer a fresh one.

**Send-as-owner** is the only path that touches a chat: it sends via the Telethon
user session and **returns the sent Message** (so its real id can be indexed).
Because a bot DM's API `chat_id` equals the owner's own user id (which would route
to Saved Messages), for positive `chat_id`s it addresses the bot by `bot_username`
instead; groups (negative `chat_id`) send as-is; if a threaded `reply_to` fails
(bot-API vs user-session id mismatch) it retries without threading so the message
still lands inline.

> The card UI itself lives **outside this repo** — a Telegram bot plugin that
> POSTs `/draft` and renders an HTML card (blockquote Goal/Now/Next + the draft),
> then POSTs `/decide` on a button tap. Document/build against the contract in
> [SETUP.md](SETUP.md), not against plugin code. `agent/service.py` is a sibling
> variant that drives its own Mirror-bot DM as the approval UI;
> `agent/bot.py`/`agent/demo.py` are a legacy scaffold and an end-to-end demo.

---

## Design choices

- **Why style embeddings ≠ topic embeddings.** Topic vectors (MiniLM) find
  *what* you talked about; they don't guarantee the exemplar is written *how* you
  write. Authorship is captured by content-independent features (function words,
  punctuation, sentence-length variance) — so a second, style-representation index
  (StyleDistance) lets exemplar selection optimize for voice, and the blend picks
  replies that mirror the incoming register. See [PRD.md](PRD.md).
- **Why edits split style vs intent.** Training one "prefer this" channel on all
  edits is wrong: if you change a *decision* (a number, a commitment), that's not a
  phrasing rule. Classifying the edit routes substance changes to durable decision
  notes and keeps them out of the voice signal.
- **Why the draft is never a training sample.** Only text you actually sent is
  genuine you. A model draft could drift the corpus toward the model's own voice,
  so it is never embedded or retrievable; it survives only as the "before" side of
  a contrastive edit-correction.
- **Why warm-up + pending persistence.** The draft path lazily loads heavy models;
  a cold first draft after a restart outran the caller's timeout. Warm-up moves
  that cost before the socket opens; persisting pendings means a restart mid-card
  doesn't turn Approve into a silent no-op (it returns a clean 410 instead).

---

## Real-samples-only: a draft is never a sample

The exemplar corpus — both the topic and style index — contains **only the
owner's real messages**: ingested history (`direction='out'`) plus
**approved/edited finals**. An approved/edited final is added to `messages` + both
embedding tables **at decide-time** by
[`agent/corpus.py::add_owner_sample`](agent/corpus.py), keyed by the **real
Telegram message id** (so a later ingest of the same id is an idempotent no-op) —
making your just-sent reply a retrievable exemplar immediately. The model's
`original_draft` is **never** inserted into `messages` or either embedding table;
it survives only as the "before" of a contrastive edit-correction.

---

## Safety notes

- **Reading your own history is the sanctioned path.** Telethon logs in as your
  own account with your own API credentials to read messages you already have. Use
  Takeout and pace ingestion (batch/chat sleeps, `FloodWait` handling); keep it
  incremental/resumable via `sync_state`. Keep the corpus private and local.
- **Nothing is sent automatically.** Every outbound reply is gated behind an
  explicit Approve or Edit; the service is loopback-only, shared-secret gated, and
  single-owner.
- **No secrets in git.** `.env`, `.telethon/` (session + credentials), `logs/`,
  and DB dumps are gitignored. This document and `.env.example` contain **names
  only, never values**.
- **Fully local models.** Embeddings and drafting run on-box, so your corpus never
  leaves the machine for a hosted embedding or chat API.
