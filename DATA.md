# Mirror — Data Architecture

Mirror is a private "learning model of one person." It ingests that person's own
Telegram history into a local Postgres + pgvector store, and when a new message
arrives it retrieves how they replied to similar things in the past and drafts a
reply in their voice for a human **Approve / Edit / Dismiss** decision. Nothing
is ever sent automatically. Embeddings and drafting run fully on-box — no
third-party API key is required.

This document covers the **data side**: the schema, the ingest → embed →
retrieve → draft → feedback pipeline, the approve service, the scripts, and the
environment. It is written to be safe to share: no secret values appear here, and
the repo's real secrets live only in gitignored files (see
[Security & sanitization](#security--sanitization)).

> **Identity config:** the owner's Telegram user id and the excluded group id are
> read from env vars — `MIRROR_OWNER_USER_ID` (the account the service verifies it
> is running as) and `MIRROR_EXCLUDED_TOPIC_CHAT_ID` (a group whose topic threads
> are never drafted). Set both in `.env` (gitignored); they are intentionally not
> hardcoded so nothing personal ships in the repo.

---

## 1. Database schema — `db/schema.sql`

Requires pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`

### `messages` — one row per Telegram message
| column | type | meaning |
|---|---|---|
| `id` | bigint | Telegram message id |
| `chat_id` | bigint | chat / dialog id |
| `chat_title` | text | chat display name |
| `sender_id` | bigint | sender's user id |
| `sender_name` | text | resolved display name |
| `text` | text | message body |
| `ts` | timestamptz | timestamp (UTC-normalized) |
| `direction` | text CHECK IN ('in','out') | `out` = the owner's own sent messages; `in` = received |
| `reply_to_id` | bigint | id of the message this replies to |
| `raw` | jsonb | compacted original payload |

PRIMARY KEY `(chat_id, id)`. Indexes: `(chat_id, ts)`, `(direction, ts)`.

### `sync_state` — per-chat forward ingestion cursor
`chat_id` PK, `chat_title`, `last_message_id` (highest id pulled), `last_pulled_at`.
Makes forward ingestion incremental and resumable.

### `message_embeddings` — one embedding per message per context bucket
`chat_id`, `message_id`, `embedding vector(384)`, `context_tag` (default `general`),
`provider`, `model`, `embedded_at`. PRIMARY KEY `(chat_id, message_id, context_tag)`,
FOREIGN KEY → `messages(chat_id, id)` ON DELETE CASCADE. Indexes on `context_tag`
and `embedded_at`. The `384` dimension matches `all-MiniLM-L6-v2`; swapping models
means changing the dim and re-embedding (the table must be empty to ALTER it).

### `feedback` — every Approve / Edit / Dismiss decision (the learning signal)
`id` PK, `original_draft`, `final_text` (null on dismiss), `action` CHECK IN
('approve','edit','dismiss'), `ts`, `source_chat_id`/`source_message_id` (the
message being answered), `target_chat_id`/`target_thread_id` (where the reply
went), `summary` jsonb (the Goal/Now/Next/Open snapshot), `metadata` jsonb,
`edit_kind` CHECK IN ('style','intent','both','trivial') (**DUAL learning** — set
at capture in the `/decide` edit path by a guarded LLM comparing `original_draft`
vs `final_text`; NULL for approve/dismiss and legacy rows), `edit_note` (the
one-line what-changed). Indexes `(action, ts)`, `(edit_kind, ts)`.

> **Learning routes on `edit_kind`.** STYLE/BOTH edits feed the **voice** channel
> (`fetch_recent_edits` and the style-sheet rule miner both filter to
> `edit_kind IN ('style','both')` ∪ NULL, excluding pure `intent`/`trivial`).
> INTENT/BOTH edits feed the **intent** channel (`intent_notes`). A pure decision
> change therefore never trains voice.

### `threads` — per-thread goal/task/stage (thread-aware summaries)
`id` PK, `title`, `goal`, `current_task`, `stage` CHECK IN
('mid-step','awaiting-owner','done'), `anchors` jsonb (distinctive keywords),
`anchor_embedding vector(384)` (for embedding pre-rank when segmenting an incoming
message), `created_at`, `updated_at`. Maintained by `agent/threads.py`: each
incoming message is segmented to the best-matching thread (embedding pre-rank +
one LLM labeler that also refreshes goal/current_task/stage and produces a
thread-aware Goal/Now/Next). Index `(updated_at DESC)`.

### `intent_notes` — durable decisions captured from INTENT/BOTH edits
`id` PK, `thread_id` → `threads(id)` ON DELETE SET NULL, `feedback_id` →
`feedback(id)` ON DELETE SET NULL, `note` (what the owner actually decided),
`source_chat_id`/`source_message_id`, `ts`. Written in the `/decide` edit path
when `edit_kind IN ('intent','both')`. Recent notes for the matched thread are
injected into the draft prompt + the thread state update, so future
goal-summaries and drafts honor real decisions. Index `(thread_id, ts DESC)`.

### `style_sheet` — the living, distilled "how the owner writes" guide
`id` PK, `guide_md` (the active markdown style guide injected into every draft),
`rubric` jsonb (the measured stylometric profile it was scored against),
`sample_size`, `active` bool (the newest `active` row is what the drafter uses),
`model`, `generated_at`. Regenerated periodically by `agent/style_sheet.py`
PRIMARILY from the owner's own corpus (`direction='out'`) and refined by edits.
Index `(active, generated_at DESC)`.

### `style_rules` — candidate correction rules mined from edits (anti-overfit staging)
`id` PK, `rule_key` UNIQUE (normalized dedupe key), `rule_text`, `status` CHECK IN
('candidate','active','decayed'), `support_edit_ids` bigint[] (the distinct
`feedback.id` edits backing the rule), `support_count`, `misses`, `first_seen`,
`last_seen`. A rule graduates to `active` only after ≥ `STYLE_RULE_PROMOTE_THRESHOLD`
(default 3) independent edits support it; one-offs stay `candidate`; promoted
rules that stop recurring accrue `misses` and `decay` out. Index `(status)`.

---

## 2. Ingest pipeline (`ingest/`)

All pullers use a **Telethon USER session** (the owner's own account — bots
cannot read account history). Session path from `TELEGRAM_SESSION`; credentials
from `TELEGRAM_API_ID` / `TELEGRAM_API_HASH`. The `.telethon/*.session` is a
single-writer SQLite file, so **only one puller may run at a time** (all run
scripts enforce this).

Shared mapping (`ingest/pull.py`): `message_row()` → the `messages` tuple;
`compact_raw()` keeps a curated JSON subset (no large binary payloads);
`upsert_messages()` = `INSERT ... ON CONFLICT (chat_id, id) DO UPDATE`
(idempotent); `advance_cursor()` bumps `sync_state` with `GREATEST(...)`;
`flush_batch()` = transactional upsert + cursor per batch. Pacing: batches of
`PULL_BATCH_SIZE` (default 100), sleeps between batches/chats, and `FloodWaitError`
→ sleep-and-resume.

- **`ingest/pull.py`** — forward/full history. Iterates all dialogs,
  `iter_messages(min_id=cursor, reverse=True)` → only messages newer than the
  cursor, oldest-first. Run: `python -m ingest.pull`.
- **`ingest/pull_recent.py`** — windowed catch-up of the last `PULL_SINCE_HOURS`
  (default 24), newest-first. Runs inside a **Telegram Takeout** session (official
  export mode, lower flood limits) by default; handles `TakeoutInitDelayError`.
  Uses `client.connect()` and **refuses if the session isn't already authorized**
  (never logs in).
- **`ingest/backfill.py`** — older history. After a recent-window pull the forward
  cursor points at the newest message, so `pull.py` never reaches older history;
  backfill walks the other way: per dialog it finds `min(id)` in `messages` and
  `iter_messages(offset_id=that)` fetches strictly-older pages until exhausted.
  Resumable with no extra schema — the `messages` table is the cursor.
- **`ingest/embed.py`** — local embeddings. Selects messages with non-empty text
  and no embedding yet (oldest-first, `EMBED_BATCH_SIZE` default 100); assigns a
  coarse `context_tag` via keyword buckets (`code-review`, `planning`, `cs`,
  `personal`, else `general`); embeds via the provider chosen by
  `EMBEDDING_PROVIDER`:
  - **`local`** (default) — `sentence-transformers` `all-MiniLM-L6-v2`, 384-dim,
    normalized, CPU, off-loop. No network / key.
  - **`openai`** — needs `EMBEDDING_API_KEY` (default `text-embedding-3-small`).
  - **`dry-run`** — zero vectors for wiring tests.
  Upserts into `message_embeddings` (idempotent). Run: `python -m ingest.embed`.

---

## 3. Retrieval — `agent/retrieve.py`

At draft time, retrieves the owner's own past replies as few-shot voice examples.
`embed_query()` embeds the incoming message with the **same local model** used at
ingest (cached encoder). `retrieve_examples(query, top_k, context_tag,
context_window=3)` runs a KNN query over `message_embeddings` JOIN `messages`,
**restricted to `direction='out'`** (genuine owner replies only), optional
`context_tag` filter, `ORDER BY embedding <=> $1` (pgvector cosine distance over
normalized vectors), `LIMIT top_k` (`RETRIEVE_TOP_K`, default 6). For each hit it
also pulls the up-to-3 preceding messages so the drafter sees what was being
replied to. No distance threshold — it takes the top-K nearest. Fully local.

> `agent/style.py` is a separate legacy/optional RAG hook doing similar retrieval
> via OpenAI embeddings; it degrades to `[]` without a key. `retrieve.py` is the
> one used by the live pipeline.

---

## 4. Drafting — `agent/draft.py` + `agent/llm.py`

`draft_reply()`:
1. Query = the incoming message (+ a tail of the last 3 thread messages as extra
   semantic anchor).
2. `retrieve_examples()` → top-K owner replies + their context.
3. When summaries are enabled, **thread-aware state** (`agent/threads.py`):
   segment the message to its thread, refresh goal/current_task/stage, and use
   that thread-aware Goal/Now/Next as the summary (this LLM call replaces the flat
   `summarize_thread` call, not adds to it). The matched thread's state + recent
   intent notes are injected into the draft prompt. Falls back to the flat
   `agent/summarize.py` summary if segmentation fails. An agent-supplied summary is
   still honored.
4. Prompt = a **system** prompt ("draft the owner's reply in their voice; match
   tone/length/directness; output ONLY the reply; don't reveal AI; don't invent
   facts; don't copy examples verbatim") + a **user** prompt (the formatted
   retrieved examples, an optional thread transcript, then "Now draft the reply to
   this incoming message: …").
5. One LLM call; on any error, a tiny heuristic fallback.

**LLM backends (`agent/llm.py`)**, selected by `LLM_PROVIDER`:
- **`codex`** (default) — shells out to `codex exec --skip-git-repo-check -s
  read-only -m <LLM_MODEL=gpt-5.5> -c model_reasoning_effort=<CODEX_REASONING_EFFORT=high>`,
  reads the final message from a temp file, 300s timeout. Auth via ChatGPT OAuth
  → **no OpenAI API key**. Fresh/stateless per draft (~13s typical).
- **`openai`** — needs `LLM_API_KEY` / `OPENAI_API_KEY`.
- **`dry-run`** — canned responses.

The **summary** (`agent/summarize.py`) returns compact JSON `{goal, now, next,
open}` with a tolerant parser and a heuristic fallback. Note: in the live fleet
integration the *sending agent* supplies Goal/Now/Next and the card uses those;
the model summary is the fallback.

---

## 5. Small agent modules

- **`agent/eligibility.py`** — pure, unit-testable filter for whether an incoming
  message should be drafted. Excludes: own outgoing, service messages, broadcast
  channels, bot commands, empty text, and the excluded group **topic** threads
  (`EXCLUDED_TOPIC_CHAT_ID` when in a forum topic). Includes DMs, other groups, and
  the the excluded group General channel.
- **`agent/summarize.py`** — the Goal/Now/Next/Open thread summary.
- **`agent/style.py`** — optional/legacy OpenAI-embedding RAG hook; superseded by
  `retrieve.py`.
- **`agent/feedback.py`** — `record_feedback()` inserts one `feedback` row
  (returning its id; accepts `edit_kind`/`edit_note`); no-ops with a warning if
  `DATABASE_URL` is missing; never blocks a send. `fetch_recent_edits()` (voice
  corrections) excludes pure `intent` edits.
- **`agent/edit_classify.py`** — DUAL learning: `classify_edit(original, final)`
  → `(kind, note)` via one guarded LLM call, heuristic fallback; classifies each
  edit as `style|intent|both|trivial` so learning routes to the right channel.
- **`agent/threads.py`** — thread segmentation + per-thread state:
  `resolve_thread(incoming, thread)` segments a message to a thread (embedding
  pre-rank + LLM labeler), updates its goal/task/stage, returns a thread-aware
  summary + recent intent notes; `record_intent_note(...)` persists a decision.
  Fully guarded — returns None on failure so drafting falls back to the flat
  summary.
- **`agent/types.py`** — shared frozen dataclasses: `ChatMessage`,
  `ThreadSummary`, `DraftRequest`, `DraftResult` (now carries `thread_id`).

---

## 6. Approve service — `agent/approve_service.py`

Headless aiohttp service on loopback (`MIRROR_APPROVE_HOST` default `127.0.0.1`,
`MIRROR_APPROVE_PORT` default `8791`). The fleet Telegram bots render the draft
card; this service does the drafting and is the **only** thing that sends as the
owner. It opens the Telethon USER client with `.connect()` only, **refuses if not
already authorized** (never logs in), and verifies the session owner id. Every
request must carry header `X-Mirror-Secret` == `MIRROR_APPROVE_SECRET` (fail
closed). Pending drafts live in memory keyed by a 12-hex `approval_id`, TTL 1h.

Endpoints:
- **POST `/draft`** — `{chat_id, question_msg_id, question, thread[], is_topic,
  bot_username?}` → drafts, stores a pending, returns `{ok, approval_id, draft,
  summary}`. Refuses the excluded group topic threads and empty questions.
- **POST `/decide`** — `{approval_id, action: approve|edit|dismiss, edited_text?}`.
  `approve` sends `draft` as the owner; `edit` sends `edited_text`; `dismiss`
  sends nothing. All three write a `feedback` row. On `edit`, after the send
  succeeds, a guarded `classify_edit` sets `edit_kind`/`edit_note`; an
  `intent`/`both` edit also writes an `intent_notes` row against the draft's
  matched thread (`DraftResult.thread_id`). Classification/DB failures are
  swallowed — never fail the request.
- **GET `/health`** — `{ok, session_owner, pending}`.

**Send-as-owner** (`send_as_owner`, the only path that touches a chat): sends via
the Telethon user session. Because a bot DM's API `chat_id` equals the owner's own
user id (which would route to **Saved Messages**), for positive chat_ids it
addresses the bot by `bot_username` instead; groups (negative chat_id) send as-is;
if a threaded `reply_to` fails (bot-API vs user-session id mismatch) it retries
without threading so the message still lands inline.

> Fleet integration lives in the shared Telegram plugin (`mirror-approve.ts` +
> `server.ts`): the `reply` tool gains `mirror_approve` + `mirror_goal/now/next`;
> a flagged reply POSTs `/draft` and renders an HTML card (blockquote
> Goal/Now/Next + a bold "✍️ &lt;Bot&gt;'s proposed reply (as you):" + the draft).
> Each bot enables it via `MIRROR_APPROVE_ENABLED=1` + URL + the shared secret in
> its channel `.env`, loaded by the bot launcher before startup.

`agent/service.py` is a sibling that instead drives its own Mirror-bot DM as the
approval UI (`MIRROR_BOT_TOKEN`); `agent/bot.py` and `agent/demo.py` are a legacy
scaffold and an end-to-end demo.

---

## 7. Scripts, commands, environment

**`Makefile`**: `schema` (apply `db/schema.sql`), `pull` (`ingest.pull`), `embed`
(`ingest.embed`), `bot` (`agent.bot`), `check` (compileall).

**Run scripts** (all `start|stop|status`, `setsid nice`, pid+log under `logs/`,
and all coordinate the single Telethon session):
- `ingest/backfill_drip.sh` → `ingest.backfill` (overnight drip, `nice -n 15`).
- `agent/service_run.sh` → `agent.service` (bot-driven approve UI).
- `agent/approve_service_run.sh` → `agent.approve_service` (the HTTP service;
  preflights that `MIRROR_APPROVE_SECRET` is set).

**`requirements.txt`**: `asyncpg`, `openai`, `pgvector`, `python-telegram-bot`,
`python-dotenv`, `telethon`, `sentence-transformers`.

**Environment variable names** (values live only in `.env`; see `.env.example`):
- Telegram user session: `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION`
- Database: `DATABASE_URL`
- Embeddings: `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBED_BATCH_SIZE`
- Drafting: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `CODEX_REASONING_EFFORT`, `OPENAI_API_KEY`
- Retrieval: `STYLE_EXAMPLE_LIMIT`, `STYLE_EMBEDDING_MODEL`, `STYLE_EMBEDDING_API_KEY`, `RETRIEVE_TOP_K`
- Approve service: `MIRROR_APPROVE_SECRET`, `MIRROR_APPROVE_HOST`, `MIRROR_APPROVE_PORT`, `MIRROR_RESUME_BACKFILL`, `MIRROR_LOG_LEVEL`, `MIRROR_DRAFT_PAYLOAD_FILE`
- Bot approve UI: `MIRROR_BOT_TOKEN`, `TELEGRAM_BOT_TOKEN`, `OWNER_APPROVAL_CHAT_ID`
- Pull/backfill tuning: `PULL_BATCH_SIZE`, `PULL_BATCH_SLEEP_SECONDS`, `PULL_CHAT_SLEEP_SECONDS`, `PULL_SINCE_HOURS`, `PULL_MAX_CHATS`, `PULL_ONE_CHAT`, `PULL_RECENT_LIMIT`, `PULL_USE_TAKEOUT`, `PULL_TAKEOUT_MAX_DELAY_SECONDS`, `BACKFILL_MAX_CHATS`, `BACKFILL_ONE_CHAT`

---

## End-to-end flow

1. **Ingest** — Telethon user session (`pull` / `pull_recent` / `backfill`) →
   `messages`, cursor in `sync_state`. Paced, FloodWait-safe, resumable.
2. **Embed** — `embed.py` → local 384-dim `all-MiniLM-L6-v2` vectors + `context_tag`
   → `message_embeddings`.
3. **Draft** — incoming → `eligibility` → `retrieve` (cosine KNN, `direction='out'`,
   top-6) → `draft` (system+user prompt) → `codex exec` GPT-5.5 → draft (+ optional
   Goal/Now/Next/Open).
4. **Decide** — `approve_service` `/draft` + `/decide`; on approve/edit the draft is
   sent **as the owner** via the Telethon session; every decision is written to
   `feedback` — the signal that tightens future retrieval and style.

---

## Security & sanitization

Confirmed gitignored and **never committed** (verified against full history):
`.env` (real secrets: API id/hash, DB URL, `MIRROR_APPROVE_SECRET`, tokens),
`.telethon/` and `*.session` / `*.session-journal` (the authenticated user
session = full account access), `logs/` (message content + pids), `.venv/`,
`__pycache__/`, build artifacts, `*.dump`, `*.sql.gz`. Only `.env.example` (names,
no values) belongs in git. The service is loopback-only and shared-secret gated;
nothing is ever sent without an explicit `/decide approve|edit`.
