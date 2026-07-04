# Mirror — Architecture

Mirror is a private "learning model of the owner." It ingests your own Telegram
history into a local vector store, then, when a new message arrives, retrieves
how you replied to similar things and drafts a reply **in your voice** for you
to Approve / Edit / Dismiss. Nothing is ever sent automatically.

Everything runs **on-box with no third-party API key**:

- **Embeddings** are computed locally with a small sentence-transformers model.
- **Drafting** uses GPT-5.5 through `codex exec` (authenticated via ChatGPT
  OAuth), one fresh stateless call per draft.

This document is a clean, secrets-free description of the full pipeline so
anyone can replicate it. No API IDs, hashes, tokens, or credentials appear here.

---

## Pipeline overview

```
                 ┌──────────────────────────────────────────────────────────┐
   Telegram      │  PHASE 1 — CORPUS (live)                                  │
   (your acct)   │                                                          │
        │        │   Telethon (user login, Takeout)                         │
        └───────►│        │                                                 │
                 │        ▼                                                 │
                 │   Postgres  ── messages, sync_state                      │
                 │        │                                                 │
                 │        ▼                                                 │
                 │   LOCAL embeddings (sentence-transformers, 384-dim)      │
                 │        │                                                 │
                 │        ▼                                                 │
                 │   pgvector  ── message_embeddings                        │
                 └────────┼─────────────────────────────────────────────────┘
                          │
                 ┌────────┼─────────────────────────────────────────────────┐
                 │  PHASE 2 — DRAFT-MY-REPLY (this build)                    │
   incoming msg  │        ▼                                                 │
        ────────►│   retrieve.py  ── embed query (same local model)         │
                 │                    → pgvector KNN over YOUR 'out' replies │
                 │        │            + surrounding thread context         │
                 │        ▼                                                 │
                 │   draft.py     ── build system+user prompt (few-shot)    │
                 │        │            → GPT-5.5 via `codex exec` (stateless)│
                 │        ▼                                                 │
                 │   draft text  ── Approve / Edit / Dismiss (human)        │
                 │        │            → feedback table → learn             │
                 └────────┴─────────────────────────────────────────────────┘
```

---

## Data store (Postgres + pgvector)

Full DDL is in [`db/schema.sql`](db/schema.sql). Key tables:

**`messages`** — one row per Telegram message.

| column | meaning |
| --- | --- |
| `chat_id`, `id` | composite primary key |
| `chat_title`, `sender_id`, `sender_name` | who/where |
| `text` | message body |
| `ts` | timestamp |
| `direction` | `'in'` (received) or `'out'` (**your own** messages) |
| `reply_to_id` | threading |
| `raw` | full original payload (jsonb) |

**`sync_state`** — per-chat cursor (`last_message_id`) so ingestion is
incremental and resumable.

**`message_embeddings`** — one embedding per message.

| column | meaning |
| --- | --- |
| `chat_id`, `message_id`, `context_tag` | composite primary key |
| `embedding` | `vector(384)` — matches the local model |
| `context_tag` | coarse bucket: `general`, `code-review`, `planning`, `cs`, `personal` |
| `provider`, `model` | provenance (e.g. `local`, `all-MiniLM-L6-v2`) |

> **Dimension note:** the vector column is `vector(384)` to match
> `sentence-transformers/all-MiniLM-L6-v2`. If you swap embedding models, change
> `384` to the new model's dimension and re-embed. The table must be empty to
> `ALTER` an existing vector length.

**`feedback`** — every Approve / Edit / Dismiss decision, with the original
draft and the final text. This is the training signal for the "learn" step.

---

## Phase 1 — Corpus ingestion

- **Telethon logs in as your own user account** (not a bot) and reads your
  history. This is the sanctioned path for reading *your own* data — see the
  safety notes below.
- Ingestion uses **Telegram Takeout** and paces itself (batch size + sleeps
  between batches and chats) to stay well under rate limits, and is
  **resumable** via `sync_state`.
- Rows land in `messages`; `direction` distinguishes messages you sent
  (`'out'`) from messages you received (`'in'`).

Entry points: `ingest/pull.py` (full history) and `ingest/pull_recent.py`
(incremental catch-up). These are not covered in detail here because the corpus
is already live.

---

## Local embeddings

[`ingest/embed.py`](ingest/embed.py) walks every message with text that isn't
already embedded, encodes it, and upserts into `message_embeddings`.

Providers are pluggable behind `EmbeddingProvider`:

- **`local`** (default) — `LocalEmbeddingProvider`, sentence-transformers,
  `all-MiniLM-L6-v2`, 384-dim, L2-normalized. CPU-only, no network, no key.
  Encoding runs off the event loop via `asyncio.to_thread`.
- `openai` — kept as an option (needs a key); imported lazily so the module
  runs fine without the `openai` package or a key.
- `dry-run` — zero vectors, for wiring tests.

Selected by `EMBEDDING_PROVIDER`. Run it with:

```bash
.venv/bin/python -m ingest.embed
```

Re-running is safe and idempotent — it only embeds messages that don't yet have
a row (and upserts on conflict).

---

## Retrieval

[`agent/retrieve.py`](agent/retrieve.py) — given a query (the incoming message):

1. Embeds the query with **the same local model** used at ingest time.
2. Runs a pgvector nearest-neighbour search (cosine distance, `<=>`, over
   normalized vectors) over `message_embeddings`, **restricted to your own
   replies** (`direction = 'out'`), optionally filtered by `context_tag`.
3. For each hit, also pulls the few messages immediately preceding it, so the
   drafter sees *what you were replying to*, not just the reply in isolation.

Returns the top-K `RetrievedExample`s — real few-shot examples of your voice.
`RETRIEVE_TOP_K` controls K (default 6).

```bash
.venv/bin/python -m agent.retrieve "can you get that done by tomorrow?"
```

---

## Drafting (GPT-5.5 via codex exec)

[`agent/draft.py`](agent/draft.py) + the `CodexLLMClient` in
[`agent/llm.py`](agent/llm.py).

For each incoming message:

1. `retrieve_examples()` fetches top-K of your similar past replies + context.
2. A prompt is built:
   - **system:** *"You are drafting the owner's reply in their voice and style. Here
     are real examples of how they reply… Match their tone, length, directness.
     Output only the reply text."*
   - **user:** the retrieved examples, optional thread context, then the
     incoming message.
3. **One fresh, stateless `codex exec` call** produces the draft. No session is
   maintained between drafts (a deliberate choice — every draft starts clean).

The exact non-interactive invocation (from `CodexLLMClient`):

```bash
codex exec --skip-git-repo-check -s read-only \
  -m gpt-5.5 -c model_reasoning_effort=high \
  -o <tmpfile> "<prompt>"   # stdin closed; final message read from <tmpfile>
```

`codex` authenticates via ChatGPT OAuth, so **no OpenAI API key is required**.
`-o/--output-last-message` captures just the model's final message; stdin is
closed so `codex` doesn't wait on it. On failure or timeout the drafter falls
back to a short heuristic reply rather than crashing.

Providers are selected by `LLM_PROVIDER` (`codex` | `openai` | `dry-run`).

---

## Demo (end-to-end proof)

[`agent/demo.py`](agent/demo.py) picks a real inbound message from the corpus
(or one you pass), loads its thread, retrieves your similar past replies, drafts
a reply as you, and prints the examples + the draft.

```bash
.venv/bin/python -m agent.demo                          # auto-pick recent inbound
.venv/bin/python -m agent.demo <chat_id> <message_id>   # target a specific message
```

Sample output (real corpus message, abridged):

```
TARGET: "Never thrown anything up on ProductHunt, but why not — <link>.
         Go upvote for me so I can see how their analytics dashboard works"
DRAFT (as the owner): "Yeah I'll upvote. Send me what the analytics dashboard looks like"
```

---

## Approve / Edit / Dismiss + learning (LIVE)

> **Now built.** The approval flow ships as the headless HTTP service
> `agent/approve_service.py` (`/draft`, `/decide`, `/health`), driven by the fleet
> Telegram bots' shared plugin. A flagged reply renders an **Approve / Edit /
> Dismiss** card with a `Goal / Now / Next` blockquote (the sending agent supplies
> that context) and a drafted reply; Approve sends it **as the owner** via the
> Telethon session. Full details in **[DATA.md](DATA.md)** §6. The description
> below is the original design note.

The draft core is done and proven. The remaining follow-up is the Telegram
approval **bot/service** that shows each draft with **Approve & Send**, **Edit**,
**Dismiss** buttons plus a `Goal / Now / Next / Open` summary (scaffolded in
`agent/bot.py`, `agent/summarize.py`, `agent/feedback.py`). Every decision is
written to the `feedback` table:

- **Approve** → send the reply, log it as a positive example.
- **Edit** → log original vs. final; the diff is the strongest learning signal.
- **Dismiss** → log the negative.

Over time this feedback tightens the retrieved examples and a running style
profile. That bot runtime is intentionally **not** built yet.

---

## `.env` template (keys only)

Copy `.env.example` to `.env` and fill in your own values. **No secrets belong
in git.**

```
# Telegram (your USER account — Telethon)
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
TELEGRAM_SESSION=.telethon/mirror

# Postgres + pgvector
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME

# Embeddings — local, no key needed
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_API_KEY=
EMBED_BATCH_SIZE=100

# Drafting — GPT-5.5 via codex exec (ChatGPT OAuth, no key needed)
LLM_PROVIDER=codex
LLM_MODEL=gpt-5.5
LLM_API_KEY=
CODEX_REASONING_EFFORT=high

# Retrieval
STYLE_EXAMPLE_LIMIT=5
STYLE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
STYLE_EMBEDDING_API_KEY=
RETRIEVE_TOP_K=6

# Approval bot (follow-up, not built yet)
TELEGRAM_BOT_TOKEN=
OWNER_APPROVAL_CHAT_ID=
```

---

## Setup & run

```bash
# 1. Python env + deps
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # includes sentence-transformers

# 2. Postgres with pgvector, then apply the schema
psql "$DATABASE_URL" -f db/schema.sql

# 3. Fill in .env (see template above) and complete the one-time Telethon login

# 4. Ingest your history (paced, resumable)
.venv/bin/python -m ingest.pull                    # full history
.venv/bin/python -m ingest.pull_recent             # incremental catch-up

# 5. Embed locally
.venv/bin/python -m ingest.embed

# 6. Prove the draft pipeline
.venv/bin/python -m agent.demo
```

Requires: a working local Postgres with the `vector` extension, and the `codex`
CLI authenticated (for drafting).

---

## Safety notes

- **Reading your own history is the sanctioned path.** Telethon logs in as your
  own account with your own API credentials to read messages you already have
  access to. Keep the corpus private and local.
- **Use Telegram Takeout and pace ingestion.** Takeout plus batch/chat sleeps
  keep you comfortably within rate limits and avoid tripping account flags. Keep
  the pull incremental and resumable via `sync_state`.
- **Nothing is sent automatically.** Every outbound reply is gated behind an
  explicit human Approve. Edit and Dismiss are equally first-class.
- **No secrets in git.** `.env` and `.telethon/` (session + credentials) are
  gitignored. This document and `.env.example` contain **keys only, never
  values**.
- **Fully local models.** Embeddings and drafting run on-box with no third-party
  API key, so your corpus never leaves the machine for a hosted embedding or
  chat API.
```
