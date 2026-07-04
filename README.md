# Mirror

**Mirror is a private, self-hosted "learning model of you."** It ingests your own
Telegram history into a local vector store, and when a new message arrives it
retrieves how *you* replied to similar things in the past and drafts a reply in
your voice — for you to **Approve / Edit / Dismiss**. Nothing is ever sent
automatically, and every decision you make trains it to sound more like you.

Embeddings and drafting run **on your own machine with no third-party API key**:
message embeddings use a local sentence-transformers model, and drafts are
produced by GPT-5.5 through `codex exec` (authenticated via ChatGPT OAuth).

> This repository is intentionally free of personal data and secrets. It
> describes *how* to run Mirror for **any** user; the owner's real ids, tokens,
> database and Telegram session live only in gitignored files (`.env`,
> `.telethon/`), never in the code.

## What it does

```
Telegram history ──► ingest ──► Postgres ──► local embeddings ──► pgvector
                                                                       │
incoming message ──► retrieve your similar past replies ──► draft in your voice
                                                                       │
                              Approve / Edit / Dismiss  ◄──────────────┘
                                        │
                                   feedback ──► learns
```

1. **Ingest** — Telethon reads your own Telegram history (your user account, not
   a bot) into Postgres, incrementally and resumably.
2. **Embed** — a local model turns each message into a 384-dim vector in pgvector.
3. **Retrieve** — for a new message, it finds your most similar *past replies*
   (plus what you were replying to) by vector similarity.
4. **Draft** — those examples become a few-shot prompt; one stateless `codex exec`
   call drafts a reply in your voice.
5. **Approve** — the draft is shown with Approve / Edit / Dismiss. Approve (or an
   edit) sends it as you; every decision is logged as a training signal.

## Requirements

- Python 3.11+ and PostgreSQL with the **pgvector** extension.
- A Telegram **user** account (for ingesting your own history) — API id/hash from
  <https://my.telegram.org>.
- For drafting: the `codex` CLI signed in with ChatGPT OAuth (or set an OpenAI
  API key). Embeddings need no key.

## Setup

```bash
# 1. Python env + deps
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Postgres with pgvector, then apply the schema
make schema                    # runs db/schema.sql against $DATABASE_URL

# 3. Configure. Copy the template and fill in your values (kept out of git).
cp .env.example .env           # then edit .env

# 4. One-time Telegram login (writes the session named by TELEGRAM_SESSION)
.venv/bin/python -m ingest.pull   # prompts for phone / code / 2FA on first run

# 5. Ingest your history (paced + resumable) and embed it locally
.venv/bin/python -m ingest.pull        # full history      (or ingest.pull_recent)
.venv/bin/python -m ingest.embed       # local embeddings

# 6. Try a draft end-to-end
.venv/bin/python -m agent.demo
```

### Running the approval service

Mirror drafts and sends only through the loopback HTTP service, which owns the
single Telegram user session:

```bash
agent/approve_service_run.sh start     # POST /draft, POST /decide, GET /health on 127.0.0.1:8791
```

Your chat bot (the UI that shows the Approve / Edit / Dismiss card) calls
`/draft` when you want a reply drafted, and `/decide` when you tap a button. The
service verifies it is running as your account (`MIRROR_OWNER_USER_ID`), requires
a shared secret on every request (`MIRROR_APPROVE_SECRET`), binds to loopback
only, and never sends anything without an explicit approve/edit.

## Configuration

All settings come from `.env` (see `.env.example` for the full list). Key ones:

- `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` / `TELEGRAM_SESSION` — your user session.
- `DATABASE_URL` — Postgres with pgvector.
- `MIRROR_OWNER_USER_ID` — your Telegram user id (the account the service runs as).
- `MIRROR_APPROVE_SECRET` — shared secret between the service and your bot.
- `EMBEDDING_PROVIDER` / `LLM_PROVIDER` — default to fully local / on-box.

## Documentation

- **[DATA.md](DATA.md)** — the full data-side reference: schema, the ingest →
  embed → retrieve → draft → feedback pipeline, the service, scripts, and env.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the pipeline and design.
- **[PRD.md](PRD.md)** — product intent.

## Safety

- Mirror reads **your own** account's history via Telethon — the sanctioned way
  to access **your own** data. Keep the session file private.
- Secrets and the session never enter git: `.env`, `.telethon/`, `*.session`,
  `logs/`, and database dumps are all gitignored.
- The service is loopback-only, shared-secret gated, single-owner, and sends
  nothing without your explicit Approve or Edit.
