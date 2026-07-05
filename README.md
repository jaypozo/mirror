# Mirror

**Mirror is a self-hosted "drafts replies in your voice, learns from your edits"
system.** It ingests your own Telegram history into a local Postgres + pgvector
store, and when a new message arrives it retrieves how *you* replied to similar
things — matched on both **topic and writing style** — and drafts a reply in
your voice for you to **Approve / Edit / Dismiss**. Nothing is ever sent without
your explicit approval, and every decision you make teaches it to sound more
like you.

Everything runs **on your own machine with no third-party API key**: embeddings
are computed locally with sentence-transformers, and drafts are produced by a
local `codex exec` call (GPT-5.5 over ChatGPT OAuth — swappable for an OpenAI
key). Your corpus never leaves the box.

> **Pointing an AI agent at this repo?** Read this file, then
> [ARCHITECTURE.md](ARCHITECTURE.md) (how it works), [SETUP.md](SETUP.md) (exact
> env + run-your-own), and [DATA.md](DATA.md) (schema + pipeline). Everything is
> abstracted to "the owner"/"you" — the owner's real ids, tokens, database and
> Telegram session live only in gitignored files (`.env`, `.telethon/`), never
> in the code. You can stand up your own instance in an afternoon.

## Features

- **Voice drafting** — every incoming message is answered with a draft written
  in your voice, built from real examples of how you actually reply.
- **Style-based retrieval** — exemplars are chosen by a blend of *topic*
  similarity (MiniLM, 384-dim) **and** *writing-style* similarity (StyleDistance,
  768-dim), plus MMR diversity, so the drafter mirrors your register (terse for
  terse, formal for formal), not just the subject.
- **Living style sheet** — a distilled, always-injected "how the owner writes"
  guide, regenerated periodically from your corpus and refined by your edits,
  with **staged rule promotion** (a correction only becomes a rule after ≥3
  independent edits back it; stale rules decay out).
- **Dual style/intent edit learning** — when you edit a draft, an LLM classifies
  the change as `style | intent | both | trivial`. Style edits train your
  **voice**; intent edits (a decision/fact/number changed) become durable
  **decision notes** and never pollute the voice channel.
- **Thread-aware Goal / Now / Next** — interleaved threads in one conversation
  are segmented per-thread, each with its own `goal` / `current_task` /
  `stage`, so the summary and draft understand *which* thread a message belongs
  to and where in the task you are.
- **Approve / Edit / Dismiss** — drafts surface as a card (with a loading
  placeholder while drafting, and expired-card handling if the service
  restarted). Approve sends as you; Edit sends your version and learns from the
  diff; Dismiss sends nothing. **Nothing is ever sent without your approval.**
- **Real-samples-only corpus** — only your genuine messages are ever
  retrievable: your ingested history plus your approved/edited *finals*. A model
  draft is **never** embedded as if you wrote it.

## Quickstart

Requires **Python 3.11+** and **PostgreSQL with the pgvector extension**.

```bash
# 1. Python env + deps (sentence-transformers pulls torch/transformers)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Configure — copy the template and fill in your values (kept out of git)
cp .env.example .env        # then edit .env (DATABASE_URL, Telegram API id/hash, secrets)

# 3. Apply the schema (needs DATABASE_URL set in .env)
make schema                 # runs db/schema.sql against $DATABASE_URL

# 4. One-time Telegram login (writes the session named by TELEGRAM_SESSION).
#    The first run of the puller performs the interactive phone/code/2FA login.
.venv/bin/python -m ingest.pull

# 5. Ingest your history (paced + resumable) and embed it locally
.venv/bin/python -m ingest.pull          # full history (re-run to catch up)
.venv/bin/python -m ingest.embed         # TOPIC embeddings (384-dim MiniLM)
make embed-style                         # STYLE embeddings (768-dim StyleDistance)

# 6. Prove the draft pipeline end-to-end (no sending)
.venv/bin/python -m agent.demo

# 7. Run the approval service (owns the Telegram session; the ONLY sender)
agent/approve_service_run.sh start       # POST /draft, POST /decide, GET /health on 127.0.0.1:8791
#    ...or install the systemd unit under deploy/systemd/ (see SETUP.md).
```

Then **point your bot/plugin** at `/draft` and `/decide` on
`http://127.0.0.1:8791`, sending the shared secret (`MIRROR_APPROVE_SECRET`) as
the `X-Mirror-Secret` header on every request. The bot renders the
Approve / Edit / Dismiss card; the service does the drafting and is the only
thing that sends as you. The card UI lives **outside this repo** (a Telegram bot
plugin) — the request/response contract is documented in
[SETUP.md](SETUP.md#bot--plugin-integration-contract).

## Requirements

- **Python 3.11+**.
- **PostgreSQL** with the **pgvector** extension (`CREATE EXTENSION vector;`).
- A **Telegram user account** — API id/hash from <https://my.telegram.org> (a
  user session, not a bot, is required to read your own history).
- **For drafting:** the `codex` CLI signed in with ChatGPT OAuth (default,
  no API key), or set `LLM_PROVIDER=openai` with an OpenAI key.
- **Embeddings need no key** — sentence-transformers models download once and
  then run offline (topic MiniLM + style StyleDistance).
- ~a few GB of disk for the models and your corpus.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the full current pipeline with a
  data + control-flow diagram, and the design rationale (why style embeddings ≠
  topic embeddings, why edits split style vs intent, why a draft is never a
  training sample, the warm-up gate + pending persistence).
- **[SETUP.md](SETUP.md)** — run your own: every env var explained, the systemd
  unit, embedding-model/offline notes, and the bot/plugin integration contract.
- **[DATA.md](DATA.md)** — the data-side reference: schema, ingest → embed →
  retrieve → draft → feedback pipeline, the approve service, scripts, and env.
- **[PRD.md](PRD.md)** — product intent and the voice/style-mimicry techniques
  (with citations).

## Safety

- Mirror reads **your own** account's history via Telethon — the sanctioned way
  to access your own data. Keep the session file private.
- Secrets never enter git: `.env`, `.telethon/`, `*.session`, `logs/`, and
  database dumps are all gitignored. Only `.env.example` (names, no values)
  ships.
- The approve service is **loopback-only**, **shared-secret gated**, verifies it
  is running as your account, and sends **nothing** without an explicit
  Approve or Edit.
