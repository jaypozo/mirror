# Mirror — Run Your Own

This is the turnkey setup reference: every environment variable explained, how to
run the approve service (script or systemd), notes on the embedding models, and
the exact bot/plugin integration contract. Start from the
[README Quickstart](README.md#quickstart); this fills in the detail.

Nothing here contains a secret value — the owner's real ids, tokens, database, and
Telegram session live only in gitignored files (`.env`, `.telethon/`).

---

## 1. Prerequisites

- **Python 3.11+**.
- **PostgreSQL** with **pgvector** (`CREATE EXTENSION vector;` — `make schema`
  runs it for you).
- A **Telegram user account** with an **API id/hash** from
  <https://my.telegram.org> → *API development tools*. (A user session, not a bot,
  is required to read your own history.)
- For drafting: the **`codex` CLI** signed in with ChatGPT OAuth (default), or an
  OpenAI API key if you set `LLM_PROVIDER=openai`.
- Outbound internet **once** to download the two sentence-transformers models;
  after that embeddings run offline.

---

## 2. Install & initialize

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # then fill it in (section 3)
make schema                     # applies db/schema.sql to $DATABASE_URL
.venv/bin/python -m ingest.pull # first run = interactive Telethon login, then pull
.venv/bin/python -m ingest.embed   # TOPIC embeddings (384-dim)
make embed-style                   # STYLE embeddings (768-dim)
.venv/bin/python -m agent.demo     # end-to-end proof, sends nothing
```

`make schema` reads `DATABASE_URL` from `.env` (the Makefile `-include .env`s and
exports), so fill `.env` **before** running it.

---

## 3. Environment variables (`.env`)

Copy [`.env.example`](.env.example) and set values. Grouped by concern; only the
first few are strictly required.

### Telegram user session (required)
| var | meaning |
| --- | --- |
| `TELEGRAM_API_ID` | your API id from my.telegram.org |
| `TELEGRAM_API_HASH` | your API hash |
| `TELEGRAM_SESSION` | session file path (default `.telethon/mirror`; gitignored) |

### Database (required)
| var | meaning |
| --- | --- |
| `DATABASE_URL` | `postgresql://USER:PASSWORD@HOST:PORT/DBNAME` — Postgres with pgvector |

### Owner identity (required for the approve service)
| var | meaning |
| --- | --- |
| `MIRROR_OWNER_USER_ID` | your Telegram user id — the account the service verifies it runs as (it refuses if the session owner differs) |
| `MIRROR_EXCLUDED_TOPIC_CHAT_ID` | a group whose forum **topic** threads are never drafted (its General channel is still drafted). Optional. |

### Topic embeddings
| var | default | meaning |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers, no key) \| `openai` \| `dry-run` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim topic model; changing it means changing the `vector(384)` dim + re-embedding |
| `EMBEDDING_API_KEY` | — | only for `openai` |
| `EMBED_BATCH_SIZE` | `100` | messages per embed batch |

### Drafting (LLM)
| var | default | meaning |
| --- | --- | --- |
| `LLM_PROVIDER` | `codex` | `codex` (GPT-5.5 via ChatGPT OAuth, no key) \| `openai` \| `dry-run` |
| `LLM_MODEL` | `gpt-5.5` | model id passed to codex/openai |
| `LLM_API_KEY` / `OPENAI_API_KEY` | — | only for `openai` |
| `CODEX_REASONING_EFFORT` | `high` | codex `model_reasoning_effort` |

### Retrieval — topic + style blend
| var | default | meaning |
| --- | --- | --- |
| `RETRIEVE_TOP_K` | `6` | exemplars returned to the drafter |
| `STYLE_RETRIEVAL_MODE` | `auto` | `auto` (style model, else stylometric-feature fallback) \| `model` \| `features` \| `off` |
| `STYLE_EMBED_MODEL` | `StyleDistance/styledistance` | 768-dim style model; changing it means changing the `vector(768)` dim + re-embedding |
| `STYLE_BLEND_TOPIC_WEIGHT` | `0.6` | topic weight in the blend |
| `STYLE_BLEND_STYLE_WEIGHT` | `0.4` | style weight in the blend |
| `STYLE_MMR_LAMBDA` | `0.7` | MMR relevance↔diversity tradeoff (1 = pure relevance) |
| `STYLE_CANDIDATE_POOL` | `40` | topic-nearest pool re-ranked by the blend |
| `STYLE_EMBED_BATCH_SIZE` | `128` | batch size for `ingest.embed_style` |
| `STYLE_BACKFILL_LIMIT` | `0` | hard cap on one backfill run (0 = none) |
| `STYLE_EXAMPLE_LIMIT`, `STYLE_EMBEDDING_MODEL`, `STYLE_EMBEDDING_API_KEY` | — | legacy `agent/style.py` RAG hook (superseded by `retrieve.py`) |

### Living style sheet
| var | default | meaning |
| --- | --- | --- |
| `STYLE_SHEET_SAMPLE_SIZE` | `400` | owner messages sampled for the stylometric profile |
| `STYLE_SHEET_MAX_EDITS` | `40` | recent owner edits fed into distillation |
| `STYLE_RULE_PROMOTE_THRESHOLD` | `3` | independent edits before a mined rule graduates to active |
| `STYLE_RULE_DECAY_MISSES` | `3` | regenerations with no supporting edit before an active rule decays out |

### Approve service (required to go live)
| var | default | meaning |
| --- | --- | --- |
| `MIRROR_APPROVE_SECRET` | — | **required** shared secret; every request must send it as `X-Mirror-Secret`. Generate: `openssl rand -hex 24`. No secret ⇒ the service refuses to start and rejects all requests. |
| `MIRROR_APPROVE_HOST` | `127.0.0.1` | bind host (keep loopback) |
| `MIRROR_APPROVE_PORT` | `8791` | bind port |
| `MIRROR_PENDING_STORE` | `logs/pending.pkl` | where undecided drafts persist across restarts |
| `MIRROR_RESUME_BACKFILL` | `0` | `1` = resume older-history backfill inside the live client (single session) |
| `MIRROR_LOG_LEVEL` | `INFO` | log level |

### Ingest pacing (optional tuning)
`PULL_BATCH_SIZE`, `PULL_BATCH_SLEEP_SECONDS`, `PULL_CHAT_SLEEP_SECONDS`,
`PULL_SINCE_HOURS`, `PULL_MAX_CHATS`, `PULL_ONE_CHAT`, `PULL_RECENT_LIMIT`,
`PULL_USE_TAKEOUT`, `PULL_TAKEOUT_MAX_DELAY_SECONDS`, `BACKFILL_MAX_CHATS`,
`BACKFILL_ONE_CHAT`.

### Legacy / alternate UIs (optional)
`MIRROR_BOT_TOKEN` (drives `agent/service.py`'s own Mirror-bot approval DM),
`TELEGRAM_BOT_TOKEN` + `OWNER_APPROVAL_CHAT_ID` (the `agent/bot.py` scaffold),
`MIRROR_DRAFT_PAYLOAD_FILE` (post one payload on that bot's startup). Not needed
for the headless `approve_service.py` path.

---

## 4. Embedding models (offline notes)

- **Topic:** `sentence-transformers/all-MiniLM-L6-v2` → 384-dim. Downloaded once
  to the local HuggingFace cache, then CPU-only and offline.
- **Style:** `StyleDistance/styledistance` → 768-dim, content-independent writing
  style. Also local/offline after the first download.
- If the **style model can't load offline**, `STYLE_RETRIEVAL_MODE=auto` falls
  back to a normalized **stylometric feature vector** on the fly (length,
  sentence shape, punctuation, casing, contractions, emoji, function-word rate) —
  retrieval degrades to topic-only rather than failing. Set `off` to disable style
  scoring entirely, or `model` to require the model.
- **Changing a model dimension** means changing the matching `vector(N)` column in
  `db/schema.sql` and re-embedding; the table must be empty to `ALTER` an existing
  vector length.
- The approve service **preloads both encoders at warm-up** so the first draft
  after a restart isn't cold (see ARCHITECTURE.md → warm-up gate).

---

## 5. Running the approve service

The service owns the single Telethon user session and is the only thing that sends
as you.

### Option A — run script

```bash
agent/approve_service_run.sh start     # detached, low-priority, logs to logs/
agent/approve_service_run.sh status
agent/approve_service_run.sh stop
```

`start` first stops the standalone backfill drip and refuses if anything else
still holds the Telethon session (single-writer), and preflights that
`MIRROR_APPROVE_SECRET` is set.

### Option B — systemd

The repo ships a systemd unit + timer for the **periodic style-sheet
regeneration** under [`deploy/systemd/`](deploy/systemd/) (edit the `User=` and
paths to your own first):

```bash
sudo cp deploy/systemd/mirror-style-sheet.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mirror-style-sheet.timer   # daily style-sheet refresh
```

To run the **approve service** under systemd, wrap `agent/approve_service_run.sh
start` (or `.venv/bin/python -m agent.approve_service`) in a `Type=simple` unit
with your `User=`, `WorkingDirectory=`, and `EnvironmentFile=` pointing at `.env`.

### Health check

```bash
curl -s -H "X-Mirror-Secret: $MIRROR_APPROVE_SECRET" http://127.0.0.1:8791/health
# {"ok": true, "session_owner": <your id>, "pending": <n>}
```

---

## 6. Bot / plugin integration contract

The Approve / Edit / Dismiss **card UI lives outside this repo** — it's a Telegram
bot plugin that talks to the service over loopback HTTP. Build against this
contract, not against plugin code. Every request must send the shared secret as a
header:

```
X-Mirror-Secret: <MIRROR_APPROVE_SECRET>
Content-Type: application/json
```

A missing/incorrect header returns `401 {"ok": false, "reason": "unauthorized"}`.

### POST `/draft_gate`

Call this before rendering any loading placeholder or card:

```json
{
  "chat_id": 123456789,
  "question_msg_id": 42,
  "question": "Can you review this before I ship it?",
  "is_topic": false
}
```

Needs-reply response:

```json
{
  "ok": true,
  "needs_reply": true,
  "gate": {"verdict": "NEEDS_REPLY", "stage": "heuristic", "reason": "question mark"}
}
```

Silent skip response:

```http
204 No Content
```

On `204`, render nothing: no placeholder, no card, no warning. Classifier
errors/timeouts/uncertainty are SKIP and also return `204`.

### POST `/draft`

Request:

```json
{
  "chat_id": 123456789,          // chat where the agent messaged the owner
  "question_msg_id": 42,         // the agent's message id (reply target)
  "question": "Can you review this before I ship it?",
  "thread": [                    // optional recent context, oldest-first
    {"text": "...", "sender_name": "Alex", "direction": "in", "message_id": 41}
  ],
  "is_topic": false,             // true if inside an excluded-group topic thread
  "bot_username": "your_bot"     // used to route the send back into the DM
}
```

Draft response:

```json
{
  "ok": true,
  "approval_id": "a1b2c3d4e5f6",
  "draft": "Yeah, send it over and I'll take a look.",
  "summary": {"goal": "...", "now": "...", "next": "...", "open": ["..."]}
}
```

Silent skip response:

```http
204 No Content
```

The service repeats the needs-reply gate as a belt-and-braces guard. It returns
`204` when the gate decides no reply is needed or when a precheck rules the
message out (`no question text`, excluded topic). Treat this as a log-only no-op:
delete any already-rendered placeholder and render no card, warning, or fallback.
Failure returns `{"ok": false, "reason": "..."}` only for messages already
classified `NEEDS_REPLY` whose draft/service path genuinely failed, including
`503` while the service is still warming up. Render a **loading placeholder** only
after `/draft_gate` returns `NEEDS_REPLY`; `/draft` includes a `codex exec`
round-trip (~13s).

### POST `/decide`

```json
{ "approval_id": "a1b2c3d4e5f6", "action": "approve" }
```

- `action: "approve"` — sends the stored `draft` as the owner.
- `action: "edit"` — include `"edited_text": "..."`; sends that instead.
- `action: "dismiss"` — sends nothing.

Response: `{"ok": true, "action": "...", "sent": true|false}`. On approve/edit the
final text is sent as the owner, logged to `feedback`, and added to the corpus;
edits are additionally classified (style/intent) and routed.

**Expired-card handling:** if the `approval_id` no longer exists (the service
restarted and swept it, or it passed the 1h TTL), `/decide` returns **HTTP 410**
with `reason: "expired: ... ask again for a fresh one"`. Treat 410 distinctly from
a normal decline — delete the stale card and offer the owner a fresh draft.

### GET `/health`

`{"ok": true, "session_owner": <owner id>, "pending": <count>}` — no side effects.

### Wiring the bot side

Each bot enables Mirror in its own channel `.env` (outside this repo):

```
MIRROR_APPROVE_ENABLED=1
MIRROR_APPROVE_URL=http://127.0.0.1:8791
MIRROR_APPROVE_SECRET=<same value as the service's .env>
```

The bot POSTs `/draft` when a reply should be drafted, renders the card
(Goal/Now/Next blockquote + the proposed reply), and POSTs `/decide` on a button
tap. The service never sends anything except through an explicit `/decide
approve|edit`.
