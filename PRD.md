# Mirror PRD

## Summary

Mirror is a learning model of the owner. It starts by building a private corpus from the owner's Telegram history and now includes the draft-approval structure that helps the owner reply in their own style without sending anything automatically.

Phase 1 builds the corpus foundation on the owner's VPS:

- Telegram history ingestion through the owner's user account with Telethon.
- Postgres storage for messages and sync cursors.
- pgvector storage for message embeddings tagged by context.
- A skeleton embedding pipeline with a pluggable provider interface.

Phase 2 is structurally implemented around live-thread summaries, draft generation, Telegram approval buttons, and feedback capture. It has explicit seams for the message-source integration and Telegram bot runtime configuration.

## Vision

Mirror should learn how the owner thinks, writes, prioritizes, and responds. The long-term product is a private digital twin that drafts responses for the owner to approve, edit, or dismiss. It should use the owner's own historical messages as ground truth and continue learning from every approval and edit.

When a message arrives that needs the owner's reply, a mirror agent will pull the thread and work context, retrieve the owner's relevant past responses using RAG over a vector store, tag the context as code-review, planning, CS, personal, or another useful category, and draft a reply as the owner. It will post the draft to the owner's Telegram with `Approve & Send`, `Edit`, and `Dismiss` buttons plus a `Goal / Now / Next / Open` context summary, mirroring Naveed's Shade bot UI. Approve sends the reply onward. Edit and Dismiss are captured. Every approval and edit feeds back into the corpus and a running style profile so Mirror improves over time.

## Goals

- Build a reliable, private Telegram corpus pipeline.
- Keep ingestion incremental, rate-limit-safe, and resumable.
- Store enough metadata to support later retrieval, filtering, and context-aware drafting.
- Make setup clear enough that `make pull` works after the owner fills `.env` and completes Telethon login.
- Avoid secrets in git.
- Keep all sending gated behind the owner's approval.

## Non-goals

- No automatic sending of Telegram replies.
- No hosted UI.
- No fine-tuning job.
- No production observability stack.
- No bot-token ingestion path.

## Users

Primary user: the owner.

Operator: the owner or an authorized builder on the owner's VPS.

## Phase Plan

### Phase 1: corpus foundation

Deliverables:

- `PRD.md` and `README.md`.
- Telethon puller at `ingest/pull.py`.
- Postgres + pgvector schema at `db/schema.sql`.
- Embedding pipeline skeleton at `ingest/embed.py`.
- `.env.example`, `requirements.txt`, and `Makefile`.
- Phase 2 stub under `agent/`.

Acceptance criteria:

- The schema applies cleanly to Postgres with pgvector installed.
- `make pull` starts the Telethon user-account login flow when credentials are configured.
- Pulls are incremental using a durable per-chat cursor.
- Re-running the puller does not duplicate messages.
- Interruption is safe: already flushed messages and cursor updates persist.
- `make embed` processes unembedded text messages through a provider interface.
- No secrets or session files are committed.

### Phase 2: mirror agent

Deliverables:

- Message triage for "needs the owner reply".
- Context gatherer for thread, work state, and active goals.
- Retrieval over `message_embeddings` filtered by context tags.
- Draft generation in the owner's voice.
- Telegram approval UI with `Approve & Send`, `Edit`, `Dismiss`.
- `Goal / Now / Next / Open` summary in the draft card.
- Feedback capture from approves, edits, and dismissals.
- Running style profile updated from accepted or edited drafts.

Current scope:

- Implement the summary, draft, approval, and feedback modules.
- Degrade gracefully when the corpus is empty or pgvector retrieval is unavailable.
- Keep message-source integration as a JSON/API seam.
- Require `TELEGRAM_BOT_TOKEN` before running the approval bot.

## Architecture

```text
Telegram user account
        |
        | Telethon, persisted session
        v
ingest/pull.py
        |
        | upsert messages + update sync_state
        v
Postgres
  - messages
  - sync_state
  - message_embeddings
        ^
        | provider interface
        |
ingest/embed.py
```

Phase 2 adds:

```text
Incoming message -> context builder -> RAG retrieval -> draft generator
       -> Telegram approval card -> send/edit/dismiss feedback -> corpus/style profile
```

Implemented Phase 2 modules:

```text
source payload -> agent.summarize -> agent.draft
       -> agent.bot approval card -> agent.feedback
```

## Data Model

### messages

Telegram message records, keyed by `(chat_id, id)` because Telegram message IDs are scoped to each chat.

Columns:

- `id bigint`: Telegram message id within the chat.
- `chat_id bigint`: Telethon dialog id.
- `chat_title text`: current known chat title.
- `sender_id bigint`: Telegram sender id when available.
- `sender_name text`: best-effort display name.
- `text text`: message text or caption.
- `ts timestamptz`: message timestamp.
- `direction text`: `in` or `out`.
- `reply_to_id bigint`: message id replied to, when available.
- `raw jsonb`: selected raw Telethon payload for future backfills.

Indexes:

- `(chat_id, ts)` for chronological thread reconstruction.
- `(direction, ts)` for outbound style retrieval.

### sync_state

Per-chat ingestion cursor.

Columns:

- `chat_id bigint primary key`
- `chat_title text`
- `last_message_id bigint`
- `last_pulled_at timestamptz`

### message_embeddings

Vector rows for retrieval.

Columns:

- `chat_id bigint`
- `message_id bigint`
- `embedding vector`
- `context_tag text`
- `provider text`
- `model text`
- `embedded_at timestamptz`

The foreign key is `(chat_id, message_id)` to `messages(chat_id, id)`.

### feedback

Draft approval feedback used as a learning signal.

Columns:

- `original_draft text`
- `final_text text`
- `action text`: `approve`, `edit`, or `dismiss`
- `ts timestamptz`
- source and target chat metadata
- `summary jsonb`
- `metadata jsonb`

## Context Tagging

Phase 1 uses a heuristic stub:

- `code-review`: review, diff, PR, tests, bug, stack trace, deploy.
- `planning`: roadmap, plan, milestone, goal, next, timeline.
- `cs`: customer, client, support, refund, invoice, onboarding.
- `personal`: family, dinner, travel, home, birthday, weekend.
- `general`: fallback.

The tagger should be replaced later by a classifier or agent pass once enough data is available.

## Security and Privacy

- The puller uses the owner's Telegram user account through Telethon.
- It must never accept or use a Telegram bot token for history ingestion.
- `.env`, Telethon session files, and local database dumps stay off git.
- The corpus remains in the owner's private repo and VPS database.
- Raw message JSON is stored for recoverability, so database access must be treated as sensitive.

## Setup Requirements

Required environment:

- Python 3.11+
- Postgres with pgvector installed.
- `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` from <https://my.telegram.org>.
- A writable Telethon session file path.
- `DATABASE_URL`.
- Embedding provider credentials when embeddings are run.

## Operational Behavior

`ingest/pull.py`:

- Loads `.env`.
- Requires user-account Telegram API credentials.
- Starts Telethon and performs one-time login if no session exists.
- Iterates dialogs.
- Reads `sync_state.last_message_id` per chat.
- Fetches messages with `min_id=last_message_id` in chronological order.
- Upserts messages.
- Advances the cursor after each flushed batch.
- Sleeps gently between batches and dialogs.
- Handles `FloodWait` explicitly while also relying on Telethon's built-in handling.

`ingest/embed.py`:

- Loads `.env`.
- Selects text messages without embeddings.
- Applies a heuristic context tag.
- Calls an embedding provider interface.
- Inserts vectors into `message_embeddings`.

`agent/summarize.py`:

- Accepts recent live-thread messages.
- Produces `Goal / Now / Next / Open`.
- Uses a pluggable LLM client when configured.
- Falls back to a heuristic summary if the LLM is unavailable.

`agent/draft.py`:

- Accepts incoming message, thread, and summary.
- Retrieves owner-style examples from pgvector when available.
- Falls back to a basic style profile when the corpus is empty.
- Uses a pluggable LLM client to draft as the owner.

`agent/bot.py`:

- Runs a Telegram approval bot.
- Posts the context summary, draft, and inline buttons.
- On `Approve & Send`, sends the draft to the target chat.
- On `Edit`, captures the owner's edited text and sends it onward.
- On `Dismiss`, drops the draft.
- Persists every action through `agent.feedback`.

## Risks

- Telegram limits can slow initial backfill. Mitigation: incremental batches and sleeps.
- Message IDs are not globally unique. Mitigation: composite keys by chat.
- Embedding provider dimensions can vary. Mitigation: schema uses pgvector's flexible `vector` type and stores provider/model metadata.
- Private data exposure. Mitigation: local-only default, ignored secrets/session files, clear docs.
- Pending approval state is currently in memory. Mitigation: durable feedback is persisted; production hardening can add a pending approvals table.
- Message-source integration is still external. Mitigation: `DraftRequest` JSON payload shape is documented.

## Open Questions

- Which embedding provider and model should be canonical for the owner's VPS?
- Should Phase 2 retrieve only the owner's outbound messages by default, or blend inbound context with outbound examples?
- What contexts beyond code-review, planning, CS, and personal matter most?
- How long should raw message JSON be retained?
