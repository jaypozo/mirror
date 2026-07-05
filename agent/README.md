# Phase 2: Mirror Agent

Phase 2 now has a structural implementation for draft approval. It does not require the historical corpus to produce a thread summary or draft. If the corpus and pgvector embeddings are available, `agent/draft.py` retrieves owner-style examples; otherwise it falls back to a basic style profile.

Two seams are intentionally still external:

- Message-source integration: another process must decide that a live message needs the owner and provide a `DraftRequest` payload.
- Telegram runtime: `TELEGRAM_BOT_TOKEN` and a target `OWNER_APPROVAL_CHAT_ID` must be configured before the approval bot can run.

## Modules

- `agent/summarize.py`: creates the `Goal / Now / Next / Open` summary from recent live-thread messages.
- `agent/draft.py`: drafts the owner's reply from the incoming message, thread, summary, and optional style examples from pgvector.
- `agent/bot.py`: posts the summary and draft to the owner with `Approve & Send`, `Edit`, and `Dismiss` inline buttons.
- `agent/feedback.py`: persists approve, edit, and dismiss signals to the `feedback` table.
- `agent/llm.py`: pluggable LLM interface with OpenAI and dry-run implementations.
- `agent/style.py`: optional corpus RAG hook and basic style fallback.
- `agent/types.py`: shared request, message, summary, and result dataclasses.

## Payload Shape

The source integration should pass this shape to `DraftRequest.from_dict()` or to the `/draft` command as JSON:

```json
{
  "incoming_message": "Can you review this before I ship it?",
  "thread": [
    {"sender_name": "Naveed", "text": "Can you review this before I ship it?", "direction": "in"}
  ],
  "source_chat_id": 123,
  "source_message_id": 456,
  "target_chat_id": 789,
  "topic_id": null,
  "context_tag": "code-review",
  "metadata": {"source": "live-thread"}
}
```

`target_chat_id` is required for `Approve & Send` and edited sends. Without it, the bot captures the failure and leaves the draft unsent.

## Commands

Run the approval bot:

```bash
make bot
```

Generate a summary from stdin:

```bash
python -m agent.summarize < payload.json
```

Generate a draft from stdin:

```bash
python -m agent.draft < payload.json
```

Post one payload file on bot startup:

```bash
MIRROR_DRAFT_PAYLOAD_FILE=payload.json make bot
```

## Approval Flow

1. Source integration submits a `DraftRequest`.
2. `summarize_thread` creates `Goal / Now / Next / Open`.
3. `draft_reply` retrieves style examples if pgvector is available, then drafts as the owner.
4. `post_approval_request` sends the owner the summary, draft, and inline buttons.
5. `Approve & Send` sends the draft to `target_chat_id` and persists `action=approve`.
6. `Edit` asks the owner to send edited final text, sends that text onward, and persists `action=edit`.
7. `Dismiss` drops the draft and persists `action=dismiss`.

## Guardrails

- Never auto-send without the owner approving or sending edited final text.
- Keep all corpus, draft, and feedback data private to the repo and VPS database.
- Preserve the original inbound thread metadata with feedback.
- Treat edits as high-signal training data.
