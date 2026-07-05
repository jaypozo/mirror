# `agent/` — the draft-and-decide layer

The modules that turn an incoming message into a voice draft and handle the
Approve / Edit / Dismiss decision. See the repo root
[ARCHITECTURE.md](../ARCHITECTURE.md) for the full pipeline and
[SETUP.md](../SETUP.md) for the integration contract; this file is a quick
module map for developers reading the code.

Two seams are intentionally external:

- **Message-source integration** — some other process decides a live message
  needs the owner and calls the service's `/draft` endpoint (or builds a
  `DraftRequest`).
- **Card UI** — the Approve / Edit / Dismiss card is rendered by a Telegram bot
  plugin that lives outside this repo; `approve_service.py` only exposes the
  loopback HTTP API it calls.

## Modules

| module | role |
| --- | --- |
| `approve_service.py` | the live headless HTTP service: `/draft`, `/decide`, `/health`; owns the Telethon user session; the only sender-as-owner. Warm-up gate + pending persistence + 410 on expired. |
| `retrieve.py` | topic + style + MMR exemplar retrieval (`direction='out'` only). |
| `style_embed.py` | the content-independent STYLE embedder (StyleDistance 768-dim, stylometric-feature fallback). |
| `corpus.py` | the one place an owner message enters the retrievable store (`add_owner_sample`, real finals only — never a draft). |
| `style_sheet.py` | the living style guide + staged rule promotion (`style_sheet` / `style_rules`). |
| `edit_classify.py` | dual learning — classify an edit `style | intent | both | trivial`. |
| `threads.py` | thread segmentation + per-thread goal/current_task/stage + intent notes. |
| `draft.py` | builds the system+user prompt (style sheet + exemplars + edits + thread state) and calls the LLM. |
| `summarize.py` | the flat Goal / Now / Next / Open summary (fallback when thread segmentation is off/failed). |
| `feedback.py` | `record_feedback()` / `fetch_recent_edits()` — writes the learning signal; voice fetch excludes pure-intent edits. |
| `eligibility.py` | pure filter for whether an incoming message should be drafted. |
| `llm.py` | pluggable LLM client (`codex` default, `openai`, `dry-run`). |
| `types.py` | shared dataclasses: `ChatMessage`, `Brief`, `DraftRequest`, `DraftResult`. |
| `service.py` | alternate UI: drives its own Mirror-bot DM as the approval card (`MIRROR_BOT_TOKEN`). |
| `bot.py` / `demo.py` | legacy python-telegram-bot scaffold, and the end-to-end demo. |

## `DraftRequest` payload shape

`DraftRequest.from_dict()` accepts:

```json
{
  "incoming_message": "Can you review this before I ship it?",
  "thread": [
    {"sender_name": "Alex", "text": "Can you review this before I ship it?", "direction": "in"}
  ],
  "source_chat_id": 123,
  "source_message_id": 456,
  "target_chat_id": 789,
  "topic_id": null,
  "context_tag": "code-review",
  "metadata": {"source": "live-thread"}
}
```

`target_chat_id` is required to actually send on approve/edit. (The live
`approve_service.py` builds this itself from the `/draft` body — see
[SETUP.md](../SETUP.md#bot--plugin-integration-contract).)

## CLI entry points

```bash
python -m agent.demo               # end-to-end proof (auto-pick or <chat_id> <message_id>)
python -m agent.draft   < payload.json     # draft from a payload on stdin
python -m agent.summarize < payload.json   # Goal/Now/Next/Open from a payload
python -m agent.style_sheet        # regenerate the living style sheet
```

## Guardrails

- Never auto-send without an explicit Approve or an edited final.
- Keep corpus, draft, and feedback data private and local.
- Treat edits as high-signal training data — but route intent changes to the
  intent channel, never the voice channel.
- A model draft is never a positive voice sample or a corpus row.
