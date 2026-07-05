CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id bigint NOT NULL,
    chat_id bigint NOT NULL,
    chat_title text,
    sender_id bigint,
    sender_name text,
    text text,
    ts timestamptz NOT NULL,
    direction text NOT NULL CHECK (direction IN ('in', 'out')),
    reply_to_id bigint,
    raw jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (chat_id, id)
);

CREATE INDEX IF NOT EXISTS messages_chat_ts_idx
    ON messages (chat_id, ts);

CREATE INDEX IF NOT EXISTS messages_direction_ts_idx
    ON messages (direction, ts);

CREATE TABLE IF NOT EXISTS sync_state (
    chat_id bigint PRIMARY KEY,
    chat_title text,
    last_message_id bigint NOT NULL DEFAULT 0,
    last_pulled_at timestamptz NOT NULL DEFAULT now()
);

-- Embedding dimension is 384 to match the local sentence-transformers model
-- `sentence-transformers/all-MiniLM-L6-v2` (EMBEDDING_PROVIDER=local).
-- If you switch embedding models, change 384 to that model's dimension and
-- re-embed. The table must be empty to ALTER an existing vector length.
CREATE TABLE IF NOT EXISTS message_embeddings (
    chat_id bigint NOT NULL,
    message_id bigint NOT NULL,
    embedding vector(384) NOT NULL,
    context_tag text NOT NULL DEFAULT 'general',
    provider text,
    model text,
    embedded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chat_id, message_id, context_tag),
    FOREIGN KEY (chat_id, message_id)
        REFERENCES messages (chat_id, id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS message_embeddings_context_tag_idx
    ON message_embeddings (context_tag);

CREATE INDEX IF NOT EXISTS message_embeddings_embedded_at_idx
    ON message_embeddings (embedded_at);

CREATE TABLE IF NOT EXISTS feedback (
    id bigserial PRIMARY KEY,
    original_draft text NOT NULL,
    final_text text,
    action text NOT NULL CHECK (action IN ('approve', 'edit', 'dismiss')),
    ts timestamptz NOT NULL DEFAULT now(),
    source_chat_id bigint,
    source_message_id bigint,
    target_chat_id bigint,
    topic_id bigint,
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS feedback_action_ts_idx
    ON feedback (action, ts);

-- Edit classification (DUAL learning). An owner edit can change STYLE (phrasing/
-- voice, same meaning), INTENT/SUBSTANCE (a fact/decision/goal changed), BOTH, or
-- be TRIVIAL. `edit_kind` is set at capture in the /decide edit path (a guarded
-- LLM compares original_draft vs final_text); `edit_note` is a one-line
-- what-changed. NULL for approve/dismiss and for legacy/unclassified rows.
--   * STYLE / BOTH  -> train the voice channel (style sheet + edit few-shots).
--   * INTENT / BOTH -> train the intent channel (intent_notes below); a pure
--     INTENT edit must NEVER train voice (a decision change is not a voice signal).
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS edit_kind text
    CHECK (edit_kind IS NULL OR edit_kind IN ('style', 'intent', 'both', 'trivial'));
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS edit_note text;

CREATE INDEX IF NOT EXISTS feedback_edit_kind_ts_idx
    ON feedback (edit_kind, ts);

-- Thread-aware state. The owner interleaves MULTIPLE threads in one
-- conversation. Each incoming message is segmented to the best-matching active
-- thread (embedding pre-rank over `anchor_embedding` + a cheap LLM labeler in
-- agent/threads.py), or a new thread is opened. Per-thread `goal`,
-- `current_task`, and `stage` are maintained (LLM update step) so the Goal/Now/
-- Next summary and the draft reflect the matched thread's objective and where in
-- the task we are — not a flat window of recent messages.
CREATE TABLE IF NOT EXISTS threads (
    id bigserial PRIMARY KEY,
    title text NOT NULL,
    goal text,
    current_task text,
    stage text NOT NULL DEFAULT 'mid-step'
        CHECK (stage IN ('mid-step', 'awaiting-owner', 'done')),
    anchors jsonb NOT NULL DEFAULT '{}'::jsonb,
    anchor_embedding vector(384),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS threads_updated_at_idx
    ON threads (updated_at DESC);

-- Intent/decision learning channel. Captured from INTENT/BOTH edits (Build 1):
-- what the owner actually decided or intended, tied to the feedback row that
-- produced it and to the thread it belongs to (Build 2). Recent notes for the
-- matched thread are injected into the draft prompt + the thread state update,
-- so future goal-summaries and drafts reflect real decisions rather than a stale
-- draft's guess.
CREATE TABLE IF NOT EXISTS intent_notes (
    id bigserial PRIMARY KEY,
    thread_id bigint REFERENCES threads (id) ON DELETE SET NULL,
    feedback_id bigint REFERENCES feedback (id) ON DELETE SET NULL,
    note text NOT NULL,
    source_chat_id bigint,
    source_message_id bigint,
    ts timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS intent_notes_thread_ts_idx
    ON intent_notes (thread_id, ts DESC);

-- Living style sheet: a distilled "how the owner writes" guide, regenerated
-- periodically (agent/style_sheet.py) PRIMARILY from the owner's own message
-- corpus (direction='out') and refined by the edit-feedback loop. The newest
-- row with active=true is what the drafter injects into every draft. `rubric`
-- stores the measured stylometric profile the guide was scored against.
CREATE TABLE IF NOT EXISTS style_sheet (
    id bigserial PRIMARY KEY,
    guide_md text NOT NULL,
    rubric jsonb NOT NULL DEFAULT '{}'::jsonb,
    sample_size integer NOT NULL DEFAULT 0,
    active boolean NOT NULL DEFAULT true,
    model text,
    generated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS style_sheet_active_idx
    ON style_sheet (active, generated_at DESC);

-- Candidate style rules mined from the owner's edits. Anti-overfit staging: a
-- rule only graduates (status='active') once it is supported by
-- >= STYLE_RULE_PROMOTE_THRESHOLD (default 3) INDEPENDENT edits — support is the
-- distinct set of feedback.id values in `support_edit_ids`. One-offs stay
-- 'candidate'; promoted rules that stop recurring across regenerations accrue
-- `misses` and decay back out (status='decayed') after STYLE_RULE_DECAY_MISSES.
CREATE TABLE IF NOT EXISTS style_rules (
    id bigserial PRIMARY KEY,
    rule_key text UNIQUE NOT NULL,
    rule_text text NOT NULL,
    status text NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'active', 'decayed')),
    support_edit_ids bigint[] NOT NULL DEFAULT '{}',
    support_count integer NOT NULL DEFAULT 0,
    misses integer NOT NULL DEFAULT 0,
    first_seen timestamptz NOT NULL DEFAULT now(),
    last_seen timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS style_rules_status_idx
    ON style_rules (status);
