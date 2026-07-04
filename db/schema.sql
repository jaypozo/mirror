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
    target_thread_id bigint,
    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS feedback_action_ts_idx
    ON feedback (action, ts);
