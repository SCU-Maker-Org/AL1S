CREATE TABLE identities (
    id INTEGER PRIMARY KEY,
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    username TEXT,
    is_bot INTEGER NOT NULL DEFAULT 0 CHECK (is_bot IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (provider, external_id)
);

CREATE TABLE chats (
    id INTEGER PRIMARY KEY,
    transport TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (transport, bot_id, external_id)
);

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    transport TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    chat_external_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    owner_key TEXT NOT NULL,
    knowledge_scope TEXT NOT NULL,
    role_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    run_id TEXT,
    model TEXT,
    status TEXT NOT NULL DEFAULT 'complete',
    created_at TEXT NOT NULL,
    source_event_key TEXT,
    UNIQUE (conversation_id, source_event_key, role)
);
CREATE INDEX messages_conversation_time_idx
    ON messages(conversation_id, created_at DESC, id DESC);

CREATE TABLE event_receipts (
    transport TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    external_event_id TEXT NOT NULL,
    received_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing',
    processing_token TEXT,
    response_json TEXT,
    delivered_at TEXT,
    delivery_attempts INTEGER NOT NULL DEFAULT 0 CHECK (delivery_attempts >= 0),
    next_attempt_at TEXT,
    last_delivery_error TEXT,
    PRIMARY KEY (transport, bot_id, external_event_id)
);
CREATE INDEX event_receipts_delivery_idx
    ON event_receipts(transport, bot_id, status, next_attempt_at, updated_at);

CREATE TABLE roles (
    owner_key TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    greeting TEXT NOT NULL,
    farewell TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner_key, name)
);

CREATE TABLE profiles (
    identity_id INTEGER PRIMARY KEY REFERENCES identities(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE command_effects (
    event_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE group_settings (
    chat_id INTEGER PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    session_scope TEXT NOT NULL DEFAULT 'topic',
    memory_enabled INTEGER NOT NULL DEFAULT 0 CHECK (memory_enabled IN (0, 1)),
    wake_words_json TEXT NOT NULL DEFAULT '[]',
    consent_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX memories_scope_time_idx ON memories(scope, updated_at DESC);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    memory_id UNINDEXED,
    scope UNINDEXED,
    content,
    tokenize='unicode61'
);

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    source_root TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    source_path TEXT NOT NULL,
    title TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    revision TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (id, scope),
    UNIQUE (scope, source_root, source_path)
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    vector_generation INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id, scope) REFERENCES documents(id, scope) ON DELETE CASCADE,
    UNIQUE (document_id, ordinal)
);
CREATE INDEX chunks_scope_idx ON chunks(scope, document_id, ordinal);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    scope UNINDEXED,
    content,
    tokenize='unicode61'
);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    lease_until TEXT,
    lease_token TEXT,
    error TEXT,
    source_event_key TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX jobs_ready_idx ON jobs(status, available_at, created_at);

CREATE TABLE tool_approvals (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    actor_key TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    expires_at TEXT NOT NULL,
    resolved_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE usage_events (
    id INTEGER PRIMARY KEY,
    identity_id INTEGER REFERENCES identities(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    run_id TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    source_event_key TEXT UNIQUE
);

CREATE TABLE app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
