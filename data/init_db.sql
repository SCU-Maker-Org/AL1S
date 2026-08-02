-- AL1S-Bot 数据库初始化脚本
-- 创建用户对话记录表

-- 用户信息表
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_user_id INTEGER UNIQUE NOT NULL,
    username TEXT,
    first_name TEXT,
    last_name TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 对话记录表
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER NOT NULL DEFAULT 0,
    session_scope TEXT NOT NULL DEFAULT 'private',
    session_owner_id INTEGER NOT NULL DEFAULT 0,
    knowledge_namespace TEXT NOT NULL DEFAULT '',
    chat_type TEXT NOT NULL DEFAULT 'private',
    role_name TEXT DEFAULT 'AI助手',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    UNIQUE(chat_id, thread_id, session_scope, session_owner_id)
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 消息记录表
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER DEFAULT 0,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
);

-- 角色使用统计表
CREATE TABLE IF NOT EXISTS role_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name TEXT NOT NULL,
    usage_count INTEGER DEFAULT 0,
    last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(role_name)
);

-- MCP工具调用记录表
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT, -- JSON格式的参数
    result TEXT,    -- 工具调用结果
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    execution_time REAL, -- 执行时间（秒）
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_conversations_user_chat ON conversations(user_id, chat_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(
    chat_id, thread_id, session_scope, session_owner_id
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation_id ON tool_calls(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_timestamp ON tool_calls(timestamp);

-- 插入初始角色统计数据
INSERT OR IGNORE INTO role_stats (role_name, usage_count) VALUES 
('天童爱丽丝', 0),
('女仆爱丽丝', 0),
('Kei人格', 0),
('游戏玩家', 0),
('AI助手', 0);

-- 创建视图：用户对话统计
CREATE VIEW IF NOT EXISTS user_conversation_stats AS
SELECT 
    u.telegram_user_id,
    u.username,
    COUNT(DISTINCT c.id) as conversation_count,
    COUNT(m.id) as message_count,
    c.role_name as current_role,
    MAX(m.timestamp) as last_activity
FROM users u
LEFT JOIN conversations c ON u.id = c.user_id
LEFT JOIN messages m ON c.id = m.conversation_id
GROUP BY u.id, c.role_name;

-- 创建视图：工具使用统计
CREATE VIEW IF NOT EXISTS tool_usage_stats AS
SELECT 
    tool_name,
    COUNT(*) as usage_count,
    COUNT(CASE WHEN success = 1 THEN 1 END) as success_count,
    COUNT(CASE WHEN success = 0 THEN 1 END) as error_count,
    AVG(execution_time) as avg_execution_time,
    MAX(timestamp) as last_used
FROM tool_calls
GROUP BY tool_name
ORDER BY usage_count DESC;

-- RAG知识库相关表

-- 知识条目表 - 存储从对话中提取的知识
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    conversation_id INTEGER,
    title TEXT NOT NULL, -- 知识条目标题
    content TEXT NOT NULL, -- 知识内容
    summary TEXT, -- 知识摘要
    keywords TEXT, -- 关键词（逗号分隔）
    category TEXT DEFAULT 'general', -- 知识分类
    importance_score REAL DEFAULT 0.0, -- 重要性评分
    embedding_id TEXT, -- 对应的向量嵌入ID
    source_message_id INTEGER, -- 来源消息ID
    knowledge_namespace TEXT NOT NULL DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (conversation_id) REFERENCES conversations (id),
    FOREIGN KEY (source_message_id) REFERENCES messages (id)
);

-- 向量嵌入表 - 存储知识的向量表示
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY, -- UUID作为主键
    knowledge_entry_id INTEGER NOT NULL,
    vector_data BLOB NOT NULL, -- 序列化的向量数据
    dimension INTEGER NOT NULL, -- 向量维度
    model_name TEXT NOT NULL, -- 生成嵌入的模型名称
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (knowledge_entry_id) REFERENCES knowledge_entries (id)
);

-- 知识检索记录表 - 记录RAG检索历史
CREATE TABLE IF NOT EXISTS knowledge_retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    conversation_id INTEGER NOT NULL,
    query TEXT NOT NULL, -- 用户查询
    retrieved_knowledge_ids TEXT, -- 检索到的知识ID列表（JSON格式）
    similarity_scores TEXT, -- 相似度分数（JSON格式）
    used_in_response BOOLEAN DEFAULT FALSE, -- 是否被用于生成回复
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
);

-- 知识更新记录表 - 跟踪知识的修改历史
CREATE TABLE IF NOT EXISTS knowledge_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_entry_id INTEGER NOT NULL,
    update_type TEXT NOT NULL CHECK (
        update_type IN (
            'create',
            'update',
            'merge',
            'delete'
        )
    ),
    old_content TEXT,
    new_content TEXT,
    reason TEXT, -- 更新原因
    updated_by_user_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (knowledge_entry_id) REFERENCES knowledge_entries (id),
    FOREIGN KEY (updated_by_user_id) REFERENCES users (id)
);

-- 创建RAG相关索引
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_user_id ON knowledge_entries (user_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_entries_conversation_id ON knowledge_entries (conversation_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_entries_category ON knowledge_entries (category);

CREATE INDEX IF NOT EXISTS idx_knowledge_entries_importance ON knowledge_entries (importance_score DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_entries_created_at ON knowledge_entries (created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_namespace ON knowledge_entries (knowledge_namespace);

CREATE INDEX IF NOT EXISTS idx_embeddings_knowledge_entry_id ON embeddings (knowledge_entry_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_retrievals_user_id ON knowledge_retrievals (user_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_retrievals_conversation_id ON knowledge_retrievals (conversation_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_retrievals_timestamp ON knowledge_retrievals (timestamp);

CREATE INDEX IF NOT EXISTS idx_knowledge_updates_knowledge_entry_id ON knowledge_updates (knowledge_entry_id);

-- 创建RAG统计视图
CREATE VIEW IF NOT EXISTS rag_stats AS
SELECT
    'knowledge_entries' as table_name,
    COUNT(*) as total_count,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(importance_score) as avg_importance,
    MAX(created_at) as last_created
FROM knowledge_entries
UNION ALL
SELECT
    'embeddings' as table_name,
    COUNT(*) as total_count,
    COUNT(DISTINCT model_name) as unique_models,
    AVG(dimension) as avg_dimension,
    MAX(created_at) as last_created
FROM embeddings
UNION ALL
SELECT
    'knowledge_retrievals' as table_name,
    COUNT(*) as total_count,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(
        CASE
            WHEN used_in_response THEN 1.0
            ELSE 0.0
        END
    ) as usage_rate,
    MAX(timestamp) as last_created
FROM knowledge_retrievals;

INSERT OR IGNORE INTO schema_migrations(version) VALUES (2);
