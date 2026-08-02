# AL1S-Bot 🤖

基于《蔚蓝档案》天童爱丽丝角色的智能 Telegram 机器人，集成 AI 对话、知识学习、工具调用等功能。

## ✨ 核心特性

- **🎭 角色扮演**: 天童爱丽丝等多种预设角色，支持角色切换
- **🧠 智能学习**: 自动从对话中学习并记忆用户信息
- **🔧 工具集成**: 支持 MCP 协议，可调用文件系统、GitHub、搜索等工具
- **🔍 图片搜索**: 基于 Ascii2D 的图片反向搜索功能
- **💾 持久存储**: SQLite 数据库存储对话历史和知识库
- **🌐 多模型支持**: 兼容 OpenAI、月之暗面、DeepSeek 等 API
- **👥 群聊与 Topic**: 支持提及、回复、唤醒词、白名单、会话隔离和群上下文旁听

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd AL1S-Bot

# 安装依赖（推荐使用 uv）
uv sync --frozen
```

锁文件中的常规 Python 包来自官方 PyPI；Linux 的 PyTorch CPU wheel 来自 PyTorch 官方仓库。项目不配置第三方镜像。

### 2. 配置设置

```bash
# 复制配置模板
cp config.example.toml config.toml

# 编辑配置文件，填入必要信息
nano config.toml
```

**必需配置**：
```toml
[openai]
api_key = "your-api-key-here"
base_url = "https://api.openai.com/v1"  # 或其他兼容API

[telegram]
bot_token = "your-telegram-bot-token"
```

### 3. 初始化数据库

```bash
# 自动创建数据库（首次运行时）
mkdir -p data
sqlite3 data/bot.db < data/init_db.sql
```

### 4. 启动机器人

```bash
# 使用 uv 运行
uv run python main.py

# 或直接运行
python main.py
```

## 📁 项目架构

```
AL1S-Bot/
├── src/
│   ├── agents/              # Agent 实现
│   │   ├── unified_agent_service.py    # 统一 Agent
│   │   └── langchain_agent_service.py  # LangChain Agent
│   ├── infra/               # 基础设施层
│   │   ├── database.py      # 数据库服务
│   │   ├── vector.py        # 向量存储服务
│   │   └── mcp.py          # MCP 工具集成
│   ├── services/            # 业务服务层
│   │   ├── conversation_service.py     # 对话管理
│   │   ├── learning_service.py         # 知识学习
│   │   └── ascii2d_service.py          # 图片搜索
│   ├── handlers/            # 消息处理器
│   │   ├── chat_handler.py             # 聊天处理
│   │   ├── command_handler.py          # 命令处理
│   │   └── image_handler.py            # 图片处理
│   ├── models.py            # 数据模型
│   ├── config.py            # 配置管理
│   └── bot.py              # 机器人主类
├── data/                    # 数据目录
├── config.example.toml      # 配置模板
└── main.py                 # 程序入口
```

## 🎮 使用指南

### 基础命令

- `/start` - 开始使用
- `/help` - 显示帮助
- `/ping` - 测试连接

### 角色管理

- `/role` - 查看当前角色
- `/role <角色名>` - 切换角色
- `/roles` - 显示所有角色

### 对话管理

- `/reset` - 重置对话
- `/stats` - 对话统计

### 知识管理

- `/knowledge search <关键词>` - 搜索知识库
- `/knowledge stats` - 知识库统计
- `/rebuild_knowledge` - 重建知识索引

## ⚙️ 配置详解

### Agent 配置

```toml
[agent]
# Agent 类型选择（这是唯一的控制开关）
type = "unified"  # 或 "langchain"

# 向量存储配置
vector_store = "faiss"
vector_store_path = "data/vector_store"

# 嵌入模型选择
embedding_model = "Qwen/Qwen3-Embedding-0.6B"
embedding_revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
embedding_device = "cpu"  # cpu | mps | cuda | auto
embedding_batch_size = 8

# 单次对话允许的工具调用上限
max_tool_rounds = 4
max_tool_calls = 12

# 自动学习
auto_learning = true
learning_threshold = 0.8
```

**重要说明**：
- `agent.type` 是唯一的 Agent 控制开关
- 当设置为 `"langchain"` 时，自动启用 LangChain 相关功能
- 当设置为 `"unified"` 时，使用统一 Agent，自动禁用 LangChain
- 无需手动设置 `langchain.enabled`，系统会自动处理
- 默认嵌入模型为 [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，支持中英文等多语言检索；`embedding_revision` 固定模型提交，查询会自动使用模型提供的 query prompt，向量会归一化后写入索引
- `cpu` 是兼容性最好的设备；Apple Silicon 可尝试 `mps`，遇到算子兼容或内存问题时切回 `cpu`
- 资源有限或不希望下载模型时可设为 `tfidf`，但语义检索效果会明显降低

#### 首次下载与索引升级

首次使用 Qwen3 时，Sentence Transformers 会从 Hugging Face 下载约 1.2 GB 的模型文件，启动时间取决于网络和磁盘速度。模型会写入本机 Hugging Face 缓存，之后启动无需重复下载。

项目锁定 PyTorch `>=2.13,<2.14`。旧版 2.8 在 Apple Silicon CPU 上加载 Qwen3 时可能触发原生 OpenMP 崩溃；不要单独把 PyTorch 降回旧版。可用 `uv sync --frozen` 恢复锁文件中的兼容版本。Linux 默认从 PyTorch 官方 CPU wheel 仓库安装，避免 Docker 镜像额外拉取整套 CUDA 运行时；需要 Linux CUDA 时应按目标 CUDA 版本调整 `tool.uv.sources` 后重新锁定。

向量目录包含索引清单，记录模型 ID、模型提交、维度、归一化方式、query prompt 内容和索引格式版本。升级前的 MiniLM 索引没有该清单，或者清单与当前模型不一致时，程序会保留 SQLite 中的知识数据，并自动使用新模型重建 FAISS 索引；不要为此删除 `data/bot.db`。重建期间知识检索尚不可用，完成后会恢复。如果日志显示重建失败，先检查模型下载、可用磁盘空间和 `data/vector_store` 的写权限，再执行 `/rebuild_knowledge`。

### MCP 工具配置

MCP 服务器由机器人按需启动。建议先启用无副作用、无密钥的服务器，再逐个开放网络、仓库或文件访问能力。下面的 [Time](https://github.com/modelcontextprotocol/servers/tree/main/src/time) 与 [Context7](https://github.com/upstash/context7) 可以作为常用基础组合：

```toml
[mcp]
enabled = true

# 当前时间与时区转换
[[mcp.servers]]
name = "time"
command = "uvx"
# Time server 尚未适配 MCP 2.x，暂时把其运行时约束到 MCP 1.x
args = ["--with", "mcp<2", "mcp-server-time", "--local-timezone", "Asia/Shanghai"]
enabled = true
access = "public"
read_only = true
connect_timeout = 90
tool_timeout = 15
max_result_chars = 5000

# 查询最新的开源库文档；无需密钥也可使用，但限额较低
[[mcp.servers]]
name = "context7"
command = "npx"
args = ["-y", "@upstash/context7-mcp@3.2.5"]
enabled = true
access = "private"
read_only = true
connect_timeout = 120
tool_timeout = 45
tool_prefix = "ctx7_"
env = { PATH = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin" }

# 文件系统工具：只开放明确目录，并要求服务器标注只读工具
[[mcp.servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/absolute/path/to/allowed/directory"]
read_only = true
access = "admin"
enabled = false
env = { PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" }
```

每个 `[[mcp.servers]]` 支持以下约束：

| 配置项 | 作用 |
| --- | --- |
| `connect_timeout` | 冷启动、下载依赖和建立连接的最长秒数 |
| `tool_timeout` | 单次工具调用的最长秒数 |
| `include_tools` | 仅暴露匹配 glob 的工具，例如 `["git_status", "git_diff*"]` |
| `exclude_tools` | 排除匹配 glob 的工具，优先级高于 `include_tools` |
| `tool_prefix` | 给该服务器的工具名加前缀，减少重名；仍冲突时程序会生成稳定别名 |
| `read_only` | 只保留 MCP 元数据中明确标注 `readOnlyHint = true` 的工具；未标注的工具也会被过滤 |
| `access` | `public` 允许群聊，`private` 仅允许私聊与管理员，`admin` 仅允许 `telegram.admin_user_ids` |
| `max_result_chars` | 截断过长的工具结果，避免一次调用耗尽模型上下文 |

环境变量可写成完整的 `${ENV_NAME}` 引用，避免把密钥写进 `config.toml`：

```toml
env = { TAVILY_API_KEY = "${TAVILY_API_KEY}" }
```

启用的服务器缺少该环境变量时，配置校验会直接报错。服务器名称必须唯一，写错字段也会在启动时被拒绝。未填写 `access` 时按 `admin` 处理；需要先在 `[telegram].admin_user_ids` 中填写自己的 Telegram 用户 ID，管理员工具才会出现在该用户的 `/tools` 和 Agent 工具列表里。权限在工具展示和实际调用两层校验，不能通过猜测工具名绕过。

#### 高权限服务器

[Fetch](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch) 可访问任意 URL，也可能访问本机或内网地址；[Git MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/git) 同时提供读取和修改仓库的工具。这两类服务器应保持默认关闭，并同时使用管理员权限与明确的工具白名单：

```toml
# 任意 URL 抓取：确认部署网络不存在 SSRF 风险后再启用
[[mcp.servers]]
name = "fetch"
command = "uvx"
args = ["--with", "mcp<2", "mcp-server-fetch"]
enabled = false
access = "admin"
include_tools = ["fetch"]

# 仓库工具：仅暴露查询类命令
[[mcp.servers]]
name = "git"
command = "uvx"
args = ["--with", "mcp<2", "mcp-server-git", "--repository", "/absolute/path/to/repository"]
enabled = false
access = "admin"
include_tools = ["git_status", "git_diff*", "git_log", "git_show", "git_branch"]
tool_prefix = "repo_"
```

不要把 Git 写工具、生产 SQLite、Home 目录或工作区根目录直接暴露给公共群聊。`read_only = true` 依赖服务器提供正确的 MCP 注解；对第三方服务器仍建议同时使用 `include_tools`。Context7 3.2.5 要求 Node.js `>=20.18.1`；如果 `npx` 启动到了错误的 Node.js，可用 `command -v node`、`command -v npx` 检查 PATH，并在服务器的 `env.PATH` 中显式把目标 Node.js 目录放在最前面。

### 角色自定义

```toml
[[roles]]
name = "自定义角色"
english_name = "Custom Role"
description = "角色描述"
personality = """
角色的详细设定和性格描述...
"""
greeting = "角色问候语"
farewell = "角色告别语"
```

### Telegram 群聊

先通过 BotFather 创建机器人并将其加入群组。Telegram 开启隐私模式时，机器人通常只能收到命令、对机器人的回复和明确提及；使用 BotFather 的 `/setprivacy` 关闭隐私模式，或将机器人设为群管理员后，Telegram 可能投递更多普通群消息。无论 Telegram 投递哪些消息，AL1S 都会在程序层再次执行触发过滤，未触发消息不会调用 Agent。

```toml
[telegram]
admin_user_ids = [123456789]

[telegram.group]
enabled = true
require_mention = true
allow_reply_trigger = true
observe_unmentioned_messages = true
ignore_bot_messages = true
session_scope = "topic" # per_user | shared | topic
allowed_chat_ids = []   # 空数组表示不限制
blocked_chat_ids = []
allowed_thread_ids = []
ignored_thread_ids = []
wake_words = ["爱丽丝", "AL1S"]
context_buffer_size = 30
context_buffer_ttl = 1800

[telegram.group.memory]
enable_long_term_learning = false
allow_admin_toggle = true
namespace_scope = "topic" # group | topic

[telegram.rate_limit]
enabled = true
per_user_requests = 10
per_user_window_seconds = 60
per_chat_requests = 30
per_chat_window_seconds = 60
```

群内默认响应条件：`@bot_username`、回复机器人消息、命中唤醒词或使用目标为当前机器人的命令。`observe_unmentioned_messages = true` 只会把普通群消息放入按群和 Topic 隔离的内存缓冲，受数量和 TTL 限制；旁听内容不会直接调用模型、写入 SQLite 或进入个人长期知识。

会话作用域：

- `per_user`：同一群和 Topic 内按成员隔离。
- `shared`：整个群共享会话，Topic 不隔离。
- `topic`：同一 Forum Topic 共享会话，普通群使用 Topic ID `0`。

管理员命令：

- `/group_status`：查看群 ID、Topic ID、作用域、旁听和长期学习状态。
- `/group_enable`、`/group_disable`：运行期启停当前群。
- `/group_scope per_user|shared|topic`：修改当前群会话作用域。
- `/group_memory on|off`：切换群知识学习；需允许管理员切换。
- `/group_wake_words 词1 词2`：修改当前群唤醒词；不带参数时查看当前值。

这些修改命令仅允许群创建者、群管理员或 `telegram.admin_user_ids` 中的全局管理员执行。运行期修改在机器人重启后恢复 TOML 配置；需要持久配置时请同步修改 `config.toml`。

获取 `chat_id` 和 `message_thread_id`：先在目标群/Topic 发送消息，然后查看 `/group_status`。如果机器人尚不能响应，可临时查看结构化日志中的 `chat_id`、`thread_id`，或使用 Telegram Bot API 的 `getUpdates`。群 ID 通常是负数，必须按整数原样写入白名单。

Forum Topic 的所有文本、图片、占位、错误和拆分回复都会携带原始 `message_thread_id`。`allowed_thread_ids` 是全局 Topic ID 列表；如果不同群存在相同 Topic ID，建议同时设置 `allowed_chat_ids`。

### 群聊数据边界

- 私聊知识使用 `private:{user_id}` 命名空间，保持现有自动学习行为。
- 群聊默认关闭长期学习；旁听缓冲只存在内存中。
- 开启群学习后使用 `group:{chat_id}` 或 `topic:{chat_id}:{message_thread_id}`，不会写入成员个人知识命名空间。
- 群聊 RAG 检索按同一命名空间过滤；LangChain 模式下群聊使用隔离的简化 RAG 路径，避免全局检索工具跨群读取。
- 请告知群成员机器人可能接收哪些消息，并按需要保持 BotFather 隐私模式开启。

完整设计见 [`docs/group-chat-design.md`](docs/group-chat-design.md)。

## 🔧 高级功能

### 1. 智能学习系统

机器人会自动从对话中学习：
- 个人信息（生日、喜好等）
- 问答对话
- 重要事实
- 用户习惯

### 2. 工具调用能力

通过 MCP 协议支持：
- 时间与时区转换
- Context7 开源库文档查询
- 受目录和只读策略限制的文件系统访问
- 经白名单过滤的 Git/GitHub 仓库查询
- 可选网络搜索与网页抓取
- 自定义工具扩展

统一 Agent 支持连续多轮工具调用，例如先让 Context7 解析库 ID，再查询对应文档；`max_tool_rounds` 和 `max_tool_calls` 会限制单次对话的调用规模。

### 3. 多模态处理

- 文本对话
- 图片分析和搜索
- 文件处理

## 🚨 故障排除

### 常见问题

1. **启动失败**
   ```bash
   # 检查配置文件
   python -c "import tomllib; print('配置OK' if tomllib.load(open('config.toml', 'rb')) else '配置错误')"
   ```

2. **API 连接问题**
   ```bash
   # 测试 API 连接
   curl -H "Authorization: Bearer YOUR_API_KEY" YOUR_BASE_URL/models
   ```

3. **数据库问题**
   ```bash
   # 查看当前迁移版本；启动时会自动幂等升级，不要删除 bot.db
   sqlite3 data/bot.db "SELECT * FROM schema_migrations ORDER BY version;"
   ```

4. **向量存储问题**
   - 先运行 `python -c "import torch; print(torch.__version__)"`，确认 PyTorch 为 2.13.x。
   - 切换嵌入模型后无需手动删除旧索引；启动时会根据索引清单自动从 SQLite 重建。
   - 查看日志中的“向量索引与当前模型不兼容”和“从数据库重建”记录，确认重建已完成。
   - 不要删除 `data/bot.db`；索引目录只是可重建的派生数据。
   - 模型下载或索引写入失败时，检查网络、磁盘空间和目录权限，然后重启或执行 `/rebuild_knowledge`。

5. **MCP 服务器连接失败**
   - 使用 `command -v npx` 和 `command -v uvx` 确认启动命令存在；macOS 上还要留意 IDE 自带 Node.js 是否遮蔽 Homebrew Node.js。
   - 首次运行 `npx`/`uvx` 服务器可能下载包，可适当提高 `connect_timeout`。
   - 查看 `/mcp_status` 和 `logs/bot.log` 中对应服务器的错误；某服务器失败不会阻止其他服务器连接。
   - `read_only = true` 后工具列表为空，通常表示该服务器没有提供 `readOnlyHint` 注解，请改用严格的 `include_tools` 白名单并自行审计工具行为。

6. **群里不响应**
   - 确认机器人已加入群，且 `[telegram.group].enabled = true`。
   - 检查群/Topic 是否被白名单、黑名单或忽略列表拒绝。
   - 开启隐私模式时请使用明确 `@机器人用户名`、回复或命令。
   - `/command@OtherBot` 不会被 AL1S 处理；用户名匹配不区分大小写。
   - Topic 已关闭、机器人没有发言权限或消息被删除时，检查 `logs/bot.log` 中的结构化拒绝原因。

7. **旁听上下文为空**
   - BotFather 隐私模式可能阻止 Telegram 投递普通群消息。
   - 确认 `observe_unmentioned_messages = true`，且消息未超过 `context_buffer_ttl`。
   - 缓冲按群和 Topic 隔离，不会从其他 Topic 读取。

### 性能优化

- **内存使用**: Qwen3 0.6B 的实际占用受设备、并发和运行库影响；资源紧张时减小 `embedding_batch_size`
- **存储空间**: 建议为模型缓存、SQLite 和向量索引至少预留 2 GB
- **网络要求**: 首次运行需下载约 1.2 GB 的嵌入模型；后续从本地缓存加载

## 🔒 安全注意事项

1. **API 密钥**: 妥善保管 API 密钥，不要提交到版本控制
2. **访问权限**: 高权限 MCP Server 使用 `access = "admin"`，并正确配置 `telegram.admin_user_ids`；目录权限与只读工具白名单仍是必要的第二层保护
3. **数据隐私**: 定期清理敏感对话记录
4. **网络安全**: 在生产环境中使用 HTTPS 和适当的防火墙设置

## 📊 监控和维护

### 日志管理

```bash
# 查看日志
tail -f logs/bot.log

# 按时间查看
ls logs/bot.*.log
```

### 数据库维护

```bash
# 数据库统计
sqlite3 data/bot.db "
SELECT 
  'messages' as table_name, COUNT(*) as count FROM messages
UNION ALL
SELECT 
  'knowledge_entries' as table_name, COUNT(*) as count FROM knowledge_entries;
"

# 清理旧数据（保留最近30天）
sqlite3 data/bot.db "
DELETE FROM messages 
WHERE created_at < datetime('now', '-30 days');
"
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [OpenAI](https://openai.com/) - AI 模型支持
- [Sentence Transformers](https://www.sbert.net/) - 语义嵌入模型
- [FAISS](https://faiss.ai/) - 向量相似性搜索
- [Model Context Protocol](https://modelcontextprotocol.io/) - 工具集成协议
- [python-telegram-bot](https://python-telegram-bot.org/) - Telegram Bot 框架

---

**邦邦卡邦！** 如有问题，请提交 Issue 或联系维护者。
