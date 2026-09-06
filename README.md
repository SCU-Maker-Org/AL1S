# AL1S 2

AL1S is a lightweight AI assistant that runs Telegram and Discord as independent,
equal transports over one application core. It uses OpenAI-compatible cloud APIs,
one SQLite database, embedded FTS5 and sqlite-vec retrieval, and trusted HTTP MCP
servers.

## Runtime

- Python 3.13
- aiogram and discord.py transports
- Official OpenAI Python SDK for chat, vision and embeddings
- SQLite, FTS5 and sqlite-vec for all persistent data and retrieval
- aiohttp for direct integrations and the bounded MCP client
- No local model, Torch, FAISS, LangChain, Qdrant, Redis or web control plane

## Local setup

```sh
uv sync --group dev --extra proxy
cp config.example.toml config.toml
umask 077
mkdir -p .secrets
printf '%s' "$TELEGRAM_BOT_TOKEN" > .secrets/telegram_bot_token
printf '%s' "$OPENAI_API_KEY" > .secrets/openai_api_key
AL1S_SECRETS_DIR=.secrets uv run al1s --config config.toml doctor
AL1S_SECRETS_DIR=.secrets uv run al1s --config config.toml run
```

Enable Discord in `config.toml` and add `.secrets/discord_bot_token` to start both
transports concurrently. A failure in one transport is retried independently and does
not stop the other. Discord text conversations require the privileged Message Content
Intent to be enabled in the Discord Developer Portal.

AL1S 2 uses a new schema and intentionally does not migrate v1 data in place. Back up
or export an existing database, then start v2 with a new `storage.database_path`.

## CLI

```text
al1s run
al1s config validate
al1s doctor
al1s status
al1s db migrate
al1s rag ingest PATH --scope public [--apply] [--prune]
al1s rag search QUERY --scope public
al1s rag rebuild
al1s mcp check
```

RAG ingestion is a dry run unless `--apply` is supplied. Deletion is never implicit;
`--prune` must be requested explicitly. RAG, migration and MCP check commands require
the Bot process to be stopped because AL1S enforces one owner for its SQLite database.

Reverse-image search is disabled by default. Enabling it uploads submitted image bytes
to the configured trace.moe and/or SauceNAO service.

## Containers

Build and run with Podman using [deploy/podman.md](deploy/podman.md). The core and
workspace server are separate OCI targets. The core image never receives the Podman
socket, Git, Node, a compiler or model weights.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module and security boundaries.
