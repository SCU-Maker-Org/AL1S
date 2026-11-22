# yachiyo 2

yachiyo is a lightweight AI assistant that runs Telegram and Discord as independent,
equal transports over one application core. It uses OpenAI-compatible cloud APIs,
one SQLite database with FTS5 full-text search, and trusted HTTP MCP
servers.

## Project layout

| Path | Contents |
| --- | --- |
| `config/` | Runtime TOML, local secrets, Ruff rules, Containerfile and Podman Quadlet |
| `src/yachiyo/` | CLI, configuration loader, application logic and adapters |
| `docs/` | [Architecture](docs/architecture.md) and [Podman setup](docs/podman.md) |

Python and uv discover `pyproject.toml`, `uv.lock` and `.python-version` at the
project root. Git and container builds read their ignore files there, and GitHub
Actions discovers workflows in `.github/workflows/`.

## Local setup

From the checkout root with Python 3.13 and uv installed, run

```sh
uv sync --group dev --extra proxy
umask 077
cp -n config/config.example.toml config/config.toml
mkdir -p config/.secrets
printf '%s' "$TELEGRAM_BOT_TOKEN" > config/.secrets/telegram_bot_token
printf '%s' "$DISCORD_BOT_TOKEN" > config/.secrets/discord_bot_token
printf '%s' "$OPENAI_API_KEY" > config/.secrets/openai_api_key
uv run yachiyo doctor
uv run yachiyo run
```

When upgrading an existing checkout, move your root `config.toml` and any
`.secrets/` directory into `config/` before running the setup commands.

Run `uv run yachiyo config validate` to see all effective settings, including defaults.

The example starts Telegram and Discord together and retries each connection
independently. Discord supports slash commands, direct messages and messages that
mention the bot with the example settings. To read other guild message content,
enable Message Content Intent in the Discord Developer Portal and set
`[transports.discord].message_content_intent` to `true`.

yachiyo 2 uses a new schema and intentionally does not migrate v1 data in place. Back up
or export an existing database, then start v2 with a new `storage.database_path`.

### Configuration lookup

Both the CLI and the workspace server use `YACHIYO_CONFIG_FILE` when set. The CLI also
accepts `--config PATH`, which takes precedence. When running the checkout's source from
its root, both discover `config/config.toml`. Installed applications use the platform's
user configuration directory. Containers read `/etc/yachiyo/config.toml`.

On a host, the core reads secrets from `YACHIYO_SECRETS_DIR`, the `.secrets/` directory
beside the selected TOML if present, or `/run/secrets`, in that order. Containers
use `/run/secrets`. Relative database paths use the platform data directory.

## CLI

```text
yachiyo run
yachiyo config validate
yachiyo doctor
yachiyo status
yachiyo db migrate
yachiyo rag ingest PATH --scope public [--apply] [--prune]
yachiyo rag search QUERY --scope public
yachiyo rag rebuild
yachiyo mcp check
```

RAG ingestion is a dry run unless `--apply` is supplied. Deletion is never implicit;
`--prune` must be requested explicitly. RAG, migration and MCP check commands require
the Bot process to be stopped because yachiyo enforces one owner for its SQLite database.

Document and memory searches run locally using keywords. Existing v2 databases can
use their full-text indexes immediately. `yachiyo rag rebuild` restores these indexes
from stored document chunks and memories.

Reverse-image search is disabled by default. Enabling it uploads submitted image bytes
to the configured trace.moe and/or SauceNAO service.

## Podman deployment

The rootless Quadlet in `config/yachiyo.container` runs the core as `yachiyo.service`.
After installation, use `systemctl --user restart yachiyo` to apply configuration
changes and `podman exec yachiyo yachiyo status` to see transport health. [Build and install the Quadlet](docs/podman.md)
