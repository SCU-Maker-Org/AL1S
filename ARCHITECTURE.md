# AL1S 2 Architecture

## Process topology

The default deployment is one core process and one SQLite file. Telegram polling and
the Discord gateway are supervised siblings:

```text
Telegram adapter --\
                    TransportSupervisor -> ChatApplication -> AgentRunner -> cloud API
Discord adapter ---/                         |       |             |
                                             |       |             +-> HTTP MCP
                                             |       +-> SQLite FTS5/sqlite-vec
                                             +-> CommandBus / policy / profiles / memory
```

No transport starts or owns another transport. Each supervisor has its own retry loop,
health state and client lifecycle. The application receives only normalized domain
events and returns a platform-neutral response plan.

## Dependency rule

`domain` contains immutable Python types and imports only the standard library.
`application` imports only configuration, domain types and ports. `adapters` implement
ports using external SDKs. `runtime` supervises adapters. `bootstrap.py` is the only
composition root. New modules must preserve this dependency direction.

## Request flow

1. A transport maps an update to `InboundEvent` with provider-qualified identity,
   conversation address, attachments and verified platform permissions.
2. SQLite atomically claims the external event ID to provide restart-safe idempotency.
3. Group policy computes a `SessionKey`; transport, bot, chat and scope always
   participate, while topic and owner dimensions are included only for their matching
   session modes.
4. Rate limits and a bounded per-session lock run before expensive work.
5. `ChatApplication` loads trusted server-side history, profile data and scoped
   public/private retrieval results.
6. `AgentRunner` executes a bounded OpenAI-compatible tool loop. It has no mutable
   conversation field and receives all state per request.
7. The user/assistant turn, response outbox, usage and optional memory job are committed
   in one SQLite transaction before transport delivery.
8. The originating transport renders text limits, buttons and media for its platform.

The durable outbox provides at-least-once delivery. Telegram and Discord do not expose
a transaction spanning multi-part sends, so a process or network failure after a
platform accepts only part of a response can duplicate that accepted prefix on replay.
Model and remote read-only tool calls are likewise at-least-once across process crashes;
non-read-only MCP calls require a separately persisted user approval.

## Data model

SQLite stores provider-qualified identities, chats, conversations, messages, roles,
profiles, group policy, memories, documents, chunks, event receipts, durable jobs,
tool approvals and usage. Foreign keys, WAL and a busy timeout are enabled at startup.
Numbered SQL files are the only migration mechanism.

Documents and memories have FTS5 rows and sqlite-vec rows in the same database. Dense
and sparse results are fused with reciprocal rank fusion. Every private query requires
an explicit scope and additionally sees `public`; it cannot query another private or
group scope. Vector tables are generation-named so an embedding dimension change can
be rebuilt and switched without mixing incompatible vectors.

## Trust boundaries

- System instructions contain only application-authored text. Profiles, group context,
  retrieval and tool output enter the model as explicitly untrusted user/tool data.
- Unmentioned group observation and group learning are disabled by default. Enabling
  group memory records consent in SQLite.
- Secrets are references in TOML and values in `/run/secrets` or explicit development
  environment variables. Effective configuration and logs never contain secret values.
- MCP accepts only configured HTTP endpoints, does not spawn commands, prefixes every
  tool name, requires an explicit non-empty allowlist, limits response bytes, filters
  access and requires persistent approval for every non-read-only tool.
- Media and reverse-image integrations validate HTTPS hosts, resolved public IPs,
  redirects, MIME types, magic bytes, response sizes and timeouts.
- The workspace MCP server is a separate container. It rejects traversal, `.git` file
  access and symlinks; arbitrary shell is unavailable and checks come from a fixed list.

## Lifecycle

Startup validates configuration, opens and migrates SQLite, loads sqlite-vec, seeds
roles, initializes integrations and then starts worker, heartbeat and all transports in
one task group. A transport exception is contained inside its supervisor. Shutdown
stops intake, closes transports, finishes active worker tasks, closes HTTP clients and
then closes SQLite. `health.json` is atomically refreshed for OCI health checks.

## Optional components

The standard core does not require the official MCP SDK because its client is a bounded
aiohttp implementation. The `mcp`/`workspace` extras exist only for building custom MCP
servers. Qdrant, Docling, Mem0 and ToolHive may be added later behind existing ports,
but are not part of the default architecture.
