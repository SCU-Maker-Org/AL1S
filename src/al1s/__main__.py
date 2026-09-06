"""AL1S command-line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from al1s.adapters.mcp import TrustedHTTPToolRegistry
from al1s.adapters.storage import Database
from al1s.bootstrap import Runtime, heartbeat_path, open_rag_resources
from al1s.config import load_config
from al1s.runtime.health import read_heartbeat
from al1s.runtime.logging import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="al1s")
    parser.add_argument("--config", type=Path, help="path to config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="run all enabled transports")
    commands.add_parser("doctor", help="validate local configuration and storage")
    commands.add_parser("status", help="read the process heartbeat")

    config = commands.add_parser("config", help="configuration operations")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("validate")

    database = commands.add_parser("db", help="database operations")
    database_commands = database.add_subparsers(dest="db_command", required=True)
    database_commands.add_parser("migrate")

    rag = commands.add_parser("rag", help="retrieval operations")
    rag_commands = rag.add_subparsers(dest="rag_command", required=True)
    ingest = rag_commands.add_parser("ingest")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--scope", default="public")
    ingest.add_argument("--apply", action="store_true")
    ingest.add_argument("--prune", action="store_true")
    search = rag_commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--scope", default="public")
    search.add_argument("--limit", type=int, default=8)
    rag_commands.add_parser("rebuild")

    mcp = commands.add_parser("mcp", help="MCP operations")
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_commands.add_parser("check")
    return parser


async def _run(args: argparse.Namespace) -> int:
    loaded = load_config(args.config)
    configure_logging(loaded.settings.logging)
    if args.command == "config":
        rendered = loaded.settings.model_dump(mode="json")
        telegram = rendered["transports"]["telegram"]
        if telegram.get("proxy_url"):
            telegram["proxy_url"] = "<redacted>"
        print(json.dumps(rendered, indent=2, ensure_ascii=False))
        return 0
    if args.command == "status":
        status = read_heartbeat(heartbeat_path(loaded))
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if status.get("state") in {"running", "degraded"} else 1
    if args.command == "doctor":
        return await _doctor(loaded)
    if args.command == "db":
        database = Database(loaded.settings, loaded.paths)
        await database.open()
        print(f"database ready: {database.path}")
        await database.close()
        return 0
    if args.command == "rag":
        return await _rag(args, loaded)
    if args.command == "mcp":
        database = Database(loaded.settings, loaded.paths)
        registry = TrustedHTTPToolRegistry(loaded.settings, loaded.secrets, database)
        try:
            await database.open()
            await registry.refresh()
            status = registry.status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0 if all(item.get("state") == "connected" for item in status) else 1
        finally:
            await registry.close()
            await database.close()
    runtime = Runtime(loaded)
    loop = asyncio.get_running_loop()
    for event_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(event_signal, runtime.stop)
    await runtime.run()
    return 0


async def _doctor(loaded) -> int:
    checks: dict[str, str] = {
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "config": "ok",
    }
    failures = 0
    for name, secret, required in (
        (
            "telegram_secret",
            loaded.settings.transports.telegram.token_secret,
            loaded.settings.transports.telegram.enabled,
        ),
        (
            "discord_secret",
            loaded.settings.transports.discord.token_secret,
            loaded.settings.transports.discord.enabled,
        ),
        ("model_secret", loaded.settings.models.chat.api_key_secret, True),
        (
            "embedding_secret",
            loaded.settings.models.embedding.api_key_secret,
            loaded.settings.rag.enabled or loaded.settings.memory.enabled,
        ),
        (
            "media_secret",
            loaded.settings.media.api_key_secret,
            loaded.settings.media.enabled,
        ),
        (
            "saucenao_secret",
            loaded.settings.reverse_image.saucenao_api_key_secret,
            loaded.settings.reverse_image.enabled
            and loaded.settings.reverse_image.saucenao_enabled,
        ),
    ):
        try:
            loaded.secrets.read(secret, required=required)
        except RuntimeError as exc:
            checks[name] = str(exc)
            failures += 1
        else:
            checks[name] = "ok" if required else "optional"
    for server in loaded.settings.mcp.servers:
        if not loaded.settings.mcp.enabled or not server.enabled or not server.bearer_secret:
            continue
        name = f"mcp_secret:{server.name}"
        try:
            loaded.secrets.read(server.bearer_secret, required=True)
        except RuntimeError as exc:
            checks[name] = str(exc)
            failures += 1
        else:
            checks[name] = "ok"
    for index, fallback in enumerate(loaded.settings.models.chat.fallback, start=1):
        name = f"fallback_model_secret:{index}"
        try:
            loaded.secrets.read(fallback.api_key_secret, required=True)
        except RuntimeError as exc:
            checks[name] = str(exc)
            failures += 1
        else:
            checks[name] = "ok"
    database = Database(loaded.settings, loaded.paths)
    try:
        await database.open()
        checks["database"] = "ok"
        checks["sqlite_vec"] = (
            "ok"
            if database.vector_enabled
            else "rebuild_required"
            if database.vector_rebuild_required
            else "unavailable"
        )
        failures += int(not database.vector_enabled and loaded.settings.rag.enabled)
    except Exception as exc:
        checks["database"] = f"{type(exc).__name__}: {exc}"
        failures += 1
    finally:
        await database.close()
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 1 if failures else 0


async def _rag(args: argparse.Namespace, loaded) -> int:
    from al1s.domain import KnowledgeScope

    database, model, index = await open_rag_resources(loaded)
    try:
        if args.rag_command == "ingest":
            report = await index.ingest(
                args.path,
                KnowledgeScope(args.scope),
                prune=args.prune,
                dry_run=not args.apply,
            )
            print(json.dumps(asdict(report), indent=2))
        elif args.rag_command == "search":
            hits = await index.search(args.query, KnowledgeScope(args.scope), limit=args.limit)
            print(json.dumps([asdict(hit) for hit in hits], indent=2, ensure_ascii=False))
        else:
            print(json.dumps({"rebuilt": await index.rebuild_vectors()}))
    finally:
        await model.close()
        await database.close()
    return 0


def main() -> None:
    try:
        code = asyncio.run(_run(_parser().parse_args()))
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        print(f"al1s: {type(exc).__name__}: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
