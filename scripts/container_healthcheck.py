#!/usr/bin/env python3
"""Container smoke and runtime health checks for the AL1S image."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from pathlib import Path

REQUIRED_COMMANDS = (
    "git",
    "github-mcp-server",
    "node",
    "npm",
    "npx",
    "uv",
    "uvx",
)
REQUIRED_MODULES = ("telegram", "torch", "sentence_transformers", "faiss", "mcp")
REQUIRED_IMAGE_FILES = (
    "main.py",
    "src/bot.py",
    "scripts/fetch_rag_sources.py",
    "scripts/ingest_rag.py",
    "knowledge/sources.toml",
    "share/init_db.sql",
)
REQUIRED_DATABASE_TABLES = {"schema_migrations", "users"}


class HealthcheckError(RuntimeError):
    """Raised when the image or running container is not healthy."""


def _check_commands() -> None:
    for command in REQUIRED_COMMANDS:
        executable = shutil.which(command)
        if not executable:
            raise HealthcheckError(f"required command is unavailable: {command}")
        subprocess.run(
            [executable, "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )


def _check_image_files(app_root: Path) -> None:
    missing = [path for path in REQUIRED_IMAGE_FILES if not (app_root / path).is_file()]
    if missing:
        raise HealthcheckError(
            f"required image files are missing: {', '.join(missing)}"
        )


def _check_imports() -> None:
    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)


def _check_config(app_root: Path) -> None:
    config_path = app_root / "config.toml"
    if not config_path.is_file():
        raise HealthcheckError("config.toml is not mounted")
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    for section in ("app", "openai", "telegram"):
        if not isinstance(config.get(section), dict):
            raise HealthcheckError(f"config.toml is missing [{section}]")


def _check_writable_directories(app_root: Path) -> None:
    for relative_path in ("data", "logs", ".cache", ".npm", ".npm-global"):
        path = app_root / relative_path
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise HealthcheckError(f"runtime directory is not writable: {path}")


def _check_database(app_root: Path) -> None:
    database_path = app_root / "data" / "bot.db"
    if not database_path.is_file():
        raise HealthcheckError("data/bot.db has not been initialized")
    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    available_tables = {row[0] for row in rows}
    missing_tables = REQUIRED_DATABASE_TABLES - available_tables
    if missing_tables:
        raise HealthcheckError(
            f"database schema is incomplete: {', '.join(sorted(missing_tables))}"
        )


def run_healthcheck(app_root: Path, smoke: bool = False) -> None:
    _check_commands()
    _check_image_files(app_root)
    if smoke:
        _check_imports()
        return
    _check_config(app_root)
    _check_writable_directories(app_root)
    _check_database(app_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--app-root",
        type=Path,
        default=Path(os.environ.get("AL1S_APP_ROOT", "/app")),
    )
    arguments = parser.parse_args(argv)
    try:
        run_healthcheck(arguments.app_root.resolve(), smoke=arguments.smoke)
    except (
        HealthcheckError,
        ImportError,
        OSError,
        sqlite3.Error,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"unhealthy: {exc}", file=sys.stderr)
        return 1
    print("healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
