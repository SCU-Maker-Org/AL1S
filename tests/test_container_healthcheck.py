import sqlite3
from pathlib import Path

import pytest

from scripts import container_healthcheck


def _create_image_files(root: Path) -> None:
    for relative_path in container_healthcheck.REQUIRED_IMAGE_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_runtime_healthcheck_validates_config_and_database(tmp_path, monkeypatch):
    _create_image_files(tmp_path)
    (tmp_path / "config.toml").write_text(
        "[app]\nname = 'AL1S'\n[openai]\napi_key = 'test'\n"
        "[telegram]\nbot_token = 'test'\n",
        encoding="utf-8",
    )
    for relative_path in ("data", "logs", ".cache", ".npm", ".npm-global"):
        (tmp_path / relative_path).mkdir(exist_ok=True)
    with sqlite3.connect(tmp_path / "data" / "bot.db") as connection:
        connection.execute("CREATE TABLE schema_migrations (version INTEGER)")
        connection.execute("CREATE TABLE users (id INTEGER)")

    monkeypatch.setattr(container_healthcheck, "_check_commands", lambda: None)

    container_healthcheck.run_healthcheck(tmp_path)


def test_smoke_healthcheck_imports_dependencies(tmp_path, monkeypatch):
    _create_image_files(tmp_path)
    calls = []
    monkeypatch.setattr(container_healthcheck, "_check_commands", lambda: None)
    monkeypatch.setattr(
        container_healthcheck, "_check_imports", lambda: calls.append(1)
    )

    container_healthcheck.run_healthcheck(tmp_path, smoke=True)

    assert calls == [1]


def test_healthcheck_rejects_incomplete_database(tmp_path, monkeypatch):
    _create_image_files(tmp_path)
    (tmp_path / "config.toml").write_text(
        "[app]\n[openai]\n[telegram]\n", encoding="utf-8"
    )
    for relative_path in ("data", "logs", ".cache", ".npm", ".npm-global"):
        (tmp_path / relative_path).mkdir(exist_ok=True)
    with sqlite3.connect(tmp_path / "data" / "bot.db") as connection:
        connection.execute("CREATE TABLE users (id INTEGER)")

    monkeypatch.setattr(container_healthcheck, "_check_commands", lambda: None)

    with pytest.raises(container_healthcheck.HealthcheckError, match="schema"):
        container_healthcheck.run_healthcheck(tmp_path)
