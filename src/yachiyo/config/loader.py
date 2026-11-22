"""TOML, environment, secret and CLI configuration loading."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import AppSettings
from .paths import AppPaths, default_paths
from .secrets import SecretStore


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    settings: AppSettings
    paths: AppPaths
    secrets: SecretStore


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _decode_env_value(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith(("[", "{")) or stripped.casefold() in {
        "true",
        "false",
        "null",
    }:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            raise ConfigError("invalid JSON environment value") from None
    return value


def _environment_overlay(prefix: str = "YACHIYO__") -> dict[str, Any]:
    result: dict[str, Any] = {}
    entries = [
        (env_name, raw_value)
        for env_name, raw_value in os.environ.items()
        if env_name.startswith(prefix)
    ]
    entries.sort(key=lambda item: len(item[0].split("__")))
    for env_name, raw_value in entries:
        parts = [part.lower() for part in env_name[len(prefix) :].split("__") if part]
        if not parts:
            continue
        cursor = result
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"conflicting environment setting: {env_name}")
            cursor = child
        if isinstance(cursor.get(parts[-1]), dict):
            raise ConfigError(f"conflicting environment setting: {env_name}")
        cursor[parts[-1]] = _decode_env_value(raw_value)
    return result


def load_config(
    config_file: str | Path | None = None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
) -> LoadedConfig:
    paths = default_paths(config_file)

    data: dict[str, Any] = {}
    if paths.config_file.exists():
        try:
            with paths.config_file.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot load configuration: {exc}") from exc
    elif config_file is not None or os.getenv("YACHIYO_CONFIG_FILE"):
        raise ConfigError(f"configuration file does not exist: {paths.config_file}")

    _deep_merge(data, _environment_overlay())
    if cli_overrides:
        _deep_merge(data, cli_overrides)

    # Existing v2 installations may still have the retired retrieval settings.
    for section, key in (("models", "embedding"), ("rag", "dense_candidates")):
        values = data.get(section)
        if isinstance(values, dict) and key in values:
            data[section] = {name: value for name, value in values.items() if name != key}

    try:
        settings = AppSettings.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"invalid configuration: {details}") from None

    database = Path(settings.storage.database_path).expanduser()
    if database.is_absolute():
        paths = replace(paths, data_dir=database.parent)
    return LoadedConfig(settings=settings, paths=paths, secrets=SecretStore(paths.secrets_dir))
