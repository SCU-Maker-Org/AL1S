"""TOML, environment, secret and CLI configuration loading."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
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


def _environment_overlay(prefix: str = "AL1S__") -> dict[str, Any]:
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
    paths = default_paths()
    if config_file is not None:
        paths = AppPaths(
            config_file=Path(config_file).expanduser(),
            data_dir=paths.data_dir,
            state_dir=paths.state_dir,
            cache_dir=paths.cache_dir,
            secrets_dir=paths.secrets_dir,
        )

    data: dict[str, Any] = {}
    if paths.config_file.exists():
        try:
            with paths.config_file.open("rb") as handle:
                parsed = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot load configuration: {exc}") from exc
        _deep_merge(data, parsed)
    elif config_file is not None or os.getenv("AL1S_CONFIG_FILE"):
        raise ConfigError(f"configuration file does not exist: {paths.config_file}")

    _deep_merge(data, _environment_overlay())
    if cli_overrides:
        _deep_merge(data, cli_overrides)

    try:
        settings = AppSettings.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"invalid configuration: {details}") from None

    data_dir = Path(settings.storage.database_path).expanduser()
    if data_dir.is_absolute():
        resolved_data_dir = data_dir.parent
    else:
        resolved_data_dir = paths.data_dir
    paths = AppPaths(
        config_file=paths.config_file,
        data_dir=resolved_data_dir,
        state_dir=paths.state_dir,
        cache_dir=paths.cache_dir,
        secrets_dir=paths.secrets_dir,
    )
    return LoadedConfig(settings=settings, paths=paths, secrets=SecretStore(paths.secrets_dir))
