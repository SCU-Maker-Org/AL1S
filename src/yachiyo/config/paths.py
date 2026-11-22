"""Cross-platform filesystem locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs


@dataclass(frozen=True, slots=True)
class AppPaths:
    config_file: Path
    data_dir: Path
    state_dir: Path
    cache_dir: Path
    secrets_dir: Path

    @property
    def heartbeat_file(self) -> Path:
        return self.state_dir / "health.json"


def _host_config(user_config_dir: Path) -> Path:
    # Only discover a checkout's config when running that checkout's source.
    source_dir = Path(__file__).resolve().parents[2]
    if source_dir == Path.cwd() / "src":
        local = source_dir.parent / "config" / "config.toml"
        if local.is_file():
            return local
    return user_config_dir / "config.toml"


def default_paths(config_file: str | Path | None = None) -> AppPaths:
    """Resolve host or container paths without creating them."""
    configured = os.getenv("YACHIYO_CONFIG_FILE")
    selected = config_file if config_file is not None else configured
    container_config = Path("/etc/yachiyo/config.toml")
    in_container = os.getenv("YACHIYO_CONTAINER") == "1" or (
        configured is None and container_config.exists()
    )
    if in_container:
        return AppPaths(
            config_file=Path(selected).expanduser() if selected else container_config,
            data_dir=Path("/var/lib/yachiyo"),
            state_dir=Path("/run/yachiyo"),
            cache_dir=Path("/var/cache/yachiyo"),
            secrets_dir=Path("/run/secrets"),
        )

    dirs = PlatformDirs("yachiyo", "yachiyo", ensure_exists=False)
    path = Path(selected).expanduser() if selected else _host_config(Path(dirs.user_config_dir))
    local_secrets = path.parent / ".secrets"
    secrets_dir = local_secrets if local_secrets.is_dir() else Path("/run/secrets")
    return AppPaths(
        config_file=path,
        data_dir=Path(dirs.user_data_dir),
        state_dir=Path(dirs.user_state_dir),
        cache_dir=Path(dirs.user_cache_dir),
        secrets_dir=Path(os.getenv("YACHIYO_SECRETS_DIR", secrets_dir)),
    )
