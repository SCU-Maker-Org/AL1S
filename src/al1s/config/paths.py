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


def default_paths() -> AppPaths:
    """Resolve host or container paths without creating them."""
    configured = os.getenv("AL1S_CONFIG_FILE")
    container_config = Path("/etc/al1s/config.toml")
    in_container = os.getenv("AL1S_CONTAINER") == "1" or (
        configured is None and container_config.exists()
    )
    if in_container:
        return AppPaths(
            config_file=Path(configured).expanduser() if configured else container_config,
            data_dir=Path("/var/lib/al1s"),
            state_dir=Path("/run/al1s"),
            cache_dir=Path("/var/cache/al1s"),
            secrets_dir=Path("/run/secrets"),
        )

    dirs = PlatformDirs("AL1S", "AL1S", ensure_exists=False)
    return AppPaths(
        config_file=(
            Path(configured).expanduser()
            if configured
            else Path(dirs.user_config_dir) / "config.toml"
        ),
        data_dir=Path(dirs.user_data_dir),
        state_dir=Path(dirs.user_state_dir),
        cache_dir=Path(dirs.user_cache_dir),
        secrets_dir=Path(os.getenv("AL1S_SECRETS_DIR", "/run/secrets")),
    )
