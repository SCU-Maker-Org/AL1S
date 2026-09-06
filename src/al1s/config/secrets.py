"""File-backed secret references."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

_SECRET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class SecretError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SecretStore:
    root: Path
    max_bytes: int = 64 * 1024
    allow_environment: bool = True

    def read(self, name: str | None, *, required: bool = False) -> str:
        if not name:
            if required:
                raise SecretError("required secret reference is missing")
            return ""
        if not _SECRET_NAME.fullmatch(name):
            raise SecretError(f"invalid secret name: {name!r}")

        path = self.root / name
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
        except FileNotFoundError:
            if not self.allow_environment:
                if required:
                    raise SecretError(f"required secret {name!r} was not found") from None
                return ""
            env_name = f"AL1S_SECRET_{name.upper().replace('-', '_').replace('.', '_')}"
            value = os.getenv(env_name, "").strip()
            if len(value.encode()) > self.max_bytes:
                raise SecretError(f"secret {name!r} exceeds the size limit") from None
            if required and not value:
                raise SecretError(f"required secret {name!r} was not found") from None
            return value
        except OSError as exc:
            raise SecretError(f"secret {name!r} is not a safe regular file") from exc
        try:
            file_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_size > self.max_bytes
                or file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
                or (hasattr(os, "geteuid") and file_stat.st_uid != os.geteuid())
            ):
                raise SecretError(f"secret {name!r} is not a safe regular file")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                value = handle.read(self.max_bytes + 1).strip()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(value.encode()) > self.max_bytes:
            raise SecretError(f"secret {name!r} exceeds the size limit")
        if required and not value:
            raise SecretError(f"required secret {name!r} is empty")
        return value
