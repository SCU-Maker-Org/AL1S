"""Platform-neutral access decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AccessLevel(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    ADMIN = "admin"
    PRIVATE_ADMIN = "private_admin"


@dataclass(frozen=True, slots=True)
class RequestPermission:
    is_private: bool
    is_admin: bool

    def allows(self, level: AccessLevel) -> bool:
        if level is AccessLevel.PUBLIC:
            return True
        if level is AccessLevel.PRIVATE:
            return self.is_private
        if level is AccessLevel.ADMIN:
            return self.is_admin
        return self.is_private and self.is_admin
