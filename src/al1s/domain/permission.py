"""Platform-neutral access decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class AccessLevel(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    ADMIN = "admin"
    PRIVATE_ADMIN = "private_admin"


class _AccessRank(IntEnum):
    PUBLIC = 0
    PRIVATE = 1
    ADMIN = 2
    PRIVATE_ADMIN = 3


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

    @property
    def rank(self) -> _AccessRank:
        if self.is_private and self.is_admin:
            return _AccessRank.PRIVATE_ADMIN
        if self.is_admin:
            return _AccessRank.ADMIN
        if self.is_private:
            return _AccessRank.PRIVATE
        return _AccessRank.PUBLIC
