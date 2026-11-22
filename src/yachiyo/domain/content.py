"""Portable inbound and outbound media types."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ArtifactKind(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"


@dataclass(frozen=True, slots=True)
class Attachment:
    external_id: str
    filename: str
    mime_type: str
    size: int
    data: bytes

    def __post_init__(self) -> None:
        if self.size != len(self.data):
            raise ValueError("attachment size does not match payload")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True, slots=True)
class Artifact:
    kind: ArtifactKind
    filename: str
    mime_type: str
    data: bytes | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.path is None):
            raise ValueError("artifact must contain exactly one of data or path")


@dataclass(frozen=True, slots=True)
class ActionButton:
    label: str
    action_id: str
    style: str = "secondary"


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    text: str = ""
    artifacts: tuple[Artifact, ...] = ()
    buttons: tuple[ActionButton, ...] = ()
    private: bool = False


@dataclass(frozen=True, slots=True)
class ResponsePlan:
    messages: tuple[OutboundMessage, ...] = ()
    track_delivery: bool = True

    @classmethod
    def text(cls, text: str, *, private: bool = False) -> ResponsePlan:
        return cls(messages=(OutboundMessage(text=text, private=private),))
