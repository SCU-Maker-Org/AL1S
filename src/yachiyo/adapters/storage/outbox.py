"""Files retained beside SQLite until a transport delivers its response."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yachiyo.domain import Artifact, ResponsePlan
from yachiyo.domain.content import ArtifactKind


@dataclass(frozen=True, slots=True)
class OutboxArtifacts:
    root: Path
    max_bytes: int

    def event_dir(self, transport: str, bot_id: str, event_id: str) -> Path:
        key = hashlib.blake2b(
            f"{transport}\x1f{bot_id}\x1f{event_id}".encode(), digest_size=16
        ).hexdigest()
        return self.root / key

    def persist(
        self,
        transport: str,
        bot_id: str,
        event_id: str,
        processing_token: str,
        response: ResponsePlan,
    ) -> list[list[dict[str, str]]]:
        root = self.event_dir(transport, bot_id, event_id) / processing_token
        result: list[list[dict[str, str]]] = []
        total_bytes = 0
        try:
            if any(message.artifacts for message in response.messages):
                for directory in (self.root, root.parent, root):
                    directory.mkdir(exist_ok=True, mode=0o700)
                    if directory.is_symlink() or not directory.is_dir():
                        raise ValueError("outbox directory is not safe")
                    directory.chmod(0o700)
            for message_index, message in enumerate(response.messages):
                artifacts: list[dict[str, str]] = []
                for artifact_index, artifact in enumerate(message.artifacts):
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact.filename)[:128]
                    safe_name = safe_name or "artifact"
                    destination = root / f"{message_index}-{artifact_index}-{safe_name}"
                    data = self._artifact_bytes(artifact)
                    total_bytes += len(data)
                    if total_bytes > self.max_bytes:
                        raise ValueError("artifacts exceed the output byte budget")
                    temporary = destination.with_name(f".{destination.name}.tmp")
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    try:
                        with os.fdopen(descriptor, "wb") as handle:
                            descriptor = -1
                            handle.write(data)
                            handle.flush()
                            os.fsync(handle.fileno())
                    finally:
                        if descriptor >= 0:
                            os.close(descriptor)
                    temporary.replace(destination)
                    artifacts.append(
                        {
                            "kind": artifact.kind.value,
                            "filename": artifact.filename,
                            "mime_type": artifact.mime_type,
                            "path": str(destination),
                        }
                    )
                result.append(artifacts)
            if root.exists():
                for directory in (root, root.parent, self.root, self.root.parent):
                    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return result

    def _artifact_bytes(self, artifact: Artifact) -> bytes:
        if artifact.data is not None:
            data = artifact.data
        else:
            descriptor = os.open(
                Path(artifact.path),
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError("artifact path is not a safe regular file")
                if metadata.st_size > self.max_bytes:
                    raise ValueError("artifact exceeds the output byte budget")
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = -1
                    data = handle.read(self.max_bytes + 1)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        if len(data) > self.max_bytes:
            raise ValueError("artifact exceeds the output byte budget")
        return data

    @staticmethod
    def restore(payload: dict[str, Any], artifact_root: Path) -> Artifact:
        if not isinstance(payload, dict):
            raise ValueError("response artifact must be an object")
        root = artifact_root.resolve(strict=True)
        candidate = Path(str(payload["path"]))
        if candidate.is_symlink():
            raise ValueError("outbox artifact must not be a symbolic link")
        path = candidate.resolve(strict=True)
        metadata = path.stat()
        if (
            not path.is_relative_to(root)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError("outbox artifact is not a safe regular file")
        return Artifact(
            kind=ArtifactKind(payload["kind"]),
            filename=str(payload["filename"]),
            mime_type=str(payload["mime_type"]),
            path=path,
        )

    def remove(self, response_json: str, artifact_root: Path | None = None) -> None:
        try:
            payload = json.loads(response_json)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("messages", []), list):
            return
        root = (artifact_root if artifact_root is not None else self.root).resolve()
        parents: set[Path] = set()
        for message in payload.get("messages", []):
            if not isinstance(message, dict) or not isinstance(message.get("artifacts", []), list):
                continue
            for artifact in message.get("artifacts", []):
                if not isinstance(artifact, dict):
                    continue
                candidate = Path(str(artifact.get("path", "")))
                if not candidate.is_absolute() or candidate.is_symlink():
                    continue
                path = candidate.resolve()
                if path.name and path.is_relative_to(root):
                    path.unlink(missing_ok=True)
                    parents.add(path.parent)
        for parent in parents:
            while parent != root and parent.is_relative_to(root):
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    def _remove_persisted_artifacts(self, messages: list[list[dict[str, str]]]) -> None:
        payload = json.dumps({"messages": [{"artifacts": item} for item in messages]})
        self.remove(payload)

    async def discard(self, messages: list[list[dict[str, str]]]) -> None:
        cleanup = asyncio.create_task(asyncio.to_thread(self._remove_persisted_artifacts, messages))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    def remove_orphans(self, referenced: set[Path]) -> int:
        root = self.root
        if not root.exists():
            return 0
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("outbox root is not a safe directory")
        removed = 0
        for event_root in root.iterdir():
            if event_root.is_symlink() or not event_root.is_dir():
                continue
            for token_root in event_root.iterdir():
                if token_root.is_symlink() or not token_root.is_dir():
                    continue
                if token_root.resolve() in referenced:
                    continue
                shutil.rmtree(token_root)
                removed += 1
            try:
                event_root.rmdir()
            except OSError:
                pass
        return removed
