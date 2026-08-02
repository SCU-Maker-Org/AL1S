from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
)

from .chunking import ChunkingConfig, chunk_document
from .documents import (
    Chunk,
    Document,
    DocumentTooLargeError,
    is_supported_document,
    load_document,
)

DocumentStatus = Literal["inserted", "updated", "unchanged"]


@dataclass(frozen=True, slots=True)
class UpsertResult:
    document_status: DocumentStatus
    inserted_chunks: int = 0
    updated_chunks: int = 0
    unchanged_chunks: int = 0
    deleted_chunks: int = 0

    def __post_init__(self) -> None:
        if self.document_status not in {"inserted", "updated", "unchanged"}:
            raise ValueError(f"Invalid document status: {self.document_status}")
        values = (
            self.inserted_chunks,
            self.updated_chunks,
            self.unchanged_chunks,
            self.deleted_chunks,
        )
        if any(value < 0 for value in values):
            raise ValueError("Upsert counters must not be negative")

    @classmethod
    def inserted(cls, chunk_count: int) -> UpsertResult:
        return cls("inserted", inserted_chunks=chunk_count)

    @classmethod
    def unchanged(cls, chunk_count: int) -> UpsertResult:
        return cls("unchanged", unchanged_chunks=chunk_count)


class DocumentUpsert(Protocol):
    def upsert_document(
        self, document: Document, chunks: Sequence[Chunk]
    ) -> UpsertResult | Awaitable[UpsertResult]:
        pass


UpsertCallback = Callable[
    [Document, Sequence[Chunk]], UpsertResult | Awaitable[UpsertResult]
]


@dataclass(frozen=True, slots=True)
class IngestionIssue:
    source: str
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    path: Path
    source_path: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    files: tuple[DiscoveredFile, ...]
    discovered_files: int
    unsupported_files: int
    oversized_files: int
    unsafe_symlinks: int
    duplicate_files: int
    failed_files: int = 0
    issues: tuple[IngestionIssue, ...] = ()


@dataclass(slots=True)
class IngestionStats:
    discovered_files: int = 0
    eligible_files: int = 0
    unsupported_files: int = 0
    oversized_files: int = 0
    unsafe_symlinks: int = 0
    duplicate_files: int = 0
    failed_files: int = 0
    empty_documents: int = 0
    excluded_documents: int = 0
    deleted_documents: int = 0
    inserted_documents: int = 0
    updated_documents: int = 0
    unchanged_documents: int = 0
    chunks_seen: int = 0
    inserted_chunks: int = 0
    updated_chunks: int = 0
    unchanged_chunks: int = 0
    deleted_chunks: int = 0
    issues: list[IngestionIssue] = field(default_factory=list)

    @property
    def processed_documents(self) -> int:
        return (
            self.inserted_documents + self.updated_documents + self.unchanged_documents
        )


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    max_file_bytes: int = 8 * 1024 * 1024
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    strict: bool = False

    def __post_init__(self) -> None:
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be greater than zero")


class RagIngestor:
    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()

    async def ingest(
        self,
        root: str | Path,
        upsert: DocumentUpsert | UpsertCallback,
        *,
        namespace: str = "default",
        default_domains: str | Iterable[str] | None = None,
        base_metadata: Mapping[str, Any] | None = None,
    ) -> IngestionStats:
        discovery = discover_files(root, max_file_bytes=self.config.max_file_bytes)
        stats = IngestionStats(
            discovered_files=discovery.discovered_files,
            eligible_files=len(discovery.files),
            unsupported_files=discovery.unsupported_files,
            oversized_files=discovery.oversized_files,
            unsafe_symlinks=discovery.unsafe_symlinks,
            duplicate_files=discovery.duplicate_files,
            failed_files=discovery.failed_files,
            issues=list(discovery.issues),
        )
        if self.config.strict and discovery.issues:
            first_issue = discovery.issues[0]
            detail = f": {first_issue.detail}" if first_issue.detail else ""
            raise ValueError(
                f"Strict discovery rejected {first_issue.source} "
                f"({first_issue.reason}){detail}"
            )
        root_path = Path(root).resolve(strict=True)
        allowed_root = root_path if root_path.is_dir() else root_path.parent
        lifecycle = _lifecycle_target(upsert)
        await _call_lifecycle(lifecycle, "begin_ingestion", str(root_path), namespace)
        try:
            for discovered in discovery.files:
                try:
                    document = load_document(
                        discovered.path,
                        root=allowed_root,
                        namespace=namespace,
                        default_domains=default_domains,
                        base_metadata=base_metadata,
                        max_file_bytes=self.config.max_file_bytes,
                    )
                    if document.metadata.get("index", True) is False:
                        stats.excluded_documents += 1
                        continue
                    if not document.domains:
                        stats.issues.append(
                            IngestionIssue(
                                discovered.source_path,
                                "missing_domains",
                                "set YAML frontmatter domains or pass --domains",
                            )
                        )
                        if self.config.strict:
                            raise ValueError(
                                "Document has no controlled domain: "
                                f"{discovered.source_path}"
                            )
                        stats.failed_files += 1
                        continue
                    if not document.content:
                        stats.empty_documents += 1
                        stats.issues.append(
                            IngestionIssue(discovered.source_path, "empty_document")
                        )
                        continue
                    chunks = chunk_document(document, self.config.chunking)
                    if not chunks:
                        stats.empty_documents += 1
                        stats.issues.append(
                            IngestionIssue(discovered.source_path, "empty_document")
                        )
                        continue
                    result = await _call_upsert(upsert, document, chunks)
                    _validate_result(result, len(chunks))
                    _add_result(stats, result, len(chunks))
                except DocumentTooLargeError as exc:
                    stats.oversized_files += 1
                    stats.issues.append(
                        IngestionIssue(
                            discovered.source_path, "file_too_large", str(exc)
                        )
                    )
                    if self.config.strict:
                        raise
                except Exception as exc:
                    stats.failed_files += 1
                    stats.issues.append(
                        IngestionIssue(
                            discovered.source_path,
                            "ingestion_failed",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                    if self.config.strict:
                        raise
        except BaseException:
            await _call_lifecycle(lifecycle, "finish_ingestion", prune=False)
            raise

        pruning_is_safe = (
            stats.failed_files == 0
            and stats.oversized_files == 0
            and stats.unsafe_symlinks == 0
        )
        pruned = await _call_lifecycle(
            lifecycle, "finish_ingestion", prune=pruning_is_safe
        )
        if isinstance(pruned, Mapping):
            stats.deleted_documents += int(pruned.get("documents", 0))
            stats.deleted_chunks += int(pruned.get("chunks", 0))
        return stats


def discover_files(
    root: str | Path, *, max_file_bytes: int = 8 * 1024 * 1024
) -> DiscoveryResult:
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be greater than zero")
    requested = Path(root)
    resolved_root = requested.resolve(strict=True)
    if resolved_root.is_file():
        return _discover_single_file(requested, resolved_root, max_file_bytes)
    if not resolved_root.is_dir():
        raise ValueError(f"Ingestion root is not a file or directory: {root}")

    files: list[DiscoveredFile] = []
    issues: list[IngestionIssue] = []
    discovered_count = unsupported = oversized = unsafe = duplicates = failed = 0
    seen_targets: set[Path] = set()

    def record_walk_error(error: OSError) -> None:
        nonlocal failed
        failed += 1
        issues.append(
            IngestionIssue(str(error.filename or root), "walk_error", str(error))
        )

    for current, dirnames, filenames in os.walk(
        resolved_root, topdown=True, followlinks=False, onerror=record_walk_error
    ):
        current_path = Path(current)
        safe_directories: list[str] = []
        for name in sorted(dirnames):
            directory = current_path / name
            if directory.is_symlink():
                try:
                    target = directory.resolve(strict=True)
                except OSError as exc:
                    unsafe += 1
                    issues.append(
                        IngestionIssue(str(directory), "broken_symlink", str(exc))
                    )
                    continue
                reason = (
                    "external_symlink"
                    if not target.is_relative_to(resolved_root)
                    else "symlink_directory_skipped"
                )
                unsafe += 1
                issues.append(IngestionIssue(str(directory), reason))
                continue
            safe_directories.append(name)
        dirnames[:] = safe_directories

        # Prefer the real path over an internal symlink that targets the same file.
        for name in sorted(
            filenames, key=lambda item: ((current_path / item).is_symlink(), item)
        ):
            candidate = current_path / name
            source_path = candidate.relative_to(resolved_root).as_posix()
            discovered_count += 1
            try:
                target = candidate.resolve(strict=True)
            except OSError as exc:
                unsafe += 1
                issues.append(IngestionIssue(source_path, "broken_symlink", str(exc)))
                continue
            if not target.is_relative_to(resolved_root):
                unsafe += 1
                issues.append(IngestionIssue(source_path, "external_symlink"))
                continue
            if not target.is_file():
                unsupported += 1
                issues.append(IngestionIssue(source_path, "not_regular_file"))
                continue
            if not is_supported_document(candidate):
                unsupported += 1
                continue
            try:
                target_size = target.stat().st_size
            except OSError as exc:
                failed += 1
                issues.append(IngestionIssue(source_path, "stat_error", str(exc)))
                continue
            if target_size > max_file_bytes:
                oversized += 1
                issues.append(IngestionIssue(source_path, "file_too_large"))
                continue
            if target in seen_targets:
                duplicates += 1
                issues.append(IngestionIssue(source_path, "duplicate_target"))
                continue
            seen_targets.add(target)
            files.append(DiscoveredFile(candidate, source_path))

    files.sort(key=lambda item: item.source_path)
    return DiscoveryResult(
        files=tuple(files),
        discovered_files=discovered_count,
        unsupported_files=unsupported,
        oversized_files=oversized,
        unsafe_symlinks=unsafe,
        duplicate_files=duplicates,
        failed_files=failed,
        issues=tuple(issues),
    )


def _discover_single_file(
    requested: Path, resolved: Path, max_file_bytes: int
) -> DiscoveryResult:
    if not is_supported_document(requested):
        return DiscoveryResult(
            files=(),
            discovered_files=1,
            unsupported_files=1,
            oversized_files=0,
            unsafe_symlinks=0,
            duplicate_files=0,
        )
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        issue = IngestionIssue(requested.name, "stat_error", str(exc))
        return DiscoveryResult(
            files=(),
            discovered_files=1,
            unsupported_files=0,
            oversized_files=0,
            unsafe_symlinks=0,
            duplicate_files=0,
            failed_files=1,
            issues=(issue,),
        )
    if size > max_file_bytes:
        issue = IngestionIssue(requested.name, "file_too_large")
        return DiscoveryResult(
            files=(),
            discovered_files=1,
            unsupported_files=0,
            oversized_files=1,
            unsafe_symlinks=0,
            duplicate_files=0,
            issues=(issue,),
        )
    return DiscoveryResult(
        files=(DiscoveredFile(requested, requested.name),),
        discovered_files=1,
        unsupported_files=0,
        oversized_files=0,
        unsafe_symlinks=0,
        duplicate_files=0,
    )


async def _call_upsert(
    upsert: DocumentUpsert | UpsertCallback,
    document: Document,
    chunks: Sequence[Chunk],
) -> UpsertResult:
    method = getattr(upsert, "upsert_document", None)
    result = (
        method(document, chunks) if method is not None else upsert(document, chunks)
    )
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, UpsertResult):
        raise TypeError("upsert must return an UpsertResult")
    return result


def _lifecycle_target(upsert: DocumentUpsert | UpsertCallback) -> Any | None:
    if hasattr(upsert, "begin_ingestion") or hasattr(upsert, "finish_ingestion"):
        return upsert
    owner = getattr(upsert, "__self__", None)
    if owner is not None and (
        hasattr(owner, "begin_ingestion") or hasattr(owner, "finish_ingestion")
    ):
        return owner
    return None


async def _call_lifecycle(
    target: Any | None, method_name: str, *args: Any, **kwargs: Any
) -> Any:
    if target is None:
        return None
    method = getattr(target, method_name, None)
    if method is None:
        return None
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _validate_result(result: UpsertResult, chunk_count: int) -> None:
    accounted = result.inserted_chunks + result.updated_chunks + result.unchanged_chunks
    if accounted != chunk_count:
        raise ValueError(
            f"upsert accounted for {accounted} current chunks; expected {chunk_count}"
        )


def _add_result(stats: IngestionStats, result: UpsertResult, chunk_count: int) -> None:
    setattr(
        stats,
        f"{result.document_status}_documents",
        getattr(stats, f"{result.document_status}_documents") + 1,
    )
    stats.chunks_seen += chunk_count
    stats.inserted_chunks += result.inserted_chunks
    stats.updated_chunks += result.updated_chunks
    stats.unchanged_chunks += result.unchanged_chunks
    stats.deleted_chunks += result.deleted_chunks
