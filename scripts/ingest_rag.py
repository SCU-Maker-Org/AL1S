from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infra.database import DatabaseService
from src.rag.chunking import ChunkingConfig
from src.rag.ingestion import IngestionConfig, RagIngestor
from src.rag.sqlite_store import SQLiteRagStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest technical documents into AL1S SQLite and vector indexes."
    )
    parser.add_argument("path", type=Path, help="UTF-8 document or directory to ingest")
    parser.add_argument(
        "--domains",
        default="",
        help="Comma-separated defaults: sys,hpc,compile,distributed,db,storage,ai_workload,cloud,security",
    )
    parser.add_argument(
        "--namespace", help="Knowledge namespace from config by default"
    )
    parser.add_argument("--collection", help="Collection name from config by default")
    parser.add_argument("--database", type=Path, default=Path("data/bot.db"))
    parser.add_argument("--strict", action="store_true", help="Stop on first bad file")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Write SQLite/FTS only; rebuild the dense index in a later run",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON result")
    return parser


async def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output = sys.stderr if args.json else sys.stdout
    with contextlib.redirect_stdout(output):
        from src.config import config

    namespace = (args.namespace or config.rag.technical_namespace).strip()
    collection = (args.collection or config.rag.technical_collection).strip()
    if not namespace or not collection:
        raise ValueError("namespace and collection must not be empty")

    database_path = args.database.expanduser()
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    _initialize_database_file(database_path)
    database = DatabaseService(str(database_path))
    store = SQLiteRagStore(database, collection=collection)
    ingestor = RagIngestor(
        IngestionConfig(
            max_file_bytes=config.rag.max_document_bytes,
            chunking=ChunkingConfig(
                max_chars=config.rag.chunk_size,
                overlap_chars=config.rag.chunk_overlap,
            ),
            strict=args.strict,
        )
    )
    stats = await ingestor.ingest(
        args.path.expanduser(),
        store,
        namespace=namespace,
        default_domains=args.domains or None,
    )

    vector_initialized = False
    vector_rebuilt = False
    if not args.no_rebuild and stats.failed_files == 0:
        from src.infra.vector import VectorService

        vector = VectorService(
            database_service=database,
            vector_store_path=config.agent.vector_store_path,
        )
        vector_initialized = await vector.initialize(
            embedding_model_type=config.agent.embedding_model,
            vector_store_backend=config.agent.vector_store,
            embedding_revision=config.agent.embedding_revision,
            embedding_device=config.agent.embedding_device,
            embedding_batch_size=config.agent.embedding_batch_size,
            force_rebuild=True,
        )
        vector_rebuilt = vector_initialized
        vector.cleanup()

    ingestion_succeeded = stats.failed_files == 0 and (
        stats.processed_documents > 0
        or stats.deleted_documents > 0
        or stats.excluded_documents > 0
    )
    result = {
        "ok": ingestion_succeeded
        and (args.no_rebuild or (vector_initialized and vector_rebuilt)),
        "source": str(args.path.expanduser().resolve()),
        "database": str(database_path.resolve()),
        "collection": collection,
        "namespace": namespace,
        "dense_index": {
            "requested": not args.no_rebuild,
            "initialized": vector_initialized,
            "rebuilt": vector_rebuilt,
        },
        "stats": asdict(stats),
        "database_totals": database.get_rag_document_stats(),
    }
    if not ingestion_succeeded:
        return 2, result
    if not args.no_rebuild and not vector_initialized:
        return 3, result
    if not args.no_rebuild and not vector_rebuilt:
        return 4, result
    return 0, result


def _initialize_database_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    init_sql = (PROJECT_ROOT / "data" / "init_db.sql").read_text(encoding="utf-8")
    with sqlite3.connect(path) as connection:
        connection.executescript(init_sql)


def _print_human(result: dict[str, Any]) -> None:
    stats = result["stats"]
    dense = result["dense_index"]
    print(
        f"RAG ingest: {result['source']} -> {result['collection']} "
        f"({result['namespace']})"
    )
    print(
        "Documents: "
        f"inserted={stats['inserted_documents']} "
        f"updated={stats['updated_documents']} "
        f"unchanged={stats['unchanged_documents']} "
        f"deleted={stats['deleted_documents']} "
        f"failed={stats['failed_files']}"
    )
    print(
        "Chunks: "
        f"inserted={stats['inserted_chunks']} updated={stats['updated_chunks']} "
        f"unchanged={stats['unchanged_chunks']} deleted={stats['deleted_chunks']}"
    )
    print(
        f"Dense index: requested={dense['requested']} "
        f"initialized={dense['initialized']} rebuilt={dense['rebuilt']}"
    )
    for issue in stats["issues"]:
        detail = f": {issue['detail']}" if issue["detail"] else ""
        print(f"Issue [{issue['reason']}] {issue['source']}{detail}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, result = asyncio.run(run(args))
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            print(f"RAG ingest failed: {result['error']}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        _print_human(result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
