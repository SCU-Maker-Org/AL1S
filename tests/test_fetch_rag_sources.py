from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from scripts import fetch_rag_sources as fetcher


class FakeContent:
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def iter_chunked(self, _size: int):
        for chunk in self.chunks:
            yield chunk


class FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        chunks: list[bytes] | None = None,
    ):
        self.status = status
        self.headers = headers or {}
        self.content = FakeContent(chunks if chunks is not None else [body])

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected network request")
        return self.responses.pop(0)


def _source(**overrides) -> fetcher.Source:
    values = {
        "id": "postgresql_mvcc",
        "url": "https://www.postgresql.org/docs/current/mvcc.html",
        "title": "PostgreSQL MVCC",
        "domains": ("db", "sys"),
        "product": "PostgreSQL",
        "version": "current",
        "license": "PostgreSQL License",
        "trust_level": "official",
    }
    values.update(overrides)
    return fetcher.Source(**values)


def _policy(**overrides) -> fetcher.FetchPolicy:
    values = {
        "allowed_hosts": frozenset({"www.postgresql.org"}),
        "max_response_bytes": 4096,
        "timeout_seconds": 5.0,
        "max_redirects": 2,
    }
    values.update(overrides)
    return fetcher.FetchPolicy(**values)


async def _public_resolver(_host: str, _port: int) -> tuple[str, ...]:
    return ("93.184.216.34",)


def _minimal_manifest(host: str, url: str, **source_overrides: str) -> str:
    values = {
        "id": "official_doc",
        "title": "Official documentation",
        "product": "PostgreSQL",
        "version": "current",
        "license": "PostgreSQL License",
        "trust_level": "official",
    }
    values.update(source_overrides)
    return f"""manifest_version = 1

[policy]
allowed_hosts = ["{host}"]
max_response_bytes = 4096
timeout_seconds = 5
max_redirects = 2

[[sources]]
id = "{values['id']}"
url = "{url}"
title = "{values['title']}"
domains = ["db", "sys"]
product = "{values['product']}"
version = "{values['version']}"
license = "{values['license']}"
trust_level = "{values['trust_level']}"
"""


def test_repository_manifest_covers_requested_official_domains():
    manifest = fetcher.load_manifest()

    assert manifest.policy.allowed_hosts == fetcher.BUILTIN_OFFICIAL_HOSTS
    assert len(manifest.sources) >= 25
    assert all(source.trust_level == "official" for source in manifest.sources)
    assert all(source.url.startswith("https://") for source in manifest.sources)
    assert {
        "postgresql_transactions",
        "postgresql_indexes",
        "postgresql_wal",
        "postgresql_mvcc",
        "postgresql_explain",
        "cocoon_documentation",
        "linux_kvm",
        "linux_memory_management",
        "linux_userfaultfd",
        "linux_fuse",
        "linux_virtio",
        "kubernetes_cni",
        "kubernetes_security",
        "llvm_optimizer",
        "llvm_codegen",
        "pytorch_distributed",
        "pytorch_cuda",
        "nvidia_nccl_collectives",
        "vllm_parallelism",
        "vllm_prefix_cache",
        "vllm_disaggregated_prefill",
    } <= manifest.by_id.keys()
    assert {domain for source in manifest.sources for domain in source.domains} == (
        fetcher.ALLOWED_DOMAINS
    )


@pytest.mark.parametrize(
    "url, message",
    [
        ("http://www.postgresql.org/docs/current/", "only HTTPS"),
        ("https://user@www.postgresql.org/docs/current/", "credentials"),
        ("https://www.postgresql.org:444/docs/current/", "default HTTPS port"),
        ("https://example.com/docs/current/", "not allowlisted"),
        ("https://www.postgresql.org./docs/current/", "not allowlisted"),
    ],
)
def test_url_validation_rejects_noncanonical_or_unofficial_targets(url, message):
    with pytest.raises(fetcher.UnsafeTargetError, match=message):
        fetcher.validate_target_url(url, {"www.postgresql.org"})


def test_manifest_rejects_host_not_in_builtin_official_set(tmp_path):
    manifest_path = tmp_path / "sources.toml"
    manifest_path.write_text(
        _minimal_manifest("example.com", "https://example.com/docs"),
        encoding="utf-8",
    )

    with pytest.raises(fetcher.ManifestError, match="not an approved official"):
        fetcher.load_manifest(manifest_path)


def test_manifest_rejects_untrusted_source_and_unsafe_id(tmp_path):
    untrusted = tmp_path / "untrusted.toml"
    untrusted.write_text(
        _minimal_manifest(
            "www.postgresql.org",
            "https://www.postgresql.org/docs/current/",
            trust_level="community",
        ),
        encoding="utf-8",
    )
    unsafe_id = tmp_path / "unsafe-id.toml"
    unsafe_id.write_text(
        _minimal_manifest(
            "www.postgresql.org",
            "https://www.postgresql.org/docs/current/",
            id="../escape",
        ),
        encoding="utf-8",
    )

    with pytest.raises(fetcher.ManifestError, match="exactly 'official'"):
        fetcher.load_manifest(untrusted)
    with pytest.raises(fetcher.ManifestError, match="unsafe source id"):
        fetcher.load_manifest(unsafe_id)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.0.2.1",
        "::1",
        "fc00::1",
        "fe80::1",
    ],
)
def test_dns_validation_rejects_non_public_addresses(address):
    with pytest.raises(fetcher.UnsafeTargetError, match="not a public address"):
        fetcher.ensure_public_addresses([address], "www.postgresql.org")


def test_dns_validation_accepts_global_addresses():
    assert fetcher.ensure_public_addresses(
        ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"],
        "www.postgresql.org",
    ) == ("93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")


@pytest.mark.asyncio
async def test_redirect_is_revalidated_and_private_dns_stops_second_request():
    session = FakeSession(
        [
            FakeResponse(302, headers={"Location": "/docs/current/indexes.html"}),
            FakeResponse(
                200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                body=b"<main><h1>Indexes</h1></main>",
            ),
        ]
    )
    resolutions = iter([("93.184.216.34",), ("127.0.0.1",)])

    async def rebinding_resolver(host: str, _port: int) -> tuple[str, ...]:
        return fetcher.ensure_public_addresses(next(resolutions), host)

    with pytest.raises(fetcher.UnsafeTargetError, match="not a public address"):
        await fetcher.fetch_html(
            _source(), _policy(), session, resolve_host=rebinding_resolver
        )

    assert len(session.calls) == 1
    assert session.calls[0][1]["allow_redirects"] is False


@pytest.mark.asyncio
async def test_redirect_to_non_allowlisted_host_is_rejected_without_requesting_it():
    session = FakeSession(
        [FakeResponse(302, headers={"Location": "https://example.com/private"})]
    )

    with pytest.raises(fetcher.UnsafeTargetError, match="not allowlisted"):
        await fetcher.fetch_html(
            _source(), _policy(), session, resolve_host=_public_resolver
        )

    assert [call[0] for call in session.calls] == [_source().url]


@pytest.mark.asyncio
async def test_static_meta_refresh_is_followed_with_full_target_revalidation():
    redirect_page = b"""<!doctype html>
    <meta http-equiv="refresh" content="0; url=../18/mvcc.html">
    <a href="../18/mvcc.html">Continue</a>"""
    session = FakeSession(
        [
            FakeResponse(
                200,
                headers={"Content-Type": "text/html"},
                body=redirect_page,
            ),
            FakeResponse(
                200,
                headers={"Content-Type": "text/html"},
                body=b"<main><h1>MVCC 18</h1></main>",
            ),
        ]
    )
    resolutions: list[str] = []

    async def recording_resolver(host: str, _port: int) -> tuple[str, ...]:
        resolutions.append(host)
        return ("93.184.216.34",)

    html, resolved_url = await fetcher.fetch_html(
        _source(), _policy(), session, resolve_host=recording_resolver
    )

    assert "MVCC 18" in html
    assert resolved_url == "https://www.postgresql.org/docs/18/mvcc.html"
    assert resolutions == ["www.postgresql.org", "www.postgresql.org"]
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_response_size_is_enforced_for_header_and_stream():
    declared = FakeSession(
        [
            FakeResponse(
                200,
                headers={
                    "Content-Type": "text/html",
                    "Content-Length": "100",
                },
                body=b"<main>small</main>",
            )
        ]
    )
    streamed = FakeSession(
        [
            FakeResponse(
                200,
                headers={"Content-Type": "text/html"},
                chunks=[b"<main>", b"x" * 50, b"</main>"],
            )
        ]
    )
    tiny_policy = _policy(max_response_bytes=32)

    with pytest.raises(fetcher.ResponseLimitError, match="exceeds 32 byte"):
        await fetcher.fetch_html(
            _source(), tiny_policy, declared, resolve_host=_public_resolver
        )
    with pytest.raises(fetcher.ResponseLimitError, match="exceeds 32 byte"):
        await fetcher.fetch_html(
            _source(), tiny_policy, streamed, resolve_host=_public_resolver
        )


@pytest.mark.asyncio
async def test_non_html_response_is_rejected():
    session = FakeSession(
        [
            FakeResponse(
                200,
                headers={"Content-Type": "application/octet-stream"},
                body=b"binary",
            )
        ]
    )

    with pytest.raises(fetcher.ResponseLimitError, match="content type"):
        await fetcher.fetch_html(
            _source(), _policy(), session, resolve_host=_public_resolver
        )


@pytest.mark.asyncio
async def test_fetch_writes_ingestible_markdown_with_complete_frontmatter(tmp_path):
    html = b"""<!doctype html><html><body>
    <nav>navigation must disappear</nav>
    <main>
      <h1>MVCC</h1>
      <p>Readers do not block <strong>writers</strong>.</p>
      <h2>Snapshots</h2>
      <ul><li>Transaction snapshot</li><li>Statement snapshot</li></ul>
      <pre><code class="language-sql">EXPLAIN SELECT 1;</code></pre>
    </main></body></html>"""
    session = FakeSession(
        [
            FakeResponse(
                200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                body=html,
            )
        ]
    )
    fetched_at = datetime(2026, 8, 3, 12, 34, 56, tzinfo=timezone.utc)

    result = await fetcher.fetch_source(
        _source(),
        _policy(),
        session,
        tmp_path,
        resolve_host=_public_resolver,
        fetched_at=fetched_at,
    )

    output = Path(result.output_path)
    raw = output.read_text(encoding="utf-8")
    _, header, body = raw.split("---", 2)
    metadata = yaml.safe_load(header)
    content = body.lstrip("\n")
    assert metadata == {
        "source_id": "postgresql_mvcc",
        "source_uri": "https://www.postgresql.org/docs/current/mvcc.html",
        "url": "https://www.postgresql.org/docs/current/mvcc.html",
        "resolved_url": "https://www.postgresql.org/docs/current/mvcc.html",
        "title": "PostgreSQL MVCC",
        "domains": ["db", "sys"],
        "product": "PostgreSQL",
        "version": "current",
        "license": "PostgreSQL License",
        "trust_level": "official",
        "fetched_at": "2026-08-03T12:34:56Z",
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    assert metadata["content_sha256"] == result.content_sha256
    assert result.changed is True
    assert "# MVCC" in content
    assert "## Snapshots" in content
    assert "**writers**" in content
    assert "```sql\nEXPLAIN SELECT 1;\n```" in content
    assert "navigation must disappear" not in content

    from src.rag.documents import load_document

    loaded = load_document(output, root=tmp_path, namespace="global:technical")
    assert loaded.metadata["url"] == _source().url
    assert loaded.metadata["source_uri"] == _source().url
    assert loaded.metadata["trust_level"] == "official"
    assert loaded.domains == ("sys", "db")
    assert loaded.content.startswith("# MVCC")


@pytest.mark.asyncio
async def test_unchanged_body_preserves_file_bytes_and_mtime(tmp_path):
    html = b"<html><main><h1>MVCC</h1><p>Stable body.</p></main></html>"
    response_headers = {"Content-Type": "text/html; charset=utf-8"}
    session = FakeSession(
        [
            FakeResponse(200, headers=response_headers, body=html),
            FakeResponse(200, headers=response_headers, body=html),
        ]
    )

    first = await fetcher.fetch_source(
        _source(),
        _policy(),
        session,
        tmp_path,
        resolve_host=_public_resolver,
        fetched_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )
    output = Path(first.output_path)
    os_timestamp = 1_700_000_000_000_000_000
    os.utime(output, ns=(os_timestamp, os_timestamp))
    first_bytes = output.read_bytes()
    first_mtime = output.stat().st_mtime_ns

    second = await fetcher.fetch_source(
        _source(),
        _policy(),
        session,
        tmp_path,
        resolve_host=_public_resolver,
        fetched_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert first.changed is True
    assert first.bytes_written > 0
    assert second.changed is False
    assert second.bytes_written == 0
    assert output.read_bytes() == first_bytes
    assert output.stat().st_mtime_ns == first_mtime
    assert b"2026-08-03T00:00:00Z" in first_bytes
    assert b"2027-01-01" not in first_bytes
    assert not fetcher.existing_document_matches(
        output,
        _source(title="Updated catalog title"),
        first.content_sha256,
        max_bytes=4096,
    )


def test_atomic_write_replaces_complete_file_and_leaves_no_temporary_files(tmp_path):
    destination = tmp_path / "source.md"
    destination.write_text("old", encoding="utf-8")

    byte_count = fetcher.atomic_write(destination, "new content\n")

    assert byte_count == len(b"new content\n")
    assert destination.read_text(encoding="utf-8") == "new content\n"
    assert list(tmp_path.glob(".source.md.*.tmp")) == []


def test_atomic_write_failure_preserves_previous_file_and_cleans_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "source.md"
    destination.write_text("previous complete document", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(fetcher.os, "replace", fail_replace)

    with pytest.raises(OSError, match="rename failure"):
        fetcher.atomic_write(destination, "partial replacement")

    assert destination.read_text(encoding="utf-8") == "previous complete document"
    assert list(tmp_path.glob(".source.md.*.tmp")) == []


@pytest.mark.asyncio
async def test_async_cli_result_is_nonzero_when_any_fetch_fails(tmp_path, monkeypatch):
    manifest = fetcher.SourceManifest(policy=_policy(), sources=(_source(),))

    async def failed_fetch(*_args, **_kwargs):
        return [], [{"source_id": "postgresql_mvcc", "error": "network failed"}]

    monkeypatch.setattr(fetcher, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr(fetcher, "fetch_selected_sources", failed_fetch)
    args = argparse.Namespace(
        all=True,
        source=None,
        manifest=tmp_path / "sources.toml",
        output=tmp_path / "output",
        json=True,
    )

    assert await fetcher._async_main(args) == 1
