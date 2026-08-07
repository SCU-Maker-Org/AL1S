from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.handlers.chat_handler import ChatHandler
from src.infra.mcp import MCPService
from src.services.conversation_service import ConversationService

TEST_NONCE = "n" * 43
TEST_OWNER_TAG = "a" * 64


def _artifact_payload(
    outbox: Path,
    relative_path: str,
    content: bytes,
    *,
    artifact_id: str = "artifact-1",
    kind: str = "photo",
    mime_type: str = "image/png",
    expires_at: float | None = None,
    capture_nonce: str = TEST_NONCE,
    owner_tag: str = TEST_OWNER_TAG,
) -> dict[str, object]:
    bound_relative_path = (
        Path("requests") / owner_tag / capture_nonce / relative_path
    ).as_posix()
    path = outbox / bound_relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "relative_path": bound_relative_path,
        "mime_type": mime_type,
        "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "expires_at": expires_at or time.time() + 300,
        "capture_nonce": capture_nonce,
        "owner_tag": owner_tag,
        "caption": "generated media",
    }


def _validate(service: MCPService, payload: dict[str, object]):
    return service.validate_media_artifact(
        payload,
        expected_capture_nonce=str(payload["capture_nonce"]),
        expected_owner_tag=str(payload["owner_tag"]),
    )


def _capture_artifact(
    service: MCPService,
    outbox: Path,
    *,
    owner: str,
    relative_path: str = "image/result.png",
    content: bytes = b"png-content",
    kind: str = "photo",
    mime_type: str = "image/png",
):
    token = service.begin_media_capture(owner)
    capture = service._media_capture.get()
    payload = _artifact_payload(
        outbox,
        relative_path,
        content,
        kind=kind,
        mime_type=mime_type,
        capture_nonce=capture.nonce,
        owner_tag=capture.owner_tag,
    )
    service._capture_media_result(
        SimpleNamespace(structuredContent={"al1s_media": payload}),
        [],
        server_name="media",
    )
    return service.finish_media_capture(token)[0]


def test_media_json_is_captured_from_mcp_text_result(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    result = SimpleNamespace(structuredContent=None)
    owner = "telegram:10:10:1:1"
    token = service.begin_media_capture(owner)
    capture = service._media_capture.get()
    payload = _artifact_payload(
        outbox,
        "image/result.png",
        b"png-content",
        capture_nonce=capture.nonce,
        owner_tag=capture.owner_tag,
    )
    text = json.dumps({"result": {"al1s_media": payload}})

    response = service._capture_media_result(result, [text], server_name="media")
    artifacts = service.finish_media_capture(token)

    assert json.loads(response) == {
        "status": "media_ready",
        "artifact_id": "artifact-1",
        "kind": "photo",
        "byte_size": len(b"png-content"),
    }
    assert len(artifacts) == 1
    assert artifacts[0].relative_path.endswith("/image/result.png")
    assert (
        service.media_artifact_path(artifacts[0], owner=owner)
        == (outbox / artifacts[0].relative_path).resolve()
    )


def test_media_server_wire_format_is_normalized_for_telegram(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    token = service.begin_media_capture("telegram:10:10:2:2")
    capture = service._media_capture.get()
    payload = _artifact_payload(
        outbox,
        "image/result.png",
        b"png-content",
        capture_nonce=capture.nonce,
        owner_tag=capture.owner_tag,
    )
    payload["kind"] = "image"
    payload["size_bytes"] = payload.pop("byte_size")
    payload["expires_at"] = (
        (datetime.now(timezone.utc) + timedelta(minutes=5))
        .isoformat()
        .replace("+00:00", "Z")
    )
    response = service._capture_media_result(
        SimpleNamespace(structuredContent=None),
        [json.dumps(payload)],
        server_name="media",
    )
    artifacts = service.finish_media_capture(token)

    assert json.loads(response)["kind"] == "photo"
    assert artifacts[0].kind == "photo"
    assert artifacts[0].byte_size == len(b"png-content")


def test_media_validation_accepts_a_safe_relative_path(tmp_path):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(
        outbox,
        "voice/answer.ogg",
        b"ogg-content",
        kind="voice",
        mime_type="audio/ogg",
    )
    service = MCPService(media_output_dir=str(outbox))

    artifact = _validate(service, payload)

    assert artifact.kind == "voice"
    assert artifact.relative_path.endswith("/voice/answer.ogg")
    assert artifact.caption == "generated media"


def test_voice_artifact_gets_ai_disclosure_when_caption_is_empty(tmp_path):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(
        outbox,
        "voice/answer.ogg",
        b"ogg-content",
        kind="voice",
        mime_type="audio/ogg",
    )
    payload["caption"] = ""

    artifact = _validate(MCPService(media_output_dir=str(outbox)), payload)

    assert artifact.caption == "AI 生成语音"


def test_media_validation_rejects_path_traversal(tmp_path):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(outbox, "image/valid.png", b"content")
    payload["relative_path"] = "../valid.png"
    service = MCPService(media_output_dir=str(outbox))

    with pytest.raises(ValueError, match="安全的相对路径"):
        _validate(service, payload)


def test_media_validation_rejects_hash_mismatch(tmp_path):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(outbox, "image/result.png", b"content")
    payload["sha256"] = "0" * 64
    service = MCPService(media_output_dir=str(outbox))

    with pytest.raises(ValueError, match="哈希校验失败"):
        _validate(service, payload)


def test_media_validation_rejects_expired_artifact(tmp_path):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(
        outbox,
        "image/result.png",
        b"content",
        expires_at=time.time() - 1,
    )
    service = MCPService(media_output_dir=str(outbox))

    with pytest.raises(ValueError, match="已经过期"):
        _validate(service, payload)


def test_media_validation_rejects_oversized_artifact(tmp_path):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(outbox, "image/result.png", b"too-large")
    service = MCPService(media_output_dir=str(outbox), max_media_bytes=4)

    with pytest.raises(ValueError, match="超过上限"):
        _validate(service, payload)


@pytest.mark.parametrize("expires_at", [float("nan"), float("inf")])
def test_media_validation_rejects_non_finite_expiry(tmp_path, expires_at):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(
        outbox, "image/result.png", b"content", expires_at=expires_at
    )

    with pytest.raises(ValueError, match="过期时间格式无效"):
        _validate(MCPService(media_output_dir=str(outbox)), payload)


def test_media_validation_rejects_expiry_beyond_configured_ttl(tmp_path):
    outbox = tmp_path / "outbox"
    payload = _artifact_payload(
        outbox, "image/result.png", b"content", expires_at=time.time() + 301
    )
    service = MCPService(media_output_dir=str(outbox), max_media_ttl_seconds=300)

    with pytest.raises(ValueError, match="超过配置上限"):
        _validate(service, payload)


def test_non_media_server_cannot_forge_telegram_artifact(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    token = service.begin_media_capture("telegram:10:10:3:3")
    capture = service._media_capture.get()
    payload = _artifact_payload(
        outbox,
        "image/forged.png",
        b"forged",
        capture_nonce=capture.nonce,
        owner_tag=capture.owner_tag,
    )

    response = service._capture_media_result(
        SimpleNamespace(structuredContent={"al1s_media": payload}),
        [],
        server_name="filesystem",
    )
    artifacts = service.finish_media_capture(token)

    assert response is None
    assert artifacts == []


def test_artifact_cannot_be_replayed_by_another_update_owner(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    artifact = _capture_artifact(service, outbox, owner="telegram:user-one")

    with pytest.raises(ValueError, match="不属于当前调用方"):
        with service.consume_media_artifact(artifact, owner="telegram:user-two"):
            pytest.fail("cross-user artifact must not be opened")

    assert (outbox / artifact.relative_path).exists()
    service.cleanup_media_artifacts([artifact], owner="telegram:user-one")
    assert not (outbox / artifact.relative_path).exists()


def test_consumed_artifact_is_deleted_after_send_attempt(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    owner = "telegram:10:10:4:4"
    artifact = _capture_artifact(service, outbox, owner=owner)
    artifact_path = outbox / artifact.relative_path

    with service.consume_media_artifact(artifact, owner=owner) as media_file:
        assert media_file.read() == b"png-content"
        assert artifact_path.exists()

    assert not artifact_path.exists()


def test_parent_symlink_swap_is_rejected_without_reading_outside_file(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    owner = "telegram:10:10:5:5"
    artifact = _capture_artifact(
        service,
        outbox,
        owner=owner,
        relative_path="image/result.png",
    )
    original_parent = (outbox / artifact.relative_path).parent
    original_parent.rename(original_parent.with_name("original-image"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.png").write_bytes(b"png-content")
    original_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="无法安全打开"):
        with service.consume_media_artifact(artifact, owner=owner):
            pytest.fail("symlink-swapped parent must not be traversed")

    assert (outside / "result.png").read_bytes() == b"png-content"


@pytest.mark.asyncio
async def test_chat_handler_sends_media_with_timeouts_and_removes_artifacts(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    owner = "telegram:10:10:6:6"
    photo_artifact = _capture_artifact(service, outbox, owner=owner)
    voice_artifact = _capture_artifact(
        service,
        outbox,
        owner=owner,
        relative_path="voice/result.ogg",
        content=b"OggS-voice-content",
        kind="voice",
        mime_type="audio/ogg",
    )
    sent = []

    class Bot:
        async def send_photo(self, **kwargs):
            sent.append(
                (
                    "photo",
                    kwargs["photo"].read(),
                    kwargs["read_timeout"],
                    kwargs["write_timeout"],
                )
            )

        async def send_voice(self, **kwargs):
            sent.append(
                (
                    "voice",
                    kwargs["voice"].read(),
                    kwargs["read_timeout"],
                    kwargs["write_timeout"],
                )
            )

        async def send_message(self, **kwargs):
            pytest.fail(f"unexpected send failure fallback: {kwargs}")

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=10),
        effective_message=SimpleNamespace(message_id=6, message_thread_id=None),
    )
    handler = ChatHandler(
        agent_service=None,
        conversation_service=ConversationService(),
        mcp_service=service,
    )

    await handler._send_media_artifacts(
        update,
        SimpleNamespace(bot=Bot()),
        [photo_artifact, voice_artifact],
        owner=owner,
    )

    assert sent == [
        ("photo", b"png-content", 60.0, 120.0),
        ("voice", b"OggS-voice-content", 60.0, 120.0),
    ]
    assert not (outbox / photo_artifact.relative_path).exists()
    assert not (outbox / voice_artifact.relative_path).exists()


@pytest.mark.asyncio
async def test_media_capture_context_is_isolated_between_async_tasks(tmp_path):
    outbox = tmp_path / "outbox"
    service = MCPService(media_output_dir=str(outbox))
    first_ready = asyncio.Event()
    second_ready = asyncio.Event()

    async def capture(
        relative_path, content, artifact_id, owner, own_ready, other_ready
    ):
        token = service.begin_media_capture(owner)
        state = service._media_capture.get()
        payload = _artifact_payload(
            outbox,
            relative_path,
            content,
            artifact_id=artifact_id,
            capture_nonce=state.nonce,
            owner_tag=state.owner_tag,
        )
        service._capture_media_result(
            SimpleNamespace(structuredContent={"al1s_media": payload}),
            [],
            server_name="media",
        )
        own_ready.set()
        await other_ready.wait()
        return service.finish_media_capture(token)

    first, second = await asyncio.gather(
        capture(
            "image/first.png",
            b"first",
            "first",
            "owner:first",
            first_ready,
            second_ready,
        ),
        capture(
            "image/second.png",
            b"second",
            "second",
            "owner:second",
            second_ready,
            first_ready,
        ),
    )

    assert [artifact.artifact_id for artifact in first] == ["first"]
    assert [artifact.artifact_id for artifact in second] == ["second"]
