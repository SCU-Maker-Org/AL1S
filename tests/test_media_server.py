import json
from datetime import datetime
from pathlib import Path

import pytest

from src.config import MediaConfig
from src.mcp_servers import media_server

CAPTURE_BINDING = {
    "al1s_capture_nonce": "n" * 43,
    "al1s_capture_owner": "a" * 64,
}
PNG_BYTES = b"\x89PNG\r\n\x1a\nqwen-image"
OPUS_BYTES = b"OggS" + (b"\0" * 24) + b"OpusHead" + b"qwen-audio"


class _FakeContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_chunked(self, _size):
        for chunk in self.chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks, headers=None, status=200):
        self.content = _FakeContent(chunks)
        self.headers = headers or {}
        self.status = status


class _FakeRequestContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args):
        return False


class _FakeSession:
    def __init__(self, responses, calls):
        self.responses = list(responses)
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeRequestContext(self.responses.pop(0))


@pytest.fixture
def media_environment(monkeypatch, tmp_path):
    output_root = tmp_path / "media"
    monkeypatch.setenv("AL1S_MEDIA_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("AL1S_MEDIA_DASHSCOPE_API_KEY", "test-secret-key")
    monkeypatch.setenv("AL1S_MEDIA_TTL_SECONDS", "3600")
    return output_root


def _parse_artifact(result: str, root: Path) -> tuple[dict, Path]:
    artifact = json.loads(result)
    path = Path(artifact["relative_path"])
    assert not path.is_absolute()
    destination = root / path
    assert destination.resolve().is_relative_to(root.resolve())
    return artifact, destination


@pytest.mark.asyncio
async def test_generate_image_calls_qwen_native_api_and_writes_artifact(
    monkeypatch, media_environment
):
    provider_calls = []
    download_calls = []

    async def fake_post(path, payload):
        provider_calls.append((path, payload))
        return {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "image": "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/image.png?sig=x"
                                }
                            ]
                        }
                    }
                ]
            }
        }

    async def fake_download(url, **kwargs):
        download_calls.append((url, kwargs))
        return PNG_BYTES

    monkeypatch.setattr(media_server, "_post_provider_json", fake_post)
    monkeypatch.setattr(media_server, "_download_remote_artifact", fake_download)

    first_result = await media_server.generate_image(
        "a storage engine diagram", **CAPTURE_BINDING
    )
    second_result = await media_server.generate_image(
        "a storage engine diagram", size="1024x1024", **CAPTURE_BINDING
    )
    first, first_path = _parse_artifact(first_result, media_environment)
    second, second_path = _parse_artifact(second_result, media_environment)

    assert first == {
        "artifact_id": first_path.stem,
        "relative_path": first["relative_path"],
        "kind": "image",
        "mime_type": "image/png",
        "size_bytes": len(PNG_BYTES),
        "sha256": media_server.hashlib.sha256(PNG_BYTES).hexdigest(),
        "expires_at": first["expires_at"],
        "capture_nonce": CAPTURE_BINDING["al1s_capture_nonce"],
        "owner_tag": CAPTURE_BINDING["al1s_capture_owner"],
    }
    assert first_path.read_bytes() == PNG_BYTES
    assert first_path.name != second_path.name
    assert second_path.read_bytes() == PNG_BYTES
    assert list(media_environment.rglob(".tmp-*")) == []
    assert datetime.fromisoformat(first["expires_at"].replace("Z", "+00:00"))
    assert "test-secret-key" not in first_result
    assert provider_calls[0] == (
        "services/aigc/multimodal-generation/generation",
        {
            "model": "qwen-image-2.0-pro",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "a storage engine diagram"}],
                    }
                ]
            },
            "parameters": {
                "size": "2048*2048",
                "n": 1,
                "prompt_extend": True,
                "watermark": False,
            },
        },
    )
    assert provider_calls[1][1]["parameters"]["size"] == "1024*1024"
    assert download_calls[0][0].startswith(
        "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/"
    )
    assert download_calls[0][1]["media_format"] == "png"


@pytest.mark.asyncio
async def test_synthesize_speech_calls_qwen_audio_and_writes_opus(
    monkeypatch, media_environment
):
    provider_calls = []
    download_calls = []

    async def fake_post(path, payload):
        provider_calls.append((path, payload))
        return {
            "output": {
                "audio": {
                    "url": "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/speech.opus?sig=x"
                }
            }
        }

    async def fake_download(url, **kwargs):
        download_calls.append((url, kwargs))
        return OPUS_BYTES

    monkeypatch.setattr(media_server, "_post_provider_json", fake_post)
    monkeypatch.setattr(media_server, "_download_remote_artifact", fake_download)

    result = await media_server.synthesize_speech(
        "Hello from AL1S",
        voice="longanlufeng",
        response_format="opus",
        speed=1.25,
        **CAPTURE_BINDING,
    )
    artifact, destination = _parse_artifact(result, media_environment)

    assert artifact["kind"] == "voice"
    assert artifact["mime_type"] == "audio/ogg"
    assert artifact["size_bytes"] == len(OPUS_BYTES)
    assert destination.suffix == ".ogg"
    assert destination.read_bytes() == OPUS_BYTES
    assert provider_calls == [
        (
            "services/audio/tts/SpeechSynthesizer",
            {
                "model": "qwen-audio-3.0-tts-plus",
                "input": {
                    "text": "Hello from AL1S",
                    "voice": "longanlufeng",
                    "format": "opus",
                    "sample_rate": 24_000,
                    "rate": 1.25,
                    "enable_aigc_tag": True,
                },
            },
        )
    ]
    assert download_calls[0][1]["media_format"] == "opus"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": " "}, "prompt must not be empty"),
        (
            {"prompt": "x" * (media_server.MAX_IMAGE_PROMPT_CHARS + 1)},
            "prompt exceeds",
        ),
        ({"prompt": "x", "size": "not-a-size"}, "invalid size"),
        ({"prompt": "x", "size": "511x512"}, "total pixels"),
        ({"prompt": "x", "size": "4096x2048"}, "total pixels"),
    ],
)
async def test_generate_image_rejects_invalid_input(monkeypatch, kwargs, message):
    monkeypatch.setattr(
        media_server,
        "_request_image",
        lambda **_kwargs: pytest.fail("invalid input must not call the provider"),
    )

    with pytest.raises(ValueError, match=message):
        await media_server.generate_image(**kwargs, **CAPTURE_BINDING)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"text": ""}, "text must not be empty"),
        (
            {"text": "x" * (media_server.MAX_SPEECH_INPUT_CHARS + 1)},
            "text exceeds",
        ),
        ({"text": "x", "voice": "unknown"}, "invalid voice"),
        ({"text": "x", "response_format": "wav"}, "invalid response_format"),
        ({"text": "x", "speed": 0.1}, "speed must be between"),
        ({"text": "x", "speed": True}, "speed must be a number"),
    ],
)
async def test_synthesize_speech_rejects_invalid_input(monkeypatch, kwargs, message):
    monkeypatch.setattr(
        media_server,
        "_request_speech",
        lambda **_kwargs: pytest.fail("invalid input must not call the provider"),
    )

    with pytest.raises(ValueError, match=message):
        await media_server.synthesize_speech(**kwargs, **CAPTURE_BINDING)


@pytest.mark.asyncio
async def test_media_tool_rejects_invalid_capture_binding_before_provider_call(
    monkeypatch, media_environment
):
    monkeypatch.setattr(
        media_server,
        "_request_image",
        lambda **_kwargs: pytest.fail("invalid binding must not call the provider"),
    )

    with pytest.raises(ValueError, match="invalid media capture nonce"):
        await media_server.generate_image(
            "diagram",
            al1s_capture_nonce="guessable",
            al1s_capture_owner=CAPTURE_BINDING["al1s_capture_owner"],
        )


def test_provider_settings_require_dashscope_credentials(monkeypatch):
    monkeypatch.delenv("AL1S_MEDIA_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="credentials are not configured"):
        media_server._provider_settings()


def test_artifact_byte_limit_honors_consumer_limit(monkeypatch):
    monkeypatch.setenv("AL1S_MEDIA_MAX_ARTIFACT_BYTES", "4096")
    assert media_server._artifact_byte_limit(media_server.MAX_SPEECH_BYTES) == 4096

    monkeypatch.setenv("AL1S_MEDIA_MAX_ARTIFACT_BYTES", "20000000")
    assert (
        media_server._artifact_byte_limit(media_server.MAX_IMAGE_BYTES)
        == media_server.MAX_IMAGE_BYTES
    )

    monkeypatch.setenv(
        "AL1S_MEDIA_MAX_ARTIFACT_BYTES", str(media_server.MAX_SPEECH_BYTES + 1)
    )
    with pytest.raises(RuntimeError, match="outside the allowed range"):
        media_server._artifact_byte_limit(media_server.MAX_IMAGE_BYTES)


def test_media_config_uses_qwen_defaults_and_dashscope_environment(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-secret")

    media = MediaConfig()

    assert media.api_key == "dashscope-secret"
    assert media.base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert media.image_model == "qwen-image-2.0-pro"
    assert media.speech_model == "qwen-audio-3.0-tts-plus"
    assert media.speech_voice == "longanlingxin"


def test_media_config_rejects_non_dashscope_base_url():
    with pytest.raises(ValueError, match="官方 DashScope"):
        MediaConfig(base_url="https://example.com/api/v1")


def test_media_config_rejects_cross_model_voice():
    with pytest.raises(ValueError, match="不适用于模型"):
        MediaConfig(
            speech_model="qwen-audio-3.0-tts-flash",
            speech_voice="longanlingxin",
        )


@pytest.mark.parametrize(
    "value",
    [
        "http://dashscope.aliyuncs.com/api/v1",
        "https://example.com/api/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://user:pass@dashscope.aliyuncs.com/api/v1",
    ],
)
def test_provider_base_url_rejects_unsafe_endpoints(value):
    with pytest.raises(RuntimeError, match="DashScope"):
        media_server._validate_provider_base_url(value)


def test_provider_base_url_accepts_legacy_and_workspace_endpoints():
    assert (
        media_server._validate_provider_base_url(
            "https://dashscope.aliyuncs.com/api/v1/"
        )
        == "https://dashscope.aliyuncs.com/api/v1"
    )
    assert (
        media_server._validate_provider_base_url(
            "https://ws-123.cn-beijing.maas.aliyuncs.com/api/v1"
        )
        == "https://ws-123.cn-beijing.maas.aliyuncs.com/api/v1"
    )


def test_artifact_url_upgrades_aliyun_http_and_rejects_unsafe_urls():
    assert (
        media_server._normalize_artifact_url(
            "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/a.opus?sig=secret"
        )
        == "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/a.opus?sig=secret"
    )
    accelerated_url = (
        "https://dashscope-7c2c.oss-accelerate.aliyuncs.com/a.png?sig=secret"
    )
    assert media_server._normalize_artifact_url(accelerated_url) == accelerated_url
    regional_url = (
        "https://provider-results.oss-ap-southeast-1.aliyuncs.com/a.png?sig=secret"
    )
    assert media_server._normalize_artifact_url(regional_url) == regional_url
    dual_stack_url = (
        "https://provider-results.cn-beijing.oss.aliyuncs.com/a.png?sig=secret"
    )
    assert media_server._normalize_artifact_url(dual_stack_url) == dual_stack_url

    for value in (
        "http://example.com/a.png",
        "https://127.0.0.1/a.png",
        "https://[::1]/a.png",
        "https://user:pass@example.com/a.png",
        "https://example.com:8443/a.png",
        "https://example.com/a.png#fragment",
        "https://public.example/a.png",
        "https://provider-results.oss-cn-beijing-internal.aliyuncs.com/a.png",
        "https://provider-results.cn-beijing-internal.oss.aliyuncs.com/a.png",
        "https://provider-results.oss-accelerate.example.com/a.png",
        "https://provider-results.oss-cn-beijing.aliyuncs.com.evil.test/a.png",
    ):
        with pytest.raises(RuntimeError):
            media_server._normalize_artifact_url(value)

    with pytest.raises(RuntimeError, match="non-public"):
        media_server._ensure_public_addresses(["127.0.0.1"], "localhost")
    with pytest.raises(RuntimeError, match="non-public"):
        media_server._ensure_public_addresses(["8.8.8.8", "127.0.0.1"], "mixed.example")


@pytest.mark.asyncio
async def test_speech_rejects_voice_from_another_qwen_model(
    monkeypatch, media_environment
):
    monkeypatch.setenv("AL1S_MEDIA_TTS_MODEL", "qwen-audio-3.0-tts-flash")

    with pytest.raises(ValueError, match="invalid voice for qwen-audio"):
        await media_server.synthesize_speech(
            "hello",
            voice="longanlingxin",
            **CAPTURE_BINDING,
        )


@pytest.mark.asyncio
async def test_limited_body_enforces_declared_and_streamed_size():
    with pytest.raises(RuntimeError, match="size limit"):
        await media_server._read_limited_body(
            _FakeResponse([b"small"], {"Content-Length": "100"}), 10
        )

    with pytest.raises(RuntimeError, match="size limit"):
        await media_server._read_limited_body(_FakeResponse([b"123456", b"789012"]), 10)


@pytest.mark.asyncio
async def test_artifact_download_revalidates_redirect_without_forwarding_key(
    monkeypatch,
):
    calls = []
    responses = [
        _FakeResponse(
            [],
            {
                "Location": "https://dashscope-7c2c.oss-accelerate.aliyuncs.com/final.opus"
            },
            status=302,
        ),
        _FakeResponse(
            [OPUS_BYTES],
            {
                "Content-Type": "audio/ogg",
                "Content-Length": str(len(OPUS_BYTES)),
            },
        ),
    ]
    fake_session = _FakeSession(responses, calls)
    monkeypatch.setattr(
        media_server.aiohttp, "TCPConnector", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        media_server.aiohttp,
        "ClientSession",
        lambda **_kwargs: fake_session,
    )

    content = await media_server._download_remote_artifact(
        "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/start.opus",
        media_format="opus",
        allowed_content_types=frozenset({"audio/ogg"}),
        max_bytes=4096,
    )

    assert content == OPUS_BYTES
    assert len(calls) == 2
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["headers"] == {"Accept": "*/*"}
    assert all("Authorization" not in call[1]["headers"] for call in calls)


@pytest.mark.asyncio
async def test_artifact_download_rejects_redirect_outside_result_buckets(monkeypatch):
    calls = []
    fake_session = _FakeSession(
        [
            _FakeResponse(
                [],
                {"Location": "https://public.example/final.png"},
                status=302,
            )
        ],
        calls,
    )
    monkeypatch.setattr(
        media_server.aiohttp, "TCPConnector", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        media_server.aiohttp,
        "ClientSession",
        lambda **_kwargs: fake_session,
    )

    with pytest.raises(RuntimeError, match="approved public OSS bucket"):
        await media_server._download_remote_artifact(
            "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/start.png",
            media_format="png",
            allowed_content_types=frozenset({"image/png"}),
            max_bytes=4096,
        )


@pytest.mark.parametrize(
    ("content", "media_format"),
    [(PNG_BYTES, "png"), (OPUS_BYTES, "opus"), (b"ID3qwen-audio", "mp3")],
)
def test_artifact_signature_accepts_expected_formats(content, media_format):
    media_server._validate_artifact_signature(content, media_format)


def test_artifact_signature_rejects_spoofed_content():
    with pytest.raises(RuntimeError, match="invalid artifact format"):
        media_server._validate_artifact_signature(b"<html>failure</html>", "png")


def test_output_root_must_be_configured(monkeypatch):
    monkeypatch.delenv("AL1S_MEDIA_OUTPUT_DIR", raising=False)

    with pytest.raises(RuntimeError, match="AL1S_MEDIA_OUTPUT_DIR"):
        media_server._output_root()


def test_artifact_rejects_empty_and_oversized_content(media_environment):
    with pytest.raises(RuntimeError, match="empty artifact"):
        media_server._write_artifact(
            b"",
            kind="image",
            suffix=".png",
            mime_type="image/png",
            max_bytes=10,
            capture_nonce=CAPTURE_BINDING["al1s_capture_nonce"],
            owner_tag=CAPTURE_BINDING["al1s_capture_owner"],
        )

    with pytest.raises(RuntimeError, match="size limit"):
        media_server._write_artifact(
            b"too large",
            kind="image",
            suffix=".png",
            mime_type="image/png",
            max_bytes=3,
            capture_nonce=CAPTURE_BINDING["al1s_capture_nonce"],
            owner_tag=CAPTURE_BINDING["al1s_capture_owner"],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "request_name", "message"),
    [
        ("generate_image", "_request_image", "image generation failed"),
        ("synthesize_speech", "_request_speech", "speech synthesis failed"),
    ],
)
async def test_provider_errors_do_not_leak_secrets(
    monkeypatch, media_environment, tool, request_name, message
):
    async def fail(**_kwargs):
        raise RuntimeError("test-secret-key")

    monkeypatch.setattr(media_server, request_name, fail)

    with pytest.raises(RuntimeError, match=f"^{message}$") as error:
        await getattr(media_server, tool)(
            "prompt" if tool == "generate_image" else "speech",
            **CAPTURE_BINDING,
        )

    assert "test-secret-key" not in str(error.value)


def test_invalid_ttl_does_not_leave_an_artifact(monkeypatch, media_environment):
    monkeypatch.setenv("AL1S_MEDIA_TTL_SECONDS", "invalid")

    with pytest.raises(RuntimeError, match="invalid media artifact TTL"):
        media_server._write_artifact(
            PNG_BYTES,
            kind="image",
            suffix=".png",
            mime_type="image/png",
            max_bytes=len(PNG_BYTES) + 1,
            capture_nonce=CAPTURE_BINDING["al1s_capture_nonce"],
            owner_tag=CAPTURE_BINDING["al1s_capture_owner"],
        )

    assert list(media_environment.rglob("*.*")) == []
