import base64
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mcp_servers import media_server

CAPTURE_BINDING = {
    "al1s_capture_nonce": "n" * 43,
    "al1s_capture_owner": "a" * 64,
}


class _FakeImages:
    def __init__(self, content: bytes):
        self.content = content
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        encoded = base64.b64encode(self.content).decode("ascii")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded)])


class _FakeSpeech:
    def __init__(self, content: bytes):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content)


class _FakeClient:
    def __init__(self, *, image: bytes = b"image", speech: bytes = b"speech"):
        self.images = _FakeImages(image)
        self.audio = SimpleNamespace(speech=_FakeSpeech(speech))


@pytest.fixture
def media_environment(monkeypatch, tmp_path):
    output_root = tmp_path / "media"
    monkeypatch.setenv("AL1S_MEDIA_OUTPUT_DIR", str(output_root))
    monkeypatch.setenv("AL1S_MEDIA_OPENAI_API_KEY", "test-secret-key")
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
async def test_generate_image_writes_random_atomic_artifact(
    monkeypatch, media_environment
):
    fake = _FakeClient(image=b"generated-image")
    monkeypatch.setattr(media_server, "_create_openai_client", lambda: fake)

    first_result = await media_server.generate_image(
        "a storage engine diagram", **CAPTURE_BINDING
    )
    second_result = await media_server.generate_image(
        "a storage engine diagram", **CAPTURE_BINDING
    )
    first, first_path = _parse_artifact(first_result, media_environment)
    second, second_path = _parse_artifact(second_result, media_environment)

    assert first == {
        "artifact_id": first_path.stem,
        "relative_path": first["relative_path"],
        "kind": "image",
        "mime_type": "image/png",
        "size_bytes": len(b"generated-image"),
        "sha256": media_server.hashlib.sha256(b"generated-image").hexdigest(),
        "expires_at": first["expires_at"],
        "capture_nonce": CAPTURE_BINDING["al1s_capture_nonce"],
        "owner_tag": CAPTURE_BINDING["al1s_capture_owner"],
    }
    assert first_path.read_bytes() == b"generated-image"
    assert first_path.name != second_path.name
    assert second_path.read_bytes() == b"generated-image"
    assert list(media_environment.rglob(".tmp-*")) == []
    assert datetime.fromisoformat(first["expires_at"].replace("Z", "+00:00"))
    assert "test-secret-key" not in first_result
    assert base64.b64encode(b"generated-image").decode("ascii") not in first_result
    assert fake.images.calls[0]["response_format"] == "b64_json"
    assert fake.images.calls[0]["model"] == "gpt-image-2"


@pytest.mark.asyncio
async def test_synthesize_speech_writes_opus_artifact(monkeypatch, media_environment):
    fake = _FakeClient(speech=b"generated-speech")
    monkeypatch.setattr(media_server, "_create_openai_client", lambda: fake)

    result = await media_server.synthesize_speech(
        "Hello from AL1S",
        voice="coral",
        response_format="opus",
        speed=1.25,
        **CAPTURE_BINDING,
    )
    artifact, destination = _parse_artifact(result, media_environment)

    assert artifact["kind"] == "voice"
    assert artifact["mime_type"] == "audio/ogg"
    assert artifact["size_bytes"] == len(b"generated-speech")
    assert destination.suffix == ".ogg"
    assert destination.read_bytes() == b"generated-speech"
    assert fake.audio.speech.calls == [
        {
            "model": "tts-1",
            "input": "Hello from AL1S",
            "voice": "coral",
            "response_format": "opus",
            "speed": 1.25,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": " "}, "prompt must not be empty"),
        (
            {"prompt": "x" * (media_server.MAX_IMAGE_PROMPT_CHARS + 1)},
            "prompt exceeds",
        ),
        ({"prompt": "x", "size": "2048x2048"}, "invalid size"),
        ({"prompt": "x", "quality": "ultra"}, "invalid quality"),
        ({"prompt": "x", "output_format": "svg"}, "invalid output_format"),
    ],
)
async def test_generate_image_rejects_invalid_input(monkeypatch, kwargs, message):
    monkeypatch.setattr(
        media_server,
        "_create_openai_client",
        lambda: pytest.fail("invalid input must not create a provider client"),
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
        "_create_openai_client",
        lambda: pytest.fail("invalid input must not create a provider client"),
    )

    with pytest.raises(ValueError, match=message):
        await media_server.synthesize_speech(**kwargs, **CAPTURE_BINDING)


@pytest.mark.asyncio
async def test_media_tool_rejects_invalid_capture_binding_before_provider_call(
    monkeypatch, media_environment
):
    monkeypatch.setattr(
        media_server,
        "_create_openai_client",
        lambda: pytest.fail("invalid binding must not create a provider client"),
    )

    with pytest.raises(ValueError, match="invalid media capture nonce"):
        await media_server.generate_image(
            "diagram",
            al1s_capture_nonce="guessable",
            al1s_capture_owner=CAPTURE_BINDING["al1s_capture_owner"],
        )


def test_client_factory_requires_credentials(monkeypatch):
    monkeypatch.delenv("AL1S_MEDIA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="credentials are not configured"):
        media_server._create_openai_client()


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
    ("tool", "provider_attribute", "message"),
    [
        ("generate_image", "images", "image generation failed"),
        ("synthesize_speech", "audio", "speech synthesis failed"),
    ],
)
async def test_provider_errors_do_not_leak_secrets(
    monkeypatch, media_environment, tool, provider_attribute, message
):
    class FailingProvider:
        def __getattr__(self, name):
            raise RuntimeError("test-secret-key")

    client = SimpleNamespace(**{provider_attribute: FailingProvider()})
    monkeypatch.setattr(media_server, "_create_openai_client", lambda: client)

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
            b"image",
            kind="image",
            suffix=".png",
            mime_type="image/png",
            max_bytes=10,
            capture_nonce=CAPTURE_BINDING["al1s_capture_nonce"],
            owner_tag=CAPTURE_BINDING["al1s_capture_owner"],
        )

    assert list(media_environment.rglob("*.*")) == []
