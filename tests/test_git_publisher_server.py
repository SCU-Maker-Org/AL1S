from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

from src.mcp_servers import git_publisher_server as publisher

TOKEN = "test-secret-value-that-must-not-leak"
HEAD = "1" * 40
OTHER_HEAD = "2" * 40


@pytest.fixture
def publisher_environment(monkeypatch, tmp_path):
    root = tmp_path / "workspaces"
    monkeypatch.setenv("AL1S_DEV_WORKSPACE_ROOT", str(root))
    monkeypatch.setenv("AL1S_DEV_GITHUB_TOKEN", TOKEN)
    monkeypatch.setenv("AL1S_DEV_ALLOWED_GITHUB_OWNERS", "TexasOct, approved-org")
    monkeypatch.setenv("AL1S_DEV_BRANCH_PREFIX", "al1s/")
    monkeypatch.setenv("AL1S_DEV_MAX_WORKSPACES", "10")
    monkeypatch.setenv("AL1S_DEV_GIT_TIMEOUT", "17")
    monkeypatch.setenv("AL1S_DEV_MAX_OUTPUT_CHARS", "4096")
    return root


def _make_repository(root: Path, workspace: str = "project") -> Path:
    repository = root / workspace
    (repository / ".git").mkdir(parents=True)
    return repository


def _install_git_mock(monkeypatch, calls, *, head=HEAD, push_output="ok"):
    askpass_paths = []

    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        askpass = Path(environment["GIT_ASKPASS"])
        askpass_paths.append(askpass)
        assert askpass.exists()
        assert stat.S_IMODE(askpass.stat().st_mode) == 0o700
        assert TOKEN not in askpass.read_text(encoding="utf-8")
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT
        assert kwargs["timeout"] == 17
        assert kwargs["check"] is False
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_CONFIG_GLOBAL"] == publisher.os.devnull
        assert environment["GIT_CONFIG_SYSTEM"] == publisher.os.devnull
        assert environment["GIT_CONFIG_VALUE_0"].endswith("/hooks")
        assert environment["GIT_CONFIG_VALUE_1"] == ""
        config_keys = {
            environment[f"GIT_CONFIG_KEY_{index}"]
            for index in range(int(environment["GIT_CONFIG_COUNT"]))
        }
        assert "credential.interactive" not in config_keys
        assert environment["GIT_TERMINAL_PROMPT"] == "0"
        assert TOKEN not in "\0".join(command)
        calls.append((list(command), dict(kwargs)))

        if "clone" in command:
            target = Path(command[-1])
            (target / ".git").mkdir(parents=True)
            return subprocess.CompletedProcess(command, 0, stdout="")
        if "config" in command:
            return subprocess.CompletedProcess(command, 1, stdout="")
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=f"{head}\n")
        if "push" in command:
            return subprocess.CompletedProcess(command, 0, stdout=push_output)
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    return askpass_paths


def test_clone_uses_shallow_https_github_and_ephemeral_askpass(
    monkeypatch, publisher_environment
):
    calls = []
    askpass_paths = _install_git_mock(monkeypatch, calls)

    result = publisher.workspace_clone_github(
        "TexasOct", "AL1S", "clone-one", ref="feature/safe"
    )

    assert result == {
        "workspace": "clone-one",
        "repository": "TexasOct/AL1S",
        "ref": "feature/safe",
        "head": HEAD,
    }
    clone_command, clone_call = calls[0]
    assert clone_command[1:7] == [
        "clone",
        "--quiet",
        "--depth=1",
        "--single-branch",
        "--no-tags",
        "--no-recurse-submodules",
    ]
    assert clone_command[-5:] == [
        "--branch",
        "feature/safe",
        "--",
        "https://github.com/TexasOct/AL1S.git",
        str(publisher_environment / "clone-one"),
    ]
    assert clone_call["env"]["AL1S_GIT_ASKPASS_TOKEN"] == TOKEN
    assert "AL1S_GIT_ASKPASS_TOKEN" not in calls[1][1]["env"]
    assert all(not path.exists() for path in askpass_paths)


def test_push_checks_head_and_uses_explicit_url_without_remote(
    monkeypatch, publisher_environment
):
    repository = _make_repository(publisher_environment)
    calls = []
    askpass_paths = _install_git_mock(monkeypatch, calls)

    result = publisher.workspace_push_github(
        "project", "approved-org", "service", "al1s/add-feature", HEAD.upper()
    )

    assert result == {
        "workspace": "project",
        "repository": "approved-org/service",
        "branch": "al1s/add-feature",
        "head": HEAD,
        "pushed": True,
    }
    assert calls[0][0][1:] == [
        "-C",
        str(repository.resolve()),
        "config",
        "--local",
        "--no-includes",
        "--name-only",
        "--get-regexp",
        publisher.UNSAFE_LOCAL_CONFIG_PATTERN,
    ]
    assert calls[1][0][1:] == [
        "-C",
        str(repository.resolve()),
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ]
    push_command, push_call = calls[2]
    assert push_command[1:] == [
        "-C",
        str(repository.resolve()),
        "push",
        "--porcelain",
        "--no-verify",
        "--recurse-submodules=no",
        "--",
        "https://github.com/approved-org/service.git",
        "HEAD:refs/heads/al1s/add-feature",
    ]
    assert "remote" not in push_command
    assert "--force" not in push_command
    assert push_call["env"]["AL1S_GIT_ASKPASS_TOKEN"] == TOKEN
    assert "AL1S_GIT_ASKPASS_TOKEN" not in calls[0][1]["env"]
    assert "AL1S_GIT_ASKPASS_TOKEN" not in calls[1][1]["env"]
    assert all(not path.exists() for path in askpass_paths)


@pytest.mark.parametrize(
    "owner",
    ["not-allowed", "bad/owner", "-leading", "double--hyphen", "owner.git/x"],
)
def test_clone_rejects_invalid_or_unapproved_owner(
    monkeypatch, publisher_environment, owner
):
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid owner must not invoke git"),
    )

    with pytest.raises(ValueError, match="owner|allowed"):
        publisher.workspace_clone_github(owner, "repo", "workspace")


@pytest.mark.parametrize(
    "repository",
    [
        "../repo",
        "repo/name",
        "repo.git",
        "repo..name",
        "repo name",
        "-option",
    ],
)
def test_clone_rejects_invalid_repository(
    monkeypatch, publisher_environment, repository
):
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid repo must not invoke git"),
    )

    with pytest.raises(ValueError, match="repository"):
        publisher.workspace_clone_github("TexasOct", repository, "workspace")


@pytest.mark.parametrize(
    "workspace",
    [
        "../escape",
        "nested/repo",
        "/absolute",
        "workspace..escape",
        "two words",
        "-option",
    ],
)
def test_workspace_name_cannot_escape_root(
    monkeypatch, publisher_environment, workspace
):
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid path must not invoke git"),
    )

    with pytest.raises(ValueError, match="workspace"):
        publisher.workspace_clone_github("TexasOct", "repo", workspace)


@pytest.mark.parametrize(
    "ref",
    [
        "-branch",
        "refs/heads/main",
        "feature/../main",
        "feature//x",
        "feature/-option",
        ".hidden",
    ],
)
def test_clone_rejects_unsafe_ref(monkeypatch, publisher_environment, ref):
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid ref must not invoke git"),
    )

    with pytest.raises(ValueError, match="ref"):
        publisher.workspace_clone_github("TexasOct", "repo", "workspace", ref)


def test_clone_requires_a_new_target(monkeypatch, publisher_environment):
    (publisher_environment / "existing").mkdir(parents=True)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("existing target must not invoke git"),
    )

    with pytest.raises(ValueError, match="already exists"):
        publisher.workspace_clone_github("TexasOct", "repo", "existing")


def test_clone_enforces_workspace_limit(monkeypatch, publisher_environment):
    (publisher_environment / "existing").mkdir(parents=True)
    monkeypatch.setenv("AL1S_DEV_MAX_WORKSPACES", "1")
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("workspace limit must be checked first"),
    )

    with pytest.raises(RuntimeError, match="count limit"):
        publisher.workspace_clone_github("TexasOct", "repo", "second")


def test_failed_clone_removes_only_its_partial_workspace(
    monkeypatch, publisher_environment
):
    keep = publisher_environment / "keep"
    keep.mkdir(parents=True)

    def failed_clone(command, **kwargs):
        target = Path(command[-1])
        target.mkdir()
        (target / "partial").write_text("incomplete", encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, stdout="clone failed")

    monkeypatch.setattr(publisher.subprocess, "run", failed_clone)

    with pytest.raises(RuntimeError, match="clone failed"):
        publisher.workspace_clone_github("TexasOct", "repo", "partial")

    assert keep.is_dir()
    assert not (publisher_environment / "partial").exists()


@pytest.mark.parametrize("branch", ["main", "master", "other/topic", "al1s/bad..ref"])
def test_push_rejects_protected_or_unsafe_branch(
    monkeypatch, publisher_environment, branch
):
    _make_repository(publisher_environment)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid branch must not invoke git"),
    )

    with pytest.raises(ValueError, match="main|master|prefix|branch"):
        publisher.workspace_push_github(
            "project", "TexasOct", "repo", branch, expected_head=HEAD
        )


def test_push_rejects_stale_expected_head_before_network(
    monkeypatch, publisher_environment
):
    _make_repository(publisher_environment)
    calls = []
    _install_git_mock(monkeypatch, calls, head=OTHER_HEAD)

    with pytest.raises(ValueError, match="does not match"):
        publisher.workspace_push_github(
            "project", "TexasOct", "repo", "al1s/topic", expected_head=HEAD
        )

    assert len(calls) == 2
    assert "config" in calls[0][0]
    assert "rev-parse" in calls[1][0]
    assert "AL1S_GIT_ASKPASS_TOKEN" not in calls[0][1]["env"]
    assert "AL1S_GIT_ASKPASS_TOKEN" not in calls[1][1]["env"]


def test_push_rejects_local_url_rewrites_before_credentials(
    monkeypatch, publisher_environment
):
    _make_repository(publisher_environment)
    calls = []

    def unsafe_config(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        assert "AL1S_GIT_ASKPASS_TOKEN" not in kwargs["env"]
        if "config" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="url.https://evil.example/.insteadof\n"
            )
        raise AssertionError("unsafe config must be rejected before other Git calls")

    monkeypatch.setattr(publisher.subprocess, "run", unsafe_config)

    with pytest.raises(ValueError, match="forbidden network"):
        publisher.workspace_push_github(
            "project", "TexasOct", "repo", "al1s/topic", expected_head=HEAD
        )

    assert len(calls) == 1


@pytest.mark.parametrize("expected_head", ["", "abc123", "g" * 40, HEAD + " "])
def test_push_requires_full_expected_head(
    monkeypatch, publisher_environment, expected_head
):
    _make_repository(publisher_environment)
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid HEAD must not invoke git"),
    )

    with pytest.raises(ValueError, match="expected_head"):
        publisher.workspace_push_github(
            "project", "TexasOct", "repo", "al1s/topic", expected_head
        )


def test_push_rejects_workspace_and_git_symlinks(
    monkeypatch, publisher_environment, tmp_path
):
    outside = tmp_path / "outside"
    _make_repository(outside)
    publisher_environment.mkdir(parents=True)
    (publisher_environment / "linked").symlink_to(
        outside / "project", target_is_directory=True
    )
    repository = publisher_environment / "git-linked"
    repository.mkdir()
    (repository / ".git").symlink_to(
        outside / "project" / ".git", target_is_directory=True
    )
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("symlink must not invoke git"),
    )

    with pytest.raises(ValueError, match="real directory"):
        publisher.workspace_push_github(
            "linked", "TexasOct", "repo", "al1s/topic", HEAD
        )
    with pytest.raises(ValueError, match="Git repository"):
        publisher.workspace_push_github(
            "git-linked", "TexasOct", "repo", "al1s/topic", HEAD
        )


def test_credentials_are_redacted_from_git_errors_and_tempfiles_removed(
    monkeypatch, publisher_environment
):
    askpass_paths = []

    def failed_run(command, **kwargs):
        askpass_paths.append(Path(kwargs["env"]["GIT_ASKPASS"]))
        assert TOKEN not in "\0".join(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=f"authentication failed for {TOKEN}",
        )

    monkeypatch.setattr(publisher.subprocess, "run", failed_run)

    with pytest.raises(RuntimeError) as error:
        publisher.workspace_clone_github("TexasOct", "repo", "workspace")

    assert TOKEN not in str(error.value)
    assert "credential redacted" in str(error.value)
    assert all(not path.exists() for path in askpass_paths)


def test_git_timeout_is_generic_and_removes_askpass(monkeypatch, publisher_environment):
    askpass_paths = []

    def timed_out(command, **kwargs):
        askpass_paths.append(Path(kwargs["env"]["GIT_ASKPASS"]))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=TOKEN)

    monkeypatch.setattr(publisher.subprocess, "run", timed_out)

    with pytest.raises(RuntimeError, match="timed out") as error:
        publisher.workspace_clone_github("TexasOct", "repo", "workspace")

    assert TOKEN not in str(error.value)
    assert all(not path.exists() for path in askpass_paths)


def test_output_is_redacted_and_limited(monkeypatch, publisher_environment):
    monkeypatch.setenv("AL1S_DEV_MAX_OUTPUT_CHARS", "256")
    output = publisher._bounded_output(TOKEN + ("x" * 1000), TOKEN)

    assert TOKEN not in output
    assert len(output) <= 256
    assert output.endswith("[output truncated]")


def test_workspace_root_must_be_absolute(monkeypatch):
    monkeypatch.setenv("AL1S_DEV_WORKSPACE_ROOT", "relative/workspaces")

    with pytest.raises(RuntimeError, match="must be absolute"):
        publisher._workspace_root()
