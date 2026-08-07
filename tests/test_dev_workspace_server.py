from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.mcp_servers import dev_workspace_server as server


@pytest.fixture
def workspace_environment(monkeypatch, tmp_path):
    root = tmp_path / "workspaces"
    monkeypatch.setenv("AL1S_DEV_WORKSPACE_ROOT", str(root))
    monkeypatch.setenv("AL1S_DEV_MAX_WORKSPACES", "10")
    monkeypatch.setenv("AL1S_DEV_MAX_FILE_BYTES", "1000000")
    monkeypatch.setenv("AL1S_DEV_MAX_OUTPUT_CHARS", "30000")
    monkeypatch.setenv("AL1S_DEV_COMMAND_TIMEOUT", "120")
    monkeypatch.setenv("AL1S_DEV_BRANCH_PREFIX", "al1s/")
    monkeypatch.setenv("AL1S_DEV_RUNNER_ENABLED", "false")
    return root


async def _create_plain_workspace(name="demo"):
    return json.loads(await server.workspace_create(name, initialize_git=False))


@pytest.mark.asyncio
async def test_workspace_create_and_list_are_bounded(
    monkeypatch, workspace_environment
):
    monkeypatch.setenv("AL1S_DEV_MAX_WORKSPACES", "1")

    created = await _create_plain_workspace()
    listed = json.loads(await server.workspace_list())

    assert created == {
        "workspace": "demo",
        "created": True,
        "git_initialized": False,
        "branch": None,
    }
    assert listed == {
        "workspaces": [{"name": "demo", "git_repository": False}],
        "count": 1,
    }
    with pytest.raises(RuntimeError, match="count limit"):
        await _create_plain_workspace("second")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["../escape", ".hidden", "name/child", "name..child", "repo.lock", ""]
)
async def test_workspace_create_rejects_invalid_names(workspace_environment, name):
    with pytest.raises(ValueError):
        await server.workspace_create(name, initialize_git=False)


@pytest.mark.asyncio
async def test_workspace_create_initializes_managed_branch_and_cleans_environment(
    monkeypatch, workspace_environment
):
    calls = []

    async def fake_execute(argv, *, cwd, environment):
        calls.append((tuple(argv), cwd, environment))
        return server.ProcessResult(0, "", "")

    monkeypatch.setattr(server, "_git_executable", lambda: "/usr/bin/git")
    monkeypatch.setattr(server, "_execute_process", fake_execute)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-forwarded")

    result = json.loads(await server.workspace_create("project"))

    assert result["branch"] == "al1s/project"
    argv, cwd, environment = calls[0]
    assert cwd == workspace_environment / "project"
    assert argv[0] == "/usr/bin/git"
    assert argv[-4:] == ("init", "--initial-branch", "al1s/project", "--")
    assert "core.hooksPath=/dev/null" in argv
    assert "commit.gpgSign=false" in argv
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert "UNRELATED_SECRET" not in environment


@pytest.mark.asyncio
async def test_file_tools_are_atomic_and_search_is_literal(workspace_environment):
    await _create_plain_workspace()

    written = json.loads(
        await server.workspace_write_file(
            "demo", "src/app.py", "value = 'needle'\nneedle = value\n"
        )
    )
    edited = json.loads(
        await server.workspace_edit_file(
            "demo", "src/app.py", "needle = value", "result = value"
        )
    )
    read = json.loads(await server.workspace_read_file("demo", "src/app.py"))
    searched = json.loads(
        await server.workspace_search_text("demo", "'NEEDLE'", case_sensitive=False)
    )
    listed = json.loads(await server.workspace_list_files("demo"))

    assert written["size_bytes"] > 0
    assert edited["replacements"] == 1
    assert read["content"] == "value = 'needle'\nresult = value\n"
    assert searched["matches"] == [
        {
            "path": "src/app.py",
            "line": 1,
            "text": "value = 'needle'",
            "line_truncated": False,
        }
    ]
    assert {item["path"] for item in listed["entries"]} == {"src", "src/app.py"}
    assert list(workspace_environment.rglob(".al1s-write-*")) == []


@pytest.mark.asyncio
async def test_file_tools_reject_traversal_git_metadata_and_symlinks(
    tmp_path, workspace_environment
):
    await _create_plain_workspace()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    (workspace_environment / "demo" / "link").symlink_to(outside)

    for path in ("../outside.txt", "/etc/passwd", ".git/config", "link"):
        with pytest.raises((FileNotFoundError, ValueError)):
            await server.workspace_read_file("demo", path)

    with pytest.raises(ValueError, match="symbolic links"):
        await server.workspace_write_file("demo", "link/child", "bad")
    assert outside.read_text(encoding="utf-8") == "private"


@pytest.mark.asyncio
async def test_file_byte_and_read_output_limits(monkeypatch, workspace_environment):
    await _create_plain_workspace()
    monkeypatch.setenv("AL1S_DEV_MAX_FILE_BYTES", "5")

    with pytest.raises(ValueError, match="byte limit"):
        await server.workspace_write_file("demo", "large.txt", "123456")

    await server.workspace_write_file("demo", "small.txt", "12345")
    monkeypatch.setenv("AL1S_DEV_MAX_OUTPUT_CHARS", "64")
    response = json.loads(await server.workspace_read_file("demo", "small.txt"))
    assert response["content"] == "12345"
    assert response["truncated"] is False


@pytest.mark.asyncio
async def test_git_metadata_symlink_is_not_a_supported_repository(
    tmp_path, workspace_environment
):
    await _create_plain_workspace()
    external_git = tmp_path / "external-git"
    external_git.mkdir()
    (workspace_environment / "demo" / ".git").symlink_to(
        external_git, target_is_directory=True
    )

    with pytest.raises(ValueError, match="supported local Git repository"):
        await server.workspace_git_status("demo")


@pytest.mark.asyncio
async def test_git_tools_validate_refs_paths_and_use_argument_vectors(
    monkeypatch, workspace_environment
):
    await _create_plain_workspace()
    (workspace_environment / "demo" / ".git").mkdir()
    calls = []

    async def fake_git_at(cwd, arguments):
        calls.append((cwd, tuple(arguments)))
        return server.ProcessResult(0, "ok", "")

    monkeypatch.setattr(server, "_run_git_at", fake_git_at)

    await server.workspace_git_diff(
        "demo", staged=True, ref="HEAD", paths=["src/app.py"]
    )
    await server.workspace_git_add("demo", ["src/app.py", "src/app.py"])

    assert calls[0][1] == (
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--cached",
        "HEAD",
        "--",
        "src/app.py",
    )
    assert calls[1][1] == ("add", "--", "src/app.py")
    with pytest.raises(ValueError, match="invalid Git ref"):
        await server.workspace_git_diff("demo", ref="HEAD; touch /tmp/pwned")
    with pytest.raises(ValueError, match="traversal"):
        await server.workspace_git_add("demo", ["../outside"])


@pytest.mark.asyncio
async def test_git_branches_are_confined_to_configured_prefix(
    monkeypatch, workspace_environment
):
    await _create_plain_workspace()
    (workspace_environment / "demo" / ".git").mkdir()
    calls = []

    async def fake_git_at(cwd, arguments):
        calls.append(tuple(arguments))
        return server.ProcessResult(0, "", "")

    monkeypatch.setattr(server, "_run_git_at", fake_git_at)

    for branch in ("main", "master", "feature/test", "al1s/../main"):
        with pytest.raises(ValueError):
            await server.workspace_git_create_branch("demo", branch)
        with pytest.raises(ValueError):
            await server.workspace_git_checkout("demo", branch)

    await server.workspace_git_create_branch("demo", "al1s/feature")
    await server.workspace_git_checkout("demo", "al1s/feature")
    assert calls == [
        ("branch", "al1s/feature", "HEAD", "--"),
        ("checkout", "--no-recurse-submodules", "al1s/feature", "--"),
    ]


@pytest.mark.asyncio
async def test_git_commit_rejects_main_and_accepts_only_managed_branch(
    monkeypatch, workspace_environment
):
    calls = []
    current_branch = "main"

    async def fake_run_git(workspace, arguments, *, require_success=True):
        calls.append(tuple(arguments))
        if arguments[0] == "symbolic-ref":
            return server.ProcessResult(0, current_branch + "\n", "")
        return server.ProcessResult(0, "committed", "")

    monkeypatch.setattr(server, "_run_git", fake_run_git)

    with pytest.raises(ValueError, match="main or master"):
        await server.workspace_git_commit("demo", "message")

    current_branch = "unmanaged/feature"
    with pytest.raises(ValueError, match="must start"):
        await server.workspace_git_commit("demo", "message")

    current_branch = "al1s/feature"
    result = json.loads(await server.workspace_git_commit("demo", "- safe message"))
    assert result["ok"] is True
    assert calls[-1] == (
        "commit",
        "--no-verify",
        "--no-gpg-sign",
        "-m",
        "- safe message",
        "--",
    )


@pytest.mark.asyncio
async def test_execute_process_uses_no_shell_and_bounds_captured_output(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AL1S_DEV_MAX_OUTPUT_CHARS", "64")
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.pid = 12345
            self.returncode = 0
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_data(b"x" * 1_000)
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self):
            return self.returncode

    async def fake_create(*argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    result = await server._execute_process(
        ("/usr/bin/true", "literal;not-shell"), cwd=tmp_path, environment={}
    )

    assert captured["argv"] == ("/usr/bin/true", "literal;not-shell")
    assert "shell" not in captured["kwargs"]
    assert captured["kwargs"]["start_new_session"] is True
    assert len(result.stdout) <= 64
    assert result.truncated is True
    assert result.stdout.endswith("[output truncated]")


@pytest.mark.asyncio
async def test_runner_is_disabled_by_default_and_rejects_arbitrary_profile(
    monkeypatch, workspace_environment
):
    await _create_plain_workspace()

    with pytest.raises(RuntimeError, match="disabled"):
        await server.workspace_run_check("demo", "pytest")

    monkeypatch.setenv("AL1S_DEV_RUNNER_ENABLED", "true")
    with pytest.raises(ValueError, match="unknown check profile"):
        await server.workspace_run_check("demo", "pytest --collect-only")


@pytest.mark.asyncio
async def test_runner_wraps_fixed_profile_in_network_denied_macos_sandbox(
    monkeypatch, workspace_environment
):
    await _create_plain_workspace()
    monkeypatch.setenv("AL1S_DEV_RUNNER_ENABLED", "true")
    monkeypatch.setattr(server.sys, "platform", "darwin")
    monkeypatch.setattr(
        server.shutil,
        "which",
        lambda command, path=None: (
            "/usr/bin/sandbox-exec" if command == "sandbox-exec" else None
        ),
    )
    calls = []

    async def fake_execute(argv, *, cwd, environment):
        calls.append((tuple(argv), cwd, environment))
        return server.ProcessResult(0, "compiled", "")

    monkeypatch.setattr(server, "_execute_process", fake_execute)

    result = json.loads(await server.workspace_run_check("demo", "python_compile"))

    argv, cwd, environment = calls[0]
    assert result["ok"] is True
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert any("(deny network*)" in argument for argument in argv)
    assert any("(deny appleevent-send)" in argument for argument in argv)
    assert any("(deny mach-lookup)" in argument for argument in argv)
    assert any("(deny signal (target others))" in argument for argument in argv)
    assert any('(literal "/dev/null")' in argument for argument in argv)
    assert any('(subpath (param "GIT_METADATA"))' in argument for argument in argv)
    assert f"GIT_METADATA={workspace_environment / 'demo' / '.git'}" in argv
    assert argv[-6:] == server.RUNNER_PROFILES["python_compile"]
    assert cwd == workspace_environment / "demo"
    assert environment["HOME"].startswith(str(workspace_environment / "demo"))
    assert "GITHUB_TOKEN" not in environment
    assert list((workspace_environment / "demo").glob(".al1s-runner-*")) == []


@pytest.mark.asyncio
async def test_runner_never_falls_back_when_sandbox_is_unavailable(
    monkeypatch, workspace_environment
):
    await _create_plain_workspace()
    monkeypatch.setenv("AL1S_DEV_RUNNER_ENABLED", "true")
    monkeypatch.setattr(server.sys, "platform", "darwin")
    monkeypatch.setattr(server.shutil, "which", lambda command, path=None: None)

    with pytest.raises(RuntimeError, match="sandbox-exec is unavailable"):
        await server.workspace_run_check("demo", "pytest")


@pytest.mark.asyncio
async def test_runner_rejects_sandbox_infrastructure_failure(
    monkeypatch, workspace_environment
):
    await _create_plain_workspace()
    monkeypatch.setenv("AL1S_DEV_RUNNER_ENABLED", "true")
    monkeypatch.setattr(server, "_sandbox_executable", lambda: "/usr/bin/sandbox-exec")

    async def failed_sandbox(argv, *, cwd, environment):
        return server.ProcessResult(71, "", "sandbox infrastructure failure")

    monkeypatch.setattr(server, "_execute_process", failed_sandbox)

    with pytest.raises(RuntimeError, match="could not reliably execute"):
        await server.workspace_run_check("demo", "pytest")


def test_environment_defaults_and_branch_prefix(monkeypatch, tmp_path):
    for name in (
        "AL1S_DEV_MAX_WORKSPACES",
        "MAX_WORKSPACES",
        "AL1S_DEV_MAX_FILE_BYTES",
        "MAX_FILE_BYTES",
        "AL1S_DEV_MAX_OUTPUT_CHARS",
        "MAX_OUTPUT_CHARS",
        "AL1S_DEV_COMMAND_TIMEOUT",
        "COMMAND_TIMEOUT",
        "AL1S_DEV_BRANCH_PREFIX",
        "BRANCH_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AL1S_DEV_WORKSPACE_ROOT", str(tmp_path / "workspaces"))

    assert server._max_workspaces() == 10
    assert server._max_file_bytes() == 1_000_000
    assert server._max_output_chars() == 30_000
    assert server._command_timeout() == 120
    assert server._branch_prefix() == "al1s/"

    monkeypatch.setenv("AL1S_DEV_BRANCH_PREFIX", "unsafe")
    with pytest.raises(RuntimeError, match="end with a slash"):
        server._branch_prefix()
