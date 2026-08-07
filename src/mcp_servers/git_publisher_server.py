"""Restricted GitHub clone and push tools for AL1S development workspaces."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from mcp.server.fastmcp import FastMCP

DEFAULT_BRANCH_PREFIX: Final = "al1s/"
DEFAULT_MAX_WORKSPACES: Final = 10
DEFAULT_GIT_TIMEOUT: Final = 60
DEFAULT_MAX_OUTPUT_CHARS: Final = 30_000
MAX_GIT_TIMEOUT: Final = 600
MAX_OUTPUT_CHARS: Final = 100_000

OWNER_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]{0,99}")
WORKSPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
REF_CHARACTER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
OBJECT_ID_PATTERN = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
UNSAFE_LOCAL_CONFIG_PATTERN = r"^(include|includeif|url|http|credential)\."

mcp = FastMCP(
    "AL1S GitHub Publisher",
    instructions=(
        "Clone approved GitHub repositories into isolated AL1S workspaces and "
        "push the current commit to approved development branches. This server "
        "never executes repository code."
    ),
)


def _integer_setting(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {name} setting") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} is outside the allowed range")
    return value


def _git_timeout() -> int:
    return _integer_setting(
        "AL1S_DEV_GIT_TIMEOUT",
        DEFAULT_GIT_TIMEOUT,
        minimum=1,
        maximum=MAX_GIT_TIMEOUT,
    )


def _max_workspaces() -> int:
    return _integer_setting(
        "AL1S_DEV_MAX_WORKSPACES",
        DEFAULT_MAX_WORKSPACES,
        minimum=1,
        maximum=1_000,
    )


def _max_output_chars() -> int:
    return _integer_setting(
        "AL1S_DEV_MAX_OUTPUT_CHARS",
        DEFAULT_MAX_OUTPUT_CHARS,
        minimum=256,
        maximum=MAX_OUTPUT_CHARS,
    )


def _validate_owner(owner: str) -> str:
    if not isinstance(owner, str) or not OWNER_PATTERN.fullmatch(owner):
        raise ValueError("invalid GitHub owner")
    if "--" in owner:
        raise ValueError("invalid GitHub owner")
    return owner


def _validate_repository(repository: str) -> str:
    if not isinstance(repository, str) or not REPOSITORY_PATTERN.fullmatch(repository):
        raise ValueError("invalid GitHub repository")
    if (
        ".." in repository
        or repository.endswith(".")
        or repository.casefold().endswith(".git")
    ):
        raise ValueError("invalid GitHub repository")
    return repository


def _validate_workspace(workspace: str) -> str:
    if not isinstance(workspace, str) or not WORKSPACE_PATTERN.fullmatch(workspace):
        raise ValueError("invalid workspace name")
    if (
        ".." in workspace
        or workspace.endswith(".")
        or workspace.casefold().endswith(".lock")
    ):
        raise ValueError("invalid workspace name")
    return workspace


def _validate_ref(value: str, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid {field}")
    if not value and allow_empty:
        return value
    if not REF_CHARACTER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid {field}")
    if value == "@" or value.startswith("refs/"):
        raise ValueError(f"invalid {field}")
    if ".." in value or "//" in value or "@{" in value:
        raise ValueError(f"invalid {field}")
    for component in value.split("/"):
        if (
            not component
            or component.startswith(".")
            or component.startswith("-")
            or component.endswith(".")
            or component.casefold().endswith(".lock")
        ):
            raise ValueError(f"invalid {field}")
    return value


def _allowed_owner(owner: str) -> str:
    owner = _validate_owner(owner)
    raw_owners = os.getenv("AL1S_DEV_ALLOWED_GITHUB_OWNERS", "")
    configured = [item.strip() for item in raw_owners.split(",") if item.strip()]
    if not configured:
        raise RuntimeError("no GitHub owners are configured")
    try:
        allowed = {_validate_owner(item).casefold() for item in configured}
    except ValueError as exc:
        raise RuntimeError("invalid configured GitHub owner") from exc
    if owner.casefold() not in allowed:
        raise ValueError("GitHub owner is not allowed")
    return owner


def _branch_prefix() -> str:
    prefix = os.getenv("AL1S_DEV_BRANCH_PREFIX", DEFAULT_BRANCH_PREFIX)
    if not prefix or not prefix.endswith("/"):
        raise RuntimeError("development branch prefix must end with a slash")
    try:
        _validate_ref(f"{prefix}branch", field="development branch prefix")
    except ValueError as exc:
        raise RuntimeError("invalid development branch prefix") from exc
    return prefix


def _validate_branch(branch: str) -> str:
    branch = _validate_ref(branch, field="branch")
    if branch.casefold() in {"main", "master"}:
        raise ValueError("pushing main or master is forbidden")
    if not branch.startswith(_branch_prefix()):
        raise ValueError("branch is outside the configured development prefix")
    return branch


def _validate_expected_head(expected_head: str) -> str:
    if not isinstance(expected_head, str) or not OBJECT_ID_PATTERN.fullmatch(
        expected_head
    ):
        raise ValueError("expected_head must be a full Git object ID")
    return expected_head.lower()


def _workspace_root() -> Path:
    raw_root = os.getenv("AL1S_DEV_WORKSPACE_ROOT", "")
    if not raw_root or "\x00" in raw_root:
        raise RuntimeError("development workspace root is not configured")
    configured = Path(raw_root)
    if not configured.is_absolute():
        raise RuntimeError("development workspace root must be absolute")
    try:
        configured.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError("development workspace root is unavailable") from exc
    if configured.is_symlink() or not configured.is_dir():
        raise RuntimeError("development workspace root must be a real directory")
    try:
        return configured.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("development workspace root is unavailable") from exc


def _new_workspace_path(root: Path, workspace: str) -> Path:
    target = root / _validate_workspace(workspace)
    if target.exists() or target.is_symlink():
        raise ValueError("workspace already exists")
    if sum(1 for _entry in root.iterdir()) >= _max_workspaces():
        raise RuntimeError("workspace count limit reached")
    if target.parent.resolve(strict=True) != root:
        raise ValueError("workspace escapes the configured root")
    return target


def _cleanup_failed_clone(target: Path, root: Path) -> None:
    if target.is_symlink() or not target.exists():
        return
    try:
        target.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        return
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


def _existing_workspace_path(root: Path, workspace: str) -> Path:
    candidate = root / _validate_workspace(workspace)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("workspace does not exist or is not a real directory")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("workspace escapes the configured root") from exc
    git_directory = resolved / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise ValueError("workspace is not a supported Git repository")
    try:
        git_directory.resolve(strict=True).relative_to(resolved)
    except (OSError, ValueError) as exc:
        raise ValueError("Git metadata escapes the workspace") from exc
    return resolved


def _github_token() -> str:
    token = os.getenv("AL1S_DEV_GITHUB_TOKEN", "")
    if (
        not token
        or len(token) > 512
        or any(ord(character) <= 32 for character in token)
    ):
        raise RuntimeError("GitHub publisher credentials are not configured")
    return token


def _github_url(owner: str, repository: str) -> str:
    return f"https://github.com/{owner}/{repository}.git"


def _git_executable() -> str:
    executable = shutil.which("git")
    if not executable:
        raise RuntimeError("git executable is unavailable")
    resolved = Path(executable).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError("git executable is unavailable")
    return str(resolved)


def _write_askpass(directory: Path) -> Path:
    askpass = directory / "askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  *sername*) printf '%s\\n' 'x-access-token' ;;\n"
        "  *assword*) printf '%s\\n' \"$AL1S_GIT_ASKPASS_TOKEN\" ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(stat.S_IRWXU)
    return askpass


def _git_environment(
    directory: Path, askpass: Path, token: str | None
) -> dict[str, str]:
    hooks = directory / "hooks"
    templates = directory / "templates"
    config_home = directory / "config"
    for path in (hooks, templates, config_home):
        path.mkdir(mode=0o700)

    environment = {
        "HOME": str(config_home),
        "XDG_CONFIG_HOME": str(config_home),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_ASKPASS": str(askpass),
        "SSH_ASKPASS": str(askpass),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TEMPLATE_DIR": str(templates),
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_CONFIG_COUNT": "6",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(hooks),
        "GIT_CONFIG_KEY_1": "credential.helper",
        "GIT_CONFIG_VALUE_1": "",
        "GIT_CONFIG_KEY_2": "core.fsmonitor",
        "GIT_CONFIG_VALUE_2": "false",
        "GIT_CONFIG_KEY_3": "protocol.allow",
        "GIT_CONFIG_VALUE_3": "never",
        "GIT_CONFIG_KEY_4": "protocol.https.allow",
        "GIT_CONFIG_VALUE_4": "always",
        "GIT_CONFIG_KEY_5": "http.followRedirects",
        "GIT_CONFIG_VALUE_5": "false",
    }
    if token is not None:
        environment["AL1S_GIT_ASKPASS_TOKEN"] = token
    return environment


def _bounded_output(output: object, token: str | None) -> str:
    if isinstance(output, bytes):
        text = output.decode("utf-8", errors="replace")
    elif output is None:
        text = ""
    else:
        text = str(output)
    if token:
        text = text.replace(token, "[credential redacted]")
    limit = _max_output_chars()
    suffix = "\n[output truncated]"
    if len(text) > limit:
        text = text[: max(0, limit - len(suffix))] + suffix
    return text.strip()


def _run_git(
    arguments: list[str],
    *,
    cwd: Path,
    token: str | None = None,
    success_codes: tuple[int, ...] = (0,),
) -> str:
    with tempfile.TemporaryDirectory(prefix="al1s-git-publisher-") as temporary:
        credential_directory = Path(temporary)
        credential_directory.chmod(stat.S_IRWXU)
        askpass = _write_askpass(credential_directory)
        environment = _git_environment(credential_directory, askpass, token)
        command = [_git_executable(), *arguments]
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_git_timeout(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _bounded_output(exc.stdout, token)
            raise RuntimeError("git command timed out") from None
        except OSError:
            raise RuntimeError("git command could not be started") from None

        output = _bounded_output(completed.stdout, token)
        if completed.returncode not in success_codes:
            detail = f": {output}" if output else ""
            raise RuntimeError(f"git command failed{detail}")
        return output


def _read_head(workspace_path: Path, root: Path) -> str:
    output = _run_git(
        ["-C", str(workspace_path), "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=root,
    )
    if not OBJECT_ID_PATTERN.fullmatch(output):
        raise RuntimeError("git returned an invalid current HEAD")
    return output.lower()


def _assert_safe_local_git_config(workspace_path: Path, root: Path) -> None:
    unsafe_names = _run_git(
        [
            "-C",
            str(workspace_path),
            "config",
            "--local",
            "--no-includes",
            "--name-only",
            "--get-regexp",
            UNSAFE_LOCAL_CONFIG_PATTERN,
        ],
        cwd=root,
        success_codes=(0, 1),
    )
    if unsafe_names:
        raise ValueError(
            "workspace Git configuration contains forbidden network or include settings"
        )


@mcp.tool()
def workspace_clone_github(
    owner: str,
    repo: str,
    workspace: str,
    ref: str = "",
) -> dict[str, str]:
    """Shallow-clone an approved GitHub repository without running project code."""
    owner = _allowed_owner(owner)
    repo = _validate_repository(repo)
    workspace = _validate_workspace(workspace)
    ref = _validate_ref(ref, field="ref", allow_empty=True)
    root = _workspace_root()
    target = _new_workspace_path(root, workspace)
    token = _github_token()

    arguments = [
        "clone",
        "--quiet",
        "--depth=1",
        "--single-branch",
        "--no-tags",
        "--no-recurse-submodules",
    ]
    if ref:
        arguments.extend(["--branch", ref])
    arguments.extend(["--", _github_url(owner, repo), str(target)])
    try:
        _run_git(arguments, cwd=root, token=token)
    except BaseException:
        _cleanup_failed_clone(target, root)
        raise

    repository_path = _existing_workspace_path(root, workspace)
    head = _read_head(repository_path, root)
    return {
        "workspace": workspace,
        "repository": f"{owner}/{repo}",
        "ref": ref,
        "head": head,
    }


@mcp.tool()
def workspace_push_github(
    workspace: str,
    owner: str,
    repo: str,
    branch: str,
    expected_head: str,
) -> dict[str, str | bool]:
    """Push the expected workspace HEAD to an approved GitHub dev branch."""
    workspace = _validate_workspace(workspace)
    owner = _allowed_owner(owner)
    repo = _validate_repository(repo)
    branch = _validate_branch(branch)
    expected_head = _validate_expected_head(expected_head)
    root = _workspace_root()
    repository_path = _existing_workspace_path(root, workspace)
    _assert_safe_local_git_config(repository_path, root)
    current_head = _read_head(repository_path, root)
    if not secrets.compare_digest(current_head, expected_head):
        raise ValueError("expected_head does not match the current workspace HEAD")

    token = _github_token()
    _run_git(
        [
            "-C",
            str(repository_path),
            "push",
            "--porcelain",
            "--no-verify",
            "--recurse-submodules=no",
            "--",
            _github_url(owner, repo),
            f"HEAD:refs/heads/{branch}",
        ],
        cwd=root,
        token=token,
    )
    return {
        "workspace": workspace,
        "repository": f"{owner}/{repo}",
        "branch": branch,
        "head": current_head,
        "pushed": True,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
