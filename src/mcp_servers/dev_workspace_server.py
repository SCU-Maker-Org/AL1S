"""A constrained local development workspace MCP server for AL1S.

The server deliberately exposes task-oriented file, Git, and check tools instead
of a general-purpose shell.  Every caller-controlled path stays below a dedicated
workspace root, and file tools never enter Git's private metadata directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP

DEFAULT_MAX_WORKSPACES = 10
DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_OUTPUT_CHARS = 30_000
DEFAULT_COMMAND_TIMEOUT = 120.0
DEFAULT_GIT_AUTHOR_NAME = "AL1S"
DEFAULT_GIT_AUTHOR_EMAIL = "al1s@localhost"
DEFAULT_BRANCH_PREFIX = "al1s/"

MAX_WORKSPACE_NAME_CHARS = 64
MAX_RELATIVE_PATH_CHARS = 512
MAX_LIST_ENTRIES = 1_000
MAX_SEARCH_RESULTS = 500
MAX_GIT_PATHS = 200
MAX_COMMIT_MESSAGE_CHARS = 2_000
MAX_QUERY_CHARS = 2_000
SAFE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"

WORKSPACE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SAFE_REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
EMAIL_PATTERN = re.compile(r"[^\s@<>]+@[^\s@<>]+")

GIT_BASE_ARGUMENTS = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "commit.gpgSign=false",
    "-c",
    "tag.gpgSign=false",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "credential.helper=",
)

RUNNER_PROFILES: dict[str, tuple[str, ...]] = {
    "python_compile": (sys.executable, "-I", "-m", "compileall", "-q", "."),
    "pytest": (sys.executable, "-I", "-m", "pytest", "-q"),
    "ruff": (sys.executable, "-I", "-m", "ruff", "check", "."),
    "black_check": (sys.executable, "-I", "-m", "black", "--check", "."),
    "isort_check": (sys.executable, "-I", "-m", "isort", "--check-only", "."),
    "npm_test": ("npm", "test", "--"),
}

SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny appleevent-send)
(deny mach-lookup)
(deny mach-per-user-lookup)
(deny mach-register)
(deny mach-priv-host-port)
(deny signal (target others))
(deny process-info* (target others))
(deny file-read-data
  (require-all
    (subpath (param \"USER_HOME\"))
    (require-not (subpath (param \"WORKSPACE\")))
    (require-not (subpath (param \"TOOLCHAIN\")))
    (require-not (subpath (param \"PYTHON_RUNTIME\")))))
(deny file-write*
  (require-all
    (require-not (subpath (param \"WORKSPACE\")))
    (require-not (literal \"/dev/null\"))))
(deny file-write*
  (subpath (param \"GIT_METADATA\")))
"""

mcp = FastMCP(
    "AL1S Dev Workspace",
    instructions=(
        "Create and edit isolated development workspaces, use constrained local "
        "Git operations, and run only administrator-approved check profiles."
    ),
)

_mutation_lock = asyncio.Lock()


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    truncated: bool = False


def _environment(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def _integer_setting(
    names: Sequence[str],
    *,
    default: int,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    raw_value = _environment(*names, default=str(default))
    try:
        value = int(raw_value or str(default))
    except ValueError as exc:
        raise RuntimeError(f"invalid {label}") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{label} is outside the allowed range")
    return value


def _max_workspaces() -> int:
    return _integer_setting(
        ("AL1S_DEV_MAX_WORKSPACES", "MAX_WORKSPACES"),
        default=DEFAULT_MAX_WORKSPACES,
        minimum=1,
        maximum=1_000,
        label="workspace count limit",
    )


def _max_file_bytes() -> int:
    return _integer_setting(
        ("AL1S_DEV_MAX_FILE_BYTES", "MAX_FILE_BYTES"),
        default=DEFAULT_MAX_FILE_BYTES,
        minimum=1,
        maximum=100_000_000,
        label="workspace file size limit",
    )


def _max_output_chars() -> int:
    return _integer_setting(
        ("AL1S_DEV_MAX_OUTPUT_CHARS", "MAX_OUTPUT_CHARS"),
        default=DEFAULT_MAX_OUTPUT_CHARS,
        minimum=64,
        maximum=100_000,
        label="command output limit",
    )


def _command_timeout() -> float:
    raw_value = _environment(
        "AL1S_DEV_COMMAND_TIMEOUT",
        "COMMAND_TIMEOUT",
        default=str(DEFAULT_COMMAND_TIMEOUT),
    )
    try:
        value = float(raw_value or str(DEFAULT_COMMAND_TIMEOUT))
    except ValueError as exc:
        raise RuntimeError("invalid command timeout") from exc
    if not 0.1 <= value <= 3_600:
        raise RuntimeError("command timeout is outside the allowed range")
    return value


def _runner_enabled() -> bool:
    raw_value = (
        (
            _environment("AL1S_DEV_RUNNER_ENABLED", "RUNNER_ENABLED", default="false")
            or "false"
        )
        .strip()
        .lower()
    )
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError("invalid runner enabled setting")


def _workspace_root() -> Path:
    configured = _environment("AL1S_DEV_WORKSPACE_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.cwd() / "data" / "dev_workspaces"
    )
    if root.exists() and root.is_symlink():
        raise RuntimeError("workspace root must not be a symbolic link")
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("workspace root is unavailable") from exc
    if not root.is_dir():
        raise RuntimeError("workspace root is not a directory")
    return root


def _validate_workspace_name(value: str) -> str:
    if not isinstance(value, str) or not WORKSPACE_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "workspace must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    if len(value) > MAX_WORKSPACE_NAME_CHARS or value in {".", ".."}:
        raise ValueError("invalid workspace name")
    if (
        value.casefold() == ".git"
        or value.casefold().endswith(".lock")
        or ".." in value
        or value.endswith(".")
    ):
        raise ValueError("reserved workspace name")
    return value


def _workspace_path(workspace: str, *, require_exists: bool = True) -> Path:
    workspace = _validate_workspace_name(workspace)
    root = _workspace_root()
    candidate = root / workspace
    if candidate.is_symlink():
        raise ValueError("workspace must not be a symbolic link")
    if not candidate.exists():
        if require_exists:
            raise FileNotFoundError(f"workspace does not exist: {workspace}")
        return candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("workspace escapes the configured root") from exc
    if not resolved.is_dir():
        raise ValueError("workspace is not a directory")
    return resolved


def _validate_relative_path(value: str, *, allow_root: bool = False) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > MAX_RELATIVE_PATH_CHARS:
        raise ValueError("path must be a non-empty relative path")
    if "\x00" in value or "\\" in value or value.startswith("/"):
        raise ValueError("path must be a safe POSIX relative path")
    raw_parts = value.split("/")
    if any(part in {"", ".."} for part in raw_parts):
        raise ValueError("path traversal is forbidden")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError("path traversal is forbidden")
    if not allow_root and path == PurePosixPath("."):
        raise ValueError("path must identify a file")
    if any(part.casefold() == ".git" for part in path.parts):
        raise ValueError("file tools cannot access .git")
    return path


def _assert_contained(candidate: Path, workspace: Path) -> Path:
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise ValueError("path escapes the workspace") from exc
    return resolved


def _reject_symlink_components(candidate: Path, workspace: Path) -> None:
    current = workspace
    relative = candidate.relative_to(workspace)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symbolic links are forbidden in workspace paths")
        if not current.exists():
            break


def _resolve_workspace_entry(
    workspace: str,
    relative_path: str,
    *,
    must_exist: bool,
    allow_root: bool = False,
) -> tuple[Path, Path]:
    root = _workspace_path(workspace)
    relative = _validate_relative_path(relative_path, allow_root=allow_root)
    candidate = root.joinpath(*relative.parts)
    _reject_symlink_components(candidate, root)
    resolved = _assert_contained(candidate, root)
    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"workspace path does not exist: {relative_path}")
    return root, resolved


def _validate_text(value: str, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    if not value or not value.strip():
        raise ValueError(f"{field} must not be empty")
    if len(value) > max_chars or "\x00" in value:
        raise ValueError(f"{field} exceeds the allowed size")
    return value


def _validate_positive_count(value: int, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return value


def _branch_prefix() -> str:
    prefix = _environment(
        "AL1S_DEV_BRANCH_PREFIX", "BRANCH_PREFIX", default=DEFAULT_BRANCH_PREFIX
    )
    if not prefix or not prefix.endswith("/"):
        raise RuntimeError("branch prefix must end with a slash")
    try:
        _validate_ref(prefix + "probe", field="branch prefix")
    except ValueError as exc:
        raise RuntimeError("invalid branch prefix") from exc
    return prefix


def _validate_ref(value: str, *, field: str = "ref") -> str:
    if not isinstance(value, str) or not SAFE_REF_PATTERN.fullmatch(value):
        raise ValueError(f"invalid Git {field}")
    if (
        value.startswith(("/", "-", "."))
        or value.endswith(("/", ".", ".lock"))
        or ".." in value
        or "//" in value
        or "@{" in value
        or any(
            part.startswith(".") or part.endswith(".lock") for part in value.split("/")
        )
    ):
        raise ValueError(f"invalid Git {field}")
    return value


def _validate_managed_branch(value: str) -> str:
    branch = _validate_ref(value, field="branch")
    prefix = _branch_prefix()
    if not branch.startswith(prefix) or branch == prefix.rstrip("/"):
        raise ValueError(f"branch must start with {prefix}")
    return branch


def _validate_git_paths(paths: list[str]) -> list[str]:
    if not isinstance(paths, list) or not paths or len(paths) > MAX_GIT_PATHS:
        raise ValueError(f"paths must contain between 1 and {MAX_GIT_PATHS} entries")
    validated: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = _validate_relative_path(raw_path, allow_root=True).as_posix()
        if path not in seen:
            validated.append(path)
            seen.add(path)
    return validated


def _git_environment() -> dict[str, str]:
    author_name = _environment(
        "AL1S_DEV_GIT_AUTHOR_NAME", "GIT_AUTHOR_NAME", default=DEFAULT_GIT_AUTHOR_NAME
    )
    author_email = _environment(
        "AL1S_DEV_GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_EMAIL",
        default=DEFAULT_GIT_AUTHOR_EMAIL,
    )
    if (
        not author_name
        or len(author_name) > 100
        or any(ord(character) < 32 for character in author_name)
    ):
        raise RuntimeError("invalid Git author name")
    if (
        not author_email
        or len(author_email) > 254
        or not EMAIL_PATTERN.fullmatch(author_email)
    ):
        raise RuntimeError("invalid Git author email")
    return {
        "PATH": SAFE_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_ASKPASS": "/usr/bin/false",
        "SSH_ASKPASS": "/usr/bin/false",
        "GIT_AUTHOR_NAME": author_name,
        "GIT_AUTHOR_EMAIL": author_email,
        "GIT_COMMITTER_NAME": author_name,
        "GIT_COMMITTER_EMAIL": author_email,
    }


def _runner_environment(workspace: Path, temporary: Path) -> dict[str, str]:
    return {
        "PATH": SAFE_PATH,
        "HOME": str(temporary),
        "TMPDIR": str(temporary),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "NPM_CONFIG_AUDIT": "false",
        "NPM_CONFIG_FUND": "false",
        "NPM_CONFIG_UPDATE_NOTIFIER": "false",
        "NPM_CONFIG_CACHE": str(temporary / "npm-cache"),
        "CI": "true",
        "AL1S_DEV_WORKSPACE": str(workspace),
    }


def _git_executable() -> str:
    executable = shutil.which("git", path=SAFE_PATH)
    if not executable:
        raise RuntimeError("Git is not available")
    return str(Path(executable).resolve())


def _sandbox_executable() -> str:
    if sys.platform != "darwin":
        raise RuntimeError("workspace checks require the macOS sandbox-exec sandbox")
    executable = shutil.which("sandbox-exec", path=SAFE_PATH)
    if not executable:
        raise RuntimeError("sandbox-exec is unavailable; refusing to run checks")
    resolved = str(Path(executable).resolve())
    if resolved != "/usr/bin/sandbox-exec":
        raise RuntimeError("untrusted sandbox-exec executable; refusing to run checks")
    return resolved


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        if process.returncode is None:
            process.kill()


async def _read_process_stream(
    stream: asyncio.StreamReader | None, byte_limit: int
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    content = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            break
        remaining = byte_limit - len(content)
        if remaining > 0:
            content.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(content), truncated


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    marker = "\n[output truncated]"
    if limit <= len(marker):
        return marker[:limit], True
    return value[: limit - len(marker)] + marker, True


def _bound_process_output(
    stdout: str, stderr: str, *, already_truncated: bool = False
) -> tuple[str, str, bool]:
    limit = _max_output_chars()
    if len(stdout) + len(stderr) <= limit:
        return stdout, stderr, already_truncated

    if stdout and stderr:
        stdout_budget = min(len(stdout), limit // 2)
        stderr_budget = min(len(stderr), limit - stdout_budget)
        remaining = limit - stdout_budget - stderr_budget
        stdout_budget += min(len(stdout) - stdout_budget, remaining)
        remaining = limit - stdout_budget - stderr_budget
        stderr_budget += min(len(stderr) - stderr_budget, remaining)
    else:
        stdout_budget = limit if stdout else 0
        stderr_budget = limit if stderr else 0

    stdout, stdout_truncated = _truncate_text(stdout, stdout_budget)
    stderr, stderr_truncated = _truncate_text(stderr, stderr_budget)
    return stdout, stderr, already_truncated or stdout_truncated or stderr_truncated


async def _execute_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> ProcessResult:
    if not argv or any(
        not isinstance(argument, str) or "\x00" in argument for argument in argv
    ):
        raise ValueError("invalid command arguments")
    max_chars = _max_output_chars()
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise RuntimeError("approved command is unavailable") from exc

    byte_limit = max_chars * 4 + 4
    stdout_task = asyncio.create_task(_read_process_stream(process.stdout, byte_limit))
    stderr_task = asyncio.create_task(_read_process_stream(process.stderr, byte_limit))
    try:
        await asyncio.wait_for(process.wait(), timeout=_command_timeout())
    except TimeoutError:
        _kill_process_group(process)
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise RuntimeError("command exceeded the configured timeout") from None
    except BaseException:
        if process.returncode is None:
            _kill_process_group(process)
            await process.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise

    (stdout_bytes, stdout_cut), (stderr_bytes, stderr_cut) = await asyncio.gather(
        stdout_task, stderr_task
    )
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    stdout, stderr, truncated = _bound_process_output(
        stdout,
        stderr,
        already_truncated=stdout_cut or stderr_cut,
    )
    return ProcessResult(process.returncode or 0, stdout, stderr, truncated)


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _command_response(result: ProcessResult, *, workspace: str, operation: str) -> str:
    return _json_response(
        {
            "workspace": workspace,
            "operation": operation,
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.truncated,
        }
    )


async def _run_git_at(cwd: Path, arguments: Sequence[str]) -> ProcessResult:
    argv = (_git_executable(), *GIT_BASE_ARGUMENTS, *arguments)
    return await _execute_process(argv, cwd=cwd, environment=_git_environment())


def _require_git_repository(workspace: str) -> Path:
    path = _workspace_path(workspace)
    git_directory = path / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise ValueError("workspace is not a supported local Git repository")
    try:
        git_directory.resolve(strict=True).relative_to(path)
    except (OSError, ValueError) as exc:
        raise ValueError("Git metadata escapes the workspace") from exc
    return path


async def _run_git(
    workspace: str,
    arguments: Sequence[str],
    *,
    require_success: bool = True,
) -> ProcessResult:
    path = _require_git_repository(workspace)
    result = await _run_git_at(path, arguments)
    if require_success and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "unknown Git error"
        raise RuntimeError(f"Git operation failed: {detail}")
    return result


def _write_text_atomic(destination: Path, content: str, workspace: Path) -> int:
    encoded = content.encode("utf-8")
    if len(encoded) > _max_file_bytes():
        raise ValueError("file content exceeds the configured byte limit")
    _reject_symlink_components(destination, workspace)
    _assert_contained(destination, workspace)
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise RuntimeError("could not create the destination directory") from exc
    _reject_symlink_components(parent, workspace)
    parent = parent.resolve(strict=True)
    try:
        parent.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("destination directory escapes the workspace") from exc
    if destination.is_symlink():
        raise ValueError("symbolic link destinations are forbidden")
    if destination.exists() and not destination.is_file():
        raise ValueError("destination is not a regular file")

    previous_mode = None
    if destination.exists():
        previous_mode = stat.S_IMODE(destination.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".al1s-write-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, previous_mode if previous_mode is not None else 0o600)
        if destination.is_symlink():
            raise ValueError("symbolic link destinations are forbidden")
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return len(encoded)


def _read_text_file(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("path is not a regular file")
    size = path.stat().st_size
    if size > _max_file_bytes():
        raise ValueError("file exceeds the configured byte limit")
    try:
        raw_content = path.read_bytes()
    except OSError as exc:
        raise RuntimeError("could not read workspace file") from exc
    if len(raw_content) > _max_file_bytes():
        raise ValueError("file exceeds the configured byte limit")
    try:
        return raw_content.decode("utf-8"), len(raw_content)
    except UnicodeDecodeError as exc:
        raise ValueError("workspace file is not UTF-8 text") from exc


@mcp.tool(structured_output=False)
async def workspace_create(name: str, initialize_git: bool = True) -> str:
    """Create an isolated workspace and, by default, initialize a local Git repo."""
    name = _validate_workspace_name(name)
    if not isinstance(initialize_git, bool):
        raise ValueError("initialize_git must be a boolean")
    root = _workspace_root()
    async with _mutation_lock:
        if (root / name).exists() or (root / name).is_symlink():
            raise FileExistsError(f"workspace already exists: {name}")
        if len(list(root.iterdir())) >= _max_workspaces():
            raise RuntimeError("workspace count limit reached")
        destination = root / name
        destination.mkdir(mode=0o700)
        branch: str | None = None
        try:
            if initialize_git:
                branch = _validate_managed_branch(_branch_prefix() + name)
                result = await _run_git_at(
                    destination, ("init", "--initial-branch", branch, "--")
                )
                if result.returncode != 0:
                    detail = (
                        result.stderr or result.stdout
                    ).strip() or "unknown Git error"
                    raise RuntimeError(f"Git initialization failed: {detail}")
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    return _json_response(
        {
            "workspace": name,
            "created": True,
            "git_initialized": initialize_git,
            "branch": branch,
        }
    )


@mcp.tool(structured_output=False)
async def workspace_list() -> str:
    """List valid workspaces below the configured workspace root."""
    root = _workspace_root()
    workspaces: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if len(workspaces) >= _max_workspaces():
            break
        try:
            _validate_workspace_name(entry.name)
            resolved = _workspace_path(entry.name)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        workspaces.append(
            {
                "name": entry.name,
                "git_repository": (resolved / ".git").is_dir()
                and not (resolved / ".git").is_symlink(),
            }
        )
    return _json_response({"workspaces": workspaces, "count": len(workspaces)})


@mcp.tool(structured_output=False)
async def workspace_list_files(
    workspace: str, path: str = ".", max_entries: int = 200
) -> str:
    """List regular files and directories without following symbolic links."""
    max_entries = _validate_positive_count(
        max_entries, field="max_entries", maximum=MAX_LIST_ENTRIES
    )
    root, start = _resolve_workspace_entry(
        workspace, path, must_exist=True, allow_root=True
    )
    if not start.is_dir():
        raise ValueError("path is not a directory")
    entries: list[dict[str, Any]] = []
    truncated = False
    for current_directory, directory_names, file_names in os.walk(
        start, topdown=True, followlinks=False
    ):
        current = Path(current_directory)
        safe_directories: list[str] = []
        for directory_name in sorted(directory_names):
            child = current / directory_name
            if directory_name.casefold() == ".git" or child.is_symlink():
                continue
            safe_directories.append(directory_name)
            if len(entries) >= max_entries:
                truncated = True
                break
            entries.append(
                {"path": child.relative_to(root).as_posix(), "type": "directory"}
            )
        directory_names[:] = safe_directories if not truncated else []
        if truncated:
            break
        for file_name in sorted(file_names):
            child = current / file_name
            if (
                file_name.casefold() == ".git"
                or child.is_symlink()
                or not child.is_file()
            ):
                continue
            if len(entries) >= max_entries:
                truncated = True
                break
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "type": "file",
                    "size_bytes": child.stat().st_size,
                }
            )
        if truncated:
            break
    return _json_response(
        {
            "workspace": workspace,
            "path": PurePosixPath(path).as_posix(),
            "entries": entries,
            "count": len(entries),
            "truncated": truncated,
        }
    )


@mcp.tool(structured_output=False)
async def workspace_read_file(workspace: str, path: str) -> str:
    """Read one UTF-8 text file within a workspace."""
    _, source = _resolve_workspace_entry(workspace, path, must_exist=True)
    content, size = _read_text_file(source)
    content, truncated = _truncate_text(content, _max_output_chars())
    return _json_response(
        {
            "workspace": workspace,
            "path": PurePosixPath(path).as_posix(),
            "content": content,
            "size_bytes": size,
            "truncated": truncated,
        }
    )


@mcp.tool(structured_output=False)
async def workspace_write_file(
    workspace: str, path: str, content: str, overwrite: bool = True
) -> str:
    """Atomically write one UTF-8 text file within a workspace."""
    if not isinstance(content, str) or "\x00" in content:
        raise ValueError("content must be text")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    root, destination = _resolve_workspace_entry(workspace, path, must_exist=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"workspace file already exists: {path}")
    size = _write_text_atomic(destination, content, root)
    return _json_response(
        {
            "workspace": workspace,
            "path": PurePosixPath(path).as_posix(),
            "written": True,
            "size_bytes": size,
        }
    )


@mcp.tool(structured_output=False)
async def workspace_edit_file(
    workspace: str,
    path: str,
    old_text: str,
    new_text: str,
    expected_replacements: int = 1,
) -> str:
    """Atomically replace an exact text fragment in one workspace file."""
    old_text = _validate_text(old_text, field="old_text", max_chars=_max_file_bytes())
    if not isinstance(new_text, str) or "\x00" in new_text:
        raise ValueError("new_text must be valid text")
    expected_replacements = _validate_positive_count(
        expected_replacements, field="expected_replacements", maximum=1_000
    )
    root, destination = _resolve_workspace_entry(workspace, path, must_exist=True)
    content, _ = _read_text_file(destination)
    occurrences = content.count(old_text)
    if occurrences != expected_replacements:
        raise ValueError(
            f"expected {expected_replacements} replacement(s), found {occurrences}"
        )
    updated = content.replace(old_text, new_text)
    size = _write_text_atomic(destination, updated, root)
    return _json_response(
        {
            "workspace": workspace,
            "path": PurePosixPath(path).as_posix(),
            "edited": True,
            "replacements": occurrences,
            "size_bytes": size,
        }
    )


@mcp.tool(structured_output=False)
async def workspace_search_text(
    workspace: str,
    query: str,
    path: str = ".",
    max_results: int = 100,
    case_sensitive: bool = False,
) -> str:
    """Search for a literal text fragment in bounded UTF-8 workspace files."""
    query = _validate_text(query, field="query", max_chars=MAX_QUERY_CHARS)
    max_results = _validate_positive_count(
        max_results, field="max_results", maximum=MAX_SEARCH_RESULTS
    )
    if not isinstance(case_sensitive, bool):
        raise ValueError("case_sensitive must be a boolean")
    root, start = _resolve_workspace_entry(
        workspace, path, must_exist=True, allow_root=True
    )
    candidates: list[Path] = []
    if start.is_file():
        candidates.append(start)
    elif start.is_dir():
        for current_directory, directory_names, file_names in os.walk(
            start, topdown=True, followlinks=False
        ):
            current = Path(current_directory)
            directory_names[:] = [
                name
                for name in sorted(directory_names)
                if name.casefold() != ".git" and not (current / name).is_symlink()
            ]
            for file_name in sorted(file_names):
                candidate = current / file_name
                if not candidate.is_symlink() and candidate.is_file():
                    candidates.append(candidate)
    else:
        raise ValueError("search path is not a regular file or directory")

    needle = query if case_sensitive else query.casefold()
    matches: list[dict[str, Any]] = []
    skipped_files = 0
    truncated = False
    for candidate in candidates:
        try:
            content, _ = _read_text_file(candidate)
        except (OSError, RuntimeError, ValueError):
            skipped_files += 1
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            haystack = line if case_sensitive else line.casefold()
            if needle not in haystack:
                continue
            preview, preview_truncated = _truncate_text(line, 500)
            matches.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "line": line_number,
                    "text": preview,
                    "line_truncated": preview_truncated,
                }
            )
            if len(matches) >= max_results:
                truncated = True
                break
        if truncated:
            break
    return _json_response(
        {
            "workspace": workspace,
            "query": query,
            "matches": matches,
            "count": len(matches),
            "skipped_files": skipped_files,
            "truncated": truncated,
        }
    )


@mcp.tool(structured_output=False)
async def workspace_git_status(workspace: str) -> str:
    """Show concise status for a supported workspace Git repository."""
    result = await _run_git(
        workspace, ("status", "--short", "--branch", "--untracked-files=all")
    )
    return _command_response(result, workspace=workspace, operation="git_status")


@mcp.tool(structured_output=False)
async def workspace_git_diff(
    workspace: str,
    staged: bool = False,
    ref: str | None = None,
    paths: list[str] | None = None,
) -> str:
    """Show a bounded working-tree or staged diff, optionally against one safe ref."""
    if not isinstance(staged, bool):
        raise ValueError("staged must be a boolean")
    arguments: list[str] = ["diff", "--no-ext-diff", "--no-textconv"]
    if staged:
        arguments.append("--cached")
    if ref is not None:
        arguments.append(_validate_ref(ref))
    arguments.append("--")
    if paths is not None:
        arguments.extend(_validate_git_paths(paths))
    result = await _run_git(workspace, arguments)
    return _command_response(result, workspace=workspace, operation="git_diff")


@mcp.tool(structured_output=False)
async def workspace_git_create_branch(
    workspace: str, branch: str, start_point: str = "HEAD"
) -> str:
    """Create a branch under the configured managed branch prefix."""
    branch = _validate_managed_branch(branch)
    start_point = _validate_ref(start_point, field="start point")
    async with _mutation_lock:
        result = await _run_git(workspace, ("branch", branch, start_point, "--"))
    return _command_response(result, workspace=workspace, operation="git_create_branch")


@mcp.tool(structured_output=False)
async def workspace_git_checkout(workspace: str, branch: str) -> str:
    """Switch to an existing branch under the configured managed prefix."""
    branch = _validate_managed_branch(branch)
    async with _mutation_lock:
        result = await _run_git(
            workspace, ("checkout", "--no-recurse-submodules", branch, "--")
        )
    return _command_response(result, workspace=workspace, operation="git_checkout")


@mcp.tool(structured_output=False)
async def workspace_git_add(workspace: str, paths: list[str]) -> str:
    """Stage an explicit bounded list of workspace paths."""
    workspace_path = _require_git_repository(workspace)
    validated_paths = _validate_git_paths(paths)
    for relative_path in validated_paths:
        candidate = workspace_path.joinpath(*PurePosixPath(relative_path).parts)
        _reject_symlink_components(candidate, workspace_path)
        _assert_contained(candidate, workspace_path)
    async with _mutation_lock:
        result = await _run_git(workspace, ("add", "--", *validated_paths))
    return _command_response(result, workspace=workspace, operation="git_add")


@mcp.tool(structured_output=False)
async def workspace_git_commit(workspace: str, message: str) -> str:
    """Commit staged changes only while on a managed, non-main branch."""
    message = _validate_text(
        message, field="message", max_chars=MAX_COMMIT_MESSAGE_CHARS
    )
    async with _mutation_lock:
        branch_result = await _run_git(
            workspace, ("symbolic-ref", "--quiet", "--short", "HEAD")
        )
        current_branch = branch_result.stdout.strip()
        if current_branch in {"main", "master"}:
            raise ValueError("commits to main or master are forbidden")
        _validate_managed_branch(current_branch)
        result = await _run_git(
            workspace,
            ("commit", "--no-verify", "--no-gpg-sign", "-m", message, "--"),
        )
    return _command_response(result, workspace=workspace, operation="git_commit")


@mcp.tool(structured_output=False)
async def workspace_git_log(
    workspace: str, max_count: int = 20, ref: str = "HEAD"
) -> str:
    """Show a bounded commit log for one validated ref."""
    max_count = _validate_positive_count(max_count, field="max_count", maximum=100)
    ref = _validate_ref(ref)
    result = await _run_git(
        workspace,
        (
            "log",
            f"--max-count={max_count}",
            "--date=iso-strict",
            "--pretty=format:%H%x09%ad%x09%an%x09%s",
            ref,
            "--",
        ),
    )
    return _command_response(result, workspace=workspace, operation="git_log")


@mcp.tool(structured_output=False)
async def workspace_run_check(workspace: str, profile: str) -> str:
    """Run one fixed check profile inside a network-denied macOS sandbox."""
    if not _runner_enabled():
        raise RuntimeError("workspace runner is disabled")
    if not isinstance(profile, str) or profile not in RUNNER_PROFILES:
        allowed = ", ".join(sorted(RUNNER_PROFILES))
        raise ValueError(f"unknown check profile; allowed profiles: {allowed}")
    workspace_path = _workspace_path(workspace)
    sandbox = _sandbox_executable()
    command = RUNNER_PROFILES[profile]
    if command[0] == "npm":
        npm = shutil.which("npm", path=SAFE_PATH)
        if not npm:
            raise RuntimeError("npm is unavailable")
        command = (str(Path(npm).resolve()), *command[1:])

    temporary = Path(tempfile.mkdtemp(prefix=".al1s-runner-", dir=workspace_path))
    try:
        argv = (
            sandbox,
            "-D",
            f"WORKSPACE={workspace_path}",
            "-D",
            f"GIT_METADATA={workspace_path / '.git'}",
            "-D",
            f"USER_HOME={Path.home().resolve()}",
            "-D",
            f"TOOLCHAIN={Path(sys.prefix).resolve()}",
            "-D",
            f"PYTHON_RUNTIME={Path(sys.base_prefix).resolve()}",
            "-p",
            SANDBOX_PROFILE,
            "--",
            *command,
        )
        result = await _execute_process(
            argv,
            cwd=workspace_path,
            environment=_runner_environment(workspace_path, temporary),
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if result.returncode in {71, 126, 127, 134}:
        raise RuntimeError("sandbox could not reliably execute the selected check")
    return _command_response(
        result, workspace=workspace, operation=f"run_check:{profile}"
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
