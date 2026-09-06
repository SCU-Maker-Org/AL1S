"""Narrow workspace file, Git and fixed-check operations."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import ClassVar


class WorkspaceError(RuntimeError):
    pass


class WorkspaceService:
    _COMPILE_CHECK: ClassVar[str] = """\
from pathlib import Path
errors = []
for path in Path('.').rglob('*.py'):
    if any(part in {'.git', '.venv', 'venv', 'node_modules'} for part in path.parts):
        continue
    try:
        compile(path.read_bytes(), str(path), 'exec')
    except (OSError, SyntaxError, UnicodeError) as exc:
        errors.append(f'{path}: {exc}')
for error in errors:
    print(error)
raise SystemExit(bool(errors))
"""
    CHECKS: ClassVar[dict[str, tuple[str, ...]]] = {
        "compile": ("python", "-c", _COMPILE_CHECK),
        "pytest": ("python", "-m", "pytest", "-q", "-p", "no:cacheprovider"),
        "ruff-check": ("ruff", "check", "--no-cache", "."),
        "ruff-format": ("ruff", "format", "--no-cache", "--check", "."),
    }

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int,
        max_results: int,
        command_timeout_seconds: int,
    ):
        self.root = root.expanduser().resolve()
        self.max_file_bytes = max_file_bytes
        self.max_results = max_results
        self.command_timeout_seconds = command_timeout_seconds
        self._mutation_lock = asyncio.Lock()

    def _workspace(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise WorkspaceError("invalid workspace name")
        path = self.root / name
        resolved = path.resolve(strict=False)
        if resolved.parent != self.root or path.is_symlink():
            raise WorkspaceError("workspace escapes the configured root")
        return resolved

    def _path(self, workspace: str, relative: str, *, must_exist: bool = False) -> Path:
        pure = PurePosixPath(relative)
        forbidden = {".git", ".gitattributes", ".gitmodules"}
        if pure.is_absolute() or ".." in pure.parts or forbidden.intersection(pure.parts):
            raise WorkspaceError("unsafe relative path")
        base = self._workspace(workspace)
        candidate = base.joinpath(*pure.parts)
        resolved = candidate.resolve(strict=must_exist)
        if resolved != base and not resolved.is_relative_to(base):
            raise WorkspaceError("path escapes the workspace")
        current = base
        for part in pure.parts:
            current /= part
            if current.is_symlink():
                raise WorkspaceError("symbolic links are forbidden")
        return resolved

    async def create(self, name: str) -> str:
        path = self._workspace(name)
        async with self._mutation_lock:
            await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True, mode=0o700)
        return str(path)

    async def list_files(self, workspace: str, relative: str = ".") -> list[str]:
        base = self._workspace(workspace)
        target = self._path(workspace, relative, must_exist=True)
        async with self._mutation_lock:
            return await asyncio.to_thread(self._list_sync, base, target)

    def _list_sync(self, base: Path, target: Path) -> list[str]:
        if not target.is_dir():
            raise WorkspaceError("path is not a directory")
        result = []
        for path in target.rglob("*"):
            relative = path.relative_to(base)
            if path.is_symlink() or ".git" in relative.parts:
                continue
            result.append(relative.as_posix() + ("/" if path.is_dir() else ""))
            if len(result) >= self.max_results:
                break
        return sorted(result)

    async def read(self, workspace: str, relative: str) -> str:
        path = self._path(workspace, relative, must_exist=True)
        async with self._mutation_lock:
            return await asyncio.to_thread(self._read_sync, path)

    def _read_sync(self, path: Path) -> str:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > self.max_file_bytes:
                raise WorkspaceError("file is missing or exceeds the size limit")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                raw = handle.read(self.max_file_bytes + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(raw) > self.max_file_bytes:
            raise WorkspaceError("file exceeds the size limit")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError("file is not UTF-8 text") from exc

    async def write(self, workspace: str, relative: str, content: str) -> str:
        encoded = content.encode()
        if len(encoded) > self.max_file_bytes:
            raise WorkspaceError("content exceeds the size limit")
        path = self._path(workspace, relative)
        async with self._mutation_lock:
            await asyncio.to_thread(self._write_sync, path, encoded)
        return path.relative_to(self._workspace(workspace)).as_posix()

    @staticmethod
    def _write_sync(path: Path, encoded: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".al1s-tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    async def replace(self, workspace: str, relative: str, old: str, new: str) -> str:
        if not old:
            raise WorkspaceError("old text must not be empty")
        path = self._path(workspace, relative, must_exist=True)
        async with self._mutation_lock:
            content = await asyncio.to_thread(self._read_sync, path)
            count = content.count(old)
            if count != 1:
                raise WorkspaceError(f"expected exactly one match, found {count}")
            encoded = content.replace(old, new, 1).encode()
            if len(encoded) > self.max_file_bytes:
                raise WorkspaceError("content exceeds the size limit")
            await asyncio.to_thread(self._write_sync, path, encoded)
        return path.relative_to(self._workspace(workspace)).as_posix()

    async def search(self, workspace: str, query: str) -> list[dict[str, object]]:
        if not query or len(query) > 500:
            raise WorkspaceError("query length is invalid")
        base = self._workspace(workspace)

        def collect() -> list[dict[str, object]]:
            result: list[dict[str, object]] = []
            scanned = 0
            scan_limit = max(1000, self.max_results * 100)
            for path in base.rglob("*"):
                relative_path = path.relative_to(base)
                if path.is_symlink() or ".git" in relative_path.parts or path.is_dir():
                    continue
                scanned += 1
                if scanned > scan_limit:
                    raise WorkspaceError("workspace search scan limit exceeded")
                relative = relative_path.as_posix()
                try:
                    text = self._read_sync(path)
                except WorkspaceError:
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if query in line:
                        result.append({"path": relative, "line": line_number, "text": line[:1000]})
                        if len(result) >= self.max_results:
                            return result
            return result

        async with self._mutation_lock:
            return await asyncio.to_thread(collect)

    async def git(self, workspace: str, args: tuple[str, ...]) -> str:
        if not args:
            raise WorkspaceError("Git operation is not allowed")
        if any("\x00" in argument or len(argument) > 1000 for argument in args):
            raise WorkspaceError("invalid Git argument")
        validated = self._git_args(workspace, args)
        root = self._workspace(workspace)
        async with self._mutation_lock:
            if args[0] == "init" and (root / ".git").exists():
                raise WorkspaceError("refusing to reinitialize an existing repository")
            if args[0] != "init":
                self._verify_owned_repository(root)
            command = (
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "diff.external=",
                "-c",
                "protocol.file.allow=never",
                *validated,
            )
            output = await self._run(
                root,
                self._sandbox_command(root, command, read_only=False),
            )
            if args[0] == "init":
                marker = root / ".git" / "al1s-owned"
                await asyncio.to_thread(marker.write_text, "managed\n", encoding="utf-8")
            return output

    @staticmethod
    def _verify_owned_repository(root: Path) -> None:
        marker = root / ".git" / "al1s-owned"
        if not marker.is_file() or marker.is_symlink():
            raise WorkspaceError("repository was not initialized by AL1S")
        if any(root.rglob(".gitattributes")) or any(root.rglob(".gitmodules")):
            raise WorkspaceError("repository contains unsafe Git control files")
        info_attributes = root / ".git" / "info" / "attributes"
        if info_attributes.exists():
            raise WorkspaceError("repository contains unsafe Git attributes")
        config = root / ".git" / "config"
        try:
            config_text = config.read_text(encoding="utf-8").casefold()
        except (OSError, UnicodeDecodeError) as exc:
            raise WorkspaceError("repository configuration is unreadable") from exc
        forbidden = ("[filter ", "[include", "fsmonitor", "external", "command =")
        if any(item in config_text for item in forbidden):
            raise WorkspaceError("repository configuration contains executable helpers")

    def _git_args(self, workspace: str, args: tuple[str, ...]) -> tuple[str, ...]:
        operation, rest = args[0], args[1:]
        if operation == "status":
            allowed = {"--short", "--branch", "--porcelain", "--untracked-files=all"}
            if any(argument not in allowed for argument in rest):
                raise WorkspaceError("unsupported git status argument")
        elif operation == "diff":
            flags = {"--cached", "--stat", "--name-only", "--color=never"}
            for argument in rest:
                if argument in flags or argument == "--":
                    continue
                if argument.startswith("-"):
                    raise WorkspaceError("unsupported git diff argument")
                self._path(workspace, argument)
        elif operation == "log":
            if any(
                not re.fullmatch(r"--(?:oneline|decorate|all|no-color)|-[1-9][0-9]{0,2}", item)
                for item in rest
            ):
                raise WorkspaceError("unsupported git log argument")
        elif operation == "init":
            if len(rest) > 1 or any(
                not re.fullmatch(r"--initial-branch=[A-Za-z0-9][A-Za-z0-9._/-]{0,100}", item)
                for item in rest
            ):
                raise WorkspaceError("unsupported git init argument")
        elif operation == "add":
            paths = tuple(item for item in rest if item != "--")
            if not paths or any(item.startswith("-") for item in paths):
                raise WorkspaceError("git add requires explicit paths")
            for item in paths:
                self._path(workspace, item)
        elif operation == "commit":
            if len(rest) != 2 or rest[0] != "-m" or not rest[1].strip():
                raise WorkspaceError("git commit requires exactly -m MESSAGE")
        elif operation == "branch":
            if len(rest) > 1 or any(not self._branch_name(item) for item in rest):
                raise WorkspaceError("unsupported git branch argument")
        elif operation == "switch":
            if len(rest) not in {1, 2}:
                raise WorkspaceError("unsupported git switch arguments")
            branch = rest[-1]
            if (len(rest) == 2 and rest[0] != "-c") or not self._branch_name(branch):
                raise WorkspaceError("unsupported git switch arguments")
        else:
            raise WorkspaceError("Git operation is not allowed")
        return args

    @staticmethod
    def _branch_name(value: str) -> bool:
        return bool(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,100}", value)
            and ".." not in value
            and not value.endswith((".", "/"))
            and value not in {"main", "master"}
        )

    async def run_check(self, workspace: str, profile: str) -> str:
        command = self.CHECKS.get(profile)
        if command is None:
            raise WorkspaceError("unknown check profile")
        async with self._mutation_lock:
            root = self._workspace(workspace)
            return await self._run(root, self._sandbox_command(root, command))

    def _sandbox_command(
        self,
        workspace: Path,
        command: tuple[str, ...],
        *,
        read_only: bool = True,
    ) -> tuple[str, ...]:
        executable = shutil.which("bwrap", path="/usr/bin:/bin")
        if executable is None or Path(executable).resolve() != Path("/usr/bin/bwrap"):
            raise WorkspaceError("bubblewrap is required for workspace checks")
        arguments = [
            executable,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--unshare-user",
            "--cap-drop",
            "ALL",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/run",
            "--dir",
            "/home",
            "--dir",
            "/root",
            "--dir",
            "/etc",
        ]
        for allowed in (
            "/usr",
            "/bin",
            "/lib",
            "/lib64",
            "/opt/venv",
            "/etc/ssl",
            "/etc/ld.so.cache",
        ):
            if Path(allowed).exists():
                arguments.extend(("--ro-bind", allowed, allowed))
        created = {Path("/tmp"), Path("/home"), Path("/root"), Path("/etc")}
        for parent in reversed(workspace.parents):
            if parent == Path("/") or parent in created:
                continue
            arguments.extend(("--dir", str(parent)))
            created.add(parent)
        arguments.extend(
            (
                "--ro-bind" if read_only else "--bind",
                str(workspace),
                str(workspace),
                "--chdir",
                str(workspace),
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "PATH",
                "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
                "--",
                *command,
            )
        )
        return tuple(arguments)

    async def _run(self, cwd: Path, command: tuple[str, ...]) -> str:
        if not await asyncio.to_thread(cwd.is_dir):
            raise WorkspaceError("workspace does not exist")
        environment = {
            "PATH": os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(cwd),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "AL1S Workspace",
            "GIT_AUTHOR_EMAIL": "al1s@localhost",
            "GIT_COMMITTER_NAME": "AL1S Workspace",
            "GIT_COMMITTER_EMAIL": "al1s@localhost",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None

        async def collect() -> bytes:
            output = bytearray()
            while chunk := await process.stdout.read(64 * 1024):
                output.extend(chunk)
                if len(output) > 256_000:
                    raise WorkspaceError("command output exceeded the size limit")
            await process.wait()
            return bytes(output)

        try:
            try:
                stdout = await asyncio.wait_for(collect(), timeout=self.command_timeout_seconds)
            except TimeoutError:
                raise WorkspaceError("command timed out") from None
            output = stdout.decode("utf-8", errors="replace")
            if process.returncode:
                raise WorkspaceError(
                    f"command failed with exit code {process.returncode}:\n{output}"
                )
            return output
        finally:
            await self._terminate_process_group(process)

    @staticmethod
    async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.2)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.returncode is None:
            await process.wait()
