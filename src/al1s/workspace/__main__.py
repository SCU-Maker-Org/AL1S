"""Optional workspace MCP console entrypoint."""

from __future__ import annotations


def main() -> None:
    try:
        from .server import main as server_main
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            raise SystemExit(
                "al1s-workspace-server requires the 'workspace' optional dependency"
            ) from None
        raise
    server_main()


if __name__ == "__main__":
    main()
