"""Import the Frontier AI subscriptions into the local rss-mcp service."""

import asyncio
from pathlib import Path

import aiohttp
from yachiyo.adapters.mcp import MCPError, _ServerClient
from yachiyo.config.models import MCPServerSettings

RSS_DIR = Path(__file__).resolve().parent


async def main() -> None:
    token = (RSS_DIR.parent / ".secrets" / "rss_mcp_token").read_text().strip()
    settings = MCPServerSettings(
        name="rss",
        url="http://127.0.0.1:8769/mcp",
        allow_insecure_http=True,
        allowed_tools=("import_opml",),
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        client = _ServerClient(settings, session, token)
        try:
            result = await client.call_tool(
                "import_opml", {"opml": (RSS_DIR / "frontier-ai.opml").read_text()}
            )
            if not result.ok:
                raise MCPError(result.content)
            print(result.content)
        finally:
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
