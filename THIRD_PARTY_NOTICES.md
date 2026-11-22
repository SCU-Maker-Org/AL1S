# Third-Party Notices

yachiyo itself is distributed under the MIT License. No third-party source code is vendored
or copied into this repository. Runtime packages are consumed through their published
APIs and retain their own licenses.

| Component | License | Purpose |
| --- | --- | --- |
| aiogram | MIT | Telegram Bot API transport |
| discord.py | MIT | Discord gateway and application commands |
| aiohttp | Apache-2.0 AND MIT | Asynchronous HTTP integrations |
| aiohttp-socks | Apache-2.0 | Optional Telegram HTTP/SOCKS proxy support |
| OpenAI Python | Apache-2.0 | OpenAI-compatible chat and tool calls |
| Pydantic | MIT | Validation and configuration |
| platformdirs | MIT | Cross-platform configuration paths |
| aiosqlite | MIT | Async bridge to the standard SQLite module |
| selectolax | MIT | HTML parsing and text extraction |
| puremagic | MIT | Attachment type validation |
| MCP Python SDK | MIT | Optional workspace MCP server |
| Ruff | MIT | Development linting and fixed workspace checks |

Release builds must regenerate an SBOM from the locked dependency graph. Any future
copied source must record its upstream repository, exact commit, original file,
license, copyright notice and local modifications in this document.
