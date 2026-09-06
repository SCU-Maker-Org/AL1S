# Security

Do not place Bot tokens or API keys in `config.toml`, command arguments, source control
or container environment dumps. Use Podman secrets mounted under `/run/secrets`.

Treat every MCP server as executable authority. Configure only reviewed endpoints,
allowlist tools, prefer `read_only=true`, and require application approval for writes.
Never mount the Podman socket into the AL1S core or workspace container.

Group observation and learning are disabled by default. Administrators must explicitly
enable group memory; users can inspect and delete scoped memory with `/knowledge` and
`/forget`.

Report vulnerabilities through GitHub private vulnerability reporting:
https://github.com/SCU-Maker-Org/AL1S/security/advisories/new. Include the affected
version, configuration, reproduction steps and impact, but never include live secrets.
