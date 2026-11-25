# Security

Do not place Bot tokens or API keys in `config/config.toml`, command arguments, source control
or container environment dumps. The user Quadlet mounts owner-only credential files read-only under `/run/secrets`.
Podman-managed secrets can also be mounted there.

Treat every MCP server as executable authority. Configure only reviewed endpoints,
allowlist tools and prefer `read_only=true`. Write confirmation is enabled by
default. Setting `confirm_writes=false` authorizes direct execution of the
server's allowlisted tools for everyone covered by its `access` setting.
Never mount the Podman socket into the yachiyo core or workspace container.

Group observation and learning are disabled by default. Administrators must explicitly
enable group memory; users can inspect and delete scoped memory with `/knowledge` and
`/forget`.

Report vulnerabilities through the repository’s Security tab on GitHub. Include the affected
version, configuration, reproduction steps and impact, but never include live secrets.
