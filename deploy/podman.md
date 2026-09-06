# Podman quick start

Build the core image for the current architecture:

```sh
podman build -f deploy/Containerfile --target core -t localhost/al1s:2.0.0 .
```

Create secrets without placing them in TOML:

```sh
printf '%s' "$TELEGRAM_BOT_TOKEN" | podman secret create telegram_bot_token -
printf '%s' "$OPENAI_API_KEY" | podman secret create openai_api_key -
printf '%s' "$WORKSPACE_MCP_TOKEN" | podman secret create workspace_mcp_token -
```

Run with a read-only configuration and one persistent data volume:

```sh
podman volume create al1s-data
podman network create --internal al1s-tools
podman run -d --name al1s --replace \
  --network podman --network al1s-tools \
  --read-only --cap-drop=all --security-opt=no-new-privileges \
  --memory=512m --pids-limit=256 --stop-timeout=360 \
  --tmpfs=/tmp:rw,noexec,nosuid,size=64m \
  --health-cmd='al1s status' --health-interval=30s --health-retries=3 \
  --secret=telegram_bot_token,uid=10001,gid=10001,mode=0400 \
  --secret=openai_api_key,uid=10001,gid=10001,mode=0400 \
  --secret=workspace_mcp_token,uid=10001,gid=10001,mode=0400 \
  -v ./config.toml:/etc/al1s/config.toml:ro,z \
  -v al1s-data:/var/lib/al1s \
  localhost/al1s:2.0.0
```

Podman's read-only mode supplies a writable `/run` tmpfs while preserving the image's
`/run/al1s` ownership. Do not overlay `/run/al1s` with a separate root-owned tmpfs.

On macOS, initialize and start `podman machine` before using the same commands.

The optional workspace server is a separate image and must not receive the Podman socket:

```sh
podman build -f deploy/Containerfile --target workspace-runner \
  -t localhost/al1s-workspace:2.0.0 .
podman run -d --name al1s-workspace --replace --network al1s-tools \
  --read-only --cap-drop=all --security-opt=no-new-privileges \
  --memory=1g --pids-limit=256 --tmpfs=/tmp:rw,noexec,nosuid,size=256m \
  --secret=workspace_mcp_token,uid=10001,gid=10001,mode=0400 \
  -v ./config.toml:/etc/al1s/config.toml:ro,z \
  -v ./workspaces:/workspaces:rw,U,Z \
  localhost/al1s-workspace:2.0.0
```

Before starting the containers, set `[mcp].enabled = true`, enable the `workspace`
server entry, and configure at least one Telegram or Discord administrator. The example
workspace tools use `private_admin`, so non-administrators cannot invoke them.
