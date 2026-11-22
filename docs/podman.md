# Rootless Podman with Quadlet

The core runs as the user service `yachiyo.service`. From the checkout root,
prepare `config/config.toml` and `config/.secrets/` as described in the
[local setup](../README.md#local-setup). Each enabled integration reads the
credential file named in its TOML settings. Keep the files owned by your
account with mode `0600`.

## Install and start

```sh
podman build --target core -f config/Containerfile -t localhost/yachiyo:2.0.0 .
quadlets="${XDG_CONFIG_HOME:-$HOME/.config}/containers/systemd"
mkdir -p "$quadlets"
ln -sT "$PWD/config" "$quadlets/yachiyo"
systemctl --user daemon-reload
systemctl --user start yachiyo.service
```

Link the directory once. Quadlet resolves the TOML and secret mounts relative to
the unit. It mounts both read-only and maps the container user to your host UID.
SQLite and pending reply attachments persist in `~/.local/share/yachiyo/`. The core
uses outbound connections and publishes no host ports.

When the build uses a proxy listening on host loopback, add `--network=host` to the
build command so uv can reach the proxy. For a host HTTP proxy on port 10808,
create `config/yachiyo.container.d/local.conf` with the following settings.
The mapped address reaches host loopback from the container.

```ini
[Container]
Network=pasta:--map-host-loopback,169.254.1.1
Environment=YACHIYO__TRANSPORTS__TELEGRAM__PROXY_URL=http://169.254.1.1:10808
Environment=YACHIYO__TRANSPORTS__DISCORD__PROXY_URL=http://169.254.1.1:10808
Environment=HTTPS_PROXY=http://169.254.1.1:10808 https_proxy=http://169.254.1.1:10808
Environment=HTTP_PROXY=http://169.254.1.1:10808 http_proxy=http://169.254.1.1:10808
Environment=ALL_PROXY=http://169.254.1.1:10808 all_proxy=http://169.254.1.1:10808
```

## Manage the service

```sh
systemctl --user status yachiyo
podman exec yachiyo yachiyo status
journalctl --user -u yachiyo -f
systemctl --user restart yachiyo
systemctl --user stop yachiyo
```

After editing the TOML or credentials, restart the service. After updating the
code, rebuild the image with the same tag and restart. Changes to the Quadlet
also require `systemctl --user daemon-reload`. Its `[Install]` section starts
it with the user's default target. For startup before login and operation after
logout, the host account needs lingering enabled.

## Optional workspace server

Deploy the workspace only when the configured MCP tools need it. Build its
separate target with

```sh
podman build --target workspace-runner -f config/Containerfile -t localhost/yachiyo-workspace:2.0.0 .
```

For a workspace deployment, add a user Quadlet named `yachiyo-workspace.container`
and an internal `yachiyo-tools.network`. Keep the core on its outbound network and
add the tools network to both services. Mount the TOML read-only in the workspace,
give it a writable `/workspaces` directory, and mount only `workspace_mcp_token`
under `/run/secrets` with mode `0400` for UID 10001. The Podman socket stays on the
host. Enable the `workspace` entry in the core's MCP settings and configure an
administrator before using its private admin tools.
