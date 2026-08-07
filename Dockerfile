FROM node:24.19.0-bookworm-slim@sha256:3638d9a6fe4030bd716be989438248074489337ba3275657f93595428be4fc03 AS node-runtime

FROM ghcr.io/github/github-mcp-server:v1.8.0@sha256:d5a18c04b92714c309eb46a2305087e91a4dbd80420f6e462656699f95093520 AS github-mcp-runtime

FROM python:3.13.13-slim-bookworm@sha256:355bfa66770995d7e9a0da4b3473b44d0cb451f6b56f5615ad9c39e3c4eca03f

ARG AL1S_VERSION=0.1.0
ARG APP_UID=10001
ARG APP_GID=10001

LABEL org.opencontainers.image.title="AL1S" \
      org.opencontainers.image.description="Telegram AI bot with RAG, MCP, media generation, and isolated development workspaces" \
      org.opencontainers.image.version="${AL1S_VERSION}" \
      org.opencontainers.image.source="https://github.com/SCU-Maker-Org/AL1S" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOME=/app \
    PATH="/app/.venv/bin:/app/.npm-global/bin:/usr/local/bin:/usr/bin:/bin" \
    UV_CACHE_DIR=/app/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    NPM_CONFIG_CACHE=/app/.npm \
    NPM_CONFIG_PREFIX=/app/.npm-global \
    NPM_CONFIG_USERCONFIG=/app/.npmrc \
    AL1S_APP_ROOT=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libatomic1 \
        libgomp1 \
        libstdc++6 \
        openssh-client \
        passwd \
        tini \
        tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" botuser \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --home-dir /app --shell /usr/sbin/nologin botuser \
    && mkdir -p \
        /app/.cache \
        /app/.npm \
        /app/.npm-global \
        /app/data \
        /app/knowledge \
        /app/logs \
        /app/share \
    && chown -R botuser:botuser /app

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=github-mcp-runtime /server/github-mcp-server /usr/local/bin/github-mcp-server
COPY --from=ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6 /uv /uvx /usr/local/bin/

RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version \
    && npm --version \
    && uv --version \
    && git --version \
    && github-mcp-server --version

COPY --chown=botuser:botuser pyproject.toml uv.lock README.md LICENSE ./

USER botuser

RUN uv sync --frozen --no-dev --no-install-project

ENV UV_FROZEN=1 \
    UV_NO_SYNC=1

COPY --chown=botuser:botuser main.py config.example.toml ./
COPY --chown=botuser:botuser src/ ./src/
COPY --chown=botuser:botuser knowledge/ ./knowledge/
COPY --chown=botuser:botuser scripts/ ./scripts/
COPY --chown=botuser:botuser data/init_db.sql ./share/init_db.sql
COPY --chown=botuser:botuser --chmod=755 scripts/container_entrypoint.sh ./scripts/container_entrypoint.sh

HEALTHCHECK --interval=60s --timeout=15s --start-period=5m --retries=3 \
    CMD ["python", "scripts/container_healthcheck.py"]

ENTRYPOINT ["/usr/bin/tini", "--", "/app/scripts/container_entrypoint.sh"]
CMD ["python", "main.py"]
