#!/bin/sh
set -eu

app_root="${AL1S_APP_ROOT:-/app}"

mkdir -p \
    "${app_root}/.cache" \
    "${app_root}/.npm" \
    "${app_root}/.npm-global" \
    "${app_root}/data" \
    "${app_root}/logs"

if [ ! -f "${app_root}/data/init_db.sql" ]; then
    cp "${app_root}/share/init_db.sql" "${app_root}/data/init_db.sql"
fi

exec "$@"
