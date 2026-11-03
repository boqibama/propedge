#!/usr/bin/env bash
# Run ON THE MAC, ONCE, before the server's first night: copies the history (database, raw recordings,
# NBA stats cache) to the server. After this the SERVER is the only one writing live data.
# Refuses to overwrite a server database that already exists.
set -euo pipefail
HOST="${PROPEDGE_HOST:-propedge-server}"
cd "$(dirname "$0")/.."
if ssh "$HOST" 'test -e ~/propedge/data/propedge.sqlite'; then
  echo "The server already has a database — not overwriting it. (Delete it there by hand if you mean to.)" >&2
  exit 1
fi
if pgrep -f "propedge (tonight|poll)" >/dev/null; then
  echo "A poller is running on this Mac — run this after it finishes." >&2
  exit 1
fi
mkdir -p data/tmp
# A consistent copy of a WAL database (never copy the live file directly).
.venv/bin/python -c "import sqlite3; s=sqlite3.connect('data/propedge.sqlite'); d=sqlite3.connect('data/tmp/seed.sqlite'); s.backup(d); d.close()"
ssh "$HOST" 'mkdir -p ~/propedge/data/logs ~/propedge/data/cache'
rsync -a data/tmp/seed.sqlite "$HOST:propedge/data/propedge.sqlite"
rsync -a data/recordings "$HOST:propedge/data/"
[ -d data/cache/nba_stats ] && rsync -a data/cache/nba_stats "$HOST:propedge/data/cache/"
rm -f data/tmp/seed.sqlite
echo "Seeded $HOST. Check with: ssh $HOST 'cd propedge && .venv/bin/python -m propedge status'"
