#!/usr/bin/env bash
# Run ON THE MAC: bring a consistent copy of the server's database (and new recordings) to
# data/server/ — never over this Mac's own data/propedge.sqlite. Safe while the server is polling.
# Then run any report against it, e.g.:
#   .venv/bin/python -m propedge --db sqlite:///data/server/propedge.sqlite rules-report
set -euo pipefail
HOST="${PROPEDGE_HOST:-propedge-server}"
cd "$(dirname "$0")/.."
ssh "$HOST" "cd ~/propedge && .venv/bin/python -c \"import sqlite3; s=sqlite3.connect('data/propedge.sqlite'); d=sqlite3.connect('data/pull-copy.sqlite'); s.backup(d); d.close()\""
mkdir -p data/server
rsync -a "$HOST:propedge/data/pull-copy.sqlite" data/server/propedge.sqlite
rsync -a "$HOST:propedge/data/recordings/" data/server/recordings/
ssh "$HOST" 'rm -f ~/propedge/data/pull-copy.sqlite'
echo "Copied to data/server/propedge.sqlite"
