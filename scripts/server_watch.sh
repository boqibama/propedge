#!/usr/bin/env bash
# Run ON THE MAC: the live read-only `watch` screen, running on the server. Ctrl-C to quit (the poller
# keeps going). `scripts/server_watch.sh log` follows the server's raw log instead.
set -euo pipefail
HOST="${PROPEDGE_HOST:-propedge-server}"
if [ "${1:-}" = "log" ]; then
  exec ssh -t "$HOST" 'journalctl -u propedge-tonight -f -n 50'
fi
exec ssh -t "$HOST" 'cd ~/propedge && .venv/bin/python -m propedge watch'
