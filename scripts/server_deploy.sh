#!/usr/bin/env bash
# Run ON THE MAC. Copies the committed code (git-tracked files only: no data, no .venv, no .env) to the
# server's ~/propedge. A run already in progress keeps its old code until its next start.
#   scripts/server_deploy.sh            copy code
#   scripts/server_deploy.sh --setup    copy code, then run deploy/setup_server.sh there (first time / new packages)
set -euo pipefail
HOST="${PROPEDGE_HOST:-propedge-server}"     # an entry in ~/.ssh/config (see deploy/README.md)
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain -- src config.yaml requirements.txt deploy)" ]; then
  echo "Uncommitted changes in src/config/deploy — commit first so the server runs a known version." >&2
  exit 1
fi
ssh "$HOST" 'mkdir -p ~/propedge'
git ls-files -z | rsync -a --from0 --files-from=- ./ "$HOST:propedge/"
ssh "$HOST" "cd ~/propedge && echo '$(git rev-parse --short HEAD)' > DEPLOYED_VERSION"
echo "Deployed $(git rev-parse --short HEAD) to $HOST:~/propedge"
if [ "${1:-}" = "--setup" ]; then
  ssh -t "$HOST" 'bash ~/propedge/deploy/setup_server.sh'
fi
