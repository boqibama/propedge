#!/usr/bin/env bash
# Run ON THE SERVER (Ubuntu 24.04, user "ubuntu") after scripts/server_deploy.sh has copied the code to
# ~/propedge. Safe to re-run. Installs Python 3.12 with uv, the packages, the systemd service + timer.
set -euo pipefail
cd "$HOME/propedge"

echo "== system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq rsync sqlite3 curl ca-certificates >/dev/null
sudo timedatectl set-timezone America/Los_Angeles     # readable log times; game days are Eastern in code

# A swap file helps on the 1 GB micro shape (pandas + the live engine); harmless elsewhere.
if [ ! -f /swapfile ] && [ "$(awk '/MemTotal/ {print $2}' /proc/meminfo)" -lt 4000000 ]; then
  echo "== adding a 2 GB swap file"
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile && echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== Python 3.12 + packages (uv)"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV" python install 3.12
[ -d .venv ] || "$UV" venv --python 3.12 .venv
# The dashboard (streamlit) isn't needed on the server.
grep -v -i streamlit requirements.txt > /tmp/requirements-server.txt
"$UV" pip install --python .venv/bin/python -q -r /tmp/requirements-server.txt -e .

echo "== tests (no network)"
.venv/bin/python -m pytest -q | tail -1

mkdir -p data/logs
echo "== systemd service + timer"
sudo cp deploy/propedge-tonight.service deploy/propedge-tonight.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now propedge-tonight.timer
systemctl list-timers propedge-tonight.timer --no-pager | head -3

echo "== quick reachability check"
.venv/bin/python - <<'EOF'
from datetime import date
from propedge.config import load_config
from propedge.data.espn import EspnLiveProvider
from propedge.data.nba_stats import NbaStatsClient
from propedge.odds.kalshi import KalshiOddsProvider
cfg = load_config()
g = EspnLiveProvider().scoreboard(date.today())
print("ESPN scoreboard:", "OK" if g is not None else "FAILED", f"({len(g or [])} games today)")
print("Kalshi fee info:", KalshiOddsProvider(config=cfg).series_fee_info("pts") or "FAILED")
print("stats.nba.com:", "reachable" if NbaStatsClient(cfg).reachable(15) else "blocked (expected on cloud; ESPN fallback)")
EOF
echo "Done. Start tonight's run now (instead of waiting for 10:00 PT):  sudo systemctl start propedge-tonight"
