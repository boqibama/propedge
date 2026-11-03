**propedge**: a live NBA player-points model that compares its projections with Kalshi's player prop prices and flags the ones that look mispriced. Paper only, it never places a bet.

Building this with some friends.

What it does:
- Before games, projects every player's points and compares that with Kalshi's prices
- During games, polls the live box score from ESPN and updates projections as things happen (fouls, blowouts, players leaving)
- Logs the bets it *would* make, then grades them later against Kalshi's closing prices (Brier score, log loss, closing line value) and counts fees
- Backtests on last season's games, rebuilt from play-by-play, with no look-ahead

Where it's at: still testing it on real game nights. Nothing's proven yet.

To run it (Python 3.11+):
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
cp .env.example .env      # add your keys
python -m propedge tonight   # the nightly run: pre-game check, then live polling until the games end
pytest                       # tests, no network needed
```

`python -m propedge --help` lists everything else (backtests, replays of recorded nights, the scoreboard vs Kalshi, and a live `watch` view).

How it's laid out: everything in `src/propedge/models/` is pure math and never touches the network, so the model can be tested with fake data. `data/` and `odds/` talk to ESPN, nba_api and Kalshi, `ingest/` writes to a SQLite database, `edge/` compares the model with the market, and `backtest/` does the backtests. All the thresholds live in `config.yaml` instead of being hard-coded.

Built with Python, pandas, NumPy, SciPy, SQLAlchemy/SQLite, and nba_api. It runs nightly on a small server with a systemd timer (see `deploy/`).
