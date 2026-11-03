"""Console + rotating-file logging, configured from config.yaml."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from propedge.config import load_config, resolve_path


def setup_logging(verbose: bool = False) -> None:
    cfg = load_config()
    level = logging.DEBUG if verbose else getattr(logging, cfg["logging"]["level"])
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    log_file = resolve_path(cfg["paths"]["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=cfg["logging"]["max_bytes"],
                                       backupCount=cfg["logging"]["backups"])
    console = logging.StreamHandler()
    root = logging.getLogger()
    root.handlers.clear()
    for h in (file_handler, console):
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(level)
    for noisy in ("urllib3", "requests_cache"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
