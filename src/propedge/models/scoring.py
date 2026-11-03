"""Proper scoring rules for probability forecasts (lower is better). Pure math."""

from __future__ import annotations

import numpy as np

EPS = 0.005    # probabilities are clipped to [EPS, 1 − EPS] for log loss (a 0.00 market mid would be −∞)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.clip(np.asarray(p, float), EPS, 1 - EPS), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")
