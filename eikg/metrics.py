"""Small metric helpers with sklearn-free fallback."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mean_squared_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    scale = float(max(np.max(np.abs(true)), np.max(np.abs(pred))))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return float(np.inf)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        difference = true / scale - pred / scale
        root_mse = scale * float(np.sqrt(np.mean(difference * difference)))
        result = root_mse * root_mse
    return float(result)


def r2_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    scale = float(max(np.max(np.abs(true)), np.max(np.abs(pred))))
    if scale == 0.0:
        return 1.0
    if not np.isfinite(scale):
        return float(np.nan)
    true_scaled = true / scale
    pred_scaled = pred / scale
    residual = true_scaled - pred_scaled
    centered = true_scaled - np.mean(true_scaled)
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum(centered * centered))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)
