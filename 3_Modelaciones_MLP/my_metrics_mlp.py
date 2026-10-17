# my_metrics_mlp.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def round_sig(value, sig=3):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(value) or np.isinf(value) or value == 0:
        return value
    return float(f"{value:.{sig}g}")


def nash_sutcliffe_efficiency(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def kling_gupta_efficiency(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    std_true = np.std(y_true, ddof=1)
    std_pred = np.std(y_pred, ddof=1)

    if std_true == 0 or mean_true == 0:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    alpha = std_pred / std_true if std_true != 0 else np.nan
    beta = mean_pred / mean_true if mean_true != 0 else np.nan

    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan

    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def pbias(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(y_true)
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(y_pred - y_true) / denom


def r_squared_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return r ** 2


def rsr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0:
        return np.nan
    numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
    return numerator / denom


def compute_regression_metrics(y_true, y_pred):
    return {
        "RMSE": round_sig(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAE": round_sig(mean_absolute_error(y_true, y_pred), 2),
        "NSE": round_sig(nash_sutcliffe_efficiency(y_true, y_pred), 2),
        "KGE": round_sig(kling_gupta_efficiency(y_true, y_pred), 2),
        "PBIAS": round_sig(pbias(y_true, y_pred), 2),
        "R2": round_sig(r_squared_corr(y_true, y_pred), 2),
        "RSR": round_sig(rsr(y_true, y_pred), 2),
    }
