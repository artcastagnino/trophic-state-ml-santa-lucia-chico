# my_metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer


def round_sig(value, sig=3):
    """
    Redondea a 'sig' cifras significativas.
    """
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0:
        return np.nan

    return 1 - (np.sum((y_true - y_pred)**2) / denom)


def kling_gupta_efficiency(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.std(y_true) == 0 or np.sum(y_true) == 0:
        return np.nan

    r = np.corrcoef(y_pred, y_true)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.sum(y_pred) / np.sum(y_true)

    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def percent_bias(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.sum(y_true)
    if denom == 0:
        return np.nan

    return 100.0 * np.sum(y_pred - y_true) / denom


def r_squared_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    return r**2


def rsr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0:
        return np.nan

    numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
    return numerator / denom


def evaluate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    metrics_train = {
        'RMSE': round_sig(np.sqrt(mean_squared_error(y_train, y_pred_train)), 2),
        'MAE': round_sig(mean_absolute_error(y_train, y_pred_train), 2),
        'NSE': round_sig(nash_sutcliffe_efficiency(y_train, y_pred_train), 2),
        'KGE': round_sig(kling_gupta_efficiency(y_train, y_pred_train), 2),
        'PBIAS': round_sig(percent_bias(y_train, y_pred_train), 2),
        'R2': round_sig(r_squared_corr(y_train, y_pred_train), 2),
        'RSR': round_sig(rsr(y_train, y_pred_train), 2),
    }

    metrics_test = {
        'RMSE': round_sig(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2),
        'MAE': round_sig(mean_absolute_error(y_test, y_pred_test), 2),
        'NSE': round_sig(nash_sutcliffe_efficiency(y_test, y_pred_test), 2),
        'KGE': round_sig(kling_gupta_efficiency(y_test, y_pred_test), 2),
        'PBIAS': round_sig(percent_bias(y_test, y_pred_test), 2),
        'R2': round_sig(r_squared_corr(y_test, y_pred_test), 2),
        'RSR': round_sig(rsr(y_test, y_pred_test), 2),
    }

    return metrics_train, metrics_test


nse_scorer = make_scorer(nash_sutcliffe_efficiency, greater_is_better=True)