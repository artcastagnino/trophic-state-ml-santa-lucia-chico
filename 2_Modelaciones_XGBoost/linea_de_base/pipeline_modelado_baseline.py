# pipeline_modelado_baseline.py
# Línea de base XGBoost independiente, sin Optuna.

import os
import re
import sys
import time
import random
import inspect
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from my_metrics import (
    nash_sutcliffe_efficiency,
    evaluate_metrics,
    round_sig
)
from visualization import (
    plot_train_vs_actual,
    plot_test_vs_actual,
    plot_real_vs_pred,
    plot_full_temporal_results,
    shap_summary_plots,
    plot_tree_evolution_curve,
    plot_tree_evolution_rmse_curve
)
from reportes_html_baseline import (
    TeeOutput,
    save_log_txt,
    save_log_html,
    generate_html_report
)

# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)

def classify_nse_performance(nse_value):
    try:
        nse_value = float(nse_value)
    except (TypeError, ValueError):
        return None
    if np.isnan(nse_value):
        return None
    if nse_value > 0.65:
        return "Muy bueno"
    if 0.50 < nse_value <= 0.65:
        return "Bueno"
    if 0.35 < nse_value <= 0.50:
        return "Satisfactorio"
    return "No satisfactorio"


# ---------------------------------------------------------
# Presets reproducibles
# ---------------------------------------------------------
PRESET_CONFIGS = {
    "lotico_old": {
        "experiment_label": "lotico_old",
        "use_early_stopping": True,
        "early_stopping_rounds": 30,
        "final_val_size": 0.15,
        "search_space_ranges": {
            "n_estimators": ("int", 80, 400),
            "max_depth": ("int", 2, 4),
            "learning_rate": ("float", 0.02, 0.08, "log"),
            "subsample": ("float", 0.80, 1.00),
            "colsample_bytree": ("float", 0.70, 1.00),
            "min_child_weight": ("int", 4, 20),
            "gamma": ("float", 0.0, 3.0),
            "reg_alpha": ("float", 1e-3, 5.0, "log"),
            "reg_lambda": ("float", 0.5, 10.0, "log"),
        }
    },
    "lentico_old": {
        "experiment_label": "lentico_old",
        "use_early_stopping": False,
        "early_stopping_rounds": None,
        "final_val_size": None,
        "search_space_ranges": {
            "n_estimators": ("int", 100, 800),
            "max_depth": ("int", 2, 7),
            "learning_rate": ("float", 0.01, 0.1, "log"),
            "subsample": ("float", 0.7, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "min_child_weight": ("int", 1, 20),
            "gamma": ("float", 1e-4, 10.0, "log"),
            "reg_alpha": ("float", 1e-8, 10.0, "log"),
            "reg_lambda": ("float", 1e-5, 10.0, "log"),
        }
    }
}

DEFAULT_PRESET = "lotico_old"

# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------
def clean_feature_names(names):
    return [re.sub(r"[^\w]", "_", str(name)) for name in names]


def get_variable_name(var, local_vars):
    matches = [name for name, val in local_vars if val is var]
    return matches[0] if matches else "diccionario_sets"


def temporal_train_test_split(X, y, train_size=0.8):
    n = len(X)
    split_idx = int(n * train_size)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def temporal_subtrain_validation_split(X, y, val_size=0.15):
    n = len(X)
    val_size_n = max(1, int(np.ceil(n * val_size)))
    split_idx = n - val_size_n

    X_subtrain = X.iloc[:split_idx].copy()
    X_val = X.iloc[split_idx:].copy()
    y_subtrain = y.iloc[:split_idx].copy()
    y_val = y.iloc[split_idx:].copy()

    return X_subtrain, X_val, y_subtrain, y_val


def build_timestamped_output_dir(base_name="Resultados"):
    now = datetime.now()
    timestamp = f"{now.day}-{now.month}--{now.hour}-{now.minute:02d}"
    return f"{base_name}_{timestamp}"


def format_elapsed_time(seconds):
    seconds = float(seconds)

def round_params_dict(params, sig=3):
    rounded = {}
    for k, v in params.items():
        if isinstance(v, (int, np.integer)):
            rounded[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            rounded[k] = round_sig(v, sig)
        else:
            rounded[k] = v
    return rounded


def get_xgb_baseline_defaults(n_estimators=2500):
    """
    Valores default/referencia explícitos de XGBoost para la línea de base.
    n_estimators no se considera default: es el máximo fijado por el experimento
    para permitir early stopping.
    """
    return {
        "n_estimators": int(n_estimators),
        "learning_rate": 0.3,
        "max_depth": 6,
        "min_child_weight": 1,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "subsample": 1,
        "colsample_bytree": 1,
    }


def get_xgb_baseline_defaults_text(n_estimators=2500):
    params = get_xgb_baseline_defaults(n_estimators=n_estimators)
    return {
        "n_estimators": f"{params['n_estimators']} (máximo fijado para early stopping)",
        "learning_rate": "0.3 (default/referencia XGBoost)",
        "max_depth": "6 (default/referencia XGBoost)",
        "min_child_weight": "1 (default/referencia XGBoost)",
        "gamma": "0 (default/referencia XGBoost)",
        "reg_alpha": "0 (default/referencia XGBoost)",
        "reg_lambda": "1 (default/referencia XGBoost)",
        "subsample": "1 (default/referencia XGBoost)",
        "colsample_bytree": "1 (default/referencia XGBoost)",
    }


def build_baseline_pipeline(n_estimators=2500, early_stopping_rounds=100):
    baseline_params = get_xgb_baseline_defaults(n_estimators=n_estimators)
    return Pipeline([
        ('model', xgb.XGBRegressor(
            **baseline_params,
            objective='reg:squarederror',
            eval_metric='rmse',
            verbosity=0,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds
        ))
    ])


def resolve_baseline_runtime_config(
    experiment_label=None,
    n_estimators=2500,
    early_stopping_rounds=100,
    final_val_size=0.15
):
    if early_stopping_rounds is None:
        raise ValueError("La línea de base requiere early_stopping_rounds para definir el número efectivo de árboles.")
    if final_val_size is None or not (0 < final_val_size < 1):
        raise ValueError("final_val_size debe estar entre 0 y 1 para la línea de base.")

    return {
        "preset": None,
        "experiment_label": experiment_label if experiment_label is not None else "xgb_baseline",
        "use_early_stopping": True,
        "early_stopping_rounds": early_stopping_rounds,
        "final_val_size": final_val_size,
        "search_space_ranges": get_xgb_baseline_defaults_text(n_estimators=n_estimators),
        "search_space_type": "xgb_baseline_defaults_explicit",
        "n_estimators": int(n_estimators),
    }


def build_baseline_run_config_df(
    dict_name,
    runtime_cfg,
    output_dir_run,
    train_size,
    cv_splits,
    make_plots,
    make_tree_evolution_plot=False,
    tree_evolution_step=5
):
    n_trials_value = np.nan
    rows = [{
        "dict_name": dict_name,
        "preset": runtime_cfg["preset"],
        "experiment_label": runtime_cfg["experiment_label"],
        "search_space_type": runtime_cfg["search_space_type"],
        "use_early_stopping": runtime_cfg["use_early_stopping"],
        "early_stopping_rounds": runtime_cfg["early_stopping_rounds"],
        "final_val_size": runtime_cfg["final_val_size"],
        "n_trials": n_trials_value,
        "train_size": train_size,
        "cv_splits": cv_splits,
        "make_plots": make_plots,
        "make_tree_evolution_plot": make_tree_evolution_plot,
        "tree_evolution_step": tree_evolution_step,
        "output_dir_run": output_dir_run,
    }]

    for param_name, param_value in runtime_cfg["search_space_ranges"].items():
        rows.append({
            "dict_name": dict_name,
            "preset": runtime_cfg["preset"],
            "experiment_label": runtime_cfg["experiment_label"],
            "search_space_type": runtime_cfg["search_space_type"],
            "use_early_stopping": runtime_cfg["use_early_stopping"],
            "early_stopping_rounds": runtime_cfg["early_stopping_rounds"],
            "final_val_size": runtime_cfg["final_val_size"],
            "n_trials": n_trials_value,
            "train_size": train_size,
            "cv_splits": cv_splits,
            "make_plots": make_plots,
            "output_dir_run": output_dir_run,
            "search_param": param_name,
            "search_range": param_value,
        })

    run_config_df = pd.DataFrame(rows)
    run_config_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_run_config.csv"), index=False)
    return run_config_df


def export_prediction_csvs(set_dir, dataframe_name, y_train, y_pred_train, y_test, y_pred_test):
    train_pred_df = pd.DataFrame({
        "Fecha": y_train.index,
        "y_observado": y_train.values,
        "y_predicho": np.asarray(y_pred_train),
    })
    train_pred_df["residuo"] = train_pred_df["y_observado"] - train_pred_df["y_predicho"]

    test_pred_df = pd.DataFrame({
        "Fecha": y_test.index,
        "y_observado": y_test.values,
        "y_predicho": np.asarray(y_pred_test),
    })
    test_pred_df["residuo"] = test_pred_df["y_observado"] - test_pred_df["y_predicho"]

    full_temporal_df = pd.concat([
        train_pred_df.assign(particion="train"),
        test_pred_df.assign(particion="test")
    ], ignore_index=True)

    train_pred_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_predicciones_train.csv"), index=False)
    test_pred_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_predicciones_test.csv"), index=False)
    full_temporal_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_serie_temporal_completa.csv"), index=False)

    return train_pred_df, test_pred_df, full_temporal_df


def export_param_importances_csv(set_dir, dataframe_name, param_importances):
    param_importances_df = pd.DataFrame([
        {"hiperparametro": k, "importancia": v}
        for k, v in param_importances.items()
    ])
    param_importances_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_hyperparam_importances.csv"), index=False)
    return param_importances_df


def export_shap_csvs(set_dir, dataframe_name, shap_values, X_train):
    shap_values_df = pd.DataFrame(shap_values, columns=X_train.columns, index=X_train.index)
    shap_values_df.index.name = "Fecha"
    shap_values_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_shap_values.csv"))

    shap_mean_abs_df = pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap": np.abs(np.asarray(shap_values)).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_mean_abs_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_shap_mean_abs.csv"), index=False)

    return shap_values_df, shap_mean_abs_df


def export_feature_importance_csvs(set_dir, dataframe_name, model, X_train):
    model_feature_importances_df = pd.DataFrame({
        "feature": X_train.columns,
        "feature_importance": getattr(model, "feature_importances_", np.zeros(X_train.shape[1]))
    }).sort_values("feature_importance", ascending=False).reset_index(drop=True)
    model_feature_importances_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_model_feature_importances.csv"), index=False)

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    cover = booster.get_score(importance_type="cover")

    booster_importances_df = pd.DataFrame({"feature": X_train.columns})
    booster_importances_df["gain"] = booster_importances_df["feature"].map(gain).fillna(0.0)
    booster_importances_df["weight"] = booster_importances_df["feature"].map(weight).fillna(0.0)
    booster_importances_df["cover"] = booster_importances_df["feature"].map(cover).fillna(0.0)
    booster_importances_df = booster_importances_df.sort_values(["gain", "weight", "cover"], ascending=False).reset_index(drop=True)
    booster_importances_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_booster_feature_importances.csv"), index=False)

    return model_feature_importances_df, booster_importances_df


def build_artifacts_inventory(output_dir):
    rows = []
    for root, _, files in os.walk(output_dir):
        for filename in files:
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, output_dir)
            ext = os.path.splitext(filename)[1].lower()
            rows.append({
                "relative_path": rel_path,
                "filename": filename,
                "extension": ext,
                "size_bytes": os.path.getsize(abs_path),
            })
    artifacts_df = pd.DataFrame(rows).sort_values(["extension", "relative_path"]).reset_index(drop=True)
    artifacts_df.to_csv(os.path.join(output_dir, "artifacts_inventory.csv"), index=False)
    return artifacts_df

def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_tree_evolution_dataframe(model, X_train_internal, y_train_internal, X_val_internal, y_val_internal, step=5):
    if model is None:
        return pd.DataFrame()

    if step is None:
        step = 5
    step = int(step)
    if step <= 0:
        raise ValueError("tree_evolution_step debe ser un entero positivo.")

    booster = model.get_booster()
    n_boosted_rounds = int(booster.num_boosted_rounds())
    if n_boosted_rounds <= 0:
        return pd.DataFrame()

    tree_counts = list(range(step, n_boosted_rounds + 1, step))
    if not tree_counts:
        tree_counts = [n_boosted_rounds]
    if n_boosted_rounds not in tree_counts:
        tree_counts.append(n_boosted_rounds)

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        try:
            best_tree = int(best_iteration) + 1
            if 1 <= best_tree <= n_boosted_rounds and best_tree not in tree_counts:
                tree_counts.append(best_tree)
        except Exception:
            pass

    tree_counts = sorted(set(tree_counts))

    rows = []
    for n_trees in tree_counts:
        y_pred_train_k = model.predict(X_train_internal, iteration_range=(0, n_trees))
        y_pred_val_k = model.predict(X_val_internal, iteration_range=(0, n_trees))

        nse_train = nash_sutcliffe_efficiency(y_train_internal, y_pred_train_k)
        nse_val = nash_sutcliffe_efficiency(y_val_internal, y_pred_val_k)
        rmse_train = _rmse(y_train_internal, y_pred_train_k)
        rmse_val = _rmse(y_val_internal, y_pred_val_k)

        rows.append({
            "n_trees": n_trees,
            "nse_train_internal": nse_train,
            "nse_val_internal": nse_val,
            "gap_nse_train_val": nse_train - nse_val,
            "rmse_train_internal": rmse_train,
            "rmse_val_internal": rmse_val,
            "gap_rmse_val_train": rmse_val - rmse_train,
            "best_iteration": best_iteration,
            "is_best_iteration_plus_one": (
                best_iteration is not None and n_trees == int(best_iteration) + 1
            )
        })

    return pd.DataFrame(rows)


def _safe_float_from_row(row, col):
    try:
        return float(row[col])
    except Exception:
        return np.nan


def _tree_evolution_row_at(df, n_trees):
    if df is None or df.empty or n_trees is None:
        return None
    subset = df[df["n_trees"] == int(n_trees)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _first_tree_at_nse_threshold(df, threshold):
    if df is None or df.empty or pd.isna(threshold):
        return None
    subset = df[df["nse_val_internal"] >= threshold]
    if subset.empty:
        return None
    return int(subset.iloc[0]["n_trees"])


def build_tree_evolution_summary_dataframe(tree_evolution_df, best_iteration=None):
    if tree_evolution_df is None or tree_evolution_df.empty:
        return pd.DataFrame()

    required_cols = {
        "n_trees",
        "nse_train_internal",
        "nse_val_internal",
        "gap_nse_train_val",
        "rmse_train_internal",
        "rmse_val_internal",
        "gap_rmse_val_train",
    }
    if not required_cols.issubset(set(tree_evolution_df.columns)):
        return pd.DataFrame()

    df = tree_evolution_df.sort_values("n_trees").reset_index(drop=True).copy()

    best_tree = None
    if best_iteration is not None:
        try:
            best_tree = int(best_iteration) + 1
        except Exception:
            best_tree = None

    idx_max_nse_val = df["nse_val_internal"].astype(float).idxmax()
    row_max_nse_val = df.loc[idx_max_nse_val]
    max_nse_val = float(row_max_nse_val["nse_val_internal"])
    tree_max_nse_val = int(row_max_nse_val["n_trees"])

    idx_min_rmse_val = df["rmse_val_internal"].astype(float).idxmin()
    row_min_rmse_val = df.loc[idx_min_rmse_val]
    min_rmse_val = float(row_min_rmse_val["rmse_val_internal"])
    tree_min_rmse_val = int(row_min_rmse_val["n_trees"])

    threshold_99 = 0.99 * max_nse_val if max_nse_val > 0 else np.nan
    threshold_95 = 0.95 * max_nse_val if max_nse_val > 0 else np.nan

    tree_99 = _first_tree_at_nse_threshold(df, threshold_99)
    tree_95 = _first_tree_at_nse_threshold(df, threshold_95)

    row_99 = _tree_evolution_row_at(df, tree_99)
    row_95 = _tree_evolution_row_at(df, tree_95)
    row_best = _tree_evolution_row_at(df, best_tree)

    def value_at(row, col):
        if row is None:
            return np.nan
        return _safe_float_from_row(row, col)

    def tree_ratio(tree):
        if tree is None or best_tree is None or best_tree == 0:
            return np.nan
        return float(tree) / float(best_tree)

    ratio_99 = tree_ratio(tree_99)
    ratio_95 = tree_ratio(tree_95)

    nse_val_99 = value_at(row_99, "nse_val_internal")
    nse_train_99 = value_at(row_99, "nse_train_internal")
    nse_val_best = value_at(row_best, "nse_val_internal")
    nse_train_best = value_at(row_best, "nse_train_internal")

    rows = [
        {"Indicador": "Best iteration + 1", "Valor": best_tree},
        {"Indicador": "Máximo NSE validación interna", "Valor": max_nse_val},
        {"Indicador": "Árbol de máximo NSE validación interna", "Valor": tree_max_nse_val},
        {"Indicador": "Mínimo RMSE validación interna", "Valor": min_rmse_val},
        {"Indicador": "Árbol de mínimo RMSE validación interna", "Valor": tree_min_rmse_val},
        {"Indicador": "Umbral 99% del máximo NSE validación", "Valor": threshold_99},
        {"Indicador": "Primer árbol que alcanza 99% del máximo NSE validación", "Valor": tree_99},
        {"Indicador": "NSE validación en 99%", "Valor": nse_val_99},
        {"Indicador": "NSE train en 99%", "Valor": nse_train_99},
        {"Indicador": "Gap NSE train-validación en 99%", "Valor": value_at(row_99, "gap_nse_train_val")},
        {"Indicador": "Proporción árboles 99% / best_iteration+1", "Valor": ratio_99},
        {"Indicador": "Reducción de árboles usando 99%", "Valor": 1 - ratio_99 if pd.notna(ratio_99) else np.nan},
        {"Indicador": "Umbral 95% del máximo NSE validación", "Valor": threshold_95},
        {"Indicador": "Primer árbol que alcanza 95% del máximo NSE validación", "Valor": tree_95},
        {"Indicador": "NSE validación en 95%", "Valor": value_at(row_95, "nse_val_internal")},
        {"Indicador": "NSE train en 95%", "Valor": value_at(row_95, "nse_train_internal")},
        {"Indicador": "Gap NSE train-validación en 95%", "Valor": value_at(row_95, "gap_nse_train_val")},
        {"Indicador": "Proporción árboles 95% / best_iteration+1", "Valor": ratio_95},
        {"Indicador": "Reducción de árboles usando 95%", "Valor": 1 - ratio_95 if pd.notna(ratio_95) else np.nan},
        {"Indicador": "NSE validación en best_iteration+1", "Valor": nse_val_best},
        {"Indicador": "NSE train en best_iteration+1", "Valor": nse_train_best},
        {"Indicador": "Gap NSE train-validación en best_iteration+1", "Valor": value_at(row_best, "gap_nse_train_val")},
        {"Indicador": "RMSE validación en best_iteration+1", "Valor": value_at(row_best, "rmse_val_internal")},
        {"Indicador": "RMSE train en best_iteration+1", "Valor": value_at(row_best, "rmse_train_internal")},
        {"Indicador": "Gap RMSE validación-train en best_iteration+1", "Valor": value_at(row_best, "gap_rmse_val_train")},
        {"Indicador": "Mejora NSE validación del 99% a best_iteration+1", "Valor": nse_val_best - nse_val_99 if pd.notna(nse_val_best) and pd.notna(nse_val_99) else np.nan},
        {"Indicador": "Mejora NSE train del 99% a best_iteration+1", "Valor": nse_train_best - nse_train_99 if pd.notna(nse_train_best) and pd.notna(nse_train_99) else np.nan},
    ]

    return pd.DataFrame(rows)


def run_xgb_baseline_temporal(
    dataframe,
    dataframe_name,
    train_size=0.8,
    cv_splits=4,
    make_plots=True,
    output_dir="Resultados",
    experiment_label=None,
    n_estimators=2500,
    early_stopping_rounds=100,
    final_val_size=0.15,
    make_tree_evolution_plot=False,
    tree_evolution_step=5
):
    start_time = time.perf_counter()

    runtime_cfg = resolve_baseline_runtime_config(
        experiment_label=experiment_label,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        final_val_size=final_val_size
    )

    config_label = runtime_cfg["experiment_label"]
    use_early_stopping = runtime_cfg["use_early_stopping"]
    search_space_type = runtime_cfg["search_space_type"]
    baseline_params = get_xgb_baseline_defaults(n_estimators=n_estimators)
    baseline_ranges_text = runtime_cfg["search_space_ranges"]

    target_column = dataframe.columns[0]
    original_feature_names = dataframe.columns[1:].tolist()
    clean_target_column = clean_feature_names([target_column])[0]

    df = dataframe.copy().sort_index().dropna()

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    X.columns = clean_feature_names(X.columns)

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, train_size=train_size)

    print(f"\n-> Dataset: {dataframe_name}")
    print(f"   Target: {target_column}")
    print("   Modelo: línea de base XGBoost sin Optuna")
    print(f"   Experiment label: {config_label}")
    print(f"   Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    print(f"   Fecha train: {X_train.index.min()} a {X_train.index.max()}")
    print(f"   Fecha test : {X_test.index.min()} a {X_test.index.max()}")
    print("")

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_scores = []
    baseline_fold_rows = []

    for fold_number, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        pipeline = build_baseline_pipeline(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds
        )

        pipeline.fit(
            X_fold_train,
            y_fold_train,
            model__eval_set=[(X_fold_val, y_fold_val)],
            model__verbose=False
        )

        y_fold_pred = pipeline.predict(X_fold_val)
        fold_nse = nash_sutcliffe_efficiency(y_fold_val, y_fold_pred)
        fold_scores.append(fold_nse)

        fold_model = pipeline.named_steps['model']
        baseline_fold_rows.append({
            "fold_number": fold_number,
            "fold_nse": fold_nse,
            "best_iteration_fold": getattr(fold_model, "best_iteration", None),
            "n_train_fold": len(X_fold_train),
            "n_val_fold": len(X_fold_val),
        })

    cv_nse_mean = round_sig(float(np.nanmean(fold_scores)), 2) if fold_scores else np.nan
    cv_nse_std = round_sig(float(np.nanstd(fold_scores)), 2) if fold_scores else np.nan

    X_subtrain, X_val, y_subtrain, y_val = temporal_subtrain_validation_split(
        X_train, y_train, val_size=final_val_size
    )

    final_pipeline = build_baseline_pipeline(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds
    )

    final_pipeline.fit(
        X_subtrain,
        y_subtrain,
        model__eval_set=[(X_val, y_val)],
        model__verbose=False
    )

    y_pred_train = final_pipeline.predict(X_train)
    y_pred_test = final_pipeline.predict(X_test)

    metrics_train, metrics_test = evaluate_metrics(y_train, y_pred_train, y_test, y_pred_test)

    set_dir = os.path.join(output_dir, dataframe_name)
    os.makedirs(set_dir, exist_ok=True)

    tree_evolution_df = pd.DataFrame()
    tree_evolution_summary_df = pd.DataFrame()
    tree_evolution_plot_path = None
    tree_evolution_rmse_plot_path = None
    tree_evolution_summary_path = None

    if make_tree_evolution_plot:
        tree_evolution_df = build_tree_evolution_dataframe(
            model=final_pipeline.named_steps['model'],
            X_train_internal=X_subtrain,
            y_train_internal=y_subtrain,
            X_val_internal=X_val,
            y_val_internal=y_val,
            step=tree_evolution_step
        )
        tree_evolution_csv_path = os.path.join(set_dir, f"{dataframe_name}_tree_evolution.csv")
        tree_evolution_df.to_csv(tree_evolution_csv_path, index=False)

        if make_plots and not tree_evolution_df.empty:
            tree_evolution_plot_path = os.path.join(set_dir, f"{dataframe_name}_tree_evolution_nse.png")
            plot_tree_evolution_curve(
                tree_evolution_df=tree_evolution_df,
                title=f"Evolución NSE según número de árboles - {dataframe_name}",
                save_path=tree_evolution_plot_path,
                best_iteration=getattr(final_pipeline.named_steps['model'], "best_iteration", None)
            )

            tree_evolution_rmse_plot_path = os.path.join(set_dir, f"{dataframe_name}_tree_evolution_rmse.png")
            plot_tree_evolution_rmse_curve(
                tree_evolution_df=tree_evolution_df,
                title=f"Evolución RMSE según número de árboles - {dataframe_name}",
                save_path=tree_evolution_rmse_plot_path,
                best_iteration=getattr(final_pipeline.named_steps['model'], "best_iteration", None)
            )

        if int(tree_evolution_step) == 1 and not tree_evolution_df.empty:
            tree_evolution_summary_df = build_tree_evolution_summary_dataframe(
                tree_evolution_df=tree_evolution_df,
                best_iteration=getattr(final_pipeline.named_steps['model'], "best_iteration", None)
            )
            if not tree_evolution_summary_df.empty:
                tree_evolution_summary_path = os.path.join(set_dir, f"{dataframe_name}_tree_evolution_summary.csv")
                tree_evolution_summary_df.to_csv(tree_evolution_summary_path, index=False)

    train_pred_df, test_pred_df, full_temporal_df = export_prediction_csvs(
        set_dir=set_dir,
        dataframe_name=dataframe_name,
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_test=y_test,
        y_pred_test=y_pred_test
    )

    baseline_fold_df = pd.DataFrame(baseline_fold_rows)
    baseline_fold_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_baseline_cv_fold_scores.csv"), index=False)

    if make_plots:
        plot_train_vs_actual(
            y_train, y_pred_train, clean_target_column,
            save_path=os.path.join(set_dir, f"{dataframe_name}_train_vs_actual.png")
        )

        plot_test_vs_actual(
            y_test, y_pred_test, clean_target_column,
            save_path=os.path.join(set_dir, f"{dataframe_name}_test_vs_actual.png")
        )

        plot_full_temporal_results(
            y_train=y_train,
            y_test=y_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            target_name=clean_target_column,
            save_path=os.path.join(set_dir, f"{dataframe_name}_full_temporal_results.png")
        )

        plot_real_vs_pred(
            y_test, y_pred_test,
            save_path=os.path.join(set_dir, f"{dataframe_name}_real_vs_pred.png")
        )

    model = final_pipeline.named_steps['model']
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    base_values = explainer.expected_value
    if isinstance(base_values, list):
        base_values = base_values[0]

    if make_plots:
        shap_summary_plots(
            shap_values,
            X_train,
            feature_names=X_train.columns.tolist(),
            out_prefix=os.path.join(set_dir, dataframe_name),
            max_display=20,
            decision_top_n=20,
            base_values=base_values
        )

    shap_values_df, shap_mean_abs_df = export_shap_csvs(set_dir, dataframe_name, shap_values, X_train)
    model_feature_importances_df, booster_feature_importances_df = export_feature_importance_csvs(set_dir, dataframe_name, model, X_train)

    elapsed_seconds = round_sig(time.perf_counter() - start_time, 3)
    elapsed_str = format_elapsed_time(elapsed_seconds)

    best_iteration = getattr(model, "best_iteration", None)

    print("-> Valores default/referencia usados por la línea de base:")
    for param_name, param_range in baseline_ranges_text.items():
        print(f"   {param_name:<18}: {param_range}")
    print("")

    print("-> Hiperparámetros usados:")
    print(round_params_dict(baseline_params, sig=3))
    print("")
    print(f"-> CV NSE promedio: {cv_nse_mean}")
    print(f"-> CV NSE desvío  : {cv_nse_std}")
    print("")
    print(f"-> Experiment label      : {config_label}")
    print(f"-> Search space type     : {search_space_type}")
    print(f"-> Use early stopping    : {use_early_stopping}")
    print(f"-> Early stopping rounds : {early_stopping_rounds}")
    print(f"-> Final val size        : {final_val_size}")
    print(f"-> Make tree evolution   : {make_tree_evolution_plot}")
    print(f"-> Tree evolution step   : {tree_evolution_step}")
    print(f"-> Best iteration        : {best_iteration}")
    print("")
    print("-> Importancia de hiperparámetros: No aplica en línea de base sin Optuna")
    print("")
    print("-> Métricas train:")
    print(metrics_train)
    print("")
    print("-> Métricas test:")
    print(metrics_test)
    print("")
    print(f"-> Tiempo de ejecución: {elapsed_str}")
    print("")

    return {
        'dataframe_name': dataframe_name,
        'target': target_column,
        'feature_names': original_feature_names,
        'config_mode': config_label,
        'search_space_type': search_space_type,
        'search_ranges': baseline_ranges_text,
        'use_early_stopping': use_early_stopping,
        'best_params': baseline_params,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'cv_nse_mean': cv_nse_mean,
        'cv_nse_std': cv_nse_std,
        'train_start': X_train.index.min(),
        'train_end': X_train.index.max(),
        'test_start': X_test.index.min(),
        'test_end': X_test.index.max(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'execution_seconds': elapsed_seconds,
        'execution_time': elapsed_str,
        'best_iteration': best_iteration,
        'early_stopping_rounds': early_stopping_rounds,
        'final_val_size': final_val_size,
        'param_importances': {},
        'top_hyperparams': [],
        'train_predictions_df': train_pred_df,
        'test_predictions_df': test_pred_df,
        'full_temporal_df': full_temporal_df,
        'trials_df': pd.DataFrame(),
        'trial_fold_df': baseline_fold_df,
        'param_importances_df': pd.DataFrame(),
        'shap_values_df': shap_values_df,
        'shap_mean_abs_df': shap_mean_abs_df,
        'model_feature_importances_df': model_feature_importances_df,
        'booster_feature_importances_df': booster_feature_importances_df,
        'tree_evolution_df': tree_evolution_df,
        'tree_evolution_summary_df': tree_evolution_summary_df,
        'tree_evolution_plot_path': tree_evolution_plot_path,
        'tree_evolution_rmse_plot_path': tree_evolution_rmse_plot_path,
        'tree_evolution_summary_path': tree_evolution_summary_path,
        'make_tree_evolution_plot': make_tree_evolution_plot,
        'tree_evolution_step': tree_evolution_step,
    }


def _integrated_safe_get(mapping, key, default=np.nan):
    if mapping is None:
        return default
    try:
        return mapping.get(key, default)
    except AttributeError:
        return default


def _integrated_metric(metrics, metric_name):
    return _integrated_safe_get(metrics, metric_name, np.nan)


def _integrated_best_iteration_plus_1(best_iteration):
    if best_iteration is None:
        return np.nan
    try:
        if pd.isna(best_iteration):
            return np.nan
    except Exception:
        pass
    try:
        return int(best_iteration) + 1
    except Exception:
        return np.nan


def _integrated_summary_value(summary_df, indicador):
    if summary_df is None or not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return np.nan
    if "Indicador" not in summary_df.columns or "Valor" not in summary_df.columns:
        return np.nan
    subset = summary_df.loc[summary_df["Indicador"] == indicador, "Valor"]
    if subset.empty:
        return np.nan
    return subset.iloc[0]


def _integrated_tree_row_value(tree_df, n_trees, column):
    if tree_df is None or not isinstance(tree_df, pd.DataFrame) or tree_df.empty:
        return np.nan
    if "n_trees" not in tree_df.columns or column not in tree_df.columns:
        return np.nan
    if n_trees is None:
        return np.nan
    try:
        if pd.isna(n_trees):
            return np.nan
        n_trees_int = int(n_trees)
    except Exception:
        return np.nan
    subset = tree_df.loc[tree_df["n_trees"] == n_trees_int]
    if subset.empty:
        return np.nan
    return subset.iloc[0][column]


def _integrated_subtract(a, b):
    try:
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return float(a) - float(b)
    except Exception:
        return np.nan


def _integrated_divide(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def _integrated_hyperimportance_families(param_importances):
    param_importances = param_importances or {}

    if not param_importances:
        return np.nan, np.nan, np.nan, np.nan

    complexity_params = {"n_estimators", "max_depth", "learning_rate"}
    regularization_params = {"min_child_weight", "gamma", "reg_alpha", "reg_lambda"}
    sampling_params = {"subsample", "colsample_bytree"}

    complexity = 0.0
    regularization = 0.0
    sampling = 0.0
    other = 0.0

    for name, value in param_importances.items():
        try:
            value_float = float(value)
        except Exception:
            value_float = 0.0

        if name in complexity_params:
            complexity += value_float
        elif name in regularization_params:
            regularization += value_float
        elif name in sampling_params:
            sampling += value_float
        else:
            other += value_float

    return complexity, regularization, sampling, other


def _integrated_feature_importance_stats(shap_mean_abs_df):
    stats = {
        "featureimp_total": np.nan,
        "featureimp_top3_sum": np.nan,
        "featureimp_top5_sum": np.nan,
        "featureimp_top3_share": np.nan,
        "featureimp_top5_share": np.nan,
    }
    top_entries = []

    if shap_mean_abs_df is None or not isinstance(shap_mean_abs_df, pd.DataFrame) or shap_mean_abs_df.empty:
        return top_entries, stats
    if "feature" not in shap_mean_abs_df.columns or "mean_abs_shap" not in shap_mean_abs_df.columns:
        return top_entries, stats

    df = shap_mean_abs_df[["feature", "mean_abs_shap"]].copy()
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce")
    df = df.dropna(subset=["mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if df.empty:
        return top_entries, stats

    total = float(df["mean_abs_shap"].sum())
    top3_sum = float(df.head(3)["mean_abs_shap"].sum())
    top5_sum = float(df.head(5)["mean_abs_shap"].sum())

    stats["featureimp_total"] = total
    stats["featureimp_top3_sum"] = top3_sum
    stats["featureimp_top5_sum"] = top5_sum
    stats["featureimp_top3_share"] = top3_sum / total if total != 0 else np.nan
    stats["featureimp_top5_share"] = top5_sum / total if total != 0 else np.nan

    for _, row in df.head(5).iterrows():
        top_entries.append((row["feature"], row["mean_abs_shap"]))

    return top_entries, stats


def build_integrated_diagnostics_dataframe(
    results,
    dict_name,
    runtime_cfg,
    n_trials,
    train_size,
    cv_splits,
    make_plots,
    make_tree_evolution_plot,
    tree_evolution_step
):
    rows = []

    param_names = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "colsample_bytree",
    ]

    tree_summary_map = {
        "tree_max_nse_val": "Máximo NSE validación interna",
        "tree_n_at_max_nse_val": "Árbol de máximo NSE validación interna",
        "tree_n_99": "Primer árbol que alcanza 99% del máximo NSE validación",
        "tree_n_99_prop_vs_best": "Proporción árboles 99% / best_iteration+1",
        "tree_n_99_reduction_pct": "Reducción de árboles usando 99%",
        "tree_nse_val_at_99": "NSE validación en 99%",
        "tree_nse_train_at_99": "NSE train en 99%",
        "tree_gap_nse_at_99": "Gap NSE train-validación en 99%",
        "tree_n_95": "Primer árbol que alcanza 95% del máximo NSE validación",
        "tree_n_95_prop_vs_best": "Proporción árboles 95% / best_iteration+1",
        "tree_n_95_reduction_pct": "Reducción de árboles usando 95%",
        "tree_nse_val_at_95": "NSE validación en 95%",
        "tree_nse_train_at_95": "NSE train en 95%",
        "tree_gap_nse_at_95": "Gap NSE train-validación en 95%",
        "tree_nse_val_at_best": "NSE validación en best_iteration+1",
        "tree_nse_train_at_best": "NSE train en best_iteration+1",
        "tree_gap_nse_at_best": "Gap NSE train-validación en best_iteration+1",
        "tree_val_gain_99_to_best": "Mejora NSE validación del 99% a best_iteration+1",
        "tree_train_gain_99_to_best": "Mejora NSE train del 99% a best_iteration+1",
        "tree_min_rmse_val": "Mínimo RMSE validación interna",
        "tree_n_at_min_rmse_val": "Árbol de mínimo RMSE validación interna",
        "tree_rmse_val_at_best": "RMSE validación en best_iteration+1",
        "tree_rmse_train_at_best": "RMSE train en best_iteration+1",
        "tree_gap_rmse_at_best": "Gap RMSE validación-train en best_iteration+1",
    }

    for result in results:
        metrics_train = result.get("metrics_train", {})
        metrics_test = result.get("metrics_test", {})
        best_params = result.get("best_params", {}) or {}
        search_ranges = result.get("search_ranges", {}) or {}
        param_importances = result.get("param_importances", {}) or {}
        shap_mean_abs_df = result.get("shap_mean_abs_df", pd.DataFrame())
        tree_evolution_df = result.get("tree_evolution_df", pd.DataFrame())
        tree_evolution_summary_df = result.get("tree_evolution_summary_df", pd.DataFrame())

        cv_nse_mean = result.get("cv_nse_mean", np.nan)
        cv_nse_std = result.get("cv_nse_std", np.nan)
        train_nse = _integrated_metric(metrics_train, "NSE")
        test_nse = _integrated_metric(metrics_test, "NSE")
        best_iteration = result.get("best_iteration", np.nan)
        best_iteration_plus_1 = _integrated_best_iteration_plus_1(best_iteration)

        row = {
            "dataset_name": result.get("dataframe_name", np.nan),
            "target": result.get("target", np.nan),
            "config_mode": result.get("config_mode", runtime_cfg.get("experiment_label", np.nan)),
            "search_space_type": result.get("search_space_type", runtime_cfg.get("search_space_type", np.nan)),
            "preset": runtime_cfg.get("preset", np.nan),
            "use_early_stopping": result.get("use_early_stopping", runtime_cfg.get("use_early_stopping", np.nan)),
            "early_stopping_rounds": result.get("early_stopping_rounds", runtime_cfg.get("early_stopping_rounds", np.nan)),
            "final_val_size": result.get("final_val_size", runtime_cfg.get("final_val_size", np.nan)),
            "n_trials": n_trials,
            "cv_splits": cv_splits,
            "train_size": train_size,
            "make_plots": make_plots,
            "make_tree_evolution_plot": make_tree_evolution_plot,
            "tree_evolution_step": tree_evolution_step,
            "train_start": result.get("train_start", np.nan),
            "train_end": result.get("train_end", np.nan),
            "test_start": result.get("test_start", np.nan),
            "test_end": result.get("test_end", np.nan),
            "n_train": result.get("n_train", np.nan),
            "n_test": result.get("n_test", np.nan),
            "n_features": result.get("n_features", np.nan),
            "cv_nse_mean": cv_nse_mean,
            "cv_nse_std": cv_nse_std,
            "cv_nse_performance": classify_nse_performance(cv_nse_mean),
            "train_rmse": _integrated_metric(metrics_train, "RMSE"),
            "train_mae": _integrated_metric(metrics_train, "MAE"),
            "train_nse": train_nse,
            "train_kge": _integrated_metric(metrics_train, "KGE"),
            "train_pbias": _integrated_metric(metrics_train, "PBIAS"),
            "train_r2": _integrated_metric(metrics_train, "R2"),
            "train_rsr": _integrated_metric(metrics_train, "RSR"),
            "test_rmse": _integrated_metric(metrics_test, "RMSE"),
            "test_mae": _integrated_metric(metrics_test, "MAE"),
            "test_nse": test_nse,
            "test_kge": _integrated_metric(metrics_test, "KGE"),
            "test_pbias": _integrated_metric(metrics_test, "PBIAS"),
            "test_r2": _integrated_metric(metrics_test, "R2"),
            "test_rsr": _integrated_metric(metrics_test, "RSR"),
            "test_nse_performance": classify_nse_performance(test_nse),
            "gap_train_cv_nse": _integrated_subtract(train_nse, cv_nse_mean),
            "gap_train_test_nse": _integrated_subtract(train_nse, test_nse),
            "gap_cv_test_nse": _integrated_subtract(cv_nse_mean, test_nse),
            "execution_seconds": result.get("execution_seconds", np.nan),
            "execution_time": result.get("execution_time", np.nan),
        }

        for param_name in param_names:
            row[f"param_{param_name}"] = best_params.get(param_name, np.nan)

        row["best_iteration"] = best_iteration
        row["best_iteration_plus_1"] = best_iteration_plus_1

        for param_name in param_names:
            row[f"range_{param_name}"] = search_ranges.get(param_name, np.nan)

        is_xgb_baseline = str(row.get("search_space_type", "")).startswith("xgb_baseline")
        sorted_importances = sorted(param_importances.items(), key=lambda item: item[1], reverse=True)
        for i in range(1, 6):
            if (not is_xgb_baseline) and len(sorted_importances) >= i:
                row[f"hyperimp_top{i}_name"] = sorted_importances[i - 1][0]
                row[f"hyperimp_top{i}_value"] = sorted_importances[i - 1][1]
            else:
                row[f"hyperimp_top{i}_name"] = np.nan
                row[f"hyperimp_top{i}_value"] = np.nan

        if is_xgb_baseline:
            row["hyperimp_complexity"] = np.nan
            row["hyperimp_regularization"] = np.nan
            row["hyperimp_sampling"] = np.nan
            row["hyperimp_other"] = np.nan
        else:
            hyperimp_complexity, hyperimp_regularization, hyperimp_sampling, hyperimp_other = _integrated_hyperimportance_families(param_importances)
            row["hyperimp_complexity"] = hyperimp_complexity
            row["hyperimp_regularization"] = hyperimp_regularization
            row["hyperimp_sampling"] = hyperimp_sampling
            row["hyperimp_other"] = hyperimp_other

        feature_top_entries, feature_stats = _integrated_feature_importance_stats(shap_mean_abs_df)
        for i in range(1, 6):
            if len(feature_top_entries) >= i:
                row[f"featureimp_top{i}_name"] = feature_top_entries[i - 1][0]
                row[f"featureimp_top{i}_value"] = feature_top_entries[i - 1][1]
            else:
                row[f"featureimp_top{i}_name"] = np.nan
                row[f"featureimp_top{i}_value"] = np.nan
        row.update(feature_stats)

        row["tree_evolution_available"] = (
            tree_evolution_df is not None and
            isinstance(tree_evolution_df, pd.DataFrame) and
            not tree_evolution_df.empty
        )
        row["tree_evolution_summary_available"] = (
            tree_evolution_summary_df is not None and
            isinstance(tree_evolution_summary_df, pd.DataFrame) and
            not tree_evolution_summary_df.empty
        )
        row["tree_evolution_step"] = result.get("tree_evolution_step", tree_evolution_step)

        for out_col, indicador in tree_summary_map.items():
            row[out_col] = _integrated_summary_value(tree_evolution_summary_df, indicador)

        row["tree_nse_train_at_max_nse_val"] = _integrated_tree_row_value(
            tree_evolution_df, row.get("tree_n_at_max_nse_val"), "nse_train_internal"
        )
        row["tree_gap_nse_at_max_nse_val"] = _integrated_tree_row_value(
            tree_evolution_df, row.get("tree_n_at_max_nse_val"), "gap_nse_train_val"
        )
        row["tree_rmse_train_at_min_rmse_val"] = _integrated_tree_row_value(
            tree_evolution_df, row.get("tree_n_at_min_rmse_val"), "rmse_train_internal"
        )
        row["tree_gap_rmse_at_min_rmse_val"] = _integrated_tree_row_value(
            tree_evolution_df, row.get("tree_n_at_min_rmse_val"), "gap_rmse_val_train"
        )

        row["tree_rmse_val_at_99"] = _integrated_tree_row_value(tree_evolution_df, row.get("tree_n_99"), "rmse_val_internal")
        row["tree_rmse_train_at_99"] = _integrated_tree_row_value(tree_evolution_df, row.get("tree_n_99"), "rmse_train_internal")
        row["tree_gap_rmse_at_99"] = _integrated_tree_row_value(tree_evolution_df, row.get("tree_n_99"), "gap_rmse_val_train")
        row["tree_rmse_val_at_95"] = _integrated_tree_row_value(tree_evolution_df, row.get("tree_n_95"), "rmse_val_internal")
        row["tree_rmse_train_at_95"] = _integrated_tree_row_value(tree_evolution_df, row.get("tree_n_95"), "rmse_train_internal")
        row["tree_gap_rmse_at_95"] = _integrated_tree_row_value(tree_evolution_df, row.get("tree_n_95"), "gap_rmse_val_train")

        row["tree_val_gain_95_to_best"] = _integrated_subtract(
            row.get("tree_nse_val_at_best"), row.get("tree_nse_val_at_95")
        )
        row["tree_train_gain_95_to_best"] = _integrated_subtract(
            row.get("tree_nse_train_at_best"), row.get("tree_nse_train_at_95")
        )

        row["tree_best_vs_99_difference"] = _integrated_subtract(best_iteration_plus_1, row.get("tree_n_99"))
        row["tree_best_vs_99_difference_pct"] = _integrated_divide(
            row.get("tree_best_vs_99_difference"), best_iteration_plus_1
        )
        row["tree_best_vs_95_difference"] = _integrated_subtract(best_iteration_plus_1, row.get("tree_n_95"))
        row["tree_best_vs_95_difference_pct"] = _integrated_divide(
            row.get("tree_best_vs_95_difference"), best_iteration_plus_1
        )

        rows.append(row)

    return pd.DataFrame(rows)

def process_sets_xgb_baseline_temporal(
    dataframes,
    train_size=0.8,
    cv_splits=4,
    make_plots=True,
    output_dir="Resultados",
    experiment_label=None,
    n_estimators=2500,
    early_stopping_rounds=100,
    final_val_size=0.15,
    make_tree_evolution_plot=False,
    tree_evolution_step=5
):
    local_vars = inspect.currentframe().f_back.f_locals.items()
    dict_name = get_variable_name(dataframes, local_vars)

    runtime_cfg = resolve_baseline_runtime_config(
        experiment_label=experiment_label,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        final_val_size=final_val_size
    )

    output_dir_run = build_timestamped_output_dir(output_dir)
    os.makedirs(output_dir_run, exist_ok=True)

    original_stdout = sys.stdout
    tee_stdout = TeeOutput(original_stdout)
    sys.stdout = tee_stdout

    results = []
    train_metrics_df = None
    test_metrics_df = None
    params_df = None
    metadata_df = None
    comparison_df = None
    global_nse_stats_df = None
    n_trials_not_applicable = np.nan

    try:
        print("")
        print("==================================================================================================")
        print("INICIO DEL PROCESAMIENTO - LÍNEA DE BASE XGBOOST")
        print("==================================================================================================")
        print(f"Diccionario procesado : {dict_name}")
        print("Tipo de modelo         : XGBoost baseline sin Optuna")
        print(f"Experiment label      : {runtime_cfg['experiment_label']}")
        print("Cantidad de trials    : No aplica")
        print(f"Train size            : {train_size}")
        print(f"CV splits             : {cv_splits}")
        print(f"Make plots            : {make_plots}")
        print(f"Make tree evolution   : {make_tree_evolution_plot}")
        print(f"Tree evolution step   : {tree_evolution_step}")
        print(f"Use early stopping    : {runtime_cfg['use_early_stopping']}")
        print(f"Early stopping rounds : {runtime_cfg['early_stopping_rounds']}")
        print(f"Final val size        : {runtime_cfg['final_val_size']}")
        print(f"Search space type     : {runtime_cfg['search_space_type']}")
        print(f"n_estimators máximo   : {runtime_cfg['n_estimators']}")
        print(f"Carpeta de salida     : {output_dir_run}")
        print("")
        print("Frase metodológica:")
        print("Se entrenó una línea de base XGBoost utilizando los valores por defecto del algoritmo y fijando n_estimators=2500 como máximo de árboles. El número efectivo de árboles fue determinado mediante early stopping sobre una partición interna de validación temporal.")
        print("")

        for dataframe_name, dataframe in dataframes.items():
            print("")
            print("######################################################################")
            print(f"Procesando {dataframe_name}...")
            print("######################################################################")
            print("")

            result = run_xgb_baseline_temporal(
                dataframe=dataframe,
                dataframe_name=dataframe_name,
                train_size=train_size,
                cv_splits=cv_splits,
                make_plots=make_plots,
                output_dir=output_dir_run,
                experiment_label=runtime_cfg["experiment_label"],
                n_estimators=runtime_cfg["n_estimators"],
                early_stopping_rounds=runtime_cfg["early_stopping_rounds"],
                final_val_size=runtime_cfg["final_val_size"],
                make_tree_evolution_plot=make_tree_evolution_plot,
                tree_evolution_step=tree_evolution_step
            )

            results.append(result)

        train_metrics_data = []
        test_metrics_data = []
        params_data = []
        metadata_data = []
        comparison_rows = []

        metric_names = ['RMSE', 'MAE', 'NSE', 'KGE', 'PBIAS', 'R2', 'RSR']

        for result in results:
            train_metrics_data.append({
                'DataFrame': result['dataframe_name'],
                **result['metrics_train']
            })

            test_metrics_data.append({
                'DataFrame': result['dataframe_name'],
                **result['metrics_test']
            })

            params_data.append({
                'DataFrame': result['dataframe_name'],
                **result['best_params']
            })

            metadata_data.append({
                'DataFrame': result['dataframe_name'],
                'Target': result['target'],
                'config_mode': result['config_mode'],
                'search_space_type': result['search_space_type'],
                'use_early_stopping': result['use_early_stopping'],
                'n_train': result['n_train'],
                'n_test': result['n_test'],
                'n_features': result['n_features'],
                'train_start': result['train_start'],
                'train_end': result['train_end'],
                'test_start': result['test_start'],
                'test_end': result['test_end'],
                'cv_nse_mean': result['cv_nse_mean'],
                'cv_nse_std': result['cv_nse_std'],
                'best_iteration': result['best_iteration'],
                'early_stopping_rounds': result['early_stopping_rounds'],
                'final_val_size': result['final_val_size'],
                'execution_seconds': result['execution_seconds'],
                'execution_time': result['execution_time']
            })

            row = {('DataFrame', ''): result['dataframe_name']}
            row[('NSE', 'CV Mean')] = result['cv_nse_mean']
            row[('NSE', 'CV Std')] = result['cv_nse_std']
            row[('NSE', 'CV Performance')] = classify_nse_performance(result['cv_nse_mean'])
            row[('NSE', 'Train')] = result['metrics_train']['NSE']
            row[('NSE', 'Test')] = result['metrics_test']['NSE']
            row[('NSE', 'Test Performance')] = classify_nse_performance(result['metrics_test']['NSE'])

            for metric in [m for m in metric_names if m != 'NSE']:
                row[(metric, 'Train')] = result['metrics_train'][metric]
                row[(metric, 'Test')] = result['metrics_test'][metric]

            row[('EarlyStop', 'BestIteration')] = result['best_iteration']
            row[('Tiempo', 'Segundos')] = result['execution_seconds']
            row[('Tiempo', 'Texto')] = result['execution_time']

            comparison_rows.append(row)

        train_metrics_df = pd.DataFrame(train_metrics_data)
        test_metrics_df = pd.DataFrame(test_metrics_data)
        params_df = pd.DataFrame(params_data)
        metadata_df = pd.DataFrame(metadata_data)

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.columns = pd.MultiIndex.from_tuples(comparison_df.columns)

        global_nse_stats_df = pd.DataFrame({
            'Estadística': ['Promedio', 'Mediana', 'Std'],
            'CV Mean': [round_sig(pd.to_numeric(metadata_df['cv_nse_mean'], errors='coerce').mean(), 2),
                        round_sig(pd.to_numeric(metadata_df['cv_nse_mean'], errors='coerce').median(), 2),
                        round_sig(pd.to_numeric(metadata_df['cv_nse_mean'], errors='coerce').std(), 2)],
            'CV Std': [round_sig(pd.to_numeric(metadata_df['cv_nse_std'], errors='coerce').mean(), 2),
                       round_sig(pd.to_numeric(metadata_df['cv_nse_std'], errors='coerce').median(), 2),
                       round_sig(pd.to_numeric(metadata_df['cv_nse_std'], errors='coerce').std(), 2)],
            'Train': [round_sig(pd.to_numeric(train_metrics_df['NSE'], errors='coerce').mean(), 2),
                      round_sig(pd.to_numeric(train_metrics_df['NSE'], errors='coerce').median(), 2),
                      round_sig(pd.to_numeric(train_metrics_df['NSE'], errors='coerce').std(), 2)],
            'Test': [round_sig(pd.to_numeric(test_metrics_df['NSE'], errors='coerce').mean(), 2),
                     round_sig(pd.to_numeric(test_metrics_df['NSE'], errors='coerce').median(), 2),
                     round_sig(pd.to_numeric(test_metrics_df['NSE'], errors='coerce').std(), 2)],
        })

        if 'NSE' in test_metrics_df.columns:
            test_metrics_df = test_metrics_df.sort_values('NSE', ascending=False).reset_index(drop=True)

            ordered_names = test_metrics_df['DataFrame'].tolist()

            train_metrics_df = (
                train_metrics_df.set_index('DataFrame')
                .loc[ordered_names]
                .reset_index()
            )

            params_df = (
                params_df.set_index('DataFrame')
                .loc[ordered_names]
                .reset_index()
            )

            metadata_df = (
                metadata_df.set_index('DataFrame')
                .loc[ordered_names]
                .reset_index()
            )

            results_by_name = {r['dataframe_name']: r for r in results}
            results = [results_by_name[name] for name in ordered_names]

        comparison_sort_col = ('NSE', 'Test')
        if comparison_sort_col in comparison_df.columns:
            comparison_df = comparison_df.sort_values(by=comparison_sort_col, ascending=False).reset_index(drop=True)

        train_metrics_df.to_csv(os.path.join(output_dir_run, f'{dict_name}_train_metrics.csv'), index=False)
        test_metrics_df.to_csv(os.path.join(output_dir_run, f'{dict_name}_test_metrics.csv'), index=False)
        params_df.to_csv(os.path.join(output_dir_run, f'{dict_name}_params.csv'), index=False)
        metadata_df.to_csv(os.path.join(output_dir_run, f'{dict_name}_metadata.csv'), index=False)
        comparison_df.to_csv(os.path.join(output_dir_run, f'{dict_name}_comparison_train_test_cv.csv'), index=False)
        global_nse_stats_df.to_csv(os.path.join(output_dir_run, f'{dict_name}_global_nse_stats.csv'), index=False)

        integrated_diagnostics_df = build_integrated_diagnostics_dataframe(
            results=results,
            dict_name=dict_name,
            runtime_cfg=runtime_cfg,
            n_trials=n_trials_not_applicable,
            train_size=train_size,
            cv_splits=cv_splits,
            make_plots=make_plots,
            make_tree_evolution_plot=make_tree_evolution_plot,
            tree_evolution_step=tree_evolution_step
        )
        integrated_diagnostics_df.to_csv(
            os.path.join(output_dir_run, f'{dict_name}_integrated_diagnostics.csv'),
            index=False
        )

        build_baseline_run_config_df(
            dict_name=dict_name,
            runtime_cfg=runtime_cfg,
            output_dir_run=output_dir_run,
            train_size=train_size,
            cv_splits=cv_splits,
            make_plots=make_plots,
            make_tree_evolution_plot=make_tree_evolution_plot,
            tree_evolution_step=tree_evolution_step
        )

        print("--------------------------------------------------------------------------------------------------")
        print(f"\nCarpeta de resultados: {output_dir_run}")

        print("\nResumen de métricas de train:\n")
        print(train_metrics_df)

        print("\nResumen de métricas de test:\n")
        print(test_metrics_df)

        print("\nResumen de hiperparámetros usados por la línea de base:\n")
        print(params_df)

        print("\nResumen de metadatos:\n")
        print(metadata_df)

        print("\nResumen comparativo final (train / test / CV / tiempo):\n")
        print(comparison_df)

        print("\nEstadísticas globales de performance NSE:\n")
        print(global_nse_stats_df)

    finally:
        sys.stdout = original_stdout

        log_text = tee_stdout.get_value()

        save_log_txt(log_text, output_dir_run, filename="log_salida.txt")
        save_log_html(
            log_text,
            output_dir_run,
            title="Log de salida del experimento baseline XGBoost",
            filename="log_salida.html"
        )

        try:
            if (
                train_metrics_df is not None and
                test_metrics_df is not None and
                params_df is not None and
                metadata_df is not None and
                comparison_df is not None and
                len(results) > 0
            ):
                report_path = generate_html_report(
                    run_output_dir=output_dir_run,
                    dict_name=dict_name,
                    train_metrics_df=train_metrics_df,
                    test_metrics_df=test_metrics_df,
                    params_df=params_df,
                    metadata_df=metadata_df,
                    comparison_df=comparison_df,
                    global_nse_stats_df=global_nse_stats_df,
                    results=results,
                    config_mode=runtime_cfg["experiment_label"],
                    n_trials=n_trials_not_applicable,
                    html_filename="reporte_resultados.html"
                )

                print(f"Reporte HTML generado correctamente en: {report_path}")

            else:
                error_msg = (
                    "No se generó reporte_resultados.html porque faltaron objetos finales.\n"
                    f"train_metrics_df is None: {train_metrics_df is None}\n"
                    f"test_metrics_df is None: {test_metrics_df is None}\n"
                    f"params_df is None: {params_df is None}\n"
                    f"metadata_df is None: {metadata_df is None}\n"
                    f"comparison_df is None: {comparison_df is None}\n"
                    f"Cantidad de resultados: {len(results)}\n"
                )
                with open(os.path.join(output_dir_run, "error_reporte_html.txt"), "w", encoding="utf-8") as f:
                    f.write(error_msg)

        except Exception:
            with open(os.path.join(output_dir_run, "error_reporte_html.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

        build_artifacts_inventory(output_dir_run)

    return train_metrics_df, test_metrics_df, params_df, metadata_df, comparison_df

