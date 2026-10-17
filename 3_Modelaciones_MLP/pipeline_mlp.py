# ============================================================
# CORE MLP + PYTORCH + OPTUNA PARA SERIES TEMPORALES TABULARES
# VERSIÓN FINAL CON REPRODCIBILIDAD 
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ast
import copy
import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from my_metrics_mlp import (
    round_sig,
    nash_sutcliffe_efficiency,
    compute_regression_metrics,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

def set_global_seed(seed=42, deterministic_torch=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SPLITS TEMPORALES
# ============================================================

def temporal_train_test_split_df(df, train_size=0.8):
    n = len(df)
    split_idx = int(n * train_size)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def temporal_subtrain_validation_split_arrays(X, y, val_size=0.15):
    n = len(X)
    split_idx = int(n * (1 - val_size))
    X_subtrain = X[:split_idx]
    y_subtrain = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    return X_subtrain, X_val, y_subtrain, y_val


# ============================================================
# ACTIVACIONES
# ============================================================

def get_activation_module(name, negative_slope=0.01):
    name = name.lower()

    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope)
    elif name == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Activación no soportada: {name}")


# ============================================================
# MODELO
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers,
        activation="relu",
        dropout=0.0,
        use_batch_norm=False,
        leaky_relu_slope=0.01
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(get_activation_module(activation, negative_slope=leaky_relu_slope))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # salida lineal

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# DATALOADERS
# ============================================================

def make_dataloader(X, y, batch_size=16, shuffle=False, drop_last=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


# ============================================================
# EARLY STOPPING
# ============================================================

@dataclass
class EarlyStoppingResult:
    best_state_dict: dict
    best_epoch: int
    best_val_loss: float
    train_losses: list
    val_losses: list


def train_one_model(
    model,
    train_loader,
    val_loader,
    optimizer_name="adam",
    learning_rate=1e-3,
    weight_decay=0.0,
    max_epochs=300,
    patience=20,
    verbose=False
):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_name}")

    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_loss = np.inf
    best_epoch = 0

    train_losses = []
    val_losses = []

    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(X_batch)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                running_val_loss += loss.item() * len(X_batch)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={epoch_train_loss:.6f} | "
                f"val_loss={epoch_val_loss:.6f}"
            )

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state_dict)

    return model, EarlyStoppingResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses
    )


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_model(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    return preds


# ============================================================
# PREPARACIÓN DE DATOS
# ============================================================

def split_target_features(df):
    y = df.iloc[:, 0].values.astype(float)
    X = df.iloc[:, 1:].values.astype(float)
    return X, y


def fit_transform_scalers(X_train, y_train, scale_y=True):
    x_scaler = StandardScaler()
    X_train_sc = x_scaler.fit_transform(X_train)

    if scale_y:
        y_scaler = StandardScaler()
        y_train_sc = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    else:
        y_scaler = None
        y_train_sc = y_train.copy()

    return X_train_sc, y_train_sc, x_scaler, y_scaler


def transform_with_scalers(X, y, x_scaler, y_scaler=None):
    X_sc = x_scaler.transform(X)
    if y_scaler is not None:
        y_sc = y_scaler.transform(y.reshape(-1, 1)).flatten()
    else:
        y_sc = y.copy()
    return X_sc, y_sc


def inverse_transform_target(y_scaled, y_scaler=None):
    if y_scaler is None:
        return y_scaled.copy()
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


# ============================================================
# PARSEO DE HIPERPARÁMETROS
# ============================================================

def parse_hidden_layers(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, str):
        return tuple(ast.literal_eval(value))
    raise ValueError(f"No pude interpretar hidden_layers={value}")


def suggest_float_from_config(trial, name, cfg):
    """
    Sugiere un hiperparámetro float desde una configuración flexible.

    Formatos aceptados:
      (low, high)          -> búsqueda continua
      (low, high, "log")   -> búsqueda logarítmica
      (low, high, step)    -> búsqueda discreta con step
    """
    if not isinstance(cfg, tuple):
        raise ValueError(
            f"La configuración de {name} debe ser una tupla. Recibido: {cfg}"
        )

    if len(cfg) == 2:
        low, high = cfg
        return trial.suggest_float(name, low, high)

    if len(cfg) == 3:
        low, high, third = cfg

        if isinstance(third, str):
            if third.lower() == "log":
                return trial.suggest_float(name, low, high, log=True)
            raise ValueError(
                f"Modo no reconocido para {name}: {third}. Usá 'log' o un step numérico."
            )

        if isinstance(third, (int, float)):
            step = third
            return trial.suggest_float(name, low, high, step=step)

        raise ValueError(
            f"Tercer elemento inválido para {name}: {third}. Debe ser 'log' o step numérico."
        )

    raise ValueError(
        f"Configuración inválida para {name}: {cfg}. Usá (low, high), (low, high, 'log') o (low, high, step)."
    )


def suggest_from_search_space(trial, search_space):
    params = {}

    hidden_choices = search_space["hidden_layers"]
    hidden_choices_as_str = [str(tuple(h)) for h in hidden_choices]
    params["hidden_layers"] = parse_hidden_layers(
        trial.suggest_categorical("hidden_layers", hidden_choices_as_str)
    )

    params["activation"] = trial.suggest_categorical(
        "activation",
        search_space["activation"]
    )

    if "dropout" in search_space:
        params["dropout"] = suggest_float_from_config(
            trial,
            "dropout",
            search_space["dropout"]
        )
    else:
        params["dropout"] = 0.0

    if "learning_rate" in search_space:
        params["learning_rate"] = suggest_float_from_config(
            trial,
            "learning_rate",
            search_space["learning_rate"]
        )
    else:
        params["learning_rate"] = 1e-3

    if "weight_decay" in search_space:
        params["weight_decay"] = suggest_float_from_config(
            trial,
            "weight_decay",
            search_space["weight_decay"]
        )
    else:
        params["weight_decay"] = 0.0

    if "batch_size" in search_space:
        params["batch_size"] = trial.suggest_categorical(
            "batch_size",
            search_space["batch_size"]
        )
    else:
        params["batch_size"] = 16

    if "optimizer" in search_space:
        params["optimizer"] = trial.suggest_categorical(
            "optimizer",
            search_space["optimizer"]
        )
    else:
        params["optimizer"] = "adam"

    if "use_batch_norm" in search_space:
        params["use_batch_norm"] = trial.suggest_categorical(
            "use_batch_norm",
            search_space["use_batch_norm"]
        )
    else:
        params["use_batch_norm"] = False

    if "leaky_relu_slope" in search_space:
        params["leaky_relu_slope"] = suggest_float_from_config(
            trial,
            "leaky_relu_slope",
            search_space["leaky_relu_slope"]
        )
    else:
        params["leaky_relu_slope"] = 0.01

    return params

# ============================================================
# OBJECTIVE DE OPTUNA
# ============================================================

def build_optuna_objective(
    df_train_full,
    search_space,
    cv_splits=3,
    final_val_size=0.15,
    max_epochs=300,
    patience=20,
    scale_y=True,
    pruning=False,
    seed=42,
    trial_fold_rows=None
):
    def objective(trial):
        set_global_seed(seed)

        params = suggest_from_search_space(trial, search_space)

        X_all, y_all = split_target_features(df_train_full)

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        fold_scores = []

        for fold_number, (train_idx, val_idx) in enumerate(tscv.split(X_all), start=1):
            X_fold_train = X_all[train_idx]
            y_fold_train = y_all[train_idx]

            X_fold_val = X_all[val_idx]
            y_fold_val = y_all[val_idx]

            X_subtrain, X_es, y_subtrain, y_es = temporal_subtrain_validation_split_arrays(
                X_fold_train, y_fold_train, val_size=final_val_size
            )

            X_subtrain_sc, y_subtrain_sc, x_scaler, y_scaler = fit_transform_scalers(
                X_subtrain, y_subtrain, scale_y=scale_y
            )

            X_es_sc, y_es_sc = transform_with_scalers(X_es, y_es, x_scaler, y_scaler)
            X_fold_val_sc, _ = transform_with_scalers(X_fold_val, y_fold_val, x_scaler, y_scaler)

            model = MLPRegressor(
                input_dim=X_subtrain_sc.shape[1],
                hidden_layers=params["hidden_layers"],
                activation=params["activation"],
                dropout=params["dropout"],
                use_batch_norm=params["use_batch_norm"],
                leaky_relu_slope=params["leaky_relu_slope"]
            )

            train_loader = make_dataloader(
                X_subtrain_sc, y_subtrain_sc,
                batch_size=params["batch_size"],
                shuffle=False,
                drop_last=params["use_batch_norm"]
            )
            es_loader = make_dataloader(
                X_es_sc, y_es_sc,
                batch_size=params["batch_size"],
                shuffle=False,
                drop_last=False
            )

            model, training_result = train_one_model(
                model=model,
                train_loader=train_loader,
                val_loader=es_loader,
                optimizer_name=params["optimizer"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                max_epochs=max_epochs,
                patience=patience,
                verbose=False
            )

            y_val_pred_sc = predict_model(model, X_fold_val_sc)
            y_val_pred = inverse_transform_target(y_val_pred_sc, y_scaler)

            fold_nse = nash_sutcliffe_efficiency(y_fold_val, y_val_pred)
            fold_scores.append(fold_nse)

            if trial_fold_rows is not None:
                trial_fold_rows.append({
                    "trial_number": trial.number,
                    "fold_number": fold_number,
                    "fold_nse": float(fold_nse),
                    "best_epoch_fold": training_result.best_epoch,
                    "best_val_loss_fold": float(training_result.best_val_loss),
                    "n_train_fold": len(X_fold_train),
                    "n_subtrain_fold": len(X_subtrain),
                    "n_es_fold": len(X_es),
                    "n_val_fold": len(X_fold_val),
                    **params,
                })

            mean_score_so_far = float(np.nanmean(fold_scores))
            trial.report(mean_score_so_far, step=fold_number)

            if pruning and trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.nanmean(fold_scores))

    return objective


# ============================================================
# ENTRENAMIENTO FINAL
# ============================================================

def train_final_model_on_train_full(
    df_train_full,
    best_params,
    final_val_size=0.15,
    max_epochs=300,
    patience=20,
    scale_y=True,
    seed=42
):
    set_global_seed(seed)

    X_all, y_all = split_target_features(df_train_full)

    X_subtrain, X_es, y_subtrain, y_es = temporal_subtrain_validation_split_arrays(
        X_all, y_all, val_size=final_val_size
    )

    X_subtrain_sc, y_subtrain_sc, x_scaler, y_scaler = fit_transform_scalers(
        X_subtrain, y_subtrain, scale_y=scale_y
    )

    X_es_sc, y_es_sc = transform_with_scalers(X_es, y_es, x_scaler, y_scaler)

    model = MLPRegressor(
        input_dim=X_subtrain_sc.shape[1],
        hidden_layers=parse_hidden_layers(best_params["hidden_layers"]),
        activation=best_params["activation"],
        dropout=best_params["dropout"],
        use_batch_norm=best_params["use_batch_norm"],
        leaky_relu_slope=best_params["leaky_relu_slope"]
    )

    train_loader = make_dataloader(
        X_subtrain_sc, y_subtrain_sc,
        batch_size=best_params["batch_size"],
        shuffle=False,
        drop_last=best_params["use_batch_norm"]
    )
    es_loader = make_dataloader(
        X_es_sc, y_es_sc,
        batch_size=best_params["batch_size"],
        shuffle=False,
        drop_last=False
    )

    model, training_result = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=es_loader,
        optimizer_name=best_params["optimizer"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        max_epochs=max_epochs,
        patience=patience,
        verbose=False
    )

    return model, x_scaler, y_scaler, training_result


# ============================================================
# PLOTEOS
# ============================================================

def plot_loss_curve(training_result, title="Curva de pérdida"):
    plt.figure(figsize=(8, 5))
    plt.plot(training_result.train_losses, label="Train loss")
    plt.plot(training_result.val_losses, label="Val interna loss")
    if training_result.best_epoch is not None and training_result.best_epoch > 0:
        plt.axvline(training_result.best_epoch - 1, linestyle="--", label="Best epoch")
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_optuna_history_simple(study, title="Historia de optimización"):
    trial_values = [
        t.value for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    if len(trial_values) == 0:
        return

    best_so_far = np.maximum.accumulate(trial_values)

    plt.figure(figsize=(8, 5))
    plt.plot(trial_values, linestyle="", marker="o", alpha=0.8, label="Trial")
    plt.plot(best_so_far, linewidth=2, label="Best so far")
    plt.title(title)
    plt.xlabel("Trial")
    plt.ylabel("NSE medio CV")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_time_series_real_vs_pred(index, y_true, y_pred, title="Real vs predicho"):
    plt.figure(figsize=(12, 5))
    plt.plot(index, y_true, label="Real", linewidth=2)
    plt.plot(index, y_pred, label="Predicho", linewidth=2)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_scatter_real_vs_pred(y_true, y_pred, title="Scatter real vs predicho"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title(title)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_residual_histogram(y_true, y_pred, title="Histograma de residuos"):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    plt.figure(figsize=(7, 4.5))
    plt.hist(residuals, bins=15, edgecolor="black")
    plt.title(title)
    plt.xlabel("Residuo (pred - real)")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def process_sets_mlp_temporal(
    sets_dict,
    search_space,
    n_trials=30,
    train_size=0.8,
    cv_splits=3,
    final_val_size=0.15,
    max_epochs=300,
    patience=20,
    scale_y=True,
    make_plots=True,
    pruning=True,
    seed=42
):
    set_global_seed(seed)

    results_rows = []
    details = {}

    for set_idx, (set_name, df) in enumerate(sets_dict.items()):
        print("\n" + "=" * 90)
        print(f"Procesando set: {set_name}")
        print("=" * 90)

        # Misma semilla para todos los datasets
        set_seed = seed
        set_global_seed(set_seed)
        set_start_time = time.perf_counter()

        # IMPORTANTE: sampler y pruner reiniciados por set
        sampler = optuna.samplers.TPESampler(seed=set_seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        df = df.dropna().copy().sort_index()

        if df.shape[1] < 2:
            print(f"Set omitido: {set_name} no tiene predictors suficientes.")
            continue

        if len(df) < 20:
            print(f"Set omitido: {set_name} tiene muy pocas observaciones ({len(df)}).")
            continue

        train_df, test_df = temporal_train_test_split_df(df, train_size=train_size)
        trial_fold_rows = []

        objective = build_optuna_objective(
            df_train_full=train_df,
            search_space=search_space,
            cv_splits=cv_splits,
            final_val_size=final_val_size,
            max_epochs=max_epochs,
            patience=patience,
            scale_y=scale_y,
            pruning=pruning,
            seed=set_seed,
            trial_fold_rows=trial_fold_rows
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            catch=(ValueError,)
        )

        if len(study.trials) == 0 or study.best_trial is None:
            print(f"No hubo trials válidos para {set_name}.")
            continue

        best_params = study.best_trial.params.copy()
        best_params["hidden_layers"] = parse_hidden_layers(best_params["hidden_layers"])

        if "optimizer" not in best_params:
            best_params["optimizer"] = "adam"
        if "use_batch_norm" not in best_params:
            best_params["use_batch_norm"] = False
        if "leaky_relu_slope" not in best_params:
            best_params["leaky_relu_slope"] = 0.01

        best_cv_nse = study.best_value

        model, x_scaler, y_scaler, training_result = train_final_model_on_train_full(
            df_train_full=train_df,
            best_params=best_params,
            final_val_size=final_val_size,
            max_epochs=max_epochs,
            patience=patience,
            scale_y=scale_y,
            seed=set_seed
        )

        X_train, y_train = split_target_features(train_df)
        X_train_sc, _ = transform_with_scalers(X_train, y_train, x_scaler, y_scaler)
        y_train_pred_sc = predict_model(model, X_train_sc)
        y_train_pred = inverse_transform_target(y_train_pred_sc, y_scaler)

        X_test, y_test = split_target_features(test_df)
        X_test_sc, _ = transform_with_scalers(X_test, y_test, x_scaler, y_scaler)
        y_test_pred_sc = predict_model(model, X_test_sc)
        y_test_pred = inverse_transform_target(y_test_pred_sc, y_scaler)

        train_metrics = compute_regression_metrics(y_train, y_train_pred)
        test_metrics = compute_regression_metrics(y_test, y_test_pred)

        execution_seconds = time.perf_counter() - set_start_time
        execution_time = _mlp_format_elapsed_time(execution_seconds) if "_mlp_format_elapsed_time" in globals() else f"{execution_seconds:.2f} s"

        result_row = {
            "DataFrame": set_name,
            "n_obs": len(df),
            "n_features": df.shape[1] - 1,
            "execution_seconds": round_sig(execution_seconds, 3),
            "execution_time": execution_time,
            "best_cv_nse": round_sig(best_cv_nse, 2),
            "best_hidden_layers": best_params["hidden_layers"],
            "best_activation": best_params["activation"],
            "best_dropout": round_sig(best_params["dropout"], 3),
            "best_learning_rate": round_sig(best_params["learning_rate"], 3),
            "best_weight_decay": round_sig(best_params["weight_decay"], 3),
            "best_batch_size": best_params["batch_size"],
            "best_optimizer": best_params["optimizer"],
            "best_use_batch_norm": best_params["use_batch_norm"],
            "best_leaky_relu_slope": (
                round_sig(best_params["leaky_relu_slope"], 3)
                if best_params["activation"] == "leaky_relu" else None
            ),
            "best_epoch": training_result.best_epoch,

            "train_RMSE": train_metrics["RMSE"],
            "train_MAE": train_metrics["MAE"],
            "train_NSE": train_metrics["NSE"],
            "train_KGE": train_metrics["KGE"],
            "train_PBIAS": train_metrics["PBIAS"],
            "train_R2": train_metrics["R2"],
            "train_RSR": train_metrics["RSR"],

            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
            "test_NSE": test_metrics["NSE"],
            "test_KGE": test_metrics["KGE"],
            "test_PBIAS": test_metrics["PBIAS"],
            "test_R2": test_metrics["R2"],
            "test_RSR": test_metrics["RSR"],
        }
        results_rows.append(result_row)

        details[set_name] = {
            "study": study,
            "best_params": best_params,
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "training_result": training_result,
            "train_df": train_df,
            "test_df": test_df,
            "y_train": y_train,
            "y_train_pred": y_train_pred,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "seed_used": set_seed,
            "execution_seconds": round_sig(execution_seconds, 3),
            "execution_time": execution_time,
            "trial_fold_rows": trial_fold_rows,
        }

        print("\nMejores hiperparámetros:")
        print(best_params)

        print(f"\nSeed usada para este set: {set_seed}")
        print(f"\nBest CV NSE: {round_sig(best_cv_nse, 2)}")

        print("\nMétricas TRAIN:")
        print(train_metrics)

        print("\nMétricas TEST:")
        print(test_metrics)

        if make_plots:
            plot_optuna_history_simple(
                study,
                title=f"{set_name} - Historia de Optuna"
            )

            plot_loss_curve(
                training_result,
                title=f"{set_name} - Curva de pérdida final"
            )

            plot_time_series_real_vs_pred(
                train_df.index,
                y_train,
                y_train_pred,
                title=f"{set_name} - TRAIN real vs predicho"
            )

            plot_time_series_real_vs_pred(
                test_df.index,
                y_test,
                y_test_pred,
                title=f"{set_name} - TEST real vs predicho"
            )

            plot_scatter_real_vs_pred(
                y_train,
                y_train_pred,
                title=f"{set_name} - Scatter TRAIN"
            )

            plot_scatter_real_vs_pred(
                y_test,
                y_test_pred,
                title=f"{set_name} - Scatter TEST"
            )

            plot_residual_histogram(
                y_test,
                y_test_pred,
                title=f"{set_name} - Histograma residuos TEST"
            )

    results_df = pd.DataFrame(results_rows)
    return results_df, details


# ============================================================
# CAPA DE REPORTE/EXPORTACIÓN

# ============================================================

_process_sets_mlp_temporal_core = process_sets_mlp_temporal


def _mlp_build_timestamped_output_dir(base_name="Resultados_MLP"):
    from datetime import datetime
    now = datetime.now()
    timestamp = f"{now.day}-{now.month}--{now.hour}-{now.minute:02d}"
    return f"{base_name}_{timestamp}"


def _mlp_format_elapsed_time(seconds):
    seconds = float(seconds)
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} s"


def _mlp_get_variable_name(var, local_vars):
    matches = [name for name, val in local_vars if val is var]
    return matches[0] if matches else "diccionario_sets_mlp"


def _mlp_search_space_to_text(search_space):
    text = {}
    for k, v in search_space.items():
        if isinstance(v, tuple):
            if len(v) == 2:
                text[k] = f"{v[0]} a {v[1]}"
            elif len(v) == 3:
                modo = "log" if v[2] == "log" else f"step={v[2]}"
                text[k] = f"{v[0]} a {v[1]} ({modo})"
            else:
                text[k] = str(v)
        else:
            text[k] = str(v)
    return text


def _mlp_classify_nse_performance(nse_value):
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


def _mlp_round_params_dict(params, sig=3):
    rounded = {}
    for k, v in params.items():
        if isinstance(v, (int, np.integer)):
            rounded[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            rounded[k] = round_sig(v, sig)
        else:
            rounded[k] = v
    return rounded


def _mlp_reconstruct_fold_scores_from_intermediate(trial):
    """
    El core original guarda en trial.intermediate_values el promedio acumulado
    hasta cada fold. Esto reconstruye folds sin tocar el objective:
    f1=m1, f2=2*m2-m1, f3=3*m3-2*m2, etc.
    """
    if trial is None or not getattr(trial, "intermediate_values", None):
        return []
    items = sorted(trial.intermediate_values.items(), key=lambda kv: kv[0])
    fold_scores = []
    prev_mean = None
    prev_k = 0
    for step, mean_so_far in items:
        try:
            k = int(step)
            m = float(mean_so_far)
        except Exception:
            continue
        if k <= 0 or np.isnan(m):
            continue
        if prev_mean is None:
            fold_score = m
        else:
            fold_score = k * m - prev_k * prev_mean
        fold_scores.append(fold_score)
        prev_mean = m
        prev_k = k
    return fold_scores


def _mlp_cv_std_from_study(study):
    try:
        folds = _mlp_reconstruct_fold_scores_from_intermediate(study.best_trial)
        if len(folds) >= 2:
            return round_sig(float(np.nanstd(folds)), 2)
        if len(folds) == 1:
            return 0.0
    except Exception:
        pass
    return np.nan


def _mlp_build_trials_dataframe(study):
    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": str(trial.state).replace("TrialState.", ""),
            "value": trial.value,
        }
        try:
            row["datetime_start"] = trial.datetime_start
            row["datetime_complete"] = trial.datetime_complete
            if trial.datetime_start is not None and trial.datetime_complete is not None:
                row["duration_seconds"] = (trial.datetime_complete - trial.datetime_start).total_seconds()
        except Exception:
            pass
        for param_name, param_value in trial.params.items():
            row[param_name] = param_value
        for step, val in sorted(trial.intermediate_values.items()):
            row[f"cv_mean_until_fold_{step}"] = val
        folds = _mlp_reconstruct_fold_scores_from_intermediate(trial)
        for i, fold_score in enumerate(folds, start=1):
            row[f"fold_{i}_nse_reconstruido"] = fold_score
        if len(folds) >= 1:
            row["cv_nse_mean_reconstruido"] = float(np.nanmean(folds))
        if len(folds) >= 2:
            row["cv_nse_std_reconstruido"] = float(np.nanstd(folds))
        elif len(folds) == 1:
            row["cv_nse_std_reconstruido"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _mlp_build_fold_scores_dataframe(detail):
    """Construye una tabla explícita trial × fold con los valores registrados durante Optuna."""
    rows = detail.get("trial_fold_rows", []) or []
    base_columns = [
        "trial_number", "trial_state", "fold_number", "fold_nse",
        "best_epoch_fold", "best_val_loss_fold",
        "n_train_fold", "n_subtrain_fold", "n_es_fold", "n_val_fold",
    ]
    if not rows:
        return pd.DataFrame(columns=base_columns)

    df = pd.DataFrame(rows)
    state_map = {
        trial.number: str(trial.state).replace("TrialState.", "")
        for trial in detail["study"].trials
    }
    df.insert(1, "trial_state", df["trial_number"].map(state_map))

    ordered = [col for col in base_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[ordered + remaining]


def _mlp_export_prediction_csvs(set_dir, dataframe_name, train_df, test_df, y_train_pred, y_test_pred):
    y_train = train_df.iloc[:, 0]
    y_test = test_df.iloc[:, 0]
    train_pred_df = pd.DataFrame({
        "Fecha": y_train.index,
        "y_observado": y_train.values,
        "y_predicho": np.asarray(y_train_pred),
    })
    train_pred_df["residuo"] = train_pred_df["y_observado"] - train_pred_df["y_predicho"]
    test_pred_df = pd.DataFrame({
        "Fecha": y_test.index,
        "y_observado": y_test.values,
        "y_predicho": np.asarray(y_test_pred),
    })
    test_pred_df["residuo"] = test_pred_df["y_observado"] - test_pred_df["y_predicho"]
    full_temporal_df = pd.concat([
        train_pred_df.assign(particion="train"),
        test_pred_df.assign(particion="test"),
    ], ignore_index=True)
    train_pred_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_predicciones_train.csv"), index=False)
    test_pred_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_predicciones_test.csv"), index=False)
    full_temporal_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_serie_temporal_completa.csv"), index=False)
    return train_pred_df, test_pred_df, full_temporal_df


def _mlp_export_loss_curve_csv(set_dir, dataframe_name, training_result):
    train_losses = list(training_result.train_losses)
    val_losses = list(training_result.val_losses)
    n = max(len(train_losses), len(val_losses))
    best_epoch = getattr(training_result, "best_epoch", None)
    loss_df = pd.DataFrame({
        "epoch": np.arange(1, n + 1),
        "train_loss": train_losses + [np.nan] * (n - len(train_losses)),
        "val_loss": val_losses + [np.nan] * (n - len(val_losses)),
    })
    loss_df["is_best_epoch"] = loss_df["epoch"] == best_epoch
    loss_df.to_csv(os.path.join(set_dir, f"{dataframe_name}_loss_curve.csv"), index=False)
    return loss_df


def _mlp_export_param_importances_csv(set_dir, dataframe_name, study):
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}
    df = pd.DataFrame([{"hiperparametro": k, "importancia": v} for k, v in importances.items()])
    df.to_csv(os.path.join(set_dir, f"{dataframe_name}_hyperparam_importances.csv"), index=False)
    return df, importances


def _mlp_compute_captum_attributions(detail):
    try:
        from captum.attr import IntegratedGradients
    except Exception:
        return None, None, None, "Captum no está instalado o no pudo importarse."
    try:
        model = detail["model"]
        train_df = detail["train_df"]
        x_scaler = detail["x_scaler"]
        feature_names = train_df.columns[1:].tolist()
        X_train = train_df.iloc[:, 1:].values.astype(float)
        X_train_sc = x_scaler.transform(X_train)
        model.eval()
        X_tensor = torch.tensor(X_train_sc, dtype=torch.float32).to(DEVICE)
        baseline = torch.zeros_like(X_tensor)
        ig = IntegratedGradients(model)
        attrs = ig.attribute(X_tensor, baselines=baseline)
        attrs_np = attrs.detach().cpu().numpy()
        attributions_df = pd.DataFrame(attrs_np, columns=feature_names, index=train_df.index)
        attributions_df.index.name = "Fecha"
        mean_abs_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_attribution": np.abs(attrs_np).mean(axis=0),
            "mean_attribution": attrs_np.mean(axis=0),
            "std_attribution": attrs_np.std(axis=0),
        }).sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)
        top_rows = []
        for idx, row in attributions_df.iterrows():
            abs_row = row.abs()
            top_feature = abs_row.idxmax()
            top_rows.append({
                "Fecha": idx,
                "top_feature": top_feature,
                "top_attribution": row[top_feature],
                "top_abs_attribution": abs_row[top_feature],
            })
        top_df = pd.DataFrame(top_rows)
        return attributions_df, mean_abs_df, top_df, None
    except Exception as e:
        return None, None, None, f"Error calculando Captum: {repr(e)}"


def _mlp_build_summary_tables(results_df, details, experiment_label, max_epochs, patience, final_val_size, scale_y, pruning):
    train_rows, test_rows, params_rows, metadata_rows, comparison_rows = [], [], [], [], []
    metric_names = ["RMSE", "MAE", "NSE", "KGE", "PBIAS", "R2", "RSR"]
    for _, row in results_df.iterrows():
        name = row["DataFrame"]
        detail = details[name]
        train_df = detail["train_df"]
        test_df = detail["test_df"]
        study = detail["study"]
        target = train_df.columns[0]
        train_metrics = detail["train_metrics"]
        test_metrics = detail["test_metrics"]
        cv_mean = row["best_cv_nse"]
        cv_std = _mlp_cv_std_from_study(study)
        train_rows.append({"DataFrame": name, **train_metrics})
        test_rows.append({"DataFrame": name, **test_metrics})
        params_rows.append({
            "DataFrame": name,
            **_mlp_round_params_dict(detail["best_params"], sig=3),
            "best_epoch": detail["training_result"].best_epoch,
            "best_val_loss": round_sig(detail["training_result"].best_val_loss, 3),
        })
        metadata_rows.append({
            "DataFrame": name,
            "Target": target,
            "experiment_label": experiment_label,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": train_df.shape[1] - 1,
            "train_start": train_df.index.min(),
            "train_end": train_df.index.max(),
            "test_start": test_df.index.min(),
            "test_end": test_df.index.max(),
            "cv_nse_mean": cv_mean,
            "cv_nse_std": cv_std,
            "best_epoch": detail["training_result"].best_epoch,
            "best_val_loss": round_sig(detail["training_result"].best_val_loss, 3),
            "max_epochs": max_epochs,
            "patience": patience,
            "final_val_size": final_val_size,
            "scale_y": scale_y,
            "pruning": pruning,
            "seed_used": detail.get("seed_used", None),
            "execution_seconds": detail.get("execution_seconds", np.nan),
            "execution_time": detail.get("execution_time", ""),
        })
        comp = {("DataFrame", ""): name}
        comp[("NSE", "CV Mean")] = cv_mean
        comp[("NSE", "CV Std")] = cv_std
        comp[("NSE", "CV Performance")] = _mlp_classify_nse_performance(cv_mean)
        comp[("NSE", "Train")] = train_metrics["NSE"]
        comp[("NSE", "Test")] = test_metrics["NSE"]
        comp[("NSE", "Test Performance")] = _mlp_classify_nse_performance(test_metrics["NSE"])
        for metric in [m for m in metric_names if m != "NSE"]:
            comp[(metric, "Train")] = train_metrics[metric]
            comp[(metric, "Test")] = test_metrics[metric]
        comp[("EarlyStop", "BestEpoch")] = detail["training_result"].best_epoch
        comp[("Tiempo", "Segundos")] = detail.get("execution_seconds", np.nan)
        comp[("Tiempo", "Texto")] = detail.get("execution_time", "")
        comparison_rows.append(comp)
    train_metrics_df = pd.DataFrame(train_rows)
    test_metrics_df = pd.DataFrame(test_rows)
    params_df = pd.DataFrame(params_rows)
    metadata_df = pd.DataFrame(metadata_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df.columns = pd.MultiIndex.from_tuples(comparison_df.columns)
    if "NSE" in test_metrics_df.columns:
        test_metrics_df = test_metrics_df.sort_values("NSE", ascending=False).reset_index(drop=True)
        ordered = test_metrics_df["DataFrame"].tolist()
        train_metrics_df = train_metrics_df.set_index("DataFrame").loc[ordered].reset_index()
        params_df = params_df.set_index("DataFrame").loc[ordered].reset_index()
        metadata_df = metadata_df.set_index("DataFrame").loc[ordered].reset_index()
    if not comparison_df.empty and ("NSE", "Test") in comparison_df.columns:
        comparison_df = comparison_df.sort_values(("NSE", "Test"), ascending=False).reset_index(drop=True)
    global_nse_stats_df = pd.DataFrame({
        "Estadística": ["Promedio", "Mediana", "Std"],
        "CV Mean": [round_sig(pd.to_numeric(metadata_df["cv_nse_mean"], errors="coerce").mean(), 2),
                    round_sig(pd.to_numeric(metadata_df["cv_nse_mean"], errors="coerce").median(), 2),
                    round_sig(pd.to_numeric(metadata_df["cv_nse_mean"], errors="coerce").std(), 2)],
        "CV Std": [round_sig(pd.to_numeric(metadata_df["cv_nse_std"], errors="coerce").mean(), 2),
                   round_sig(pd.to_numeric(metadata_df["cv_nse_std"], errors="coerce").median(), 2),
                   round_sig(pd.to_numeric(metadata_df["cv_nse_std"], errors="coerce").std(), 2)],
        "Train": [round_sig(pd.to_numeric(train_metrics_df["NSE"], errors="coerce").mean(), 2),
                  round_sig(pd.to_numeric(train_metrics_df["NSE"], errors="coerce").median(), 2),
                  round_sig(pd.to_numeric(train_metrics_df["NSE"], errors="coerce").std(), 2)],
        "Test": [round_sig(pd.to_numeric(test_metrics_df["NSE"], errors="coerce").mean(), 2),
                 round_sig(pd.to_numeric(test_metrics_df["NSE"], errors="coerce").median(), 2),
                 round_sig(pd.to_numeric(test_metrics_df["NSE"], errors="coerce").std(), 2)],
    })
    return train_metrics_df, test_metrics_df, params_df, metadata_df, comparison_df, global_nse_stats_df


def _mlp_hyperimportance_families(param_importances):
    param_importances = param_importances or {}

    architecture_params = {"hidden_layers", "activation", "use_batch_norm", "leaky_relu_slope"}
    regularization_params = {"dropout", "weight_decay"}
    optimization_params = {"learning_rate", "batch_size", "optimizer"}

    architecture = 0.0
    regularization = 0.0
    optimization = 0.0
    other = 0.0

    for name, value in param_importances.items():
        try:
            value_float = float(value)
        except Exception:
            value_float = 0.0

        if name in architecture_params:
            architecture += value_float
        elif name in regularization_params:
            regularization += value_float
        elif name in optimization_params:
            optimization += value_float
        else:
            other += value_float

    return architecture, regularization, optimization, other


def _mlp_feature_importance_stats(captum_mean_abs_df):
    stats = {
        "featureimp_total": np.nan,
        "featureimp_top3_sum": np.nan,
        "featureimp_top5_sum": np.nan,
        "featureimp_top3_share": np.nan,
        "featureimp_top5_share": np.nan,
    }
    top_entries = []

    if captum_mean_abs_df is None or not isinstance(captum_mean_abs_df, pd.DataFrame) or captum_mean_abs_df.empty:
        return top_entries, stats
    if "feature" not in captum_mean_abs_df.columns or "mean_abs_attribution" not in captum_mean_abs_df.columns:
        return top_entries, stats

    df = captum_mean_abs_df[["feature", "mean_abs_attribution"]].copy()
    df["mean_abs_attribution"] = pd.to_numeric(df["mean_abs_attribution"], errors="coerce")
    df = df.dropna(subset=["mean_abs_attribution"]).sort_values("mean_abs_attribution", ascending=False).reset_index(drop=True)
    if df.empty:
        return top_entries, stats

    total = float(df["mean_abs_attribution"].sum())
    top3_sum = float(df.head(3)["mean_abs_attribution"].sum())
    top5_sum = float(df.head(5)["mean_abs_attribution"].sum())

    stats["featureimp_total"] = total
    stats["featureimp_top3_sum"] = top3_sum
    stats["featureimp_top5_sum"] = top5_sum
    stats["featureimp_top3_share"] = top3_sum / total if total != 0 else np.nan
    stats["featureimp_top5_share"] = top5_sum / total if total != 0 else np.nan

    for _, row in df.head(5).iterrows():
        top_entries.append((row["feature"], row["mean_abs_attribution"]))

    return top_entries, stats


MLP_DIAGNOSTIC_COLUMNS = [
    "dataset_name", "target", "model_type", "config_mode", "search_space_type",
    "use_early_stopping", "final_val_size", "n_trials", "cv_splits", "train_size",
    "max_epochs", "patience", "scale_y", "pruning", "seed", "make_plots",
    "train_start", "train_end", "test_start", "test_end", "n_train", "n_test", "n_features",
    "cv_nse_mean", "cv_nse_std", "cv_nse_performance",
    "train_rmse", "train_mae", "train_nse", "train_kge", "train_pbias", "train_r2", "train_rsr",
    "test_rmse", "test_mae", "test_nse", "test_kge", "test_pbias", "test_r2", "test_rsr",
    "test_nse_performance", "gap_train_cv_nse", "gap_train_test_nse", "gap_cv_test_nse",
    "execution_seconds", "execution_time",
    "param_hidden_layers", "param_activation", "param_dropout", "param_learning_rate",
    "param_weight_decay", "param_batch_size", "param_optimizer", "param_use_batch_norm",
    "param_leaky_relu_slope",
    "best_epoch", "best_val_loss", "n_epochs_trained", "early_stopped", "train_loss_at_best",
    "last_train_loss", "last_val_loss", "gap_loss_at_best", "epochs_after_best",
    "range_hidden_layers", "range_activation", "range_dropout", "range_learning_rate",
    "range_weight_decay", "range_batch_size", "range_optimizer", "range_use_batch_norm",
    "range_leaky_relu_slope",
    "hyperimp_top1_name", "hyperimp_top1_value", "hyperimp_top2_name", "hyperimp_top2_value",
    "hyperimp_top3_name", "hyperimp_top3_value", "hyperimp_top4_name", "hyperimp_top4_value",
    "hyperimp_top5_name", "hyperimp_top5_value",
    "hyperimp_architecture", "hyperimp_regularization", "hyperimp_optimization", "hyperimp_other",
    "featureimp_top1_name", "featureimp_top1_value", "featureimp_top2_name", "featureimp_top2_value",
    "featureimp_top3_name", "featureimp_top3_value", "featureimp_top4_name", "featureimp_top4_value",
    "featureimp_top5_name", "featureimp_top5_value",
    "featureimp_total", "featureimp_top3_sum", "featureimp_top5_sum",
    "featureimp_top3_share", "featureimp_top5_share",
]

if len(MLP_DIAGNOSTIC_COLUMNS) != 102:
    raise RuntimeError(f"MLP_DIAGNOSTIC_COLUMNS debe contener 102 columnas, contiene {len(MLP_DIAGNOSTIC_COLUMNS)}.")


def _mlp_build_integrated_diagnostics_dataframe(
    details,
    metadata_df,
    search_space,
    experiment_label,
    n_trials,
    train_size,
    cv_splits,
    final_val_size,
    max_epochs,
    patience,
    scale_y,
    pruning,
    seed,
    make_plots,
):
    rows = []
    search_ranges = _mlp_search_space_to_text(search_space)

    if metadata_df is not None and not metadata_df.empty and "DataFrame" in metadata_df.columns:
        ordered_names = metadata_df["DataFrame"].tolist()
        meta_by_name = metadata_df.set_index("DataFrame").to_dict(orient="index")
    else:
        ordered_names = list(details.keys())
        meta_by_name = {}

    param_names = [
        "hidden_layers", "activation", "dropout", "learning_rate", "weight_decay",
        "batch_size", "optimizer", "use_batch_norm", "leaky_relu_slope",
    ]

    for name in ordered_names:
        if name not in details:
            continue

        detail = details[name]
        meta = meta_by_name.get(name, {})
        train_df = detail["train_df"]
        test_df = detail["test_df"]
        train_metrics = detail.get("train_metrics", {}) or {}
        test_metrics = detail.get("test_metrics", {}) or {}
        best_params = detail.get("best_params", {}) or {}
        training_result = detail["training_result"]
        param_importances = detail.get("param_importances", {}) or {}
        captum_mean_abs_df = detail.get("captum_mean_abs_df", pd.DataFrame())

        cv_nse_mean = meta.get("cv_nse_mean", np.nan)
        cv_nse_std = meta.get("cv_nse_std", np.nan)
        train_nse = train_metrics.get("NSE", np.nan)
        test_nse = test_metrics.get("NSE", np.nan)

        train_losses = list(getattr(training_result, "train_losses", []) or [])
        val_losses = list(getattr(training_result, "val_losses", []) or [])
        n_epochs_trained = max(len(train_losses), len(val_losses))
        best_epoch = getattr(training_result, "best_epoch", None)
        best_val_loss = getattr(training_result, "best_val_loss", np.nan)

        train_loss_at_best = np.nan
        if best_epoch is not None:
            try:
                best_idx = int(best_epoch) - 1
                if 0 <= best_idx < len(train_losses):
                    train_loss_at_best = float(train_losses[best_idx])
            except Exception:
                pass

        last_train_loss = float(train_losses[-1]) if train_losses else np.nan
        last_val_loss = float(val_losses[-1]) if val_losses else np.nan
        gap_loss_at_best = (
            float(best_val_loss) - float(train_loss_at_best)
            if pd.notna(best_val_loss) and pd.notna(train_loss_at_best)
            else np.nan
        )
        try:
            epochs_after_best = n_epochs_trained - int(best_epoch) if best_epoch is not None else np.nan
        except Exception:
            epochs_after_best = np.nan

        row = {
            "dataset_name": name,
            "target": train_df.columns[0],
            "model_type": "MLP",
            "config_mode": experiment_label,
            "search_space_type": "custom_mlp",
            "use_early_stopping": True,
            "final_val_size": final_val_size,
            "n_trials": n_trials,
            "cv_splits": cv_splits,
            "train_size": train_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "scale_y": scale_y,
            "pruning": pruning,
            "seed": seed,
            "make_plots": make_plots,
            "train_start": train_df.index.min(),
            "train_end": train_df.index.max(),
            "test_start": test_df.index.min(),
            "test_end": test_df.index.max(),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": train_df.shape[1] - 1,
            "cv_nse_mean": cv_nse_mean,
            "cv_nse_std": cv_nse_std,
            "cv_nse_performance": _mlp_classify_nse_performance(cv_nse_mean),
            "train_rmse": train_metrics.get("RMSE", np.nan),
            "train_mae": train_metrics.get("MAE", np.nan),
            "train_nse": train_nse,
            "train_kge": train_metrics.get("KGE", np.nan),
            "train_pbias": train_metrics.get("PBIAS", np.nan),
            "train_r2": train_metrics.get("R2", np.nan),
            "train_rsr": train_metrics.get("RSR", np.nan),
            "test_rmse": test_metrics.get("RMSE", np.nan),
            "test_mae": test_metrics.get("MAE", np.nan),
            "test_nse": test_nse,
            "test_kge": test_metrics.get("KGE", np.nan),
            "test_pbias": test_metrics.get("PBIAS", np.nan),
            "test_r2": test_metrics.get("R2", np.nan),
            "test_rsr": test_metrics.get("RSR", np.nan),
            "test_nse_performance": _mlp_classify_nse_performance(test_nse),
            "gap_train_cv_nse": float(train_nse) - float(cv_nse_mean) if pd.notna(train_nse) and pd.notna(cv_nse_mean) else np.nan,
            "gap_train_test_nse": float(train_nse) - float(test_nse) if pd.notna(train_nse) and pd.notna(test_nse) else np.nan,
            "gap_cv_test_nse": float(cv_nse_mean) - float(test_nse) if pd.notna(cv_nse_mean) and pd.notna(test_nse) else np.nan,
            "execution_seconds": detail.get("execution_seconds", np.nan),
            "execution_time": detail.get("execution_time", ""),
            "best_epoch": best_epoch,
            "best_val_loss": round_sig(best_val_loss, 3),
            "n_epochs_trained": n_epochs_trained,
            "early_stopped": bool(n_epochs_trained < max_epochs),
            "train_loss_at_best": round_sig(train_loss_at_best, 3),
            "last_train_loss": round_sig(last_train_loss, 3),
            "last_val_loss": round_sig(last_val_loss, 3),
            "gap_loss_at_best": round_sig(gap_loss_at_best, 3),
            "epochs_after_best": epochs_after_best,
        }

        for param_name in param_names:
            row[f"param_{param_name}"] = best_params.get(param_name, np.nan)
            row[f"range_{param_name}"] = search_ranges.get(param_name, np.nan)

        sorted_importances = sorted(param_importances.items(), key=lambda item: item[1], reverse=True)
        for i in range(1, 6):
            if len(sorted_importances) >= i:
                row[f"hyperimp_top{i}_name"] = sorted_importances[i - 1][0]
                row[f"hyperimp_top{i}_value"] = sorted_importances[i - 1][1]
            else:
                row[f"hyperimp_top{i}_name"] = np.nan
                row[f"hyperimp_top{i}_value"] = np.nan

        architecture, regularization, optimization, other = _mlp_hyperimportance_families(param_importances)
        row["hyperimp_architecture"] = architecture
        row["hyperimp_regularization"] = regularization
        row["hyperimp_optimization"] = optimization
        row["hyperimp_other"] = other

        feature_top_entries, feature_stats = _mlp_feature_importance_stats(captum_mean_abs_df)
        for i in range(1, 6):
            if len(feature_top_entries) >= i:
                row[f"featureimp_top{i}_name"] = feature_top_entries[i - 1][0]
                row[f"featureimp_top{i}_value"] = feature_top_entries[i - 1][1]
            else:
                row[f"featureimp_top{i}_name"] = np.nan
                row[f"featureimp_top{i}_value"] = np.nan
        row.update(feature_stats)

        rows.append(row)

    diagnostics_df = pd.DataFrame(rows)
    if diagnostics_df.empty:
        return pd.DataFrame(columns=MLP_DIAGNOSTIC_COLUMNS)

    for col in MLP_DIAGNOSTIC_COLUMNS:
        if col not in diagnostics_df.columns:
            diagnostics_df[col] = np.nan

    return diagnostics_df[MLP_DIAGNOSTIC_COLUMNS]


def _mlp_generate_results_for_html(details, metadata_df, search_space, experiment_label):
    search_ranges = _mlp_search_space_to_text(search_space)
    meta_by_name = metadata_df.set_index("DataFrame").to_dict(orient="index") if metadata_df is not None and not metadata_df.empty else {}
    results = []
    for name, detail in details.items():
        train_df = detail["train_df"]
        test_df = detail["test_df"]
        meta = meta_by_name.get(name, {})
        results.append({
            "dataframe_name": name,
            "target": train_df.columns[0],
            "feature_names": train_df.columns[1:].tolist(),
            "config_mode": experiment_label,
            "search_space_type": "custom_mlp",
            "search_ranges": search_ranges,
            "use_early_stopping": True,
            "metrics_train": detail["train_metrics"],
            "metrics_test": detail["test_metrics"],
            "best_params": detail["best_params"],
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": train_df.shape[1] - 1,
            "train_start": train_df.index.min(),
            "train_end": train_df.index.max(),
            "test_start": test_df.index.min(),
            "test_end": test_df.index.max(),
            "cv_nse_mean": meta.get("cv_nse_mean", np.nan),
            "cv_nse_std": meta.get("cv_nse_std", np.nan),
            "best_epoch": detail["training_result"].best_epoch,
            "best_val_loss": round_sig(detail["training_result"].best_val_loss, 3),
            "max_epochs": meta.get("max_epochs", np.nan),
            "patience": meta.get("patience", np.nan),
            "final_val_size": meta.get("final_val_size", np.nan),
            "scale_y": meta.get("scale_y", ""),
            "pruning": meta.get("pruning", ""),
            "execution_seconds": meta.get("execution_seconds", np.nan),
            "execution_time": meta.get("execution_time", ""),
        })
    return results


def _mlp_build_artifacts_inventory(output_dir):
    rows = []
    for root, _, files in os.walk(output_dir):
        for filename in files:
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, output_dir)
            rows.append({
                "relative_path": rel_path,
                "filename": filename,
                "extension": os.path.splitext(filename)[1].lower(),
                "size_bytes": os.path.getsize(abs_path),
            })
    df = pd.DataFrame(rows).sort_values(["extension", "relative_path"]).reset_index(drop=True)
    df.to_csv(os.path.join(output_dir, "artifacts_inventory.csv"), index=False)
    return df


def _mlp_save_basic_plots(set_dir, dataframe_name, detail, target_column):
    try:
        from visualization_mlp import (
            plot_loss_curve_from_df,
            plot_objective_curve,
            plot_train_vs_actual,
            plot_test_vs_actual,
            plot_full_temporal_results,
            plot_real_vs_pred,
            plot_residual_histogram,
            plot_hyperparameter_importances,
            plot_slice_plots,
            plot_parallel_coordinates,
            plot_contour_grid,
            plot_captum_bar,
            plot_captum_beeswarm,
            plot_captum_heatmap,
            plot_captum_top_temporal,
        )
    except Exception:
        return
    train_df = detail["train_df"]
    test_df = detail["test_df"]
    y_train = pd.Series(detail["y_train"], index=train_df.index)
    y_test = pd.Series(detail["y_test"], index=test_df.index)
    y_train_pred = detail["y_train_pred"]
    y_test_pred = detail["y_test_pred"]
    study = detail["study"]
    trial_values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    plot_objective_curve(trial_values, title=f"Curva objetivo - {dataframe_name}", save_path=os.path.join(set_dir, f"{dataframe_name}_objective_curve.png"))
    loss_df = pd.read_csv(os.path.join(set_dir, f"{dataframe_name}_loss_curve.csv"))
    plot_loss_curve_from_df(loss_df, title=f"{dataframe_name} - Curva de pérdida final", save_path=os.path.join(set_dir, f"{dataframe_name}_loss_curve.png"))
    plot_train_vs_actual(y_train, y_train_pred, target_column, save_path=os.path.join(set_dir, f"{dataframe_name}_train_vs_actual.png"))
    plot_test_vs_actual(y_test, y_test_pred, target_column, save_path=os.path.join(set_dir, f"{dataframe_name}_test_vs_actual.png"))
    plot_full_temporal_results(y_train, y_test, y_train_pred, y_test_pred, target_column, save_path=os.path.join(set_dir, f"{dataframe_name}_full_temporal_results.png"))
    plot_real_vs_pred(y_test, y_test_pred, save_path=os.path.join(set_dir, f"{dataframe_name}_real_vs_pred.png"))
    plot_residual_histogram(y_test.values, y_test_pred, title=f"{dataframe_name} - Histograma residuos TEST", save_path=os.path.join(set_dir, f"{dataframe_name}_residual_histogram_test.png"))
    trials_path = os.path.join(set_dir, f"{dataframe_name}_optuna_trials.csv")
    imp_path = os.path.join(set_dir, f"{dataframe_name}_hyperparam_importances.csv")
    if os.path.exists(trials_path) and os.path.exists(imp_path):
        trials_df = pd.read_csv(trials_path)
        imp_df = pd.read_csv(imp_path)
        if not imp_df.empty:
            importances = dict(zip(imp_df["hiperparametro"], imp_df["importancia"]))
            param_names = list(importances.keys())
            plot_hyperparameter_importances(importances, title=f"Importancia de hiperparámetros - {dataframe_name}", save_path=os.path.join(set_dir, f"{dataframe_name}_hyperparam_importances.png"))
            plot_slice_plots(trials_df, param_names=param_names, target_col="value", title=f"Slice plots - {dataframe_name}", save_path=os.path.join(set_dir, f"{dataframe_name}_hyperparam_slice.png"))
            plot_parallel_coordinates(trials_df, param_names=param_names, target_col="value", top_n=20, title=f"Parallel coordinate - top trials - {dataframe_name}", save_path=os.path.join(set_dir, f"{dataframe_name}_hyperparam_parallel.png"))
            top_params = param_names[:4]
            if len(top_params) >= 2:
                plot_contour_grid(trials_df, param_names=top_params, target_col="value", title=f"Contour plots - {dataframe_name}", save_path=os.path.join(set_dir, f"{dataframe_name}_hyperparam_contour.png"), top_n=100)
    cap_mean = os.path.join(set_dir, f"{dataframe_name}_captum_mean_abs.csv")
    cap_attr = os.path.join(set_dir, f"{dataframe_name}_captum_attributions.csv")
    if os.path.exists(cap_mean) and os.path.exists(cap_attr):
        mean_abs_df = pd.read_csv(cap_mean)
        attributions_df = pd.read_csv(cap_attr, index_col=0)
        X_df = train_df.iloc[:, 1:].copy()
        plot_captum_bar(mean_abs_df, title=None, save_path=os.path.join(set_dir, f"{dataframe_name}_captum_bar.png"))
        plot_captum_beeswarm(attributions_df, X_df, mean_abs_df, title=None, save_path=os.path.join(set_dir, f"{dataframe_name}_captum_beeswarm.png"))
        plot_captum_heatmap(attributions_df, mean_abs_df, title=None, save_path=os.path.join(set_dir, f"{dataframe_name}_captum_heatmap.png"))
        plot_captum_top_temporal(attributions_df, mean_abs_df, title="Atribuciones temporales", save_path=os.path.join(set_dir, f"{dataframe_name}_captum_top_temporal.png"))


def process_sets_mlp_temporal(
    sets_dict=None,
    dataframes=None,
    search_space=None,
    n_trials=30,
    train_size=0.8,
    cv_splits=3,
    final_val_size=0.15,
    max_epochs=300,
    patience=20,
    scale_y=True,
    make_plots=True,
    output_dir=None,
    pruning=True,
    seed=42,
    experiment_label="mlp_experiment"
):
    if dataframes is None:
        dataframes = sets_dict
    if dataframes is None:
        raise ValueError("Debes pasar sets_dict=... o dataframes=...")
    if search_space is None:
        raise ValueError("Debes pasar search_space=...")

    if output_dir is None:
        return _process_sets_mlp_temporal_core(
            sets_dict=dataframes,
            search_space=search_space,
            n_trials=n_trials,
            train_size=train_size,
            cv_splits=cv_splits,
            final_val_size=final_val_size,
            max_epochs=max_epochs,
            patience=patience,
            scale_y=scale_y,
            make_plots=make_plots,
            pruning=pruning,
            seed=seed
        )

    import inspect, sys, time, traceback
    try:
        from reportes_html_mlp import TeeOutput, save_log_txt, save_log_html, generate_html_report
    except Exception:
        TeeOutput = save_log_txt = save_log_html = generate_html_report = None

    local_vars = inspect.currentframe().f_back.f_locals.items()
    dict_name = _mlp_get_variable_name(dataframes, local_vars)
    output_dir_run = _mlp_build_timestamped_output_dir(output_dir)
    os.makedirs(output_dir_run, exist_ok=True)

    original_stdout = sys.stdout
    tee_stdout = TeeOutput(original_stdout) if TeeOutput is not None else None
    if tee_stdout is not None:
        sys.stdout = tee_stdout

    start_time = time.perf_counter()
    train_metrics_df = test_metrics_df = params_df = metadata_df = comparison_df = global_nse_stats_df = None
    results_for_html = []

    try:
        print("")
        print("==================================================================================================")
        print("INICIO DEL PROCESAMIENTO MLP")
        print("==================================================================================================")
        print(f"Diccionario procesado : {dict_name}")
        print(f"Experiment label      : {experiment_label}")
        print(f"Cantidad de trials    : {n_trials}")
        print(f"Train size            : {train_size}")
        print(f"CV splits             : {cv_splits}")
        print(f"Make plots reporte    : {make_plots}")
        print(f"Final val size        : {final_val_size}")
        print(f"Max epochs            : {max_epochs}")
        print(f"Patience              : {patience}")
        print(f"Scale y               : {scale_y}")
        print(f"Pruning               : {pruning}")
        print(f"Seed                  : {seed}")
        print(f"Carpeta de salida     : {output_dir_run}")
        print("")

        results_df, details = _process_sets_mlp_temporal_core(
            sets_dict=dataframes,
            search_space=search_space,
            n_trials=n_trials,
            train_size=train_size,
            cv_splits=cv_splits,
            final_val_size=final_val_size,
            max_epochs=max_epochs,
            patience=patience,
            scale_y=scale_y,
            make_plots=False,
            pruning=pruning,
            seed=seed
        )

        elapsed_total = time.perf_counter() - start_time
        run_timing_df = pd.DataFrame([{
            "execution_scope": "corrida_completa",
            "run_execution_seconds": round_sig(elapsed_total, 3),
            "run_execution_time": _mlp_format_elapsed_time(elapsed_total),
        }])
        run_timing_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_run_timing.csv"), index=False)

        train_metrics_df, test_metrics_df, params_df, metadata_df, comparison_df, global_nse_stats_df = _mlp_build_summary_tables(
            results_df=results_df,
            details=details,
            experiment_label=experiment_label,
            max_epochs=max_epochs,
            patience=patience,
            final_val_size=final_val_size,
            scale_y=scale_y,
            pruning=pruning
        )

        train_metrics_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_train_metrics.csv"), index=False)
        test_metrics_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_test_metrics.csv"), index=False)
        params_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_params.csv"), index=False)
        metadata_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_metadata.csv"), index=False)
        comparison_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_comparison_train_test_cv.csv"), index=False)
        global_nse_stats_df.to_csv(os.path.join(output_dir_run, f"{dict_name}_global_nse_stats.csv"), index=False)

        run_config_base = {
            "dict_name": dict_name,
            "experiment_label": experiment_label,
            "search_space_type": "custom_mlp",
            "use_early_stopping": True,
            "n_trials": n_trials,
            "train_size": train_size,
            "cv_splits": cv_splits,
            "final_val_size": final_val_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "scale_y": scale_y,
            "make_plots": make_plots,
            "pruning": pruning,
            "seed": seed,
            "output_dir_run": output_dir_run,
            "run_execution_seconds": round_sig(elapsed_total, 3),
            "run_execution_time": _mlp_format_elapsed_time(elapsed_total),
        }
        run_config_rows = [dict(run_config_base)]
        for p, r in _mlp_search_space_to_text(search_space).items():
            row_cfg = dict(run_config_base)
            row_cfg["search_param"] = p
            row_cfg["search_range"] = r
            run_config_rows.append(row_cfg)
        pd.DataFrame(run_config_rows).to_csv(os.path.join(output_dir_run, f"{dict_name}_run_config.csv"), index=False)

        for name, detail in details.items():
            set_dir = os.path.join(output_dir_run, name)
            os.makedirs(set_dir, exist_ok=True)
            _mlp_export_prediction_csvs(set_dir, name, detail["train_df"], detail["test_df"], detail["y_train_pred"], detail["y_test_pred"])
            _mlp_export_loss_curve_csv(set_dir, name, detail["training_result"])
            trials_df = _mlp_build_trials_dataframe(detail["study"])
            trials_df.to_csv(os.path.join(set_dir, f"{name}_optuna_trials.csv"), index=False)

            fold_scores_df = _mlp_build_fold_scores_dataframe(detail)
            fold_scores_df.to_csv(os.path.join(set_dir, f"{name}_optuna_fold_scores.csv"), index=False)

            param_importances_df, param_importances = _mlp_export_param_importances_csv(set_dir, name, detail["study"])
            detail["param_importances_df"] = param_importances_df
            detail["param_importances"] = param_importances

            attrs, mean_abs, top_df, cap_err = _mlp_compute_captum_attributions(detail)
            detail["captum_attributions_df"] = attrs if attrs is not None else pd.DataFrame()
            detail["captum_mean_abs_df"] = mean_abs if mean_abs is not None else pd.DataFrame()
            detail["captum_top_features_by_date_df"] = top_df if top_df is not None else pd.DataFrame()
            if attrs is not None:
                attrs.to_csv(os.path.join(set_dir, f"{name}_captum_attributions.csv"))
                mean_abs.to_csv(os.path.join(set_dir, f"{name}_captum_mean_abs.csv"), index=False)
                top_df.to_csv(os.path.join(set_dir, f"{name}_captum_top_features_by_date.csv"), index=False)
            elif cap_err is not None:
                with open(os.path.join(set_dir, f"{name}_captum_error.txt"), "w", encoding="utf-8") as f:
                    f.write(str(cap_err))
            if make_plots:
                _mlp_save_basic_plots(set_dir, name, detail, detail["train_df"].columns[0])

        integrated_diagnostics_df = _mlp_build_integrated_diagnostics_dataframe(
            details=details,
            metadata_df=metadata_df,
            search_space=search_space,
            experiment_label=experiment_label,
            n_trials=n_trials,
            train_size=train_size,
            cv_splits=cv_splits,
            final_val_size=final_val_size,
            max_epochs=max_epochs,
            patience=patience,
            scale_y=scale_y,
            pruning=pruning,
            seed=seed,
            make_plots=make_plots,
        )
        integrated_diagnostics_df.to_csv(
            os.path.join(output_dir_run, f"{dict_name}_integrated_diagnostics.csv"),
            index=False,
        )

        results_for_html = _mlp_generate_results_for_html(details, metadata_df, search_space, experiment_label)

        print("--------------------------------------------------------------------------------------------------")
        print(f"\nCarpeta de resultados: {output_dir_run}")
        print(f"Tiempo total de corrida: {_mlp_format_elapsed_time(elapsed_total)}")
        print("\nResumen de métricas de train:\n")
        print(train_metrics_df)
        print("\nResumen de métricas de test:\n")
        print(test_metrics_df)
        print("\nResumen de mejores parámetros:\n")
        print(params_df)
        print("\nResumen de metadatos:\n")
        print(metadata_df)
        print("\nResumen comparativo final (train / test / CV):\n")
        print(comparison_df)
        print("\nEstadísticas globales de performance NSE:\n")
        print(global_nse_stats_df)

    finally:
        if tee_stdout is not None:
            sys.stdout = original_stdout
            log_text = tee_stdout.get_value()
            if save_log_txt is not None:
                save_log_txt(log_text, output_dir_run, filename="log_salida.txt")
            if save_log_html is not None:
                save_log_html(log_text, output_dir_run, title="Log de salida del experimento MLP", filename="log_salida.html")
        try:
            if generate_html_report is not None and train_metrics_df is not None and len(results_for_html) > 0:
                generate_html_report(
                    run_output_dir=output_dir_run,
                    dict_name=dict_name,
                    train_metrics_df=train_metrics_df,
                    test_metrics_df=test_metrics_df,
                    params_df=params_df,
                    metadata_df=metadata_df,
                    comparison_df=comparison_df,
                    global_nse_stats_df=global_nse_stats_df,
                    results=results_for_html,
                    config_mode=experiment_label,
                    n_trials=n_trials,
                    html_filename="reporte_resultados.html"
                )
        except Exception:
            with open(os.path.join(output_dir_run, "error_reporte_html.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        try:
            _mlp_build_artifacts_inventory(output_dir_run)
        except Exception:
            pass

    return train_metrics_df, test_metrics_df, params_df, metadata_df, comparison_df
