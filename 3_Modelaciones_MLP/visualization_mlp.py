import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

plt.style.use("ggplot")
plt.rcParams["axes.facecolor"] = "#E5E5E5"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["grid.color"] = "white"
plt.rcParams["grid.alpha"] = 1.0
plt.rcParams["grid.linewidth"] = 1.0

SOFT_BLUE = "#EAF3FF"


def _save_show_close(fig, save_path=None):
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _apply_soft_grid(ax, axis="both"):
    ax.set_facecolor(SOFT_BLUE)
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color="white", linewidth=1.0)


def plot_objective_curve(nse_scores, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nse_scores)
    ax.set_xlabel("Trial")
    ax.set_ylabel("NSE promedio CV")
    ax.set_title(title if title else "Curva de la función objetivo")
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_loss_curve_from_df(loss_df, title="Curva de pérdida", save_path=None):
    if loss_df is None or loss_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_df["epoch"], loss_df["train_loss"], label="Train loss", linewidth=2)
    ax.plot(loss_df["epoch"], loss_df["val_loss"], label="Val interna loss", linewidth=2)

    if "is_best_epoch" in loss_df.columns and loss_df["is_best_epoch"].any():
        best_epoch = loss_df.loc[loss_df["is_best_epoch"], "epoch"].iloc[0]
        ax.axvline(best_epoch, linestyle="--", color="black", label="Best epoch")

    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE loss")
    ax.legend()
    _apply_soft_grid(ax)
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_train_vs_actual(y_train, y_pred_train, target_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_train.index, y_train.values, label="Real", linewidth=2)
    ax.plot(y_train.index, y_pred_train, label="Predicción train", linewidth=2)
    ax.set_title(f"Train - {target_name}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(target_name)
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_test_vs_actual(y_test, y_pred_test, target_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.index, y_test.values, label="Real", linewidth=2)
    ax.plot(y_test.index, y_pred_test, label="Predicción test", linewidth=2)
    ax.set_title(f"Test - {target_name}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(target_name)
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_full_temporal_results(y_train, y_test, y_pred_train, y_pred_test, target_name, save_path=None, point_size=13, alpha_points=0.70):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)

    x_all = np.concatenate([y_train.index.values, y_test.index.values])
    y_all = np.concatenate([y_train.values, y_test.values])

    ax.plot(x_all, y_all, color="black", linewidth=1.2, alpha=0.9)
    ax.scatter(x_all, y_all, s=point_size, color="black", alpha=alpha_points, label="Datos reales")

    ax.plot(y_train.index, y_pred_train, color="lightskyblue", linewidth=1.2, alpha=0.9)
    ax.scatter(y_train.index, y_pred_train, s=point_size, color="lightskyblue", alpha=alpha_points, label="Predicción train")

    ax.plot(y_test.index, y_pred_test, color="red", linewidth=1.2, alpha=0.9)
    ax.scatter(y_test.index, y_pred_test, s=point_size, color="red", alpha=alpha_points, label="Predicción test")

    ax.set_title(f"Resultados temporales completos - {target_name}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(target_name)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_real_vs_pred(y_test, y_pred_test, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, label="Observaciones")

    lims = [min(np.min(y_test), np.min(y_pred_test)), max(np.max(y_test), np.max(y_pred_test))]
    ax.plot(lims, lims, "--", color="black", label="1:1")

    if len(y_test) > 1:
        coef = np.polyfit(y_test, y_pred_test, 1)
        x_fit = np.linspace(lims[0], lims[1], 200)
        y_fit = coef[0] * x_fit + coef[1]
        ax.plot(x_fit, y_fit, color="red", linewidth=2, label="Recta de ajuste")

    ax.set_xlabel("Valor real")
    ax.set_ylabel("Predicción")
    ax.set_title("Real vs predicho")
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_residual_histogram(y_true, y_pred, title="Histograma de residuos", save_path=None):
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=15, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Residuo (pred - real)")
    ax.set_ylabel("Frecuencia")
    _apply_soft_grid(ax)
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_captum_bar(mean_abs_df, title=None, save_path=None, top_n=20):
    if mean_abs_df is None or mean_abs_df.empty:
        return
    df = mean_abs_df.sort_values("mean_abs_attribution", ascending=False).head(top_n).copy()
    df = df.sort_values("mean_abs_attribution", ascending=True)
    fig_height = max(4, 0.35 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(df["feature"], df["mean_abs_attribution"])
    ax.set_xlabel("mean(|atribución|)")
    if title:
        ax.set_title(title)
    _apply_soft_grid(ax, axis="x")
    plt.tight_layout()
    _save_show_close(fig, save_path)


def _relative_feature_values(values):
    """Normaliza una variable a [0, 1] únicamente para el color del beeswarm."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full(values.shape, 0.5, dtype=float)
    if not finite.any():
        return out

    vmin = np.nanmin(values[finite])
    vmax = np.nanmax(values[finite])
    if vmax > vmin:
        out[finite] = (values[finite] - vmin) / (vmax - vmin)
    return out


def plot_captum_beeswarm(attributions_df, X_df, mean_abs_df, title=None, save_path=None, max_display=20):
    if attributions_df is None or attributions_df.empty or X_df is None or X_df.empty:
        return

    top_features = mean_abs_df.sort_values("mean_abs_attribution", ascending=False)["feature"].head(max_display).tolist()
    top_features = [f for f in top_features if f in attributions_df.columns and f in X_df.columns]
    if not top_features:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_features) + 1.5)))

    y_positions = np.arange(len(top_features))
    rng = np.random.default_rng(42)
    color_norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("cool")

    for i, feature in enumerate(reversed(top_features)):
        vals = attributions_df[feature].values.astype(float)
        relative_colors = _relative_feature_values(X_df[feature].values.astype(float))
        jitter = rng.normal(0, 0.07, size=len(vals))
        ax.scatter(
            vals,
            np.full_like(vals, i, dtype=float) + jitter,
            c=relative_colors,
            cmap=cmap,
            norm=color_norm,
            s=18,
            alpha=0.75,
            edgecolors="none",
        )

    ax.axvline(0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(reversed(top_features)))
    ax.set_xlabel("Atribución")
    if title:
        ax.set_title(title)
    _apply_soft_grid(ax, axis="x")

    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", rotation=90)

    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_captum_heatmap(attributions_df, mean_abs_df, title=None, save_path=None, max_display=20):
    if attributions_df is None or attributions_df.empty:
        return

    top_features = mean_abs_df.sort_values("mean_abs_attribution", ascending=False)["feature"].head(max_display).tolist()
    top_features = [f for f in top_features if f in attributions_df.columns]
    if not top_features:
        return

    data = attributions_df[top_features].T.values
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(top_features) + 2)))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xlabel("Observaciones temporales")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Atribución")
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_captum_top_temporal(attributions_df, mean_abs_df, title="Atribuciones temporales", save_path=None, top_n=5):
    if attributions_df is None or attributions_df.empty:
        return
    top_features = mean_abs_df.sort_values("mean_abs_attribution", ascending=False)["feature"].head(top_n).tolist()
    top_features = [f for f in top_features if f in attributions_df.columns]
    if not top_features:
        return

    date_index = pd.to_datetime(attributions_df.index, errors="coerce")
    use_dates = not pd.isna(date_index).all()
    x_values = date_index if use_dates else attributions_df.index

    fig, ax = plt.subplots(figsize=(12, 5))
    for feature in top_features:
        ax.plot(x_values, attributions_df[feature].values, linewidth=1.4, label=feature)

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.7)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Atribución")

    if use_dates:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelrotation=0)

    ax.legend(fontsize=8)
    _apply_soft_grid(ax)
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_hyperparameter_importances(importances, title=None, save_path=None):
    if not importances:
        return
    labels = list(importances.keys())
    values = list(importances.values())
    fig_height = max(4, 0.6 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia relativa")
    ax.set_title(title if title else "Importancia de hiperparámetros")
    _apply_soft_grid(ax)
    plt.tight_layout()
    _save_show_close(fig, save_path)


def _encode_categorical_for_plot(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), None
    codes, uniques = pd.factorize(series.astype(str))
    return pd.Series(codes, index=series.index, dtype=float), list(uniques)


def plot_slice_plots(trials_df, param_names, target_col="value", title=None, save_path=None):
    if trials_df is None or trials_df.empty or not param_names:
        return
    param_names = [p for p in param_names if p in trials_df.columns]
    if not param_names:
        return

    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = int(math.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols + 0.9, 4.8 * n_rows), squeeze=False)
    axes = axes.ravel()
    color_col = "trial_number" if "trial_number" in trials_df.columns else None
    cmap = plt.cm.viridis
    norm = Normalize(vmin=trials_df[color_col].min(), vmax=trials_df[color_col].max()) if color_col else None
    last_scatter = None

    for ax, param in zip(axes, param_names):
        cols = [param, target_col] + ([color_col] if color_col else [])
        subset = trials_df[cols].dropna().copy()
        if subset.empty:
            ax.set_title(param)
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            _apply_soft_grid(ax)
            continue

        x_vals, labels = _encode_categorical_for_plot(subset[param])
        colors = subset[color_col] if color_col else np.arange(len(subset))
        scatter = ax.scatter(x_vals, subset[target_col], c=colors, cmap=cmap, norm=norm, alpha=0.75, s=22, edgecolors="none")
        last_scatter = scatter
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.set_ylabel("CV NSE")
        if labels is not None and len(labels) <= 12:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
        _apply_soft_grid(ax)

    for ax in axes[n_params:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, y=0.995, fontsize=14)
    if last_scatter is not None and color_col is not None:
        fig.tight_layout(rect=[0, 0, 0.90, 0.97 if title else 0.98])
        cax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
        cbar = fig.colorbar(last_scatter, cax=cax)
        cbar.set_label("Trial")
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.97 if title else 0.98])
    _save_show_close(fig, save_path)


def plot_parallel_coordinates(trials_df, param_names, target_col="value", top_n=20, title=None, save_path=None):
    if trials_df is None or trials_df.empty or not param_names:
        return
    available_params = [p for p in param_names if p in trials_df.columns]
    cols = available_params + [target_col]
    df = trials_df[cols].dropna().copy()
    if df.empty:
        return
    df = df.sort_values(target_col, ascending=False).head(min(top_n, len(df))).reset_index(drop=True)

    encoded = pd.DataFrame(index=df.index)
    for col in cols:
        encoded[col], _ = _encode_categorical_for_plot(df[col])

    scaled = pd.DataFrame(index=df.index)
    for col in cols:
        col_min = encoded[col].min()
        col_max = encoded[col].max()
        scaled[col] = 0.5 if col_max - col_min == 0 else (encoded[col] - col_min) / (col_max - col_min)

    x_positions = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(cols)), 6.8))
    target_values = df[target_col].values.astype(float)
    norm_vals = np.ones_like(target_values) * 0.5 if np.max(target_values) == np.min(target_values) else (target_values - np.min(target_values)) / (np.max(target_values) - np.min(target_values))
    cmap = plt.cm.viridis

    for idx in range(len(df)):
        y_vals = scaled.iloc[idx].values
        points = np.array([x_positions, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=[cmap(norm_vals[idx])], linewidths=1.2, alpha=0.45)
        ax.add_collection(lc)

    for i, col in enumerate(cols):
        ax.axvline(i, color="white", linewidth=1.2, zorder=0)
        ax.text(i, -0.08, f"{df[col].iloc[0] if df[col].nunique()==1 else df[col].min()}", ha="center", va="top", fontsize=8)
        ax.text(i, 1.03, f"{df[col].iloc[0] if df[col].nunique()==1 else df[col].max()}", ha="center", va="bottom", fontsize=8)

    ax.set_xlim(x_positions[0], x_positions[-1])
    ax.set_ylim(-0.12, 1.08)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title(title if title else f"Parallel coordinate - top {min(top_n, len(df))} trials")
    ax.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=df[target_col].min(), vmax=df[target_col].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("CV NSE")
    plt.tight_layout()
    _save_show_close(fig, save_path)


def _pairwise(iterable):
    items = list(iterable)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def _compute_plot_bounds(values, margin_ratio=0.06):
    vmin = np.min(values)
    vmax = np.max(values)
    if vmax == vmin:
        margin = 1.0 if vmin == 0 else abs(vmin) * margin_ratio
    else:
        margin = (vmax - vmin) * margin_ratio
    return vmin - margin, vmax + margin


def _smooth_surface_from_points(x, y, z, grid_n=220):
    x_min, x_max = _compute_plot_bounds(x)
    y_min, y_max = _compute_plot_bounds(y)
    xi = np.linspace(x_min, x_max, grid_n)
    yi = np.linspace(y_min, y_max, grid_n)
    Xi, Yi = np.meshgrid(xi, yi)
    points = np.column_stack([x, y])
    Zi_cubic = None
    if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3 and len(points) >= 4:
        try:
            Zi_cubic = griddata(points, z, (Xi, Yi), method="cubic")
        except Exception:
            Zi_cubic = None
    if Zi_cubic is None or np.all(np.isnan(Zi_cubic)):
        Zi_cubic = griddata(points, z, (Xi, Yi), method="linear")
    Zi_nearest = griddata(points, z, (Xi, Yi), method="nearest")
    Zi_full = np.where(np.isnan(Zi_cubic), Zi_nearest, Zi_cubic)
    return Xi, Yi, Zi_full, (x_min, x_max, y_min, y_max)


def _contour_or_scatter(ax, subset, x_col, y_col, z_col):
    x_raw, _ = _encode_categorical_for_plot(subset[x_col])
    y_raw, _ = _encode_categorical_for_plot(subset[y_col])
    x = x_raw.values.astype(float)
    y = y_raw.values.astype(float)
    z = subset[z_col].values.astype(float)
    enough_points = len(subset) >= 4 and len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2
    if enough_points:
        try:
            Xi, Yi, Zi, bounds = _smooth_surface_from_points(x, y, z, grid_n=220)
            mesh = ax.pcolormesh(Xi, Yi, Zi, shading="auto", cmap="viridis")
            ax.contour(Xi, Yi, Zi, levels=8, colors="white", linewidths=0.6, alpha=0.75)
            ax.scatter(x, y, color="black", edgecolor="black", linewidth=0.3, s=16, alpha=0.75)
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            return mesh
        except Exception:
            pass
    scatter = ax.scatter(x, y, color="black", edgecolor="black", s=28, alpha=0.9)
    x_min, x_max = _compute_plot_bounds(x)
    y_min, y_max = _compute_plot_bounds(y)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return scatter


def plot_contour_grid(trials_df, param_names, target_col="value", title=None, save_path=None, top_n=100):
    if trials_df is None or trials_df.empty:
        return
    param_names = [p for p in param_names if p in trials_df.columns]
    if top_n is not None and len(trials_df) > top_n:
        trials_df = trials_df.sort_values(target_col, ascending=False).head(top_n).copy()
    pairs = list(_pairwise(param_names))
    if not pairs:
        return
    n_plots = len(pairs)
    n_cols = min(2, n_plots)
    n_rows = int(math.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.1 * n_rows), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    mappable = None
    for ax, (x_col, y_col) in zip(axes_flat, pairs):
        subset = trials_df[[x_col, y_col, target_col]].dropna()
        if subset.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{x_col} vs {y_col}")
            continue
        mappable = _contour_or_scatter(ax, subset, x_col, y_col, target_col)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.grid(False)
    for ax in axes_flat[n_plots:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.01, fontsize=14)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes_flat[:n_plots].tolist(), shrink=0.94, pad=0.02, aspect=30, location="right")
        cbar.set_label("CV NSE")
    _save_show_close(fig, save_path)
