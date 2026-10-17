import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata

# Estilo general del gráfico
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
    # plt.show()
    plt.close(fig)


def _apply_soft_grid(ax, axis="both"):
    ax.set_facecolor(SOFT_BLUE)
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color="white", linewidth=1.0)


def _coerce_shap_explanation(shap_values, X_data, feature_names=None, base_values=None):
    if isinstance(shap_values, shap.Explanation):
        return shap_values

    values = np.asarray(shap_values)
    data = X_data.values if isinstance(X_data, pd.DataFrame) else np.asarray(X_data)
    if feature_names is None:
        if isinstance(X_data, pd.DataFrame):
            feature_names = X_data.columns.tolist()
        else:
            feature_names = None

    if base_values is None:
        base_values = np.zeros(values.shape[0]) if values.ndim > 1 else 0.0

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names
    )


def _annotate_barh_values(ax, fmt="{:.3f}", pad_ratio=0.02):
    patches = [patch for patch in ax.patches if hasattr(patch, "get_width")]
    if not patches:
        return
    widths = [patch.get_width() for patch in patches]
    max_width = max(widths) if widths else 0.0
    pad = max_width * pad_ratio if max_width > 0 else 0.01
    for patch in patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(width + pad, y, fmt.format(width), va="center", ha="left", fontsize=9, color="black")


def _truncate_colormap(cmap, minval=0.0, maxval=0.82, n=256):
    return LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}",
        cmap(np.linspace(minval, maxval, n))
    )


def _focused_band_bounds(y_true, y_pred, system_type, pad_ratio=0.12, min_pad=0.8):
    values = np.concatenate([
        pd.Series(y_true).dropna().values.astype(float),
        pd.Series(y_pred).dropna().values.astype(float)
    ])
    if len(values) == 0:
        return _interval_bounds_for_plot(system_type)

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    span = max(data_max - data_min, min_pad)
    pad = max(span * pad_ratio, min_pad)

    y_min = data_min - pad
    y_max = data_max + pad

    intervals = TROPHIC_INTERVALS[system_type]
    bounds = []
    for lower, upper, label in intervals:
        band_lower = y_min if np.isneginf(lower) else lower
        band_upper = y_max if np.isposinf(upper) else upper
        clipped_lower = max(band_lower, y_min)
        clipped_upper = min(band_upper, y_max)
        if clipped_upper > clipped_lower:
            bounds.append((clipped_lower, clipped_upper, label))

    return y_min, y_max, bounds


def plot_objective_curve(nse_scores, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nse_scores)
    ax.set_xlabel('Trial')
    ax.set_ylabel('NSE promedio CV')
    ax.set_title(title if title else 'Curva de la función objetivo')
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_tree_evolution_curve(
    tree_evolution_df,
    title=None,
    save_path=None,
    best_iteration=None,
    selected_n_trees=None,
    selected_label=None
):
    if tree_evolution_df is None or tree_evolution_df.empty:
        return

    required_cols = {"n_trees", "nse_train_internal", "nse_val_internal"}
    if not required_cols.issubset(set(tree_evolution_df.columns)):
        return

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    ax.plot(
        tree_evolution_df["n_trees"],
        tree_evolution_df["nse_train_internal"],
        marker="o",
        markersize=2.5,
        linewidth=1.5,
        label="Train interno"
    )
    ax.plot(
        tree_evolution_df["n_trees"],
        tree_evolution_df["nse_val_internal"],
        marker="o",
        markersize=2.5,
        linewidth=1.5,
        label="Validación interna"
    )

    if best_iteration is not None:
        try:
            best_tree = int(best_iteration) + 1
            ax.axvline(
                best_tree,
                linestyle="--",
                linewidth=1.3,
                color="black",
                alpha=0.75,
                label=f"Best iteration + 1 = {best_tree}"
            )
        except Exception:
            pass

    if selected_n_trees is not None:
        try:
            selected_tree = int(selected_n_trees)
            label_text = selected_label if selected_label else f"Árbol reportado = {selected_tree}"
            ax.axvline(
                selected_tree,
                linestyle=":",
                linewidth=1.8,
                color="dimgray",
                alpha=0.90,
                label=label_text
            )
        except Exception:
            pass

    ax.set_xlabel("Número acumulado de árboles")
    ax.set_ylabel("NSE")
    ax.set_title(title if title else "Evolución del NSE según número de árboles")
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)


def plot_tree_evolution_rmse_curve(
    tree_evolution_df,
    title=None,
    save_path=None,
    best_iteration=None,
    selected_n_trees=None,
    selected_label=None
):
    if tree_evolution_df is None or tree_evolution_df.empty:
        return

    required_cols = {"n_trees", "rmse_train_internal", "rmse_val_internal"}
    if not required_cols.issubset(set(tree_evolution_df.columns)):
        return

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    ax.plot(
        tree_evolution_df["n_trees"],
        tree_evolution_df["rmse_train_internal"],
        marker="o",
        markersize=2.5,
        linewidth=1.5,
        label="Train interno"
    )
    ax.plot(
        tree_evolution_df["n_trees"],
        tree_evolution_df["rmse_val_internal"],
        marker="o",
        markersize=2.5,
        linewidth=1.5,
        label="Validación interna"
    )

    if best_iteration is not None:
        try:
            best_tree = int(best_iteration) + 1
            ax.axvline(
                best_tree,
                linestyle="--",
                linewidth=1.3,
                color="black",
                alpha=0.75,
                label=f"Best iteration + 1 = {best_tree}"
            )
        except Exception:
            pass

    if selected_n_trees is not None:
        try:
            selected_tree = int(selected_n_trees)
            label_text = selected_label if selected_label else f"Árbol reportado = {selected_tree}"
            ax.axvline(
                selected_tree,
                linestyle=":",
                linewidth=1.8,
                color="dimgray",
                alpha=0.90,
                label=label_text
            )
        except Exception:
            pass

    ax.set_xlabel("Número acumulado de árboles")
    ax.set_ylabel("RMSE")
    ax.set_title(title if title else "Evolución del RMSE según número de árboles")
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)



def plot_train_vs_actual(y_train, y_pred_train, target_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_train.index, y_train.values, label='Real', linewidth=2)
    ax.plot(y_train.index, y_pred_train, label='Predicción train', linewidth=2)
    ax.set_title(f'Train - {target_name}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel(target_name)
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)



def plot_test_vs_actual(y_test, y_pred_test, target_name, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.index, y_test.values, label='Real', linewidth=2)
    ax.plot(y_test.index, y_pred_test, label='Predicción test', linewidth=2)
    ax.set_title(f'Test - {target_name}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel(target_name)
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)



def plot_real_vs_pred(y_test, y_pred_test, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.6, label='Observaciones')

    lims = [
        min(np.min(y_test), np.min(y_pred_test)),
        max(np.max(y_test), np.max(y_pred_test))
    ]

    ax.plot(lims, lims, '--', color='black', label='1:1')

    if len(y_test) > 1:
        coef = np.polyfit(y_test, y_pred_test, 1)
        x_fit = np.linspace(lims[0], lims[1], 200)
        y_fit = coef[0] * x_fit + coef[1]
        ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Recta de ajuste')

    ax.set_xlabel('Valor real')
    ax.set_ylabel('Predicción')
    ax.set_title('Real vs predicho')
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)



def plot_full_temporal_results(
    y_train,
    y_test,
    y_pred_train,
    y_pred_test,
    target_name,
    save_path=None,
    point_size=13,
    alpha_points=0.70
):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#E5E5E5")
    fig.patch.set_facecolor("white")
    ax.grid(True, color="white", linewidth=1.0)

    x_all = np.concatenate([y_train.index.values, y_test.index.values])
    y_all = np.concatenate([y_train.values, y_test.values])

    ax.plot(x_all, y_all, color='black', linewidth=1.2, alpha=0.9)
    ax.scatter(x_all, y_all, s=point_size, color='black', alpha=alpha_points, label='Datos reales')

    ax.plot(y_train.index, y_pred_train, color='lightskyblue', linewidth=1.2, alpha=0.9)
    ax.scatter(y_train.index, y_pred_train, s=point_size, color='lightskyblue', alpha=alpha_points, label='Predicción train')

    ax.plot(y_test.index, y_pred_test, color='red', linewidth=1.2, alpha=0.9)
    ax.scatter(y_test.index, y_pred_test, s=point_size, color='red', alpha=alpha_points, label='Predicción test')

    ax.set_title(f'Resultados temporales completos - {target_name}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel(target_name)
    ax.legend()
    plt.tight_layout()
    _save_show_close(fig, save_path)



def shap_summary_plots(shap_values, X_data, feature_names, out_prefix=None, max_display=20, decision_top_n=20, base_values=None):
    shap_exp = _coerce_shap_explanation(shap_values, X_data, feature_names=feature_names, base_values=base_values)

    plt.figure(figsize=(6, 3.6))
    shap.plots.beeswarm(shap_exp, max_display=max_display, color=plt.get_cmap("cool"), show=False)
    ax = plt.gca()
    ax.set_xlabel('SHAP value')
    plt.tight_layout()
    if out_prefix is not None:
        plt.savefig(f"{out_prefix}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    plt.figure(figsize=(6, 3.6))
    shap.plots.bar(shap_exp, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_xlabel('mean(|SHAP| value)')
    plt.tight_layout()
    if out_prefix is not None:
        plt.savefig(f"{out_prefix}_shap_bar.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 9))
    shap.plots.heatmap(shap_exp, max_display=max_display, instance_order=np.arange(shap_exp.values.shape[0]), show=False)
    fig = plt.gcf()
    if fig.axes:
        main_ax = fig.axes[0]
        main_ax.tick_params(axis="y", labelsize=8)
    if len(fig.axes) >= 2:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel("SHAP value")
        cbar_ax.tick_params(labelsize=8)
    plt.tight_layout()
    if out_prefix is not None:
        plt.savefig(f"{out_prefix}_shap_heatmap.png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    n_obs = min(decision_top_n, shap_exp.values.shape[0])
    if n_obs > 0:
        values_2d = np.asarray(shap_exp.values)
        row_strength = np.abs(values_2d).sum(axis=1)
        selected_idx = np.argsort(row_strength)[-n_obs:]
        selected_idx = selected_idx[np.argsort(row_strength[selected_idx])[::-1]]
        expected_value = shap_exp.base_values
        if isinstance(expected_value, np.ndarray):
            expected_value = np.ravel(expected_value)[0]

        plt.figure(figsize=(11, 7))
        shap.decision_plot(
            expected_value,
            values_2d[selected_idx],
            X_data.iloc[selected_idx] if isinstance(X_data, pd.DataFrame) else np.asarray(X_data)[selected_idx],
            feature_display_range=slice(None, -max_display - 1, -1),
            show=False
        )
        plt.tight_layout()
        if out_prefix is not None:
            plt.savefig(f"{out_prefix}_shap_decision.png", dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()


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
    _apply_soft_grid(ax, axis="both")
    plt.tight_layout()
    _save_show_close(fig, save_path)



def plot_slice_plots(trials_df, param_names, target_col="value", title=None, save_path=None):
    if trials_df is None or trials_df.empty or not param_names:
        return

    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = int(math.ceil(n_params / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols + 0.9, 4.8 * n_rows),
        squeeze=False
    )
    axes = axes.ravel()

    color_col = "trial_number" if "trial_number" in trials_df.columns else None
    cmap = plt.cm.viridis
    norm = None
    if color_col is not None and not trials_df[color_col].dropna().empty:
        norm = Normalize(vmin=trials_df[color_col].min(), vmax=trials_df[color_col].max())

    last_scatter = None

    for ax, param in zip(axes, param_names):
        cols = [param, target_col] + ([color_col] if color_col is not None else [])
        subset = trials_df[cols].dropna()
        if subset.empty:
            ax.set_title(param)
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            _apply_soft_grid(ax)
            continue

        colors = subset[color_col] if color_col is not None else np.arange(len(subset))
        scatter = ax.scatter(
            subset[param],
            subset[target_col],
            c=colors,
            cmap=cmap,
            norm=norm,
            alpha=0.75,
            s=22,
            edgecolors="none"
        )
        last_scatter = scatter
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.set_ylabel("CV NSE")
        _apply_soft_grid(ax)

    for ax in axes[n_params:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, y=0.995, fontsize=14)

    if last_scatter is not None and color_col is not None:
        right_margin = 0.90
        top_margin = 0.97 if title else 0.98
        fig.tight_layout(rect=[0, 0, right_margin, top_margin])
        cax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
        cbar = fig.colorbar(last_scatter, cax=cax)
        cbar.set_label("Trial")
    else:
        top_margin = 0.97 if title else 0.98
        fig.tight_layout(rect=[0, 0, 1, top_margin])

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

    scaled = pd.DataFrame(index=df.index)
    for col in cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.isna(col_min) or pd.isna(col_max):
            scaled[col] = 0.5
        elif col_max - col_min == 0:
            scaled[col] = 0.5
        else:
            scaled[col] = (df[col] - col_min) / (col_max - col_min)

    x_positions = np.arange(len(cols))

    fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(cols)), 6.8))

    target_values = df[target_col].values
    if np.max(target_values) == np.min(target_values):
        norm_vals = np.ones_like(target_values) * 0.5
    else:
        norm_vals = (target_values - np.min(target_values)) / (np.max(target_values) - np.min(target_values))

    cmap = plt.cm.viridis

    for idx in range(len(df)):
        y_vals = scaled.iloc[idx].values
        points = np.array([x_positions, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=[cmap(norm_vals[idx])], linewidths=1.2, alpha=0.45)
        ax.add_collection(lc)

    for i, col in enumerate(cols):
        ax.axvline(i, color="white", linewidth=1.2, zorder=0)
        actual_min = df[col].min()
        actual_max = df[col].max()
        ax.text(i, -0.08, f"{actual_min:.3g}", ha="center", va="top", fontsize=9)
        ax.text(i, 1.03, f"{actual_max:.3g}", ha="center", va="bottom", fontsize=9)

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
    x = subset[x_col].values.astype(float)
    y = subset[y_col].values.astype(float)
    z = subset[z_col].values.astype(float)

    enough_points = len(subset) >= 4 and len(np.unique(x)) >= 2 and len(np.unique(y)) >= 2

    if enough_points:
        try:
            Xi, Yi, Zi, bounds = _smooth_surface_from_points(x, y, z, grid_n=220)

            mesh = ax.pcolormesh(
                Xi,
                Yi,
                Zi,
                shading="auto",
                cmap="viridis"
            )
            ax.contour(
                Xi,
                Yi,
                Zi,
                levels=8,
                colors="white",
                linewidths=0.6,
                alpha=0.75
            )
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



def plot_contour_grid(trials_df, param_names, target_col="value", title=None, save_path=None, top_n=20):
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

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.1 * n_rows),
        squeeze=False,
        constrained_layout=True
    )
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
        cbar = fig.colorbar(
            mappable,
            ax=axes_flat[:n_plots].tolist(),
            shrink=0.94,
            pad=0.02,
            aspect=30,
            location="right"
        )
        cbar.set_label("CV NSE")

    _save_show_close(fig, save_path)

