# reportes_html.py

from __future__ import annotations

import io
import html
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


HYPERPARAM_SIG_KEYS = {"learning_rate", "weight_decay", "dropout", "leaky_relu_slope"}


# ---------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------
def round_sig_html(value: Any, sig: int = 3) -> Any:
    try:
        if value is None:
            return None
        value = float(value)
        if pd.isna(value) or value == 0:
            return value
        return float(f"{value:.{sig}g}")
    except Exception:
        return value


def safe_round(value: Any, ndigits: int = 3, key: str | None = None) -> Any:
    try:
        if value is None:
            return None

        if key in HYPERPARAM_SIG_KEYS:
            return round_sig_html(value, sig=3)

        return round(float(value), ndigits)
    except Exception:
        return value


def round_dict_values(d: Dict[str, Any], ndigits: int = 3) -> Dict[str, Any]:
    return {k: safe_round(v, ndigits, key=k) for k, v in d.items()}


def round_numeric_df(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
    df2 = df.copy()
    numeric_cols = df2.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        col_name = col[-1] if isinstance(col, tuple) else col
        if col_name in HYPERPARAM_SIG_KEYS:
            df2[col] = df2[col].apply(lambda x: safe_round(x, ndigits, key=col_name))
        else:
            df2[col] = df2[col].round(ndigits)

    return df2


def dataframe_to_html_table(df: pd.DataFrame, best_col: str | None = None, maximize: bool = True, highlight_best: bool = True, ndigits: int = 3) -> str:
    if df is None or df.empty:
        return "<p>No hay datos.</p>"

    df_show = round_numeric_df(df, ndigits).copy()

    best_idx = None
    if highlight_best and best_col is not None and best_col in df_show.columns:
        try:
            best_idx = df_show[best_col].astype(float).idxmax() if maximize else df_show[best_col].astype(float).idxmin()
        except Exception:
            best_idx = None

    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in df_show.columns)

    rows_html = []
    for idx, row in df_show.iterrows():
        row_class = "best-row" if best_idx is not None and idx == best_idx else ""
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values)
        rows_html.append(f'<tr class="{row_class}">{cells}</tr>')

    return f"""
    <table class="data-table">
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    """


def multiindex_dataframe_to_html_table(df: pd.DataFrame, best_col: tuple | None = None, maximize: bool = True, highlight_best: bool = True, ndigits: int = 3) -> str:
    if df is None or df.empty:
        return "<p>No hay datos.</p>"

    df_show = round_numeric_df(df, ndigits).copy()

    if not isinstance(df_show.columns, pd.MultiIndex):
        return dataframe_to_html_table(
            df_show,
            best_col=best_col if isinstance(best_col, str) else None,
            maximize=maximize,
            highlight_best=highlight_best,
            ndigits=ndigits
        )

    level0 = df_show.columns.get_level_values(0)
    level1 = df_show.columns.get_level_values(1)

    top_row = ""
    i = 0
    while i < len(df_show.columns):
        current = level0[i]
        span = 1
        j = i + 1
        while j < len(df_show.columns) and level0[j] == current:
            span += 1
            j += 1
        top_row += f'<th colspan="{span}">{html.escape(str(current))}</th>'
        i = j

    second_row = "".join(f"<th>{html.escape(str(v))}</th>" for v in level1)

    best_idx = None
    if highlight_best and best_col is not None and best_col in df_show.columns:
        try:
            best_idx = df_show[best_col].astype(float).idxmax() if maximize else df_show[best_col].astype(float).idxmin()
        except Exception:
            best_idx = None

    body_rows = []
    for idx, row in df_show.iterrows():
        row_class = "best-row" if best_idx is not None and idx == best_idx else ""
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values)
        body_rows.append(f'<tr class="{row_class}">{cells}</tr>')

    return f"""
    <table class="data-table">
        <thead>
            <tr>{top_row}</tr>
            <tr>{second_row}</tr>
        </thead>
        <tbody>
            {''.join(body_rows)}
        </tbody>
    </table>
    """


def dict_to_html_table(d: Dict[str, Any], title: str | None = None) -> str:
    if not d:
        return ""

    rows = []
    for k, v in d.items():
        rows.append(
            f"""
            <tr>
                <th>{html.escape(str(k))}</th>
                <td>{html.escape(str(v))}</td>
            </tr>
            """
        )

    table_html = f"""
    <table class="kv-table">
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """

    if title:
        return f"<h4>{html.escape(title)}</h4>{table_html}"
    return table_html


def fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def series_barplot_base64(series: pd.Series, title: str, ylabel: str = "") -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig_to_base64(fig)


def image_file_to_base64(image_path: str | Path) -> str | None:
    path = Path(image_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded


def datasets_summary_to_html_table(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "<p>No hay datos.</p>"

    rows = []
    for result in results:
        dataset_name = result.get("dataframe_name", "")
        features = result.get("feature_names", [])
        features_text = ", ".join(str(v) for v in features) if features else ""

        rows.append(
            f"""
            <tr>
                <th>{html.escape(str(dataset_name))}</th>
                <td>{html.escape(features_text)}</td>
            </tr>
            """
        )

    return f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>Dataset</th>
                <th>Variables</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def dataset_variables_table(feature_names: List[str], dataset_name: str | None = None) -> str:
    if not feature_names:
        return "<p>No hay variables.</p>"

    left_header = html.escape(dataset_name) if dataset_name else "Dataset"

    rows = "".join(
        f"<tr><td>{html.escape(str(feature))}</td></tr>"
        for feature in feature_names
    )

    return f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>{left_header}</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


def get_image_html_size(filename: str) -> str:
    name = filename.lower()

    if "objective_curve" in name:
        return 'style="width:50%; max-width:50%;"'

    if "loss_curve" in name:
        return 'style="width:45%; max-width:45%;"'

    if "real_vs_pred" in name:
        return 'style="width:50%; max-width:50%;"'

    if "residual_histogram" in name:
        return 'style="width:40%; max-width:40%;"'

    if "captum_bar" in name:
        return 'style="width:35%; max-width:35%;"'

    if "captum_beeswarm" in name:
        return 'style="width:42%; max-width:42%;"'

    if "captum_heatmap" in name:
        return 'style="width:58%; max-width:58%;"'

    if "hyperparam_importances" in name:
        return 'style="width:40%; max-width:40%;"'

    if "hyperparam_slice" in name:
        return 'style="width:60%; max-width:60%;"'

    if "hyperparam_contour" in name:
        return 'style="width:50%; max-width:50%;"'

    return ""


def params_summary_with_ranges_table(params_df: pd.DataFrame, results: List[Dict[str, Any]]) -> str:
    if params_df is None or params_df.empty:
        return "<p>No hay datos.</p>"

    df_show = round_numeric_df(params_df.copy(), ndigits=3)

    search_ranges = {}
    for result in results:
        search_ranges = result.get("search_ranges", {}) or {}
        if search_ranges:
            break

    ordered_cols = list(df_show.columns)
    header_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in ordered_cols)

    range_cells = []
    for col in ordered_cols:
        if col == "DataFrame":
            range_cells.append("<td>Rango de búsqueda</td>")
        else:
            range_cells.append(f"<td>{html.escape(str(search_ranges.get(col, '')))}</td>")

    body_rows = [f"<tr>{''.join(range_cells)}</tr>"]

    for _, row in df_show.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in ordered_cols)
        body_rows.append(f"<tr>{cells}</tr>")

    return f"""
    <table class="data-table">
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {''.join(body_rows)}
        </tbody>
    </table>
    """


# ---------------------------------------------------------
# Orden de figuras
# ---------------------------------------------------------
PLOT_ORDER_PATTERNS: List[Tuple[str, int]] = [
    ("objective_curve", 1),
    ("loss_curve", 2),
    ("train_vs_actual", 3),
    ("test_vs_actual", 4),
    ("full_temporal_results", 5),
    ("real_vs_pred", 6),
    ("residual_histogram", 7),
    ("captum_beeswarm", 8),
    ("captum_bar", 9),
    ("captum_heatmap", 10),
    ("captum_top_temporal", 11),
    ("hyperparam_importances", 12),
    ("hyperparam_slice", 13),
    ("hyperparam_parallel", 14),
    ("hyperparam_contour", 15),
]


def get_plot_priority(filename: str) -> int:
    """
    Orden deseado definido en PLOT_ORDER_PATTERNS para los gráficos MLP,
    incluyendo pérdida, residuos, atribuciones e hiperparámetros.
    """
    name = filename.lower()

    for pattern, priority in PLOT_ORDER_PATTERNS:
        if pattern in name:
            return priority

    return 99


def sort_image_paths(image_paths: List[Path]) -> List[Path]:
    unique_paths = list(dict.fromkeys(image_paths))
    return sorted(unique_paths, key=lambda p: (get_plot_priority(p.name), p.name.lower()))


# ---------------------------------------------------------
# Estilos
# ---------------------------------------------------------
def build_common_css() -> str:
    return """
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            margin: 0;
            padding: 0;
            background: #f4f6f8;
            color: #1f2937;
        }

        .container {
            max-width: 1500px;
            margin: 0 auto;
            padding: 24px;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }

        h1, h2, h3, h4 {
            margin-top: 0;
            color: #111827;
        }

        .muted {
            color: #6b7280;
            font-size: 0.95rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.95rem;
        }

        th, td {
            border: 1px solid #d1d5db;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }

        th {
            background: #eef2f7;
        }

        tr:nth-child(even) td {
            background: #fafafa;
        }

        .best-row td {
            background: #e8f7e8 !important;
            font-weight: bold;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .kv-table th {
            width: 38%;
            background: #f8fafc;
        }

        .img-box {
            text-align: center;
            margin-top: 12px;
        }

        .img-box img {
            max-width: 100%;
            height: auto;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: white;
        }

        .code-block {
            background: #0f172a;
            color: #e5e7eb;
            border-radius: 10px;
            padding: 16px;
            overflow-x: auto;
            white-space: pre-wrap;
            font-family: Consolas, "Courier New", monospace;
            line-height: 1.35;
            font-size: 0.92rem;
        }

        .dataset-title {
            padding-top: 8px;
            border-top: 2px solid #e5e7eb;
            margin-top: 20px;
        }

        .dataset-title h2 {
            font-size: 2.8rem;
            margin-bottom: 18px;
        }

        .small {
            font-size: 0.9rem;
        }
    </style>
    """


# ---------------------------------------------------------
# Captura de salida
# ---------------------------------------------------------
class TeeOutput:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stream.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original_stream.flush()

    def get_value(self):
        return self.buffer.getvalue()


def save_log_txt(log_text: str, output_dir: str | Path, filename: str = "log_salida.txt") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(log_text, encoding="utf-8")
    return out_path


def save_log_html(
    log_text: str,
    output_dir: str | Path,
    title: str = "Log de salida del experimento",
    filename: str = "log_salida.html"
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    css = build_common_css()
    escaped = html.escape(log_text)

    html_doc = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>{html.escape(title)}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>{html.escape(title)}</h1>
                                <div class="code-block">{escaped}</div>
            </div>
        </div>
    </body>
    </html>
    """

    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------
# Imágenes de cada dataset
# ---------------------------------------------------------
def build_dataset_images_section(run_output_dir: str | Path, dataset_name: str) -> str:
    dataset_dir = Path(run_output_dir) / dataset_name
    if not dataset_dir.exists():
        return ""

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_paths: List[Path] = []

    for pattern in image_extensions:
        image_paths.extend(dataset_dir.glob(pattern))

    image_paths = sort_image_paths(image_paths)
    if not image_paths:
        return ""

    image_blocks = []

    for img_path in image_paths:
        img_b64 = image_file_to_base64(img_path)
        if img_b64 is not None:
            suffix = img_path.suffix.lower()
            if suffix in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif suffix == ".webp":
                mime_type = "image/webp"
            else:
                mime_type = "image/png"

            size_attr = get_image_html_size(img_path.name)

            image_blocks.append(
                f"""
                <div class="card">
                    <h4>{html.escape(img_path.name)}</h4>
                    <div class="img-box">
                        <img src="data:{mime_type};base64,{img_b64}" alt="{html.escape(img_path.name)}" {size_attr}>
                    </div>
                </div>
                """
            )

    return "".join(image_blocks)




def global_nse_stats_to_multiindex_html_table(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "<p>No hay datos.</p>"
    df_show = round_numeric_df(df, 2).copy()
    cols = pd.MultiIndex.from_tuples([("Performance NSE", "Estadística"), ("Performance NSE", "CV Mean"), ("Performance NSE", "CV Std"), ("Performance NSE", "Train"), ("Performance NSE", "Test")])
    df_show.columns = cols
    return multiindex_dataframe_to_html_table(df_show, highlight_best=False, ndigits=2)


def load_run_timing_summary(run_output_dir: str | Path, dict_name: str) -> Dict[str, Any]:
    """Lee el tiempo total de corrida guardado por pipeline_mlp.py."""
    path = Path(run_output_dir) / f"{dict_name}_run_timing.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        out = {}
        if "run_execution_time" in row:
            out["Tiempo total de ejecución"] = row.get("run_execution_time")
        if "run_execution_seconds" in row:
            out["Tiempo total de ejecución [s]"] = row.get("run_execution_seconds")
        return out
    except Exception:
        return {}


# ---------------------------------------------------------
# Reporte principal
# ---------------------------------------------------------
def generate_html_report(
    run_output_dir: str | Path,
    dict_name: str,
    train_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    params_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    global_nse_stats_df: pd.DataFrame | None = None,
    results: List[Dict[str, Any]] | None = None,
    config_mode: str = "",
    n_trials: int = 0,
    html_filename: str = "reporte_resultados.html"
) -> Path:
    run_output_dir = Path(run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_output_dir / html_filename

    css = build_common_css()

    if results is None:
        results = []

    summary_meta = {
        "Diccionario procesado": dict_name,
        "Modo de configuración": config_mode,
        "Cantidad de datasets": len(results),
        "Cantidad de trials": n_trials,
        "Carpeta de salida": str(run_output_dir),
    }
    summary_meta.update(load_run_timing_summary(run_output_dir, dict_name))
    charts_html = ""

    datasets_summary_html = datasets_summary_to_html_table(results)

    dataset_details_html = []
    for result in results:
        dataset_name = result["dataframe_name"]

        detail_meta = {
            "Target": result["target"],
            "config_mode": result["config_mode"],
            "search_space_type": result["search_space_type"],
            "use_early_stopping": result["use_early_stopping"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "n_features": result["n_features"],
            "train_start": result["train_start"],
            "train_end": result["train_end"],
            "test_start": result["test_start"],
            "test_end": result["test_end"],
            "cv_nse_mean": result["cv_nse_mean"],
            "cv_nse_std": result["cv_nse_std"],
            "best_epoch": result.get("best_epoch", ""),
            "best_val_loss": result.get("best_val_loss", ""),
            "max_epochs": result.get("max_epochs", ""),
            "patience": result.get("patience", ""),
            "final_val_size": result["final_val_size"],
            "scale_y": result.get("scale_y", ""),
            "pruning": result.get("pruning", ""),
            "execution_seconds": result.get("execution_seconds", ""),
            "execution_time": result.get("execution_time", ""),
        }

        search_ranges = result.get("search_ranges", {})
        feature_names = result.get("feature_names", [])

        dataset_block = f"""
        <div class="card dataset-title">
            <h2>Resultados para {html.escape(dataset_name)}</h2>
            <div>
                {dataset_variables_table(feature_names, dataset_name)}
            </div>
            <div class="grid-2">
                <div>
                    {dict_to_html_table(round_dict_values(result["metrics_train"], 2), "Métricas de train")}
                    {dict_to_html_table(round_dict_values(result["metrics_test"], 2), "Métricas de test")}
                    {dict_to_html_table(search_ranges, "Rangos de búsqueda de hiperparámetros")}
                </div>
                <div>
                    {dict_to_html_table(result["best_params"], "Mejores hiperparámetros")}
                    {dict_to_html_table(detail_meta, "Metadatos")}
                </div>
            </div>
        </div>
        """

        dataset_block += build_dataset_images_section(run_output_dir, dataset_name)
        dataset_details_html.append(dataset_block)

    comparison_html = (
        multiindex_dataframe_to_html_table(
            comparison_df,
            best_col=("NSE", "Test"),
            maximize=True,
            highlight_best=False,
            ndigits=2
        )
        if isinstance(comparison_df.columns, pd.MultiIndex)
        else dataframe_to_html_table(
            comparison_df,
            best_col="NSE_test",
            maximize=True,
            highlight_best=False,
            ndigits=2
        )
    )

    html_doc = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>Reporte de resultados</title>
        {css}
    </head>
    <body>
        <div class="container">

            <div class="card">
                <h1>Reporte de resultados del modelado</h1>
            </div>

            <div class="card">
                <h2>Resumen general</h2>
                {dict_to_html_table(summary_meta)}
            </div>

            <div class="card">
                <h2>Datasets analizados</h2>
                {datasets_summary_html}
            </div>

            <div class="card">
                <h2>Resumen de métricas de train</h2>
                {dataframe_to_html_table(train_metrics_df, ndigits=2)}
            </div>

            <div class="card">
                <h2>Resumen de métricas de test</h2>
                {dataframe_to_html_table(test_metrics_df, best_col="NSE", maximize=True, highlight_best=False, ndigits=2)}
            </div>

            <div class="card">
                <h2>Resumen de mejores parámetros</h2>
                {params_summary_with_ranges_table(params_df, results)}
            </div>

            <div class="card">
                <h2>Resumen de metadatos</h2>
                {dataframe_to_html_table(metadata_df, ndigits=2)}
            </div>

            <div class="card">
                <h2>Resumen comparativo final</h2>
                {comparison_html}
            </div>

            <div class="card">
                <h2>Estadísticas globales de performance</h2>
                {global_nse_stats_to_multiindex_html_table(global_nse_stats_df)}
            </div>

            {charts_html}

            {''.join(dataset_details_html)}

        </div>
    </body>
    </html>
    """

    out_path.write_text(html_doc, encoding="utf-8")
    return out_path