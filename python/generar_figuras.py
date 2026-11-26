from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Rutas absolutas basadas en la ubicación de este archivo.
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LABELS_DIR = RESULTS / "labels_grid"
OUT = RESULTS / "figuras"
RANDOM_STATE = 42
SIL_SAMPLE_SIZE = 2000


def find_col(df, candidates, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(
            f"No encuentro columnas {candidates}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    return None


def save_plot(fig, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG CREADO] {path}")


def parse_label_filename(path):
    # Formato real de los archivos del laboratorio:
    # benchmark_1000_2d_eps0p5_min3_classic.csv
    # benchmark_1000_2d_eps0p5_min3_kdtree.csv
    name = path.stem.lower()
    pattern = (
        r"(?:labels_)?benchmark_(\d+)_(\d+)d_"
        r"eps([0-9p.]+)_min(\d+)_(classic|kdtree)"
    )
    match = re.search(pattern, name)
    if not match:
        return None
    n, dims, eps, min_samples, algorithm = match.groups()
    return {
        "puntos": int(n),
        "dimensiones": int(dims),
        "algoritmo": algorithm,
        "eps": float(eps.replace("p", ".")),
        "min_samples": int(min_samples),
    }


def load_label_csv(path):
    df = pd.read_csv(path)
    cluster_col = find_col(df, ["cluster", "label", "labels"])
    feature_cols = [c for c in df.columns if c != cluster_col]
    if not feature_cols:
        raise ValueError(f"{path.name}: no hay columnas de características.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[cluster_col], errors="coerce").to_numpy()
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid].astype(int)
    if len(y) == 0:
        raise ValueError(f"{path.name}: no contiene filas numéricas válidas.")
    return X, y


def build_metrics_summary():
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de etiquetas: {LABELS_DIR}")

    files = sorted(LABELS_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay archivos CSV dentro de: {LABELS_DIR}")

    rows = []
    ignored = []
    for path in files:
        meta = parse_label_filename(path)
        if meta is None:
            ignored.append(path.name)
            continue

        X, y = load_label_csv(path)
        non_noise = y != -1
        X_clustered = X[non_noise]
        y_clustered = y[non_noise]
        cluster_ids = np.unique(y_clustered)
        silhouette = np.nan

        if len(cluster_ids) >= 2 and len(y_clustered) > len(cluster_ids):
            X_scaled = StandardScaler().fit_transform(X_clustered)
            sample_size = min(SIL_SAMPLE_SIZE, len(y_clustered))
            silhouette = float(silhouette_score(
                X_scaled,
                y_clustered,
                sample_size=sample_size if sample_size < len(y_clustered) else None,
                random_state=RANDOM_STATE
            ))

        rows.append({
            **meta,
            "clusters": int(len(cluster_ids)),
            "ruido": int(np.sum(y == -1)),
            "silhouette": silhouette,
            "archivo": path.name,
        })

    if not rows:
        raise ValueError(
            "Encontré CSV en labels_grid, pero sus nombres no coinciden con el "
            "formato esperado. Ejemplo esperado: "
            "benchmark_10000_2d_classic_eps1p500000_min5.csv"
        )

    summary = pd.DataFrame(rows).sort_values(
        ["puntos", "dimensiones", "eps", "min_samples", "algoritmo"]
    )
    summary_path = RESULTS / "resumen_metricas_90_experimentos.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[CSV CREADO] {summary_path} ({len(summary)} filas)")
    if ignored:
        print(f"[AVISO] Se ignoraron {len(ignored)} CSV por nombre no reconocido.")
    # Se lee nuevamente el CSV generado y esta lectura alimenta las gráficas.
    return pd.read_csv(summary_path)


def read_benchmark():
    path = RESULTS / "benchmark_grid_results.csv"
    if not path.exists():
        print(f"[AVISO] No existe {path}. Se omitirán las gráficas de tiempos.")
        return None

    df = pd.read_csv(path)
    c_points = find_col(df, ["puntos", "points", "n"], required=False)
    c_dims = find_col(df, ["dimensiones", "dimensions", "dims"], required=False)
    c_classic = find_col(df, ["classic_ms", "classicms", "tiempo_clasico_ms"])
    c_kd = find_col(df, ["kdtree_ms", "kd_ms", "kdtree_ms", "kdtreeMs",
                         "tiempo_kdtree_ms"])
    c_speed = find_col(df, ["speedup"], required=False)

    # Si el CSV no trae puntos/dimensiones como columnas, se extraen del dataset.
    if c_points is None or c_dims is None:
        c_dataset = find_col(df, ["dataset", "archivo"], required=False)
        if c_dataset is None:
            raise ValueError(
                "El benchmark necesita columnas puntos/dimensiones o dataset."
            )
        extracted = df[c_dataset].astype(str).str.extract(r"(\d+)_(\d+)d")
        df["_puntos"] = pd.to_numeric(extracted[0], errors="coerce")
        df["_dimensiones"] = pd.to_numeric(extracted[1], errors="coerce")
        c_points, c_dims = "_puntos", "_dimensiones"

    for col in [c_points, c_dims, c_classic, c_kd]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if c_speed is None:
        df["_speedup"] = df[c_classic] / df[c_kd].replace(0, np.nan)
        c_speed = "_speedup"
    else:
        df[c_speed] = pd.to_numeric(df[c_speed], errors="coerce")

    df = df.dropna(subset=[c_points, c_dims, c_classic, c_kd, c_speed]).copy()
    if df.empty:
        raise ValueError("benchmark_grid_results.csv no contiene datos numéricos válidos.")

    df["_puntos"] = df[c_points]
    df["_dimensiones"] = df[c_dims]
    df["_classic"] = df[c_classic]
    df["_kdtree"] = df[c_kd]
    df["_speedup"] = df[c_speed]
    print(f"[INFO] Benchmark leído: {path} ({len(df)} filas)")
    return df


def plot_benchmarks(bench):
    if bench is None:
        return
    df = bench.sort_values(["_dimensiones", "_puntos"])
    labels = [f"{int(n):,}\n{int(d)}D"
              for n, d in zip(df["_puntos"], df["_dimensiones"])]
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, df["_classic"], width, label="DBSCAN clásico")
    ax.bar(x + width / 2, df["_kdtree"], width, label="DBSCAN con KDTree")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cantidad de puntos y dimensionalidad")
    ax.set_ylabel("Tiempo promedio (ms)")
    ax.set_title("Comparación de tiempos de ejecución")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "01_comparacion_tiempos.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(labels, df["_speedup"])
    ax.axhline(1, linestyle="--", linewidth=1, label="Mismo tiempo")
    ax.set_xlabel("Cantidad de puntos y dimensionalidad")
    ax.set_ylabel("Speedup (clásico / KDTree)")
    ax.set_title("Aceleración obtenida mediante KDTree")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "02_speedup_kdtree.png")


def plot_metric_curves(summary):
    # Las dos implementaciones deberían producir la misma partición.
    # Para no dibujar dos curvas superpuestas, se usa una sola implementación.
    data = summary[summary["algoritmo"] == "classic"].copy()
    if data.empty:
        data = summary.copy()

    for (n, dims), group in data.groupby(["puntos", "dimensiones"]):
        group = group.sort_values("eps")
        suffix = f"{int(n)}_{int(dims)}d"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(group["eps"], group["silhouette"], marker="o")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Coeficiente Silhouette")
        ax.set_title(f"Silhouette según epsilon ({n} puntos, {dims}D)")
        ax.grid(alpha=0.25)
        save_plot(fig, f"silhouette_{suffix}.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(group["eps"], group["ruido"], marker="o")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Cantidad de puntos clasificados como ruido")
        ax.set_title(f"Puntos de ruido según epsilon ({n} puntos, {dims}D)")
        ax.grid(alpha=0.25)
        save_plot(fig, f"ruido_{suffix}.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(group["eps"], group["clusters"], marker="o")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Cantidad de clusters")
        ax.set_title(f"Clusters según epsilon ({n} puntos, {dims}D)")
        ax.grid(alpha=0.25)
        save_plot(fig, f"clusters_{suffix}.png")


def find_visualization_file(meta):
    candidates = []
    for path in LABELS_DIR.glob("*.csv"):
        parsed = parse_label_filename(path)
        if parsed is None:
            continue
        if (parsed["puntos"] == meta["puntos"]
                and parsed["dimensiones"] == meta["dimensiones"]
                and parsed["algoritmo"] == "classic"
                and parsed["eps"] == meta["eps"]
                and parsed["min_samples"] == meta["min_samples"]):
            candidates.append(path)
    return candidates[0] if candidates else None


def plot_example(meta):
    path = find_visualization_file(meta)
    if path is None:
        print(f"[AVISO] No encontré etiquetas para la figura {meta}; la omito.")
        return

    X, labels = load_label_csv(path)
    if X.shape[1] == 2:
        coords = X
        axis_names = ("Variable 1", "Variable 2")
        projection = "coordenadas originales"
    elif X.shape[1] > 2:
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(X_scaled)
        variance = pca.explained_variance_ratio_.sum() * 100
        axis_names = ("Componente principal 1", "Componente principal 2")
        projection = f"PCA 2D ({variance:.1f}% de varianza)"
    else:
        print(f"[AVISO] {path.name}: no se puede graficar con una sola dimensión.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in sorted(np.unique(labels)):
        mask = labels == label
        if label == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], c="black",
                       marker="x", s=22, alpha=0.75, label="Ruido")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], s=22,
                       alpha=0.75, label=f"Cluster {label}")
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_title(
        f"DBSCAN {meta['dimensiones']}D: n={meta['puntos']}, "
        f"eps={meta['eps']}, min_samples={meta['min_samples']}\n{projection}"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    filename = (
        f"visualizacion_{meta['puntos']}_{meta['dimensiones']}d_"
        f"eps{str(meta['eps']).replace('.', 'p')}_min{meta['min_samples']}.png"
    )
    save_plot(fig, filename)


def main():
    print("=== GENERADOR DE FIGURAS DBSCAN + KDTREE ===")
    print(f"[INFO] Proyecto: {ROOT}")
    print(f"[INFO] Salida:   {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    summary = build_metrics_summary()
    benchmark = read_benchmark()
    plot_benchmarks(benchmark)
    plot_metric_curves(summary)

    examples = [
        {"puntos": 10000, "dimensiones": 2, "eps": 0.5, "min_samples": 5},
        {"puntos": 10000, "dimensiones": 2, "eps": 1.5, "min_samples": 5},
        {"puntos": 10000, "dimensiones": 5, "eps": 0.5, "min_samples": 5},
        {"puntos": 10000, "dimensiones": 5, "eps": 1.5, "min_samples": 5},
    ]
    for meta in examples:
        plot_example(meta)

    generated = list(OUT.glob("*.png"))
    print("\n=== PROCESO FINALIZADO ===")
    print(f"Figuras PNG en {OUT}: {len(generated)}")
    print(f"Resumen CSV: {RESULTS / 'resumen_metricas_90_experimentos.csv'}")
    if not generated:
        raise RuntimeError("El script terminó, pero no se generó ningún PNG.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        print("Revisa el mensaje anterior: indica qué archivo o columna falta.")
        sys.exit(1)
