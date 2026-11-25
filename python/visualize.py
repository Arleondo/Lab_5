"""
Genera las figuras del informe DBSCAN + KDTree usando SOLO los archivos reales:
- results/benchmark_grid_results.csv
- results/labels_grid/*.csv

No requiere resumen_metricas_90_experimentos.csv.
Calcula clusters, ruido y Silhouette Score directamente desde cada CSV
de etiquetas. Para Silhouette se excluye ruido (-1) y se usa una muestra
reproducible de hasta 2.000 puntos para controlar el tiempo.

Ejecutar desde la raíz del proyecto:
    python python/generar_figuras.py
"""

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LABELS_DIR = RESULTS / "labels_grid"
OUT = RESULTS / "figuras"
RANDOM_STATE = 42
SIL_SAMPLE_SIZE = 2000


def save_plot(fig, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path.relative_to(ROOT)}")


def find_col(df, candidates):
    lookup = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"No encuentro ninguna de estas columnas: {candidates}\n"
        f"Columnas disponibles: {list(df.columns)}"
    )


def read_benchmark():
    path = RESULTS / "benchmark_grid_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_csv(path)

    # Nombres producidos por el benchmark C++/Python del laboratorio.
    c_points = find_col(df, ["puntos", "points", "n"])
    c_dims = find_col(df, ["dimensiones", "dimensions", "dims"])
    c_classic = find_col(df, ["classic_ms", "classicMs", "tiempo_clasico_ms"])
    c_kd = find_col(df, ["kdtree_ms", "kd_ms", "kdtreeMs", "tiempo_kdtree_ms"])
    c_speed = find_col(df, ["speedup"])

    for c in [c_points, c_dims, c_classic, c_kd, c_speed]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c_points, c_dims, c_classic, c_kd, c_speed])
    if df.empty:
        raise ValueError("benchmark_grid_results.csv no contiene filas numéricas válidas.")

    df["_points"] = df[c_points]
    df["_dims"] = df[c_dims]
    df["_classic"] = df[c_classic]
    df["_kd"] = df[c_kd]
    df["_speedup"] = df[c_speed]
    return df


def plot_timing(bench):
    df = bench.sort_values(["_dims", "_points"]).copy()
    labels = [f"{int(n):,}\n{int(d)}D" for n, d in zip(df["_points"], df["_dims"])]
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, df["_classic"], width, label="DBSCAN clásico")
    ax.bar(x + width / 2, df["_kd"], width, label="DBSCAN con KDTree")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cantidad de puntos y dimensionalidad")
    ax.set_ylabel("Tiempo promedio (ms)")
    ax.set_title("Comparación de tiempos de ejecución")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "01_comparacion_tiempos.png")


def plot_speedup(bench):
    df = bench.sort_values(["_dims", "_points"]).copy()
    labels = [f"{int(n):,} pts, {int(d)}D" for n, d in zip(df["_points"], df["_dims"])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, df["_speedup"])
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Mismo tiempo")
    ax.bar_label(bars, fmt="%.2fx", padding=3)
    ax.set_ylabel("Speedup (tiempo clásico / tiempo KDTree)")
    ax.set_title("Aceleración obtenida con KDTree")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, "02_speedup_kdtree.png")


def parse_label_filename(path):
    # Ejemplo: benchmark_10000_2d_eps0p5_min5_classic.csv
    m = re.match(
        r"benchmark_(\d+)_(\d+)d_eps([0-9p]+)_min(\d+)_(classic|kdtree)\.csv$",
        path.name,
        re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "n": int(m.group(1)),
        "dims": int(m.group(2)),
        "eps": float(m.group(3).replace("p", ".")),
        "min_samples": int(m.group(4)),
        "algorithm": m.group(5).lower(),
    }


def load_label_file(path):
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError(f"{path.name}: falta la columna 'cluster'.")
    feature_cols = [c for c in df.columns if c.lower() != "cluster"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["cluster"].to_numpy(dtype=int)
    return X, y


def calculate_metrics_for_labels():
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta: {LABELS_DIR}")
    rows = []
    files = sorted(LABELS_DIR.glob("*.csv"))
    for path in files:
        meta = parse_label_filename(path)
        if meta is None:
            continue
        X, y = load_label_file(path)
        non_noise = y != -1
        X_valid = X[non_noise]
        y_valid = y[non_noise]
        clusters = np.unique(y_valid)
        n_noise = int(np.sum(y == -1))
        sil = np.nan

        # Silhouette solo está definido si hay al menos dos clusters
        # y menos clusters que observaciones evaluadas.
        if len(clusters) >= 2 and len(y_valid) > len(clusters):
            X_scaled = StandardScaler().fit_transform(X_valid)
            sample_size = min(SIL_SAMPLE_SIZE, len(y_valid))
            sil = float(
                silhouette_score(
                    X_scaled,
                    y_valid,
                    sample_size=sample_size if sample_size < len(y_valid) else None,
                    random_state=RANDOM_STATE,
                )
            )

        rows.append({
            **meta,
            "clusters": int(len(clusters)),
            "noise": n_noise,
            "silhouette": sil,
            "file": path.name,
        })

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError(
            "No encontré CSV de etiquetas con el formato esperado en results/labels_grid. "
            "Revisa los nombres de los archivos."
        )

    # Guardar el resumen calculado a partir de los CSV reales.
    summary_path = RESULTS / "metricas_calculadas_desde_labels.csv"
    metrics.to_csv(summary_path, index=False)
    print(f"[OK] Métricas calculadas desde etiquetas: {summary_path.relative_to(ROOT)}")
    print(f"[INFO] Archivos de etiquetas analizados: {len(metrics)}")
    return metrics


def plot_parameter_curves(metrics, metric, ylabel, prefix, title):
    for n in sorted(metrics["n"].unique()):
        for dims in sorted(metrics["dims"].unique()):
            subset = metrics[(metrics["n"] == n) & (metrics["dims"] == dims)]
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for mins in sorted(subset["min_samples"].unique()):
                curve = subset[subset["min_samples"] == mins].sort_values("eps")
                ax.plot(curve["eps"], curve[metric], marker="o",
                        label=f"min_samples={mins}")
            ax.set_xlabel("eps")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}: n={n}, dimensionalidad={dims}D")
            ax.legend()
            ax.grid(alpha=0.25)
            save_plot(fig, f"{prefix}_n{n}_{dims}d.png")


def plot_cluster_visualization(metrics):
    # Elegir n=10.000, min_samples=5 y eps=0.5/1.5, si están disponibles.
    for dims in (2, 5):
        for eps in (0.5, 1.5):
            candidates = metrics[
                (metrics["n"] == 10000)
                & (metrics["dims"] == dims)
                & (metrics["min_samples"] == 5)
                & (np.isclose(metrics["eps"], eps))
                & (metrics["algorithm"] == "classic")
                ]
            if candidates.empty:
                print(f"[AVISO] No hay etiquetas para {dims}D, eps={eps}, min_samples=5.")
                continue

            meta = candidates.iloc[0]
            path = LABELS_DIR / meta["file"]
            X, y = load_label_file(path)

            if dims == 2:
                coords = X
                xlabel, ylabel = "Característica 1", "Característica 2"
                projection = "Coordenadas originales"
            else:
                Xs = StandardScaler().fit_transform(X)
                pca = PCA(n_components=2, random_state=RANDOM_STATE)
                coords = pca.fit_transform(Xs)
                variance = pca.explained_variance_ratio_.sum() * 100
                xlabel, ylabel = "Componente principal 1", "Componente principal 2"
                projection = f"PCA 2D, varianza explicada: {variance:.1f}%"

            fig, ax = plt.subplots(figsize=(8.5, 6.2))
            for label in sorted(np.unique(y)):
                mask = y == label
                if label == -1:
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                               marker="x", s=20, label="Ruido (-1)")
                else:
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                               s=18, alpha=0.72, label=f"Cluster {label}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"DBSCAN {dims}D | n=10000 | eps={eps} | min_samples=5\n{projection}"
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            save_plot(fig, f"06_clusters_{dims}d_eps{str(eps).replace('.', 'p')}_min5.png")


def main():
    print("Leyendo benchmark_grid_results.csv...")
    bench = read_benchmark()
    plot_timing(bench)
    plot_speedup(bench)

    print("\nCalculando métricas desde results/labels_grid...")
    metrics = calculate_metrics_for_labels()

    plot_parameter_curves(
        metrics, "silhouette", "Silhouette Score (muestra reproducible)",
        "03_silhouette", "Silhouette según hiperparámetros"
    )
    plot_parameter_curves(
        metrics, "noise", "Cantidad de puntos de ruido",
        "04_ruido", "Puntos de ruido según hiperparámetros"
    )
    plot_parameter_curves(
        metrics, "clusters", "Número de clusters",
        "05_clusters", "Clusters según hiperparámetros"
    )
    plot_cluster_visualization(metrics)

    print(f"\nListo. Figuras guardadas en: {OUT}")
    print("Nota: Silhouette se calcula sobre puntos no etiquetados como ruido,")
    print(f"      con una muestra reproducible de hasta {SIL_SAMPLE_SIZE} puntos.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
