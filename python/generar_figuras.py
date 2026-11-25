from pathlib import Path
import argparse
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def buscar_archivo(root: Path, nombres):
    for nombre in nombres:
        candidatos = [root / nombre, root / "results" / nombre]
        for ruta in candidatos:
            if ruta.exists():
                return ruta
    return None


def guardar(fig, destino: Path):
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destino, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {destino}")


def grafica_tiempos(bench: pd.DataFrame, out: Path):
    df = bench.copy()
    df["etiqueta"] = df["puntos"].astype(str) + " puntos\n" + df["dimensiones"].astype(str) + "D"
    df = df.sort_values(["dimensiones", "puntos"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(x - width / 2, df["classic_ms"], width, label="DBSCAN clásico")
    ax.bar(x + width / 2, df["kdtree_ms"], width, label="DBSCAN con KDTree")
    ax.set_xticks(x)
    ax.set_xticklabels(df["etiqueta"])
    ax.set_ylabel("Tiempo promedio (ms)")
    ax.set_title("Comparación de tiempos de ejecución")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    guardar(fig, out / "01_comparacion_tiempos.png")


def grafica_speedup(bench: pd.DataFrame, out: Path):
    df = bench.copy()
    df["etiqueta"] = df["puntos"].astype(str) + " puntos, " + df["dimensiones"].astype(str) + "D"
    df = df.sort_values(["dimensiones", "puntos"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(df["etiqueta"], df["speedup"])
    ax.axhline(1.0, linestyle="--", linewidth=1, label="Mismo tiempo")
    ax.bar_label(bars, fmt="%.2fx", padding=3)
    ax.set_ylabel("Speedup (clásico / KDTree)")
    ax.set_title("Aceleración obtenida con KDTree")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    guardar(fig, out / "02_speedup_kdtree.png")


def grafica_hiperparametros(metrics: pd.DataFrame, out: Path):
    # Se grafica la métrica del algoritmo clásico; ARI=1 demuestra que
    # KDTree produce la misma partición para las parejas comparadas.
    df = metrics.copy()
    required = {"n", "dims", "eps", "min_samples",
                "clusters_classic", "noise_classic", "silhouette_classic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el resumen de métricas: {sorted(missing)}")

    # Gráfica de silueta en función de eps, una línea por min_samples,
    # separada por tamaño y dimensionalidad.
    for n in sorted(df["n"].unique()):
        for dims in sorted(df["dims"].unique()):
            sub = df[(df["n"] == n) & (df["dims"] == dims)]
            fig, ax = plt.subplots(figsize=(8, 5))
            for mins in sorted(sub["min_samples"].unique()):
                curva = sub[sub["min_samples"] == mins].sort_values("eps")
                ax.plot(curva["eps"], curva["silhouette_classic"],
                        marker="o", label=f"min_samples={mins}")
            ax.set_xlabel("eps")
            ax.set_ylabel("Silhouette Score")
            ax.set_title(f"Silhouette según hiperparámetros: n={n}, {dims}D")
            ax.legend()
            ax.grid(alpha=0.25)
            guardar(fig, out / f"03_silhouette_n{n}_{dims}d.png")

    # Cantidad de ruido en función de eps.
    for n in sorted(df["n"].unique()):
        for dims in sorted(df["dims"].unique()):
            sub = df[(df["n"] == n) & (df["dims"] == dims)]
            fig, ax = plt.subplots(figsize=(8, 5))
            for mins in sorted(sub["min_samples"].unique()):
                curva = sub[sub["min_samples"] == mins].sort_values("eps")
                ax.plot(curva["eps"], curva["noise_classic"],
                        marker="o", label=f"min_samples={mins}")
            ax.set_xlabel("eps")
            ax.set_ylabel("Cantidad de puntos de ruido")
            ax.set_title(f"Puntos de ruido según hiperparámetros: n={n}, {dims}D")
            ax.legend()
            ax.grid(alpha=0.25)
            guardar(fig, out / f"04_ruido_n{n}_{dims}d.png")

    # Cantidad de clusters en función de eps.
    for n in sorted(df["n"].unique()):
        for dims in sorted(df["dims"].unique()):
            sub = df[(df["n"] == n) & (df["dims"] == dims)]
            fig, ax = plt.subplots(figsize=(8, 5))
            for mins in sorted(sub["min_samples"].unique()):
                curva = sub[sub["min_samples"] == mins].sort_values("eps")
                ax.plot(curva["eps"], curva["clusters_classic"],
                        marker="o", label=f"min_samples={mins}")
            ax.set_xlabel("eps")
            ax.set_ylabel("Número de clusters")
            ax.set_title(f"Clusters según hiperparámetros: n={n}, {dims}D")
            ax.legend()
            ax.grid(alpha=0.25)
            guardar(fig, out / f"05_clusters_n{n}_{dims}d.png")


def localizar_labels(root: Path):
    carpetas = [
        root / "results" / "labels_grid",
        root / "results" / "extracted_labels",
        root / "extracted_labels",
        root / "labels_grid",
    ]
    for carpeta in carpetas:
        if carpeta.exists() and list(carpeta.glob("*.csv")):
            return carpeta
    return None


def nombre_label(n, dims, eps, mins, algoritmo):
    eps_txt = str(float(eps)).replace(".", "p")
    return f"benchmark_{n}_{dims}d_eps{eps_txt}_min{mins}_{algoritmo}.csv"


def grafica_clusters(labels_dir: Path, out: Path, n, dims, eps, mins):
    # Los CSV contienen las características originales y la etiqueta "cluster".
    archivo = labels_dir / nombre_label(n, dims, eps, mins, "classic")
    if not archivo.exists():
        # Alternativa: localizar por patrón si la escritura del eps difiere.
        patron = f"benchmark_{n}_{dims}d_eps*_min{mins}_classic.csv"
        candidatos = sorted(labels_dir.glob(patron))
        objetivo = round(float(eps), 6)
        for candidato in candidatos:
            m = re.search(r"_eps([0-9p]+)_", candidato.name)
            if m and abs(float(m.group(1).replace("p", ".")) - objetivo) < 1e-6:
                archivo = candidato
                break
    if not archivo.exists():
        print(f"[AVISO] No se encontró el CSV de etiquetas: {archivo.name}")
        return

    df = pd.read_csv(archivo)
    if "cluster" not in df.columns:
        raise ValueError(f"El archivo {archivo} no contiene la columna cluster.")
    columnas = [c for c in df.columns if c != "cluster"]
    X = df[columnas].to_numpy(dtype=float)
    y = df["cluster"].to_numpy(dtype=int)

    if X.shape[1] == 2:
        coords = X
        eje_x, eje_y = "Característica 1", "Característica 2"
        nota = "coordenadas originales"
    else:
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_std)
        var = pca.explained_variance_ratio_.sum() * 100
        eje_x, eje_y = "Componente principal 1", "Componente principal 2"
        nota = f"PCA 2D, varianza explicada: {var:.1f}%"

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for etiqueta in sorted(np.unique(y)):
        mask = y == etiqueta
        if etiqueta == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       marker="x", s=20, label="Ruido (-1)")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=18, alpha=0.72, label=f"Cluster {etiqueta}")
    ax.set_xlabel(eje_x)
    ax.set_ylabel(eje_y)
    ax.set_title(f"DBSCAN {dims}D | n={n} | eps={eps} | min_samples={mins}\n{nota}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    eps_tag = str(float(eps)).replace(".", "p")
    guardar(fig, out / f"06_clusters_{dims}d_n{n}_eps{eps_tag}_min{mins}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Raíz del proyecto")
    parser.add_argument("--output", default="results/figuras_informe",
                        help="Carpeta de salida relativa a la raíz")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.output).resolve()

    benchmark_path = buscar_archivo(root, ["benchmark_results.csv", "benchmark_grid_results.csv"])
    metrics_path = buscar_archivo(root, ["resumen_metricas_90_experimentos.csv"])

    if benchmark_path is None:
        sys.exit("No encuentro benchmark_results.csv ni benchmark_grid_results.csv.")
    if metrics_path is None:
        sys.exit("No encuentro resumen_metricas_90_experimentos.csv.")

    bench = pd.read_csv(benchmark_path)
    metrics = pd.read_csv(metrics_path)

    print(f"Benchmark: {benchmark_path}")
    print(f"Métricas:  {metrics_path}")
    grafica_tiempos(bench, out)
    grafica_speedup(bench, out)
    grafica_hiperparametros(metrics, out)

    labels_dir = localizar_labels(root)
    if labels_dir is None:
        print("[AVISO] No encontré la carpeta con los CSV de etiquetas. Se omiten las figuras de clusters.")
    else:
        print(f"Etiquetas: {labels_dir}")
        # Configuraciones representativas para mostrar el efecto de eps,
        # en los dos espacios dimensionales solicitados.
        for dims in (2, 5):
            for eps in (0.5, 1.5):
                grafica_clusters(labels_dir, out, 10000, dims, eps, 5)

    print(f"\nFiguras generadas en: {out}")


if __name__ == "__main__":
    main()
