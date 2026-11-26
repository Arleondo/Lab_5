


from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Ejecutar desde la carpeta principal del proyecto:
# python python/graficar_resultados.py

ARCHIVO = Path("results/labels_output_kdtree_eps1p800000_min20.csv")
SALIDA = Path("results/figuras/clusterizacion_output_kdtree_eps1p8_min20.png")

def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO.resolve()}")

    df = pd.read_csv(ARCHIVO)

    columnas = {"x1", "x2", "cluster"}
    faltantes = columnas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas {sorted(faltantes)}. "
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    df = df[["x1", "x2", "cluster"]].apply(pd.to_numeric, errors="coerce").dropna()
    X = df[["x1", "x2"]].to_numpy()
    labels = df["cluster"].astype(int).to_numpy()

    outliers = int((labels == -1).sum())
    clusters = sorted(set(labels) - {-1})
    print(f"Puntos procesados: {len(df)}")
    print(f"Clusters encontrados: {len(clusters)}")
    print(f"Outliers: {outliers}")

    # Calcular Silhouette excluyendo el ruido (-1).
    mask = labels != -1
    X_clustered = X[mask]
    labels_clustered = labels[mask]
    n_clusters = len(set(labels_clustered))

    if n_clusters >= 2 and len(X_clustered) > n_clusters:
        score = silhouette_score(X_clustered, labels_clustered)
        print(f"Silhouette Score (sin ruido): {score:.6f}")
    else:
        score = None
        print("Silhouette Score: no se puede calcular (se necesitan al menos 2 clusters).")

    # Gráfica de clusters y ruido.
    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster in sorted(set(labels)):
        selected = labels == cluster
        if cluster == -1:
            ax.scatter(
                X[selected, 0], X[selected, 1],
                c="black", marker="x", s=25, alpha=0.75,
                label=f"Ruido / outliers ({outliers})"
            )
        else:
            ax.scatter(
                X[selected, 0], X[selected, 1],
                s=22, alpha=0.75, label=f"Cluster {cluster}"
            )

    ax.set_title("DBSCAN con KDTree (eps=1.8, min_samples=20)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfica guardada en: {SALIDA.resolve()}")

if __name__ == "__main__":
    main()

