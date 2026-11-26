from pathlib import Path
import pandas as pd
from sklearn.metrics import silhouette_score

# Ejecutar desde la carpeta principal:
# python python/calcular_silhouette.py

ARCHIVO = Path("results/labels_output_kdtree_eps1p800000_min20.csv")

def main():
    if not ARCHIVO.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO.resolve()}")

    df = pd.read_csv(ARCHIVO)

    required = {"x1", "x2", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas {sorted(missing)}. "
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    df = df[["x1", "x2", "cluster"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    X = df[["x1", "x2"]].to_numpy()
    labels = df["cluster"].astype(int).to_numpy()

    # DBSCAN identifica los puntos de ruido con la etiqueta -1.
    mask = labels != -1
    X_clustered = X[mask]
    labels_clustered = labels[mask]

    outliers = int((labels == -1).sum())
    unique_clusters = sorted(set(labels_clustered))
    n_clusters = len(unique_clusters)

    print(f"Archivo: {ARCHIVO}")
    print(f"Puntos válidos: {len(df)}")
    print(f"Outliers excluidos: {outliers}")
    print(f"Clusters considerados: {n_clusters}")

    if n_clusters < 2:
        print(
            "No se puede calcular Silhouette Score: "
            "se necesitan al menos 2 clusters (sin contar el ruido)."
        )
        return

    if len(labels_clustered) <= n_clusters:
        print("No se puede calcular Silhouette Score: hay muy pocos puntos.")
        return

    score = silhouette_score(X_clustered, labels_clustered)
    print(f"Silhouette Score (excluyendo ruido): {score:.6f}")

if __name__ == "__main__":
    main()
