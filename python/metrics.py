


from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def evaluate_file(path: Path):
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError(f"{path}: falta la columna 'cluster'.")

    feature_cols = [c for c in df.columns if c != "cluster"]
    if not feature_cols:
        raise ValueError(f"{path}: no hay columnas de coordenadas.")

    X = df[feature_cols].to_numpy(dtype=float)
    labels = df["cluster"].to_numpy(dtype=int)

    noise_count = int((labels == -1).sum())
    valid = labels != -1
    X_valid = X[valid]
    labels_valid = labels[valid]
    clusters = sorted(set(labels_valid))
    cluster_count = len(clusters)

    score = None
    note = ""
    if cluster_count >= 2 and len(labels_valid) > cluster_count:
        X_scaled = StandardScaler().fit_transform(X_valid)
        score = float(silhouette_score(X_scaled, labels_valid))
    else:
        note = "No calculable: se necesitan al menos 2 clusters y más muestras que clusters."

    return {
        "archivo": path.name,
        "puntos": len(df),
        "dimensiones": len(feature_cols),
        "clusters": cluster_count,
        "ruido": noise_count,
        "porcentaje_ruido": round(100 * noise_count / len(df), 3) if len(df) else 0,
        "silhouette": round(score, 6) if score is not None else "",
        "observacion": note
    }

def main():
    parser = argparse.ArgumentParser(description="Calcula métricas de resultados DBSCAN.")
    parser.add_argument("csv", nargs="?", help="CSV de etiquetas. Si se omite, procesa results/labels_*.csv")
    parser.add_argument("--output", default="results/metrics_results.csv")
    args = parser.parse_args()

    files = [Path(args.csv)] if args.csv else sorted(Path("results").glob("labels_*.csv"))
    if not files:
        raise SystemExit("No se encontraron CSV. Ejecuta primero DBSCAN y exporta las etiquetas.")

    rows = []
    for file in files:
        try:
            row = evaluate_file(file)
            rows.append(row)
            print(f"\n{row['archivo']}")
            print(f"  Puntos: {row['puntos']} | Dimensiones: {row['dimensiones']}")
            print(f"  Clusters: {row['clusters']} | Ruido: {row['ruido']} ({row['porcentaje_ruido']}%)")
            print(f"  Silhouette: {row['silhouette'] if row['silhouette'] != '' else row['observacion']}")
        except Exception as exc:
            print(f"ERROR procesando {file}: {exc}")

    if rows:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"\nResumen guardado en: {output}")

if __name__ == "__main__":
    main()

