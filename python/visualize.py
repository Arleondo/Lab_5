from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_file(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    if "cluster" not in df.columns:
        raise ValueError(f"{csv_path}: falta la columna 'cluster'.")

    feature_cols = [c for c in df.columns if c != "cluster"]
    X = df[feature_cols].to_numpy(dtype=float)
    labels = df["cluster"].to_numpy(dtype=int)

    if X.shape[1] == 1:
        raise ValueError("Se necesita un dataset de al menos 2 dimensiones para graficar.")

    if X.shape[1] == 2:
        coords = X
        axis_labels = ("x1", "x2")
        title_suffix = "coordenadas originales"
    else:
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        variance = pca.explained_variance_ratio_.sum() * 100
        axis_labels = ("Componente principal 1", "Componente principal 2")
        title_suffix = f"PCA 2D ({variance:.1f}% varianza explicada)"

    fig, ax = plt.subplots(figsize=(9, 7))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c="black", marker="x", s=22, alpha=0.75,
                       label="Ruido (-1)")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=22, alpha=0.75, label=f"Cluster {label}")

    ax.set_title(f"DBSCAN: {csv_path.stem}\n{title_suffix}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico guardado: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualiza etiquetas DBSCAN.")
    parser.add_argument("csv", nargs="?", help="CSV de etiquetas. Si se omite, procesa results/labels_*.csv")
    parser.add_argument("--output-dir", default="results/figuras")
    args = parser.parse_args()

    files = [Path(args.csv)] if args.csv else sorted(Path("results").glob("labels_*.csv"))
    if not files:
        raise SystemExit("No se encontraron CSV. Ejecuta primero DBSCAN y exporta las etiquetas.")

    for file in files:
        try:
            plot_file(file, Path(args.output_dir))
        except Exception as exc:
            print(f"ERROR procesando {file}: {exc}")

if __name__ == "__main__":
    main()
