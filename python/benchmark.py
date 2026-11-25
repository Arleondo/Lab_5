import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ajusta esta ruta solo si tu ejecutable se genera en otra carpeta.
EXE = ROOT / "cmake-build-release" / "DBSCAN_Benchmark.exe"

DATASET_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
LABELS_DIR = RESULTS_DIR / "labels_grid"

# Barrido de hiperparametros. Son 5 * 3 * 6 = 90 experimentos.
EPS_VALUES = [0.5, 0.8, 1.0, 1.5, 2.0]
MIN_SAMPLES_VALUES = [3, 5, 10]
REPETITIONS = 10

DATASETS = [
    "benchmark_1000_2d.csv",
    "benchmark_5000_2d.csv",
    "benchmark_10000_2d.csv",
    "benchmark_1000_5d.csv",
    "benchmark_5000_5d.csv",
    "benchmark_10000_5d.csv",
]


def safe_number(value):
    return str(value).replace(".", "p")


def main():
    if not EXE.exists():
        print(f"ERROR: No encuentro el ejecutable:\n{EXE}")
        print("Compila primero el target Laboratorio5_DBSCAN_KDTREE en modo Release.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = RESULTS_DIR / "benchmark_grid_results.csv"
    fields = [
        "dataset", "puntos", "dimensiones", "eps", "min_samples",
        "repeticiones", "classic_ms", "kdtree_ms", "speedup",
        "clusters", "outliers", "labels_equivalent",
        "labels_classic", "labels_kdtree"
    ]

    rows = []
    total = len(DATASETS) * len(EPS_VALUES) * len(MIN_SAMPLES_VALUES)
    current = 0

    for dataset_name in DATASETS:
        dataset_path = DATASET_DIR / dataset_name
        if not dataset_path.exists():
            print(f"\n[AVISO] No existe {dataset_path}. Se omite.")
            continue

        for eps in EPS_VALUES:
            for min_samples in MIN_SAMPLES_VALUES:
                current += 1
                prefix = LABELS_DIR / (
                    f"{Path(dataset_name).stem}"
                    f"_eps{safe_number(eps)}"
                    f"_min{min_samples}"
                )

                command = [
                    str(EXE),
                    str(dataset_path),
                    str(eps),
                    str(min_samples),
                    str(REPETITIONS),
                    str(prefix),
                ]

                print(
                    f"\n[{current}/{total}] {dataset_name} | "
                    f"eps={eps} | min_samples={min_samples}"
                )

                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                if process.returncode != 0:
                    print("[ERROR]", process.stderr.strip() or process.stdout.strip())
                    continue

                result_line = next(
                    (line.strip() for line in process.stdout.splitlines()
                     if line.strip().startswith("RESULT,")),
                    None
                )

                if result_line is None:
                    print("[ERROR] C++ no devolvio una linea RESULT.")
                    print(process.stdout)
                    continue

                values = next(csv.reader([result_line]))
                if len(values) != 15:
                    print(f"[ERROR] Se esperaban 15 campos y llegaron {len(values)}.")
                    print(result_line)
                    continue

                row = dict(zip(fields, values[1:]))
                # El primer campo RESULT no forma parte de fields.
                rows.append(row)

                print(
                    f"  Clasico: {row['classic_ms']} ms | "
                    f"KDTree: {row['kdtree_ms']} ms | "
                    f"Speedup: {row['speedup']}x | "
                    f"Clusters: {row['clusters']} | "
                    f"Ruido: {row['outliers']} | "
                    f"Etiquetas iguales: {row['labels_equivalent']}"
                )

    if not rows:
        print("\nNo se generaron resultados. Revisa datasets y ejecutable.")
        sys.exit(1)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("BARRIDO DE HIPERPARAMETROS FINALIZADO")
    print(f"Experimentos completados: {len(rows)} de {total}")
    print(f"Resultados: {output_csv}")
    print(f"Etiquetas: {LABELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
