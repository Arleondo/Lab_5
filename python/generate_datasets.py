import numpy as np
import os

os.makedirs("../datasets", exist_ok=True)

def generate_dataset(n, dimensions, filename):
    rng = np.random.default_rng(42)

    # Generar varios grupos de puntos
    clusters = 4
    points_per_cluster = n // clusters

    data = []

    for i in range(clusters):
        center = rng.uniform(-10, 10, dimensions)

        points = rng.normal(
            loc=center,
            scale=0.5,
            size=(points_per_cluster, dimensions)
        )

        data.append(points)

    dataset = np.vstack(data)

    np.savetxt(
        f"../datasets/{filename}",
        dataset,
        delimiter=",",
        fmt="%.6f"
    )

    print(f"Generado: {filename}")
    print(f"Puntos: {len(dataset)}")
    print(f"Dimensiones: {dimensions}\n")


generate_dataset(1000, 2, "benchmark_1000_2d.csv")
generate_dataset(5000, 2, "benchmark_5000_2d.csv")
generate_dataset(10000, 2, "benchmark_10000_2d.csv")

generate_dataset(1000, 5, "benchmark_1000_5d.csv")
generate_dataset(5000, 5, "benchmark_5000_5d.csv")
generate_dataset(10000, 5, "benchmark_10000_5d.csv")