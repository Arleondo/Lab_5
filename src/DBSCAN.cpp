#include "DBSCAN.hpp"
#include "KDTree.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <iomanip>


// Busca todos los puntos dentro del radio eps.
// Incluye al propio punto consultado - explicacion en clase
std::vector<int> regionQuery(const Dataset& dataset,int pointIndex,double eps) {

    std::vector<int> neighbors;
    for (int i = 0; i < static_cast<int>(dataset.size()); i++) {

        double distance = euclideanDistance(dataset[pointIndex],dataset[i]);
        if (distance <= eps) neighbors.push_back(i);

    }

    return neighbors;
}

// Algoritmo DBSCAN clasico sin kd
DBSCANResult runDBSCAN(const Dataset& dataset,double eps,int minSamples) {

    if (!validateDataset(dataset)) throw std::invalid_argument("Dataset invalido.");
    if (eps <= 0.0 || !std::isfinite(eps)) throw std::invalid_argument("eps debe ser positivo y finito.");
    if (minSamples < 1) throw std::invalid_argument("minSamples debe ser >= 1.");

    const int UNVISITED = 0;
    const int NOISE = -1;

    int n = static_cast<int>(dataset.size());

    std::vector<int> labels(n, UNVISITED);
    std::vector<bool> visited(n, false);

    int clusterId = 0;

    // Procesar todos los puntos de manera lineal *(no paralela)
    for (int i = 0; i < n; i++) {

        if (visited[i]) continue;

        visited[i] = true;

        // Buscar vecinos del punto actual.
        std::vector<int> neighbors =
            regionQuery(dataset, i, eps);

        // No es un punto núcleo.
        if (static_cast<int>(neighbors.size()) < minSamples) {
            labels[i] = NOISE;
            continue;
        }

        // Encontramos un nuevo cluster.
        clusterId++;
        labels[i] = clusterId;

        // Lista de puntos pendientes de expandir.
        std::vector<int> seeds = neighbors;
        std::vector<bool> inQueue(n, false);

        for (int neighbor : seeds) inQueue[neighbor] = true;

        // Expandir el cluster.
        for (std::size_t j = 0; j < seeds.size(); j++) {

            int q = seeds[j];

            // Procesar solamente puntos no visitados.
            if (!visited[q]) {

                visited[q] = true;

                std::vector<int> qNeighbors = regionQuery(dataset, q, eps);

                // Si q es punto núcleo, agregar sus vecinos.
                if (static_cast<int>(qNeighbors.size()) >= minSamples) {

                    for (int neighbor : qNeighbors) {

                        if (!inQueue[neighbor]) {
                            seeds.push_back(neighbor);
                            inQueue[neighbor] = true;
                        }
                    }
                }
            }

            // Asignar puntos sin cluster, incluyendo
            // los que inicialmente fueron clasificados como ruido.
            if (labels[q] <= 0) labels[q] = clusterId;

        }
    }

    // Contar puntos clasificados como ruido.
    int noiseCount = 0;

    for (int label : labels) {
        if (label == NOISE) noiseCount++;
    }

    return {labels, clusterId, noiseCount};
}


DBSCANResult runDBSCAN_KDTree(const Dataset& dataset,double eps,int minSamples) {

    if (!validateDataset(dataset)) throw std::invalid_argument("Dataset invalido.");
    if (eps <= 0.0 || !std::isfinite(eps)) throw std::invalid_argument("eps debe ser positivo y finito.");
    if (minSamples < 1) throw std::invalid_argument("minSamples debe ser >= 1.");

    const int UNVISITED = 0;
    const int NOISE = -1;

    int n = static_cast<int>(dataset.size());

    // Construir KDTree una sola vez.
    KDTree tree(dataset);

    std::vector<int> labels(n, UNVISITED);
    std::vector<bool> visited(n, false);

    int clusterId = 0;

    // Procesar todos los puntos.
    for (int i = 0; i < n; i++) {

        if (visited[i]) continue;

        visited[i] = true;

        // Consultar vecinos mediante KDTree.
        std::vector<int> neighbors = tree.queryRadius(dataset[i], eps);

        if (static_cast<int>(neighbors.size()) < minSamples) {
            labels[i] = NOISE;
            continue;
        }

        // Crear nuevo cluster.
        clusterId++;
        labels[i] = clusterId;

        std::vector<int> seeds = neighbors;
        std::vector<bool> inQueue(n, false);

        for (int neighbor : seeds) inQueue[neighbor] = true;

        // Expandir cluster.
        for (std::size_t j = 0; j < seeds.size(); j++) {

            int q = seeds[j];

            if (!visited[q]) {

                visited[q] = true;

                std::vector<int> qNeighbors = tree.queryRadius(dataset[q], eps);

                if (static_cast<int>(qNeighbors.size()) >= minSamples) {

                    for (int neighbor : qNeighbors) {
                        if (!inQueue[neighbor]) {
                            seeds.push_back(neighbor);
                            inQueue[neighbor] = true;
                        }
                    }
                }
            }
            if (labels[q] <= 0) labels[q] = clusterId;
        }
    }

    // Contar ruido.
    int noiseCount = 0;

    for (int label : labels) { if (label == NOISE) noiseCount++; }

    return {labels, clusterId, noiseCount};
}

// Exportar coordenadas originales y etiqueta final de cada punto.
bool saveDBSCANLabelsCSV(const Dataset& dataset,const DBSCANResult& result,const std::string& filename) {

    if (dataset.size() != result.labels.size()) throw std::invalid_argument("La cantidad de etiquetas no coincide con la cantidad de puntos.");
    if (!validateDataset(dataset)) throw std::invalid_argument("Dataset invalido al exportar etiquetas.");

    std::ofstream file(filename.c_str());
    if (!file.is_open()) return false;

    file << std::setprecision(17);

    // Encabezados x1, x2, ..., xN, cluster
    const std::size_t dimensions = dataset[0].size();
    for (std::size_t d = 0; d < dimensions; d++) {
        if (d > 0) file << ",";
        file << "x" << (d + 1);
    }
    file << ",cluster\n";

    for (std::size_t i = 0; i < dataset.size(); i++) {
        for (std::size_t d = 0; d < dimensions; d++) {
            if (d > 0) file << ",";
            file << dataset[i][d];
        }
        file << "," << result.labels[i] << "\n";
    }
}