#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

#include "Dataset.hpp"
#include "DBSCAN.hpp"

using namespace std;
using namespace chrono;
namespace fs = std::filesystem;

static void saveLabelsCSV(
    const Dataset& dataset,
    const DBSCANResult& result,
    const string& outputPath
) {
    if (dataset.size() != result.labels.size()) {
        throw runtime_error("La cantidad de etiquetas no coincide con el dataset.");
    }

    fs::path path(outputPath);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }

    ofstream file(outputPath);
    if (!file.is_open()) {
        throw runtime_error("No se pudo crear el archivo de etiquetas: " + outputPath);
    }

    const size_t dimensions = getDimensions(dataset);

    for (size_t d = 0; d < dimensions; ++d) {
        if (d > 0) file << ",";
        file << "x" << (d + 1);
    }
    file << ",cluster\n";

    file << setprecision(17);
    for (size_t i = 0; i < dataset.size(); ++i) {
        for (size_t d = 0; d < dimensions; ++d) {
            if (d > 0) file << ",";
            file << dataset[i][d];
        }
        file << "," << result.labels[i] << "\n";
    }
}

static double measureClassic(const Dataset& data, double eps, int minSamples) {
    const auto start = steady_clock::now();
    DBSCANResult result = runDBSCAN(data, eps, minSamples);
    const auto end = steady_clock::now();
    return duration<double, milli>(end - start).count();
}

static double measureKDTree(const Dataset& data, double eps, int minSamples) {
    const auto start = steady_clock::now();
    DBSCANResult result = runDBSCAN_KDTree(data, eps, minSamples);
    const auto end = steady_clock::now();
    return duration<double, milli>(end - start).count();
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 6) {
            cerr << "Uso: benchmark_main <dataset.csv> <eps> <min_samples> <repeticiones> <prefijo_salida>\n";
            return 2;
        }

        const string datasetPath = argv[1];
        const double eps = stod(argv[2]);
        const int minSamples = stoi(argv[3]);
        const int repetitions = stoi(argv[4]);
        const string outputPrefix = argv[5];

        if (!isfinite(eps) || eps <= 0.0 || minSamples < 1 || repetitions < 1) {
            throw invalid_argument("eps, min_samples o repeticiones tienen valores invalidos.");
        }

        Dataset dataset = loadCSV(datasetPath);
        if (dataset.empty()) {
            throw runtime_error("El dataset esta vacio.");
        }

        // Calentamiento fuera de las mediciones.
        DBSCANResult classicLabels = runDBSCAN(dataset, eps, minSamples);
        DBSCANResult kdLabels = runDBSCAN_KDTree(dataset, eps, minSamples);

        double totalClassic = 0.0;
        double totalKDTree = 0.0;

        for (int i = 0; i < repetitions; ++i) {
            if (i % 2 == 0) {
                totalClassic += measureClassic(dataset, eps, minSamples);
                totalKDTree += measureKDTree(dataset, eps, minSamples);
            } else {
                totalKDTree += measureKDTree(dataset, eps, minSamples);
                totalClassic += measureClassic(dataset, eps, minSamples);
            }
        }

        const double classicMs = totalClassic / repetitions;
        const double kdTreeMs = totalKDTree / repetitions;
        const double speedup = kdTreeMs > 0.0 ? classicMs / kdTreeMs : 0.0;
        const bool sameLabels = classicLabels.labels == kdLabels.labels;

        const string classicPath = outputPrefix + "_classic.csv";
        const string kdPath = outputPrefix + "_kdtree.csv";
        saveLabelsCSV(dataset, classicLabels, classicPath);
        saveLabelsCSV(dataset, kdLabels, kdPath);

        cout << fixed << setprecision(6);
        cout << "RESULT,"
             << fs::path(datasetPath).filename().string() << ","
             << dataset.size() << ","
             << getDimensions(dataset) << ","
             << eps << ","
             << minSamples << ","
             << repetitions << ","
             << classicMs << ","
             << kdTreeMs << ","
             << speedup << ","
             << classicLabels.clusterCount << ","
             << classicLabels.noiseCount << ","
             << (sameLabels ? 1 : 0) << ","
             << classicPath << ","
             << kdPath << "\n";

        return 0;
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
