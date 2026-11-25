#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <limits>

#include "Dataset.hpp"
#include "DBSCAN.hpp"

using namespace std;
using namespace chrono;
namespace fs = std::filesystem;

// ================= CONFIGURACION =================

struct Config {
    string datasetPath = "";
    double eps = 1.5;
    int minSamples = 5;
    int repetitions = 10;
};

struct BenchmarkResult {
    double classicTime = 0.0;
    double kdTreeTime = 0.0;
    int clusters = 0;
    int outliers = 0;
};

// ================= UTILIDADES =================

void clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void pauseScreen() {
    cout << "\nPresione ENTER para continuar...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();
}

void printLine() {
    cout << "==================================================\n";
}


// ================= EXPORTACION DE ETIQUETAS =================

// Genera nombres distintos para cada dataset y configuracion.
string makeLabelsFilename(const Config& config, const string& algorithm) {
    string epsText = to_string(config.eps);
    replace(epsText.begin(), epsText.end(), '.', 'p');

    string datasetName = fs::path(config.datasetPath).stem().string();

    return "results/labels_" + datasetName + "_" + algorithm +
           "_eps" + epsText +
           "_min" + to_string(config.minSamples) + ".csv";
}

void exportLabels(const Config& config,
                  const Dataset& dataset,
                  const DBSCANResult& result,
                  const string& algorithm) {
    fs::create_directories("results");

    string filename = makeLabelsFilename(config, algorithm);

    if (saveDBSCANLabelsCSV(dataset, result, filename)) {
        cout << "Etiquetas exportadas: " << filename << "\\n";
    } else {
        cout << "No se pudo exportar el archivo: " << filename << "\\n";
    }
}

void printHeader() {
    clearScreen();

    cout << "\n";
    printLine();
    cout << "             DBSCAN LABORATORY v1.0\n";
    cout << "              C++17 + KDTree\n";
    printLine();
}

int readInt(const string& message, int minValue, int maxValue) {
    int value;

    while (true) {
        cout << message;

        if (cin >> value && value >= minValue && value <= maxValue) {
            return value;
        }

        cout << "Entrada invalida. Intente nuevamente.\n";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
}

double readDouble(const string& message, double minValue) {
    double value;

    while (true) {
        cout << message;

        if (cin >> value && value >= minValue && value <= 1e100) {
            return value;
        }

        cout << "Valor invalido. Intente nuevamente.\n";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }
}

// ================= DATASETS =================

vector<string> listDatasets() {
    vector<string> files;

    fs::path directory = "datasets";

    if (!fs::exists(directory)) {
        return files;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".csv") {
            files.push_back(entry.path().string());
        }
    }

    sort(files.begin(), files.end());

    return files;
}

bool selectDataset(Config& config) {
    vector<string> files = listDatasets();

    if (files.empty()) {
        cout << "\nNo se encontraron archivos CSV en datasets/.\n";
        return false;
    }

    cout << "\n============== DATASETS DISPONIBLES ==============\n\n";

    for (size_t i = 0; i < files.size(); i++) {
        cout << "[" << i + 1 << "] "
             << fs::path(files[i]).filename().string() << "\n";
    }

    cout << "[0] Cancelar\n\n";

    int option = readInt("Seleccione un dataset: ", 0,
                         static_cast<int>(files.size()));

    if (option == 0) {
        return false;
    }

    string selected = files[option - 1];

    try {
        Dataset dataset = loadCSV(selected);

        config.datasetPath = selected;

        cout << "\nDataset cargado correctamente.\n";
        cout << "Archivo: " << selected << "\n";
        cout << "Puntos: " << dataset.size() << "\n";
        cout << "Dimensiones: " << getDimensions(dataset) << "\n";

        return true;

    } catch (const exception& e) {
        cout << "\nError al cargar dataset: " << e.what() << "\n";
        return false;
    }
}

// ================= BENCHMARK =================

BenchmarkResult runBenchmark(const Config& config) {
    Dataset dataset = loadCSV(config.datasetPath);

    // Calentamiento: no se incluye en las mediciones.
    runDBSCAN(dataset, config.eps, config.minSamples);
    runDBSCAN_KDTree(dataset, config.eps, config.minSamples);

    BenchmarkResult result;

    double totalClassic = 0.0;
    double totalKDTree = 0.0;

    DBSCANResult classic;
    DBSCANResult optimized;

    for (int i = 0; i < config.repetitions; i++) {

        auto startClassic = steady_clock::now();

        classic = runDBSCAN(
            dataset, config.eps, config.minSamples
        );

        auto endClassic = steady_clock::now();

        auto startKDTree = steady_clock::now();

        optimized = runDBSCAN_KDTree(
            dataset, config.eps, config.minSamples
        );

        auto endKDTree = steady_clock::now();

        totalClassic += duration<double, milli>(
            endClassic - startClassic
        ).count();

        totalKDTree += duration<double, milli>(
            endKDTree - startKDTree
        ).count();
    }

    result.classicTime = totalClassic / config.repetitions;
    result.kdTreeTime = totalKDTree / config.repetitions;
    result.clusters = classic.clusterCount;
    result.outliers = classic.noiseCount;

    // Exportar las etiquetas de la ultima repeticion.
    // La escritura ocurre despues de las mediciones, por lo que no altera los tiempos.
    exportLabels(config, dataset, classic, "classic");
    exportLabels(config, dataset, optimized, "kdtree");

    return result;
}

void showResult(const Config& config, const BenchmarkResult& result) {
    printLine();
    cout << "                 RESULTADOS\n";
    printLine();

    cout << fixed << setprecision(6);

    cout << "Dataset: " << config.datasetPath << "\n";
    cout << "Epsilon: " << config.eps << "\n";
    cout << "Min Samples: " << config.minSamples << "\n";
    cout << "Repeticiones: " << config.repetitions << "\n\n";

    cout << "DBSCAN clasico: " << result.classicTime << " ms\n";
    cout << "DBSCAN KDTree:  " << result.kdTreeTime << " ms\n";

    cout << "\nClusters: " << result.clusters << "\n";
    cout << "Outliers: " << result.outliers << "\n";

    if (result.kdTreeTime > 0.0) {
        double speedup = result.classicTime / result.kdTreeTime;

        cout << "\nSpeedup: " << speedup << "x\n";

        if (speedup > 1.0) {
            cout << "KDTree fue mas rapido.\n";
        } else {
            cout << "DBSCAN clasico fue mas rapido.\n";
        }
    }
}

void saveResult(const Config& config, const BenchmarkResult& result) {
    fs::create_directories("results");

    const string filename = "results/benchmark_interactivo.csv";

    bool writeHeader = !fs::exists(filename) ||
                       fs::file_size(filename) == 0;

    ofstream file(filename, ios::app);

    if (!file.is_open()) {
        cout << "\nNo se pudo guardar el resultado.\n";
        return;
    }

    if (writeHeader) {
        file << "dataset,puntos,dimensiones,eps,min_samples,"
                "repeticiones,classic_ms,kdtree_ms,clusters,outliers\n";
    }

    Dataset dataset = loadCSV(config.datasetPath);

    file << fs::path(config.datasetPath).filename().string() << ","
         << dataset.size() << ","
         << getDimensions(dataset) << ","
         << config.eps << ","
         << config.minSamples << ","
         << config.repetitions << ","
         << result.classicTime << ","
         << result.kdTreeTime << ","
         << result.clusters << ","
         << result.outliers << "\n";

    cout << "\nResultado guardado en " << filename << "\n";
}

// ================= EJECUCION =================

void executeComparison(const Config& config) {
    if (config.datasetPath.empty()) {
        cout << "\nPrimero debe seleccionar un dataset.\n";
        return;
    }

    cout << "\nEjecutando comparacion...\n";

    try {
        BenchmarkResult result = runBenchmark(config);

        showResult(config, result);
        saveResult(config, result);

    } catch (const exception& e) {
        cout << "\nError: " << e.what() << "\n";
    }
}

void executeClassic(const Config& config) {
    if (config.datasetPath.empty()) {
        cout << "\nPrimero debe seleccionar un dataset.\n";
        return;
    }

    try {
        Dataset dataset = loadCSV(config.datasetPath);

        auto start = steady_clock::now();

        DBSCANResult result = runDBSCAN(
            dataset, config.eps, config.minSamples
        );

        auto end = steady_clock::now();

        exportLabels(config, dataset, result, "classic");

        cout << "\nDBSCAN CLASICO\n";
        cout << "Clusters: " << result.clusterCount << "\n";
        cout << "Outliers: " << result.noiseCount << "\n";
        cout << "Tiempo: "
             << duration<double, milli>(end - start).count()
             << " ms\n";

    } catch (const exception& e) {
        cout << "\nError: " << e.what() << "\n";
    }
}

void executeKDTree(const Config& config) {
    if (config.datasetPath.empty()) {
        cout << "\nPrimero debe seleccionar un dataset.\n";
        return;
    }

    try {
        Dataset dataset = loadCSV(config.datasetPath);

        auto start = steady_clock::now();

        DBSCANResult result = runDBSCAN_KDTree(
            dataset, config.eps, config.minSamples
        );

        auto end = steady_clock::now();

        exportLabels(config, dataset, result, "kdtree");

        cout << "\nDBSCAN CON KDTREE\n";
        cout << "Clusters: " << result.clusterCount << "\n";
        cout << "Outliers: " << result.noiseCount << "\n";
        cout << "Tiempo: "
             << duration<double, milli>(end - start).count()
             << " ms\n";

    } catch (const exception& e) {
        cout << "\nError: " << e.what() << "\n";
    }
}

// ================= MENU =================

void showMenu(const Config& config) {
    printHeader();

    cout << "\nCONFIGURACION ACTUAL\n";
    printLine();

    cout << "Dataset: "
         << (config.datasetPath.empty()
             ? "Ninguno seleccionado"
             : fs::path(config.datasetPath).filename().string())
         << "\n";

    cout << "Epsilon: " << config.eps << "\n";
    cout << "Min Samples: " << config.minSamples << "\n";
    cout << "Repeticiones: " << config.repetitions << "\n";

    printLine();

    cout << "\nMENU PRINCIPAL\n\n";
    cout << "[1] Seleccionar dataset\n";
    cout << "[2] Cambiar Epsilon\n";
    cout << "[3] Cambiar Min Samples\n";
    cout << "[4] Cambiar repeticiones\n";
    cout << "[5] Ejecutar DBSCAN clasico\n";
    cout << "[6] Ejecutar DBSCAN KDTree\n";
    cout << "[7] Comparar ambos algoritmos\n";
    cout << "[8] Salir\n\n";
}

int main() {
    Config config;
    bool running = true;

    while (running) {
        showMenu(config);

        int option = readInt("Seleccione una opcion: ", 1, 8);

        switch (option) {

            case 1:
                selectDataset(config);
                pauseScreen();
                break;

            case 2:
                config.eps = readDouble(
                    "\nNuevo Epsilon: ", 0.000001
                );
                break;

            case 3:
                config.minSamples = readInt(
                    "\nNuevo Min Samples: ", 1, 1000000
                );
                break;

            case 4:
                config.repetitions = readInt(
                    "\nNumero de repeticiones: ", 1, 1000
                );
                break;

            case 5:
                executeClassic(config);
                pauseScreen();
                break;

            case 6:
                executeKDTree(config);
                pauseScreen();
                break;

            case 7:
                executeComparison(config);
                pauseScreen();
                break;

            case 8:
                running = false;
                cout << "\nGracias por utilizar DBSCAN Laboratory!\n";
                break;
        }
    }

    return 0;
}
