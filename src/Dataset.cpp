#include "Dataset.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <fstream>

double euclideanDistance(const Point& a, const Point& b) {

    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Los puntos tienen dimensiones diferentes."
        );
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); i++) {
        double difference = a[i] - b[i];
        sum += difference * difference;
    }

    return std::sqrt(sum);
}

bool validateDataset(const Dataset& dataset) {

    if (dataset.empty()) {
        return false;
    }

    std::size_t dimensions = dataset[0].size();

    if (dimensions == 0) return false;

    for (const Point& point : dataset) {

        if (point.size() != dimensions) return false;

        for (double value : point) {
            if (!std::isfinite(value)) return false;
        }
    }

    return true;
}

std::size_t getDimensions(const Dataset& dataset) {
    if (dataset.empty()) return 0;
    return dataset[0].size();
}

Dataset loadCSV(const std::string& filename) {

    std::ifstream file(filename);

    if (!file.is_open()) throw std::runtime_error("No se pudo abrir el archivo: " + filename);


    Dataset dataset;
    std::string line;

    std::size_t expectedDimensions = 0;
    std::size_t lineNumber = 0;

    while (std::getline(file, line)) {

        lineNumber++;

        // Ignorar lineas vacias.
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;


        std::stringstream ss(line);
        std::string value;

        Point point;

        while (std::getline(ss, value, ',')) {

            // Eliminar espacios alrededor del valor.
            std::size_t first = value.find_first_not_of(" \t\r");
            std::size_t last = value.find_last_not_of(" \t\r");

            if (first == std::string::npos) throw std::runtime_error("Valor vacio en linea " +std::to_string(lineNumber));

            value = value.substr(first, last - first + 1);

            try {
                std::size_t processed = 0;

                double number = std::stod(value, &processed);
                if (processed != value.size() || !std::isfinite(number)) throw std::runtime_error("Valor numerico invalido.");

                point.push_back(number);

            } catch (const std::exception&) {
                throw std::runtime_error("Dato invalido en linea " +std::to_string(lineNumber) +": " + value);
            }
        }

        if (expectedDimensions == 0) {
            expectedDimensions = point.size();
        }

        if (point.size() != expectedDimensions) {
            throw std::runtime_error(
                "Dimensiones inconsistentes en linea " +
                std::to_string(lineNumber)
            );
        }

        dataset.push_back(point);
    }

    file.close();

    if (!validateDataset(dataset)) throw std::runtime_error("El dataset esta vacio o contiene datos invalidos.");


    return dataset;
}