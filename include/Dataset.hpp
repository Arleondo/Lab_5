#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <cstddef>
#include <string>

using Point = std::vector<double>;
using Dataset = std::vector<Point>;

// Calcula la distancia euclidiana entre dos puntos.
double euclideanDistance(const Point& a, const Point& b);

// Comprueba que todos los puntos tengan las mismas dimensiones.
bool validateDataset(const Dataset& dataset);

// Devuelve la cantidad de dimensiones del dataset.
std::size_t getDimensions(const Dataset& dataset);

// Nueva funcion: cargar dataset desde CSV.
Dataset loadCSV(const std::string& filename);

#endif