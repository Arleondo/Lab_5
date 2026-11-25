#ifndef DBSCAN_HPP
#define DBSCAN_HPP

#include "Dataset.hpp"

#include <vector>

struct DBSCANResult {std::vector<int> labels;int clusterCount;int noiseCount;};

// Búsqueda de vecinos dentro del radio eps.
std::vector<int> regionQuery(const Dataset& dataset,int pointIndex,double eps);

// Implementación clásica de DBSCAN.
DBSCANResult runDBSCAN(const Dataset& dataset,double eps,int minSamples);

// DBSCAN utilizando KDTree.
DBSCANResult runDBSCAN_KDTree(const Dataset& dataset,double eps,int minSamples);

// Exporta las coordenadas y la etiqueta de cada punto a un CSV. automatizacion python
// La última columna se llama "cluster"; -1 representa ruido.
bool saveDBSCANLabelsCSV(const Dataset& dataset,const DBSCANResult& result,const std::string& filename);

#endif
