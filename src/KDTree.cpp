#include "KDTree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

KDTree::KDTree(const Dataset& data) : dataset(&data) {

    if (!validateDataset(data)) throw std::invalid_argument("Dataset invalido para KDTree.");

    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    root = build(indices, 0);
}

// Construccion recursiva del KDTree.
std::unique_ptr<KDTree::Node> KDTree::build(std::vector<int>& indices,int depth) {

    if (indices.empty()) return nullptr;

    int dimensions = static_cast<int>((*dataset)[0].size());

    int axis = depth % dimensions;

    std::size_t median = indices.size() / 2;

    std::nth_element(
        indices.begin(),
        indices.begin() + median,
        indices.end(),
        [&](int a, int b) {return (*dataset)[a][axis] < (*dataset)[b][axis];}
    );

    int medianIndex = indices[median];

    auto node = std::make_unique<Node>(medianIndex, axis);

    std::vector<int> leftIndices(indices.begin(),indices.begin() + median);
    std::vector<int> rightIndices(indices.begin() + median + 1,indices.end());

    node->left = build(leftIndices, depth + 1);
    node->right = build(rightIndices, depth + 1);

    return node;
}

// Consulta de rango por radio.
void KDTree::rangeSearch(const Node* node,const Point& query,double radius,std::vector<int>& neighbors) const {

    if (node == nullptr) return;

    const Point& currentPoint = (*dataset)[node->pointIndex];

    double distance = euclideanDistance(query,currentPoint);

    if (distance <= radius) neighbors.push_back(node->pointIndex);

    int axis = node->axis;

    double difference = query[axis] - currentPoint[axis];

    // Explorar el lado donde se encuentra la consulta.
    if (difference <= 0.0) {

        rangeSearch(node->left.get(),query,radius,neighbors);

        // Explorar el otro lado si puede contener vecinos.
        if (std::abs(difference) <= radius) {
            rangeSearch(node->right.get(),query,radius,neighbors);
        }

    } else {

        rangeSearch(node->right.get(),query,radius,neighbors);

        if (std::abs(difference) <= radius) {
            rangeSearch(node->left.get(),query,radius,neighbors);
        }
    }
}

// Interfaz publica de consulta.
std::vector<int> KDTree::queryRadius(const Point& query,double radius) const {

    if (query.size() != (*dataset)[0].size()) throw std::invalid_argument("Dimensiones incompatibles.");
    if (radius < 0.0 || !std::isfinite(radius)) throw std::invalid_argument("Radio invalido.");

    std::vector<int> neighbors;
    rangeSearch(root.get(),query,radius,neighbors);

    return neighbors;
}
