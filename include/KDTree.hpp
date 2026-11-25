#ifndef KDTREE_HPP
#define KDTREE_HPP

#include "Dataset.hpp"

#include <memory>
#include <vector>

class KDTree {
    
    struct Node {
        int pointIndex;
        int axis;

        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;

        Node(int index, int splitAxis): pointIndex(index), axis(splitAxis) {}
    };

    const Dataset* dataset;
    std::unique_ptr<Node> root;

    // Construccion recursiva.
    std::unique_ptr<Node> build(std::vector<int>& indices,int depth);

    // Busqueda recursiva por radio.
    void rangeSearch(const Node* node,const Point& query,double radius,std::vector<int>& neighbors) const;

public:

    explicit KDTree(const Dataset& data);

    // Devuelve indices de puntos dentro del radio.
    std::vector<int> queryRadius(const Point& query,double radius) const;
};

#endif
