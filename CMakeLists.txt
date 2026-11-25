cmake_minimum_required(VERSION 4.1)
project(Laboratorio5_DBSCAN_KDTREE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

set(COMMON_SOURCES
        src/DBSCAN.cpp
        src/KDTree.cpp
        src/Dataset.cpp
)

# Programa interactivo
add_executable(DBSCAN_Laboratory
        src/main.cpp
        ${COMMON_SOURCES}
)

# Programa de benchmarking
add_executable(DBSCAN_Benchmark
        src/benchmark_main.cpp
        ${COMMON_SOURCES}
)