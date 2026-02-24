This repository contains the implementation for the paper: **PredicatePQ: Unifying Scalar and Vector Predicates for Disk-Resident Hybrid Retrieval through Vectorized Execution**.

PredicatePQ is a disk-native hybrid retrieval framework that reimagines vector search as a vectorized scan operator. By unifying scalar filtering and vector similarity computation using SIMD-accelerated primitives, it eliminates the random I/O overhead inherent in graph-based methods for disk-resident scenarios.

## Features Implemented
* **Cost-based Planner (Section 3.2)**: Dynamically selects between Pre-Filtering and Post-Filtering based on predicate selectivity and Laplace smoothed probabilities.
* **Vectorized Cluster Pruning (Section 3.3)**: Implementation of the SIMD-accelerated `ClusterReduce` operator to transform sparse filtering signals into continuous memory access.
* **Batch-Oriented Disk Refinement (Section 3.4)**: Coalescing exact vector distance computations into sequential SSD batch reads.

## Dependencies
* C++17 Compatible Compiler (GCC/Clang)
* CMake >= 3.14
* [Faiss](https://github.com/facebookresearch/faiss)
* OpenMP

## Build Instructions

You can build the project using standard CMake commands:

```bash
mkdir build
cd build
cmake ..
make -j
```

## Running the Demo

After a successful build, you can run the synthetic benchmark demo which simulates both high-selectivity (Pre-Filtering) and low-selectivity (Post-Filtering) hybrid queries:

```bash
./PredicatePQ
```

## Project Structure
* `include/` & `src/`: Core implementation of PredicatePQ (Planner, ClusterReduce, DiskRefiner).
* `main.cpp`: Entry point for generating synthetic data and testing the hybrid retrieval pipeline.
* `CMakeLists.txt`: Build configuration.

