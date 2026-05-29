# PredicatePQ C++ Prototype

`PredicatePQ` is a disk-resident hybrid retrieval prototype implementing:

* Adaptive plan selection: Pre-Filtering / Post-Filtering
* Cluster-aligned pruning: `ClusterReduce`
* PQ scan with SIMD-friendly on-the-fly transpose
* Batch-oriented SSD refinement: `io_uring` / `mmap`
* Update path: delta buffer + tombstone + compaction

This is a **system prototype**, not a FAISS-only wrapper. The hybrid execution path, planner, cluster operators, and refinement scheduler are implemented in this project.

---

## 1. Project Layout

```text
predicatepq/
├── CMakeLists.txt
├── README.md
├── cmake/
│   └── FindFAISS.cmake
├── configs/
│   └── default.yaml
├── include/
│   └── predicatepq/
│       └── ...
├── src/
│   └── ...
├── apps/
│   ├── build_index.cpp
│   ├── query_cli.cpp
│   ├── benchmark.cpp
│   └── microbench_transpose.cpp
└── tests/
    └── ...
```

---

## 2. Dependencies

Required dependencies:

* C++20 compiler

  * GCC >= 11 is recommended
* CMake >= 3.20
* OpenMP
* FAISS CPU
* Apache Arrow C++
* Parquet C++
* yaml-cpp
* liburing

  * Optional, but recommended for SSD refinement

Ubuntu example packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libomp-dev libyaml-cpp-dev liburing-dev \
  libarrow-dev libparquet-dev
```

FAISS can be installed from source or through a package manager.

If CMake cannot find FAISS automatically, set `CMAKE_PREFIX_PATH` manually:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH=/path/to/faiss/install
```

---

## 3. Build

Create a build directory and configure the project:

```bash
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPREDICATEPQ_ENABLE_IO_URING=ON \
  -DPREDICATEPQ_ENABLE_TESTS=OFF
```

Build the binaries:

```bash
make -j
```

Generated binaries include:

* `build_index`
* `query_cli`
* `benchmark`
* `microbench_transpose`

---

## 4. Build Index

The input vector file format is raw `float32` binary.

Expected shape:

```text
[ntotal * dim]
```

The vectors should be stored in row-major order.

Command format:

```bash
./build_index \
  <vec_fbin> <out_dir> <ntotal> <dim> \
  <nlist> <pq_m> <pq_nbits> <train_n> \
  <scalar_parquet_or_->
```

Example:

```bash
./build_index \
  ./data/base.fbin ./out_index \
  1000000 128 \
  4096 16 8 200000 \
  ./data/scalar.parquet
```

Output artifacts include:

* `ivfpq.faiss`
* `pq_codes_rowwise.bin`
* `vectors_clustered.fbin`
* `id_to_cluster.bin`
* `id_to_disk_offset.bin`
* `coarse_centroids.fbin`
* `cluster_offsets.bin`
* `scalar_original.arrow`
* `scalar_clustered.arrow`

`scalar_original.arrow` and `scalar_clustered.arrow` are generated only when a valid Parquet scalar file is provided.

---

## 5. Query

Command format:

```bash
./query_cli \
  <index_dir> <query_fbin> <qid> <topk> <use_io_uring:0|1> "<predicate_sql>"
```

Example:

```bash
./query_cli \
  ./out_index ./data/query.fbin \
  0 100 1 \
  "price >= 100 AND category IN ('shoe','coat')"
```

Supported predicate operators in the scalar engine:

| Category       | Operators                       |
| -------------- | ------------------------------- |
| Comparison     | `=`, `!=`, `<`, `<=`, `>`, `>=` |
| Set membership | `IN (...)`                      |
| Logical        | `AND`, `OR`                     |
| Grouping       | `( ... )`                       |

---

## 6. Configuration

Runtime defaults are provided in:

```text
configs/default.yaml
```

The configuration file includes defaults for:

* Planner behavior
* PQ scan behavior
* Refinement scheduling
* SIMD-related options
* Update behavior
* I/O options

Users can modify this file to tune query execution, scan parameters, refinement strategy, and update-path behavior.

---

## 7. Notes

The `io_uring` refiner currently defaults to buffered I/O.

To enable direct I/O, explicitly configure `use_o_direct` and ensure that alignment constraints are correctly handled.

For best performance on NUMA machines:

* Pin worker threads.
* Place data on local NUMA nodes.
* Avoid cross-socket memory access when possible.

This prototype focuses on the retrieval path and reproducible systems behavior, including:

* Hybrid predicate-vector query execution
* Planner-controlled pre-filtering and post-filtering
* Cluster-aware pruning
* PQ scan optimization
* SSD-based vector refinement
* Update handling through delta buffer, tombstones, and compaction
