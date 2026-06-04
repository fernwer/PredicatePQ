# PredicatePQ (C++ Prototype)

`PredicatePQ` is a disk-resident hybrid retrieval prototype in C++.

This repository implements core components described in the paper:

* Adaptive planning (`Pre-Filtering` / `Post-Filtering`)
* Cluster-aligned pruning (`ClusterReduce`)
* PQ scan + SIMD on-the-fly transpose
* Batch-oriented refinement (`io_uring` / `mmap`)
* Update path (`delta buffer + tombstone + compaction`)

> **Note:** This is a **single-binary prototype** build.
> There are no `apps/` directories or test scripts.

---

## 1. Project Structure

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
│       ├── types.hpp
│       ├── predicate.hpp
│       ├── scalar_store.hpp
│       ├── metadata.hpp
│       ├── planner.hpp
│       ├── cluster_reduce.hpp
│       ├── pq_scanner.hpp
│       ├── simd_transpose.hpp
│       ├── refiner.hpp
│       ├── build_layout.hpp
│       ├── engine.hpp
│       └── update_manager.hpp
└── src/
    ├── main.cpp
    ├── predicate/predicate.cpp
    ├── scalar/arrow_scalar_store.cpp
    ├── planner/planner.cpp
    ├── core/cluster_reduce.cpp
    ├── core/engine.cpp
    ├── pq/pq_scanner.cpp
    ├── pq/simd_transpose.cpp
    ├── io/refiner_mmap.cpp
    ├── io/refiner_io_uring.cpp
    ├── index/build_layout.cpp
    └── update/update_manager.cpp
```

---

## 2. Dependencies

* CMake >= 3.20
* C++20 compiler

  * GCC >= 11 recommended
  * Clang >= 14 recommended
* OpenMP
* FAISS (CPU)
* Apache Arrow + Parquet
* yaml-cpp
* liburing

  * Optional, required only if enabling the `io_uring` refiner

### Ubuntu Example

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libomp-dev libyaml-cpp-dev liburing-dev \
  libarrow-dev libparquet-dev
```

If FAISS is installed in a non-system path, configure CMake with:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/faiss/install
```

---

## 3. Build

```bash
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPREDICATEPQ_ENABLE_IO_URING=ON
make -j
```

Output binary:

```text
build/predicatepq
```

---

## 4. Run

### Help

```bash
./build/predicatepq --help
```

### Version

```bash
./build/predicatepq --version
```

### Basic Execution

Prototype sanity run:

```bash
./build/predicatepq run --config configs/default.yaml
```

This `run` command performs:

* Config loading from YAML
* Predicate parser/evaluator quick check (`AND` / `OR` / `IN`)
* SIMD transpose quick check

---

## 5. Configuration

Default config file:

```text
configs/default.yaml
```

Key sections:

* `index`
* `planner`
* `engine`
* `refiner`
* `simd`
* `update`

You can tune planner thresholds, probe budget, queue depth, and update/compaction policy in this file.

---

## 6. Notes

If you get the following error:

```text
cannot open include file <yaml-cpp/yaml.h>
```

Install `libyaml-cpp-dev`, then re-run CMake configure:

```bash
sudo apt-get install -y libyaml-cpp-dev
cmake ..
```

Additional notes:

* The `io_uring` path requires Linux kernel/runtime support.
* The current prototype focuses on the core execution path and reproducibility, not production hardening.

---

## 7. License

Research prototype.

Add your preferred license file, such as MIT or Apache-2.0, before public release.
