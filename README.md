# PredicatePQ

`PredicatePQ` is a C++ research prototype for **disk-resident hybrid vector retrieval** over vector data with scalar predicates.

PredicatePQ organizes predicate evaluation, PQ-based similarity computation, and full-vector SSD refinement around **IVF clusters as shared physical execution units**. The implementation preserves cluster identity across the retrieval pipeline so that sparse predicate results can be converted into regular CPU batches and schedulable SSD requests.

This repository accompanies the paper:

> **PredicatePQ: Execution-Regularized Hybrid Retrieval over Disk-Resident Vector Data**

---

## 1. Implemented Components

The current prototype implements the main execution mechanisms described in the paper:

- Adaptive execution planning
  - `Pre-Filtering`
  - `Post-Filtering`
  - cluster-stratified selectivity estimation
- Cluster-aligned candidate materialization
  - `ClusterReduce`
  - histogram / prefix-sum / scatter organization
- IVFPQ-based approximate scoring
  - FAISS IVFPQ index construction
  - PQ scanning
  - SIMD on-the-fly PQ-code transposition
- Batch-oriented SSD refinement
  - cluster-local candidate ordering
  - request coalescing
  - asynchronous `io_uring`
  - `mmap` fallback path
- Scalar data processing
  - Apache Arrow columnar storage
  - vectorized predicate evaluation
- Lightweight update support
  - append-oriented delta buffer
  - tombstone-based logical deletion
  - periodic compaction / retraining path

> **Prototype scope**
>
> This repository is a research prototype intended to expose the execution path evaluated in the paper. It is not a production-hardened vector database. Update-heavy online reclustering and production distributed coordination are outside the evaluated path.

---

## 2. Project Structure

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
    ├── predicate/
    │   └── predicate.cpp
    ├── scalar/
    │   └── arrow_scalar_store.cpp
    ├── planner/
    │   └── planner.cpp
    ├── core/
    │   ├── cluster_reduce.cpp
    │   └── engine.cpp
    ├── pq/
    │   ├── pq_scanner.cpp
    │   └── simd_transpose.cpp
    ├── io/
    │   ├── refiner_mmap.cpp
    │   └── refiner_io_uring.cpp
    ├── index/
    │   └── build_layout.cpp
    └── update/
        └── update_manager.cpp
```

The repository uses a **single-binary prototype** layout rather than separate `apps/` or benchmark binaries.

---

## 3. Dependencies

### Required

- Linux
- CMake >= 3.20
- C++20 compiler
  - GCC >= 11 recommended
  - Clang >= 14 recommended
- OpenMP
- FAISS CPU
- Apache Arrow
- Apache Parquet
- yaml-cpp

### Optional

- `liburing`
  - required when the `io_uring` refinement path is enabled

### Ubuntu Example

```bash
sudo apt-get update

sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  libomp-dev \
  libyaml-cpp-dev \
  liburing-dev \
  libarrow-dev \
  libparquet-dev
```

FAISS can be installed system-wide or supplied through `CMAKE_PREFIX_PATH`.

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH=/path/to/faiss/install
```

---

## 4. Build

```bash
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPREDICATEPQ_ENABLE_IO_URING=ON

make -j
```

The resulting binary is:

```text
build/predicatepq
```

To disable the `io_uring` implementation:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPREDICATEPQ_ENABLE_IO_URING=OFF
```

---

## 5. Quick Start

### Help

```bash
./build/predicatepq --help
```

### Version

```bash
./build/predicatepq --version
```

### Prototype Sanity Run

```bash
./build/predicatepq run \
  --config configs/default.yaml
```

The current sanity path checks:

- YAML configuration loading
- predicate parsing and evaluation
- logical operators such as `AND`, `OR`, and `IN`
- SIMD transpose support
- initialization of the core PredicatePQ execution components

The benchmark configuration used in the paper is documented in the following sections.

---

## 6. Execution Pipeline

PredicatePQ follows a three-stage **Plan--Execute--Refine** pipeline.

### 6.1 Selectivity Estimation and Planning

Scalar predicates are sampled at IVF-cluster granularity.

For cluster `i`:

```text
p_i = m_i / n_i
```

where:

- `n_i` is the number of sampled objects in cluster `i`
- `m_i` is the number satisfying the predicate

Cluster-local estimates use Laplace smoothing:

```text
delta_i = (m_i + 1) / (n_i + 2)
```

Global selectivity is population weighted:

```text
delta_global =
    sum_i |C_i| * p_i
    -----------------
       sum_i |C_i|
```

This weighting is important because adaptive sampling can use different sampling fractions for clusters of different sizes.

The implementation uses:

```text
sampling ratio:        approximately 1%--2%
minimum samples:       64 per sufficiently large cluster
```

Small clusters below the minimum size can be scanned directly.

The planner selects between:

```text
Pre-Filtering
Post-Filtering
```

using estimated predicate selectivity, expected PQ work, and expected refinement cost.

### 6.2 ClusterReduce

For Pre-Filtering, predicate-qualified global IDs are regrouped by IVF cluster.

`ClusterReduce` performs:

1. cluster histogram construction
2. prefix-sum computation
3. cluster-local ID materialization

The output representation contains:

```text
Counts      valid objects per cluster
Offsets     start offset of each cluster segment
IDs_out     valid IDs grouped by cluster
```

This representation is reused by both PQ scoring and SSD refinement rather than being reconstructed at every stage.

The implementation considers clusters in small execution batches, typically:

```text
16 or 32 clusters
```

before passing dense cluster-local work to the SIMD PQ path.

### 6.3 Post-Filtering / Stat-Guided Pruning

For Post-Filtering, PredicatePQ avoids materializing the complete predicate result first.

For cluster `i`, the expected valid population is estimated as:

```text
V_i = |C_i| * delta_i
```

Clusters are ranked using both:

- query-to-centroid distance
- estimated valid population

A geometric safety floor retains approximately:

```text
16--32 nearest clusters
```

even when predicate statistics estimate low valid density.

### 6.4 SIMD PQ Scan

PQ codes are stored row-wise for efficient ID-based candidate gathering.

During scoring, the implementation performs on-the-fly SIMD transposition inside registers to provide a FastScan-style layout without maintaining a persistent second copy of PQ codes.

```text
persistent layout:    row-wise PQ codes
compute layout:       SIMD-oriented transient representation
```

### 6.5 SSD Refinement

Full-precision vectors remain on SSD.

After PQ scoring, shortlisted candidates retain:

```text
cluster ID
physical offset
```

Candidates are ordered by cluster and physical offset before refinement. Nearby candidates can therefore be combined into cluster-bounded requests and submitted asynchronously using `io_uring`.

The `mmap` refiner provides an alternative implementation.

Request coalescing intentionally trades some read amplification for:

- larger requests
- fewer I/O operations
- greater SSD bandwidth utilization
- higher effective queue depth

---

## 7. Configuration

The default configuration is stored in:

```text
configs/default.yaml
```

The configuration is divided into the following logical sections:

```text
index
planner
engine
refiner
simd
update
```

Important paper-aligned parameters include:

| Parameter | Representative value |
|---|---:|
| `nprobe` | 32 |
| Candidate / refinement budget | 1024 |
| Asynchronous I/O queue depth | 32 |
| DRAM budget used for baseline comparison | 64 GiB |
| Latency query threads | 1 |
| Throughput threads | all physical CPU cores |
| Planner sampling ratio | 1%--2% |
| Minimum per-cluster sample size | 64 |
| Geometric safety floor | 16--32 clusters |

The exact values can be adjusted through `configs/default.yaml`.

---

## 8. Dataset and IVFPQ Configurations

The following index configurations correspond to the evaluation in the paper.

| Dataset | Vectors | Dimension | `nlist` | `M` | `nbits` |
|---|---:|---:|---:|---:|---:|
| SIFT1M | 1M | 128 | 4,096 | 32 | 4 |
| GIST1M | 1M | 960 | 4,096 | 64 | 4 |
| Deep1M | 1M | 96 | 4,096 | 24 | 4 |
| LAION-400M | 400M | 512 | 32,768 | 64 | 4 |
| Deep1B | 1B | 96 | 65,536 | 24 | 4 |

The IVFPQ indexes are constructed using the standard FAISS training and index construction pipeline.

PredicatePQ does **not** replace FAISS clustering, coarse quantization, or PQ codebook training. It adds the execution structures required to preserve cluster identity across predicate evaluation, PQ scoring, and SSD refinement.

---

## 9. IVFPQ Construction Cost

The measured FAISS IVFPQ construction settings used in the evaluation are:

| Dataset | `nlist` | PQ Config | Training Size | Build Time | Index Size |
|---|---:|---:|---:|---:|---:|
| SIFT1M | 4,096 | 32 x 4 | 1.0M | 35.7 s | 24.9 MiB |
| GIST1M | 4,096 | 64 x 4 | 1.0M | 4.2 min | 53.2 MiB |
| Deep1M | 4,096 | 24 x 4 | 1.0M | 29.8 s | 20.6 MiB |
| LAION-400M | 32,768 | 64 x 4 | 8.4M | 4.8 h | 15.0 GiB |
| Deep1B | 65,536 | 24 x 4 | 16.8M | 4.1 h | 18.7 GiB |

These values describe the underlying FAISS IVFPQ construction.

PredicatePQ-specific preprocessing consists primarily of:

- ID-to-cluster extraction
- cluster statistics
- selectivity samples
- physical offsets
- auxiliary cluster metadata

These structures are created offline and reused during query execution.

---

## 10. Data Layout and Memory Residency

### In Memory

Frequently accessed structures include:

- IVF/PQ index structures
- compact PQ codes
- cluster statistics
- ID-to-cluster mappings
- selectivity samples
- predicate metadata
- temporary query and I/O buffers

The ID-to-cluster map uses a 16-bit cluster ID when the number of clusters is at most 65,536.

Its approximate storage cost is:

| Number of vectors | Mapping size |
|---:|---:|
| 10K | 19.53 KiB |
| 100K | 195.3 KiB |
| 1M | 1.95 MiB |
| 10M | 19.53 MiB |
| 100M | 195.31 MiB |

### On SSD

Full-precision vectors remain SSD-resident.

The large-scale evaluation uses approximately:

| Dataset | IVFPQ Index | ID-to-Cluster | Scalar Index & Other | Raw Vectors on SSD |
|---|---:|---:|---:|---:|
| LAION-400M | 15.0 GiB | 0.75 GiB | 1.8 GiB | 762.9 GiB |
| Deep1B | 18.7 GiB | 1.86 GiB | 4.1 GiB | 357.6 GiB |

The query path therefore does not require the full raw-vector collection to reside in DRAM.

---

## 11. Evaluation Platform

The paper evaluation was performed on:

```text
CPU:       2 x Intel Xeon Gold 5218R
SIMD:      AVX-512
DRAM:      256 GiB
Storage:   4 x 1 TiB NVMe SSD
Layout:    RAID 0
I/O:       O_DIRECT
```

Graph-baseline DRAM residency was limited using Linux cgroups.

For the representative matched-recall experiments:

```text
DRAM cap:                  64 GiB
configured I/O QD:         32
single-query latency:      1 query thread
throughput experiments:    all physical CPU cores
```

Latency measurements exclude client-side RPC and network transport.

Reported PredicatePQ latency includes:

- planning
- predicate processing
- PQ scanning
- SSD refinement
- final reranking

---

## 12. Workload Generation

Hybrid queries combine:

```text
query vector + scalar predicate
```

Unless otherwise specified, the main controlled evaluation uses range predicates with selectivities spanning approximately:

```text
2% -- 95%
```

Each reported operating point averages:

```text
1,000 queries
```

For controlled experiments, scalar attributes are generated independently of the vector values so that execution effects can be isolated.

Robustness experiments additionally vary:

- predicate type
- scalar attribute distribution
- scalar-vector correlation
- unfavorable / adversarial placement

The controlled workload is intended to isolate execution behavior rather than claim to reproduce a particular production metadata trace.

For reproducibility, keep the random seed fixed when regenerating scalar attributes and predicate selectivities.

---

## 13. Representative Baseline Configurations

All systems are independently tuned to matched retrieval quality rather than forced to use the same search parameters.

The representative operating point used for `Recall@100 ~= 0.95` is:

| Method | Implementation | Main Search Configuration | I/O | DRAM Cap | Achieved R@100 |
|---|---|---|---:|---:|---:|
| FilteredDiskANN | Native | `L=128`, `beam=32` | QD=32 | 64 GiB | 0.951 |
| Milvus-Pre | Milvus | `nprobe=32`, `cand=1024` | QD=32 | 64 GiB | 0.949 |
| Milvus-Post | Milvus | `nprobe=32`, `cand=1024` | QD=32 | 64 GiB | 0.952 |
| Milvus-Hybrid | Milvus | `nprobe=32`, `cand=1024` | QD=32 | 64 GiB | 0.951 |
| NaviX | original Kùzu-integrated implementation | `width=128`, `beam=32` | QD=32 | 64 GiB | 0.948 |
| PredicatePQ | Native C++ | `nprobe=32`, `cand=1024` | QD=32 | 64 GiB | 0.952 |

### Fairness Policy

The following rules are used throughout comparison experiments:

1. All methods use the same storage configuration.
2. All methods are constrained to the same DRAM budget.
3. Each method is tuned independently to the target recall.
4. Graph search width / beam parameters are tuned independently from IVF `nprobe` and candidate budgets.
5. Asynchronous I/O depth is tuned independently where applicable.
6. Single-query latency uses one query thread.
7. Throughput experiments use all physical CPU cores.
8. RPC and network transport are excluded from latency measurements.

This avoids using one common parameter setting that is favorable to one index family but suboptimal for another.

---

## 14. Large-Scale Evaluation Configuration

The large-scale experiments use:

```text
LAION-400M
Deep1B
```

Full-precision vectors remain SSD-resident, while frequently used index metadata remain in DRAM.

For Deep1B at 30% selectivity, the representative PredicatePQ query path scans approximately:

```text
0.49M PQ codes / query
```

and refines:

```text
1,024 full-precision vectors
```

corresponding to approximately:

```text
0.38 MiB useful vector payload
```

before request coalescing.

Physical bytes read can be larger than the useful payload because cluster-local request coalescing deliberately trades bounded read amplification for larger and more schedulable SSD requests.

---

## 15. Storage-Side Measurements

The execution-regularity evaluation records:

- I/O requests per query
- average request size
- total bytes read
- read amplification
- configured queue depth
- average in-flight I/O depth
- achieved SSD bandwidth

At 30% selectivity under configured `QD=32`, the evaluated setup observes an average in-flight depth of approximately:

```text
PredicatePQ:         28
FilteredDiskANN:      8
```

For PredicatePQ, total physical read volume over the evaluated SIFT1M selectivity range is approximately:

```text
4.5 -- 17.6 MiB/query
```

and is approximately:

```text
8.9 MiB/query
```

at 30% selectivity.

These measurements are intended to distinguish execution regularity from a simple reduction in byte volume.

---

## 16. Metrics

The evaluation reports:

### Retrieval quality

- Recall@100
- Recall@10
- Recall--latency trade-off
- Recall--QPS trade-off

### Performance

- mean query latency
- P95 latency
- throughput / QPS

### Planner

- selected execution plan
- oracle-relative regret
- plan-hit rate
- misprediction overhead

### Storage behavior

- I/O request count
- average request size
- total bytes read
- read amplification
- queue depth / in-flight depth
- SSD bandwidth

### System overhead

- ID-to-cluster mapping size
- preprocessing latency
- IVFPQ build time
- IVFPQ index size
- large-scale DRAM / SSD footprint

---

## 17. Update Path

PredicatePQ primarily targets **static and read-mostly workloads**.

### Append-Oriented Insertions

New vectors are:

1. assigned to an IVF cluster
2. inserted into a memory-resident delta buffer
3. searched together with the main index
4. periodically flushed into cluster-aligned disk segments

This converts small random updates into batched writes.

Long-term distribution drift can require:

- background compaction
- IVFPQ retraining

### Logical Deletions

Deleted vector IDs are recorded in memory-resident tombstone bitmaps.

Tombstones are checked during:

- predicate evaluation
- candidate generation
- refinement

Physical space reclamation is deferred to periodic compaction.

### Scope

The prototype supports buffered insertion and logical deletion while preserving query-time visibility.

Update-heavy online reclustering is outside the evaluated workload.

---

## 18. Reproducing the Paper Configuration

A typical reproduction workflow is:

### Step 1: Build PredicatePQ

```bash
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPREDICATEPQ_ENABLE_IO_URING=ON

make -j
```

### Step 2: Select the Dataset Configuration

Use the corresponding IVFPQ parameters from Section 8.

For example, the SIFT1M configuration is:

```text
dimension = 128
nlist     = 4096
M         = 32
nbits     = 4
```

Deep1B uses:

```text
dimension = 96
nlist     = 65536
M         = 24
nbits     = 4
```

### Step 3: Configure the Query Path

Representative PredicatePQ settings are:

```text
nprobe              = 32
candidate budget    = 1024
I/O queue depth     = 32
DRAM budget         = 64 GiB
```

### Step 4: Generate Scalar Workloads

Generate scalar predicates at the desired selectivity while preserving the same random seed.

The principal selectivities used in the paper are drawn from:

```text
2%, 5%, 10%, 30%, 50%, 70%, 90%, 95%
```

Not every experiment uses every point.

### Step 5: Run Queries

For latency:

```text
1 query thread
1,000 queries / operating point
```

For throughput:

```text
all physical CPU cores
fixed query batch
```

### Step 6: Match Recall

Tune the search budget for each method independently.

Do not compare methods using the same raw parameter values unless those values also produce comparable recall.

The main controlled operating point is approximately:

```text
Recall@100 = 0.95
```

### Step 7: Collect Metrics

Record at least:

```text
latency
P95 latency
QPS
Recall
I/O request count
request size
bytes read
read amplification
in-flight queue depth
SSD bandwidth
```

---

## 19. Notes on Artifact Scope

The artifact is designed to reproduce the execution mechanisms and configuration used by PredicatePQ.

The following parts rely on external implementations:

- FAISS for IVFPQ training and base index construction
- Apache Arrow for columnar scalar processing
- FilteredDiskANN for the graph baseline
- Milvus for IVFPQ filtering baselines
- NaviX through the original Kùzu-integrated implementation

PredicatePQ-specific additions are:

- cluster-stratified selectivity metadata
- population-weighted selectivity aggregation
- planner logic
- ID-to-cluster mapping
- `ClusterReduce`
- SIMD-oriented candidate preparation
- cluster-preserving refinement scheduling
- asynchronous batched SSD refinement

---

## 20. Troubleshooting

### yaml-cpp Header Not Found

If compilation fails with:

```text
cannot open include file <yaml-cpp/yaml.h>
```

install:

```bash
sudo apt-get install -y libyaml-cpp-dev
```

then rerun:

```bash
cmake ..
make -j
```

### FAISS Not Found

If FAISS is installed outside the system prefix:

```bash
cmake .. \
  -DCMAKE_PREFIX_PATH=/path/to/faiss/install
```

### liburing Not Found

Install:

```bash
sudo apt-get install -y liburing-dev
```

or disable the `io_uring` path:

```bash
cmake .. \
  -DPREDICATEPQ_ENABLE_IO_URING=OFF
```

### Arrow / Parquet Not Found

Install the corresponding Arrow and Parquet development packages or provide their installation prefix through CMake.

### O_DIRECT

The evaluation uses `O_DIRECT` for disk-resident experiments.

Make sure that:

- the backing filesystem supports direct I/O
- request buffers satisfy required alignment constraints
- the target vector files are placed on the intended NVMe device / RAID array

---

## 21. Reproducibility Notes

For exact experiment reproduction, record the following together with each result:

```text
dataset
query seed
predicate seed
predicate selectivity
target Recall@k
nlist
M
nbits
nprobe
candidate budget
refinement budget
queue depth
number of query threads
DRAM cap
refiner backend
```

The paper reports latency without RPC/network transport overhead.

When comparing against server-style systems such as Milvus, use the same measurement boundary consistently and exclude client-side RPC/network time if reproducing the paper numbers.

## 22. License

This repository is released as a research prototype for artifact evaluation and research reproducibility.

