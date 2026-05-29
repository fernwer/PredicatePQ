#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ppq {

struct ClusterLayoutMeta {
    uint32_t nlist = 0;
    uint64_t ntotal = 0;
    uint32_t dim = 0;
    uint32_t pq_m = 0;
    uint32_t pq_nbits = 0;

    std::vector<uint64_t> cluster_offsets; // size nlist+1, in vectors
    std::vector<uint32_t> cluster_counts;  // size nlist
};

} // namespace ppq