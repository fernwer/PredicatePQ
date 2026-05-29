#include "predicatepq/cluster_reduce.hpp"
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <vector>

namespace ppq {
namespace {
constexpr uint32_t CACHELINE_BYTES = 64;
constexpr uint32_t SLOT_BYTES = sizeof(uint32_t);
constexpr uint32_t PAD = CACHELINE_BYTES / SLOT_BYTES; // 16
} // namespace

ClusterReduceOutput ClusterReducer::run(const std::vector<Id>& ids_in,
                                        const std::vector<ClusterId>& id_to_cluster,
                                        uint32_t K) {
    ClusterReduceOutput out;
    out.counts.assign(K, 0);
    out.offsets.assign(K + 1, 0);
    out.ids_out.resize(ids_in.size());

    if (ids_in.empty()) return out;

    const int T = omp_get_max_threads();
    const uint32_t stride = K + PAD; // padding to reduce false sharing

    // thread-local histograms
    std::vector<uint32_t> local_hist(static_cast<size_t>(T) * stride, 0);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t* my = local_hist.data() + static_cast<size_t>(tid) * stride;

#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(ids_in.size()); ++i) {
            Id id = ids_in[static_cast<size_t>(i)];
            ClusterId c = id_to_cluster[id];
            my[c] += 1;
        }
    }

    // reduce histograms
    for (int t = 0; t < T; ++t) {
        const uint32_t* h = local_hist.data() + static_cast<size_t>(t) * stride;
        for (uint32_t c = 0; c < K; ++c) out.counts[c] += h[c];
    }

    // global prefix
    for (uint32_t c = 0; c < K; ++c) out.offsets[c + 1] = out.offsets[c] + out.counts[c];

    // compute per-thread start offsets for each cluster:
    // thread_offset[t][c] = global_offset[c] + sum_{tt<t} local_hist[tt][c]
    std::vector<uint64_t> thread_offsets(static_cast<size_t>(T) * K, 0);
    for (uint32_t c = 0; c < K; ++c) {
        uint64_t base = out.offsets[c];
        for (int t = 0; t < T; ++t) {
            thread_offsets[static_cast<size_t>(t) * K + c] = base;
            base += local_hist[static_cast<size_t>(t) * stride + c];
        }
    }

    // second pass: scatter without atomics
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t* my_ptr = thread_offsets.data() + static_cast<size_t>(tid) * K;

#pragma omp for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(ids_in.size()); ++i) {
            Id id = ids_in[static_cast<size_t>(i)];
            ClusterId c = id_to_cluster[id];
            uint64_t pos = my_ptr[c]++;
            out.ids_out[pos] = id;
        }
    }

    return out;
}

} // namespace ppq