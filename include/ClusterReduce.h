#pragma once
#include "Common.h"
#include <vector>
#include <cstdint>
#include <faiss/IndexIVFPQ.h>

class ClusterReduce {
public:
    ClusterReduce(int num_clusters, const std::vector<int>& id_to_cluster);

    // Algorithm 1: Vectorized ClusterReduce (Mode A: Pre-Filtering)
    void execute_mode_a(const std::vector<int64_t>& valid_ids,
                        std::vector<int64_t>& ids_out,
                        std::vector<int>& counts,
                        std::vector<int>& offsets);

    // Mode B: Stat-Guided Pruning (Post-Filtering)
    std::vector<int> execute_mode_b(const std::vector<float>& centroid_distances,
                                    const std::vector<float>& cluster_probs,
                                    float alpha,
                                    int top_k_clusters);

private:
    int num_clusters_;
    const std::vector<int>& id_to_cluster_;
};