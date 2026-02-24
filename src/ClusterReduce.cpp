#include "ClusterReduce.h"
#include <numeric>
#include <algorithm>

ClusterReduce::ClusterReduce(int num_clusters, const std::vector<int>& id_to_cluster)
    : num_clusters_(num_clusters), id_to_cluster_(id_to_cluster) {
}

void ClusterReduce::execute_mode_a(const std::vector<int64_t>& valid_ids,
                                   std::vector<int64_t>& ids_out,
                                   std::vector<int>& counts,
                                   std::vector<int>& offsets) {
    counts.assign(num_clusters_, 0);
    offsets.assign(num_clusters_ + 1, 0);
    ids_out.resize(valid_ids.size());

// Step 1: Vectorized Histogram Construction
#pragma omp simd
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        int cid = id_to_cluster_[valid_ids[i]];
        counts[cid]++;
    }

    // Step 2: Layout Calculation (Prefix Sums)
    for (int i = 0; i < num_clusters_; ++i) {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    // Step 3: Batch Materialization (Scatter to Buckets)
    std::vector<int> current_offsets = offsets;
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        int64_t vid = valid_ids[i];
        int cid = id_to_cluster_[vid];
        int pos = current_offsets[cid]++;
        ids_out[pos] = vid;
    }
}

std::vector<int> ClusterReduce::execute_mode_b(const std::vector<float>& centroid_distances,
                                               const std::vector<float>& cluster_probs,
                                               float alpha,
                                               int top_k_clusters) {
    std::vector<int> indices(num_clusters_);
    std::iota(indices.begin(), indices.end(), 0);

    // Compute Ranks
    std::vector<int> rank_dist(num_clusters_), rank_stat(num_clusters_);
    auto dist_indices = indices;
    std::sort(
        dist_indices.begin(), dist_indices.end(), [&](int a, int b) { return centroid_distances[a] < centroid_distances[b]; });
    for (int i = 0; i < num_clusters_; ++i) rank_dist[dist_indices[i]] = i;

    auto stat_indices = indices;
    std::sort(stat_indices.begin(), stat_indices.end(), [&](int a, int b) {
        return cluster_probs[a] > cluster_probs[b]; // Prob越大越靠前
    });
    for (int i = 0; i < num_clusters_; ++i) rank_stat[stat_indices[i]] = i;

    // Joint Scoring
    std::vector<float> scores(num_clusters_);
    for (int i = 0; i < num_clusters_; ++i) {
        scores[i] = alpha * rank_dist[i] + (1.0f - alpha) * rank_stat[i];
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return scores[a] < scores[b]; });

    indices.resize(std::min(top_k_clusters, num_clusters_));
    return indices;
}