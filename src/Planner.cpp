#include "Planner.h"

Planner::Planner(int num_clusters, float threshold) : num_clusters_(num_clusters), threshold_(threshold) {
}

void Planner::build_strata_samples(const std::vector<std::vector<int64_t>>& cluster_to_ids, float sample_rate) {
    strata_samples_.resize(num_clusters_);
    for (int i = 0; i < num_clusters_; ++i) {
        int sample_size = std::max(64, static_cast<int>(cluster_to_ids[i].size() * sample_rate));
        sample_size = std::min(sample_size, static_cast<int>(cluster_to_ids[i].size()));

        for (int j = 0; j < sample_size; ++j) {
            int idx = (j * cluster_to_ids[i].size()) / sample_size;
            strata_samples_[i].push_back(cluster_to_ids[i][idx]);
        }
    }
}

ExecutionMode Planner::plan(const ScalarFilter& filter, std::vector<float>& cluster_probs) {
    cluster_probs.resize(num_clusters_, 0.0f);
    int total_m = 0;
    int total_n = 0;

    for (int i = 0; i < num_clusters_; ++i) {
        int m_i = 0;
        int n_i = strata_samples_[i].size();
        for (int64_t id : strata_samples_[i]) {
            if (filter.test(id)) {
                m_i++;
            }
        }
        total_m += m_i;
        total_n += n_i;

        // Equation (5): Laplace Smoothing
        cluster_probs[i] = static_cast<float>(m_i + 1) / (n_i + 2);
    }

    float global_selectivity = total_n > 0 ? static_cast<float>(total_m) / total_n : 0.0f;
    return (global_selectivity < threshold_) ? ExecutionMode::PRE_FILTERING : ExecutionMode::POST_FILTERING;
}