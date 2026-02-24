#pragma once
#include "Common.h"
#include <vector>
#include <memory>
#include <cmath>

enum class ExecutionMode { PRE_FILTERING, POST_FILTERING };

class Planner {
public:
    Planner(int num_clusters, float threshold = 0.3f);

    void build_strata_samples(const std::vector<std::vector<int64_t>>& cluster_to_ids, float sample_rate = 0.02f);

    ExecutionMode plan(const ScalarFilter& filter, std::vector<float>& cluster_probs);

private:
    int num_clusters_;
    float threshold_;
    std::vector<std::vector<int64_t>> strata_samples_;
};