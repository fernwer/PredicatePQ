#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

struct Candidate {
    int64_t id;
    int cluster_id;
    int offset_in_cluster; // 在Cluster内的局部偏移
    float distance;
};

struct ScalarFilter {
    virtual ~ScalarFilter() = default;
    virtual bool test(int64_t id) const = 0;
};