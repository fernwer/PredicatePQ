#pragma once
#include "Common.h"
#include <string>
#include <vector>

class DiskRefiner {
public:
    DiskRefiner(const std::string& db_path, int dim);
    ~DiskRefiner();

    void build_disk_layout(const std::vector<std::vector<std::vector<float>>>& clustered_vectors);

    void refine(const std::vector<Candidate>& candidates, const float* query, std::vector<Candidate>& final_results, int k);

private:
    std::string path_;
    int dim_;
    int fd_;
    std::vector<size_t> cluster_disk_offsets_;
};