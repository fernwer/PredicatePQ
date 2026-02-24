#include "DiskRefiner.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <iostream>
#include <cstring>

DiskRefiner::DiskRefiner(const std::string& db_path, int dim) : path_(db_path), dim_(dim), fd_(-1) {
    fd_ = open(path_.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
        throw std::runtime_error("Failed to open disk file.");
    }
}

DiskRefiner::~DiskRefiner() {
    if (fd_ != -1) close(fd_);
}

void DiskRefiner::build_disk_layout(const std::vector<std::vector<std::vector<float>>>& clustered_vectors) {
    cluster_disk_offsets_.push_back(0);
    size_t current_offset = 0;

    for (const auto& cluster : clustered_vectors) {
        for (const auto& vec : cluster) {
            write(fd_, vec.data(), dim_ * sizeof(float));
            current_offset += dim_ * sizeof(float);
        }
        cluster_disk_offsets_.push_back(current_offset);
    }
    fsync(fd_);
}

void DiskRefiner::refine(const std::vector<Candidate>& candidates,
                         const float* query,
                         std::vector<Candidate>& final_results,
                         int k) {
    auto sorted_cands = candidates;
    std::sort(sorted_cands.begin(), sorted_cands.end(), [](const Candidate& a, const Candidate& b) {
        if (a.cluster_id != b.cluster_id) return a.cluster_id < b.cluster_id;
        return a.offset_in_cluster < b.offset_in_cluster;
    });

    std::vector<float> buffer(dim_);
    for (auto& cand : sorted_cands) {
        size_t disk_pos = cluster_disk_offsets_[cand.cluster_id] + cand.offset_in_cluster * dim_ * sizeof(float);
        pread(fd_, buffer.data(), dim_ * sizeof(float), disk_pos);

        float exact_dist = 0.0f;
#pragma omp simd reduction(+ : exact_dist)
        for (int i = 0; i < dim_; ++i) {
            float diff = buffer[i] - query[i];
            exact_dist += diff * diff;
        }
        cand.distance = exact_dist;
        final_results.push_back(cand);
    }

    std::sort(final_results.begin(), final_results.end(), [](const Candidate& a, const Candidate& b) {
        return a.distance < b.distance;
    });
    if (final_results.size() > static_cast<size_t>(k)) {
        final_results.resize(k);
    }
}