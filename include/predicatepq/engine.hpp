#pragma once
#include "types.hpp"
#include "planner.hpp"
#include "cluster_reduce.hpp"
#include "pq_scanner.hpp"
#include "refiner.hpp"
#include "scalar_store.hpp"
#include <memory>
#include <vector>

namespace ppq {

struct EngineConfig {
    uint32_t nprobe = 32;
    uint32_t candidate_budget = 4096;
    uint32_t min_geo_floor = 16;
    float alpha = 0.7f;         // score weight for distance-rank
    float sample_ratio = 0.01f; // per-cluster sampling
    uint32_t sample_nmin = 64;
};

class PredicatePQEngine {
public:
    PredicatePQEngine(Planner planner,
                      std::shared_ptr<ScalarStore> scalar,
                      std::shared_ptr<PQScanner> scanner,
                      std::unique_ptr<Refiner> refiner,
                      std::vector<ClusterId> id_to_cluster,
                      uint32_t K,
                      uint32_t dim,
                      std::vector<float> coarse_centroids,   // K * dim
                      std::vector<uint64_t> cluster_offsets, // K+1
                      EngineConfig cfg);

    std::vector<Result> search(const Query& q);

private:
    std::vector<Id> prefilter_candidates(const Query& q);
    std::vector<Id> postfilter_candidates(const Query& q);

    std::vector<float> compute_centroid_dist_(const float* q) const;
    std::vector<float> estimate_cluster_density_(const std::string& pred_sql) const;
    std::vector<uint32_t> select_clusters_post_(const std::vector<float>& dists, const std::vector<float>& est_counts) const;

private:
    Planner planner_;
    std::shared_ptr<ScalarStore> scalar_;
    std::shared_ptr<PQScanner> scanner_;
    std::unique_ptr<Refiner> refiner_;

    std::vector<ClusterId> id_to_cluster_;
    uint32_t K_;
    uint32_t dim_;
    std::vector<float> coarse_centroids_;
    std::vector<uint64_t> cluster_offsets_;

    std::vector<std::vector<Id>> cluster_to_ids_; // built once from id_to_cluster
    EngineConfig cfg_;
};

} // namespace ppq