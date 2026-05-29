#include "predicatepq/engine.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>

namespace ppq {
namespace {

static inline float l2(const float* a, const float* b, uint32_t d) {
    float s = 0.f;
    for (uint32_t i = 0; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

} // namespace

PredicatePQEngine::PredicatePQEngine(Planner planner,
                                     std::shared_ptr<ScalarStore> scalar,
                                     std::shared_ptr<PQScanner> scanner,
                                     std::unique_ptr<Refiner> refiner,
                                     std::vector<ClusterId> id_to_cluster,
                                     uint32_t K,
                                     uint32_t dim,
                                     std::vector<float> coarse_centroids,
                                     std::vector<uint64_t> cluster_offsets,
                                     EngineConfig cfg)
    : planner_(std::move(planner)),
      scalar_(std::move(scalar)),
      scanner_(std::move(scanner)),
      refiner_(std::move(refiner)),
      id_to_cluster_(std::move(id_to_cluster)),
      K_(K),
      dim_(dim),
      coarse_centroids_(std::move(coarse_centroids)),
      cluster_offsets_(std::move(cluster_offsets)),
      cfg_(cfg) {
    cluster_to_ids_.assign(K_, {});
    for (Id id = 0; id < static_cast<Id>(id_to_cluster_.size()); ++id) {
        cluster_to_ids_[id_to_cluster_[id]].push_back(id);
    }
}

std::vector<float> PredicatePQEngine::compute_centroid_dist_(const float* q) const {
    std::vector<float> d(K_, 0.f);
    for (uint32_t c = 0; c < K_; ++c) {
        d[c] = l2(q, &coarse_centroids_[static_cast<size_t>(c) * dim_], dim_);
    }
    return d;
}

std::vector<float> PredicatePQEngine::estimate_cluster_density_(const std::string& pred_sql) const {
    // 返回每个 cluster 的估计“有效数量” = \tilde{delta}_i * |C_i|
    std::vector<float> est(K_, 0.f);
    std::mt19937_64 rng(1234567);

    for (uint32_t c = 0; c < K_; ++c) {
        const auto& ids = cluster_to_ids_[c];
        uint32_t sz = static_cast<uint32_t>(ids.size());
        if (sz == 0) {
            est[c] = 0.f;
            continue;
        }

        uint32_t ni = 0;
        if (sz < cfg_.sample_nmin)
            ni = sz;
        else
            ni = std::max<uint32_t>(cfg_.sample_nmin, static_cast<uint32_t>(std::floor(cfg_.sample_ratio * sz)));

        uint32_t hit = 0;
        if (ni == sz) {
            for (auto id : ids)
                if (scalar_->eval_id(id, pred_sql)) ++hit;
        } else {
            std::uniform_int_distribution<uint32_t> dis(0, sz - 1);
            for (uint32_t t = 0; t < ni; ++t) {
                Id id = ids[dis(rng)];
                if (scalar_->eval_id(id, pred_sql)) ++hit;
            }
        }

        float smoothed = static_cast<float>(hit + 1) / static_cast<float>(ni + 2); // Laplace
        est[c] = smoothed * static_cast<float>(sz);
    }
    return est;
}

std::vector<uint32_t> PredicatePQEngine::select_clusters_post_(const std::vector<float>& dists,
                                                               const std::vector<float>& est_counts) const {
    // Rank_dist: asc(dists), Rank_stat: desc(est_counts)
    std::vector<uint32_t> idx(K_);
    std::iota(idx.begin(), idx.end(), 0);

    std::vector<uint32_t> by_dist = idx;
    std::sort(by_dist.begin(), by_dist.end(), [&](uint32_t a, uint32_t b) { return dists[a] < dists[b]; });

    std::vector<uint32_t> by_stat = idx;
    std::sort(by_stat.begin(), by_stat.end(), [&](uint32_t a, uint32_t b) { return est_counts[a] > est_counts[b]; });

    std::vector<uint32_t> rank_dist(K_), rank_stat(K_);
    for (uint32_t r = 0; r < K_; ++r) rank_dist[by_dist[r]] = r;
    for (uint32_t r = 0; r < K_; ++r) rank_stat[by_stat[r]] = r;

    struct Node {
        uint32_t c;
        float score;
    };
    std::vector<Node> scored;
    scored.reserve(K_);
    for (uint32_t c = 0; c < K_; ++c) {
        float s = cfg_.alpha * static_cast<float>(rank_dist[c]) + (1.0f - cfg_.alpha) * static_cast<float>(rank_stat[c]);
        scored.push_back({c, s});
    }
    std::sort(scored.begin(), scored.end(), [](const Node& a, const Node& b) { return a.score < b.score; });

    std::vector<uint32_t> out;
    out.reserve(cfg_.nprobe);

    // safety floor: 强制纳入最近的 min_geo_floor
    std::unordered_set<uint32_t> used;
    for (uint32_t i = 0; i < std::min<uint32_t>(cfg_.min_geo_floor, K_); ++i) {
        out.push_back(by_dist[i]);
        used.insert(by_dist[i]);
        if (out.size() >= cfg_.nprobe) return out;
    }

    for (auto& n : scored) {
        if (used.insert(n.c).second) {
            out.push_back(n.c);
            if (out.size() >= cfg_.nprobe) break;
        }
    }
    return out;
}

std::vector<Id> PredicatePQEngine::prefilter_candidates(const Query& q) {
    auto valid = scalar_->scan_ids(q.predicate_sql);
    auto reduced = ClusterReducer::run(valid, id_to_cluster_, K_);
    if (reduced.ids_out.size() > cfg_.candidate_budget) {
        reduced.ids_out.resize(cfg_.candidate_budget);
    }
    return reduced.ids_out;
}

std::vector<Id> PredicatePQEngine::postfilter_candidates(const Query& q) {
    auto dists = compute_centroid_dist_(q.qvec.data());
    auto est_counts = estimate_cluster_density_(q.predicate_sql);
    auto selected = select_clusters_post_(dists, est_counts);

    std::vector<Id> out;
    out.reserve(cfg_.candidate_budget);

    for (auto c : selected) {
        for (auto id : cluster_to_ids_[c]) {
            if (scalar_->eval_id(id, q.predicate_sql)) {
                out.push_back(id);
                if (out.size() >= cfg_.candidate_budget) return out;
            }
        }
    }
    return out;
}

std::vector<Result> PredicatePQEngine::search(const Query& q) {
    float est_sel = scalar_->estimate_selectivity(q.predicate_sql);
    auto d = planner_.decide(scalar_->size(), cfg_.candidate_budget, 64, 128, est_sel);

    std::vector<Id> ids = (d.type == PlanType::PreFilter) ? prefilter_candidates(q) : postfilter_candidates(q);

    auto approx = scanner_->scan_candidates(q.qvec.data(), ids, std::max(4u * q.topk, q.topk));
    return refiner_->refine(q.qvec.data(), approx, q.topk);
}

} // namespace ppq