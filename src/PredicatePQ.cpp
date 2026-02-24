#include "PredicatePQ.h"
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include <algorithm>
#include <iostream>
#include <cstring>

PredicatePQ::PredicatePQ(int dim, int num_clusters, int M, int nbits, const std::string& disk_path)
    : dim_(dim), num_clusters_(num_clusters), ntotal_(0), disk_path_(disk_path), M_(M), ksub_(1 << nbits) {
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dim);
    ivfpq_ = std::make_unique<faiss::IndexIVFPQ>(quantizer, dim, num_clusters, M, nbits);
    ivfpq_->own_fields = true;
    disk_refiner_ = std::make_unique<DiskRefiner>(disk_path, dim);
}

void PredicatePQ::train_and_add(const float* vectors, int n) {
    ntotal_ = n;
    id_to_cluster_.resize(n);
    id_to_offset_.resize(n);
    cluster_to_ids_.resize(num_clusters_);

    std::cout << "Training IVFPQ..." << std::endl;
    ivfpq_->train(n, vectors);
    ivfpq_->add(n, vectors);

    centroids_.resize(num_clusters_, std::vector<float>(dim_));
    for (int i = 0; i < num_clusters_; ++i) {
        ivfpq_->quantizer->reconstruct(i, centroids_[i].data());
    }

    global_pq_codes_.resize(n * M_);
    std::vector<std::vector<std::vector<float>>> clustered_vectors(num_clusters_);

    for (int cid = 0; cid < num_clusters_; ++cid) {
        size_t list_size = ivfpq_->invlists->list_size(cid);
        const uint8_t* codes = ivfpq_->invlists->get_codes(cid);
        const faiss::idx_t* ids = ivfpq_->invlists->get_ids(cid);

        for (size_t j = 0; j < list_size; ++j) {
            int64_t vid = ids[j];
            id_to_cluster_[vid] = cid;
            id_to_offset_[vid] = j;
            cluster_to_ids_[cid].push_back(vid);

            std::memcpy(global_pq_codes_.data() + vid * M_, codes + j * M_, M_);

            std::vector<float> vec(vectors + vid * dim_, vectors + (vid + 1) * dim_);
            clustered_vectors[cid].push_back(vec);
        }
    }

    std::cout << "Building Disk Layout..." << std::endl;
    disk_refiner_->build_disk_layout(clustered_vectors);

    planner_ = std::make_unique<Planner>(num_clusters_, 0.3f);
    planner_->build_strata_samples(cluster_to_ids_);

    cluster_reduce_ = std::make_unique<ClusterReduce>(num_clusters_, id_to_cluster_);
}

void PredicatePQ::compute_centroid_distances(const float* query, std::vector<float>& distances) {
    distances.resize(num_clusters_);
    for (int i = 0; i < num_clusters_; ++i) {
        float dist = 0.0f;
        const float* centroid = centroids_[i].data();
#pragma omp simd reduction(+ : dist)
        for (int j = 0; j < dim_; ++j) {
            float diff = centroid[j] - query[j];
            dist += diff * diff;
        }
        distances[i] = dist;
    }
}

void PredicatePQ::fast_scan_pq(const float* query,
                               int cluster_id,
                               const std::vector<int64_t>& candidates,
                               std::vector<Candidate>& results) {
    std::vector<float> residual(dim_);
    const float* centroid = centroids_[cluster_id].data();

#pragma omp simd
    for (int i = 0; i < dim_; ++i) {
        residual[i] = query[i] - centroid[i];
    }

    std::vector<float> dist_table(M_ * ksub_);
    ivfpq_->pq.compute_distance_table(residual.data(), dist_table.data());

    for (int64_t vid : candidates) {
        const uint8_t* code = global_pq_codes_.data() + vid * M_;
        float approx_dist = 0.0f;

        for (int m = 0; m < M_; ++m) {
            approx_dist += dist_table[m * ksub_ + code[m]];
        }

        Candidate c;
        c.id = vid;
        c.cluster_id = cluster_id;
        c.offset_in_cluster = id_to_offset_[vid];
        c.distance = approx_dist;
        results.push_back(c);
    }
}

std::vector<Candidate> PredicatePQ::search(const float* query, const ScalarFilter& filter, int k, int nprobe) {
    std::vector<float> cluster_probs;
    ExecutionMode mode = planner_->plan(filter, cluster_probs);

    std::vector<float> centroid_distances;
    compute_centroid_distances(query, centroid_distances);

    std::vector<Candidate> pq_candidates;

    if (mode == ExecutionMode::PRE_FILTERING) {
        std::vector<int64_t> valid_ids;
        for (int i = 0; i < ntotal_; ++i) {
            if (filter.test(i)) valid_ids.push_back(i);
        }

        std::vector<int64_t> ids_out;
        std::vector<int> counts, offsets;
        cluster_reduce_->execute_mode_a(valid_ids, ids_out, counts, offsets);

        std::vector<std::pair<float, int>> dist_cid;
        for (int i = 0; i < num_clusters_; ++i) {
            if (counts[i] > 0) dist_cid.push_back({centroid_distances[i], i});
        }
        std::sort(dist_cid.begin(), dist_cid.end());

        int scanned = 0;
        for (auto& dc : dist_cid) {
            if (scanned >= nprobe) break;
            int cid = dc.second;
            std::vector<int64_t> cands(ids_out.begin() + offsets[cid], ids_out.begin() + offsets[cid] + counts[cid]);
            fast_scan_pq(query, cid, cands, pq_candidates);
            scanned++;
        }
    } else {
        std::vector<int> target_clusters = cluster_reduce_->execute_mode_b(centroid_distances, cluster_probs, 0.5f, nprobe);

        for (int cid : target_clusters) {
            std::vector<int64_t> cands;
            for (int64_t vid : cluster_to_ids_[cid]) {
                if (filter.test(vid)) cands.push_back(vid);
            }
            fast_scan_pq(query, cid, cands, pq_candidates);
        }
    }

    std::sort(pq_candidates.begin(), pq_candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.distance < b.distance;
    });
    if (pq_candidates.size() > static_cast<size_t>(k * 10)) {
        pq_candidates.resize(k * 10);
    }

    std::vector<Candidate> final_results;
    disk_refiner_->refine(pq_candidates, query, final_results, k);

    return final_results;
}