#pragma once
#include "Common.h"
#include "Planner.h"
#include "ClusterReduce.h"
#include "DiskRefiner.h"
#include <faiss/IndexIVFPQ.h>
#include <memory>
#include <vector>
#include <string>

class PredicatePQ {
public:
    PredicatePQ(int dim, int num_clusters, int M, int nbits, const std::string& disk_path);

    void train_and_add(const float* vectors, int n);

    std::vector<Candidate> search(const float* query, const ScalarFilter& filter, int k, int nprobe = 32);

private:
    int dim_;
    int num_clusters_;
    int ntotal_;
    std::string disk_path_;
    int M_;     
    int ksub_;  

    std::unique_ptr<faiss::IndexIVFPQ> ivfpq_;

    std::vector<int> id_to_cluster_;
    std::vector<int> id_to_offset_;            
    std::vector<std::vector<int64_t>> cluster_to_ids_;
    
    std::vector<std::vector<float>> centroids_; 
    std::vector<uint8_t> global_pq_codes_;      

    std::unique_ptr<Planner> planner_;
    std::unique_ptr<ClusterReduce> cluster_reduce_;
    std::unique_ptr<DiskRefiner> disk_refiner_;

    void compute_centroid_distances(const float* query, std::vector<float>& distances);
    
    void fast_scan_pq(const float* query, int cluster_id, const std::vector<int64_t>& candidates, std::vector<Candidate>& results);
};