#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace faiss {
class IndexIVFPQ;
}

namespace ppq {

struct LayoutBuildResult {
    std::vector<uint16_t> id_to_cluster;
    std::vector<uint32_t> id_to_local;
    std::vector<uint32_t> cluster_counts;
    std::vector<uint64_t> cluster_offsets; // size nlist+1
    std::vector<uint32_t> perm_new_to_old;
    std::vector<uint32_t> old_to_new;
    std::vector<uint64_t> id_to_disk_offset;
};

LayoutBuildResult build_layout_from_ivfpq(const faiss::IndexIVFPQ& index, uint64_t ntotal, uint32_t dim);

void write_clustered_vectors(const std::vector<float>& xb, // old-id order
                             uint32_t dim,
                             const std::vector<uint32_t>& perm_new_to_old,
                             const std::string& out_fbin);

template <typename T>
void write_binary_vector(const std::string& path, const std::vector<T>& v);

} // namespace ppq