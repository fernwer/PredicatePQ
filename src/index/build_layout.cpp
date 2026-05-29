#include "predicatepq/build_layout.hpp"
#include <faiss/IndexIVFPQ.h>
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace ppq {

template <typename T>
void write_binary_vector(const std::string& path, const std::vector<T>& v) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("open out file failed: " + path);
    fout.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    if (!fout) throw std::runtime_error("write out file failed: " + path);
}

// 显式实例化
template void write_binary_vector<uint16_t>(const std::string&, const std::vector<uint16_t>&);
template void write_binary_vector<uint32_t>(const std::string&, const std::vector<uint32_t>&);
template void write_binary_vector<uint64_t>(const std::string&, const std::vector<uint64_t>&);
template void write_binary_vector<float>(const std::string&, const std::vector<float>&);
template void write_binary_vector<uint8_t>(const std::string&, const std::vector<uint8_t>&);

LayoutBuildResult build_layout_from_ivfpq(const faiss::IndexIVFPQ& index, uint64_t ntotal, uint32_t dim) {
    (void)dim; // dim在offset字节计算时外部可用，这里仅构建ID映射

    LayoutBuildResult r;
    const uint32_t nlist = static_cast<uint32_t>(index.nlist);

    r.id_to_cluster.assign(ntotal, 0);
    r.id_to_local.assign(ntotal, 0);
    r.cluster_counts.assign(nlist, 0);
    r.cluster_offsets.assign(nlist + 1, 0);

    for (uint32_t c = 0; c < nlist; ++c) {
        size_t sz = index.invlists->list_size(c);
        r.cluster_counts[c] = static_cast<uint32_t>(sz);

        for (size_t j = 0; j < sz; ++j) {
            faiss::idx_t id = index.invlists->get_single_id(c, j);
            if (id < 0 || static_cast<uint64_t>(id) >= ntotal) continue;
            r.id_to_cluster[static_cast<size_t>(id)] = static_cast<uint16_t>(c);
            r.id_to_local[static_cast<size_t>(id)] = static_cast<uint32_t>(j);
        }
    }

    for (uint32_t c = 0; c < nlist; ++c) {
        r.cluster_offsets[c + 1] = r.cluster_offsets[c] + r.cluster_counts[c];
    }

    r.perm_new_to_old.assign(ntotal, 0);
    r.old_to_new.assign(ntotal, 0);

    for (uint64_t old = 0; old < ntotal; ++old) {
        uint16_t c = r.id_to_cluster[old];
        uint32_t j = r.id_to_local[old];
        uint64_t new_pos = r.cluster_offsets[c] + j;
        r.perm_new_to_old[new_pos] = static_cast<uint32_t>(old);
        r.old_to_new[old] = static_cast<uint32_t>(new_pos);
    }

    // id_to_disk_offset 外部按 dim 计算更清晰，这里先占位0
    r.id_to_disk_offset.assign(ntotal, 0);
    return r;
}

void write_clustered_vectors(const std::vector<float>& xb,
                             uint32_t dim,
                             const std::vector<uint32_t>& perm_new_to_old,
                             const std::string& out_fbin) {
    const uint64_t ntotal = static_cast<uint64_t>(perm_new_to_old.size());
    if (xb.size() != static_cast<size_t>(ntotal) * dim) {
        throw std::runtime_error("write_clustered_vectors: xb size mismatch");
    }

    std::vector<float> clustered(static_cast<size_t>(ntotal) * dim);
    for (uint64_t new_pos = 0; new_pos < ntotal; ++new_pos) {
        uint32_t old = perm_new_to_old[new_pos];
        const float* src = xb.data() + static_cast<size_t>(old) * dim;
        float* dst = clustered.data() + static_cast<size_t>(new_pos) * dim;
        std::copy(src, src + dim, dst);
    }

    write_binary_vector<float>(out_fbin, clustered);
}

} // namespace ppq