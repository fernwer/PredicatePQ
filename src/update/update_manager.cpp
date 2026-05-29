#include "predicatepq/update_manager.hpp"
#include <fstream>
#include <stdexcept>
#include <vector>

namespace ppq {

UpdateManager::UpdateManager(std::shared_ptr<ScalarStore> scalar, uint32_t dim, std::string delta_vec_file)
    : scalar_(std::move(scalar)), dim_(dim), delta_vec_file_(std::move(delta_vec_file)) {
}

Id UpdateManager::insert(const std::vector<float>& vec, const ScalarRow& row) {
    if (vec.size() != dim_) throw std::runtime_error("insert vec dim mismatch");
    Id id = scalar_->append_row(row);
    append_vector_to_delta_(vec);
    return id;
}

void UpdateManager::logical_delete(Id id) {
    scalar_->mark_deleted(id);
}

std::vector<int64_t> UpdateManager::compact_all() {
    auto remap = scalar_->compact();

    // truncate delta vec file
    {
        std::ofstream fout(delta_vec_file_, std::ios::binary | std::ios::trunc);
        if (!fout) throw std::runtime_error("truncate delta vec file failed");
    }
    return remap;
}

void UpdateManager::append_vector_to_delta_(const std::vector<float>& vec) {
    std::ofstream fout(delta_vec_file_, std::ios::binary | std::ios::app);
    if (!fout) throw std::runtime_error("open delta vec file failed");
    fout.write(reinterpret_cast<const char*>(vec.data()), static_cast<std::streamsize>(vec.size() * sizeof(float)));
}

} // namespace ppq