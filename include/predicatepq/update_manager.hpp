#pragma once
#include "scalar_store.hpp"
#include "types.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ppq {

class UpdateManager {
public:
    UpdateManager(std::shared_ptr<ScalarStore> scalar, uint32_t dim, std::string delta_vec_file);

    // append one vector + scalar row
    Id insert(const std::vector<float>& vec, const ScalarRow& row);

    // logical delete
    void logical_delete(Id id);

    std::vector<int64_t> compact_all();

private:
    void append_vector_to_delta_(const std::vector<float>& vec);

    std::shared_ptr<ScalarStore> scalar_;
    uint32_t dim_;
    std::string delta_vec_file_;
};

} // namespace ppq