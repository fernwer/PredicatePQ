#pragma once
#include "types.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ppq {

class Refiner {
public:
    virtual ~Refiner() = default;
    virtual std::vector<Result> refine(const float* q, const std::vector<Result>& approx, uint32_t topk) = 0;
};

std::unique_ptr<Refiner> make_mmap_refiner(const std::string& vec_file, uint32_t dim);

struct IoUringRefinerOptions {
    uint32_t queue_depth = 256;
    uint32_t submit_batch = 64;
    uint32_t max_merge_vecs = 16; // coalesce contiguous vectors
    uint32_t max_inflight = 256;
    bool use_o_direct = false; // prototype 默认 false，避免对齐复杂性
};

std::unique_ptr<Refiner> make_io_uring_refiner(const std::string& vec_file,
                                               uint32_t dim,
                                               std::vector<uint64_t> id_to_disk_offset_bytes,
                                               IoUringRefinerOptions opt = {});

} // namespace ppq