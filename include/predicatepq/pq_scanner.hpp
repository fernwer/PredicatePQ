#pragma once
#include "types.hpp"
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <vector>

namespace ppq {

class PQScanner {
public:
    PQScanner(uint32_t d,
              uint32_t M,
              uint32_t nbits,
              std::vector<uint8_t> rowwise_codes,
              uint32_t ntotal,
              const faiss::ProductQuantizer& trained_pq);

    std::vector<Result> scan_candidates(const float* q, const std::vector<Id>& ids, uint32_t topk) const;

private:
    uint32_t d_, M_, nbits_, code_size_, ntotal_;
    faiss::ProductQuantizer pq_;
    std::vector<uint8_t> codes_; // AoS row-wise
};

} // namespace ppq