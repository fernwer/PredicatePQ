#include "predicatepq/pq_scanner.hpp"
#include <algorithm>
#include <queue>

namespace ppq {

PQScanner::PQScanner(uint32_t d,
                     uint32_t M,
                     uint32_t nbits,
                     std::vector<uint8_t> rowwise_codes,
                     uint32_t ntotal,
                     const faiss::ProductQuantizer& trained_pq)
    : d_(d),
      M_(M),
      nbits_(nbits),
      code_size_(M * nbits / 8),
      ntotal_(ntotal),
      pq_(trained_pq),
      codes_(std::move(rowwise_codes)) {
}

std::vector<Result> PQScanner::scan_candidates(const float* q, const std::vector<Id>& ids, uint32_t topk) const {
    std::vector<float> table((1u << nbits_) * M_);
    pq_.compute_distance_table(q, table.data());

    auto cmp = [](const Result& a, const Result& b) { return a.distance < b.distance; };
    std::priority_queue<Result, std::vector<Result>, decltype(cmp)> heap(cmp);

    for (Id id : ids) {
        const uint8_t* code = &codes_[size_t(id) * code_size_];
        float dist = 0.f;
        for (uint32_t m = 0; m < M_; ++m) {
            uint8_t c = code[m];
            dist += table[m * (1u << nbits_) + c];
        }
        if (heap.size() < topk)
            heap.push({id, dist});
        else if (dist < heap.top().distance) {
            heap.pop();
            heap.push({id, dist});
        }
    }

    std::vector<Result> out(heap.size());
    for (int i = int(out.size()) - 1; i >= 0; --i) {
        out[i] = heap.top();
        heap.pop();
    }
    return out;
}

} // namespace ppq