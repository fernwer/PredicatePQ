// src/io/refiner_mmap.cpp
#include "predicatepq/refiner.hpp"
#include <algorithm>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace ppq {

class MMapRefiner final : public Refiner {
public:
    MMapRefiner(const std::string& file, uint32_t d) : d_(d) {
        fd_ = ::open(file.c_str(), O_RDONLY);
        off_t sz = lseek(fd_, 0, SEEK_END);
        data_ = static_cast<float*>(mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd_, 0));
        n_ = sz / sizeof(float) / d_;
    }
    ~MMapRefiner() override {
        // omitted: munmap/close for brevity
    }

    std::vector<Result> refine(const float* q, const std::vector<Result>& approx, uint32_t topk) override {
        std::vector<Result> r = approx;
        for (auto& x : r) {
            const float* v = data_ + size_t(x.id) * d_;
            float s = 0.f;
            for (uint32_t i = 0; i < d_; ++i) {
                float t = q[i] - v[i];
                s += t * t;
            }
            x.distance = s;
        }
        std::partial_sort(r.begin(), r.begin() + std::min<size_t>(topk, r.size()), r.end(), [](auto& a, auto& b) {
            return a.distance < b.distance;
        });
        if (r.size() > topk) r.resize(topk);
        return r;
    }

private:
    int fd_{-1};
    float* data_{nullptr};
    uint64_t n_{0};
    uint32_t d_{0};
};

std::unique_ptr<Refiner> make_mmap_refiner(const std::string& vec_file, uint32_t dim) {
    return std::make_unique<MMapRefiner>(vec_file, dim);
}

} // namespace ppq