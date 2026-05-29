#include "predicatepq/refiner.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <liburing.h>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace ppq {
namespace {

struct ReadTask {
    uint64_t offset;     // byte offset in vec file
    uint32_t nvec;       // number of contiguous vectors
    std::vector<Id> ids; // same order as file
};

struct Inflight {
    ReadTask task;
    void* buf{nullptr};
    uint32_t bytes{0};
};

class IoUringRefiner final : public Refiner {
public:
    IoUringRefiner(const std::string& vec_file, uint32_t dim, std::vector<uint64_t> id_to_off, IoUringRefinerOptions opt)
        : dim_(dim), vec_bytes_(dim * sizeof(float)), id_to_offset_(std::move(id_to_off)), opt_(opt) {
        int flags = O_RDONLY;
        if (opt_.use_o_direct) flags |= O_DIRECT;
        fd_ = ::open(vec_file.c_str(), flags);
        if (fd_ < 0) throw std::runtime_error("open vec file failed");

        if (io_uring_queue_init(opt_.queue_depth, &ring_, 0) < 0) {
            ::close(fd_);
            throw std::runtime_error("io_uring_queue_init failed");
        }
    }

    ~IoUringRefiner() override {
        io_uring_queue_exit(&ring_);
        if (fd_ >= 0) ::close(fd_);
    }

    std::vector<Result> refine(const float* q, const std::vector<Result>& approx, uint32_t topk) override {
        if (approx.empty()) return {};

        // 1) build read tasks by id->offset and merge contiguous
        std::vector<std::pair<uint64_t, Id>> off_id;
        off_id.reserve(approx.size());
        for (auto& r : approx) {
            if (r.id >= id_to_offset_.size()) continue;
            off_id.emplace_back(id_to_offset_[r.id], r.id);
        }
        std::sort(off_id.begin(), off_id.end(), [](auto& a, auto& b) { return a.first < b.first; });

        auto tasks = build_tasks_(off_id);

        // 2) submit/harvest in pipeline
        std::vector<Result> exact;
        exact.reserve(off_id.size());

        size_t next_submit = 0;
        uint32_t inflight = 0;
        std::vector<std::unique_ptr<Inflight>> slots(tasks.size());

        while (next_submit < tasks.size() || inflight > 0) {
            // submit phase
            uint32_t submitted_now = 0;
            while (next_submit < tasks.size() && inflight < opt_.max_inflight && submitted_now < opt_.submit_batch) {
                auto sqe = io_uring_get_sqe(&ring_);
                if (!sqe) break;

                auto req = std::make_unique<Inflight>();
                req->task = std::move(tasks[next_submit]);
                req->bytes = req->task.nvec * vec_bytes_;

                // aligned alloc (even without O_DIRECT 也可用)
                void* p = nullptr;
                if (posix_memalign(&p, 4096, req->bytes) != 0) {
                    throw std::runtime_error("posix_memalign failed");
                }
                req->buf = p;

                io_uring_prep_read(sqe, fd_, req->buf, req->bytes, static_cast<off_t>(req->task.offset));
                sqe->user_data = static_cast<__u64>(next_submit);

                slots[next_submit] = std::move(req);
                ++next_submit;
                ++submitted_now;
                ++inflight;
            }

            if (submitted_now > 0) {
                int rc = io_uring_submit(&ring_);
                if (rc < 0) throw std::runtime_error("io_uring_submit failed");
            }

            // harvest at least one CQE
            io_uring_cqe* cqe = nullptr;
            int w = io_uring_wait_cqe(&ring_, &cqe);
            if (w < 0) throw std::runtime_error("io_uring_wait_cqe failed");

            do {
                uint64_t idx = static_cast<uint64_t>(cqe->user_data);
                auto& req = slots[idx];
                if (!req) {
                    io_uring_cqe_seen(&ring_, cqe);
                    continue;
                }

                if (cqe->res < 0) {
                    free(req->buf);
                    throw std::runtime_error("io_uring read failed");
                }
                if (static_cast<uint32_t>(cqe->res) != req->bytes) {
                    free(req->buf);
                    throw std::runtime_error("short read in io_uring");
                }

                // 计算 exact distance
                const float* base = static_cast<const float*>(req->buf);
                for (uint32_t i = 0; i < req->task.nvec; ++i) {
                    const float* v = base + static_cast<size_t>(i) * dim_;
                    float d = l2_(q, v, dim_);
                    exact.push_back({req->task.ids[i], d});
                }

                free(req->buf);
                req.reset();
                --inflight;

                io_uring_cqe_seen(&ring_, cqe);
            } while (io_uring_peek_cqe(&ring_, &cqe) == 0);
        }

        // 3) topk
        if (exact.size() > topk) {
            std::nth_element(exact.begin(), exact.begin() + topk, exact.end(), [](const Result& a, const Result& b) {
                return a.distance < b.distance;
            });
            exact.resize(topk);
        }
        std::sort(exact.begin(), exact.end(), [](const Result& a, const Result& b) { return a.distance < b.distance; });
        return exact;
    }

private:
    static float l2_(const float* a, const float* b, uint32_t d) {
        float s = 0.f;
        for (uint32_t i = 0; i < d; ++i) {
            float t = a[i] - b[i];
            s += t * t;
        }
        return s;
    }

    std::vector<ReadTask> build_tasks_(const std::vector<std::pair<uint64_t, Id>>& off_id) const {
        std::vector<ReadTask> tasks;
        if (off_id.empty()) return tasks;

        ReadTask cur;
        cur.offset = off_id[0].first;
        cur.nvec = 1;
        cur.ids.push_back(off_id[0].second);

        for (size_t i = 1; i < off_id.size(); ++i) {
            uint64_t expected = cur.offset + static_cast<uint64_t>(cur.nvec) * vec_bytes_;
            bool contiguous = (off_id[i].first == expected);
            bool can_merge = (cur.nvec < opt_.max_merge_vecs);
            if (contiguous && can_merge) {
                ++cur.nvec;
                cur.ids.push_back(off_id[i].second);
            } else {
                tasks.push_back(std::move(cur));
                cur = ReadTask{};
                cur.offset = off_id[i].first;
                cur.nvec = 1;
                cur.ids.push_back(off_id[i].second);
            }
        }
        tasks.push_back(std::move(cur));
        return tasks;
    }

private:
    int fd_{-1};
    io_uring ring_{};

    uint32_t dim_;
    uint32_t vec_bytes_;
    std::vector<uint64_t> id_to_offset_;
    IoUringRefinerOptions opt_;
};

} // namespace

std::unique_ptr<Refiner> make_io_uring_refiner(const std::string& vec_file,
                                               uint32_t dim,
                                               std::vector<uint64_t> id_to_disk_offset_bytes,
                                               IoUringRefinerOptions opt) {
    return std::make_unique<IoUringRefiner>(vec_file, dim, std::move(id_to_disk_offset_bytes), opt);
}

} // namespace ppq