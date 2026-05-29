#include "predicatepq/engine.hpp"
#include "predicatepq/planner.hpp"
#include "predicatepq/pq_scanner.hpp"
#include "predicatepq/refiner.hpp"
#include "predicatepq/scalar_store.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string index_dir;
    std::string query_fbin;
    uint64_t qid = 0;
    uint32_t topk = 100;
    bool use_io_uring = true;
    std::string predicate_sql;
};

static Args parse_args(int argc, char** argv) {
    if (argc < 7) {
        throw std::runtime_error("Usage:\n"
                                 "  query_cli <index_dir> <query_fbin> <qid> <topk> <use_io_uring:0|1> \"<predicate_sql>\"\n"
                                 "Example:\n"
                                 "  query_cli ./out ./query.fbin 0 100 1 \"price >= 100 AND category IN ('shoe','coat')\"\n");
    }
    Args a;
    a.index_dir = argv[1];
    a.query_fbin = argv[2];
    a.qid = std::stoull(argv[3]);
    a.topk = static_cast<uint32_t>(std::stoul(argv[4]));
    a.use_io_uring = (std::stoi(argv[5]) != 0);
    a.predicate_sql = argv[6];
    return a;
}

template <typename T>
static std::vector<T> read_bin_vec(const std::string& path) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("open failed: " + path);
    auto sz = fin.tellg();
    fin.seekg(0);
    if (sz < 0) throw std::runtime_error("invalid file size: " + path);
    if ((sz % static_cast<std::streamoff>(sizeof(T))) != 0) {
        throw std::runtime_error("file size not aligned with type: " + path);
    }
    std::vector<T> out(static_cast<size_t>(sz / sizeof(T)));
    fin.read(reinterpret_cast<char*>(out.data()), sz);
    if (!fin) throw std::runtime_error("read failed: " + path);
    return out;
}

static std::shared_ptr<arrow::Table> read_arrow_ipc_table(const std::string& path) {
    auto infile_res = arrow::io::ReadableFile::Open(path);
    if (!infile_res.ok()) throw std::runtime_error("open arrow file failed: " + infile_res.status().ToString());

    auto rb_res = arrow::ipc::RecordBatchFileReader::Open(*infile_res);
    if (!rb_res.ok()) throw std::runtime_error("RecordBatchFileReader::Open failed: " + rb_res.status().ToString());

    auto table_res = (*rb_res)->ReadAll();
    if (!table_res.ok()) throw std::runtime_error("ReadAll failed: " + table_res.status().ToString());

    return *table_res;
}

static std::vector<float> load_one_query(const std::string& qpath, uint32_t dim, uint64_t qid) {
    std::ifstream fin(qpath, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("open query file failed");
    auto sz = fin.tellg();
    fin.seekg(0);

    uint64_t total_float = static_cast<uint64_t>(sz / static_cast<std::streamoff>(sizeof(float)));
    if (total_float % dim != 0) throw std::runtime_error("query file size mismatch with dim");
    uint64_t nq = total_float / dim;
    if (qid >= nq) throw std::runtime_error("qid out of range");

    fin.seekg(static_cast<std::streamoff>(qid * dim * sizeof(float)), std::ios::beg);
    std::vector<float> q(dim);
    fin.read(reinterpret_cast<char*>(q.data()), static_cast<std::streamsize>(dim * sizeof(float)));
    if (!fin) throw std::runtime_error("read query failed");
    return q;
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);

        const std::string p_ivfpq = args.index_dir + "/ivfpq.faiss";
        const std::string p_codes = args.index_dir + "/pq_codes_rowwise.bin";
        const std::string p_id2c = args.index_dir + "/id_to_cluster.bin";
        const std::string p_id2off = args.index_dir + "/id_to_disk_offset.bin";
        const std::string p_centroids = args.index_dir + "/coarse_centroids.fbin";
        const std::string p_coffsets = args.index_dir + "/cluster_offsets.bin";
        const std::string p_scalar = args.index_dir + "/scalar_original.arrow";
        const std::string p_vec_clustered = args.index_dir + "/vectors_clustered.fbin";

        if (!fs::exists(p_ivfpq)) throw std::runtime_error("missing ivfpq.faiss");
        if (!fs::exists(p_codes)) throw std::runtime_error("missing pq_codes_rowwise.bin");
        if (!fs::exists(p_id2c)) throw std::runtime_error("missing id_to_cluster.bin");
        if (!fs::exists(p_id2off)) throw std::runtime_error("missing id_to_disk_offset.bin");
        if (!fs::exists(p_centroids)) throw std::runtime_error("missing coarse_centroids.fbin");
        if (!fs::exists(p_coffsets)) throw std::runtime_error("missing cluster_offsets.bin");
        if (!fs::exists(p_scalar)) throw std::runtime_error("missing scalar_original.arrow");
        if (!fs::exists(p_vec_clustered)) throw std::runtime_error("missing vectors_clustered.fbin");

        // 1) load faiss index
        std::unique_ptr<faiss::Index> base(faiss::read_index(p_ivfpq.c_str()));
        auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(base.get());
        if (!ivfpq) throw std::runtime_error("ivfpq.faiss is not IndexIVFPQ");

        uint32_t dim = static_cast<uint32_t>(ivfpq->d);
        uint32_t nlist = static_cast<uint32_t>(ivfpq->nlist);
        uint32_t M = static_cast<uint32_t>(ivfpq->pq.M);
        uint32_t nbits = static_cast<uint32_t>(ivfpq->pq.nbits);
        uint64_t ntotal = static_cast<uint64_t>(ivfpq->ntotal);

        // 2) load metadata
        auto pq_codes = read_bin_vec<uint8_t>(p_codes);
        auto id_to_cluster = read_bin_vec<uint16_t>(p_id2c);
        auto id_to_disk_offset = read_bin_vec<uint64_t>(p_id2off);
        auto coarse_centroids = read_bin_vec<float>(p_centroids);
        auto cluster_offsets = read_bin_vec<uint64_t>(p_coffsets);

        if (id_to_cluster.size() != ntotal) throw std::runtime_error("id_to_cluster size mismatch");
        if (id_to_disk_offset.size() != ntotal) throw std::runtime_error("id_to_disk_offset size mismatch");
        if (coarse_centroids.size() != static_cast<size_t>(nlist) * dim)
            throw std::runtime_error("coarse_centroids size mismatch");
        if (cluster_offsets.size() != static_cast<size_t>(nlist) + 1) throw std::runtime_error("cluster_offsets size mismatch");

        const uint32_t code_size = M * nbits / 8;
        if (pq_codes.size() != static_cast<size_t>(ntotal) * code_size) {
            throw std::runtime_error("pq_codes size mismatch");
        }

        // 3) scalar store
        auto scalar_table = read_arrow_ipc_table(p_scalar);
        if (static_cast<uint64_t>(scalar_table->num_rows()) != ntotal) {
            throw std::runtime_error("scalar rows != ntotal");
        }
        auto scalar_store = ppq::make_arrow_scalar_store(scalar_table);

        // 4) planner
        ppq::PlannerConfig pcfg;
        pcfg.tau = 0.5f;
        pcfg.C_scalar = 1.0f;
        pcfg.C_pq = 2.0f;
        pcfg.C_io = 5.0f;
        ppq::Planner planner(pcfg);

        // 5) scanner with trained PQ
        auto scanner =
            std::make_shared<ppq::PQScanner>(dim, M, nbits, std::move(pq_codes), static_cast<uint32_t>(ntotal), ivfpq->pq);

        // 6) refiner
        std::unique_ptr<ppq::Refiner> refiner;
        if (args.use_io_uring) {
            ppq::IoUringRefinerOptions opt;
            opt.queue_depth = 256;
            opt.submit_batch = 64;
            opt.max_inflight = 256;
            opt.max_merge_vecs = 16;
            opt.use_o_direct = false;
            refiner = ppq::make_io_uring_refiner(p_vec_clustered, dim, std::move(id_to_disk_offset), opt);
        } else {
            refiner = ppq::make_mmap_refiner(p_vec_clustered, dim);
        }

        // 7) engine
        ppq::EngineConfig ecfg;
        ecfg.nprobe = 32;
        ecfg.candidate_budget = 4096;
        ecfg.min_geo_floor = 16;
        ecfg.alpha = 0.7f;
        ecfg.sample_ratio = 0.01f;
        ecfg.sample_nmin = 64;

        ppq::PredicatePQEngine engine(planner,
                                      scalar_store,
                                      scanner,
                                      std::move(refiner),
                                      std::move(id_to_cluster),
                                      nlist,
                                      dim,
                                      std::move(coarse_centroids),
                                      std::move(cluster_offsets),
                                      ecfg);

        // 8) load query vector
        auto qvec = load_one_query(args.query_fbin, dim, args.qid);
        ppq::Query q;
        q.qvec = std::move(qvec);
        q.topk = args.topk;
        q.predicate_sql = args.predicate_sql;

        // 9) run
        auto t0 = std::chrono::high_resolution_clock::now();
        auto results = engine.search(q);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "Query done, latency(ms)=" << ms << "\n";
        std::cout << "Top-" << results.size() << " results:\n";
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << i << "\t" << results[i].id << "\t" << results[i].distance << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "query_cli failed: " << e.what() << std::endl;
        return 1;
    }
}