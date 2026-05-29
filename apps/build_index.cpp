#include "predicatepq/metadata.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string vec_fbin; // [ntotal * dim] float32
    std::string out_dir;
    uint64_t ntotal = 0;
    uint32_t dim = 0;

    uint32_t nlist = 4096;
    uint32_t pq_m = 16;
    uint32_t pq_nbits = 8;
    uint64_t train_n = 200000;

    // optional scalar metadata
    std::string scalar_parquet; // optional
};

static Args parse_args(int argc, char** argv) {
    if (argc < 10) {
        throw std::runtime_error(
            "Usage:\n"
            "  build_index <vec_fbin> <out_dir> <ntotal> <dim> <nlist> <pq_m> <pq_nbits> <train_n> <scalar_parquet_or_->\n");
    }
    Args a;
    a.vec_fbin = argv[1];
    a.out_dir = argv[2];
    a.ntotal = std::stoull(argv[3]);
    a.dim = static_cast<uint32_t>(std::stoul(argv[4]));
    a.nlist = static_cast<uint32_t>(std::stoul(argv[5]));
    a.pq_m = static_cast<uint32_t>(std::stoul(argv[6]));
    a.pq_nbits = static_cast<uint32_t>(std::stoul(argv[7]));
    a.train_n = std::stoull(argv[8]);
    a.scalar_parquet = argv[9];
    if (a.scalar_parquet == "-") a.scalar_parquet.clear();
    return a;
}

template <typename T>
static void write_bin(const std::string& path, const std::vector<T>& v) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("open out file failed: " + path);
    fout.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

static std::vector<float> load_fbin(const std::string& path, uint64_t n, uint32_t d) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("open vec file failed");
    std::vector<float> x(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(x.size() * sizeof(float)));
    if (!fin) throw std::runtime_error("read vec file failed");
    return x;
}

static std::shared_ptr<arrow::Table> read_parquet_table(const std::string& path) {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    auto st = arrow::io::ReadableFile::Open(path);
    if (!st.ok()) throw std::runtime_error("open parquet failed: " + st.status().ToString());
    infile = *st;

    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto pst = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
    if (!pst.ok()) throw std::runtime_error("parquet::OpenFile failed: " + pst.ToString());

    std::shared_ptr<arrow::Table> table;
    auto rst = reader->ReadTable(&table);
    if (!rst.ok()) throw std::runtime_error("ReadTable failed: " + rst.ToString());
    return table;
}

static void write_arrow_ipc(const std::shared_ptr<arrow::Table>& table, const std::string& path) {
    auto of = arrow::io::FileOutputStream::Open(path);
    if (!of.ok()) throw std::runtime_error("open arrow ipc out failed: " + of.status().ToString());

    auto writer_res = arrow::ipc::MakeFileWriter((*of).get(), table->schema());
    if (!writer_res.ok()) throw std::runtime_error("MakeFileWriter failed: " + writer_res.status().ToString());

    auto writer = *writer_res;
    auto st = writer->WriteTable(*table);
    if (!st.ok()) throw std::runtime_error("WriteTable failed: " + st.ToString());
    st = writer->Close();
    if (!st.ok()) throw std::runtime_error("ipc writer close failed: " + st.ToString());
    st = (*of)->Close();
    if (!st.ok()) throw std::runtime_error("FileOutputStream close failed: " + st.ToString());
}

static std::shared_ptr<arrow::Table> add_id_column(const std::shared_ptr<arrow::Table>& t, uint64_t ntotal) {
    arrow::UInt32Builder b;
    auto r = b.Reserve(static_cast<int64_t>(ntotal));
    if (!r.ok()) throw std::runtime_error("reserve id builder failed");
    for (uint64_t i = 0; i < ntotal; ++i) b.UnsafeAppend(static_cast<uint32_t>(i));

    std::shared_ptr<arrow::Array> id_arr;
    auto f = b.Finish(&id_arr);
    if (!f.ok()) throw std::runtime_error("finish id array failed");

    auto chunked = std::make_shared<arrow::ChunkedArray>(id_arr);
    auto field = arrow::field("__id", arrow::uint32());

    auto t2res = t->AddColumn(t->num_columns(), field, chunked);
    if (!t2res.ok()) throw std::runtime_error("AddColumn __id failed: " + t2res.status().ToString());
    return *t2res;
}

static std::shared_ptr<arrow::Table> reorder_table_by_perm(const std::shared_ptr<arrow::Table>& table,
                                                           const std::vector<uint32_t>& perm) {
    // perm[new_pos] = old_id, 所以 take(table, perm) 得到 cluster-ordered table
    arrow::UInt32Builder ib;
    auto rs = ib.Reserve(static_cast<int64_t>(perm.size()));
    if (!rs.ok()) throw std::runtime_error("reserve perm builder failed");
    for (auto v : perm) ib.UnsafeAppend(v);

    std::shared_ptr<arrow::Array> idx_arr;
    auto fs = ib.Finish(&idx_arr);
    if (!fs.ok()) throw std::runtime_error("finish idx arr failed");

    auto idx_d = arrow::Datum(idx_arr);

    std::vector<std::shared_ptr<arrow::ChunkedArray>> out_cols;
    out_cols.reserve(table->num_columns());
    for (int c = 0; c < table->num_columns(); ++c) {
        auto col = table->column(c);
        auto take_res = arrow::compute::Take(arrow::Datum(col), idx_d);
        if (!take_res.status().ok()) {
            throw std::runtime_error("arrow::compute::Take failed: " + take_res.status().ToString());
        }
        auto out = take_res->chunked_array();
        out_cols.push_back(out);
    }
    return arrow::Table::Make(table->schema(), out_cols);
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        fs::create_directories(args.out_dir);

        std::cout << "[1/9] loading vectors..." << std::endl;
        auto xb = load_fbin(args.vec_fbin, args.ntotal, args.dim);

        std::cout << "[2/9] train/build IVFPQ..." << std::endl;
        faiss::IndexFlatL2 quantizer(args.dim);
        faiss::IndexIVFPQ index(&quantizer, args.dim, args.nlist, args.pq_m, args.pq_nbits);
        index.metric_type = faiss::METRIC_L2;

        uint64_t train_n = std::min(args.train_n, args.ntotal);
        index.train(train_n, xb.data());
        index.add(args.ntotal, xb.data());

        std::cout << "[3/9] extract id_to_cluster / id_to_local..." << std::endl;
        std::vector<uint16_t> id_to_cluster(args.ntotal, 0);
        std::vector<uint32_t> id_to_local(args.ntotal, 0);
        std::vector<uint32_t> cluster_counts(args.nlist, 0);

        for (uint32_t c = 0; c < args.nlist; ++c) {
            size_t list_sz = index.invlists->list_size(c);
            cluster_counts[c] = static_cast<uint32_t>(list_sz);
            for (size_t j = 0; j < list_sz; ++j) {
                faiss::idx_t id = index.invlists->get_single_id(c, j);
                if (id < 0 || static_cast<uint64_t>(id) >= args.ntotal) continue;
                id_to_cluster[static_cast<size_t>(id)] = static_cast<uint16_t>(c);
                id_to_local[static_cast<size_t>(id)] = static_cast<uint32_t>(j);
            }
        }

        std::vector<uint64_t> cluster_offsets(args.nlist + 1, 0);
        for (uint32_t c = 0; c < args.nlist; ++c) cluster_offsets[c + 1] = cluster_offsets[c] + cluster_counts[c];

        std::cout << "[4/9] build permutation old<->new..." << std::endl;
        // new_pos -> old_id
        std::vector<uint32_t> perm(args.ntotal, 0);
        // old_id -> new_pos
        std::vector<uint32_t> old_to_new(args.ntotal, 0);

        for (uint64_t old = 0; old < args.ntotal; ++old) {
            uint16_t c = id_to_cluster[old];
            uint32_t j = id_to_local[old];
            uint64_t new_pos = cluster_offsets[c] + j;
            perm[new_pos] = static_cast<uint32_t>(old);
            old_to_new[old] = static_cast<uint32_t>(new_pos);
        }

        std::cout << "[5/9] write cluster-ordered vectors..." << std::endl;
        std::vector<float> x_clustered(static_cast<size_t>(args.ntotal) * args.dim);
        for (uint64_t new_pos = 0; new_pos < args.ntotal; ++new_pos) {
            uint32_t old = perm[new_pos];
            const float* src = xb.data() + static_cast<size_t>(old) * args.dim;
            float* dst = x_clustered.data() + static_cast<size_t>(new_pos) * args.dim;
            std::copy(src, src + args.dim, dst);
        }
        write_bin(args.out_dir + "/vectors_clustered.fbin", x_clustered);

        std::vector<uint64_t> id_to_disk_offset(args.ntotal, 0);
        for (uint64_t old = 0; old < args.ntotal; ++old) {
            uint64_t new_pos = old_to_new[old];
            id_to_disk_offset[old] = new_pos * args.dim * sizeof(float);
        }

        std::cout << "[6/9] dump index + metadata..." << std::endl;
        write_bin(args.out_dir + "/id_to_cluster.bin", id_to_cluster);
        write_bin(args.out_dir + "/id_to_local.bin", id_to_local);
        write_bin(args.out_dir + "/cluster_offsets.bin", cluster_offsets);
        write_bin(args.out_dir + "/cluster_counts.bin", cluster_counts);
        write_bin(args.out_dir + "/old_to_new.bin", old_to_new);
        write_bin(args.out_dir + "/perm_new_to_old.bin", perm);
        write_bin(args.out_dir + "/id_to_disk_offset.bin", id_to_disk_offset);

        faiss::write_index(&index, (args.out_dir + "/ivfpq.faiss").c_str());

        // row-wise PQ code by old_id
        const uint32_t code_size = index.pq.code_size;
        std::vector<uint8_t> pq_codes(args.ntotal * code_size, 0);
        for (uint32_t c = 0; c < args.nlist; ++c) {
            size_t list_sz = index.invlists->list_size(c);
            const uint8_t* list_codes = index.invlists->get_codes(c);
            for (size_t j = 0; j < list_sz; ++j) {
                faiss::idx_t id = index.invlists->get_single_id(c, j);
                if (id < 0 || static_cast<uint64_t>(id) >= args.ntotal) continue;
                const uint8_t* src = list_codes + j * code_size;
                uint8_t* dst = pq_codes.data() + static_cast<size_t>(id) * code_size;
                std::copy(src, src + code_size, dst);
            }
            index.invlists->release_codes(c, list_codes);
        }
        write_bin(args.out_dir + "/pq_codes_rowwise.bin", pq_codes);

        std::cout << "[7/9] dump centroids..." << std::endl;
        // coarse centroids
        std::vector<float> coarse_centroids(static_cast<size_t>(args.nlist) * args.dim, 0.f);
        auto* cq = dynamic_cast<faiss::IndexFlat*>(index.quantizer);
        if (!cq) throw std::runtime_error("quantizer is not IndexFlat");
        std::copy(cq->get_xb(), cq->get_xb() + static_cast<size_t>(args.nlist) * args.dim, coarse_centroids.data());
        write_bin(args.out_dir + "/coarse_centroids.fbin", coarse_centroids);

        std::cout << "[8/9] optional scalar parquet import..." << std::endl;
        if (!args.scalar_parquet.empty()) {
            auto t = read_parquet_table(args.scalar_parquet);
            if (static_cast<uint64_t>(t->num_rows()) != args.ntotal) {
                throw std::runtime_error("scalar parquet num_rows != ntotal");
            }
            auto t_with_id = add_id_column(t, args.ntotal);
            write_arrow_ipc(t_with_id, args.out_dir + "/scalar_original.arrow");

            // cluster-ordered table (same order as vectors_clustered)
            auto t_clustered = reorder_table_by_perm(t_with_id, perm);
            write_arrow_ipc(t_clustered, args.out_dir + "/scalar_clustered.arrow");
        }

        std::cout << "[9/9] done: " << args.out_dir << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "build_index failed: " << e.what() << std::endl;
        return 1;
    }
}