#include "predicatepq/predicate.hpp"
#include "predicatepq/scalar_store.hpp"
#include "predicatepq/simd_transpose.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

static void print_help() {
    std::cout <<
        R"(PredicatePQ prototype

Usage:
  predicatepq --help
  predicatepq --version
  predicatepq run [--config <path>]

Examples:
  predicatepq run --config configs/default.yaml
)";
}

static void print_version() {
    std::cout << "PredicatePQ prototype v0.1.0\n";
#ifdef PREDICATEPQ_USE_IO_URING
    std::cout << "io_uring: enabled\n";
#else
    std::cout << "io_uring: disabled\n";
#endif
}

static std::string get_opt(int argc, char** argv, const std::string& key, const std::string& defv) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return argv[i + 1];
    }
    return defv;
}

static bool run_basic(const std::string& config_path) {
    std::cout << "[1] Loading config: " << config_path << "\n";
    if (!fs::exists(config_path)) {
        std::cerr << "Config not found: " << config_path << "\n";
        return false;
    }

    YAML::Node cfg = YAML::LoadFile(config_path);

    // 打印关键配置
    auto nlist = cfg["index"] && cfg["index"]["nlist"] ? cfg["index"]["nlist"].as<int>() : -1;
    auto dim = cfg["index"] && cfg["index"]["dim"] ? cfg["index"]["dim"].as<int>() : -1;
    auto tau = cfg["planner"] && cfg["planner"]["tau"] ? cfg["planner"]["tau"].as<double>() : -1.0;
    auto nprobe = cfg["engine"] && cfg["engine"]["nprobe"] ? cfg["engine"]["nprobe"].as<int>() : -1;

    std::cout << "[config] dim=" << dim << ", nlist=" << nlist << ", tau=" << tau << ", nprobe=" << nprobe << "\n";

    // [2] predicate quick check
    std::cout << "[2] Predicate quick check...\n";
    std::unordered_map<std::string, ppq::ScalarValue> row{
        {"price", int64_t(120)}, {"category", std::string("shoe")}, {"in_stock", true}};

    const std::string expr = "price >= 100 AND (category IN ('shoe','coat') OR in_stock = TRUE)";
    bool ok = ppq::pred::evaluate(expr, row);
    std::cout << "  expr: " << expr << "\n";
    std::cout << "  eval: " << (ok ? "true" : "false") << "\n";

    // [3] SIMD transpose quick check
    std::cout << "[3] SIMD transpose quick check...\n";
    constexpr uint32_t batch = 1024;
    constexpr uint32_t code_size = 16;
    std::vector<uint8_t> row_codes(static_cast<size_t>(batch) * code_size);
    std::vector<uint8_t> soa_out(static_cast<size_t>(batch) * code_size);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> ud(0, 255);
    for (auto& x : row_codes) x = static_cast<uint8_t>(ud(rng));

    // warmup
    for (int i = 0; i < 20; ++i) {
        ppq::transpose_codes_on_the_fly(row_codes.data(), batch, code_size, soa_out.data());
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; ++i) {
        ppq::transpose_codes_on_the_fly(row_codes.data(), batch, code_size, soa_out.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 200.0;

    std::cout << "  transpose avg: " << us << " us (batch=" << batch << ", code_size=" << code_size << ")\n";

    std::cout << "[done] Basic execution finished.\n";
    return true;
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        print_help();
        return 0;
    }

    const std::string cmd = argv[1];
    if (cmd == "--help" || cmd == "-h") {
        print_help();
        return 0;
    }
    if (cmd == "--version" || cmd == "-v") {
        print_version();
        return 0;
    }
    if (cmd == "run") {
        const std::string cfg = get_opt(argc, argv, "--config", "configs/default.yaml");
        bool ok = run_basic(cfg);
        return ok ? 0 : 1;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    print_help();
    return 1;
