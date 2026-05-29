#include "predicatepq/simd_transpose.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    using clk = std::chrono::high_resolution_clock;

    const std::vector<uint32_t> batches = {128, 512, 2048, 8192, 16384};
    const uint32_t code_size = 16; // e.g., M=16, nbits=8 => 16 bytes

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> ud(0, 255);

    std::cout << "batch,code_size,us\n";
    for (auto b : batches) {
        std::vector<uint8_t> row(static_cast<size_t>(b) * code_size);
        std::vector<uint8_t> soa(static_cast<size_t>(b) * code_size);
        for (auto& x : row) x = static_cast<uint8_t>(ud(rng));

        // warmup
        for (int i = 0; i < 100; ++i) {
            ppq::transpose_codes_on_the_fly(row.data(), b, code_size, soa.data());
        }

        auto t0 = clk::now();
        const int rounds = 2000;
        for (int r = 0; r < rounds; ++r) {
            ppq::transpose_codes_on_the_fly(row.data(), b, code_size, soa.data());
        }
        auto t1 = clk::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / rounds;
        std::cout << b << "," << code_size << "," << us << "\n";
    }
    return 0;
}