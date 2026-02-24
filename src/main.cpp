#include "PredicatePQ.h"
#include <iostream>
#include <vector>
#include <random>

struct MyFilter : public ScalarFilter {
    std::vector<int> attributes;
    int target_val;

    MyFilter(int n, int target) : attributes(n), target_val(target) {
        std::mt19937 rng(42);
        for (int i = 0; i < n; ++i) attributes[i] = rng() % 100;
    }

    bool test(int64_t id) const override {
        return attributes[id] < target_val;
    }
};

int main() {
    int d = 128;
    int n = 100000;
    int nq = 1;
    int num_clusters = 100;
    int M = 16;
    int nbits = 8;

    std::cout << "Generating Synthetic Data..." << std::endl;
    std::vector<float> database(n * d);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    for (int i = 0; i < n * d; ++i) {
        database[i] = distrib(rng);
    }

    std::vector<float> query(d);
    for (int i = 0; i < d; ++i) query[i] = distrib(rng);

    PredicatePQ system(d, num_clusters, M, nbits, "vectors_data.bin");
    system.train_and_add(database.data(), n);

    std::cout << "\n=== Test 1: High Selectivity (10% pass) ===" << std::endl;
    MyFilter filter_high(n, 10);
    auto res1 = system.search(query.data(), filter_high, 10, 16);
    for (const auto& r : res1) {
        std::cout << "ID: " << r.id << " | Dist: " << r.distance << std::endl;
    }

    std::cout << "\n=== Test 2: Low Selectivity (90% pass) ===" << std::endl;
    MyFilter filter_low(n, 90);
    auto res2 = system.search(query.data(), filter_low, 10, 16);
    for (const auto& r : res2) {
        std::cout << "ID: " << r.id << " | Dist: " << r.distance << std::endl;
    }

    return 0;
}