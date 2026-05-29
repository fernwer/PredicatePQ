// include/predicatepq/planner.hpp
#pragma once
#include "types.hpp"
#include <vector>

namespace ppq {

struct PlannerConfig {
  float tau = 0.5f;
  float C_scalar = 1.0f;
  float C_pq = 2.0f;
  float C_io = 5.0f;
  float sample_ratio = 0.01f;
  uint32_t N_min = 64;
};

struct ClusterStat {
  uint32_t n_sample{0};
  uint32_t n_hit{0};
  float smoothed_sel{0.0f};
};

class Planner {
public:
  explicit Planner(PlannerConfig cfg) : cfg_(cfg) {}
  PlanDecision decide(uint64_t N, uint64_t Nscan, uint64_t Bpre, uint64_t Bpost,
                      float est_global_sel) const;
  static float laplace(uint32_t m, uint32_t n) { return float(m + 1) / float(n + 2); }

private:
  PlannerConfig cfg_;
};

} // namespace ppq