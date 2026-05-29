// src/planner/planner.cpp
#include "predicatepq/planner.hpp"

namespace ppq {

PlanDecision Planner::decide(
    uint64_t N, uint64_t Nscan, uint64_t Bpre, uint64_t Bpost, float est_global_sel) const {
  float cost_pre = N * cfg_.C_scalar + (N * est_global_sel) * cfg_.C_pq + Bpre * cfg_.C_io;
  float cost_post = Nscan * (cfg_.C_pq + cfg_.C_scalar) + Bpost * cfg_.C_io;
  float fw = cost_pre / (cost_post + 1e-6f);

  PlanDecision d;
  d.est_global_selectivity = est_global_sel;
  d.est_filtered_workload = fw;
  d.type = (fw <= cfg_.tau ? PlanType::PreFilter : PlanType::PostFilter);
  return d;
}

} // namespace ppq