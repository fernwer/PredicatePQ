#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <span>

namespace ppq {

using Id = uint32_t;
using ClusterId = uint16_t;

struct Query {
  std::vector<float> qvec;
  std::string predicate_sql; // e.g. "price >= 10 AND category = 'shoe'"
  uint32_t topk{100};
};

struct Result {
  Id id;
  float distance;
};

enum class PlanType : uint8_t { PreFilter, PostFilter };

struct PlanDecision {
  PlanType type;
  float est_global_selectivity;
  float est_filtered_workload;
};

} // namespace ppq