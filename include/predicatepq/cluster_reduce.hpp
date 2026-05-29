// include/predicatepq/cluster_reduce.hpp
#pragma once
#include "types.hpp"
#include <vector>

namespace ppq {

struct ClusterReduceOutput {
  std::vector<uint32_t> counts;   // K
  std::vector<Id> ids_out;        // compacted ids
  std::vector<uint64_t> offsets;  // K+1
};

class ClusterReducer {
public:
  static ClusterReduceOutput run(
      const std::vector<Id>& ids_in,
      const std::vector<ClusterId>& id_to_cluster,
      uint32_t K);
};

} // namespace ppq