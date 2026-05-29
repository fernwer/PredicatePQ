#pragma once
#include "types.hpp"
#include <arrow/api.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ppq {

using ScalarValue = std::variant<int64_t, double, std::string, bool>;

struct ScalarRow {
    std::unordered_map<std::string, ScalarValue> values;
};

class ScalarStore {
public:
    virtual ~ScalarStore() = default;

    virtual uint64_t size() const = 0;
    virtual std::vector<Id> all_ids() const = 0;

    virtual std::vector<Id> scan_ids(const std::string& predicate_sql) const = 0;
    virtual bool eval_id(Id id, const std::string& predicate_sql) const = 0;
    virtual float estimate_selectivity(const std::string& predicate_sql, size_t sample_n = 4096) const = 0;

    // update path
    virtual Id append_row(const ScalarRow& row) = 0;
    virtual void append_rows(const std::vector<ScalarRow>& rows) = 0;
    virtual void mark_deleted(Id id) = 0;
    virtual bool is_deleted(Id id) const = 0;

    // return old_id -> new_id, deleted -> -1
    virtual std::vector<int64_t> compact() = 0;
};

std::shared_ptr<ScalarStore> make_arrow_scalar_store(const std::shared_ptr<arrow::Table>& table);

} // namespace ppq