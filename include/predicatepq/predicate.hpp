#pragma once
#include "scalar_store.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ppq::pred {

enum class CompareOp { Eq, Ne, Lt, Le, Gt, Ge, In };

class Predicate {
public:
    virtual ~Predicate() = default;
    virtual bool eval(const std::unordered_map<std::string, ScalarValue>& row) const = 0;
};

using PredicatePtr = std::unique_ptr<Predicate>;

PredicatePtr compile(const std::string& expr);

bool evaluate(const Predicate& p, const std::unordered_map<std::string, ScalarValue>& row);
bool evaluate(const std::string& expr, const std::unordered_map<std::string, ScalarValue>& row);

} // namespace ppq::pred