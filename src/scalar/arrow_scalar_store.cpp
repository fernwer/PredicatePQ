#include "predicatepq/scalar_store.hpp"
#include <arrow/api.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string_view>

namespace ppq {
namespace {

enum class TokType { Ident, Number, String, Bool, And, Or, In, Eq, Ne, Lt, Le, Gt, Ge, LParen, RParen, Comma, End };

struct Token {
    TokType t;
    std::string s;
};

class Lexer {
public:
    explicit Lexer(std::string_view sv) : sv_(sv) {
    }

    Token next() {
        skip_ws();
        if (p_ >= sv_.size()) return {TokType::End, ""};
        char c = sv_[p_];

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t st = p_++;
            while (p_ < sv_.size()) {
                char x = sv_[p_];
                if (!(std::isalnum(static_cast<unsigned char>(x)) || x == '_')) break;
                ++p_;
            }
            std::string w(sv_.substr(st, p_ - st));
            std::string u = upper(w);
            if (u == "AND") return {TokType::And, w};
            if (u == "OR") return {TokType::Or, w};
            if (u == "IN") return {TokType::In, w};
            if (u == "TRUE" || u == "FALSE") return {TokType::Bool, u};
            return {TokType::Ident, w};
        }

        if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+') {
            size_t st = p_++;
            bool dot = false;
            while (p_ < sv_.size()) {
                char x = sv_[p_];
                if (x == '.') {
                    dot = true;
                    ++p_;
                    continue;
                }
                if (!std::isdigit(static_cast<unsigned char>(x))) break;
                ++p_;
            }
            return {TokType::Number, std::string(sv_.substr(st, p_ - st))};
        }

        if (c == '\'' || c == '"') {
            char q = c;
            ++p_;
            std::string out;
            while (p_ < sv_.size()) {
                char x = sv_[p_++];
                if (x == q) break;
                if (x == '\\' && p_ < sv_.size())
                    out.push_back(sv_[p_++]);
                else
                    out.push_back(x);
            }
            return {TokType::String, out};
        }

        ++p_;
        switch (c) {
            case '(': return {TokType::LParen, "("};
            case ')': return {TokType::RParen, ")"};
            case ',': return {TokType::Comma, ","};
            case '=': return {TokType::Eq, "="};
            case '!':
                if (peek('=')) {
                    ++p_;
                    return {TokType::Ne, "!="};
                }
                break;
            case '<':
                if (peek('=')) {
                    ++p_;
                    return {TokType::Le, "<="};
                }
                return {TokType::Lt, "<"};
            case '>':
                if (peek('=')) {
                    ++p_;
                    return {TokType::Ge, ">="};
                }
                return {TokType::Gt, ">"};
            default: break;
        }
        throw std::runtime_error("Invalid token in predicate");
    }

private:
    bool peek(char x) const {
        return p_ < sv_.size() && sv_[p_] == x;
    }
    void skip_ws() {
        while (p_ < sv_.size() && std::isspace(static_cast<unsigned char>(sv_[p_]))) ++p_;
    }
    static std::string upper(const std::string& s) {
        std::string u = s;
        for (char& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return u;
    }
    std::string_view sv_;
    size_t p_{0};
};

enum class Op { Eq, Ne, Lt, Le, Gt, Ge, In };

struct Expr {
    virtual ~Expr() = default;
    virtual bool eval(const std::unordered_map<std::string, ScalarValue>& row) const = 0;
};

using ExprPtr = std::unique_ptr<Expr>;

static bool to_double(const ScalarValue& v, double& out) {
    if (auto p = std::get_if<int64_t>(&v)) {
        out = static_cast<double>(*p);
        return true;
    }
    if (auto p = std::get_if<double>(&v)) {
        out = *p;
        return true;
    }
    if (auto p = std::get_if<bool>(&v)) {
        out = *p ? 1.0 : 0.0;
        return true;
    }
    return false;
}

static bool scalar_eq(const ScalarValue& a, const ScalarValue& b) {
    double da, db;
    if (to_double(a, da) && to_double(b, db)) return da == db;
    if (auto sa = std::get_if<std::string>(&a)) {
        if (auto sb = std::get_if<std::string>(&b)) return *sa == *sb;
    }
    if (auto ba = std::get_if<bool>(&a)) {
        if (auto bb = std::get_if<bool>(&b)) return *ba == *bb;
    }
    return false;
}

static bool scalar_cmp(const ScalarValue& a, const ScalarValue& b, Op op) {
    double da, db;
    if (to_double(a, da) && to_double(b, db)) {
        switch (op) {
            case Op::Lt: return da < db;
            case Op::Le: return da <= db;
            case Op::Gt: return da > db;
            case Op::Ge: return da >= db;
            default: return false;
        }
    }
    if (auto sa = std::get_if<std::string>(&a)) {
        if (auto sb = std::get_if<std::string>(&b)) {
            switch (op) {
                case Op::Lt: return *sa < *sb;
                case Op::Le: return *sa <= *sb;
                case Op::Gt: return *sa > *sb;
                case Op::Ge: return *sa >= *sb;
                default: return false;
            }
        }
    }
    return false;
}

struct CmpExpr final : Expr {
    std::string col;
    Op op;
    ScalarValue rhs;
    std::vector<ScalarValue> rhs_list;

    bool eval(const std::unordered_map<std::string, ScalarValue>& row) const override {
        auto it = row.find(col);
        if (it == row.end()) return false;
        const auto& lhs = it->second;

        if (op == Op::In) {
            for (const auto& x : rhs_list) {
                if (scalar_eq(lhs, x)) return true;
            }
            return false;
        }
        if (op == Op::Eq) return scalar_eq(lhs, rhs);
        if (op == Op::Ne) return !scalar_eq(lhs, rhs);
        return scalar_cmp(lhs, rhs, op);
    }
};

struct AndExpr final : Expr {
    ExprPtr l, r;
    bool eval(const std::unordered_map<std::string, ScalarValue>& row) const override {
        return l->eval(row) && r->eval(row);
    }
};
struct OrExpr final : Expr {
    ExprPtr l, r;
    bool eval(const std::unordered_map<std::string, ScalarValue>& row) const override {
        return l->eval(row) || r->eval(row);
    }
};

class Parser {
public:
    explicit Parser(std::string_view sv) : lex_(sv) {
        cur_ = lex_.next();
    }

    ExprPtr parse_expr() {
        return parse_or();
    }

private:
    ExprPtr parse_or() {
        auto lhs = parse_and();
        while (cur_.t == TokType::Or) {
            consume(TokType::Or);
            auto rhs = parse_and();
            auto n = std::make_unique<OrExpr>();
            n->l = std::move(lhs);
            n->r = std::move(rhs);
            lhs = std::move(n);
        }
        return lhs;
    }

    ExprPtr parse_and() {
        auto lhs = parse_factor();
        while (cur_.t == TokType::And) {
            consume(TokType::And);
            auto rhs = parse_factor();
            auto n = std::make_unique<AndExpr>();
            n->l = std::move(lhs);
            n->r = std::move(rhs);
            lhs = std::move(n);
        }
        return lhs;
    }

    ExprPtr parse_factor() {
        if (cur_.t == TokType::LParen) {
            consume(TokType::LParen);
            auto e = parse_expr();
            consume(TokType::RParen);
            return e;
        }
        return parse_cmp();
    }

    ExprPtr parse_cmp() {
        std::string col = expect_ident();

        if (cur_.t == TokType::In) {
            consume(TokType::In);
            consume(TokType::LParen);
            std::vector<ScalarValue> vals;
            vals.push_back(parse_literal());
            while (cur_.t == TokType::Comma) {
                consume(TokType::Comma);
                vals.push_back(parse_literal());
            }
            consume(TokType::RParen);
            auto e = std::make_unique<CmpExpr>();
            e->col = std::move(col);
            e->op = Op::In;
            e->rhs_list = std::move(vals);
            return e;
        }

        Op op;
        switch (cur_.t) {
            case TokType::Eq: op = Op::Eq; break;
            case TokType::Ne: op = Op::Ne; break;
            case TokType::Lt: op = Op::Lt; break;
            case TokType::Le: op = Op::Le; break;
            case TokType::Gt: op = Op::Gt; break;
            case TokType::Ge: op = Op::Ge; break;
            default: throw std::runtime_error("Expected comparison operator");
        }
        cur_ = lex_.next();

        auto rhs = parse_literal();
        auto e = std::make_unique<CmpExpr>();
        e->col = std::move(col);
        e->op = op;
        e->rhs = std::move(rhs);
        return e;
    }

    ScalarValue parse_literal() {
        if (cur_.t == TokType::Number) {
            const std::string s = cur_.s;
            cur_ = lex_.next();
            if (s.find('.') != std::string::npos) return std::stod(s);
            return static_cast<int64_t>(std::stoll(s));
        }
        if (cur_.t == TokType::String) {
            std::string s = cur_.s;
            cur_ = lex_.next();
            return s;
        }
        if (cur_.t == TokType::Bool) {
            std::string u = cur_.s;
            cur_ = lex_.next();
            return (u == "TRUE");
        }
        throw std::runtime_error("Expected literal");
    }

    std::string expect_ident() {
        if (cur_.t != TokType::Ident) throw std::runtime_error("Expected column ident");
        std::string s = cur_.s;
        cur_ = lex_.next();
        return s;
    }

    void consume(TokType t) {
        if (cur_.t != t) throw std::runtime_error("Unexpected token");
        cur_ = lex_.next();
    }

    Lexer lex_;
    Token cur_;
};

class ArrowScalarStore final : public ScalarStore {
public:
    explicit ArrowScalarStore(const std::shared_ptr<arrow::Table>& table) {
        import_table_(table);
        tombstone_.assign(rows_.size(), 0);
    }

    uint64_t size() const override {
        return rows_.size();
    }

    std::vector<Id> all_ids() const override {
        std::vector<Id> out;
        out.reserve(rows_.size());
        for (Id i = 0; i < static_cast<Id>(rows_.size()); ++i) {
            if (!tombstone_[i]) out.push_back(i);
        }
        return out;
    }

    std::vector<Id> scan_ids(const std::string& predicate_sql) const override {
        auto ast = parse_(predicate_sql);
        std::vector<Id> out;
        out.reserve(rows_.size() / 4 + 16);
        for (Id i = 0; i < static_cast<Id>(rows_.size()); ++i) {
            if (tombstone_[i]) continue;
            if (ast->eval(rows_[i])) out.push_back(i);
        }
        return out;
    }

    bool eval_id(Id id, const std::string& predicate_sql) const override {
        if (id >= rows_.size() || tombstone_[id]) return false;
        auto ast = parse_(predicate_sql);
        return ast->eval(rows_[id]);
    }

    float estimate_selectivity(const std::string& predicate_sql, size_t sample_n) const override {
        auto ast = parse_(predicate_sql);
        std::vector<Id> alive;
        alive.reserve(rows_.size());
        for (Id i = 0; i < static_cast<Id>(rows_.size()); ++i)
            if (!tombstone_[i]) alive.push_back(i);
        if (alive.empty()) return 0.0f;

        sample_n = std::min(sample_n, alive.size());
        std::mt19937_64 rng(42);
        std::uniform_int_distribution<size_t> dist(0, alive.size() - 1);

        size_t hit = 0;
        for (size_t i = 0; i < sample_n; ++i) {
            Id id = alive[dist(rng)];
            if (ast->eval(rows_[id])) ++hit;
        }
        return static_cast<float>(hit) / static_cast<float>(sample_n);
    }

    Id append_row(const ScalarRow& row) override {
        rows_.push_back(row.values);
        tombstone_.push_back(0);
        return static_cast<Id>(rows_.size() - 1);
    }

    void append_rows(const std::vector<ScalarRow>& rows) override {
        rows_.reserve(rows_.size() + rows.size());
        tombstone_.reserve(tombstone_.size() + rows.size());
        for (const auto& r : rows) {
            rows_.push_back(r.values);
            tombstone_.push_back(0);
        }
    }

    void mark_deleted(Id id) override {
        if (id < tombstone_.size()) tombstone_[id] = 1;
    }

    bool is_deleted(Id id) const override {
        if (id >= tombstone_.size()) return true;
        return tombstone_[id] != 0;
    }

    std::vector<int64_t> compact() override {
        std::vector<int64_t> remap(rows_.size(), -1);
        std::vector<std::unordered_map<std::string, ScalarValue>> new_rows;
        new_rows.reserve(rows_.size());

        int64_t nid = 0;
        for (size_t i = 0; i < rows_.size(); ++i) {
            if (tombstone_[i]) continue;
            remap[i] = nid++;
            new_rows.push_back(std::move(rows_[i]));
        }
        rows_.swap(new_rows);
        tombstone_.assign(rows_.size(), 0);
        return remap;
    }

private:
    static ExprPtr parse_(const std::string& sql) {
        Parser p(sql);
        return p.parse_expr();
    }

    void import_table_(const std::shared_ptr<arrow::Table>& table) {
        if (!table) return;
        const auto nrows = table->num_rows();
        const auto ncols = table->num_columns();
        rows_.assign(static_cast<size_t>(nrows), {});

        for (int c = 0; c < ncols; ++c) {
            const auto name = table->field(c)->name();
            auto carr = table->column(c); // ChunkedArray

            int64_t base = 0;
            for (const auto& chunk : carr->chunks()) {
                switch (chunk->type_id()) {
                    case arrow::Type::INT64: {
                        auto a = std::static_pointer_cast<arrow::Int64Array>(chunk);
                        for (int64_t i = 0; i < a->length(); ++i) {
                            if (!a->IsNull(i)) rows_[base + i][name] = a->Value(i);
                        }
                        break;
                    }
                    case arrow::Type::DOUBLE: {
                        auto a = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                        for (int64_t i = 0; i < a->length(); ++i) {
                            if (!a->IsNull(i)) rows_[base + i][name] = a->Value(i);
                        }
                        break;
                    }
                    case arrow::Type::STRING: {
                        auto a = std::static_pointer_cast<arrow::StringArray>(chunk);
                        for (int64_t i = 0; i < a->length(); ++i) {
                            if (!a->IsNull(i)) rows_[base + i][name] = a->GetString(i);
                        }
                        break;
                    }
                    case arrow::Type::BOOL: {
                        auto a = std::static_pointer_cast<arrow::BooleanArray>(chunk);
                        for (int64_t i = 0; i < a->length(); ++i) {
                            if (!a->IsNull(i)) rows_[base + i][name] = a->Value(i);
                        }
                        break;
                    }
                    default: throw std::runtime_error("Unsupported Arrow type in ScalarStore");
                }
                base += chunk->length();
            }
        }
    }

private:
    std::vector<std::unordered_map<std::string, ScalarValue>> rows_;
    std::vector<uint8_t> tombstone_;
};

} // namespace

std::shared_ptr<ScalarStore> make_arrow_scalar_store(const std::shared_ptr<arrow::Table>& table) {
    return std::make_shared<ArrowScalarStore>(table);
}

} // namespace ppq