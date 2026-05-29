#include "predicatepq/predicate.hpp"
#include <cctype>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace ppq::pred {
namespace {

enum class TokType { Ident, Number, String, Bool, And, Or, In, Eq, Ne, Lt, Le, Gt, Ge, LParen, RParen, Comma, End };

struct Token {
    TokType type;
    std::string text;
};

class Lexer {
public:
    explicit Lexer(std::string_view s) : s_(s) {
    }

    Token next() {
        skip_ws();
        if (pos_ >= s_.size()) return {TokType::End, ""};

        char c = s_[pos_];
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') return lex_ident();
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+') return lex_number();
        if (c == '\'' || c == '"') return lex_string();

        ++pos_;
        switch (c) {
            case '(': return {TokType::LParen, "("};
            case ')': return {TokType::RParen, ")"};
            case ',': return {TokType::Comma, ","};
            case '=': return {TokType::Eq, "="};
            case '!':
                if (pos_ < s_.size() && s_[pos_] == '=') {
                    ++pos_;
                    return {TokType::Ne, "!="};
                }
                break;
            case '<':
                if (pos_ < s_.size() && s_[pos_] == '=') {
                    ++pos_;
                    return {TokType::Le, "<="};
                }
                return {TokType::Lt, "<"};
            case '>':
                if (pos_ < s_.size() && s_[pos_] == '=') {
                    ++pos_;
                    return {TokType::Ge, ">="};
                }
                return {TokType::Gt, ">"};
            default: break;
        }
        throw std::runtime_error("predicate lexer: invalid token");
    }

private:
    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    static std::string upper(std::string x) {
        for (char& c : x) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return x;
    }

    Token lex_ident() {
        size_t st = pos_++;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) break;
            ++pos_;
        }
        std::string w(s_.substr(st, pos_ - st));
        std::string u = upper(w);
        if (u == "AND") return {TokType::And, w};
        if (u == "OR") return {TokType::Or, w};
        if (u == "IN") return {TokType::In, w};
        if (u == "TRUE" || u == "FALSE") return {TokType::Bool, u};
        return {TokType::Ident, w};
    }

    Token lex_number() {
        size_t st = pos_++;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '.')) break;
            ++pos_;
        }
        return {TokType::Number, std::string(s_.substr(st, pos_ - st))};
    }

    Token lex_string() {
        char q = s_[pos_++];
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == q) break;
            if (c == '\\' && pos_ < s_.size())
                out.push_back(s_[pos_++]);
            else
                out.push_back(c);
        }
        return {TokType::String, out};
    }

    std::string_view s_;
    size_t pos_{0};
};

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

static bool scalar_cmp(const ScalarValue& a, const ScalarValue& b, CompareOp op) {
    double da, db;
    if (to_double(a, da) && to_double(b, db)) {
        switch (op) {
            case CompareOp::Lt: return da < db;
            case CompareOp::Le: return da <= db;
            case CompareOp::Gt: return da > db;
            case CompareOp::Ge: return da >= db;
            default: return false;
        }
    }
    if (auto sa = std::get_if<std::string>(&a)) {
        if (auto sb = std::get_if<std::string>(&b)) {
            switch (op) {
                case CompareOp::Lt: return *sa < *sb;
                case CompareOp::Le: return *sa <= *sb;
                case CompareOp::Gt: return *sa > *sb;
                case CompareOp::Ge: return *sa >= *sb;
                default: return false;
            }
        }
    }
    return false;
}

struct CmpNode final : Predicate {
    std::string col;
    CompareOp op;
    ScalarValue rhs;
    std::vector<ScalarValue> rhs_list;

    bool eval(const std::unordered_map<std::string, ScalarValue>& row) const override {
        auto it = row.find(col);
        if (it == row.end()) return false;

        if (op == CompareOp::In) {
            for (const auto& x : rhs_list) {
                if (scalar_eq(it->second, x)) return true;
            }
            return false;
        }

        if (op == CompareOp::Eq) return scalar_eq(it->second, rhs);
        if (op == CompareOp::Ne) return !scalar_eq(it->second, rhs);
        return scalar_cmp(it->second, rhs, op);
    }
};

struct AndNode final : Predicate {
    PredicatePtr l, r;
    bool eval(const std::unordered_map<std::string, ScalarValue>& row) const override {
        return l->eval(row) && r->eval(row);
    }
};

struct OrNode final : Predicate {
    PredicatePtr l, r;
    bool eval(const std::unordered_map<std::string, ScalarValue>& row) const override {
        return l->eval(row) || r->eval(row);
    }
};

class Parser {
public:
    explicit Parser(std::string_view s) : lex_(s) {
        cur_ = lex_.next();
    }

    PredicatePtr parse_expr() {
        return parse_or();
    }

private:
    PredicatePtr parse_or() {
        auto lhs = parse_and();
        while (cur_.type == TokType::Or) {
            consume(TokType::Or);
            auto rhs = parse_and();
            auto n = std::make_unique<OrNode>();
            n->l = std::move(lhs);
            n->r = std::move(rhs);
            lhs = std::move(n);
        }
        return lhs;
    }

    PredicatePtr parse_and() {
        auto lhs = parse_factor();
        while (cur_.type == TokType::And) {
            consume(TokType::And);
            auto rhs = parse_factor();
            auto n = std::make_unique<AndNode>();
            n->l = std::move(lhs);
            n->r = std::move(rhs);
            lhs = std::move(n);
        }
        return lhs;
    }

    PredicatePtr parse_factor() {
        if (cur_.type == TokType::LParen) {
            consume(TokType::LParen);
            auto n = parse_expr();
            consume(TokType::RParen);
            return n;
        }
        return parse_cmp();
    }

    PredicatePtr parse_cmp() {
        std::string col = expect_ident();

        if (cur_.type == TokType::In) {
            consume(TokType::In);
            consume(TokType::LParen);

            std::vector<ScalarValue> vals;
            vals.push_back(parse_literal());
            while (cur_.type == TokType::Comma) {
                consume(TokType::Comma);
                vals.push_back(parse_literal());
            }
            consume(TokType::RParen);

            auto n = std::make_unique<CmpNode>();
            n->col = std::move(col);
            n->op = CompareOp::In;
            n->rhs_list = std::move(vals);
            return n;
        }

        CompareOp op;
        switch (cur_.type) {
            case TokType::Eq: op = CompareOp::Eq; break;
            case TokType::Ne: op = CompareOp::Ne; break;
            case TokType::Lt: op = CompareOp::Lt; break;
            case TokType::Le: op = CompareOp::Le; break;
            case TokType::Gt: op = CompareOp::Gt; break;
            case TokType::Ge: op = CompareOp::Ge; break;
            default: throw std::runtime_error("predicate parser: expected comparison operator");
        }
        cur_ = lex_.next();

        auto n = std::make_unique<CmpNode>();
        n->col = std::move(col);
        n->op = op;
        n->rhs = parse_literal();
        return n;
    }

    ScalarValue parse_literal() {
        if (cur_.type == TokType::Number) {
            std::string s = cur_.text;
            cur_ = lex_.next();
            if (s.find('.') != std::string::npos) return std::stod(s);
            return static_cast<int64_t>(std::stoll(s));
        }
        if (cur_.type == TokType::String) {
            std::string s = cur_.text;
            cur_ = lex_.next();
            return s;
        }
        if (cur_.type == TokType::Bool) {
            bool v = (cur_.text == "TRUE");
            cur_ = lex_.next();
            return v;
        }
        throw std::runtime_error("predicate parser: expected literal");
    }

    std::string expect_ident() {
        if (cur_.type != TokType::Ident) {
            throw std::runtime_error("predicate parser: expected identifier");
        }
        std::string s = cur_.text;
        cur_ = lex_.next();
        return s;
    }

    void consume(TokType t) {
        if (cur_.type != t) throw std::runtime_error("predicate parser: unexpected token");
        cur_ = lex_.next();
    }

    Lexer lex_;
    Token cur_;
};

} // namespace

PredicatePtr compile(const std::string& expr) {
    Parser p(expr);
    return p.parse_expr();
}

bool evaluate(const Predicate& p, const std::unordered_map<std::string, ScalarValue>& row) {
    return p.eval(row);
}

bool evaluate(const std::string& expr, const std::unordered_map<std::string, ScalarValue>& row) {
    auto p = compile(expr);
    return p->eval(row);
}

} // namespace ppq::pred