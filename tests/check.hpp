#pragma once
// Minimal single-TU test helpers. Each test file includes this once, runs
// CHECK* macros in main(), and returns RETURN_TESTS().

#include <cstdio>
#include <string>

namespace arcaine::check {

inline int g_failures = 0;

inline void fail(const char* file, int line, const std::string& msg) {
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, msg.c_str());
    ++g_failures;
}

}  // namespace arcaine::check

#define CHECK(cond) \
    do { if (!(cond)) ::arcaine::check::fail(__FILE__, __LINE__, #cond); } while (0)

#define CHECK_EQ(a, b)                                                       \
    do {                                                                     \
        auto _a = (a);                                                        \
        auto _b = (b);                                                        \
        if (!(_a == _b))                                                      \
            ::arcaine::check::fail(__FILE__, __LINE__, #a " != " #b);         \
    } while (0)

#define CHECK_THROWS(expr)                                                   \
    do {                                                                     \
        bool _caught = false;                                                \
        try { (expr); } catch (...) { _caught = true; }                      \
        if (!_caught) ::arcaine::check::fail(__FILE__, __LINE__, #expr " did not throw"); \
    } while (0)

#define RETURN_TESTS() return ::arcaine::check::g_failures ? 1 : 0
