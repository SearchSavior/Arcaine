#pragma once

#include <vector>

namespace arcaine::bench {

struct Bench {
    const char* name;
    const char* description;
    int (*run)(int argc, char** argv);
};

class Registry {
public:
    static Registry& get() {
        static Registry r;
        return r;
    }
    void add(Bench b) { benches_.push_back(b); }
    const std::vector<Bench>& all() const { return benches_; }
private:
    std::vector<Bench> benches_;
};

struct Registrar {
    Registrar(const char* name, const char* desc, int (*fn)(int, char**)) {
        Registry::get().add({name, desc, fn});
    }
};

}  // namespace arcaine::bench

#define REGISTER_BENCH(name, desc, fn) \
    static ::arcaine::bench::Registrar \
        arcaine_bench_registrar_##__COUNTER__(name, desc, fn);
