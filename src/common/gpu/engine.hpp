#pragma once
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <algorithm>
#include <stdexcept>
#include <vector>

// Per-device GPU context. GpuEngine::get(idx) returns the context for GPU idx.
// GpuEngine::get() (no arg) defaults to GPU 0 for backward compatibility.
struct GpuEngine {
    dnnl::engine engine;
    dnnl::stream stream;
    sycl::queue  queue;
    int          index = 0;

    // Returns the context for GPU idx (clamped to [0, count()-1]).
    static GpuEngine& get(int idx = 0);
    static int        count();
};

namespace gpu_detail {
    struct Engines {
        std::vector<GpuEngine> devs;

        Engines() {
            int n = (int)dnnl::engine::get_count(dnnl::engine::kind::gpu);
            if (n == 0)
                throw std::runtime_error("No oneDNN GPU engines found");
            devs.resize(n);
            for (int i = 0; i < n; ++i) {
                devs[i].index  = i;
                devs[i].engine = dnnl::engine(dnnl::engine::kind::gpu, i);
                auto dev = dnnl::sycl_interop::get_device(devs[i].engine);
                auto ctx = dnnl::sycl_interop::get_context(devs[i].engine);
                devs[i].queue = sycl::queue(ctx, dev, sycl::property::queue::in_order{});
                devs[i].stream = dnnl::sycl_interop::make_stream(devs[i].engine, devs[i].queue);
            }
        }
    };

    inline Engines& engines() { static Engines e; return e; }
}

inline GpuEngine& GpuEngine::get(int idx) {
    auto& e = gpu_detail::engines();
    return e.devs[(idx >= 0 && idx < (int)e.devs.size()) ? idx : 0];
}

inline int GpuEngine::count() { return (int)gpu_detail::engines().devs.size(); }
