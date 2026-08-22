s = open("/workspace/ngen_lab/ovino/gdn_onednn_full.cpp").read()

old1 = "        for (int i = 0; i < 2; ++i) run_once();"
new1 = (
    "        for (int i = 0; i < 2; ++i) run_once();\n"
    "        auto run_mm_only = [&] {\n"
    "            for (int t = 0; t < T; ++t) {\n"
    "                void* kt = (char*)d_k + (size_t)t * nv * dk * 2;\n"
    "                void* qt = (char*)d_q + (size_t)t * nv * dk * 2;\n"
    "                mm.execute(stream, {{DNNL_ARG_SRC, mk(s_desc, d_s)},\n"
    "                                    {DNNL_ARG_WEIGHTS, mk(x_desc, kt)},\n"
    "                                    {DNNL_ARG_DST, mk(y_desc, d_hk)}});\n"
    "                mm.execute(stream, {{DNNL_ARG_SRC, mk(s_desc, d_s)},\n"
    "                                    {DNNL_ARG_WEIGHTS, mk(x_desc, qt)},\n"
    "                                    {DNNL_ARG_DST, mk(y_desc, d_o)}});\n"
    "            }\n"
    "            stream.wait();\n"
    "        };\n"
    "        for (int i = 0; i < 2; ++i) run_mm_only();\n"
    "        std::vector<double> mm_samp;\n"
    "        for (int i = 0; i < iters; ++i) {\n"
    "            auto t0 = std::chrono::steady_clock::now();\n"
    "            run_mm_only();\n"
    "            auto t1 = std::chrono::steady_clock::now();\n"
    "            mm_samp.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());\n"
    "        }\n"
    "        double mm_mean = 0; for (double s : mm_samp) mm_mean += s; mm_mean /= mm_samp.size();"
)
assert old1 in s
s = s.replace(old1, new1, 1)

old2 = (
    '        std::printf("gdn_nonfused_full p=%d runs=%d mean_ms=%.3f tok_per_s=%.1f '
    "(decay+gemm+update+gemm x%d)\\n\",\n"
    "                    T, iters, mean, T * 1000.0 / mean, T);"
)
new2 = (
    '        std::printf("gdn_nonfused_full p=%d runs=%d mean_ms=%.3f tok_per_s=%.1f '
    '| gemm-only %.3f ms (%.1f%% of full) (decay+gemm+update+gemm x%d)\\n",\n'
    "                    T, iters, mean, T * 1000.0 / mean, mm_mean, 100.0 * mm_mean / mean, T);"
)
assert old2 in s
s = s.replace(old2, new2, 1)

open("/workspace/ngen_lab/ovino/gdn_onednn_full.cpp", "w").write(s)
print("patched")
