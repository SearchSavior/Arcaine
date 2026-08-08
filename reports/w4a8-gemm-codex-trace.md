# W4A8 GEMM — full codepath trace (vLLM oneDNN shim → ngen DPAS kernel)

Date: 2026-08-08

Scope: trace `int4_gemm_w4a8` from the torch binding down to the DPAS hardware
instruction, using the *current* reference checkouts (HEAD). Reviewed code only;
no build/run in the container.

## Provenance (commit-pinned)

| Repo | Checkout | File |
| --- | --- | --- |
| vllm-xpu-kernels | `de8eae16cdf2d864c86a203d9431b9bcdb42fb3c` | `reference/vllm-xpu-kernels` |
| oneDNN | `fd819b3e5188eef6dbcca287332890bef48a7fb1` | `reference/oneDNN` |

`libdnnl.so.3.13` exists in the Arcaine container (`/opt/onednn`), but the code
below is traced against reference HEAD. Any 3.13-vs-HEAD difference is out of
scope here.

---

## Layer-by-layer trace

### 1. Torch custom-op registration
- `csrc/xpu/torch_bindings.cpp:39-43` — `xpu_ops.def("int4_gemm_w4a8(...)")` →
  `xpu_ops.impl("int4_gemm_w4a8", torch::kXPU, &int4_gemm_w4a8)`.
- Wrapper `tests/register_ops.py:380-392`; tensor shapes per
  `tests/test_int4_gemm_onednn.py:103-160`:
  - `input` int8 `[m,k]`; `input_scales`/`input_zero_points` per-token `[b*m,1]`
  - `weight` packed u4 `[k/8, n]` (2 int4/byte)
  - `wei_scales` `[k/group_size, n]`; `wei_zero_points` `[k/group_size, n/8]`
  - `group_size`, optional `g_idx` (GPTQ act-order), optional `bias`

### 2. Host shim
- `csrc/xpu/onednn/onednn_matmul.cpp:220-256` — dispatch: contiguity checks,
  `g_idx` column reorder (`index_select` :238-239), per-token vs per-tensor
  validation (:242-249).
- `csrc/xpu/onednn/int4_gemm_w4a8.h`:
  - flattens `[b,m,k]` → `[b*m,k]` (:25-28)
  - picks `joint_dtypes_t::s8_int4` (or `u8_int4`) by src scalar type (:33-40)
  - `ldb = mat2.strides()[last] * 8` (:59) — 2 int4/byte → wei gemm dims
  - `primitive_attr`: src scales+ZP (mask 3 per-token / mask 0 per-tensor),
    wei scales (mask 3 grouped), u4 wei ZP (:62-107)
- oneDNN dtype mapping: `csrc/xpu/onednn/onednn_ext.h:83-91` — `(src=s8, wei=u4,
  dst=f16)`.

### 3. Primitive cache + ABI
- `onednn_ext.h:895-1174` — `matmul_primitive_create_and_cache` → generates
  `matmul::primitive_desc(aengine, src s8, wei u4, dst f16, pattr)` with `nt`
  strides (`get_strides<nt>`, :449-460). Cached on (strides, m, n, k, b_type,
  scale/zp group sizes).
- `onednn_ext.h:706-731` — `primitive_ext::execute` reuses cached
  `dnnl_exec_arg_t` slots; only USM data handles are swapped per call.
- `onednn_ext.h:149-185` — src ZP is passed via `make_onednn_memory` with the
  tensor's own descriptor; the grouped u4 weight ZP is rebuilt as a
  `{num_groups, n}` desc with stride `{n,1}` (:177-184).

### 4. Engine / stream binding (the "to the metal" plumbing)
- `csrc/xpu/onednn/onednn_runtime.h:47-54` — one `dnnl::engine` per device,
  built from `at::xpu::get_raw_device(dev)` + `at::xpu::get_device_context()`:
  the same SYCL device/context PyTorch XPU already bound to Level Zero.
- `onednn_runtime.h:62-77` — `dnnl::stream` wraps the *current PyTorch XPU
  stream's SYCL `queue()`* (keyed by device+priority+stream identity).
- `onednn_runtime.h:17-24` — all memory is USM
  (`make_onednn_memory` → `sycl_interop::memory_kind::usm`), consumed in place.
- **Consequence:** oneDNN submits its kernels onto the exact Level Zero queue
  PyTorch is already using — in-order with the rest of the model, no copies.

### 5. non-grouped matmul → `intel::matmul::gemm_t`
- `oneDNN/src/gpu/intel/gpu_matmul_list.cpp:50` routes to `gemm_t`.
- `oneDNN/src/gpu/intel/matmul/gemm.cpp:27-91` — `gemm_t::execute` strips the
  oneDNN quantization attrs into raw gemm operands, swapping roles because gemm
  is row-major convention while impl is column-major:

  ```cpp
  // gemm.cpp:51-59
  args.a_zero_point = b0;     // weights ZP   (gemm-B col-major)
  args.b_zero_point = a0;     // src ZP       (gemm-A)
  args.a_scales   = ...SCALES | DNNL_ARG_WEIGHTS; // weight scales
  args.b_scales   = ...SCALES | DNNL_ARG_SRC;     // src scales
  ```
- `create_gemm_desc` — `oneDNN/src/gpu/intel/gemm/utils.hpp:102-118` builds the
  gemm desc with `a=src s8`, `b=weights u4`, `c=f16`.

### 6. gemm dtype / support gate (`gen_t::pd_t::init`)
- `oneDNN/src/gpu/intel/gemm/jit.hpp:159-168` — the int8|int4 dense path:

  ```cpp
  if (c_type in {s32, f16, bf16, f32, u8, s8} && a_type in {u8,s8,u4,s4}) {
    // b_type in {u8,s8} OR wei_decomp_; c_type f16 requires arch_>=xe_hp
  }
  ```
- `wei_decomp()` — `oneDNN/src/gpu/intel/gemm/jit/pd.cpp:216-227`: false here
  (needs `bits(a)<bits(b)`; s8 is wider than u4).
- ZP/scales support validation — `pd.cpp:343-426` (`zp_ok`): the grouped (mask
  != 0) src ZP is allowed only under dequantization/precomputed-reduction
  conditions (`pd.cpp:373`). **Flagged for verification** — grouped A ZP (mask
  `{1,k}`) is exactly the w4a8 case and the most likely silent-fallback point.

### 7. gemmstone kernel selection
- `jit.hpp:292-309` — constructs `gemmstone::GEMMProblem` and calls
  `kernel_desc_.select_kernel(...)`: tiling, workgroup, k-parallel, and
  dpasv/dpasw strategy search.
- `jit.cpp:277-285` — launch grid: `gws = (m/unrollM, n/unrollN, kParallel or
  wgK)`, `lws = (wgM, wgN, wgK)`; `swap_ab_` (`jit.hpp:123-156`) swaps M/N for
  column-major gemm convention.

### 8. ngen DPAS microkernel + inline scales
- `jit.cpp:359+` (`gen_t::execute`) → `launch_nocopy` (`jit.cpp:45+`): A and B
  stay **nocopy in native form**; scales/ZP are separate kernel args (aScale2D /
  bScale2D / b_zp).
- `oneDNN/src/gpu/intel/gemm/jit/generator/pieces/matrix_multiply.cxx:386-571`:
  emits `dpas(sdepth, rc, C0, srcC0, V, N, ...)` and `dpasw(...)` for the XMX
  variant, with **scales injected inline** in the k-loop via
  `bdpasScaleArg`/`bdpas(...)` (:481-536) — A-scales (`AS`) and B-scales (`BS`)
  become register operands on the DPAS, not post-ops.

### 9. Fusion + Level Zero submission
- `oneDNN/src/gpu/intel/gemm/jit/generator/microkernel/fuser.cpp:134` — the
  ngen-compiled microkernel ELF is spliced into an IGC-generated **zebin**.
- Kernel object created via `zeModuleCreate`/`zeKernelCreate` through SYCL
  interop — `oneDNN/src/gpu/intel/sycl/utils.cpp:84-147`.
- Dispatched on the stream's SYCL queue → Level Zero → Xe matrix engine (the
  same DPAS hardware the Arcaine `ngen_lab` targets).

---

## Net path

```
torch.ops._xpu_C.int4_gemm_w4a8
  → onednn_matmul.cpp:int4_gemm_w4a8
  → dnnl_matmul_w4a8_int4 (int4_gemm_w4a8.h)   [s8 src × u4 wei × f16 dst]
  → matmul::primitive_desc (onednn_ext.h)      [USM, cached primitive]
  → intel::matmul::gemm_t (matmul/gemm.cpp)    [strips attrs, A/B swap]
  → gemm jit::pd_t::init → gemmstone GEMMProblem (gemm/jit.hpp)
  → gemmstone select_kernel → ngen DPAS microkernel (gemm/jit/generator)
  → fused zebin (fuser.cpp) → Level Zero (sycl utils)
  → DPAS execute on current PyTorch XPU queue
```

---

## Key observations / open items

1. **The live backend is gemmstone (ngen), not the legacy `jit_xe_hp_systolic`**
   path (`src/gpu/intel/gemm/jit_xe_hp_systolic.*`). The old systolic copy
   kernels are not the operative path for this shape.
2. **No per-call weight repack.** In gemmstone "nocopy", A and B are consumed
   in native form; scales/ZP ride as separate args. The "src ZP forces unpacked
   B" behavior lives only in the legacy systolic file and does **not** apply.
3. **Missing `fpmath_mode` in the shim.** `int4_gemm_w4a8.h` never calls
   `set_fpmath_mode`, unlike the w4a16 twin (`int4_gemm_w4a16.h:80-83`). Worth
   confirming whether s8×u4 accumulations need `apply_to_int=true` on the
   gemmstone path.
4. **Grouped src ZP (mask `{1,k}`) is the risky case** (`pd.cpp:373`). This is
   precisely what the w4a8 shim sets (`int4_gemm_w4a8.h:66-75`). Highest-value
   thing to verify with oneDNN verbose/`--check` at small M.
5. **Contiguity guards cover only scales** (`onednn_matmul.cpp:233-235`), not
   `A_zp`/`B_zp`. The u4 weight-ZP memory is rebuilt with a fixed `{n,1}`
   stride (`int4_gemm_w4a8.h:177-184`); if the tensor isn't actually in that
   layout, the descriptor mismatches.
