# ngen_lab — bring-up findings and architecture

Standalone Level Zero + nGEN bring-up for a fixed-shape BF16 16x16x32 matmul on
Intel Arc (Xe2 / BMG). This document records the critical findings: the register
and message-level contracts that caused hangs and wrong results, the ground
truth sources used to resolve them, and the layout decisions that make the
kernels correct.

Everything here was verified against the vendored ngen source
(`reference/oneDNN/third_party/ngen/`) and oneDNN GPU code, and confirmed on
hardware via the `l0_min` stage ladder and the final correctness gates.

---

## 1. The two kernels

Both compute `C[16x16] (fp32) += A[16x32] (bf16) * B[32x16] (bf16)`, one tile per
work-group (SIMD16, one thread), `iters` accumulation passes.

- **DPAS kernel** (`dpasKernelBody`): 4x `dpas.8x8` per tile (2 K-chunks x 2
  N-halves). ~170 TFLOPS bf16 at groups=8192, iters=4096.
- **Scalar kernel** (`scalarKernelBody`): same math with per-lane fp32
  `mul`+`add` (bf16->fp32 unpack of A once, B streamed per K). ~5 TFLOPS
  (launch-bound at these sizes).

Register layouts (Xe2, 64B GRFs, SIMD16, lane m = row m of C):

- **A (src1, per-lane)**: GRF j, lane m = `{A[m][2j], A[m][2j+1]}` (bf16 pair).
  Loaded with 16x `scattered(D32,1)` at `rPtrA + 4*j` — the scattered layout
  (GRF j = element j across lanes) matches the k-pair layout directly.
- **B (src2, broadcast/VNNI)**: host packer `packB` produces 4 pieces; each
  piece dword `r*8+j` = `{B[16c+2j][8h+r], B[16c+2j+1][8h+r]}`. Loaded as 4x
  `block(D32,16)` per piece (contiguous 64B chunks).
- **C (accumulator)**: GRF n, lane m = `C[m][n]` (fp32). Stored with 16x
  `scattered(D32,1)` at `rPtrC + 4*n` — column-per-GRF matches the scattered
  data layout, and the row-major result matches the CPU reference.

---

## 2. Critical findings (each caused a real bug)

### 2.1 GRF subregister offsets cannot cross a GRF boundary
`GRF(14).uq(8)(1)` in a SIMD8 op does **not** address GRF 15 — it wraps to
`r14.0` (IGA disassembly proved the second `add` read garbage r20/r21 halves).
Use separate GRFs for the two halves of a 16-lane address vector.

### 2.2 Missing SWSB annotations
ngen kernels emit zero SWSB sync by default; without them the first store races
ahead of the instructions producing its addresses/data. Enable via a derived
generator exposing the protected setters (oneDNN's pattern):
`setDefaultAutoSWSB(true)` **before** `prologue()`, `setDefaultNoMask(true)`
**after**. Verified in `l0_min` stages 3-5: stores landed at wrong addresses
before the fix, correct after.

### 2.3 `GRF(x)[1]` is NOT GRF(x+1)
`Register::operator[]` returns `sub(offset, type)` — a *subregister of the same
GRF*, not the next GRF. The address-vector high-half code originally used
`rPtrA[1]`, producing lanes 8-15 with garbage addresses -> wild scattered
stores/loads -> **GPU hang / lost device**. Fix: use explicit `GRF(base+1)`.
This was the root cause of the remaining hang after SWSB.

### 2.4 Ternary `mad` src0/src1 are forced SCALAR on Gen12LP+
For 3-source ALU ops the ternary encoding has no width field for src0/src1
(only `vs`/`hs`); the only vector operand is src2 (the addend). A "full-GRF"
region on src0 silently degrades to a broadcast of element 0. oneDNN's
`fixup_ternary_rgn` (`gemm/jit/dsl/ir/codegen/kernel.hpp:1118`) codifies this:
full-vector `(hs==1 && vs==width)` regions are converted to `(1,1,0)` scalar.

Consequence: `mad(acc, A_row, B_scalar, acc)` computes `A[0][k]*B` for *all*
lanes. The scalar kernel therefore uses **`mul` + `add`** (binary ops, which
support full per-lane regions `<16;16,1>` on both operands) instead of a single
`mad`. This was the scalar kernel's wrong-result bug (cos=-0.08).

### 2.5 Xe2 memory messages are `DataSpecLSC` only
The legacy `block_hword`/`block_oword`/`scattered_dword` specs throw
`unsupported_message` on Xe2 under `NGEN_SAFE`. Use `DataSpecLSC` built via
`scattered(dtype, vcount)` or `block(dtype, vcount)` (which adds
`createTranspose`).

- `block(D32, n)` (transposed): `messageLen=1`, `dataLen = dbytes()*vc` GRFs.
  A 64B load is `block(D32,16)` -> 1 GRF; a 256B piece is 4 of those.
- `scattered(D32, vcount)`: vcount elements per lane; `vcount==1` is the
  proven lane-mapped pattern (l0_min stages 3/4), with the address vector =
  2 GRFs (16 lanes x uq).

### 2.6 `GRF(x) + offset` arithmetic is a byte displacement
In load/store addresses, `rPtrA + 4*j` builds a `GRFDisp` (byte offset in the
message descriptor), not a register index. Correct for scattered per-column
access; keep register-index arithmetic on explicit GRFs.

### 2.7 Integer widening must never mix widths in one instruction
ngen's emulation (`ngen_emulation.hpp` `emov`/`splitToDW`) never mixes integer
widths in a single mov/add. For `uw->ud` and `ud->uq` widening, write low
subwords with a same-type strided mov then zero the high subwords (or use
explicit same-type sequences). The ladder's address build (`setupTileAddresses`)
follows this.

---

## 3. Environment and ABI notes

- **Interface**: `ngen::InterfaceHandler` (not NEO) + the r0.0-indirect
  loadargs path. `setInlineGRFCount(1)` declared inline delivery but the driver
  left r1-r2 all zero on BMG — the loadargs path is what oneDNN relies on.
- **Argument slots**: first uq slots are system; user args land at r1.4+ (the
  `mov r96 <- r1.5` style shown in disassembly). Positional `zeKernelSetArgumentValue`
  matches interface registration order.
- **Device selection**: `NGEN_LAB_DEVIDX` / `L0_MIN_DEVIDX` envs pick the GPU;
  both devices currently enumerate as "Intel(R) Arc(TM) Pro B70 Graphics".
- **GPU hangs recover**: a lost/unavailable device recovers on its own after
  ~20s. Always re-run a known-good probe (`l0_min_s3`) to confirm hardware
  health before interpreting new results.
- **Disassembly**: extract the `.text` section from the zebin
  (`dd` from the ELF section offset/length) and feed it to `build/iga_dis`.
  Note the textual asm from `ngen_lab_asmdump` uses `AsmCodeGenerator` whose
  `fixup` is a no-op, so printed regions may differ from the binary — verify
  binary regions with IGA.

---

## 4. Layouts and ground truth sources

| Topic | Source |
|---|---|
| SWSB/NoMask generator pattern | oneDNN `gemm/jit/generator/pieces/...` |
| Widening idiom | `ngen_emulation.hpp` `emov`/`splitToDW` |
| LSC message encoding | `ngen_core.hpp` `encodeLSCMessage`, `DataSpecLSC` |
| Ternary mad scalar constraint | oneDNN `gemm/jit/dsl/ir/codegen/kernel.hpp` `fixup_ternary_rgn` |
| Scattered-store reference | oneDNN `conv/jit/zero_out.hpp` |
| DPAS src2 full-load requirement | oneDNN `gemm/jit/dsl/ir/pass/dpasw.cpp` |

---

## 5. Files

- `ngen_dpas_8x8_matmul_bf16_16x16x32.cpp` + `.hpp` — bf16 x bf16 (L0 harness,
  correctness vs CPU ref, perf).
- `ngen_dpas_8x8_matmul_s8_16x16x32.cpp` + `.hpp` — s8 x s8 -> s32 (PASS,
  ~363 TFLOPS). Header namespace `ngen_lab::s8`.
- `ngen_dpas_8x8_matmul_s4_16x16x32.cpp` + `.hpp` — s8 x s4 -> s32, int4 weights
  (PASS). Header namespace `ngen_lab::s4`.
- `ngen_dpas_8x8_matmul_w8a2_16x16x32.cpp` + `.hpp` — s8 x s2 -> s32, int2
  weights (PASS). Header namespace `ngen_lab::w8a2`.
- `ngen_dpas_8x8_matmul_w4a8_16x16x32.cpp` + `.hpp` — s4 x s8 -> s32, int4
  activations (PASS). Header namespace `ngen_lab::w4a8`.
- Each `.hpp` holds the kernel-body authors templated on the generator, so the
  same body compiles against both the binary generator (harness) and the text
  generator (asm dump). Distinct nested namespaces avoid cross-kernel symbol
  collisions when a TU includes multiple headers.
- `ngen_lab_asm.cpp` — textual asm dump binary (`ngen_lab_asmdump`); dumps all
  five kernels (bf16 dpas+scalar, s8, s4, w8a2, w4a8).
- `l0_min.cpp` — compile-time stage ladder (0-7) for bisection; stage 6/7 run
  the real DPAS/scalar bodies.
- `iga_dis.cpp` — raw Gen-ISA disassembler (optional; wired into CMake if IGA
  is present).

## 6. Build and run

```sh
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j$(nproc)
# One build target per kernel file (name = operand types + shape):
NGEN_LAB_DEVIDX=1 NGEN_LAB_KERNELS=both ./build/ngen_lab  # bf16 x bf16
NGEN_LAB_DEVIDX=1 ./build/ngen_dpas_8x8_matmul_s8_16x16x32     # s8 x s8 (PASS)
NGEN_LAB_DEVIDX=1 ./build/ngen_dpas_8x8_matmul_s4_16x16x32     # s8 x s4, int4 weights (PASS)
NGEN_LAB_DEVIDX=1 ./build/ngen_dpas_8x8_matmul_w8a2_16x16x32   # s8 x s2, int2 weights (PASS)
NGEN_LAB_DEVIDX=1 ./build/ngen_dpas_8x8_matmul_w4a8_16x16x32   # s4 x s8, int4 activations (PASS)
./build/ngen_lab_asmdump                                   # asm dumps for all 5 kernels
L0_MIN_DEVIDX=1 ./build/l0_min_s3                             # hardware health probe
```

## 7. What does NOT work on Xe2 (documented, no code kept)

The following operand-type combinations were implemented and tested on BMG; they
fail and are intentionally not kept as buildable kernels. Evidence and exact
layouts were recorded in project memory (`ngen_lab_int2.md`, `ngen_lab_int4.md`,
`ngen_lab_fp.md`).

| Combo | Failure mode |
|---|---|
| s4 x s4 (w4a4) | ngen encodes; hardware computes garbage (deterministic wrong values) |
| s2 x s2 | ngen encodes; hardware computes garbage |
| s2 x s8 (w2a8) | ngen encodes; hardware computes garbage |
| fp8 x fp8 | ngen encodes; hardware returns `inf` (misreads fp8 operands) |
| fp4 x fp4 (e2m1) | ngen rejects at encode (`invalid_type_exception`; fp4 is Xe3/`bdpas` only) |
| f16 x s4 (w4a16) | ngen rejects at encode (ternary operand type-family mismatch) |
| dpas sdepth < 8 | ngen encodes and reads correct registers, but computes wrong results for every B layout tried; oneDNN always uses sdepth 8 |

General rules that follow (all verified empirically):
- Sub-byte precision (s4, s2) works on **one** dpas operand only, and only on the
  src2 (weights) side; not in src1 (activations), not both.
- 16-bit float (bf16/f16) and 8-bit int (s8) are the only native dpas types;
  everything else requires upconversion or Xe3's `bdpas`.
- True fp4/fp8/int2-both-sides matmul needs Xe3 (`bdpas`, fused scale operands
  src3/src4).
