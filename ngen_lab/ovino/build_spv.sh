#!/bin/sh
# Lower ngen_lab/ovino/gdn_seq_ref_resolved.cl to SPIR-V for the L0 harness.
set -e
cd "$(dirname "$0")"
CLANG=/opt/intel/oneapi/compiler/2026.1/bin/compiler/clang
SPV=/opt/intel/oneapi/compiler/2026.1/bin/compiler/llvm-spirv
$CLANG -x cl -cl-std=CL2.0 -c -emit-llvm -target spir64-unknown-unknown \
    -o gdn_seq_ref.bc gdn_seq_ref_resolved.cl
$SPV gdn_seq_ref.bc -o gdn_seq_ref.spv
echo "gdn_seq_ref.spv written"
