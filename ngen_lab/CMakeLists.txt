cmake_minimum_required(VERSION 3.20)
project(ngen_lab CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ngen headers (oneDNN's bundled copy) + its single source file.
set(NGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../reference/oneDNN/third_party/ngen)

add_compile_definitions(NGEN_SAFE)
add_compile_options(-fno-operator-names)

include_directories(${NGEN_DIR})

add_executable(ngen_lab ngen_dpas_8x8_matmul_bf16_16x16x32.cpp ${NGEN_DIR}/ngen_register_allocator.cpp)
target_link_libraries(ngen_lab PRIVATE ze_loader ${CMAKE_DL_LIBS})

# One target per fixed-shape kernel file. The kernel name encodes its operand
# types and shape (e.g. s4 = s8 x s4 int4 weights); each file represents exactly
# one shape. Target name = file stem (ngen_dpas_8x8_matmul_s4_16x16x32 -> ...).
# Only kernels that pass correctness on Xe2 are kept; the types that the
# hardware does not support are documented in docs/findings.md (never built).
set(NGEN_LAB_WORKING_KERNELS
        ngen_dpas_8x8_matmul_s8_16x16x32
        ngen_dpas_8x8_matmul_s4_16x16x32
        ngen_dpas_8x8_matmul_w8a2_16x16x32
        ngen_dpas_8x8_matmul_w4a8_16x16x32
        ngen_dpas_w4a8_gemm)

foreach(kernel_stem IN LISTS NGEN_LAB_WORKING_KERNELS)
    add_executable(${kernel_stem} ${kernel_stem}.cpp
            ${NGEN_DIR}/ngen_register_allocator.cpp)
    target_link_libraries(${kernel_stem} PRIVATE ze_loader ${CMAKE_DL_LIBS})
endforeach()

add_executable(ngen_lab_asmdump ngen_lab_asm.cpp)
target_compile_definitions(ngen_lab_asmdump PRIVATE NGEN_ASM)

# iga_dis: raw Gen-ISA disassembler (IGA from oneAPI debugger). Optional;
# the core build does not depend on it.
find_path(IGA_INCLUDE_DIR iga/iga.h
        HINTS /opt/intel/oneapi/debugger/latest/opt/debugger/include)
find_library(IGA_LIBRARY NAMES iga64
        HINTS /opt/intel/oneapi/debugger/latest/opt/debugger/lib)
if(IGA_INCLUDE_DIR AND IGA_LIBRARY)
    add_executable(iga_dis iga_dis.cpp)
    target_include_directories(iga_dis PRIVATE ${IGA_INCLUDE_DIR})
    target_link_libraries(iga_dis PRIVATE ${IGA_LIBRARY})
endif()

foreach(STAGE RANGE 0 7)
    add_executable(l0_min_s${STAGE} l0_min.cpp ${NGEN_DIR}/ngen_register_allocator.cpp)
    target_compile_definitions(l0_min_s${STAGE} PRIVATE TEST_STAGE=${STAGE})
    target_link_libraries(l0_min_s${STAGE} PRIVATE ze_loader ${CMAKE_DL_LIBS})
endforeach()

# ovino: OpenVINO GDN kernel lowered to SPIR-V and loaded via ZE_MODULE_FORMAT_SPIRV.
add_executable(ovino_gdn_spv ovino/ovino_gdn_spv.cpp)
target_link_libraries(ovino_gdn_spv PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_subgroup_probe ovino/subgroup_probe.cpp)
target_link_libraries(ovino_subgroup_probe PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_probe2 ovino/probe2.cpp)
target_link_libraries(ovino_probe2 PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag ovino/diag.cpp)
target_link_libraries(ovino_diag PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag2 ovino/diag2.cpp)
target_link_libraries(ovino_diag2 PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag3 ovino/diag3.cpp)
target_link_libraries(ovino_diag3 PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag3_small ovino/diag3_small.cpp)
target_link_libraries(ovino_diag3_small PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag3_512 ovino/diag3_512.cpp)
target_link_libraries(ovino_diag3_512 PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_gdn_spv_x ovino/ovino_gdn_spv_x.cpp)
target_link_libraries(ovino_gdn_spv_x PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag4 ovino/diag4.cpp)
target_link_libraries(ovino_diag4 PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag5 ovino/diag5.cpp)
target_link_libraries(ovino_diag5 PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diagB ovino/diagB.cpp)
target_link_libraries(ovino_diagB PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diagC ovino/diagC.cpp)
target_link_libraries(ovino_diagC PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_state_pass ovino/state_pass.cpp)
target_link_libraries(ovino_state_pass PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_red_test ovino/red_test.cpp)
target_link_libraries(ovino_red_test PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_diag2_big ovino/diag2_big.cpp)
target_link_libraries(ovino_diag2_big PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_decay_test ovino/decay_test.cpp)
target_link_libraries(ovino_decay_test PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_scalar ovino/ovino_scalar.cpp)
target_link_libraries(ovino_scalar PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_slm_probe ovino/slm_probe.cpp)
target_link_libraries(ovino_slm_probe PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_vec ovino/ovino_vec.cpp)
target_link_libraries(ovino_vec PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_barrier_probe ovino/barrier_probe.cpp)
target_link_libraries(ovino_barrier_probe PRIVATE ze_loader ${CMAKE_DL_LIBS})


add_executable(ovino_native ovino/ovino_native.cpp)
target_link_libraries(ovino_native PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(ovino_native_ocl ovino/ovino_native_ocl.cpp)
target_link_libraries(ovino_native_ocl PRIVATE ze_loader ${CMAKE_DL_LIBS})

add_executable(gdn_onednn_gemm ovino/gdn_onednn_gemm.cpp)
target_link_libraries(gdn_onednn_gemm PRIVATE dnnl ${CMAKE_DL_LIBS})
target_include_directories(gdn_onednn_gemm PRIVATE /opt/onednn/include)
