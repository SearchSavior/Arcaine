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
