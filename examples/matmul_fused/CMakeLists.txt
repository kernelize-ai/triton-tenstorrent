cmake_minimum_required(VERSION 3.22...3.30)
project(triton_tenstorrent_matmul_example)

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="${CMAKE_CURRENT_SOURCE_DIR}/../")

add_subdirectory(../matmul_common matmul_common)

add_executable(matmul_fused_add)
target_sources(matmul_fused_add PRIVATE matmul_fused_add.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

find_package(OpenMP REQUIRED)

target_compile_options(matmul_fused_add PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(
    matmul_fused_add
    PRIVATE
        TT::Metalium
        Matmul::Common
        OpenMP::OpenMP_CXX
)
