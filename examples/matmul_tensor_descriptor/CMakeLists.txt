cmake_minimum_required(VERSION 3.22...3.30)
project(triton_tenstorrent_matmul_example)

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="${CMAKE_CURRENT_SOURCE_DIR}/../")

add_subdirectory(../matmul_common matmul_common)

add_executable(metal_example_matmul_multi_core_tma)
target_sources(metal_example_matmul_multi_core_tma PRIVATE matmul_multi_core_tma.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

find_package(OpenMP REQUIRED)

target_compile_options(metal_example_matmul_multi_core_tma PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(
    metal_example_matmul_multi_core_tma
    PRIVATE
        TT::Metalium
        Matmul::Common
        OpenMP::OpenMP_CXX
)
