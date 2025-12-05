cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_add_2_integers_in_compute_multicore)

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="${CMAKE_CURRENT_SOURCE_DIR}/../")

add_executable(metal_example_add_2_integers_in_compute_multicore)
target_sources(metal_example_add_2_integers_in_compute_multicore PRIVATE add_2_integers_in_compute_multicore.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_add_2_integers_in_compute_multicore PUBLIC TT::Metalium)
