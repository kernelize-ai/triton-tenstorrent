cmake_minimum_required(VERSION 3.22...3.30)
project(kforge_histc_atomic)

add_executable(kforge_histc_atomic)
target_sources(kforge_histc_atomic PRIVATE histc_atomic.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(kforge_histc_atomic PUBLIC TT::Metalium)
