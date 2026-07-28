cmake_minimum_required(VERSION 3.22...3.30)
project(kforge_histc_hybrid)

add_executable(kforge_histc_hybrid)
target_sources(kforge_histc_hybrid PRIVATE histc_hybrid.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(kforge_histc_hybrid PUBLIC TT::Metalium)
