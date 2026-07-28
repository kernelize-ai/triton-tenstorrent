cmake_minimum_required(VERSION 3.22...3.30)
project(kforge_histc)

add_executable(kforge_histc)
target_sources(kforge_histc PRIVATE histc.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(kforge_histc PUBLIC TT::Metalium)
