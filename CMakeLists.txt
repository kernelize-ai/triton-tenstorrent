include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/)

# Tenstorrent includes
if(DEFINED ENV{TENSTORRENT_MLIR})
  set(TENSTORRENT_MLIR $ENV{TENSTORRENT_MLIR})
else()
  set(TENSTORRENT_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tt-mlir")
endif()
set(TTMLIR_LIB_DIR "${TENSTORRENT_MLIR}/build/lib")

set(TTMLIR_LIB_NAMES
    MLIRTTKernelDialect
    MLIRTTKernelTransforms
    TTKernelTargetCpp
    TTMLIRTTKernelToEmitC
    MLIRTTMetalDialect
    MLIRTTCoreDialect
)

# Create imported libraries for each TT-MLIR dep
set(TTMLIR_LIBS)
set(TTMLIR_LIB_PATHS)
foreach(lib ${TTMLIR_LIB_NAMES})
  set(LIB_PATH "${TTMLIR_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  list(APPEND TTMLIR_LIB_PATHS ${LIB_PATH})
  add_library(${lib} STATIC IMPORTED)
  set_target_properties(${lib} PROPERTIES
    IMPORTED_LOCATION "${LIB_PATH}"
  )
  add_dependencies(${lib} build_tt_mlir)
  list(APPEND TTMLIR_LIBS
    ${lib}
  )
endforeach()

add_subdirectory(third_party)

include_directories(
    "${TENSTORRENT_MLIR}/include"
    "${TENSTORRENT_MLIR}/build/include"
)


# Lit test support
set(triton_cpu_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Triton plugin
add_subdirectory(npu)
add_subdirectory(utils)
add_subdirectory(cpu)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonCPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_cpu.cc
    LINK_LIBS
    TritonCPUToLLVM
    TritonNPUToTenstorrent
    MLIRQuantDialect
    MLIREmitCDialect
    MLIRArithToEmitC
    MLIRMemRefToEmitC
    MLIRSCFToEmitC
    MLIRTargetCpp
    ${TTMLIR_LIBS}

    DEPENDS
    build_tt_mlir)
  target_link_libraries(TritonCPU PRIVATE Python3::Module pybind11::headers)

  target_include_directories(TritonCPU PUBLIC
      "${TENSTORRENT_MLIR}/include"
      "${TENSTORRENT_MLIR}/build/include"
  )

  # copy ttmlir headers to the build directory so dependent targets in Triton can see them
  set(TTMLIR_HEADER_SRC_DIR "${TENSTORRENT_MLIR}/build/include")
  set(TTMLIR_HEADER_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  add_custom_target(copy_ttmlir_headers ALL
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${TTMLIR_HEADER_DST_DIR}/ttmlir"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TTMLIR_HEADER_DST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TTMLIR_HEADER_SRC_DIR}" "${TTMLIR_HEADER_DST_DIR}"
    COMMENT "Copying tt-mlir headers to build directory"
    VERBATIM
  )

  # ensure headers exist before compiling TritonCPU
  add_dependencies(TritonCPU copy_ttmlir_headers)
  # Ensure tt-mlir build step runs before we copy the built headers
  add_dependencies(copy_ttmlir_headers build_tt_mlir)
  target_include_directories(TritonCPU PUBLIC
      "${TTMLIR_HEADER_DST_DIR}"
  )

endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()

add_subdirectory(test)

# uKernel Libs

option(BUILD_SLEEF "Build SLEEF library" OFF)
if(BUILD_SLEEF)
  include(ExternalProject)

  # Sleef

  set(SLEEF_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/sleef-src")

  # Sleef clones TLFloat as an external project which is problematic, as it will
  # cache temp ninja paths and not get the new path on a re-configure. Delete the
  # directory manually if it exists to force a re-install for each re-configure.
  if(EXISTS "${SLEEF_SRC_DIR}/src/sleef/submodules/tlfloat")
    file(REMOVE_RECURSE "${SLEEF_SRC_DIR}/src/sleef/submodules/tlfloat")
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_deps/sleef-build/ext_tlfloat-prefix")
  endif()

  set(SLEEF_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/sleef-install)

  ExternalProject_Add(sleef
    PREFIX ${SLEEF_SRC_DIR}
    # Or if you want to pull from git directly:
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG master

    CMAKE_ARGS
      -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
      -DCMAKE_INSTALL_PREFIX=${SLEEF_INSTALL_DIR}
      -DCMAKE_BUILD_TYPE=Release
      -DSLEEF_BUILD_SHARED_LIBS=OFF
      -DSLEEF_BUILD_DFT=OFF
      -DSLEEF_BUILD_GNUABI_LIBS=OFF
      -DSLEEF_BUILD_TESTS=OFF
      -DSLEEF_BUILD_SCALAR_LIB=OFF

    BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/sleef-build

    # Control which steps happen
    UPDATE_DISCONNECTED 1    # don’t re-fetch every time
    INSTALL_DIR ${SLEEF_INSTALL_DIR}
  )

  # Add a custom step to copy libsleef.a after install
  ExternalProject_Add_Step(sleef copy_lib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${SLEEF_INSTALL_DIR}/lib/libsleef.a
            ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    DEPENDEES install
    ALWAYS 1
  )

  # So you can depend on a phony target to ensure the copy happens
  add_custom_target(sleef_artifact ALL DEPENDS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  add_dependencies(sleef_artifact sleef)
endif()
