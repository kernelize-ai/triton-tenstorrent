find_program(CLANG_BIN  clang   REQUIRED)
find_program(CLANGXX_BIN clang++ REQUIRED)

set(TTMLIR_TOOLCHAIN_DIR "/opt/ttmlir-toolchain" CACHE PATH "Toolchain dir - don't change this without updating tt-mlir env script")
set(TRITON_VENV_DIR      "$ENV{HOME}/.venv" CACHE PATH "Triton venv dir")
set(LLVM_BUILD_DIR       "$ENV{HOME}/.triton/llvm/llvm-ubuntu-x86" CACHE PATH "LLVM build dir")
set(TTMLIR_PYTHON_VERSION "python3.12" CACHE STRING "Python for tt-mlir")

file(MAKE_DIRECTORY "${TTMLIR_TOOLCHAIN_DIR}")

add_custom_target(build_tt_mlir
  COMMAND ${CMAKE_COMMAND} -E env
          CC=${CLANG_BIN}
          CXX=${CLANGXX_BIN}
          TRITON_VENV_DIR=${TRITON_VENV_DIR}
          LLVM_BUILD_DIR=${LLVM_LIBRARY_DIR}/../
          TTMLIR_PYTHON_VERSION=${TTMLIR_PYTHON_VERSION}
          TTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR}
          bash ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build-tt-mlir.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
  USES_TERMINAL
  COMMENT "Building tt-mlir via scripts/build-tt-mlir.sh"
)
