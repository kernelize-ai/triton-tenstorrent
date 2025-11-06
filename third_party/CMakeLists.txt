find_program(CLANG_BIN  clang   REQUIRED)
find_program(CLANGXX_BIN clang++ REQUIRED)

set(TTMLIR_TOOLCHAIN_DIR "/opt/ttmlir-toolchain" CACHE PATH "Toolchain dir - don't change this without updating tt-mlir env script")
if (DEFINED ENV{TRITON_VENV_DIR} AND NOT "$ENV{TRITON_VENV_DIR}" STREQUAL "")
  set(TRITON_VENV_DIR      "$ENV{TRITON_VENV_DIR}")
else()
  set(TRITON_VENV_DIR      "$ENV{HOME}/.venv")
endif()

# detect environment specific variables using python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
       "import sys; sys.path.insert(0, '${CMAKE_CURRENT_SOURCE_DIR}'); \
        from get_llvm_path import get_llvm_path; print(get_llvm_path())"

      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE _llvm_dir
      ERROR_VARIABLE  _llvm_err
      RESULT_VARIABLE _llvm_rc
      OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _llvm_rc EQUAL 0 OR _llvm_dir STREQUAL "")
  message(FATAL_ERROR "Failed to detect LLVM dir via Python:\n${_llvm_err}")
endif()
set(LLVM_SYM_NAME "${_llvm_dir}" CACHE PATH "LLVM build dir" FORCE)

set(LLVM_BUILD_DIR       "$ENV{HOME}/.triton/llvm/${LLVM_SYM_NAME}" CACHE PATH "LLVM build dir")

execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"

      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE _python_vers
      ERROR_VARIABLE  _python_vers_err
      RESULT_VARIABLE _python_vers_rc
      OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _python_vers_rc EQUAL 0 OR _python_vers STREQUAL "")
      message(FATAL_ERROR "Failed to detect Python version:\n${_python_vers_err}")
endif()
set(TTMLIR_PYTHON_VERSION "${_python_vers}" CACHE STRING "Python for tt-mlir")

file(MAKE_DIRECTORY "${TTMLIR_TOOLCHAIN_DIR}")

if (DEFINED ENV{TTMLIR_RUNTIME_ENABLED})
  set(NO_TTMLIR_RUNTIME "")
else()
  set(NO_TTMLIR_RUNTIME "ON")
endif()

add_custom_target(build_tt_mlir
  COMMAND ${CMAKE_COMMAND} -E env
          CC=${CLANG_BIN}
          CXX=${CLANGXX_BIN}
          TRITON_VENV_DIR=${TRITON_VENV_DIR}
          LLVM_BUILD_DIR=${LLVM_LIBRARY_DIR}/../
          TTMLIR_PYTHON_VERSION=${TTMLIR_PYTHON_VERSION}
          TTMLIR_TOOLCHAIN_DIR=${TTMLIR_TOOLCHAIN_DIR}
          NO_TTMLIR_RUNTIME=${NO_TTMLIR_RUNTIME}
          bash ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/build-tt-mlir.sh
  BYPRODUCTS ${TTMLIR_LIB_PATHS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
  USES_TERMINAL
  COMMENT "Building tt-mlir via scripts/build-tt-mlir.sh"
)
