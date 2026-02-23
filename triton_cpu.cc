#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/Dialect/TritonCPU/Transforms/Passes.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "npu/include/Dialect/TritonTenstorrent/Transforms/Passes.h"
#include "npu/include/TritonNPUToTenstorrent/Passes.h"

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Host.h"

#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include <pybind11/pybind11.h>

#include <iostream>

namespace py = pybind11;

std::string getDefaultTargerOrProcessTriple() {
  // Return process triple iff the default target triple is empty.
  std::string triple = llvm::sys::getDefaultTargetTriple();
  if (triple.empty()) {
    // host
    triple = llvm::sys::getProcessTriple();
  }
  return triple;
}

void init_triton_cpu_passes(py::module &&m) {
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertTritonCPUToLLVMPass());
  });
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createAllocateSharedMemoryPass());
  });
  m.def("add_shared_memory_global_conversion", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createSharedMemoryGlobalConversionPass());
  });
  m.def("add_masked_ops_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createConvertMaskedOpsToLLVM());
  });
}

void init_triton_npu_passes_tenstorrent(py::module &&m) {
  m.def("add_core_specialize", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createCoreSpecialize());
  });
  m.def("add_materialize_multicasts", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::npu::createTritonTenstorrentMaterializeMulticasts());
  });
  m.def("add_register_allocation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTritonTenstorrentRegAlloc());
  });
  m.def("remove_redundant_masks", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::npu::createTritonTenstorrentRemoveRedundantMasks());
  });
  m.def("add_convert_compute_ops", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTritonTenstorrentConvertComputeOps());
  });
  m.def("add_accelerate_matmul", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createTritonTenstorrentAccelerateMatmul());
  });
  m.def("add_remove_dot_load_layout_conversions", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::
                   createTritonTenstorrentRemoveDotLoadLayoutConversions());
  });
  m.def("add_convert_tensor_desc", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::
                   createTritonTenstorrentConvertTensorDescToLoadStore());
  });
  m.def("add_canonicalize_matmul_loops", [](mlir::PassManager &pm) {
    pm.addPass(
        mlir::triton::npu::createTritonTenstorrentCanonicalizeMatmulLoops());
  });
  m.def("add_to_ttkernel_dialect", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createConvertTritonNPUToTTKernel());
  });

  m.def("add_drop_function",
        [](mlir::PassManager &pm, const std::string &funcName) {
          pm.addPass(mlir::triton::npu::createDropFunction({funcName}));
        });

  // tt-mlir specific passes
  m.def("add_ttkernel_control_dst_selection", [](mlir::PassManager &pm) {
    pm.addPass(mlir::tt::ttkernel::createTTKernelControlDstSection());
  });
  m.def("add_ttkernel_device_zone_scopes", [](mlir::PassManager &pm) {
    pm.addPass(mlir::tt::ttkernel::createTTKernelInsertDeviceZoneScopes());
  });

  // emit-c -- TODO should this be part of a different namespace?
  m.def("add_ttkernel_to_emitc", [](mlir::PassManager &pm) {
    pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
  });
  m.def("add_form_expressions_pass", [](mlir::PassManager &pm) {
    pm.addPass(mlir::emitc::createFormExpressionsPass());
  });

  // tt-core
  m.def(
      "add_ttcore_register_device_pass",
      [](mlir::PassManager &pm, const std::string &systemDescPath) {
        mlir::tt::ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
        if (!systemDescPath.empty()) {
          registerDeviceOptions.systemDescPath = systemDescPath;
        }
        pm.addPass(mlir::tt::ttcore::createTTCoreRegisterDevicePass(
            registerDeviceOptions));
      });

  // tt-nn
  m.def("add_create_ttnn_generic_op", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::npu::createCreateTTNNGenericOp());
  });
}

void init_triton_npu_passes_common(py::module &&m) {
  m.def("add_arith_int_range_opts", [](mlir::PassManager &pm) {
    pm.addPass(mlir::arith::createArithIntRangeOpts());
  });
  m.def("add_arith_expand", [](mlir::PassManager &pm) {
    pm.addPass(mlir::arith::createArithExpandOpsPass());
  });
}

void init_triton_cpu_passes_ttgpuir(py::module &&m) {
  m.def("add_coalesce", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createTritonCPUCoalesce());
  });
  m.def("add_make_persistent_kernel", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::cpu::createMakePersistentKernelPass());
  });
}

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  // Triton to TritonGPU passes specific to the Triton CPU plugin
  init_triton_cpu_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  // TritonGPU to LLVM passes specific to the Triton CPU plugin
  init_triton_cpu_passes(passes.def_submodule("ttcpuir"));

  init_triton_npu_passes_tenstorrent(passes.def_submodule("tenstorrent"));
  init_triton_npu_passes_common(passes.def_submodule("common"));

  m.def(
      "translate_to_cpp",
      [](mlir::ModuleOp moduleOp, const std::string &symbolName) -> py::object {
        mlir::SymbolTable symbolTable(moduleOp);
        auto entry = symbolTable.lookup<mlir::func::FuncOp>(symbolName);
        assert(entry && "expected kernel func with given symbol name");
        assert(entry->hasAttr(mlir::tt::ttkernel::ThreadTypeAttr::name) &&
               "expected thread type attr on kernel func");

        std::string cppCode;
        llvm::raw_string_ostream os(cppCode);
        assert(mlir::succeeded(
                   mlir::tt::ttkernel::translateKernelFuncToCpp(entry, os)) &&
               "failed to translate kernel func to C++");
        return py::str(cppCode);
      });

  m.def("load_dialects",
        [](mlir::MLIRContext &context, const std::string &device) {
          mlir::DialectRegistry registry;
          registry.insert<mlir::triton::cpu::TritonCPUDialect>();

          if (device == "Tenstorrent") {
            // register tenstorrent dialects
            registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
          }

          context.appendDialectRegistry(registry);
          context.loadAllAvailableDialects();
        });

  m.def("get_default_target_triple",
        []() { return getDefaultTargerOrProcessTriple(); });

  m.def("get_processor_name",
        []() { return llvm::sys::getHostCPUName().str(); });

  m.def("attach_target_triple",
        [](llvm::Module *module, const std::string &triple) {
          module->setTargetTriple(llvm::Triple(triple));
        });
}
