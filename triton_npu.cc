#include "npu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "npu/include/TritonNPUToLLVM/Passes.h"
#include "npu/include/JIT/CpuJIT.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/TargetSelect.h"

#include <pybind11/pybind11.h>

#include <iostream>
#include <cstdint>


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

void init_triton_npu_passes_ttgpuir(py::module &&m) {
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertTritonNPUToLLVMPass());
  });
}

void init_triton_npu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_npu_passes_ttgpuir(passes.def_submodule("ttnpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect>();
    // registry.insert<mlir::vector::VectorDialect>();
    // mlir::triton::cpu::registerTritonOpScalarizeExternalModels(registry);
    // mlir::registerAMXDialectTranslation(registry);
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

  m.def("load_binary",
        [](const std::string& name, const std::string& kernel) {
          llvm::InitializeNativeTarget();
          llvm::InitializeNativeTargetAsmPrinter();

          std::unique_ptr<NPU::CpuJIT> jit = llvm::cantFail(NPU::CpuJIT::Create());

          auto buffer = llvm::MemoryBuffer::getMemBufferCopy(kernel.c_str());
          llvm::cantFail(jit->addObjectFile(std::move(buffer)));

          return jit;
        });

    m.def("lookup_function",
      [](NPU::CpuJIT* jit, const std::string &name) {
          auto objSymbol = llvm::cantFail(jit->lookup(name));
          auto fnPtr = objSymbol.toPtr<void (*)()>();
          return reinterpret_cast<uintptr_t>(fnPtr);
      });
}


