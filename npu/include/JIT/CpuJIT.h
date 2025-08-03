#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::orc;

namespace NPU {

class CpuJIT {
public:
  CpuJIT(std::unique_ptr<ExecutionSession> ES,
                  JITTargetMachineBuilder JTMB, DataLayout DL)
      : ES(std::move(ES)), DL(std::move(DL)), Mangle(*this->ES, this->DL),
        ObjectLayer(*this->ES,
                    [](const MemoryBuffer &) {
                      return std::make_unique<SectionMemoryManager>();
                    }),
        TransformLayer(*this->ES, ObjectLayer),
        CompileLayer(*this->ES, ObjectLayer,
                     std::make_unique<ConcurrentIRCompiler>(std::move(JTMB))),
        // OptimizeLayer(*this->ES, CompileLayer, optimizeModule),
        JD(this->ES->createBareJITDylib("<main>")) {
    JD.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            DL.getGlobalPrefix())));
  }

  ~CpuJIT() {
    if (auto Err = ES->endSession())
      ES->reportError(std::move(Err));
  }

  static Expected<std::unique_ptr<CpuJIT>> Create() {
    auto EPC = SelfExecutorProcessControl::Create();
    if (!EPC)
      return EPC.takeError();

    auto ES = std::make_unique<ExecutionSession>(std::move(*EPC));

    JITTargetMachineBuilder JTMB(
        ES->getExecutorProcessControl().getTargetTriple());

    auto DL = JTMB.getDefaultDataLayoutForTarget();
    if (!DL)
      return DL.takeError();

    return std::make_unique<CpuJIT>(std::move(ES), std::move(JTMB),
                                             std::move(*DL));
  }

 Error addObjectFile(std::unique_ptr<MemoryBuffer> ObjBuffer) {
#if 1
        return TransformLayer.add(std::move(JD.getDefaultResourceTracker()),
                                  std::move(ObjBuffer));
#else
        auto Obj = object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef());
        if (!Obj)
            return Obj.takeError();

        auto K = ES->allocateVModule();
        return ObjectLayer.add(JD, K, std::move(ObjBuffer));
#endif
    }

    Error addModule(std::unique_ptr<Module> M) {
        ThreadSafeModule TSM(std::move(M), std::make_unique<LLVMContext>());
        return CompileLayer.add(JD, std::move(TSM));
    }

    Expected<ExecutorSymbolDef> lookup(StringRef Name) {
        return ES->lookup({&JD}, Mangle(Name.str()));
    }

    const DataLayout& getDataLayout() const { return DL; }

private:
    std::unique_ptr<ExecutionSession> ES;

    DataLayout DL;
    MangleAndInterner Mangle;

    RTDyldObjectLinkingLayer ObjectLayer;
    ObjectTransformLayer TransformLayer;
    IRCompileLayer CompileLayer;

    JITDylib& JD;
};

}
