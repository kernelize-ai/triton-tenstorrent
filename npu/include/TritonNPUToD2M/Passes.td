#ifndef TRITONNPU_TENSORRNT_D2M_CONVERSION_PASSES
#define TRITONNPU_TENSORRNT_D2M_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonNPUToD2M : Pass<"convert-triton-npu-to-d2m", "mlir::ModuleOp"> {
    let summary = "Convert TritonNPU to D2M";
    let description = [{
        The `convert-triton-npu-to-d2m` pass transforms Triton NPU operations
        into D2M operations, preparing the IR for D2M kernel generation.
    }];

    let dependentDialects = ["mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::d2m::D2MDialect",
                             "mlir::arith::ArithDialect",
                             "func::FuncDialect"
                            ];
}

#endif
