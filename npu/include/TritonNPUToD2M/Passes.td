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
                             "func::FuncDialect",
                             "mlir::triton::npu::tt::TritonTenstorrentDialect"
                            ];
}

def HoistCBAllocs : Pass<"hoist-cb-allocs", "mlir::ModuleOp"> {
    let summary = "Hoist CB allocs from D2M generic op regions to the function level";

    let description = [{
        Triton loads/stores and some intermediates are converted from registers to CB buffers during D2M lowering. The CB buffers must be initialized/allocated per D2M generic op outside the op region. This pass identifies such buffers and hoists their allocs to the function level, and replaces the original allocs with arguments forwarded through the D2M generic op via additionalArgs.
    }];

    let dependentDialects = ["mlir::d2m::D2MDialect"];
}

#endif
