#ifndef TRITONNPU_TENSTORRENT_CONVERSION_PASSES
#define TRITONNPU_TENSTORRENT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonNPUToTenstorrent : Pass<"convert-triton-npu-to-tenstorrent", "mlir::ModuleOp"> {
    let summary = "Convert TritonNPU to Tenstorrent";
    let description = [{
        The `convert-triton-npu-to-tenstorrent` pass converts Triton NPU operations
        into Tenstorrent operations, enabling further optimizations and code generation.
    }];

    let dependentDialects = ["mlir::triton::TritonDialect",
                             "mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::func::FuncDialect",
                             "mlir::tt::ttkernel::TTKernelDialect"
                            ];
}

def ConvertMathToD2M : Pass<"convert-math-to-d2m", "mlir::ModuleOp"> {
    let summary = "Convert Math to D2M";
    let description = [{
        The `convert-math-to-d2m` pass converts mathematical operations
        into D2M operations, optimizing for Tenstorrent hardware.
    }];

    let dependentDialects = ["mlir::triton::TritonDialect",
                             "arith::ArithDialect",
                             "mlir::tt::d2m::D2MDialect"
                            ];
}

#endif
