#ifndef TRITONNPU_TENSTORRENT_CONVERSION_PASSES
#define TRITONNPU_TENSTORRENT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonNPUToTTKernel : Pass<"convert-triton-npu-to-ttkernel", "mlir::ModuleOp"> {
    let summary = "Convert TritonNPU to TTKernel";
    let description = [{
        The `convert-triton-npu-to-ttkernel` pass transforms Triton NPU operations
        into TTKernel operations, preparing the IR for Tenstorrent kernel generation.
    }];

    let dependentDialects = ["mlir::triton::TritonDialect",
                            "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::tt::ttkernel::TTKernelDialect",
                             "mlir::arith::ArithDialect",
                             "func::FuncDialect",
                            ];
}

def DropFunction : Pass<"drop-function", "mlir::ModuleOp"> {
    let summary = "Drop named functions from the module";

    let description = [{
        Drops named function definitions from the module. Useful for debugging.
    }];

    let options = [
        Option<"functionName", "function-name", "std::string", /*default=*/"",
               "Name of the function to drop">
    ];

    let dependentDialects = ["mlir::func::FuncDialect"];
}

def FinalizeCBTransaction : Pass<"finalize-cb-transaction", "mlir::ModuleOp"> {
    let summary = "Finalize CB transactions in TTKernel ops";

    let description = [{
        Finalizes circular buffer (CB) transactions in TTKernel operations by inserting necessary pop/push operations.
    }];

    let dependentDialects = ["mlir::tt::ttkernel::TTKernelDialect",
                             "mlir::arith::ArithDialect"
                            ];
}

#endif
