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
                             "mlir::triton::npu::tt::TritonTenstorrentDialect",
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

def CreateTTNNGenericOp : Pass<"create-ttnn-generic-op", "mlir::ModuleOp"> {
    let summary = "Wrap the Triton kernels in a top-level TTNN.generic op in the module.";

    let description = [{
        This pass wraps all Triton kernels in the module into a single top-level TTNN.generic operation.
    }];

    let dependentDialects = ["mlir::tt::ttnn::TTNNDialect"];
}

#endif
