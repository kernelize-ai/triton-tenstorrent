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

def ConvertTritonFuncToFunc : Pass<"convert-triton-func-to-func", "mlir::ModuleOp"> {
    let summary = "Convert tt::FuncOp to canonical func:FuncOp";

    let description = [{
        Converts tt:FuncOp to canonical func::FuncOp. Preserves arguments and types but drops Triton-specific function attributes. This must run before converting ops to ttkernel, as ttkernel ops often expect a FuncOp op container.
    }];

    let dependentDialects = ["mlir::triton::TritonDialect",
                             "func::FuncDialect"
                            ];
}

#endif
