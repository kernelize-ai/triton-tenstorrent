#ifndef TRITONTENSTORRENT_PASSES
#define TRITONTENSTORRENT_PASSES

include "mlir/Pass/PassBase.td"

def TritonTenstorrentPropagateRegisterIndices : Pass<"tritontenstorrent-propagate-register-indices", "mlir::ModuleOp"> {
    let summary = "Propagate register indices from compute ops to local loads";

    let description = [{
        coming soon
    }];

    let dependentDialects = ["mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::triton::TritonDialect"
                            ];
}

def TritonTenstorrentConvertComputeOps : Pass<"tritontenstorrent-convert-compute-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton compute ops to Tenstorrent compute ops";

    let description = [{
        coming soon
    }];

    let dependentDialects = ["mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::triton::TritonDialect",
                             "arith::ArithDialect"
                            ];
}

def TritonTenstorrentPtrRotate : Pass<"tritontenstorrent-ptr-rotate", "mlir::ModuleOp"> {
    let summary = "Move all pointer offsetting to base pointer instead of array of pointers";

    let description = [{
        coming soon
    }];

    let dependentDialects = ["mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::func::FuncDialect"
                            ];
}

#endif
