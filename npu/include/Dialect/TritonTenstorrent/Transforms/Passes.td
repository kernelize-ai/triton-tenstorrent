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

#endif
