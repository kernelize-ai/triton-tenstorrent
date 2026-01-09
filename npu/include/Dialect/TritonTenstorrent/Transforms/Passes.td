#ifndef TRITONTENSTORRENT_PASSES
#define TRITONTENSTORRENT_PASSES

include "mlir/Pass/PassBase.td"

def TritonTenstorrentConvertTensorDescToLoadStore : Pass<"tritontenstorrent-convert-tensor-desc", "mlir::ModuleOp"> {
    let summary = "Convert Triton tensor descriptors to Triton load/store operations";

    let description = "Can we convert a Tensor Descriptor load/store into a Triton load/store? We will try to convert TensorDescriptor to ptr, and then apply the appropriate offsets using addptr.";

    let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonTenstorrentPropagateTileEncoding : Pass<"tritontenstorrent-propagate-tile-encoding", "mlir::ModuleOp"> {
    let summary = "Propagate tile encoding from compute ops to local loads";

    let description = [{
        Local loads from SRAM to registers need to know which tile register index to load into. This pass propagates the
        register index information from the compute ops (which are users of the tile registers and dicatate placement based
        on use pattern) to local loads, which generically copy data from SRAM into registers. The register index is stored
        in the RegisterEncodingAttr attached to the tensor type.
    }];

    let dependentDialects = ["mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::triton::TritonDialect"
                            ];
}

def TritonTenstorrentConvertComputeOps : Pass<"tritontenstorrent-convert-compute-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton compute ops to Tenstorrent compute ops";

    let description = [{
        Converts compute ops on Tensors to Triton Tenstorrent compute ops (e.g. arith.addf to BinaryComputeOp
        with opcode "addf"). This allows for easy identification of candidate ops for compute kernel lowering in later passes.
    }];

    let dependentDialects = ["mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::triton::TritonDialect",
                             "arith::ArithDialect"
                            ];
}

def TritonTenstorrentRemoveRedundantMasks : Pass<"tritontenstorrent-remove-redundant-masks", "mlir::ModuleOp"> {
    let summary = "Remove redundant mask operations in Tenstorrent kernels";

    let description = [{
        coming soon
    }];

    let dependentDialects = ["mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::triton::TritonDialect"
                            ];
}

def TritonTenstorrentCanonicalizeMatmulLoops : Pass<"tritontenstorrent-canonicalize-matmul-loops", "mlir::ModuleOp"> {
    let summary = "Canonicalize matmul loops in Tenstorrent kernels";

    let description = [{
        Remove unnecessary pointer arithmetic and accumulator handling in matmul loops within Tenstorrent kernels.
    }];

    let dependentDialects = ["mlir::triton::npu::tt::TritonTenstorrentDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::scf::SCFDialect"
                            ];
}

#endif
