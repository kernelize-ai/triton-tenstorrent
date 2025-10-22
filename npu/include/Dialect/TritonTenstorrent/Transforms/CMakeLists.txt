set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonTenstorrent)
add_public_tablegen_target(TritonTenstorrentTransformsIncGen)
