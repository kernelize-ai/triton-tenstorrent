add_triton_library(TritonTenstorrentTransforms
    CanonicalizeMatmulLoops.cpp
    ConvertComputeOps.cpp
    ConvertTensorDescToLoadStore.cpp
    PropagateTileEncoding.cpp
    RemoveRedundantMasks.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
