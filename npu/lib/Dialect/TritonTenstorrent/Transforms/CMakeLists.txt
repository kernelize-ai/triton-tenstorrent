add_triton_library(TritonTenstorrentTransforms
    AccelerateMatmul.cpp
    CanonicalizeMatmulLoops.cpp
    ConvertComputeOps.cpp
    ConvertTensorDescToLoadStore.cpp
    PropagateTileEncoding.cpp
    RemoveDotLoadLayoutConversions.cpp
    RemoveRedundantMasks.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
