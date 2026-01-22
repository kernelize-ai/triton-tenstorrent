add_triton_library(TritonTenstorrentTransforms
    AccelerateMatmul.cpp
    CanonicalizeMatmulLoops.cpp
    ConvertComputeOps.cpp
    ConvertTensorDescToLoadStore.cpp
    CoreSpecialize.cpp
    RemoveDotLoadLayoutConversions.cpp
    RegAlias.cpp
    RegAlloc.cpp
    RemoveRedundantMasks.cpp
    Utility.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
