add_triton_library(TritonTenstorrentTransforms
    AccelerateMatmul.cpp
    CanonicalizeMatmulLoops.cpp
    ConvertComputeOps.cpp
    ConvertTensorDescToLoadStore.cpp
    CoreSpecialize.cpp
    MaterializeMulticasts.cpp
    RegAlias.cpp
    RegAlloc.cpp
    RemoveDotLoadLayoutConversions.cpp
    RemoveRedundantMasks.cpp
    TagInputOutputs.cpp
    Utility.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
