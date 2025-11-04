add_triton_library(TritonTenstorrentTransforms
    ConvertComputeOps.cpp
    PtrRotate.cpp
    PropagateTileEncoding.cpp
    RemoveRedundantMasks.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
