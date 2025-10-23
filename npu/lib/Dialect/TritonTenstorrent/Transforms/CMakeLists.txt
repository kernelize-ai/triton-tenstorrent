add_triton_library(TritonTenstorrentTransforms
    ConvertComputeOps.cpp
    PropagateRegisterIndices.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
