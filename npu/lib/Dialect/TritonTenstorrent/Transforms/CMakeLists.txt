add_triton_library(TritonTenstorrentTransforms
    ConvertComputeOps.cpp
    PtrRotate.cpp
    PropagateRegisterIndices.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
