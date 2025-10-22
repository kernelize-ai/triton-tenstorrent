add_triton_library(TritonTenstorrentTransforms
    PropagateRegisterIndices.cpp

    DEPENDS
    TritonTenstorrentTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRTransforms
    TritonTenstorrentIR
)
