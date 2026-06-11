add_triton_library(TritonNPUAnalysis
    Utility.cpp

    DEPENDS
    TritonTableGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonTenstorrentTableGen
    TritonTenstorrentAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonIR
    TritonGPUIR
)
