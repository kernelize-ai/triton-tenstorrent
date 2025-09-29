add_triton_library(TritonNPUToTenstorrent
    ControlFlowOpToFuncOp.cpp
    FuncOpToMLIRFunc.cpp
    TritonNPUToTenstorrent.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
