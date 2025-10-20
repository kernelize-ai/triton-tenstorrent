add_triton_library(TritonNPUToTenstorrent
    FuncOpToMLIRFunc.cpp
    TritonNPUToTenstorrent.cpp

    TritonFuncOpToFuncOp.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
