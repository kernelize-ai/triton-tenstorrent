add_triton_library(TritonNPUToTenstorrent
    ControlFlowOpToFuncOp.cpp
    FuncOpToMLIRFunc.cpp
    MakeRangeOpToTenstorrent.cpp
    SPMDOpToTenstorrent.cpp
    TritonNPUToTenstorrent.cpp
    TypeConverter.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
