add_triton_library(TritonNPUToTenstorrent
    ControlFlowOpToFuncOp.cpp
    ElementwiseOpToTenstorrent.cpp
    FuncOpToMLIRFunc.cpp
    LoadStoreOpToTenstorrent.cpp
    MakeRangeOpToTenstorrent.cpp
    SPMDOpToTenstorrent.cpp
    TritonNPUToTenstorrent.cpp
    TypeConverter.cpp
    ViewOpToTenstorrent.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
