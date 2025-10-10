add_triton_library(TritonNPUToTenstorrent
    ConvertMathToD2M.cpp
    ConvertTritonFuncToFunc.cpp

    FuncOpToMLIRFunc.cpp
    TritonNPUToTenstorrent.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
