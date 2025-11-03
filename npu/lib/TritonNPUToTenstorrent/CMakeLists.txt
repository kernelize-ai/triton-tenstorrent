add_triton_library(TritonNPUToTenstorrent
    DropFunctionPass.cpp
    FinalizeCBTransaction.cpp
    TritonFuncOpToFuncOp.cpp
    TritonNPUToTTKernel.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
