add_triton_library(TritonNPUToTenstorrent
    ComputeOpToTTKernel.cpp
    DropFunctionPass.cpp
    FinalizeCBTransaction.cpp
    TritonFuncOpToFuncOp.cpp
    TritonNPUToTTKernel.cpp
    Utility.cpp

    DEPENDS
    TritonNPUTenstorrentConversionPassIncGen

    LINK_LIBS ${TTMLIR_LIBS}
)
